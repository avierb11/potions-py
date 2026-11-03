"""Batched hydrological zone model (port of the reference ``hydro.rs``).

The four storage zones (``SnowZone``, ``SurfaceZone``, ``GroundZone``,
``GroundZoneB``) each hold a single water-storage state ``s`` [mm] and solve a
scalar implicit-Euler water-balance equation at every step:

``f(s) = (s_0 - s) + dt * mass_balance(s, d) = 0``

with ``mass_balance = q_in + forc_flux - vap_flux - lat_flux - vert_flux``.

**Batched convention:** unlike the scalar reference (which steps one zone at a
time with plain ``f64``), every flux method and ``step`` here operates on
broadcastable values and returns ``torch.Tensor`` fields, so many replicate
zones / parameter sets can be stepped in one forward pass.  Calling ``step``
with a plain ``float`` state returns 0-dim tensors, matching the scalar
reference lane-for-lane.

**Root finder (damped Newton with an analytic derivative):** the
implicit-Euler root is located with a batched damped Newton iteration
(:func:`_newton_storage_solve`, the structure of
``scripts/hydrologic_zone_ode_ad.ipynb``).  Each trip re-evaluates the residual
``r = f(s)`` at the current trial ``s`` and pairs it with the *analytic*
derivative ``dr = df/ds = -1 + dt * d(mass_balance)/ds`` (each zone's
:meth:`HydrologicZone.mass_balance_deriv` mirrors its
:meth:`HydrologicZone.mass_balance` flux-by-flux, so no ``torch.autograd.grad``
is required on the hot path and no graph is ever built).  The Newton step is
floored at ``NEWTON_MIN_DERIV`` and clamped to non-negative storage (the clamp
keeps ``s**alpha``-type flux laws NaN-free when a trial crosses ``s = 0``), and
is accepted through an Armijo-style backtracking line search on the residual
magnitude.  The damping is load-bearing, not an optimisation: groundwater
baseflow laws ``k * s**alpha`` with ``0 < alpha < 1`` are kinked at ``s = 0``
(infinite analytic one-sided slope, zero subgradient on the clamped boundary),
so an undamped full Newton step can jump across the kink and the iterate
2-cycles forever; the sufficient-decrease guard makes each trip reduce ``|f|``
and kills the cycle.

The analytic derivative is written to match ``torch.autograd.grad`` bit-for-bit
at the non-smooth points (and in the *order of the backward operations*), so
that the damped-Newton iterates of the analytic and autograd paths are
bit-identical.  The pinned subgradient conventions (measured against
``torch.autograd.grad`` on the installed torch, f64/CPU; 0-dim and 1-dim
tensors agree in every case) are:

* ``minimum(x, c)``: ``1`` (``x < c``), ``0.5`` (tie), ``0`` (``x > c``),
  **``nan -> 1``** (:func:`_d_min`);
* ``maximum(x, c)``: ``0`` (``x < c``), ``0.5`` (tie), ``1`` (``x > c``),
  **``nan -> 1``** (:func:`_d_max`);
* ``clamp_max(x, c)``: **mirrors the forward threshold's dispatch** — a
  Python ``float`` ``c`` is a hard step (``1`` for ``x < c``, ``0`` for
  ``x >= c``: tie/nan/``+inf``, ``1`` for ``x == -inf``); a ``Tensor`` ``c``
  is bit-identical to ``minimum``'s subgradient (:func:`_d_clamp_max`);
* ``clamp_min(x, c)``: likewise — a ``float`` ``c`` is a hard step (``1``
  for ``x > c``, ``0`` for ``x <= c``: tie/nan/``-inf``, ``1`` for
  ``x == +inf``); a ``Tensor`` ``c`` is bit-identical to ``maximum``'s
  subgradient (:func:`_d_clamp_min`);
* ``x ** b``: ``g * (b * x**(b-1))`` — the slope is computed through
  ``torch.pow`` (``0**negative -> +inf``, ``0**0 -> 1``, ``0**positive -> 0``,
  negative base with fractional exponent ``-> nan``), multiplied by ``b``
  first, then by the upstream gradient ``g``, *last*; and the inner
  ``x**(b-1)`` **mirrors the dispatch of the forward exponent** — a Python
  ``float`` ``b`` (the default for every zone parameter) keeps the
  scalar-overload ``pow`` in the slope, a 0-dim ``Tensor`` ``b`` uses the
  tensor-tensor ``pow`` (the two differ in low-order bits at some finite
  ``x`` and in the ``-0`` / ``-inf`` conventions) (:func:`_d_pow`);
* true divisions backpropagate as ``grad / c`` (true division, not
  ``grad * (1/c)``).

``torch.where(mask, a, b)``'s backward gates each branch with a *where* (not a
multiply): the discarded branch's gradient is exactly ``+0``, so the analytic
chains reproduce that with ``torch.where(mask, grad, zeros)`` rather than
``grad * mask`` (which would give ``-0`` for a negative ``grad``).

The residual and derivative are plain batched torch expressions evaluated
eagerly under ``torch.no_grad``; there is no ``torch.compile`` (or any
tracing/fusion) anywhere in the package.  The driver still accepts a
residual-only callable (a bare tensor return) and falls back to
``torch.autograd.grad`` for it, so arbitrary test residuals keep working; the
zone hot path always supplies the analytic ``(r, dr)`` pair.  A lane that has
not met ``|f| <= atol + rtol*|s|`` after ``NEWTON_MAX_ITER`` trips raises
:class:`ScalarRootFindingError` (the reference ``math.rs`` secant + bisection
fallback is not reproduced here).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Callable, TypeAlias

import torch

from .common_types import HydroForcing, HydroStep
from .solvers import RootFindingError

Tensor: TypeAlias = torch.Tensor
Scalar: TypeAlias = float | torch.Tensor

#: Maximum AD-Newton trips per lane before a :class:`ScalarRootFindingError`
#: is raised (the notebook's ``max_iterations=20``).
NEWTON_MAX_ITER: int = 20
#: Absolute convergence bar for full-precision (f64) solves, i.e. the
#: reference ``FIND_ROOT_TOL``.  f64 uses a *pure absolute* bar (``rtol = 0``),
#: so this path keeps the legacy f64 tolerance exactly.
NEWTON_ATOL_F64: float = 1e-6
#: Relative convergence term for reduced-precision (f32) solves: the bar is
#: ``|f| <= atol + rtol*|s|`` (the notebook's two-term form), because a pure
#: absolute bar is unreachable at the f32 ulp floor for mm-scale storages.
NEWTON_RTOL_F32: float = 1e-6
#: Storage magnitude [mm] that the f32 absolute bar is clamped against
#: (states in this model reach ~1000 mm; the reference bisected on ``[0, 1000]``).
_STORAGE_MM_SCALE: float = 1000.0
#: Derivatives with ``|df/ds| < NEWTON_MIN_DERIV`` are floored to this
#: (sign-preserving) value before the Newton update (the notebook's
#: ``minimum_derivative``); a flat residual cannot then blow up the step.
NEWTON_MIN_DERIV: float = 1e-12
#: Armijo sufficient-decrease factor for the backtracking acceptance test: a
#: trial ``s - alpha*step`` is accepted only if ``|f| <= NEWTON_BACKTRACK_C*|r|``
#: (the same convention as the RT solver's Armijo line search).
NEWTON_BACKTRACK_C: float = 0.5
#: Maximum step halvings per trip in the backtracking search; a lane that never
#: meets the decrease condition keeps the candidate with the smallest ``|f|``
#: observed (the search is a per-trip guard, so this is bounded work).
NEWTON_MAX_BACKTRACK: int = 10


class ScalarRootFindingError(RootFindingError):
    """Storage root could not be found.

    Mirrors the Rust ``ScalarRootFindingError`` enum (``IterationError`` /
    ``NanError`` variants), surfaced as a single exception type: the AD-Newton
    driver raises it when any lane (non-converging or NaN residual) has not
    met the convergence bar after ``NEWTON_MAX_ITER`` trips.  Subclasses
    :class:`~potions.solvers.RootFindingError` (like the batched RT solver's
    ``BatchedRootFindingError``), so a package-wide ``except
    RootFindingError`` — or ``except RuntimeError`` — catches every
    non-converged root.
    """

# Default dtype for scalar coercion in hydrology. Can be overridden via
# ``set_hydro_dtype`` to enable float32 hydrology when the RT solver runs f32.
_DEFAULT_HYDRO_DTYPE: torch.dtype = torch.float64


def set_hydro_dtype(dtype: torch.dtype) -> None:
    """Set the default dtype for hydrology scalar coercion (``_t``)."""
    global _DEFAULT_HYDRO_DTYPE
    _DEFAULT_HYDRO_DTYPE = dtype


def _t(x: Scalar) -> Tensor:
    """Coerce a scalar or tensor to a tensor, preserving differentiability.

    Plain Python scalars are lifted to 0-dim tensors of
    ``_DEFAULT_HYDRO_DTYPE``; torch tensors are returned **unmodified** (no
    detach, no cast) so a trial that carries a gradient (the AD-Newton
    residual input) keeps its differentiable path to the solver's ``s``.
    Callers that need a fresh differentiable input (the state carried out of
    :meth:`HydrologicZone.step`, warm-start seeds) detach explicitly.
    """
    if isinstance(x, (int, float)):
        return torch.tensor(float(x), dtype=_DEFAULT_HYDRO_DTYPE)
    return x


def _solver_tols(dtype: torch.dtype) -> tuple[float, float]:
    """dtype-aware ``(atol, rtol)`` convergence bar for the AD-Newton solve.

    The test is ``|f| <= atol + rtol*|s|`` (the notebook's two-term form).
    float64 keeps the reference ``FIND_ROOT_TOL`` as a *pure absolute* bar
    (``rtol = 0``), exactly matching the legacy f64 tolerance and the
    residual-satisfaction contract the stepping tests assert.  float32 clamps
    the absolute bar above the f32 ulp floor (states reach
    ``_STORAGE_MM_SCALE`` mm here) and adds the relative term, so the
    implicit-Euler root is actually reachable at reduced precision.
    """
    if dtype != torch.float32:
        return NEWTON_ATOL_F64, 0.0
    eps: float = float(torch.finfo(torch.float32).eps)
    atol: float = max(NEWTON_ATOL_F64, 64.0 * eps * _STORAGE_MM_SCALE)
    return atol, NEWTON_RTOL_F32


# ---------------------------------------------------------------------- #
# Analytic flux-law derivative primitives (match torch.autograd bit-for-bit)
# ---------------------------------------------------------------------- #
#
# These are the subgradients of the non-smooth building blocks the zone flux
# laws are composed of.  ``torch.autograd`` does not document these
# conventions, so each was pinned down empirically against ``autograd.grad``
# on the installed torch (f64 / CPU; 0-dim and 1-dim tensors agree in every
# case; see ``scripts/parity_hydro_derivative.py``):
#
#   * ``torch.minimum(x, c)``  ->  1 (x < c), 0.5 (tie), 0 (x > c), nan -> 1
#   * ``torch.maximum(x, c)``  ->  0 (x < c), 0.5 (tie), 1 (x > c), nan -> 1
#   * ``torch.clamp_max(x, c)``->  float c: 1 (x < c), 0 (x >= c: tie/nan/
#                                  +inf), 1 (x == -inf) — a hard step;
#                                  Tensor c: bit-identical to the
#                                  ``minimum`` subgradient (0.5 tie,
#                                  nan -> 1)
#   * ``torch.clamp_min(x, c)``->  float c: 1 (x > c), 0 (x <= c: tie/nan/
#                                  -inf), 1 (x == +inf) — a hard step whose
#                                  tie belongs to the flat side; Tensor c:
#                                  bit-identical to the ``maximum``
#                                  subgradient (0.5 tie, nan -> 1)
#
# The clamp backward therefore *mirrors the dispatch of the forward
# threshold*, like ``pow`` mirrors the forward exponent: a Python float
# binds the scalar-threshold kernel (hard step, tie and NaN on the flat
# side), while any Tensor threshold (0-dim or broadcastable) binds the
# tensor-threshold kernel (min/max subgradient).  The ``_d_clamp_*``
# functions branch on ``isinstance(c, Tensor)`` to stay bit-identical for
# either kind of threshold.
#
# The clamp *backward* is a *where gate* on the input (scalar path:
# ``where(x < c, grad, 0)`` / ``where(x > c, grad, 0)``), not a multiply:
# the gate drops whatever the downstream sub-chain computed — including
# NaNs (e.g. ``+0 * +inf`` from a ``pow`` slope at ``u = 0`` with
# ``alpha < 1``) — so zone chains apply the gate with ``torch.where`` and
# never fold it into a multiplication.
#   * ``x ** b``               ->  ``g * (b * x**(b-1))``: ATen computes the
#                                  slope ``b * x**(b-1)`` via ``torch.pow``
#                                  (0**negative -> +inf, 0**0 -> 1,
#                                  0**positive -> 0, negative base +
#                                  fractional exponent -> nan), multiplies it
#                                  by ``b`` first and by the upstream
#                                  gradient ``g`` *last*, and the inner
#                                  ``x**(b-1)`` mirrors the *dispatch* of the
#                                  forward exponent (a float ``b`` keeps the
#                                  scalar-overload pow, a 0-dim tensor ``b``
#                                  uses the tensor-tensor pow — the two
#                                  kernels differ in low-order bits at some
#                                  finite x and in the -0 / -inf conventions)
#   * true divisions           ->  ``grad / c`` (true division, not
#                                  ``grad * (1/c)``)
#
# ``torch.where(mask, a, b)``'s backward gates each branch with a *where*
# (not a multiply): the discarded branch's gradient is exactly ``+0``, so the
# zone chains gate with ``torch.where(mask, grad, zeros)`` rather than
# ``grad * mask`` (which would give ``-0`` for a negative ``grad``).
#
# The half-slope ``0.5`` at the ``minimum`` / ``maximum`` tie is written as
# ``0.5 (+/-) 0.5 * sign(x - c)`` (one ``sign`` each) rather than nested
# ``torch.where`` masks, to keep the hot path lean; the two forms agree to
# the last bit (``sign`` returns exactly ``-1`` / ``0`` / ``+1``).  The NaN
# input is routed to the ``1`` side with an explicit ``where(x.isnan(), ...)``
# because ``sign(nan)`` is NaN, not a valid slope.


def _d_min(x: Tensor, c: Scalar) -> Tensor:
    """Subgradient of ``torch.minimum(x, c)`` w.r.t. ``x`` (autograd-matching).

    ``1`` where ``x < c``, ``0.5`` at the tie ``x == c``, ``0`` where
    ``x > c``, and ``1`` for ``x = nan`` (autograd routes a NaN comparison
    to the ``x`` side).
    """
    return torch.where(x.isnan(), 1.0, 0.5 - 0.5 * torch.sign(x - c))


def _d_max(x: Tensor, c: Scalar) -> Tensor:
    """Subgradient of ``torch.maximum(x, c)`` w.r.t. ``x`` (autograd-matching).

    ``0`` where ``x < c``, ``0.5`` at the tie ``x == c``, ``1`` where
    ``x > c``, and ``1`` for ``x = nan``.
    """
    return torch.where(x.isnan(), 1.0, 0.5 + 0.5 * torch.sign(x - c))


def _d_clamp_max(x: Tensor, c: Scalar) -> Tensor:
    """Subgradient of ``torch.clamp_max(x, c)`` w.r.t. ``x`` (autograd-matching).

    Mirrors the *dispatch* of the forward threshold: a Python ``float`` ``c``
    runs the scalar-threshold kernel, a hard step — ``1`` where ``x < c``,
    ``0`` where ``x >= c`` (the tie, NaN, and ``+inf`` all fall on the flat
    side), ``1`` where ``x == -inf``.  Any ``Tensor`` ``c`` (0-dim or
    broadcastable) runs the tensor-threshold kernel, whose backward is
    bit-identical to ``torch.minimum``'s subgradient (:func:`_d_min`):
    ``1`` / ``0.5`` (tie) / ``0`` with ``nan -> 1``.
    """
    if isinstance(c, Tensor):
        return _d_min(x, c)
    return torch.where(x < c, torch.ones_like(x), torch.zeros_like(x))


def _d_clamp_min(x: Tensor, c: Scalar) -> Tensor:
    """Subgradient of ``torch.clamp_min(x, c)`` w.r.t. ``x`` (autograd-matching).

    Mirrors the *dispatch* of the forward threshold: a Python ``float`` ``c``
    runs the scalar-threshold kernel, a hard step whose tie belongs to the
    flat side — ``1`` where ``x > c``, ``0`` where ``x <= c`` (the tie, NaN,
    and ``-inf`` all fall on the flat side), ``1`` where ``x == +inf``.  Any
    ``Tensor`` ``c`` (0-dim or broadcastable) runs the tensor-threshold
    kernel, whose backward is bit-identical to ``torch.maximum``'s
    subgradient (:func:`_d_max`): ``0`` / ``0.5`` (tie) / ``1`` with
    ``nan -> 1``.
    """
    if isinstance(c, Tensor):
        return _d_max(x, c)
    return torch.where(x > c, torch.ones_like(x), torch.zeros_like(x))


def _d_pow(x: Tensor, b: Scalar) -> Tensor:
    """Slope ``b * x**(b-1)`` of the ``pow`` backward (autograd-matching).

    ``x ** b`` backpropagates as ``g * (b * x**(b-1))``: the slope is computed
    through ``torch.pow`` first (``0**negative -> +inf``, ``0**0 -> 1``,
    ``0**positive -> 0``, negative base with fractional exponent ``-> nan``),
    is multiplied by ``b`` first, and the upstream gradient ``g`` is applied
    by the caller *last* (pinned in ``scripts/parity_hydro_derivative.py``).

    The inner ``x ** (b - 1.0)`` mirrors the dispatch of the *forward*
    exponent with no branching: a Python ``float`` ``b`` (the default for
    every zone parameter) keeps the scalar-overload ``pow``, while a 0-dim
    ``Tensor`` ``b`` uses the tensor-tensor ``pow`` — the two kernels differ
    in low-order bits at some finite ``x`` and in the ``-0`` / ``-inf``
    conventions, so mirroring the forward's dispatch is what makes the slope
    bit-identical to autograd's.
    """
    return b * (x ** (b - 1.0))


# ---------------------------------------------------------------------- #
# Batched damped Newton storage solve (analytic derivative; AD fallback)
# ---------------------------------------------------------------------- #


def _candidate_value(
    f: Callable[[Tensor], Tensor | tuple[Tensor, Tensor]],
    f_value: Callable[[Tensor], Tensor] | None,
    candidate: Tensor,
    analytic: bool,
) -> Tensor:
    """Residual value at a backtracking candidate (decrease-test evaluation).

    The Armijo decrease test needs only ``|f(candidate)|`` — never the
    derivative — so this picks the cheapest available residual evaluator:

    * in the analytic pair mode (``analytic``) with an ``f_value`` companion
      supplied, it evaluates ``f_value(candidate)`` directly, skipping the
      zone's ``residual_deriv`` chain that ``f`` would otherwise recompute for
      a value-only use (a candidate's derivative is wasted work whenever the
      decrease test discards the candidate);
    * otherwise it evaluates ``f(candidate)`` and, if ``f`` returns a
      ``(residual, derivative)`` pair, unwraps the residual.  A bare-residual
      ``f`` (the generic / autograd path) is already value-only, so this is
      exactly the legacy decrease-test evaluation.

    ``f_value`` must return the same residual as ``f``'s first element for the
    same input; the zone hot path satisfies this by calling the identical
    ``_hydro_residual`` in both closures.  The caller invokes this under
    ``torch.no_grad``.
    """
    if analytic and f_value is not None:
        return f_value(candidate)
    out: Tensor | tuple[Tensor, Tensor] = f(candidate)
    return out[0] if isinstance(out, tuple) else out


def _newton_storage_solve(
    f: Callable[[Tensor], Tensor | tuple[Tensor, Tensor]],
    s_0: Tensor,
    max_iter: int = NEWTON_MAX_ITER,
    f_value: Callable[[Tensor], Tensor] | None = None,
) -> Tensor:
    """Solve ``f(s) = 0`` with a batched, damped Newton iteration.

    The driver is the notebook pattern (``scripts/hydrologic_zone_ode_ad.ipynb``)
    plus an Armijo-style backtracking guard on the residual magnitude.  Each
    trip evaluates the residual ``r = f(s)`` at the current trial and pairs it
    with a derivative ``dr``:

    * if ``f`` returns a ``(residual, derivative)`` pair (the zone hot path),
      the derivative is the zone's *analytic* :meth:`~HydrologicZone.residual_deriv`
      and every evaluation runs under ``torch.no_grad`` — no graph is built;
    * if ``f`` returns a bare residual tensor (a generic test residual), the
      derivative is obtained with ``torch.autograd.grad`` from a fresh
      differentiable trial (the legacy automatic-differentiation path).

    The driver then floors a near-zero (or NaN) derivative to
    ``NEWTON_MIN_DERIV`` (sign-preserving) and forms the Newton direction
    ``r / dr``.  The step is accepted by a *per-lane* Armijo step-halving
    search run under ``torch.no_grad``: each lane starts from its own
    ``alpha = 1`` and keeps halving only its own ``alpha`` until its candidate
    ``(s - alpha*step).clamp_min(0.0)`` satisfies the sufficient-decrease
    condition ``|f| <= NEWTON_BACKTRACK_C*|r|`` in that lane (at most
    ``NEWTON_MAX_BACKTRACK`` halvings); a lane that never satisfies keeps the
    smallest-``|f|`` candidate it observed (NaN candidates are never
    preferred).  Lanes therefore accept at their own pace and the search ends
    once every lane has accepted, instead of the whole batch being held to the
    strictest lane's halving cadence as with a shared scalar ``alpha`` (for a
    one-lane batch the two schedules are bit-identical).  The decrease test
    needs only the residual, so candidates are evaluated through the optional
    residual-only ``f_value`` when ``f`` is the analytic pair, and through
    ``f`` itself otherwise (a bare-residual ``f`` is already value-only).
    Lanes that already meet ``|r| <= atol + rtol*|s|`` are frozen by
    ``torch.where`` and stop updating; the loop breaks once every lane has
    converged.

    The damping is required for this model, not an optimisation.  Flux laws
    such as the groundwater baseflow ``k * s**alpha`` (``0 < alpha < 1``) are
    kinked at ``s = 0``: the analytic one-sided slope ``k*alpha*s**(alpha-1)``
    is infinite there, while the subgradient on the clamped ``clamp_min(0.0)``
    boundary is ``0``.  An undamped full Newton step from a near-zero trial
    therefore jumps across the kink to the opposite side, and the iterate
    2-cycles between the two sides forever (e.g. the ``GroundZoneB`` residual
    ``f(s) = 1.0344 - s - 2.5119*s**0.4`` alternates ``s = 1.0344 <-> 0``).
    Backtracking breaks the cycle: for continuous ``f`` with ``r != 0``,
    ``|f(s - alpha*step)| -> |r|`` as ``alpha -> 0``, so some halved step
    reduces ``|f|`` and the per-trip candidate only moves down the residual,
    while a genuine non-convergence (flat or NaN residual) still exhausts the
    budget and raises.

    Args:
        f:      storage residual ``(...,) -> (...,)`` or
                ``(...,) -> ((...,), (...,))`` returning ``(r, dr)``.  A bare
                tensor return is differentiated with ``torch.autograd.grad``;
                a pair is used verbatim (no graph).  Either way it must accept
                the same tensor shape for its backtracking re-evaluations.
        s_0:    warm-start guess (any shape); detached and cloned internally.
        max_iter: per-lane trip budget; exhausting it raises.
        f_value: optional residual-only companion ``(...,) -> (...,)`` used for
                 the backtracking *candidate* evaluations in the analytic pair
                 mode: the decrease test needs only ``|f|``, never the
                 derivative, so this skips the ``residual_deriv`` chain ``f``
                 would otherwise recompute for each (mostly discarded)
                 candidate.  It must return the same residual as ``f``'s first
                 element for the same input (the zone hot path passes a
                 closure over the identical ``_hydro_residual``).  Ignored when
                 ``f`` is a bare residual (already value-only) or when not
                 supplied.

    Returns:
        The storage root(s), detached (never carries a grad graph into the
        next step).

    Raises:
        ScalarRootFindingError: when any lane (including NaN residuals) has
            not converged after ``max_iter`` trips.
    """
    s: Tensor = s_0.detach().clone()
    atol: float
    rtol: float
    atol, rtol = _solver_tols(s.dtype)
    analytic: bool | None = None
    for _ in range(max_iter):
        if analytic is None:
            # First trip: make the trial differentiable and evaluate once.  A
            # ``(r, dr)`` pair return means the caller supplies the analytic
            # derivative (the graph built here is unused, a one-time cost); a
            # bare tensor return is differentiated through that same graph, so
            # neither mode pays an extra ``f`` evaluation.
            s = s.detach().requires_grad_(True)
            out: Tensor | tuple[Tensor, Tensor] = f(s)
            if isinstance(out, tuple):
                analytic = True
                r: Tensor = out[0]
                deriv: Tensor = out[1]
            else:
                analytic = False
                r = out
                deriv = torch.autograd.grad(
                    outputs=r,
                    inputs=s,
                    grad_outputs=torch.ones_like(r),
                    create_graph=False,
                )[0]
        elif analytic:
            # Stripping the grad flag keeps every later trip graph-free.
            s = s.detach()
            out = f(s)
            assert isinstance(out, tuple)
            r = out[0]
            deriv = out[1]
        else:
            s = s.detach().requires_grad_(True)
            r = f(s)
            deriv = torch.autograd.grad(
                outputs=r,
                inputs=s,
                grad_outputs=torch.ones_like(r),
                create_graph=False,
            )[0]
        # Near-zero (or NaN) derivatives are floored to a sign-preserving
        # ``NEWTON_MIN_DERIV`` so a flat residual cannot blow up the update.
        safe_deriv: Tensor = torch.where(
            deriv.abs() >= NEWTON_MIN_DERIV,
            deriv,
            torch.where(
                deriv >= 0,
                torch.full_like(deriv, NEWTON_MIN_DERIV),
                torch.full_like(deriv, -NEWTON_MIN_DERIV),
            ),
        )
        newton_step: Tensor = r / safe_deriv
        # Armijo-style backtracking on the residual magnitude, mirroring the
        # RT solver's line search (``solvers.py``) but run *per lane*: each
        # lane carries its own halving schedule (a shape-``s`` ``alpha``
        # tensor) and freezes on the smallest-``|f|`` candidate it observed
        # once that lane's candidate satisfies the decrease condition, so
        # lanes accept at their own pace.  A shared scalar ``alpha`` would
        # couple the lanes and force the whole batch to keep halving (and
        # re-evaluating) while any one lane is still searching; the per-lane
        # schedule ends the trip as soon as every lane has accepted.  As in
        # the RT search, every evaluation is a plain value call under
        # ``torch.no_grad`` (the analytic ``f`` ignores gradients by
        # construction and the AD ``f`` only needs the value for the test),
        # and the candidate is evaluated through :func:`_candidate_value` so
        # the analytic mode never recomputes the derivative of a candidate.
        alpha: Tensor = torch.ones_like(s)
        accepted: Tensor = torch.zeros_like(s, dtype=torch.bool)
        with torch.no_grad():
            candidate: Tensor = (s - alpha * newton_step).clamp_min(0.0)
            fc: Tensor = _candidate_value(f, f_value, candidate, analytic is True)
            best_cand: Tensor = candidate
            best_fc: Tensor = fc
            improved: Tensor = fc.abs() <= NEWTON_BACKTRACK_C * r.abs()
            accepted = accepted | improved
            for _ in range(NEWTON_MAX_BACKTRACK):
                if bool(accepted.all()):
                    break
                alpha = torch.where(accepted, alpha, alpha * 0.5)
                candidate = torch.where(
                    accepted,
                    best_cand,
                    (s - alpha * newton_step).clamp_min(0.0),
                )
                fc = _candidate_value(f, f_value, candidate, analytic is True)
                better: Tensor = fc.abs() < best_fc.abs()
                best_cand = torch.where(better, candidate, best_cand)
                best_fc = torch.where(better, fc, best_fc)
                improved = fc.abs() <= NEWTON_BACKTRACK_C * r.abs()
                accepted = accepted | improved
        converged: Tensor = r.abs() <= atol + rtol * s.abs()
        s = torch.where(converged, s, best_cand)
        # One host read of a 0-dim reduction per trip (plus the per-trip
        # ``improved.all()`` break above; the same sync profile as the legacy
        # masked loops); frozen lanes are idempotent, so stopping once all
        # lanes are converged is exact.
        if bool(converged.all()):
            break
    else:
        raise ScalarRootFindingError("Failed to find root in Python step")
    return s.detach()


# ---------------------------------------------------------------------- #
# Zone classes
# ---------------------------------------------------------------------- #


@dataclass(eq=False)
class HydrologicZone:
    """Base class for a water-storage zone (Rust ``HydrologicZone`` pyclass).

    Subclasses override the individual flux methods; the base returns ``0.0``
    for every flux (a fully conserving, inert zone).  The single scalar state
    is ``s`` [mm] of water storage.

    Zones are dataclasses: tunable parameters are declared as fields by
    subclasses (with defaults, so ``SurfaceZone()`` is valid), and the Rust
    introspection py-methods (:meth:`parameter_names`, :meth:`num_parameters`,
    :meth:`param_list`, :meth:`from_array`) are derived from
    :func:`dataclasses.fields`.  ``eq=False`` keeps identity equality/hashing
    (zones are never value-compared).
    """

    #: Unique zone identifier (Rust ``__name`` field); ``None`` resolves to
    #: :meth:`base_name` in :meth:`__post_init__`.
    name: str | None = None

    def __post_init__(self) -> None:
        # Invoked through the MRO by every subclass (whether or not it
        # declares its own), so the base default name is always resolved.
        if self.name is None:
            self.name = self.__class__.base_name()

    # ---- flux methods (broadcastable tensor arithmetic) -------------- #

    def forc_flux(self, s: Scalar, d: HydroForcing) -> Tensor:
        """Forcing flux retained by the zone (e.g. snowfall) [mm/day]."""
        return torch.zeros_like(_t(s))

    def vap_flux(self, s: Scalar, d: HydroForcing) -> Tensor:
        """Evaporative / evapotranspiration loss [mm/day]."""
        return torch.zeros_like(_t(s))

    def lat_flux(self, s: Scalar, d: HydroForcing) -> Tensor:
        """Lateral outflow carrying solutes [mm/day]."""
        return torch.zeros_like(_t(s))

    def vert_flux(self, s: Scalar, d: HydroForcing) -> Tensor:
        """Vertical outflow carrying solutes [mm/day]."""
        return torch.zeros_like(_t(s))

    def lat_flux_ext(self, s: Scalar, d: HydroForcing) -> Tensor:
        """Total lateral outflow (defaults to the internal lateral flux)."""
        return self.lat_flux(s, d)

    def vert_flux_ext(self, s: Scalar, d: HydroForcing) -> Tensor:
        """Total vertical outflow (defaults to the internal vertical flux)."""
        return self.vert_flux(s, d)

    def mass_balance(self, s: Scalar, d: HydroForcing) -> Tensor:
        """Net water balance ``q_in + forc - vap - lat - vert`` [mm/day]."""
        return (
            _t(d.q_in)
            + self.forc_flux(s, d)
            - self.vap_flux(s, d)
            - self.lat_flux(s, d)
            - self.vert_flux(s, d)
        )

    def mass_balance_deriv(self, s: Scalar, d: HydroForcing) -> Tensor:
        """Analytic ``d(mass_balance)/ds`` (the water-balance Jacobian).

        The base zone is inert (every flux is constant in ``s``), so the
        derivative is zero.  Subclasses that override :meth:`mass_balance`
        override this method with their dt-folded flux chains evaluated at
        ``dt = 1.0``, matching ``torch.autograd.grad`` bit-for-bit at the
        non-smooth points (see the module docstring and
        ``scripts/parity_hydro_derivative.py``).
        """
        return torch.zeros_like(_t(s))

    def residual_deriv(self, s: Scalar, s_0: Scalar, d: HydroForcing, dt: Scalar) -> Tensor:
        """Analytic derivative of :meth:`residual` w.r.t. ``s``.

        ``df/ds = -1 + dt * d(mass_balance)/ds``: the leading ``-s``
        contributes ``-1`` and the balance term is scaled by ``dt``.  For the
        inert base zone the balance derivative is zero, so this exact form
        (``dt*0 - 1``) matches autograd bit-for-bit for every ``dt``.
        Flux-carrying zones override this method with their dt-*folded*
        chains — each flux's chain carries its own ``-dt`` at the position
        the autograd backward places it — because scaling the *summed*
        ``d(mass_balance)/ds`` by ``dt`` afterwards is not bit-exact for
        non-dyadic ``dt`` (a dyadic ``dt`` scales exactly and the two forms
        coincide).  This is the companion of :meth:`residual` on the
        damped-Newton driver's ``(r, dr)`` hot path (no
        ``torch.autograd.grad``, no graph).
        """
        return _t(dt) * self.mass_balance_deriv(s, d) - 1.0

    def residual(self, s: Scalar, s_0: Scalar, d: HydroForcing, dt: Scalar) -> Tensor:
        """Implicit-Euler residual ``f(s) = (s_0 - s) + dt*mass_balance(s, d)``.

        The leading ``-s`` makes ``f`` depend on ``s`` for *every* zone
        (``df/ds = -1 + dt*d(mass_balance)/ds``; see :meth:`residual_deriv`).
        """
        return (_t(s_0) - _t(s)) + _t(dt) * self.mass_balance(s, d)

    # ---- parameter introspection (Rust pymethods) -------------------- #

    def param_list(self) -> list[Scalar]:
        """Ordered parameter values (Rust ``param_list``)."""
        return [getattr(self, n) for n in self.parameter_names()]

    @classmethod
    def parameter_names(cls) -> list[str]:
        """Ordered parameter names (Rust ``parameter_names``).

        Derived from :func:`dataclasses.fields`: every field of this class
        except the inherited ``name``.
        """
        return [f.name for f in fields(cls) if f.name != "name"]

    @classmethod
    def num_parameters(cls) -> int:
        """Number of tunable parameters (Rust ``num_parameters``)."""
        return len(cls.parameter_names())

    @classmethod
    def base_name(cls) -> str:
        """Class default zone name (Rust ``base_name``)."""
        return "unnamed"

    @classmethod
    def default_parameter_range(cls) -> dict[str, tuple[float, float]]:
        """Default calibration ranges keyed by parameter name."""
        return {}

    @classmethod
    def default_init_state(cls) -> float:
        """Default initial storage [mm] (Rust ``default_init_state``)."""
        return 0.0

    @classmethod
    def from_array(cls, arr: Tensor) -> HydrologicZone:
        """Build a zone from a parameter array (Rust ``from_array``).

        Maps ``arr[i]`` onto the ``i``-th :meth:`parameter_names` field.
        Ground zones override this to unpack the drainage coefficient from its
        log10 natural scale.
        """
        if arr.shape[0] != cls.num_parameters():
            raise ValueError("Wrong number of arguments")
        kwargs: dict[str, Any] = dict(
            zip(cls.parameter_names(), (float(v) for v in arr))
        )
        return cls(**kwargs)

    @classmethod
    def from_dict(cls, params: dict[str, float]) -> HydrologicZone:
        raise NotImplementedError(
            f"{cls.__name__}.from_dict is not implemented in the batched port; "
            "construct the zone directly or use from_array"
        )

    # ---- stepping ---------------------------------------------------- #

    def step(self, s_0: Scalar, d: HydroForcing, dt: Scalar = 1.0) -> HydroStep:
        """Solve one implicit-Euler storage step (Rust ``step`` pymethod).

        Governing equation::

            f(s) = (s_0 - s) + dt * mass_balance(s, d) = 0

        solved for the new storage ``s`` with the batched damped Newton
        driver (:func:`_newton_storage_solve`) using the analytic
        :meth:`residual_deriv` (no ``torch.autograd.grad`` on the hot path);
        the state carried out is detached and clamped to ``s >= 0``.

        Args:
            s_0: initial storage [mm] (scalar or broadcastable tensor).
            d:   forcing for this step.
            dt:  step length [days].

        Returns:
            A :class:`HydroStep` whose fields carry the leading shape of ``s_0``.

        Raises:
            ScalarRootFindingError: when any lane fails to converge within
                ``NEWTON_MAX_ITER`` trips.
        """
        ref: Tensor = _t(s_0)
        s0: Tensor = _t(s_0).to(dtype=ref.dtype, device=ref.device)
        dt_t: Tensor = _t(dt).to(dtype=ref.dtype, device=ref.device)
        d_t: HydroForcing = HydroForcing(
            precip=_t(d.precip).to(dtype=ref.dtype, device=ref.device),
            temp=_t(d.temp).to(dtype=ref.dtype, device=ref.device),
            pet=_t(d.pet).to(dtype=ref.dtype, device=ref.device),
            q_in=_t(d.q_in).to(dtype=ref.dtype, device=ref.device),
        )

        def f(s: Tensor) -> tuple[Tensor, Tensor]:
            # ``s`` is the Newton trial state; ``s0`` / ``d_t`` / ``dt_t`` are
            # plain constant tensors fixed for this step.  Returning the
            # analytic ``(r, dr)`` pair (no ``torch.autograd.grad``) keeps the
            # driver graph-free on every trip after the one-time mode probe.
            r: Tensor = _hydro_residual(self, s, s0, d_t, dt_t)
            dr: Tensor = _hydro_residual_deriv(self, s, s0, d_t, dt_t)
            return r, dr

        def f_value(s: Tensor) -> Tensor:
            # Residual-only companion for the driver's Armijo *candidate*
            # evaluations: the decrease test needs only ``|f|``, never the
            # derivative, so in the analytic mode candidates run this
            # expression instead of the full ``(r, dr)`` pair, skipping the
            # ``residual_deriv`` chain each (mostly discarded) candidate would
            # otherwise recompute.  It is the identical residual to ``f``'s
            # first element, so candidate values stay bit-identical.
            return _hydro_residual(self, s, s0, d_t, dt_t)

        s_new: Tensor = _newton_storage_solve(f, s0, f_value=f_value)
        s_new = s_new.maximum(torch.zeros_like(s_new))  # clamp to >= 0

        return HydroStep(
            state=s_new,
            forc_flux=self.forc_flux(s_new, d_t),
            lat_flux=self.lat_flux(s_new, d_t),
            vert_flux=self.vert_flux(s_new, d_t),
            vap_flux=self.vap_flux(s_new, d_t),
            q_in=d_t.q_in,
            lat_flux_ext=self.lat_flux_ext(s_new, d_t),
            vert_flux_ext=self.vert_flux_ext(s_new, d_t),
        )


def _hydro_residual(zone: HydrologicZone, s: Scalar, s_0: Scalar, d: HydroForcing, dt: Scalar) -> Tensor:
    """Call the zone's own :meth:`HydrologicZone.residual` (virtual dispatch).

    The damped Newton driver calls this eagerly at every trip (no
    ``torch.compile`` wrapping of any kind).  It is a plain dispatching
    function rather than a module alias of ``HydrologicZone.residual``: an
    alias pins the *base* method object, which is fine for the residual
    itself (its body is generic — the ``self.mass_balance`` call dispatches
    dynamically) but **not** for the derivative companion below (the base
    body scales the summed balance derivative by ``dt`` instead of running
    the zone's dt-folded chain), so per-zone overrides must be reachable
    through these entry points.
    """
    return zone.residual(s, s_0, d, dt)


def _hydro_residual_deriv(
    zone: HydrologicZone, s: Scalar, s_0: Scalar, d: HydroForcing, dt: Scalar
) -> Tensor:
    """Call the zone's own :meth:`HydrologicZone.residual_deriv` (virtual dispatch).

    See :func:`_hydro_residual`: a plain dispatching function (one frame of
    indirection per trip, negligible next to the torch ops) so the zone's
    dt-folded analytic chain — not the base's ``dt * d(mass_balance)/ds - 1``
    form — is what the damped Newton driver runs.
    """
    return zone.residual_deriv(s, s_0, d, dt)


# ---------------------------------------------------------------------- #
# SnowZone
# ---------------------------------------------------------------------- #


@dataclass(eq=False)
class SnowZone(HydrologicZone):
    """Snow accumulation / melt storage zone (Rust ``SnowZone``).

    Parameters:
        tt:   threshold temperature for snowfall vs melt [deg C] (default ``0.0``).
        fmax: maximum melt rate factor [mm/day per deg C] (default ``1.0``).
    """

    tt: float | Tensor = 0.0
    fmax: float | Tensor = 1.0

    def forc_flux(self, s: Scalar, d: HydroForcing) -> Tensor:
        """Snowfall: the full precipitation when ``temp <= tt``, else 0."""
        t: Tensor = _t(d.temp)
        p: Tensor = _t(d.precip)
        return torch.where(t <= self.tt, p, torch.zeros_like(t))

    def vert_flux(self, s: Scalar, d: HydroForcing) -> Tensor:
        """Melt: ``min(s, fmax*(temp - tt))`` when ``temp > tt``, else 0."""
        s_t: Tensor = _t(s)
        t: Tensor = _t(d.temp)
        melt: Tensor = self.fmax * (t - self.tt)
        return torch.where(t > self.tt, s_t.minimum(melt), torch.zeros_like(s_t))

    def vap_flux(self, s: Scalar, d: HydroForcing) -> Tensor:
        """No evaporation from snow."""
        return torch.zeros_like(_t(s))

    def lat_flux(self, s: Scalar, d: HydroForcing) -> Tensor:
        """No lateral flow from snow."""
        return torch.zeros_like(_t(s))

    def lat_flux_ext(self, s: Scalar, d: HydroForcing) -> Tensor:
        return self.lat_flux(s, d)

    def vert_flux_ext(self, s: Scalar, d: HydroForcing) -> Tensor:
        """Melt + rain-on-snow precipitation when melting, else melt only."""
        t: Tensor = _t(d.temp)
        p: Tensor = _t(d.precip)
        v: Tensor = self.vert_flux(s, d)
        return torch.where(t > self.tt, v + p, v)

    def mass_balance(self, s: Scalar, d: HydroForcing) -> Tensor:
        """Snow conserves: ``q_in + forc - vert`` (no vapor/lateral loss)."""
        return _t(d.q_in) + self.forc_flux(s, d) - self.vert_flux(s, d)

    def _mass_balance_deriv_prefolded(self, s: Scalar, d: HydroForcing, dt: Scalar) -> Tensor:
        """dt-folded derivative of the snow mass balance, autograd-matching.

        Governing equations (mirror of the autograd backward, op-for-op;
        ``t`` = air temperature, ``tt`` the melt threshold,
        ``melt = fmax * (t - tt)``)::

            mass_balance = q_in + forc - vert
            vert         = where(t > tt, min(s, melt), 0)

        Backward from the residual's ``dt * mass_balance`` node: the ``min``
        node's incoming gradient is the ``where``-backward of ``vert``, which
        gates with a *where* (discarded lane exactly ``+0``) and carries the
        ``-dt`` factor; the ``minimum`` subgradient :func:`_d_min` (half-slope
        at the ``s == melt`` tie) is multiplied last::

            M(s) = where(t > tt, -dt, 0) * d_min(s, melt)
        """
        s_t: Tensor = _t(s)
        t: Tensor = _t(d.temp)
        melt: Tensor = self.fmax * (t - self.tt)
        gate: Tensor = torch.where(t > self.tt, -_t(dt), torch.zeros_like(s_t))
        return gate * _d_min(s_t, melt)

    def mass_balance_deriv(self, s: Scalar, d: HydroForcing) -> Tensor:
        """d/ds of :meth:`mass_balance` (mirrors ``q_in + forc - vert``).

        The prefolded melt chain evaluated at ``dt = 1.0``: only the melt
        ``vert_flux = where(t > tt, min(s, fmax*(t-tt)), 0)`` depends on
        ``s``; its subgradient is ``where(t > tt, -1, 0) * d_min(s, melt)``
        with the ``minimum`` half-slope (``0.5``) at the ``s == melt`` tie
        (autograd-matching, see :func:`_d_min` and the module docstring).
        """
        return self._mass_balance_deriv_prefolded(s, d, 1.0)

    def residual_deriv(self, s: Scalar, s_0: Scalar, d: HydroForcing, dt: Scalar) -> Tensor:
        """Analytic ``df/ds`` of :meth:`residual` (the damped-Newton hot path).

        ``df/ds = M - 1`` where ``M`` is the dt-folded melt chain from
        :meth:`_mass_balance_deriv_prefolded`.  The grouping ``M + (-1)``
        mirrors the autograd accumulation order: the residual's
        ``dt * mass_balance`` node is created *after* its ``(s_0 - s)``
        sibling, so backward (reverse node-creation order) folds the melt
        chain into ``s``'s gradient buffer before the leading ``-1`` —
        required for bit-parity of the Newton iterates, not just values.
        """
        return self._mass_balance_deriv_prefolded(s, d, dt) + _t(-1.0)

    @classmethod
    def base_name(cls) -> str:
        return "snow"

    @classmethod
    def default_parameter_range(cls) -> dict[str, tuple[float, float]]:
        return {"tt": (-1.0, 1.0), "fmax": (0.5, 5.0)}

    @classmethod
    def default_init_state(cls) -> float:
        return 0.0

    def __repr__(self) -> str:
        return f"SnowZone(tt={self.tt:.2f},fmax={self.fmax:.2f})"


# ---------------------------------------------------------------------- #
# SurfaceZone
# ---------------------------------------------------------------------- #


@dataclass(eq=False)
class SurfaceZone(HydrologicZone):
    """Soil / land-surface storage zone (Rust ``SurfaceZone``).

    Parameters:
        fc:   field capacity / maximum storage [mm] (default ``100.0``).
        lp:   limiting factor for PET, fraction of ``fc`` (default ``0.5``).
        beta: precipitation-partitioning exponent (default ``1.0``).
        k0:   lateral drainage coefficient [1/day] (default ``0.1``).
        thr:  lateral-drainage threshold storage [mm] (default ``10.0``).
    """

    fc: float | Tensor = 100.0
    lp: float | Tensor = 0.5
    beta: float | Tensor = 1.0
    k0: float | Tensor = 0.1
    thr: float | Tensor = 10.0

    def forc_flux(self, s: Scalar, d: HydroForcing) -> Tensor:
        """No additional forcing flux for the surface zone."""
        return torch.zeros_like(_t(s))

    def vap_flux(self, s: Scalar, d: HydroForcing) -> Tensor:
        """Actual ET = ``pet * min(s/(fc*lp), 1)``."""
        s_t: Tensor = _t(s)
        pet: Tensor = _t(d.pet)
        return pet * (s_t / (self.fc * self.lp)).minimum(torch.ones_like(s_t))

    def lat_flux(self, s: Scalar, d: HydroForcing) -> Tensor:
        """Quickflow: ``max(k0 * (s - thr), 0)``."""
        s_t: Tensor = _t(s)
        return (self.k0 * (s_t - self.thr)).maximum(torch.zeros_like(s_t))

    def vert_flux(self, s: Scalar, d: HydroForcing) -> Tensor:
        """Percolation: ``q_in * (s / fc)**beta``."""
        s_t: Tensor = _t(s)
        qin: Tensor = _t(d.q_in)
        return qin * (s_t / self.fc) ** self.beta

    def _mass_balance_deriv_prefolded(self, s: Scalar, d: HydroForcing, dt: Scalar) -> Tensor:
        """dt-folded derivative of the surface mass balance, autograd-matching.

        Governing equations (mirror of the autograd backward, op-for-op;
        ``x_pc = s / fc``, ``x_et = s / (fc * lp)``)::

            mass_balance = q_in - vap - lat - vert      (forc == 0)
            vap  = pet * min(x_et, 1)
            lat  = max(k0 * (s - thr), 0)
            vert = q_in * x_pc ** beta

        Each chain carries its ``-dt`` factor from the residual's
        ``dt * mass_balance`` node at the position the autograd backward
        places it (the balance sub-nodes negate the upstream gradient before
        the flux's own backward runs)::

            V = ((-dt*q_in) * (beta*x_pc**(beta-1))) / fc
            L = ((-dt)      * d_max(k0*(s-thr), 0))   * k0
            P = ((-dt*pet)  * d_min(x_et, 1))         / (fc*lp)

        ``V`` and ``P`` use the ``pow`` / ``minimum`` backpropagations
        (:func:`_d_pow` — ``b * x**(b-1)`` through ``torch.pow``, slope
        multiplied by ``b`` first, upstream gradient last; :func:`_d_min` —
        half-slope at the ``x_et == 1`` tie), ``L`` the ``maximum``
        subgradient :func:`_d_max` (half-slope at the ``s == thr`` kink), and
        the divisions backpropagate as true divisions (``grad / c``, not
        ``grad * (1/c)``).
        """
        s_t: Tensor = _t(s)
        dt_neg: Tensor = -_t(dt)
        pet: Tensor = _t(d.pet)
        qin: Tensor = _t(d.q_in)
        x_pc: Tensor = s_t / self.fc
        x_lat: Tensor = self.k0 * (s_t - self.thr)
        x_et: Tensor = s_t / (self.fc * self.lp)
        v: Tensor = ((dt_neg * qin) * _d_pow(x_pc, self.beta)) / self.fc
        l: Tensor = (dt_neg * _d_max(x_lat, 0.0)) * self.k0
        p: Tensor = ((dt_neg * pet) * _d_min(x_et, 1.0)) / (self.fc * self.lp)
        # Autograd accumulates into ``s`` in reverse node-creation order:
        # the vertical-flux chain first, then lateral, then vapor.
        return (v + l) + p

    def mass_balance_deriv(self, s: Scalar, d: HydroForcing) -> Tensor:
        """d/ds of the base :meth:`~HydrologicZone.mass_balance` (mirrors
        ``q_in - vap - lat - vert``; ``forc`` is identically zero here).

        The prefolded chains evaluated at ``dt = 1.0``, written to match
        ``torch.autograd.grad`` bit-for-bit at each kink (``minimum`` /
        ``maximum`` half-slopes at their ties, ``pow`` backward
        ``b*x**(b-1)`` through ``torch.pow``, true divisions ``grad / c`` —
        see the module docstring and
        ``scripts/parity_hydro_derivative.py``):

        * vap  ``pet * min(s/(fc*lp), 1)``
          -> ``(-1*pet) * d_min(x_et, 1) / (fc*lp)``
        * lat  ``max(k0*(s-thr), 0)``
          -> ``(-1) * d_max(k0*(s-thr), 0) * k0``
        * vert ``q_in * (s/fc)**beta``
          -> ``(-1*q_in) * (beta*(s/fc)**(beta-1)) / fc``
        """
        return self._mass_balance_deriv_prefolded(s, d, 1.0)

    def residual_deriv(self, s: Scalar, s_0: Scalar, d: HydroForcing, dt: Scalar) -> Tensor:
        """Analytic ``df/ds`` of :meth:`residual` (the damped-Newton hot path).

        ``df/ds = (V + L) + P + (-1)`` with ``V`` / ``L`` / ``P`` the dt-
        folded flux chains of :meth:`_mass_balance_deriv_prefolded` (vertical
        first, then lateral, then vapor) and the leading ``-s`` contributing
        the final ``-1`` — exactly the autograd accumulation order (reverse
        node-creation order), required for bit-parity of the Newton iterates
        (see the module docstring for the pinned kink conventions).
        """
        return self._mass_balance_deriv_prefolded(s, d, dt) + _t(-1.0)

    def lat_flux_ext(self, s: Scalar, d: HydroForcing) -> Tensor:
        return self.lat_flux(s, d)

    def vert_flux_ext(self, s: Scalar, d: HydroForcing) -> Tensor:
        return self.vert_flux(s, d)

    @classmethod
    def base_name(cls) -> str:
        return "surface"

    @classmethod
    def default_parameter_range(cls) -> dict[str, tuple[float, float]]:
        return {
            "fc": (50.0, 1000.0),
            "lp": (0.05, 1.0),
            "beta": (0.05, 5.0),
            "k0": (0.0, 1.0),
            "thr": (0.0, 1000.0),
        }

    @classmethod
    def default_init_state(cls) -> float:
        return 25.0

    def __repr__(self) -> str:
        return (
            f"SurfaceZone(fc={self.fc:.2f},lp={self.lp:.2f},beta={self.beta:.2f},"
            f"k0={self.k0:.2f},thr={self.thr:.2f})"
        )


# ---------------------------------------------------------------------- #
# GroundZone
# ---------------------------------------------------------------------- #


@dataclass(eq=False)
class GroundZone(HydrologicZone):
    """Groundwater storage zone with percolation (Rust ``GroundZone``).

    Parameters:
        k:     baseflow drainage coefficient [1/day] (default ``1e-2``).
        alpha: non-linearity exponent of the drainage law (default ``1.0``).
        perc:  maximum percolation to the deeper zone [mm/day] (default ``1.0``).
    """

    k: float | Tensor = 1e-2
    alpha: float | Tensor = 1.0
    perc: float | Tensor = 1.0

    def forc_flux(self, s: Scalar, d: HydroForcing) -> Tensor:
        return torch.zeros_like(_t(s))

    def vap_flux(self, s: Scalar, d: HydroForcing) -> Tensor:
        return torch.zeros_like(_t(s))

    def lat_flux(self, s: Scalar, d: HydroForcing) -> Tensor:
        """Baseflow: ``k * max(s, 0)**alpha`` when ``s >= 1e-12``, else 0."""
        s_t: Tensor = _t(s)
        active: Tensor = s_t >= 1e-12
        baseflow: Tensor = self.k * s_t.clamp_min(0.0) ** self.alpha
        return torch.where(active, baseflow, torch.zeros_like(s_t))

    def vert_flux(self, s: Scalar, d: HydroForcing) -> Tensor:
        """Percolation ``min(s, perc)`` to the deeper zone."""
        s_t: Tensor = _t(s)
        return s_t.clamp_max(self.perc)

    def lat_flux_ext(self, s: Scalar, d: HydroForcing) -> Tensor:
        return self.lat_flux(s, d)

    def vert_flux_ext(self, s: Scalar, d: HydroForcing) -> Tensor:
        return self.vert_flux(s, d)

    def mass_balance(self, s: Scalar, d: HydroForcing) -> Tensor:
        return _t(d.q_in) - self.vert_flux(s, d) - self.lat_flux(s, d)

    def _mass_balance_deriv_prefolded(self, s: Scalar, d: HydroForcing, dt: Scalar) -> Tensor:
        """dt-folded derivative of the ground mass balance, autograd-matching.

        Governing equations (mirror of the autograd backward, op-for-op;
        ``u = clamp_min(s, 0)`` — a *scalar*-threshold clamp, whose backward
        is the hard-step where-gate ``where(s > 0, grad, 0)``)::

            mass_balance = q_in - vert - lat
            vert         = clamp_max(s, perc)
            lat          = where(s >= 1e-12, k * u ** alpha, 0)

            Lg = where(s > 0, ((where(s >= 1e-12, -dt, 0) * k)
                               * (alpha*u**(alpha-1))), 0)
            Vg = where(s < perc, -dt, 0)                          (float perc)
               = (-dt) * d_min(s, perc)                           (Tensor perc)

        Two backward structures must not be flattened into multiplications.
        First, the ``where``-backward of ``lat`` (discarded lane exactly
        ``+0``) and the ``clamp_min`` backward gate *after* the ``pow``
        node: for ``s <= 0`` the gate's ``+0`` meets the ``pow`` slope
        :func:`_d_pow` at ``u = 0``, which is ``+inf`` for ``alpha < 1`` —
        ``+0 * (+inf) = nan`` inside the chain, and only the *where* form of
        the clamp gate (``where(s > 0, ..., 0)``) drops that NaN exactly as
        autograd does (a multiply by ``_d_clamp_min`` would keep it, and a
        ``where(active, slope, 0)`` form would miss it on the ``0 < s <
        1e-12`` lanes where the inner chain is finite).  Second, the
        ``clamp_max`` backward of ``vert`` mirrors the forward threshold's
        dispatch (:func:`_d_clamp_max`): a ``float`` ``perc`` is the
        hard-step where-gate ``where(s < perc, -dt, 0)`` (tie/nan/``+inf``
        on the flat side), while a ``Tensor`` ``perc`` is the
        ``minimum``-style subgradient multiply ``(-dt) * d_min(s, perc)``
        (``0.5`` at the ``s == perc`` tie, ``nan -> 1``).
        """
        s_t: Tensor = _t(s)
        dt_neg: Tensor = -_t(dt)
        active: Tensor = s_t >= 1e-12
        u: Tensor = s_t.clamp_min(0.0)
        gate: Tensor = torch.where(active, dt_neg, torch.zeros_like(s_t))
        # ``u = 0`` (i.e. ``s <= 0``) makes the pow slope ``+inf`` for
        # alpha < 1, so the inner product below is ``nan`` there; the
        # scalar-threshold clamp_min backward is the where-gate ``s > 0``,
        # which discards it (discarded lane exactly ``+0``).
        lg_inner: Tensor = (gate * self.k) * _d_pow(u, self.alpha)
        lg: Tensor = torch.where(s_t > 0.0, lg_inner, torch.zeros_like(s_t))
        vg: Tensor = (
            dt_neg * _d_min(s_t, self.perc)
            if isinstance(self.perc, Tensor)
            else torch.where(s_t < self.perc, dt_neg, torch.zeros_like(s_t))
        )
        # Autograd accumulates into ``s`` in reverse node-creation order:
        # the lateral (baseflow) chain first, then vertical (percolation).
        return lg + vg

    def mass_balance_deriv(self, s: Scalar, d: HydroForcing) -> Tensor:
        """d/ds of :meth:`mass_balance` (mirrors ``q_in - vert - lat``).

        The prefolded chains evaluated at ``dt = 1.0``:

        * vert  ``clamp_max(s, perc)`` -> ``where(s < perc, -1, 0)``
          (float ``perc``) or ``-1 * d_min(s, perc)`` (Tensor ``perc``) —
          see :meth:`_mass_balance_deriv_prefolded`.
        * lat   ``where(s >= 1e-12, k*clamp_min(s,0)**alpha, 0)``
          -> ``where(s > 0, where(s >= 1e-12, -1, 0) * k *
             (alpha*u**(alpha-1)), 0)`` with ``u = clamp_min(s, 0)`` — the
          ``where``-backward gate (discarded lane ``+0``) *before* the
          ``pow`` slope, and the scalar-threshold ``clamp_min``
          where-gate *after* it (both applied with ``where``, never as a
          multiply, so the ``+0 * inf = nan`` on ``s <= 0`` lanes for
          ``alpha < 1`` is dropped exactly as in autograd).
        """
        return self._mass_balance_deriv_prefolded(s, d, 1.0)

    def residual_deriv(self, s: Scalar, s_0: Scalar, d: HydroForcing, dt: Scalar) -> Tensor:
        """Analytic ``df/ds`` of :meth:`residual` (the damped-Newton hot path).

        ``df/ds = (Lg + Vg) + (-1)`` with ``Lg`` / ``Vg`` the dt-folded
        baseflow / percolation chains of
        :meth:`_mass_balance_deriv_prefolded` (baseflow first) and the
        leading ``-s`` contributing the final ``-1`` — exactly the autograd
        accumulation order (reverse node-creation order), required for
        bit-parity of the Newton iterates.
        """
        return self._mass_balance_deriv_prefolded(s, d, dt) + _t(-1.0)

    @classmethod
    def base_name(cls) -> str:
        return "ground"

    @classmethod
    def default_parameter_range(
        cls, natural_scales: bool = True
    ) -> dict[str, tuple[float, float]]:
        k_range: tuple[float, float] = (
            (-5.0, -1.0) if natural_scales else (1e-5, 0.1)
        )
        return {"k": k_range, "alpha": (0.5, 3.0), "perc": (0.0, 5.0)}

    @classmethod
    def default_init_state(cls) -> float:
        return 10.0

    @classmethod
    def from_array(
        cls, arr: Tensor, natural_scales: bool = True
    ) -> GroundZone:
        if arr.shape[0] != cls.num_parameters():
            raise ValueError("Wrong number of arguments")
        k: float = 10.0 ** float(arr[0]) if natural_scales else float(arr[0])
        return cls(k=k, alpha=float(arr[1]), perc=float(arr[2]))

    def __repr__(self) -> str:
        return (
            f"GroundZone(k={self.k:.2e},alpha={self.alpha:.2f},"
            f"perc={self.perc:.2f})"
        )


# ---------------------------------------------------------------------- #
# GroundZoneB
# ---------------------------------------------------------------------- #


@dataclass(eq=False)
class GroundZoneB(HydrologicZone):
    """Deepest groundwater zone, no percolation (Rust ``GroundZoneB``).

    Parameters:
        k:     baseflow drainage coefficient [1/day] (default ``1e-2``).
        alpha: non-linearity exponent of the drainage law (default ``1.0``).
    """

    k: float | Tensor = 1e-2
    alpha: float | Tensor = 1.0

    def forc_flux(self, s: Scalar, d: HydroForcing) -> Tensor:
        return torch.zeros_like(_t(s))

    def vap_flux(self, s: Scalar, d: HydroForcing) -> Tensor:
        return torch.zeros_like(_t(s))

    def lat_flux(self, s: Scalar, d: HydroForcing) -> Tensor:
        """Baseflow: ``k * max(s, 0)**alpha`` when ``s >= 1e-12``, else 0."""
        s_t: Tensor = _t(s)
        active: Tensor = s_t >= 1e-12
        baseflow: Tensor = self.k * s_t.clamp_min(0.0) ** self.alpha
        return torch.where(active, baseflow, torch.zeros_like(s_t))

    def vert_flux(self, s: Scalar, d: HydroForcing) -> Tensor:
        return torch.zeros_like(_t(s))

    def lat_flux_ext(self, s: Scalar, d: HydroForcing) -> Tensor:
        return self.lat_flux(s, d)

    def vert_flux_ext(self, s: Scalar, d: HydroForcing) -> Tensor:
        return self.vert_flux(s, d)

    def mass_balance(self, s: Scalar, d: HydroForcing) -> Tensor:
        return _t(d.q_in) - self.lat_flux(s, d)

    def _mass_balance_deriv_prefolded(self, s: Scalar, d: HydroForcing, dt: Scalar) -> Tensor:
        """dt-folded derivative of the groundB mass balance, autograd-matching.

        Governing equations (mirror of the autograd backward, op-for-op;
        ``u = clamp_min(s, 0)`` — a *scalar*-threshold clamp, whose backward
        is the hard-step where-gate ``where(s > 0, grad, 0)``)::

            mass_balance = q_in - lat
            lat          = where(s >= 1e-12, k * u ** alpha, 0)

            Lg = where(s > 0, ((where(s >= 1e-12, -dt, 0) * k)
                               * (alpha*u**(alpha-1))), 0)

        The baseflow chain of
        :meth:`GroundZone._mass_balance_deriv_prefolded` without the
        percolation branch.  Both backward gates are applied with
        ``torch.where``, never as multiplications: the ``where``-backward
        gate (discarded lane exactly ``+0``) multiplies the ``pow`` slope
        :func:`_d_pow` *after* it is formed (an inactive lane with
        ``alpha < 1`` yields ``+0 * (+inf) = nan`` inside the chain, exactly
        as autograd), and the scalar-threshold ``clamp_min`` backward gate
        ``s > 0`` is what finally *drops* that NaN for ``s <= 0`` — a
        multiply by ``_d_clamp_min`` would keep it, and a
        ``where(active, slope, 0)`` form would miss the ``s <= 0`` lanes.
        """
        s_t: Tensor = _t(s)
        active: Tensor = s_t >= 1e-12
        u: Tensor = s_t.clamp_min(0.0)
        gate: Tensor = torch.where(active, -_t(dt), torch.zeros_like(s_t))
        # ``u = 0`` (i.e. ``s <= 0``) makes the pow slope ``+inf`` for
        # alpha < 1, so the inner product below is ``nan`` there; the
        # scalar-threshold clamp_min backward is the where-gate ``s > 0``,
        # which discards it (discarded lane exactly ``+0``).
        lg_inner: Tensor = (gate * self.k) * _d_pow(u, self.alpha)
        return torch.where(s_t > 0.0, lg_inner, torch.zeros_like(s_t))

    def mass_balance_deriv(self, s: Scalar, d: HydroForcing) -> Tensor:
        """d/ds of :meth:`mass_balance` (mirrors ``q_in - lat``).

        The prefolded baseflow chain evaluated at ``dt = 1.0``:
        ``lat = where(s >= 1e-12, k*clamp_min(s,0)**alpha, 0)`` ->
        ``where(s > 0, where(s >= 1e-12, -1, 0) * k *
        (alpha*u**(alpha-1)), 0)`` with ``u = clamp_min(s, 0)`` (identical
        chain to :meth:`GroundZone.mass_balance_deriv`'s baseflow term —
        both gates applied with ``where``, so the ``+0 * inf = nan`` on
        ``s <= 0`` lanes for ``alpha < 1`` is dropped exactly as in
        autograd).
        """
        return self._mass_balance_deriv_prefolded(s, d, 1.0)

    def residual_deriv(self, s: Scalar, s_0: Scalar, d: HydroForcing, dt: Scalar) -> Tensor:
        """Analytic ``df/ds`` of :meth:`residual` (the damped-Newton hot path).

        ``df/ds = Lg + (-1)`` with ``Lg`` the dt-folded baseflow chain of
        :meth:`_mass_balance_deriv_prefolded` and the leading ``-s``
        contributing the final ``-1`` — exactly the autograd accumulation
        order (reverse node-creation order), required for bit-parity of the
        Newton iterates.
        """
        return self._mass_balance_deriv_prefolded(s, d, dt) + _t(-1.0)

    @classmethod
    def base_name(cls) -> str:
        return "ground_bottom"

    @classmethod
    def default_parameter_range(
        cls, natural_scales: bool = True
    ) -> dict[str, tuple[float, float]]:
        k_range: tuple[float, float] = (
            (-5.0, -1.0) if natural_scales else (1e-5, 0.1)
        )
        return {"k": k_range, "alpha": (0.5, 3.0)}

    @classmethod
    def default_init_state(cls) -> float:
        return 10.0

    @classmethod
    def from_array(
        cls, arr: Tensor, natural_scales: bool = True
    ) -> GroundZoneB:
        if arr.shape[0] != cls.num_parameters():
            raise ValueError("Wrong number of arguments")
        k: float = 10.0 ** float(arr[0]) if natural_scales else float(arr[0])
        return cls(k=k, alpha=float(arr[1]))

    def __repr__(self) -> str:
        return f"GroundZone(k={self.k:.2e},alpha={self.alpha:.2f})"
