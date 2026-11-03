"""Batched root solvers mirroring the Rust solver chain in ``math.rs``.

The reference implements a Newton-with-analytic-Jacobian solve (Armijo
backtracking line search) followed by a Levenberg-Marquardt fallback that
resumes from Newton's best iterate.  Every numerical detail — the sufficient
decrease test, the step/decrease tolerances, the damping update, the stagnation
acceptance rule and the error metrics — is transcribed verbatim from ``math.rs``
so batched results agree with the scalar Rust kernels.

All callables operate on tensors with an explicit leading *batch* dimension:
residual functions take ``(B, m)``, Jacobians are ``(B, m, n)``.  Functions are
(usually) evaluated on the whole batch every iteration, while termination,
line-search halving, damping factors etc. are tracked *per element* with masks,
so each batch lane reproduces the scalar Rust control flow exactly.

Convergence is declared when the component-wise residual meets the two-term
bar ``max|f| <= atol + rtol * max|x|`` (the same contract as the hydro
AD-Newton solver in :mod:`potions.hydro`).  For float64 the bar is the
reference *pure absolute* one (``atol = MULTI_TOL``, ``rtol = 0``);
``MULTI_TOL_STEP`` (1e-12) backstops stalled iterates as a relative-step bar
and the relative stagnation bar accepts a stalled iterate whose best residual
is already small.  All four bars are float64-era constants;
:func:`_solver_tols` additionally clamps them above the working dtype's ulp
(and sets the f32 relative term ``MULTI_RTOL_F32``) so reduced-precision
(float32) runs terminate instead of oscillating below the convergence floor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TypeAlias, cast

import torch

Tensor: TypeAlias = torch.Tensor

MULTI_TOL: float = 1e-6
MULTI_MAXITER: int = 100
MULTI_TOL_STEP: float = 1e-12
#: Relative residual term for reduced-precision (f32) solves: the convergence
#: bar is the two-term form ``max|f| <= atol + rtol * max|x|`` (the hydro
#: two-term contract), because a pure absolute bar is unreachable at the f32
#: ulp floor for O(1)-scale iterates.
MULTI_RTOL_F32: float = 1e-6
ARMIJO_C: float = 1e-4
LINE_SEARCH_MAX: int = 30
STAGNATION_REL_TOL: float = 1e-9

# Per-element status codes for the masked solve machine.
_RUNNING: int = 0
_DONE: int = 1
_FAILED: int = 2


class RootFindingError(RuntimeError):
    """A root-finding solve did not converge (base class for all of them).

    Every solver exception in the package subclasses this (and therefore
    :class:`RuntimeError`), so callers that catch ``RuntimeError`` keep
    working: the batched Newton/LM solvers in this module raise
    :class:`BatchedRootFindingError`, and the scalar hydro AD-Newton driver
    raises ``hydro.ScalarRootFindingError``.
    """


class BatchedRootFindingError(RootFindingError):
    """A batched Newton/LM solve has at least one failed lane.

    Raised by :func:`newton_fused_then_lm` / :func:`newton_analytic_then_lm`
    with ``raise_on_failure=True`` when a batch lane does not meet the
    convergence bar in Newton *and* the Levenberg-Marquardt fallback.
    """

# A fused evaluation: residual (B, m) and analytic Jacobian (B, m, n) at once.
FusedEval: TypeAlias = Callable[[Tensor], tuple[Tensor, Tensor]]
# A plain vector-valued function (B, m).
VectorFn: TypeAlias = Callable[[Tensor], Tensor]
# An analytic Jacobian function (B, m, n).
JacobianFn: TypeAlias = Callable[[Tensor], Tensor]


@dataclass(frozen=True)
class NewtonOutcome:
    """Result of a masked Newton solve (before any LM fallback).

    Attributes:
        result:   (B, n) final iterate: current ``x`` for direct convergence,
                  ``best_x`` for the post-loop/stagnation accept, and the last
                  iterate for failed lanes (consumed by the LM fallback).
        best_x:   (B, n) lowest-residual iterate ever reached.
        best_err: (B,) component-wise max absolute residual at ``best_x``.
        status:   (B,) element status; ``_DONE`` or ``_FAILED`` (LM territory).
        err:      (B,) current ``max|f|`` after the loop (diagnostics only).
    """

    result: Tensor
    best_x: Tensor
    best_err: Tensor
    status: Tensor
    err: Tensor


def _solver_tols(
    dtype: torch.dtype,
    compute_dtype: torch.dtype | None = None,
) -> tuple[float, float, float, float]:
    """dtype-aware ``(atol, rtol, relative step bar, relative stagnation)``.

    The residual test is the two-term form ``max|f| <= atol + rtol *
    max|x|`` (the same contract as the hydro AD-Newton solver).  float64
    keeps the reference ``MULTI_TOL`` / ``MULTI_TOL_STEP`` /
    ``STAGNATION_REL_TOL`` constants exactly as a *pure absolute* bar
    (``rtol = 0``; golden parity is unchanged).  float32 clamps each
    absolute bar above the f32 ulp floor and sets
    ``rtol = MULTI_RTOL_F32`` so the Newton/LM convergence tests are
    satisfiable on reduced precision.

    If ``compute_dtype`` is provided, use it for the convergence tolerances
    (allows float32 compute with float64 convergence).
    """
    tol_dtype = compute_dtype if compute_dtype is not None else dtype
    if tol_dtype != torch.float32:
        return MULTI_TOL, 0.0, MULTI_TOL_STEP, STAGNATION_REL_TOL
    eps: float = float(torch.finfo(torch.float32).eps)
    return (
        max(MULTI_TOL, 64.0 * eps),
        MULTI_RTOL_F32,
        max(MULTI_TOL_STEP, eps),
        max(STAGNATION_REL_TOL, eps),
    )


def _max_abs(v: Tensor) -> Tensor:
    """Batch-wise component-wise max ``|v|`` over the last axis (Rust ``max_abs``)."""
    return v.abs().amax(dim=-1)


def _convergence_bar(
    x: Tensor,
    atol: float,
    rtol: float,
    rtol_active: bool,
) -> Tensor:
    """Two-term convergence bar ``atol + rtol * max(|x|, 1)`` as a (B,) tensor.

    ``rtol_active=False`` (the float64 golden path, ``rtol = 0``) returns the
    pure absolute bar ``atol`` *without touching* ``x`` — so the f64 test
    stays bit-identical to the reference (no relative term, no NaN
    propagation from the iterate).  ``rtol_active=True`` (float32) adds the
    relative term scaled by the iterate's component-wise ``max(|x|, 1)``
    magnitude (the hydro two-term form), which keeps the test satisfiable at
    the f32 ulp floor for O(1)-scale states.
    """
    if not rtol_active:
        return torch.full((x.shape[0],), atol, dtype=x.dtype, device=x.device)
    x_scale: Tensor = _max_abs(x)
    return atol + rtol * torch.maximum(x_scale, torch.ones_like(x_scale))


def _solves_lu(a: Tensor, b: Tensor) -> Tensor:
    """Solve ``a x = b`` element-wise, batched, with partial-pivoting LU.

    Mirrors ``math.rs::solve_f64`` (faer ``PartialPivLu``), which performs the
    solve for *any* matrix (a singular pivot yields Inf/NaN rather than an
    error); no least-squares fallback is needed because the reference only
    reaches it when the matrix is non-contiguous, which never happens here.

    ``lu_factor_ex(check_errors=False)`` is used instead of ``lu_factor`` so
    rank-deficient inputs produce ``Inf``/``NaN`` silently like the reference
    rather than raising: the masked solve machine feeds *every* batch lane into
    one batched factorization, and a single singular submatrix must not abort
    the batch or desynchronise the per-lane status masks.

    Plain batched tensor ops (``lu_factor_ex``/``lu_solve``), run eagerly on
    whatever device/dtype the caller selected; no ``torch.compile`` wrapping
    of any kind.
    """
    lu, pivots, _info = torch.linalg.lu_factor_ex(a, check_errors=False)
    if b.dim() == a.dim() - 1:
        x: Tensor = torch.linalg.lu_solve(lu, pivots, b.unsqueeze(-1))
        return x.squeeze(-1)
    return cast(Tensor, torch.linalg.lu_solve(lu, pivots, b))


def _satisfies_sufficient_decrease(
    f_new: Tensor, f_x: Tensor, alpha: Tensor
) -> Tensor:
    """Armijo sufficient-decrease mask (Rust ``satisfies_sufficient_decrease``).

    ``||f_new||^2 <= (1 - ARMIJO_C*alpha) * ||f_x||^2`` in squared-residual
    norm, with NaN residuals rejected outright.
    """
    no_nan: Tensor = ~f_new.isnan().any(dim=-1)
    norm_new: Tensor = (f_new * f_new).sum(dim=-1)
    norm_x: Tensor = (f_x * f_x).sum(dim=-1)
    decrease: Tensor = norm_new <= (1.0 - ARMIJO_C * alpha) * norm_x
    return no_nan & decrease


def _newton_trial_head(
    eval_fn: FusedEval,
    x: Tensor,
    f_x: Tensor,
    jac_x: Tensor,
    err: Tensor,
    status: Tensor,
    result: Tensor,
    eye_batch: Tensor,
    zero_rhs: Tensor,
    done_t: Tensor,
    failed_t: Tensor,
    bar: Tensor,
) -> tuple[Tensor, ...]:
    """Head of one masked Newton trial (Rust ``newton_fused_core`` step 1-6).

    Finalise/NaN-guard the running lanes, solve the (identity-masked) step and run
    the *full-step* (``alpha = 1``) Armijo trial.  The head is a sync-free
    sequence of batched torch ops; the single host sync a Newton trial can have
    (the rare Armijo *halving* guard) lives in :func:`_newton_fused_core`
    between the head and :func:`_newton_trial_tail`.

    ``eval_fn`` is passed (not captured) so the evaluation closure is inlined
    per call -- the solver remains agnostic to which model supplies the
    residual/Jacobian.

    ``bar`` is the (B,) two-term convergence bar (``atol + rtol*max(|x|, 1)``
    at the current iterate; the pure absolute reference bar on f64) used for
    the top-of-trip done test.

    Returns the updated ``(status, result)`` and the running state carrying the
    full-step trial outcome ``(running, step, alpha, trial_left, x_new, f_new,
    jac_new, accepted)``.
    """
    running: Tensor = status == _RUNNING
    done: Tensor = running & (err <= bar)
    status = torch.where(done, done_t, status)
    result = torch.where(done.unsqueeze(-1), x, result)
    running = status == _RUNNING
    nan: Tensor = running & (
        x.isnan().any(dim=-1) | f_x.isnan().any(dim=-1) | jac_x.isnan().any(dim=(-1, -2))
    )
    status = torch.where(nan, failed_t, status)
    running = status == _RUNNING

    safe_jac: Tensor = torch.where(running[:, None, None], jac_x, eye_batch)
    safe_f: Tensor = torch.where(running[:, None], f_x, zero_rhs)
    step: Tensor = _solves_lu(safe_jac, safe_f)
    step = torch.where(running[:, None], step, torch.zeros_like(x))

    batch: int = x.shape[0]
    alpha: Tensor = x.new_ones(batch)
    trial_left: Tensor = running.clone()
    x_new: Tensor = x.clone()
    f_new: Tensor = f_x.clone()
    jac_new: Tensor = jac_x.clone()
    accepted: Tensor = torch.zeros(batch, dtype=torch.bool, device=x.device)

    # Full-step (alpha=1) trial, unrolled with no host guard (the common path).
    x_trial: Tensor = x - alpha.unsqueeze(-1) * step
    f_trial: Tensor
    jac_trial: Tensor
    f_trial, jac_trial = eval_fn(x_trial)
    decrease: Tensor = _satisfies_sufficient_decrease(f_trial, f_x, alpha)
    new_acc: Tensor = trial_left & decrease
    new_acc_v: Tensor = new_acc.unsqueeze(-1)
    x_new = torch.where(new_acc_v, x_trial, x_new)
    f_new = torch.where(new_acc_v, f_trial, f_new)
    jac_new = torch.where(new_acc_v.unsqueeze(-1), jac_trial, jac_new)
    accepted = accepted | new_acc
    trial_left = trial_left & ~new_acc
    alpha = torch.where(trial_left, alpha * 0.5, alpha)
    return status, result, running, step, alpha, trial_left, x_new, f_new, jac_new, accepted


def _newton_trial_tail(
    x: Tensor,
    f_x: Tensor,
    jac_x: Tensor,
    running: Tensor,
    x_new: Tensor,
    f_new: Tensor,
    jac_new: Tensor,
    accepted: Tensor,
    best_x: Tensor,
    best_err: Tensor,
    status: Tensor,
    result: Tensor,
    failed_t: Tensor,
    tol: float,
) -> tuple[Tensor, ...]:
    """Tail of one masked Newton trial (Rust ``newton_fused_core`` step 8).

    Accept/reject the (possibly line-searched) step, recompute the carried residual
    and update the best-iterate track.  Like the head, this is a sync-free
    sequence of batched torch ops.  ``x`` / ``f_x`` / ``jac_x`` are the
    *pre-trial* iterate carried in the loop; ``x_new`` / ``f_new`` / ``jac_new``
    / ``accepted`` carry the (head or head+halving) trial outcome.
    """
    stalled: Tensor = running & ~accepted
    status = torch.where(stalled, failed_t, status)
    moved: Tensor = running & accepted
    x = torch.where(moved.unsqueeze(-1), x_new, x)
    f_x = torch.where(moved.unsqueeze(-1), f_new, f_x)
    jac_x = torch.where(moved.unsqueeze(-1).unsqueeze(-1), jac_new, jac_x)
    err: Tensor = _max_abs(f_x)
    better: Tensor = moved & (err < best_err) & ~err.isnan()
    best_x = torch.where(better.unsqueeze(-1), x, best_x)
    best_err = torch.where(better, err, best_err)
    return x, f_x, jac_x, err, best_x, best_err, status, result


def _newton_fused_core(
    eval_fn: FusedEval,
    x_0: Tensor,
    verbose: bool = False,
    compute_dtype: torch.dtype | None = None,
) -> NewtonOutcome:
    """Masked Newton iterate body (Rust ``newton_fused_core``) as a bounded
    Python loop.

    Each lane follows the scalar control flow: check the two-term residual bar
    (``max|f| <= atol + rtol*max|x|``, pure absolute on f64) at the top,
    NaN-guard the current iterate, solve the (partial-pivoting) system,
    backtrack along the Armijo line (a bounded Python ``for`` loop),
    accept/reject.  Lanes that stall, hit NaN, or exhaust the budget are marked
    ``_FAILED`` (the LM fallback then resumes from ``best_x``); lanes that
    succeed return their iterate directly, or the best iterate when the loop
    exits via the post-loop accept.

    The iteration is a bounded Python ``for`` loop rather than
    ``torch.while_loop``: the eager ``while_loop`` compiles its wrapper with
    ``fullgraph=True`` and inlines this body into a single traced graph, where
    every form of early exit is a hard error in torch 2.7 — a nested Armijo
    ``torch.while_loop`` (``UncapturedHigherOrderOpError``), a ``break`` on a
    tensor condition (``Unsupported: Data-dependent branching``), and a
    ``.item()`` host read in the traced body (``Unsupported: Tensor.item``).
    Outside the trace all three are ordinary Python, so the Newton solve is a
    hand-rolled loop; the per-iteration guards (``running.any()`` and
    ``trial_left.any()``) are single 0-dim reads, which cost nothing next to
    the fused eval.  Frozen (done/failed) lanes are bitwise-idempotent under
    the masked ``torch.where`` updates.

    Args:
        eval_fn: Fused residual/Jacobian evaluation.
        x_0: Initial guesses (B, n).
        verbose: Print per-iteration diagnostics.
        compute_dtype: Optional dtype for internal computation (e.g., float32 for
            mixed precision with float64 convergence checks).
    """
    batch: int = x_0.shape[0]
    n: int = x_0.shape[1]
    device: torch.device = x_0.device
    atol: float
    rtol: float
    tol_step: float
    stagn_tol: float
    atol, rtol, tol_step, stagn_tol = _solver_tols(x_0.dtype, compute_dtype)
    rtol_active: bool = rtol > 0.0

    x: Tensor = x_0.clone()
    f_x: Tensor
    jac_x: Tensor
    f_x, jac_x = eval_fn(x)
    # An eval may hand back strided views (e.g. broadcasts over a strided
    # batch column), while the masked ``torch.where`` body outputs are always
    # contiguous; normalise the eval-derived state so the Newton iterate carries
    # contiguous tensors throughout (matching the LM core's convention).
    f_x = f_x.contiguous()
    jac_x = jac_x.contiguous()

    # Degenerate systems (no free variables, or an empty residual) are already
    # at the trivial root.  This must be checked *before* ``_max_abs`` because
    # ``amax`` over a zero-length axis raises.
    if n == 0 or f_x.shape[-1] == 0:
        zeros: Tensor = torch.zeros(batch, dtype=x.dtype, device=device)
        status: Tensor = torch.full((batch,), _DONE, dtype=torch.long, device=device)
        return NewtonOutcome(x.clone(), x.clone(), zeros, status, zeros)

    # For mixed precision: use compute_dtype for internal iterations,
    # but keep convergence checks in x.dtype (float64)
    compute_dtype = compute_dtype or x.dtype

    err: Tensor = _max_abs(f_x)

    # Warm-start fast path: if every lane is already within the residual bar at
    # the initial iterate ``x``, return it directly -- no workspace build, no
    # iteration.  This is bit-identical to what the *first* masked trip would
    # produce: a trip on all-converged lanes only re-labels them ``_DONE`` and
    # copies ``x`` into ``result`` / ``best_x`` (the LU / Armijo / NaN / step
    # updates in the trip body all become masked no-ops for frozen lanes), so
    # short-circuiting here changes neither the returned iterate nor the status.
    # It is the common case for the batched runner, which warm-starts every zone
    # from the previous step (and the speciation solve from the previous
    # ``x_free``), so near-steady steps that are already at the root skip an
    # entire fused eval + LU + Armijo trial.  ``err <= bar`` is ``False`` for
    # a NaN residual or iterate, so a non-finite or still-far lane is never
    # short-circuited.
    if bool((err <= _convergence_bar(x, atol, rtol, rtol_active)).all()):
        return NewtonOutcome(
            x.clone(),
            x.clone(),
            err.clone(),
            torch.full((batch,), _DONE, dtype=torch.long, device=device),
            err,
        )

    best_x: Tensor = x.clone()
    best_err: Tensor = err.clone()
    status = torch.full((batch,), _RUNNING, dtype=torch.long, device=device)
    result: Tensor = x.clone()

    # Frozen lanes keep identity jacobian + zero rhs (their LU solve returns an
    # exact zero step) so NaN/singular states from already-failed lanes can
    # never poison the full-batch ``lu_factor``.
    eye_batch: Tensor = (
        torch.eye(n, dtype=x.dtype, device=device).unsqueeze(0).expand(batch, n, n)
    )
    zero_rhs: Tensor = torch.zeros_like(f_x)
    done_t: Tensor = torch.full((batch,), _DONE, dtype=torch.long, device=device)
    failed_t: Tensor = torch.full((batch,), _FAILED, dtype=torch.long, device=device)

    def newton_body(
        x: Tensor,
        f_x: Tensor,
        jac_x: Tensor,
        err: Tensor,
        best_x: Tensor,
        best_err: Tensor,
        status: Tensor,
        result: Tensor,
    ) -> tuple[Tensor, ...]:
        """One masked Newton trial, split across two eager regions.

        The expensive, sync-free work is delegated to the module-level
        :func:`_newton_trial_head` (finalise + LU + full-step Armijo trial) and
        :func:`_newton_trial_tail` (accept/reject + best-iterate update), each a
        flat sequence of batched torch ops.  The only host sync in a trial --
        the *rare* Armijo *halving* guard -- lives here, between the two
        regions; on the common full-step path (``trial_left`` all False) it is a
        single sync and the body is head -> tail.  The arithmetic performed is
        byte-for-byte the original inlined ``newton_body`` (``armijo_trial`` is
        the same expression as the head's inlined full-step trial), so the
        per-lane control flow and values are unchanged.
        """
        bar: Tensor = _convergence_bar(x, atol, rtol, rtol_active)
        (
            status,
            result,
            running,
            step,
            alpha,
            trial_left,
            x_new,
            f_new,
            jac_new,
            accepted,
        ) = _newton_trial_head(
            eval_fn, x, f_x, jac_x, err, status, result,
            eye_batch, zero_rhs, done_t, failed_t, bar,
        )

        if trial_left.any().item():  # rare: some lane needs backtracking

            def armijo_trial(
                alpha: Tensor,
                trial_left: Tensor,
                x_new: Tensor,
                f_new: Tensor,
                jac_new: Tensor,
                accepted: Tensor,
            ) -> tuple[Tensor, ...]:
                """One masked Armijo halving trial (the shared halving body)."""
                x_trial: Tensor = x - alpha.unsqueeze(-1) * step
                f_trial: Tensor
                jac_trial: Tensor
                f_trial, jac_trial = eval_fn(x_trial)
                decrease: Tensor = _satisfies_sufficient_decrease(f_trial, f_x, alpha)
                new_acc: Tensor = trial_left & decrease
                new_acc_v: Tensor = new_acc.unsqueeze(-1)
                x_new = torch.where(new_acc_v, x_trial, x_new)
                f_new = torch.where(new_acc_v, f_trial, f_new)
                jac_new = torch.where(new_acc_v.unsqueeze(-1), jac_trial, jac_new)
                accepted = accepted | new_acc
                trial_left = trial_left & ~new_acc
                alpha = torch.where(trial_left, alpha * 0.5, alpha)
                return alpha, trial_left, x_new, f_new, jac_new, accepted

            for _ in range(LINE_SEARCH_MAX - 1):
                alpha, trial_left, x_new, f_new, jac_new, accepted = armijo_trial(
                    alpha, trial_left, x_new, f_new, jac_new, accepted
                )
                if not trial_left.any().item():
                    break

        (
            x,
            f_x,
            jac_x,
            err,
            best_x,
            best_err,
            status,
            result,
        # The tail's trailing ``tol`` parameter is vestigial (unused in the
        # body); ``atol`` is passed to keep the signature populated.
        ) = _newton_trial_tail(
            x, f_x, jac_x, running, x_new, f_new, jac_new, accepted,
            best_x, best_err, status, result, failed_t, atol,
        )
        if verbose:
            print(f"newton iter, max err {err.max().item():.3e}")
        return x, f_x, jac_x, err, best_x, best_err, status, result

    # Bounded eager Python loop.  The termination check reads the 0-dim
    # ``running.any()`` mask to the host every trip.  (Phase 2 originally
    # checked only every 4th trip to cut host syncs, but that was measured to
    # *slow* the solve: each skipped check lets up to 3 extra masked body
    # evaluations — an ``eval_fn`` + LU + Armijo trial, i.e. the expensive
    # part — run after the last lane freezes, and those extra evaluations cost
    # far more than the one host sync saved.  Checking every trip is the
    # faster, bit-identical behaviour, so it is kept.)
    for _it in range(MULTI_MAXITER):
        x, f_x, jac_x, err, best_x, best_err, status, result = newton_body(
            x, f_x, jac_x, err, best_x, best_err, status, result
        )
        if not (status == _RUNNING).any().item():
            break

    still: Tensor = status == _RUNNING
    ok: Tensor = still & (
        best_err <= _convergence_bar(best_x, atol, rtol, rtol_active)
    )
    status = torch.where(still, torch.where(ok, done_t, failed_t), status)
    result = torch.where(ok.unsqueeze(-1), best_x, result)
    return NewtonOutcome(result, best_x, best_err, status, err)


def _levenberg_marquardt_core(
    eval_fn: FusedEval,
    x_0: Tensor,
    verbose: bool = False,
    compute_dtype: torch.dtype | None = None,
    guard: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Masked Levenberg-Marquardt with analytic Jacobian (Rust LM implement).

    The iterate is a bounded eager Python ``for`` loop (successor to the
    ``torch.while_loop`` HOP).  The body is one masked LM iteration; the
    per-trip termination is a 0-dim ``running.any()`` mask read to the host
    every trip, and it is unnecessary to check on the last trip since
    remaining running lanes are folded by the post-loop accept below.
    Frozen lanes are bit-idempotent under the masked ``torch.where``
    updates (each update is gated by the running mask, and the re-evaluation
    at an unchanged iterate is deterministic), so trips after the last lane
    freezes are bit-identical.

    When ``guard`` is supplied, ``guard.any()`` is read once before the loop
    (it never changes over the LM trips); if false, the loop is skipped
    entirely — so a caller that only needs the fallback for lanes flagged by
    ``guard`` gets zero LM work (and the initial ``result``/``status``
    returned untouched) when there is nothing to rescue.

    Returns ``(result, status)``.  ``result`` is the accepted iterate when the
    residual bar is met, ``best_x`` when the solve is accepted through step
    stagnation, and NaN for lanes that failed outright.

    Args:
        eval_fn: Fused residual/Jacobian evaluation.
        x_0: (B, n) initial guesses.
        verbose: Print per-iteration diagnostics.
        compute_dtype: Optional dtype for internal computation.
        guard: Optional (B,) mask; if empty, the loop is skipped entirely.
    """
    batch: int = x_0.shape[0]
    n: int = x_0.shape[1]
    device: torch.device = x_0.device
    atol: float
    rtol: float
    tol_step: float
    stagn_tol: float
    atol, rtol, tol_step, stagn_tol = _solver_tols(x_0.dtype, compute_dtype)
    rtol_active: bool = rtol > 0.0

    # The empty system (no free variables) is trivially solved; degenerate-case
    # detection must precede the initial ``eval_fn`` here so a one-shot caller
    # that just short-circuited Newton (e.g. ``n == 0``) pays no extra residual
    # evaluation, and before any ``_max_abs``/norm reduction that would raise on
    # a zero-length axis.
    if n == 0:
        status: Tensor = torch.full((batch,), _DONE, dtype=torch.long, device=device)
        return x_0.clone(), status

    x: Tensor = x_0.clone()
    f_x: Tensor
    jac_x: Tensor
    f_x, jac_x = eval_fn(x)
    # Same stride-normalisation rationale as ``_newton_fused_core``: the loop's
    # carried f_x/jac_x must match their body-output metadata exactly.
    f_x = f_x.contiguous()
    jac_x = jac_x.contiguous()

    if f_x.shape[-1] == 0:
        done: Tensor = torch.full((batch,), _DONE, dtype=torch.long, device=device)
        return x.clone(), done

    err: Tensor = 0.5 * (f_x * f_x).sum(dim=-1)
    err_max: Tensor = _max_abs(f_x)
    initial_err_max: Tensor = err_max.clone()
    best_x: Tensor = x.clone()
    best_err: Tensor = err_max.clone()
    lam: Tensor = x.new_full((batch,), 1e-6)
    status = torch.full((batch,), _RUNNING, dtype=torch.long, device=device)
    result: Tensor = x.clone()

    eye_batch: Tensor = (
        torch.eye(n, dtype=x.dtype, device=device).unsqueeze(0).expand(batch, n, n)
    )
    zero_rhs: Tensor = torch.zeros_like(f_x)
    done_t: Tensor = torch.full((batch,), _DONE, dtype=torch.long, device=device)
    failed_t: Tensor = torch.full((batch,), _FAILED, dtype=torch.long, device=device)

    # ``guard`` (the lanes that failed Newton) never changes over the LM trips,
    # so its ``.any()`` is read once to the host here rather than every trip
    # (Phase 2).  An all-``guard``-false caller gets zero LM work.
    guard_active: bool = True if guard is None else bool(guard.any())

    def lm_body(
        x: Tensor,
        f_x: Tensor,
        jac_x: Tensor,
        err: Tensor,
        err_max: Tensor,
        initial_err_max: Tensor,
        best_x: Tensor,
        best_err: Tensor,
        lam: Tensor,
        status: Tensor,
        result: Tensor,
    ) -> tuple[Tensor, ...]:
        """One masked LM iteration."""
        running: Tensor = status == _RUNNING
        done: Tensor = running & (err_max <= _convergence_bar(x, atol, rtol, rtol_active))
        status = torch.where(done, done_t, status)
        result = torch.where(done.unsqueeze(-1), x, result)
        running = status == _RUNNING
        nan: Tensor = running & (
            x.isnan().any(dim=-1) | f_x.isnan().any(dim=-1) | jac_x.isnan().any(dim=(-1, -2))
        )
        status = torch.where(nan, failed_t, status)
        running = status == _RUNNING

        jac_x = eval_fn(x)[1].contiguous()
        jac_t: Tensor = jac_x.transpose(-1, -2)
        a_base: Tensor = jac_t @ jac_x
        diag_add: Tensor = lam.unsqueeze(-1) * torch.diagonal(a_base, dim1=-2, dim2=-1)
        a_damped: Tensor = a_base + torch.diag_embed(diag_add)
        b: Tensor = (jac_t @ f_x.unsqueeze(-1)).squeeze(-1)

        safe_a: Tensor = torch.where(running[:, None, None], a_damped, eye_batch)
        safe_b: Tensor = torch.where(running[:, None], b, zero_rhs)
        step: Tensor = _solves_lu(safe_a, safe_b)
        step = torch.where(running[:, None], step, torch.zeros_like(x))

        x_scale: Tensor = torch.maximum(x.abs().amax(dim=-1), torch.ones_like(err_max))
        step_converged: Tensor = step.abs().amax(dim=-1) <= tol_step * x_scale

        x_test: Tensor = x - step
        f_test, _ = eval_fn(x_test)
        err_test: Tensor = 0.5 * (f_test * f_test).sum(dim=-1)
        err_test_max: Tensor = _max_abs(f_test)

        better: Tensor = running & (err_test_max < best_err) & ~err_test_max.isnan()
        best_x = torch.where(better.unsqueeze(-1), x_test, best_x)
        best_err = torch.where(better, err_test_max, best_err)

        actual_reduction: Tensor = err - err_test
        predicted_reduction: Tensor = (
            (step * b).sum(dim=-1)
            - 0.5 * (step * (a_base @ step.unsqueeze(-1)).squeeze(-1)).sum(dim=-1)
        )
        rho: Tensor = torch.where(
            predicted_reduction.abs() < 1e-12,
            torch.zeros_like(predicted_reduction),
            actual_reduction / predicted_reduction,
        )

        accept: Tensor = running & (rho > 0)
        any_accept: Tensor = accept.any()
        x = torch.where(accept.unsqueeze(-1), x_test, x)
        f_x = torch.where(accept.unsqueeze(-1), f_test, f_x)
        err = torch.where(accept, err_test, err)
        err_max = torch.where(accept, err_test_max, err_max)
        factor: Tensor = torch.maximum(
            1.0 / 3.0 * torch.ones_like(rho), 1.0 - (2.0 * rho - 1.0).pow(3)
        )
        lam = torch.where(accept, (lam * factor).clamp_min(1e-16), lam)
        reject: Tensor = running & (rho <= 0) & ~any_accept
        lam = torch.where(reject, (lam * 2.0).clamp_max(1e16), lam)
        step_ok: Tensor = accept & step_converged & (
            err_max <= _convergence_bar(x, atol, rtol, rtol_active)
        )
        status = torch.where(step_ok, done_t, status)
        result = torch.where(step_ok.unsqueeze(-1), x, result)

        running = status == _RUNNING
        bar: Tensor = torch.maximum(
            _convergence_bar(x, atol, rtol, rtol_active),
            stagn_tol * initial_err_max,
        )
        stagnate: Tensor = running & step_converged & (best_err <= bar)
        status = torch.where(stagnate, done_t, status)
        result = torch.where(stagnate.unsqueeze(-1), best_x, result)
        if verbose:
            print(f"lm iter, max err {err_max.max().item():.3e}, accept {accept.sum().item()}")
        return (
            x,
            f_x,
            jac_x,
            err,
            err_max,
            initial_err_max.clone(),
            best_x,
            best_err,
            lam,
            status,
            result,
        )

    # Bounded eager Python loop (successor to the ``torch.while_loop``
    # HOP).  Each trip is one masked LM iteration; the Python ``range``
    # carries the per-lane trip budget (``MULTI_MAXITER``) that was formerly
    # the carried ``it < multi_max`` check, and ``guard_active`` was hoisted
    # out of the per-trip condition (``guard`` never changes over the LM
    # trips, so reading it once is always a strict saving).  The per-trip
    # ``running.any()`` check runs every trip for the same reason as in
    # ``_newton_fused_core`` (a periodic check would save a cheap host sync
    # at the cost of up to 3 extra body evaluations — measured, the net is
    # slower).
    if guard_active:
        for it in range(MULTI_MAXITER):
            (
                x,
                f_x,
                jac_x,
                err,
                err_max,
                initial_err_max,
                best_x,
                best_err,
                lam,
                status,
                result,
            ) = lm_body(
                x, f_x, jac_x, err, err_max,
                initial_err_max, best_x, best_err, lam, status, result,
            )
            if not (status == _RUNNING).any().item():
                break

    still: Tensor = status == _RUNNING
    ok: Tensor = still & (
        best_err <= _convergence_bar(best_x, atol, rtol, rtol_active)
    )
    status = torch.where(still, torch.where(ok, done_t, failed_t), status)
    result = torch.where(ok.unsqueeze(-1), best_x, result)
    return result, status


def _newton_fused_then_lm_core(
    eval_fn: FusedEval,
    x_0: Tensor,
    verbose: bool,
    compute_dtype: torch.dtype | None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Newton-then-LM dispatch: LM is device-resident (no host syncs), Newton
    is a bounded Python loop with per-iteration host guard reads.

    Returns ``(result, still_failed, best_err)``.  Lanes that failed in both
    solvers hold NaN in ``result`` and are marked in the ``still_failed`` mask;
    ``best_err`` is Newton's best per-lane residual for the caller's diagnostics.
    The LM fallback only runs when at least one lane failed: the LM loop's
    ``guard`` stops it before its first trip otherwise, so the common
    all-converged case pays only the on-device ``guard.any()`` check.
    """
    outcome: NewtonOutcome = _newton_fused_core(eval_fn, x_0, verbose, compute_dtype)
    go_lm: Tensor = outcome.status == _FAILED
    lm_result, lm_status = _levenberg_marquardt_core(
        eval_fn, outcome.best_x.clone(), verbose, compute_dtype, guard=go_lm
    )
    result: Tensor = torch.where(go_lm.unsqueeze(-1), lm_result, outcome.result)
    still_failed: Tensor = go_lm & (lm_status == _FAILED)
    result = torch.where(
        still_failed.unsqueeze(-1), torch.full_like(result, float("nan")), result
    )
    return result, still_failed, outcome.best_err


def newton_fused_then_lm(
    eval_fn: FusedEval,
    x_0: Tensor,
    *,
    raise_on_failure: bool = False,
    verbose: bool = False,
    compute_dtype: torch.dtype | None = None,
) -> Tensor:
    """Newton (fused analytic Jacobian) then LM fallback, batched (Rust fused entry).

    Args:
        eval_fn:            fused (residual, Jacobian) evaluation.
        x_0:                (B, n) initial guesses.
        raise_on_failure:   raise :class:`BatchedRootFindingError` if any lane
                            fails; otherwise failed lanes are returned as NaN.
        verbose:            print per-iteration summaries.

    Returns:
        (B, n) roots.  Lanes that failed in both Newton and LM hold NaN.
    """
    result, still_failed, best_err = _newton_fused_then_lm_core(
        eval_fn, x_0, verbose, compute_dtype
    )
    if raise_on_failure:
        # One cheap host read of a 0-dim mask, only on the raise-on-failure path.
        if bool(still_failed.any()):
            raise BatchedRootFindingError(
                f"Newton + Levenberg-Marquardt failed for {int(still_failed.sum())} "
                f"of {x_0.shape[0]} batch lanes; best max|f| = "
                f"{best_err[still_failed].max().item():.3e}"
            )
    return result


def newton_analytic_then_lm(
    f: VectorFn,
    jf: JacobianFn,
    x_0: Tensor,
    *,
    raise_on_failure: bool = False,
    verbose: bool = False,
) -> Tensor:
    """Newton with a separately-supplied analytic Jacobian, then LM (Rust entry).

    Delegates to :func:`newton_fused_then_lm` with a fused evaluation.  Along
    the line search each trial evaluates the residual and the Jacobian once, and
    the accepted step's Jacobian is reused for the next Newton step, mirroring
    the reference's fused single-evaluation property.

    Args:
        f:                  (B, m) residual function.
        jf:                 (B, m, n) analytic Jacobian function.
        x_0:                (B, n) initial guesses.
        raise_on_failure:   raise :class:`BatchedRootFindingError` if any lane
                            fails.
        verbose:            print per-iteration summaries.

    Returns:
        (B, n) roots.
    """
    fused: FusedEval = lambda x: (f(x), jf(x))
    return newton_fused_then_lm(fused, x_0, raise_on_failure=raise_on_failure, verbose=verbose)