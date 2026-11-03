"""Soil (``RtZone``) and river (``RiverZone``) reactive-transport zones.

Mirrors ``rt_zone.rs`` and ``river_zone.rs``: the Newton-to-LM fused solve, the
``num_aq + i`` hardcode quirk in the kinetic derivative, the double-conversion
of aqueous concentrations in ``reaction_rate``, and every dropped sign / zeroing
convention.  ``run_zone_steps`` (the multi-zone Gauss-Seidel driver) is deferred
to M4; this module covers single-zone stepping plus the diagnostic rate methods.

**Batched convention:** every diagnostic method accepts a 1-D species vector and
returns a 1-D vector; the internal ``step`` path promotes all species vectors to
a leading batch dimension ``(B, num_spec)`` (restoring the original leading shape
on exit) so a single step can drive many replicate zones at once and match the
scalar reference lane-by-lane.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

import torch

class WaterVolumeError(ValueError):
    """Raised when the simulated water volume exceeds the zone geometry.

    Mirrors the golden ``WaterVolumeError`` (which subclasses the reference
    ``HydrologyError``).
    """



from .common_types import MiscData, RtForcing, RtStep, ZERO_CONC
from .equilibrium import EquilibriumParameters
from .kinetic_structures import (
    MineralParameters,
    MonodParameters,
    RiverDimensions,
    RiverParameters,
    RtParameters,
    TstParameters,
    ZoneDimensions,
    monod_inhib_rates,
    smoothed_conc,
    smoothed_rel_deriv,
    tst_rates,
)
from .reaction_network import ReactionNetwork
from .solvers import FusedEval, newton_fused_then_lm

Tensor: TypeAlias = torch.Tensor
Scalar: TypeAlias = float | torch.Tensor

# Module-level cache for identity matrices
_EYE_CACHE: dict[tuple[int, torch.dtype, torch.device], torch.Tensor] = {}


def _get_eye(num_spec: int, dtype: torch.dtype, device: torch.device) -> Tensor:
    """Cached identity matrix for (num_spec, dtype, device)."""
    key = (num_spec, dtype, device)
    if key not in _EYE_CACHE:
        _EYE_CACHE[key] = torch.eye(num_spec, dtype=dtype, device=device)
    return _EYE_CACHE[key]


# Module-level cache for the aqueous-species index range (``range(num_aq)`` as a
# tensor).  It is a loop-invariant: ``num_aq`` is fixed per network, so the
# per-zone-step re-allocation of ``torch.arange(num_aq)`` in
# ``_assemble_ode_jacobian`` / ``_zone_step`` is pure churn otherwise.
_AQ_RANGE_CACHE: dict[tuple[int, torch.device], torch.Tensor] = {}


def _aq_range(num_aq: int, device: torch.device) -> Tensor:
    """Cached ``arange(num_aq)`` index tensor for the given device."""
    key = (num_aq, device)
    r: Tensor | None = _AQ_RANGE_CACHE.get(key)
    if r is None:
        r = torch.arange(num_aq, dtype=torch.long, device=device)
        _AQ_RANGE_CACHE[key] = r
    return r


# Module-level cache for the (num_spec, num_min) mineral stoichiometry cast to
# the working dtype/device.  The ``stoich`` tensor is constant per zone *and*
# per step (it is built once in ``_build_fused_eval``), so the cast in
# ``_assemble_ode_jacobian`` / ``_kinetic_ode_from_terms`` is pure overhead on
# every solver iteration.  The original tensor is stored alongside the cast so
# an ``id`` reuse after garbage collection cannot return a stale buffer.
_STOICH_CACHE: dict[tuple[int, torch.dtype, torch.device], tuple[Tensor, Tensor]] = {}


def _get_stoich(
    stoich: Tensor, dtype: torch.dtype, device: torch.device
) -> Tensor:
    """Cached ``stoich.to(dtype=dtype, device=device)`` for the hot RT path."""
    key = (id(stoich), dtype, device)
    cached: tuple[Tensor, Tensor] | None = _STOICH_CACHE.get(key)
    if cached is not None and cached[0] is stoich:
        return cached[1]
    cast_stoich: Tensor = stoich.to(dtype=dtype, device=device)
    _STOICH_CACHE[key] = (stoich, cast_stoich)
    return cast_stoich


# ---------------------------------------------------------------------- #
# Cached network constants and context-safe workspaces (Phase 3)
# ---------------------------------------------------------------------- #
#
# The kinetic hot path re-cast the same frozen per-network tensors (Monod /
# TST matrices, mineral parameters, misc arrays) on *every* zone step and,
# inside the Newton loop, on every solver trip, and rebuilt the same masks /
# zero fills / scratch buffers on top.  Those are loop invariants: each
# source tensor is immutable for the life of its zone (frozen dataclass
# field, or the zone-owned ``MiscData`` / ``MineralParameters`` objects that
# ``run_rt_batched`` passes through unchanged, or the per-zone 0-dim depth
# tensors of its ``per_zone_*`` lists).  These id-keyed caches collapse the
# churn to one build per (source, dtype, device) pair, mirroring the
# ``_get_stoich`` pattern above: the source tensor is held next to the
# derived value so an ``id()`` reuse after garbage collection can never
# return a stale entry.
#
# Two kinds of cached values:
#   * *read-only* caches (casts, masks, zero fills, ``delta``): safe to read
#     from any context — reading an inference-mode tensor in normal mode is
#     allowed and yields a normal tensor.
#   * *scratch* workspaces (written on every call): PyTorch forbids
#     in-place updates to inference tensors *outside* an inference context
#     (the reverse direction is fine).  ``run_rt_batched`` runs the whole RT
#     loop under ``torch.inference_mode()`` while direct ``zone.step()``
#     calls run in normal mode, so scratch workspaces track the context they
#     were built in and are rebuilt once on a context switch
#     (:func:`_ws_buffers_ok`).

_CAST_CACHE: dict[tuple[int, torch.dtype, torch.device], tuple[Tensor, Tensor]] = {}


def _get_cast(t: Tensor, dtype: torch.dtype, device: torch.device) -> Tensor:
    """Cached ``t.to(dtype=dtype, device=device)`` keyed on the source id.

    Use only for tensors whose identity is stable for the duration of a run
    (zone-owned network constants): the held reference keeps the source
    alive, which both bounds the cache and makes the id key collision-proof.
    A no-op cast still costs a dispatch, so even identical dtype/device pays
    the cache.
    """
    key: tuple[int, torch.dtype, torch.device] = (id(t), dtype, device)
    entry: tuple[Tensor, Tensor] | None = _CAST_CACHE.get(key)
    if entry is not None and entry[0] is t:
        return entry[1]
    casted: Tensor = t.to(dtype=dtype, device=device)
    _CAST_CACHE[key] = (t, casted)
    return casted


# Cached 1-D zero vectors: the ``where(mask, x, 0)`` fallback lanes of the
# kinetic path rebuilt ``(num_spec,)`` / ``(num_min,)`` zero tensors on
# every trip although the zeros are a loop invariant.
_ZERO1D_CACHE: dict[tuple[int, torch.dtype, torch.device], Tensor] = {}


def _zero_tensor(num: int, dtype: torch.dtype, device: torch.device) -> Tensor:
    """Cached ``(num,)`` zero vector for the given dtype/device."""
    key: tuple[int, torch.dtype, torch.device] = (num, dtype, device)
    z: Tensor | None = _ZERO1D_CACHE.get(key)
    if z is None:
        z = torch.zeros(num, dtype=dtype, device=device)
        _ZERO1D_CACHE[key] = z
    return z


# Cached inverse of a 0-dim (scalar) depth tensor: ``where(|depth| > 0,
# 1/depth, 0)`` is recomputed on every Newton trip in
# ``_assemble_ode_jacobian`` although it is a loop invariant of a zone step.
# The per-lane ``(B,)`` depth form is *not* cached: it is per-step data.
_DEPTH_INV_CACHE: dict[tuple[int, torch.device], tuple[Tensor, Tensor]] = {}


def _get_depth_inv(depth: Tensor) -> Tensor:
    """Cached ``where(|depth| > 0, 1/depth, 0)`` for a 0-dim depth tensor.

    Computed in ``depth``'s own dtype/device (the caller's ``p``-row
    assignment performs any target-dtype conversion, exactly as before).
    """
    key: tuple[int, torch.device] = (id(depth), depth.device)
    entry: tuple[Tensor, Tensor] | None = _DEPTH_INV_CACHE.get(key)
    if entry is not None and entry[0] is depth:
        return entry[1]
    inv: Tensor = torch.where(depth.abs() > 0.0, 1.0 / depth, torch.zeros_like(depth))
    _DEPTH_INV_CACHE[key] = (depth, inv)
    return inv


@dataclass(frozen=True)
class _ReactionConsts:
    """Per-network loop invariants of :func:`_reaction_terms`.

    Everything here depends only on the network's *constant* tensors — never
    on the per-step batched state and never on the per-lane auxiliary ``ssa``
    (see :func:`_get_reaction_consts`) — so each field is built once per
    (network, compute_dtype, device) and reused by every Newton trip:

    Attributes:
        mk_fin:       (num_min, num_spec) finite mask of the Monod matrix
            (``NaN`` marks "no Monod term for this species").
        ik_fin:       (num_min, num_spec) finite mask of the inhibition matrix.
        const_active: (num_min,) True where the mineral's rate constant,
            molar mass and TST equilibrium constant are all finite and
            ``|min_eq_const| > 0`` (the constant half of the reference's
            per-call boolean activity chain).
        rate_scale:   (num_min,) ``86400 * rate_const`` — the first
            left-to-right factor of the per-call base scale; the per-call
            chain continues ``((rate_scale * ssa) * mineral_molar_mass) *
            aux_factor`` in the reference's exact order.
        zero_mn:      (num_min,) broadcastable zero fill.
        zero_mn_ns:   (num_min, num_spec) broadcastable zero fill.
    """

    mk_fin: Tensor
    ik_fin: Tensor
    const_active: Tensor
    rate_scale: Tensor
    zero_mn: Tensor
    zero_mn_ns: Tensor


_REACTION_CONSTS_CACHE: dict[
    tuple[int, int, int, int, int, torch.dtype, torch.device],
    tuple[Tensor, Tensor, Tensor, Tensor, Tensor, _ReactionConsts],
] = {}


def _get_reaction_consts(
    *,
    monod_mat: Tensor,
    inhib_mat: Tensor,
    min_eq_const: Tensor,
    rate_const: Tensor,
    mineral_molar_mass: Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> _ReactionConsts:
    """Build or fetch the per-network invariants of ``_reaction_terms``.

    The five inputs are the *compute-cast* network constants (id-stable
    across trips and steps for a given zone / compute dtype, see
    :func:`_get_cast`), so the derived masks / scales / zero fills are
    computed once.  The per-lane ``ssa`` is deliberately *not* an input: on
    the ``step_with_params`` path it is a per-call batched tensor, and the
    per-call base chain consumes it there in the reference's exact
    left-to-right order.  The source tensors are held in the cache entry so
    an ``id()`` reuse after garbage collection can never return a stale
    result.
    """
    key: tuple[int, int, int, int, int, torch.dtype, torch.device] = (
        id(monod_mat),
        id(inhib_mat),
        id(min_eq_const),
        id(rate_const),
        id(mineral_molar_mass),
        dtype,
        device,
    )
    cached: (
        tuple[Tensor, Tensor, Tensor, Tensor, Tensor, _ReactionConsts] | None
    ) = _REACTION_CONSTS_CACHE.get(key)
    if (
        cached is not None
        and cached[0] is monod_mat
        and cached[1] is inhib_mat
        and cached[2] is min_eq_const
        and cached[3] is rate_const
        and cached[4] is mineral_molar_mass
    ):
        return cached[5]
    num_min: int = int(monod_mat.shape[0])
    num_spec: int = int(monod_mat.shape[1])
    mk_fin: Tensor = monod_mat.isfinite()
    ik_fin: Tensor = inhib_mat.isfinite()
    const_active: Tensor = (
        rate_const.isfinite()
        & mineral_molar_mass.isfinite()
        & min_eq_const.isfinite()
        & (min_eq_const.abs() > 0.0)
    )
    rate_scale: Tensor = 86_400.0 * rate_const
    rc: _ReactionConsts = _ReactionConsts(
        mk_fin=mk_fin,
        ik_fin=ik_fin,
        const_active=const_active,
        rate_scale=rate_scale,
        zero_mn=_zero_tensor(num_min, dtype, device),
        zero_mn_ns=torch.zeros(num_min, num_spec, dtype=dtype, device=device),
    )
    _REACTION_CONSTS_CACHE[key] = (
        monod_mat,
        inhib_mat,
        min_eq_const,
        rate_const,
        mineral_molar_mass,
        rc,
    )
    return rc


def _ws_buffers_ok(ws: dict[str, Tensor]) -> bool:
    """True if the cached scratch buffers are writable in the current context.

    A buffer created inside ``torch.inference_mode`` is an *inference*
    tensor, and PyTorch rejects in-place updates to inference tensors outside
    an inference context (the reverse direction — writing a normal buffer
    inside one — is allowed; verified empirically on torch 2.14).  Scratch
    workspaces are therefore rebuilt when a context switch is detected
    (e.g. a direct ``zone.step()`` after a batched inference-mode run): at
    most one rebuild per key per process, after which the buffers are plain
    tensors again.  Read-only caches (casts, masks, zero fills, ``delta``)
    do not need this guard.

    Note on the torch 2.14 API: ``torch.is_inference(input)`` is the
    *tensor* check (hence the ``first.is_inference()`` below), while the
    *context* check is ``torch.is_inference_mode_enabled()`` (renamed from
    ``torch.is_inference_mode`` in older releases).
    """
    if torch.is_inference_mode_enabled():
        return True
    first: Tensor | None = next(iter(ws.values()), None)
    return first is not None and not first.is_inference()


@dataclass(frozen=True)
class _FusedEvalConstants:
    """Constant tensors for a fused residual/Jacobian evaluation.

    All fields are pre-cast to the working dtype/device and have batch dimension.
    Field names MUST match the parameter names of ``_kinetic_residual_and_jacobian``.
    """
    c_0: Tensor
    dt: Tensor
    v0: Tensor
    depth: float | Tensor
    q_in: Tensor
    c_in: Tensor
    stoich: Tensor
    monod_mat: Tensor
    inhib_mat: Tensor
    tst_stoich: Tensor
    tst_dep: Tensor
    min_eq_const: Tensor
    rate_const: Tensor
    mineral_molar_mass: Tensor
    ssa: Tensor
    aux_factor: Tensor | None
    has_aux: bool
    consts: _ReactionConsts
    num_aq: int
    num_min: int
    num_spec: int
    min_start: int
    do_reactions: bool
    compute_dtype: torch.dtype | None = None


class _FusedEvaluator:
    """Callable fused residual/Jacobian evaluator.

    Replaces a lambda closure with a picklable callable class.  Holds all
    constant tensors internally so ``__call__`` only takes ``conc``.
    """
    __slots__ = ("consts",)

    def __init__(self, consts: _FusedEvalConstants) -> None:
        self.consts = consts

    def __call__(self, conc: Tensor) -> tuple[Tensor, Tensor]:
        c: _FusedEvalConstants = self.consts
        return _kinetic_residual_and_jacobian(
            conc,
            c_0=c.c_0,
            dt=c.dt,
            v0=c.v0,
            depth=c.depth,
            q_in=c.q_in,
            c_in=c.c_in,
            stoich=c.stoich,
            monod_mat=c.monod_mat,
            inhib_mat=c.inhib_mat,
            tst_stoich=c.tst_stoich,
            tst_dep=c.tst_dep,
            min_eq_const=c.min_eq_const,
            rate_const=c.rate_const,
            mineral_molar_mass=c.mineral_molar_mass,
            ssa=c.ssa,
            aux_factor=c.aux_factor,
            has_aux=c.has_aux,
            consts=c.consts,
            num_aq=c.num_aq,
            num_min=c.num_min,
            num_spec=c.num_spec,
            min_start=c.min_start,
            do_reactions=c.do_reactions,
            compute_dtype=c.compute_dtype,
        )


# ---------------------------------------------------------------------- #
# Pre-allocated workspaces for RT solver (eliminates ~2M allocs/record)
# ---------------------------------------------------------------------- #

_KINETIC_WS_CACHE: dict[tuple, dict[str, Tensor]] = {}


def _get_kinetic_workspace(
    batch: int,
    num_spec: int,
    num_min: int,
    num_aq: int,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, Tensor]:
    """Get or create the scratch buffers consumed by :func:`_reaction_terms`.

    Every buffer is written *before* it is first read (``zero_ns`` is
    re-zeroed and ``am`` is copy-and-ANDed on each call; ``delta`` is a
    zeroed construction-time constant), so the same buffers are safely
    reused across Newton trips: the per-trip allocation churn (the dominant
    cost of the pre-Phase-3 hot path) collapses into one construction per
    ``(batch, num_spec, num_min, num_aq, dtype, device)`` key.  The four values that *escape* the call (``chms``, ``dm``, ``base``,
    ``r_full`` of the returned :class:`KineticTerms`) are deliberately not
    cached: they are read by the caller across solver trips and must not
    alias the scratch buffers.

    Buffer groups:

    * ``(B, num_min)``: TST exponent/rate values (``q_vals``, ``d_vals``,
      ``mn_vals``, ``b``, ``r1``..``r3``) and the activity mask ``am`` (the
      per-call finiteness factors are ANDed into it in place).
    * ``(B, num_spec)``: concentration guards (``chms_abs``, ``abs_ok``,
      ``safe``) and the broadcast zero fill ``zero_ns``.
    * ``(B, num_min, num_spec)``: Monod/inhibition derivative scratch
      (``mk_fin``, ``ik_fin``, ``{add,mul,div,g}_mk``, ``{add,mul,div,g}_ik``,
      ``g``, ``dmn``, ``tn*``, ``td*``, ``dts``, ``dm*``,
      ``dm_raw``).  The ``tn3`` / ``tn4`` / ``td1`` intermediates are
      ``(B, num_min, 1)``: their operands (``q_col`` / ``k_row``) carry only
      the mineral axis, so the result of those divisions/subtractions has a
      singleton species axis before it is re-broadcast in the following
      multiplies.
    * ``(num_min, num_spec)``: ``delta``, the mineral-row indicator with
      ``delta[i, num_aq + i] = 1.0`` and zero elsewhere (the reference
      hardcodes ``num_aq`` here).  A construction-time constant: zeroed once
      at build and then only read (the reference's per-call fresh
      ``torch.zeros`` is done once here instead), which also makes it immune
      to the inference-mode guard.

    The buffers are written on every call, so the entry is rebuilt if the
    current context cannot write the cached tensors (:func:`_ws_buffers_ok`).
    """
    key: tuple = ("kinetic", batch, num_spec, num_min, num_aq, dtype, device)
    ws: dict[str, Tensor] | None = _KINETIC_WS_CACHE.get(key)
    if ws is not None and not _ws_buffers_ok(ws):
        ws = None
    if ws is None:

        def empty(*shape: int, bool: bool = False) -> Tensor:
            return torch.empty(
                *shape, dtype=torch.bool if bool else dtype, device=device
            )

        ws = {
            # TST exponents / rates (B, num_min)
            "mn_vals": empty(batch, num_min),
            "q_vals": empty(batch, num_min),
            "d_vals": empty(batch, num_min),
            "b": empty(batch, num_min),
            "r1": empty(batch, num_min),
            "r2": empty(batch, num_min),
            "r3": empty(batch, num_min),
            # activity mask (B, num_min); the per-call finiteness factors are
            # ANDed in place (``isfinite`` has no ``out=`` variant, so they
            # are small fresh tensors, not buffers)
            "am": empty(batch, num_min, bool=True),
            # concentration guards (B, num_spec)
            "chms_abs": empty(batch, num_spec),
            "abs_ok": empty(batch, num_spec, bool=True),
            "safe": empty(batch, num_spec),
            "zero_ns": empty(batch, num_spec),
            # Monod / inhibition derivative scratch (B, num_min, num_spec)
            "mk_fin": empty(batch, num_min, num_spec, bool=True),
            "ik_fin": empty(batch, num_min, num_spec, bool=True),
            "add_mk": empty(batch, num_min, num_spec),
            "mul_mk": empty(batch, num_min, num_spec),
            "div_mk": empty(batch, num_min, num_spec),
            "g_mk": empty(batch, num_min, num_spec),
            "add_ik": empty(batch, num_min, num_spec),
            "mul_ik": empty(batch, num_min, num_spec),
            "div_ik": empty(batch, num_min, num_spec),
            "g_ik": empty(batch, num_min, num_spec),
            "g": empty(batch, num_min, num_spec),
            "dmn": empty(batch, num_min, num_spec),
            "tn1": empty(batch, num_min, num_spec),
            "tn2": empty(batch, num_min, num_spec),
            # (B, num_min, 1): q_col / k_row and 1 - (q_col / k_row) keep the
            # singleton species axis until the next multiply re-broadcasts.
            "tn3": empty(batch, num_min, 1),
            "tn4": empty(batch, num_min, 1),
            "tn5": empty(batch, num_min, num_spec),
            "td1": empty(batch, num_min, 1),
            "td2": empty(batch, num_min, num_spec),
            "td3": empty(batch, num_min, num_spec),
            "td4": empty(batch, num_min, num_spec),
            "dts": empty(batch, num_min, num_spec),
            "dm1": empty(batch, num_min, num_spec),
            "dm3": empty(batch, num_min, num_spec),
            "dm4": empty(batch, num_min, num_spec),
            "dm5": empty(batch, num_min, num_spec),
            "dm_raw": empty(batch, num_min, num_spec),
        }
        # (num_min, num_spec) mineral-row indicator: ``delta[i, num_aq + i] =
        # 1.0`` and zero elsewhere.  It is a construction-time constant (read
        # on every call, never rewritten), and the off-diagonal entries MUST
        # stay exactly 0.0: they are read (``delta * r_full``) with no mask,
        # so a plain ``torch.empty`` would leak allocator garbage into every
        # non-mineral column of ``dm`` (the reference mints a fresh per-call
        # ``torch.zeros``; the zeroing is done once here instead).
        delta: Tensor = torch.zeros(num_min, num_spec, dtype=dtype, device=device)
        for i in range(num_min):
            delta[i, num_aq + i] = 1.0
        ws["delta"] = delta
        _KINETIC_WS_CACHE[key] = ws
    return ws


def _get_jacobian_workspace(
    batch: int,
    num_spec: int,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, Tensor]:
    """Get or create the scratch buffers for Jacobian / ODE assembly.

    ``jac`` and ``ode`` are zeroed at the start of every call (``p`` is fully
    overwritten by its two slice writes) and the same buffers are reused
    across Newton trips, replacing the per-trip fresh ``torch.zeros``
    allocations.  ``react`` (the ``(B, num_spec, num_spec)`` block of
    ``stoich @ dm`` written by ``_assemble_ode_jacobian``) and ``ode_react``
    (the ``(B, num_spec, 1)`` ``stoich @ mcr`` product of
    ``_kinetic_ode_from_terms``) are likewise written before read on every
    call.  Only *read* results escape the assembling functions (the
    mixed-precision cast, or the fresh tensors the caller builds from them),
    so re-using the buffers in place cannot corrupt a live result.  The
    entry is rebuilt on a context switch (:func:`_ws_buffers_ok`) because the
    buffers are written on every call.
    """
    key: tuple = ("jacobian", batch, num_spec, dtype, device)
    ws: dict[str, Tensor] | None = _KINETIC_WS_CACHE.get(key)
    if ws is not None and not _ws_buffers_ok(ws):
        ws = None
    if ws is None:
        ws = {
            "jac": torch.empty(batch, num_spec, num_spec, dtype=dtype, device=device),
            "p": torch.empty(batch, num_spec, dtype=dtype, device=device),
            "ode": torch.empty(batch, num_spec, dtype=dtype, device=device),
            "react": torch.empty(batch, num_spec, num_spec, dtype=dtype, device=device),
            "ode_react": torch.empty(batch, num_spec, 1, dtype=dtype, device=device),
        }
        _KINETIC_WS_CACHE[key] = ws
    return ws


def _flatten_batch(x: Tensor) -> tuple[Tensor, tuple[int, ...]]:
    """Promote a 1-D ``(num_spec,)`` tensor to ``(1, num_spec)``.

    Returns ``(promoted, leading)`` where ``leading`` is the original shape
    without the trailing species axis (``()`` for a 1-D input).
    """
    if x.ndim == 1:
        return x.unsqueeze(0), ()
    leading: tuple[int, ...] = x.shape[:-1]
    return x.reshape(-1, x.shape[-1]), leading


def _restore(x: Tensor, leading: tuple[int, ...]) -> Tensor:
    """Reshape a ``(B, n)`` tensor back to ``(*leading, n)``."""
    return x.reshape(*leading, x.shape[-1])


def _to_vec(x: Scalar, b: int, dtype: torch.dtype, device: torch.device) -> Tensor:
    """Materialise a hydro scalar field as a ``(b,)`` tensor."""
    if isinstance(x, (int, float)):
        return torch.full((b,), float(x), dtype=dtype, device=device)
    t: Tensor = x.detach().to(dtype=dtype, device=device)
    return t.expand(b) if t.ndim == 0 else t.reshape(b)


def _coerce_dt(dt: Scalar, dtype: torch.dtype, device: torch.device) -> Tensor:
    """Promote the time step to a 0-dim tensor (broadcasts with everything)."""
    if isinstance(dt, (int, float)):
        return torch.tensor(float(dt), dtype=dtype, device=device)
    return dt.detach().to(dtype=dtype, device=device)


@dataclass(frozen=True)
class _HydroTensors:
    """Batch-normalised hydrological fields for one step (all ``(b,)``)."""

    state: Tensor
    q_in: Tensor
    q_internal: Tensor
    q_external: Tensor
    lat_flux_ext: Tensor
    vert_flux_ext: Tensor
    lat_flux: Tensor
    vert_flux: Tensor
    vap_flux: Tensor
    s_w: Tensor
    temp: Tensor


def _normalise_hydro(
    d: RtForcing, b: int, dtype: torch.dtype, device: torch.device
) -> _HydroTensors:
    """Coerce every hydrological scalar in ``d`` to ``(b,)`` tensors."""
    hs = d.hydro_step
    assert hs is not None, "RtForcing must carry a hydro_step"
    if d.hydro_forc is None:
        temp: Tensor = torch.full((b,), 20.0, dtype=dtype, device=device)
    else:
        temp = _to_vec(d.hydro_forc.temp, b, dtype, device)
    state = _to_vec(hs.state, b, dtype, device)
    lat_ext = _to_vec(hs.lat_flux_ext, b, dtype, device)
    vert_ext = _to_vec(hs.vert_flux_ext, b, dtype, device)
    lat = _to_vec(hs.lat_flux, b, dtype, device)
    vert = _to_vec(hs.vert_flux, b, dtype, device)
    q_internal: Tensor = lat + vert
    return _HydroTensors(
        state=state,
        q_in=_to_vec(hs.q_in, b, dtype, device),
        q_internal=q_internal,
        q_external=lat_ext + vert_ext - q_internal,
        lat_flux_ext=lat_ext,
        vert_flux_ext=vert_ext,
        lat_flux=lat,
        vert_flux=vert,
        vap_flux=_to_vec(hs.vap_flux, b, dtype, device),
        s_w=_to_vec(d.s_w, b, dtype, device),
        temp=temp,
    )


# ---------------------------------------------------------------------- #
# Moles <-> concentration (the mol/L storage convention)
# ---------------------------------------------------------------------- #


def moles_to_conc(
    tot_moles: Tensor,
    v0: Tensor,
    depth: float | Tensor,
    num_aq: int,
    out: Tensor | None = None,
) -> Tensor:
    """Convert total-moles storage to concentration (Rust ``moles_to_conc_rust``).

    Aqueous rows: ``m/v0`` (``ZERO_CONC`` when ``v0 < 1e-6``); non-aqueous rows:
    ``m/depth`` unconditionally.

    Args:
        tot_moles: (B, num_spec) storage vector.
        v0:        (B,) water volume (``state``).
        depth:     zone depth [mm] scalar, or a ``(B,)`` tensor when the depth
            varies across batch lanes (batched parameter sets).
        num_aq:    number of aqueous species.
        out:       optional pre-allocated output tensor.

    Returns:
        (B, num_spec) concentrations.
    """
    if out is not None:
        # Write both column blocks in place: this avoids the intermediate
        # ``aq`` / ``naq`` temporaries, the ``cat``, and the final copy.
        # ``torch.where`` with the scalar ``ZERO_CONC`` fill and ``torch.div``
        # are the same kernels as the non-out path, so values are
        # bit-identical.
        zero: Tensor = torch.tensor(
            ZERO_CONC, dtype=tot_moles.dtype, device=tot_moles.device
        )
        torch.where(
            v0.unsqueeze(-1) < 1e-6,
            zero,
            tot_moles[..., :num_aq] / v0.unsqueeze(-1),
            out=out[:, :num_aq],
        )
        if isinstance(depth, Tensor):
            torch.div(
                tot_moles[..., num_aq:], depth.unsqueeze(-1), out=out[:, num_aq:]
            )
        else:
            torch.div(tot_moles[..., num_aq:], depth, out=out[:, num_aq:])
        return out
    aq: Tensor = torch.where(
        v0.unsqueeze(-1) < 1e-6,
        torch.tensor(ZERO_CONC, dtype=tot_moles.dtype, device=tot_moles.device),
        tot_moles[..., :num_aq] / v0.unsqueeze(-1),
    )
    if isinstance(depth, Tensor):
        naq: Tensor = tot_moles[..., num_aq:] / depth.unsqueeze(-1)
    else:
        naq = tot_moles[..., num_aq:] / depth
    return torch.cat([aq, naq], dim=-1)


def get_tot_moles(
    conc: Tensor, v0: Tensor, depth: float | Tensor, num_aq: int
) -> Tensor:
    """Concentration to total-moles storage (Rust ``get_tot_moles_rust``).

    Aqueous rows: ``c * v0``; non-aqueous rows: ``c * depth``.

    Args:
        conc:  (B, num_spec) concentration vector.
        v0:    (B,) water volume (``state``).
        depth: zone depth [mm] scalar, or a ``(B,)`` tensor when the depth
            varies across batch lanes.
        num_aq: number of aqueous species.

    Returns:
        (B, num_spec) total-moles storage.
    """
    aq: Tensor = conc[..., :num_aq] * v0.unsqueeze(-1)
    if isinstance(depth, Tensor):
        naq: Tensor = conc[..., num_aq:] * depth.unsqueeze(-1)
    else:
        naq = conc[..., num_aq:] * depth
    return torch.cat([aq, naq], dim=-1)


# ---------------------------------------------------------------------- #
# Shared kinetic terms (port of ``reaction_terms_impl``)
# ---------------------------------------------------------------------- #


@dataclass(frozen=True)
class KineticTerms:
    """Kinetic quantities evaluated once and reused by residual and Jacobian.

    Attributes:
        chms:   (B, num_spec) renormalised concentrations.
        dm:     (B, num_min, num_spec) derivative ``dM/dchms``.
        base:   (B, num_min) prefactor ``A_i``.
        r_full: (B, num_min) full rate factor ``Mn + D*(1 - Q/K)``.
    """

    chms: Tensor
    dm: Tensor
    base: Tensor
    r_full: Tensor


def _reaction_terms(
    *,
    conc: Tensor,
    v0: Tensor,
    depth: float | Tensor,
    monod_mat: Tensor,
    inhib_mat: Tensor,
    tst_stoich: Tensor,
    tst_dep: Tensor,
    min_eq_const: Tensor,
    rate_const: Tensor,
    mineral_molar_mass: Tensor,
    ssa: Tensor,
    aux_factor: Tensor | None,
    has_aux: bool,
    num_aq: int,
    num_min: int,
    num_spec: int,
    do_reactions: bool,
    consts: _ReactionConsts | None = None,
) -> KineticTerms:
    """Batched port of ``reaction_terms_impl`` (``rt_zone.rs``).

    In the fused solve the kinetic terms act on the *renormalised* state
    ``chms = moles_to_conc(conc)``: aqueous rows are divided by ``state`` yet
    again (the reference quirk), non-aqueous rows by ``depth``.

    Governing quantities (per mineral ``i``, all tensors batched over lanes
    ``B``):

    * ``Q_i = 10**(log10 s(c) @ stoich_i) = prod_j s(c_j)**nu_ij``,
      ``D_i = 10**(log10 s(c) @ dep_i) = prod_j s(c_j)**mu_ij`` — the TST
      solubility product / dependence, power products over the smoothed
      concentrations ``s =
      :func:`~potions.kinetic_structures.smoothed_conc`` (Rust
      ``calculate_solubility_product`` / ``calculate_dependence_term``):
      every log-contribution stays finite (>= -300) for zero or negative
      trial concentrations as well.
    * ``Mn_i = prod_j c_j/(K_j+c_j) * prod_j K_j/(K_j+c_j)`` — the Monod rate
      (:func:`monod_inhib_rates`).
    * ``r_full_i = Mn_i + D_i * (1 - Q_i / K_i)`` — full rate factor, computed
      for **all** minerals (no activity gate).
    * ``g_i = d(Mn part)/dchms - d(inhib part)/dchms`` — the per-species
      derivative of the Monod/inhibition product, with a ``0`` fill where a
      constant is ``NaN`` (no term) or the concentration is non-positive.
    * ``dts_i = D_i * dep_i * (s'/s) * (1 - Q_i/K_i)
               - D_i * Q_i * stoich_i * (s'/s) / K_i`` — per-species derivative of
      the TST factor, with ``s'/s =
      :func:`~potions.kinetic_structures.smoothed_rel_deriv`` finite for
      every trial ``c`` (no non-positive-concentration gate, whose jump was
      the zero-plane kink the smoothing removes).
    * ``base_i = 86400 * rc_i * ssa_i * mm_i * aux_i`` — the prefactor ``A_i``,
      zero where ``ssa`` is non-finite (and zero entirely without aux).
    * ``dm_i = A_i * (delta_i * r_full_i + c_min_i * (Mn_i * g_i + dts_i))``
      — the derivative of the mineral source term, gated by the per-mineral
      activity mask (all relevant constants/values finite, and aux present).

    Every per-trip temporary is written into the id-keyed scratch workspace
    (:func:`_get_kinetic_workspace`); the four escaping values (``chms``,
    ``dm``, ``base``, ``r_full``) are fresh tensors so they never alias a
    reused buffer.  The network-constant invariants (finite masks, ``86400*rc``
    scale, zero fills) come from :class:`_ReactionConsts` via
    :func:`_get_reaction_consts`.

    Args:
        conc:     (B, num_spec) concentration state ``c``.
        v0:       (B,) water volume (``state``).
        depth:    zone depth [mm].
        monod_mat, inhib_mat: (num_min, num_spec) Monod/inhibition constants
            (``NaN`` where the reaction defines none).
        tst_stoich, tst_dep: (num_min, num_spec) Q / D exponent matrices.
        min_eq_const: (num_min,) equilibrium constants.
        rate_const, mineral_molar_mass, ssa: (num_min,) auxiliary constants.
        aux_factor: (B, num_min) combined soil-water/temperature factor, or
            ``None`` when ``has_aux`` is false.
        has_aux:   whether mineral auxiliary parameters were supplied.
        num_aq:    number of aqueous species.
        num_min:   number of minerals.
        num_spec:  total species count.
        do_reactions: if false return empty kinetic blocks (transport only).
        consts:    precomputed :class:`_ReactionConsts`; auto-built (cached)
            from the passed network constants when ``None``.

    Returns:
        :class:`KineticTerms` with the shared values.
    """
    batch: int = conc.shape[0]
    dtype: torch.dtype = conc.dtype
    device: torch.device = conc.device

    if consts is None:
        consts = _get_reaction_consts(
            monod_mat=monod_mat,
            inhib_mat=inhib_mat,
            min_eq_const=min_eq_const,
            rate_const=rate_const,
            mineral_molar_mass=mineral_molar_mass,
            dtype=dtype,
            device=device,
        )

    if not do_reactions:
        return KineticTerms(
            chms=conc,
            dm=torch.zeros(batch, 0, num_spec, dtype=dtype, device=device),
            base=torch.zeros(batch, 0, dtype=dtype, device=device),
            r_full=torch.zeros(batch, 0, dtype=dtype, device=device),
        )

    ws: dict[str, Tensor] = _get_kinetic_workspace(
        batch, num_spec, num_min, num_aq, dtype, device
    )

    # ---- chms = moles_to_conc(conc): the renormalised state (B, ns) ----
    chms: Tensor = torch.empty(batch, num_spec, dtype=dtype, device=device)
    moles_to_conc(conc, v0, depth, num_aq, out=chms)
    monod_inhib_rates(monod_mat, inhib_mat, chms, out=ws["mn_vals"])  # (B, nm)

    if num_min == 0:
        return KineticTerms(
            chms=chms,
            dm=torch.zeros(batch, 0, num_spec, dtype=dtype, device=device),
            base=torch.zeros(batch, 0, dtype=dtype, device=device),
            r_full=torch.zeros(batch, 0, dtype=dtype, device=device),
        )

    # ---- Q = 10**(log10 s(c) @ stoich.T), D = 10**(log10 s(c) @ dep.T)  (B, nm) ----
    # Power products over the smoothed concentrations (Rust
    # calculate_solubility_product / calculate_dependence_term): s(c) keeps
    # every log-contribution finite (>= -300) for zero / negative / NaN trial
    # states, and for every physical c (c > 5e-29) s(c) is bit-identical to
    # c, so the values are unchanged there.
    sc: Tensor = smoothed_conc(chms)
    log_conc: Tensor = torch.log10(sc)
    torch.matmul(log_conc, tst_stoich.T, out=ws["q_vals"])
    torch.pow(10.0, ws["q_vals"], out=ws["q_vals"])
    torch.matmul(log_conc, tst_dep.T, out=ws["d_vals"])
    torch.pow(10.0, ws["d_vals"], out=ws["d_vals"])

    # ---- base = A_i = (86400*rc) * ssa * mm * aux  (aux present & ssa finite) ----
    if has_aux:
        assert aux_factor is not None, "has_aux requires a non-None aux_factor"
        ssa_finite: Tensor = ssa.isfinite()  # (nm,) or (B, nm)
        # The first factors stay on the mineral axis ((nm,) when ``ssa`` is a
        # single parameter set) exactly as in the reference chain; only the
        # final multiply against the batched ``aux_factor`` promotes to
        # (B, nm), which is what lands in the workspace buffer.
        scale: Tensor = consts.rate_scale * ssa
        scale = scale * mineral_molar_mass
        b: Tensor = ws["b"]
        torch.mul(scale, aux_factor, out=b)
        base: Tensor = torch.where(ssa_finite, b, consts.zero_mn)  # fresh
    else:
        base = torch.zeros(batch, num_min, dtype=dtype, device=device)

    # ---- r_full = Mn + D * (1 - Q / K)  (ALL minerals, no activity gate) ----
    torch.div(ws["q_vals"], min_eq_const, out=ws["r1"])
    torch.sub(1.0, ws["r1"], out=ws["r2"])
    torch.mul(ws["d_vals"], ws["r2"], out=ws["r3"])
    r_full: Tensor = torch.empty(batch, num_min, dtype=dtype, device=device)
    torch.add(ws["mn_vals"], ws["r3"], out=r_full)

    # ---- concentration guards (B, ns) ----
    # ``abs > 0.0`` (not ``!= 0``): NaN must read as non-positive here.
    # Only the Monod/inhibition block gates on positivity (``safe``); the
    # TST derivative uses ``smoothed_rel_deriv``, which is finite for every
    # trial ``c`` and needs no such guard.
    torch.abs(chms, out=ws["chms_abs"])
    torch.gt(ws["chms_abs"], 0.0, out=ws["abs_ok"])
    ws["zero_ns"].zero_()
    torch.where(ws["abs_ok"], chms, ws["zero_ns"], out=ws["safe"])

    # ---- broadcast views (no copies) ----
    c1: Tensor = ws["safe"].unsqueeze(1)               # (B, 1, ns)
    mk: Tensor = monod_mat.unsqueeze(0)                # (1, nm, ns)
    ik: Tensor = inhib_mat.unsqueeze(0)                # (1, nm, ns)
    mn_l: Tensor = ws["mn_vals"].unsqueeze(-1)         # (B, nm, 1)
    d_col: Tensor = ws["d_vals"].unsqueeze(-1)         # (B, nm, 1)
    q_col: Tensor = ws["q_vals"].unsqueeze(-1)         # (B, nm, 1)
    # Mineral axis kept in the middle so this broadcasts for any ``num_min``.
    k_row: Tensor = min_eq_const.reshape(1, -1, 1)     # (1, nm, 1)
    # Fresh (not workspace-cached) each call: it is a pure function of the
    # per-trip ``chms`` and stays safe under torch.inference_mode().
    rel_c: Tensor = smoothed_rel_deriv(chms).unsqueeze(1)  # (B, 1, ns)
    dep0: Tensor = tst_dep.unsqueeze(0)                # (1, nm, ns)
    stoich0: Tensor = tst_stoich.unsqueeze(0)          # (1, nm, ns)
    min_conc_aq: Tensor = chms[:, num_aq : num_aq + num_min]  # (B, nm) view
    mca_l: Tensor = min_conc_aq.unsqueeze(-1)          # (B, nm, 1)
    r_full_l: Tensor = r_full.unsqueeze(-1)            # (B, nm, 1)
    base_l: Tensor = base.unsqueeze(-1)                # (B, nm, 1)
    delta0: Tensor = ws["delta"].unsqueeze(0)          # (1, nm, ns)

    # ---- g = d(Monod part)/dc - d(inhib part)/dc : (B, nm, ns) ----
    # The finite masks combine the per-network constant mask (cached) with the
    # per-call ``abs_ok`` (a species must be positive for the derivative to be
    # meaningful); non-finite / non-positive lanes fall back to the zero fill.
    torch.bitwise_and(consts.mk_fin, ws["abs_ok"].unsqueeze(1), out=ws["mk_fin"])
    torch.bitwise_and(consts.ik_fin, ws["abs_ok"].unsqueeze(1), out=ws["ik_fin"])
    torch.add(mk, c1, out=ws["add_mk"])
    torch.mul(c1, ws["add_mk"], out=ws["mul_mk"])
    torch.div(mk, ws["mul_mk"], out=ws["div_mk"])
    torch.where(ws["mk_fin"], ws["div_mk"], consts.zero_mn_ns, out=ws["g_mk"])
    torch.add(ik, c1, out=ws["add_ik"])
    torch.mul(c1, ws["add_ik"], out=ws["mul_ik"])
    torch.div(ik, ws["mul_ik"], out=ws["div_ik"])
    torch.where(ws["ik_fin"], ws["div_ik"], consts.zero_mn_ns, out=ws["g_ik"])
    torch.sub(ws["g_mk"], ws["g_ik"], out=ws["g"])

    # ---- dts : (B, nm, ns)  (derivative of D*(1 - Q/K) w.r.t. chms) ----
    # Q and D are power products over the smoothed concentrations, so the
    # per-species derivative uses ``s'(c)/s(c)`` (:func:`smoothed_rel_deriv`)
    # in place of ``1/c``.  It is finite for every trial ``c`` (7.2e29 at
    # zero, smoothly decaying to 0 below ``c ~= -7.5e-28``, 0 for NaN), so
    # the old non-positive gate — whose 0 <-> finite jump was the zero-plane
    # kink that stalled Newton / LM — is gone.  Operation order mirrors the
    # Rust scalar chain: ``(d*dep*rel)*(1-q/k) - ((d*q*stoich)*rel)/k``.
    torch.mul(d_col, dep0, out=ws["tn1"])
    torch.mul(ws["tn1"], rel_c, out=ws["tn2"])
    torch.div(q_col, k_row, out=ws["tn3"])  # (B, nm, 1)
    torch.sub(1.0, ws["tn3"], out=ws["tn4"])
    torch.mul(ws["tn2"], ws["tn4"], out=ws["tn5"])
    torch.mul(d_col, q_col, out=ws["td1"])  # (B, nm, 1)
    torch.mul(ws["td1"], stoich0, out=ws["td2"])
    torch.mul(ws["td2"], rel_c, out=ws["td3"])
    torch.div(ws["td3"], k_row, out=ws["td4"])
    torch.sub(ws["tn5"], ws["td4"], out=ws["dts"])

    # ---- dm = A_i * (delta*r_full + c_min*(Mn*g + dts)) : (B, nm, ns) ----
    torch.mul(mn_l, ws["g"], out=ws["dmn"])
    torch.mul(delta0, r_full_l, out=ws["dm1"])
    torch.add(ws["dmn"], ws["dts"], out=ws["dm3"])
    torch.mul(mca_l, ws["dm3"], out=ws["dm4"])
    torch.add(ws["dm1"], ws["dm4"], out=ws["dm5"])
    torch.mul(base_l, ws["dm5"], out=ws["dm_raw"])

    # ---- activity mask am : (B, nm) bool ----
    # Starts from the cached constant half of the reference's boolean chain
    # (rate / molar-mass / equilibrium-constant finiteness and |K| > 0), then
    # ANDs in the per-call finite pieces; the aux gate is the final factor
    # (zero mask entirely when aux is absent).  Boolean AND is exact, so this
    # regrouping is value-identical to the reference's 8-term chain.
    am: Tensor = ws["am"]
    am.copy_(consts.const_active)
    # torch 2.14 exposes no ``out=`` variant of ``isfinite`` (nor of
    # ``isnan``/``isinf``), so each per-call finiteness check uses the method
    # form — a small fresh ``(B, num_min)`` bool — and is ANDed straight into
    # ``am``.  The values are exactly the reference's 8-term chain, and the
    # op count (4 ``isfinite`` + 4 in-place ``&=``) stays below the original
    # (7 ``isfinite`` + 6 ``&``) because the constant half is cached.
    am &= min_conc_aq.isfinite()
    am &= ws["mn_vals"].isfinite()
    am &= ws["d_vals"].isfinite()
    am &= ws["q_vals"].isfinite()
    if has_aux:
        am &= ssa_finite
    else:
        am.zero_()

    dm: Tensor = torch.where(am.unsqueeze(-1), ws["dm_raw"], consts.zero_mn_ns)

    return KineticTerms(chms=chms, dm=dm, base=base, r_full=r_full)


# ---------------------------------------------------------------------- #
# ODE Jacobian assembly (port of ``assemble_ode_jacobian``)
# ---------------------------------------------------------------------- #


def _assemble_ode_jacobian(
    terms: KineticTerms,
    stoich: Tensor,
    *,
    num_aq: int,
    num_spec: int,
    depth: float | Tensor,
    v0: Tensor,
    q_in: Tensor,
    do_reactions: bool,
    compute_dtype: torch.dtype | None = None,
    f: Tensor | None = None,
) -> Tensor:
    """Batched port of ``assemble_ode_jacobian`` (``rt_zone.rs``).

    The Jacobian is ``-f`` on the aqueous diagonal (transport) plus the kinetic
    block ``stoich @ dm`` scaled by the ``moles_to_conc`` Jacobian ``p`` on the
    aqueous rows, all on a workspace buffer that is zeroed (``jac``) or fully
    overwritten (``p``) at the top of the call and reused across Newton trips.

    Args:
        terms:   precomputed kinetic terms.
        stoich:  (num_spec, num_min) mineral stoichiometry.
        num_aq:  number of aqueous species.
        num_spec: total species count.
        depth:   zone depth [mm] scalar, or a ``(B,)`` tensor when the depth
            varies across batch lanes.
        v0:      (B,) water volume.
        q_in:    (B,) water flux driving transport.
        do_reactions: whether kinetics are active.
        compute_dtype: Optional dtype for computation (e.g., float32 for
            mixed precision with float64 convergence checks).
        f:       precomputed inflow flux ``where(|v0| >= 1e-6, q_in/v0, 0)``
            (``(B,)``); computed here when ``None`` so a fused caller can
            share one evaluation with :func:`_kinetic_ode_from_terms`.

    Returns:
        (B, num_spec, num_spec) ODE Jacobian.
    """
    batch: int = terms.chms.shape[0]
    dtype: torch.dtype = terms.chms.dtype
    device: torch.device = terms.chms.device
    compute_dtype = compute_dtype or dtype

    # ``jac`` is zeroed here; the workspace guard (:func:`_ws_buffers_ok`)
    # rebuilds the buffers on a context switch, so the zero-then-accumulate
    # pattern is safe even after a mode change.
    ws = _get_jacobian_workspace(batch, num_spec, compute_dtype, device)
    jac: Tensor = ws["jac"]
    jac.zero_()

    # NOTE: no ``if v0_ok.any()`` host guard here -- it would cost a
    # host-device sync per call, and the batch mixes dry and wet lanes, so the
    # flux must be computed per lane; ``torch.where`` already yields a zero
    # flux for dry lanes, so the masked assignment is the same arithmetic the
    # original guard performed whenever any lane was wet.
    if f is None:
        v0_ok: Tensor = v0.abs() >= 1e-6                          # (B,)
        f = torch.where(v0_ok, q_in / v0, torch.zeros_like(v0))
    # Cached index range (loop invariant of a zone step).
    aq_idx = _aq_range(num_aq, device)
    jac[:, aq_idx, aq_idx] = -f.unsqueeze(-1).expand(-1, num_aq)

    if not do_reactions:
        return jac.to(dtype)

    s: Tensor = _get_stoich(stoich, compute_dtype, device)        # (ns, nm)

    # P = d(moles_to_conc)/dc, diagonal.  The two slice writes below cover
    # every column, so the cached buffer needs no pre-zeroing.  The raw
    # ``1/v0`` (inf/NaN on dry lanes) is intentional: it reproduces the
    # reference's 0*inf == NaN quirk on dead lanes.
    p: Tensor = ws["p"]
    p[:, :num_aq] = (1.0 / v0).unsqueeze(-1)
    if isinstance(depth, Tensor):
        if depth.ndim == 0:
            # 0-dim depth is a per-zone loop invariant: cached inverse.
            depth_inv: Tensor = _get_depth_inv(depth)
        else:
            # (B,) per-lane depth is per-step data: computed fresh.
            depth_inv = torch.where(
                depth.abs() > 0.0, 1.0 / depth, torch.zeros_like(depth)
            )
        p[:, num_aq:] = depth_inv.unsqueeze(-1)
    else:
        depth_inv_scalar: float = 1.0 / depth if abs(depth) > 0.0 else 0.0
        p[:, num_aq:] = depth_inv_scalar

    react: Tensor = ws["react"]
    torch.matmul(s.unsqueeze(0), terms.dm.to(compute_dtype), out=react)
    jac[:, :num_aq, :] += react[:, :num_aq, :] * p.unsqueeze(1)

    return jac.to(dtype)


def _kinetic_ode_from_terms(
    terms: KineticTerms,
    stoich: Tensor,
    *,
    conc: Tensor,
    c_in: Tensor,
    v0: Tensor,
    q_in: Tensor,
    num_aq: int,
    num_spec: int,
    min_start: int,
    do_reactions: bool,
    compute_dtype: torch.dtype | None = None,
    f: Tensor | None = None,
) -> Tensor:
    """Batched port of ``kinetic_ode_from_terms`` (``rt_zone.rs``).

    The right-hand side is the transport exchange on the aqueous rows
    (``f * (c_in - conc)``) plus the reaction source (``stoich @ mcr``, zeroed
    on the mineral rows), accumulated on a workspace buffer that is zeroed at
    the top of the call and reused across Newton trips.

    Args:
        terms:    precomputed kinetic terms.
        stoich:   (num_spec, num_min) mineral stoichiometry.
        conc:     (B, num_spec) concentration state.
        c_in:     (B, num_spec) incoming concentration.
        v0:       (B,) water volume.
        q_in:     (B,) water flux driving transport.
        num_aq:   number of aqueous species.
        num_spec: total species count.
        min_start: index of the first mineral row in the species vector.
        do_reactions: whether kinetics are active.
        compute_dtype: Optional dtype for computation (e.g., float32 for
            mixed precision with float64 convergence checks).
        f:       precomputed inflow flux ``where(|v0| >= 1e-6, q_in/v0, 0)``
            (``(B,)``); computed here when ``None`` so a fused caller can
            share one evaluation with :func:`_assemble_ode_jacobian`.

    Returns:
        (B, num_spec) ODE right-hand side.
    """
    batch: int = conc.shape[0]
    dtype: torch.dtype = conc.dtype
    device: torch.device = conc.device
    compute_dtype = compute_dtype or dtype

    # Workspace buffer (zeroed here; the workspace guard rebuilds it on a
    # context switch).  The two writes below (aqueous slice, then reaction
    # addition) consume it before it escapes as ``ode.to(dtype)``.
    ws = _get_jacobian_workspace(batch, num_spec, compute_dtype, device)
    ode: Tensor = ws["ode"]
    ode.zero_()

    # No ``if v0_ok.any()`` host guard (it would cost a host-device sync per
    # call, and the batch mixes dry and wet lanes, so the flux must be per
    # lane): ``f`` is zero for dry lanes via ``torch.where``, which
    # reproduces the original arithmetic on the mixed/wet path.
    if f is None:
        v0_ok: Tensor = v0.abs() >= 1e-6
        f = torch.where(v0_ok, q_in / v0, torch.zeros_like(v0))
    ode[:, :num_aq] = f.unsqueeze(-1) * (c_in[:, :num_aq] - conc[:, :num_aq])

    if do_reactions:
        num_min: int = terms.base.shape[1]
        mcr: Tensor = (
            terms.base * terms.chms[:, min_start : min_start + num_min] * terms.r_full
        )  # (B, nm)
        s: Tensor = _get_stoich(stoich, compute_dtype, device)
        # ``stoich @ mcr`` into the workspace, then zero the mineral rows on
        # the view before adding into ``ode`` (the reference's quirk).
        ode_react: Tensor = ws["ode_react"]
        torch.matmul(s, mcr.unsqueeze(-1), out=ode_react)  # (B, ns, 1)
        react: Tensor = ode_react.squeeze(-1)              # (B, ns) view
        react[:, min_start:] = 0.0
        ode += react

    return ode.to(dtype)


# ---------------------------------------------------------------------- #
# Fused residual + Jacobian (port of ``kinetic_residual_and_jacobian_impl``)
# ---------------------------------------------------------------------- #


def _kinetic_residual_and_jacobian(
    conc: Tensor,
    *,
    c_0: Tensor,
    dt: Tensor,
    v0: Tensor,
    depth: float | Tensor,
    q_in: Tensor,
    c_in: Tensor,
    stoich: Tensor,
    monod_mat: Tensor,
    inhib_mat: Tensor,
    tst_stoich: Tensor,
    tst_dep: Tensor,
    min_eq_const: Tensor,
    rate_const: Tensor,
    mineral_molar_mass: Tensor,
    ssa: Tensor,
    aux_factor: Tensor | None,
    has_aux: bool,
    num_aq: int,
    num_min: int,
    num_spec: int,
    min_start: int,
    do_reactions: bool,
    compute_dtype: torch.dtype | None = None,
    consts: _ReactionConsts | None = None,
) -> tuple[Tensor, Tensor]:
    """Batched port of ``kinetic_residual_and_jacobian_impl`` (``rt_zone.rs``).

    Computes the implicit-step residual and Jacobian in one fused evaluation:
    ``res = c_0 - conc + dt*ODE(conc)`` and ``jac = -I + dt*dODE/dc``, where
    the ODE and its Jacobian are assembled from the kinetic terms at ``conc``.

    Args:
        conc:  (B, num_spec) candidate state; all other tensors are per-zone /
               per-step constants cast to ``conc``'s dtype/device.
        consts: precomputed :class:`_ReactionConsts` shared with
            :func:`_reaction_terms` (auto-built when ``None``).

    Returns:
        ``(res, jac)`` with ``res = c_0 - conc + dt*ODE`` and
        ``jac = -I + dt*dODE/dc``.
    """
    dtype: torch.dtype = conc.dtype
    device: torch.device = conc.device
    eye: Tensor = _get_eye(num_spec, dtype, device)

    # The inflow flux is a loop invariant shared by the Jacobian and the ODE
    # assembly; evaluate it once per trip instead of twice.
    v0_ok: Tensor = v0.abs() >= 1e-6
    f: Tensor = torch.where(v0_ok, q_in / v0, torch.zeros_like(v0))

    terms = _reaction_terms(
        conc=conc,
        v0=v0,
        depth=depth,
        monod_mat=monod_mat,
        inhib_mat=inhib_mat,
        tst_stoich=tst_stoich,
        tst_dep=tst_dep,
        min_eq_const=min_eq_const,
        rate_const=rate_const,
        mineral_molar_mass=mineral_molar_mass,
        ssa=ssa,
        aux_factor=aux_factor,
        has_aux=has_aux,
        num_aq=num_aq,
        num_min=num_min,
        num_spec=num_spec,
        do_reactions=do_reactions,
        consts=consts,
    )

    j_ode: Tensor = _assemble_ode_jacobian(
        terms,
        stoich,
        num_aq=num_aq,
        num_spec=num_spec,
        depth=depth,
        v0=v0,
        q_in=q_in,
        do_reactions=do_reactions,
        compute_dtype=compute_dtype,
        f=f,
    )
    jac_x: Tensor = -eye.unsqueeze(0) + dt * j_ode

    ode: Tensor = _kinetic_ode_from_terms(
        terms,
        stoich,
        conc=conc,
        c_in=c_in,
        v0=v0,
        q_in=q_in,
        num_aq=num_aq,
        num_spec=num_spec,
        min_start=min_start,
        do_reactions=do_reactions,
        compute_dtype=compute_dtype,
        f=f,
    )
    res: Tensor = (c_0 - conc) + dt * ode
    return res, jac_x


# ---------------------------------------------------------------------- #
# Standalone reaction-rate evaluation (port of ``reaction_rate_rust``)
# ---------------------------------------------------------------------- #


def _reaction_rate_rust(
    tot_moles: Tensor,
    *,
    v0: Tensor,
    depth: float | Tensor,
    num_aq: int,
    num_min: int,
    num_spec: int,
    min_start: int,
    stoich: Tensor,
    monod_mat: Tensor,
    inhib_mat: Tensor,
    tst_stoich: Tensor,
    tst_dep: Tensor,
    min_eq_const: Tensor,
    rate_const: Tensor,
    mineral_molar_mass: Tensor,
    ssa: Tensor,
    aux_factor: Tensor | None,
    has_aux: bool,
    zone_name: str,
    minerals_only: bool = False,
) -> tuple[Tensor, Tensor | None]:
    """Batched port of ``reaction_rate_rust`` (``rt_zone.rs``).

    The input ``tot_moles`` is already a concentration vector, yet it goes
    through ``moles_to_conc`` first — the double-conversion quirk (aqueous rows
    are divided by ``state`` again, non-aqueous by ``depth``).

    Args:
        tot_moles: (B, num_spec) concentration vector.
        v0:        (B,) water volume.
        depth:     zone depth.
        min_start: index of the first mineral row in the species vector.
        minerals_only: if true only the per-mineral rates ``(B, num_min)`` are
            returned (``all_species`` is ``None``), else both.

    Returns:
        ``(mineral_conc_rates, all_species_rates)``; the second is ``None`` when
        ``minerals_only``.
    """
    name_disp: str = zone_name if has_aux else "zone"
    if not has_aux:
        raise ValueError(
            f"Trying to calculate kinetic rate in zone '{name_disp}'"
            " with kinetic rates disabled"
        )
    assert aux_factor is not None, "has_aux requires a non-None aux_factor"
    chms: Tensor = moles_to_conc(tot_moles, v0, depth, num_aq)
    mn_vals: Tensor = monod_inhib_rates(monod_mat, inhib_mat, chms)
    tst_rate: Tensor = tst_rates(tst_stoich, tst_dep, min_eq_const, chms)

    min_conc: Tensor = chms[:, min_start : min_start + num_min]  # (B, nm)
    mineral_rates: Tensor = (
        86_400.0
        * rate_const
        * ssa
        * mineral_molar_mass
        * min_conc
        * aux_factor
        * (mn_vals + tst_rate)
    )  # (B, nm)

    if minerals_only:
        return mineral_rates, None

    s: Tensor = stoich.to(dtype=tot_moles.dtype, device=tot_moles.device)
    all_species: Tensor = (s @ mineral_rates.unsqueeze(-1)).squeeze(-1)
    all_species[:, min_start:] = 0.0
    return mineral_rates, all_species


# ---------------------------------------------------------------------- #
# Shared per-zone step internals
# ---------------------------------------------------------------------- #


@dataclass(frozen=True)
class _FusedEvalContext:
    """Everything one reactive-transport fused solve needs from a zone step.

    Bundles the batch-normalised hydro fields, the cast network constants and
    the fused ``(residual, Jacobian)`` closure produced while setting up a
    step.  Both the internal step kernel (:func:`_zone_step`) and the public
    microbenchmark entry point (:meth:`RtZone.build_fused_eval`) build it once
    from the same code path, so the kinetic solve can be timed in isolation
    without duplicating (and risking drift from) the casting rules.

    Attributes:
        fused:               fused ``(residual, Jacobian)`` callable on
                             ``(B, num_spec)`` iterates.
        c0b:                 ``(B, num_spec)`` flattened initial iterate.
        leading:             original leading dims of ``c_0`` (``()`` for 1-D).
        batch:               number of lanes ``B``.
        dtype:               working dtype.
        device:              working device.
        hydro:               batch-normalised hydrological fields.
        q_in:                selected inflow flux (``q_in`` or ``q_internal``).
        cin_b:               ``(B, num_spec)`` incoming concentration.
        num_spec:            number of species.
        num_aq:              number of aqueous species.
        num_min:             number of minerals.
        min_start:           index of the first mineral row.
        stoich:              cast mineral stoichiometry.
        monod_mat:           cast Monod matrix.
        inhib_mat:           cast inhibition matrix.
        tst_stoich:          cast TST stoichiometry.
        tst_dep:             cast TST dependency matrix.
        min_eq_const:        cast TST mineral equilibrium constants.
        rate_const:          cast rate constants.
        mineral_molar_mass:  cast mineral molar masses.
        ssa:                 cast (or zero) specific surface areas.
        aux_factor:          ``(B, num_min)`` soil-water/temperature factors,
            or ``None`` when no auxiliary parameters were supplied (never
            read in that case).
        has_aux:             whether auxiliary kinetic parameters were supplied.
        consts:              per-network :class:`_ReactionConsts` (finite
            masks, rate scale, zero fills) shared with the fused evaluation.
    """

    fused: FusedEval
    c0b: Tensor
    leading: tuple[int, ...]
    batch: int
    dtype: torch.dtype
    device: torch.device
    hydro: _HydroTensors
    q_in: Tensor
    cin_b: Tensor
    num_spec: int
    num_aq: int
    num_min: int
    min_start: int
    stoich: Tensor
    monod_mat: Tensor
    inhib_mat: Tensor
    tst_stoich: Tensor
    tst_dep: Tensor
    min_eq_const: Tensor
    rate_const: Tensor
    mineral_molar_mass: Tensor
    ssa: Tensor
    aux_factor: Tensor | None
    has_aux: bool
    consts: _ReactionConsts


def _build_fused_eval(
    *,
    c_0: Tensor,
    d: RtForcing,
    dt: Scalar,
    network: ReactionNetwork,
    monod: MonodParameters,
    tst: TstParameters,
    aux: MineralParameters | None,
    misc: MiscData,
    depth: float | Tensor,
    min_start: int,
    q_in_field: str,
    do_reactions: bool,
    compute_dtype: torch.dtype | None = None,
) -> _FusedEvalContext:
    """Flatten inputs, cast network constants and build the fused closure.

    Shared setup for a single reactive-transport step: flattens ``c_0`` /
    ``conc_in`` to a leading batch dim, normalises the hydro forcing, casts
    every network constant once onto the working dtype/device and wraps
    ``_kinetic_residual_and_jacobian`` in the ``fused(conc) -> (residual,
    Jacobian)`` closure consumed by the Newton-to-LM solver.

    Args:
        c_0:         (num_spec,) or (B, num_spec) initial concentrations.
        d:           reactive-transport forcing for this step/zone.
        dt:          time step length [days].
        network:     the reaction network.
        monod:       per-mineral Monod/inhibition matrices.
        tst:         per-mineral TST matrices.
        aux:         per-mineral auxiliary kinetics, or ``None``.
        misc:        static per-network arrays.
        depth:       zone depth [mm] scalar or ``(B,)`` tensor.
        min_start:   index of the first mineral row.
        q_in_field:  ``"q_in"`` or ``"q_internal"`` inflow selection.
        do_reactions: whether mineral kinetics are active.

    Returns:
        A fully populated :class:`_FusedEvalContext`.
    """
    c0b, leading = _flatten_batch(c_0)
    batch: int = c0b.shape[0]
    dtype: torch.dtype = c0b.dtype
    device: torch.device = c0b.device
    compute_dtype = compute_dtype or dtype
    # ``ctx`` fields stay on the *external* dtype (parity / bookkeeping /
    # convergence), while the fused-constant copies below are cast to
    # ``compute_dtype`` so the whole RT solve runs on one mixed-precision dtype
    # (e.g. float32 compute with float64 convergence checks).
    num_spec: int = network.num_species
    num_aq: int = network.num_aqueous_species
    num_min: int = network.num_minerals
    h: _HydroTensors = _normalise_hydro(d, batch, dtype, device)
    q_in: Tensor = h.q_in if q_in_field == "q_in" else h.q_internal
    dtb: Tensor = _coerce_dt(dt, dtype, device)

    # ---- incoming concentration ----
    cin_raw: Tensor = d.conc_in
    if cin_raw.ndim == 0 or cin_raw.shape[-1] != num_spec:
        raise ValueError(
            f"RtForcing.conc_in must be (num_spec={num_spec},)-shaped, got {tuple(cin_raw.shape)}"
        )
    cin_f, _ = _flatten_batch(cin_raw.to(dtype=dtype, device=device))
    if cin_f.shape[0] == 1 and batch > 1:
        cin_b: Tensor = cin_f.expand(batch, -1)
    elif cin_f.shape[0] == batch:
        cin_b = cin_f
    else:
        raise ValueError(
            f"RtForcing.conc_in has {cin_f.shape[0]} rows but c_0 has {batch}"
        )

    # ---- cast network constants (id-keyed cache) ----
    # The sources are zone-owned frozen tensors whose identity is stable for
    # the life of the run, so ``_get_cast`` holds exactly one cast per
    # (source, dtype, device): each step pays a dict lookup instead of a
    # dispatch, and the cache stays bounded.  (The old per-step ``t.to(...)``
    # closure minted a fresh tensor every call, which grew the downstream
    # ``_REACTION_CONSTS_CACHE`` / ``_STOICH_CACHE`` by one entry per zone
    # step under mixed-dtype configurations.)
    monod_mat = _get_cast(monod.monod_mat, dtype, device)
    inhib_mat = _get_cast(monod.inhib_mat, dtype, device)
    tst_stoich = _get_cast(tst.stoich, dtype, device)
    tst_dep = _get_cast(tst.dep, dtype, device)
    min_eq_const = _get_cast(tst.min_eq_const, dtype, device)
    rate_const = _get_cast(misc.rate_const, dtype, device)
    mineral_molar_mass = _get_cast(misc.mineral_molar_mass, dtype, device)
    # Stoichiometry is a network constant.  ``network.mineral_stoichiometry``
    # returns a stable base tensor, so obtain its working (compute_dtype/
    # device) form from the id-keyed ``_get_stoich`` cache instead of a fresh
    # ``.to()``: the key ``id(base)`` is stable, so the cast is built once, this
    # step pays only a dict lookup, and the cache stays bounded.
    stoich = _get_stoich(network.mineral_stoichiometry, compute_dtype, device)
    if aux is None:
        ssa = _zero_tensor(num_min, dtype, device)
        aux_factor: Tensor | None = None
        has_aux = False
    else:
        ssa = _get_cast(aux.ssa, dtype, device)
        # The soil-water/temperature factor depends on the per-step hydro
        # fields (``s_w``, ``temp``), so it is computed fresh each step and
        # deliberately never cached.
        aux_factor = aux.factor(h.s_w, h.temp).to(dtype=dtype, device=device)
        has_aux = True

    # The fused solver runs entirely on ``compute_dtype``; these compute-cast
    # copies are id-keyed too (their sources are the external-dtype casts
    # above, which are id-stable: one entry per source per compute dtype).
    monod_mat_c = _get_cast(monod_mat, compute_dtype, device)
    inhib_mat_c = _get_cast(inhib_mat, compute_dtype, device)
    tst_stoich_c = _get_cast(tst_stoich, compute_dtype, device)
    tst_dep_c = _get_cast(tst_dep, compute_dtype, device)
    min_eq_const_c = _get_cast(min_eq_const, compute_dtype, device)
    rate_const_c = _get_cast(rate_const, compute_dtype, device)
    mineral_molar_mass_c = _get_cast(mineral_molar_mass, compute_dtype, device)
    ssa_c = _get_cast(ssa, compute_dtype, device)
    # Per-network invariants (finite masks, rate scale, zero fills) for
    # ``_reaction_terms``: built once per (network, compute dtype, device).
    consts = _get_reaction_consts(
        monod_mat=monod_mat_c,
        inhib_mat=inhib_mat_c,
        min_eq_const=min_eq_const_c,
        rate_const=rate_const_c,
        mineral_molar_mass=mineral_molar_mass_c,
        dtype=compute_dtype,
        device=device,
    )

    fused: FusedEval = _FusedEvaluator(
        _FusedEvalConstants(
            c_0=c0b.to(compute_dtype),
            dt=dtb.to(compute_dtype),
            v0=h.state.to(compute_dtype),
            depth=depth,
            q_in=q_in.to(compute_dtype),
            c_in=cin_b.to(compute_dtype),
            stoich=stoich,
            monod_mat=monod_mat_c,
            inhib_mat=inhib_mat_c,
            tst_stoich=tst_stoich_c,
            tst_dep=tst_dep_c,
            min_eq_const=min_eq_const_c,
            rate_const=rate_const_c,
            mineral_molar_mass=mineral_molar_mass_c,
            ssa=ssa_c,
            aux_factor=(
                aux_factor.to(compute_dtype) if aux_factor is not None else None
            ),
            has_aux=has_aux,
            consts=consts,
            num_aq=num_aq,
            num_min=num_min,
            num_spec=num_spec,
            min_start=min_start,
            do_reactions=do_reactions,
            compute_dtype=compute_dtype,
        )
    )
    return _FusedEvalContext(
        fused=fused,
        c0b=c0b,
        leading=leading,
        batch=batch,
        dtype=dtype,
        device=device,
        hydro=h,
        q_in=q_in,
        cin_b=cin_b,
        num_spec=num_spec,
        num_aq=num_aq,
        num_min=num_min,
        min_start=min_start,
        stoich=stoich,
        monod_mat=monod_mat,
        inhib_mat=inhib_mat,
        tst_stoich=tst_stoich,
        tst_dep=tst_dep,
        min_eq_const=min_eq_const,
        rate_const=rate_const,
        mineral_molar_mass=mineral_molar_mass,
        ssa=ssa,
        aux_factor=aux_factor,
        has_aux=has_aux,
        consts=consts,
    )


def _zone_step(
    zone: RtZone | RiverZone,
    c_0: Tensor,
    d: RtForcing,
    dt: Scalar,
    *,
    aux: MineralParameters | None,
    depth: float | Tensor,
    verbose: bool = False,
    compute_dtype: torch.dtype | None = None,
) -> tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor | None,
]:
    """Batched single-zone RT step, shared by ``RtZone`` and ``RiverZone``.

    ``zone`` supplies the network-derived parameters (monod / TST /
    equilibrium / misc), the per-zone geometry (``default_depth``,
    ``q_in_field``, the mineral-row start index ``_min_start``), the
    ``do_reactions`` / ``do_speciation`` flags, the name, and the speciation
    warm-start cache.  ``aux`` and ``depth`` are the *effective* values for
    this step: the zone's own in :meth:`RtZone.step` / :meth:`RiverZone.step`
    and per-lane overrides in :meth:`RtZone.step_with_params` (they are
    explicit rather than defaulting to the zone's, so an explicit
    ``aux=None`` still means "no kinetics" on a parameterised zone).

    Returns 9 outputs plus the (updated) speciation warm-start cache:
    ``(state, total_moles, mass_in, conc_in, lat_conc, vert_conc, lat_mass,
    vert_mass, mineral_rates, last_x_free)``.  All tiles the species dimension of
    ``c_0`` and restores its leading shape.
    """
    eq: EquilibriumParameters = zone.eq
    zone_name: str = zone.name
    min_start: int = zone._min_start()
    ctx: _FusedEvalContext = _build_fused_eval(
        c_0=c_0,
        d=d,
        dt=dt,
        network=zone.network,
        monod=zone.monod,
        tst=zone.tst,
        aux=aux,
        misc=zone.misc,
        depth=depth,
        min_start=min_start,
        q_in_field=zone.q_in_field,
        do_reactions=zone.do_reactions,
        compute_dtype=compute_dtype,
    )
    c0b: Tensor = ctx.c0b
    leading: tuple[int, ...] = ctx.leading
    batch: int = ctx.batch
    dtype: torch.dtype = ctx.dtype
    device: torch.device = ctx.device
    num_spec: int = ctx.num_spec
    num_min: int = ctx.num_min
    h: _HydroTensors = ctx.hydro
    cin_b: Tensor = ctx.cin_b
    fused: FusedEval = ctx.fused
    stoich: Tensor = ctx.stoich
    monod_mat: Tensor = ctx.monod_mat
    inhib_mat: Tensor = ctx.inhib_mat
    tst_stoich: Tensor = ctx.tst_stoich
    tst_dep: Tensor = ctx.tst_dep
    eq_const: Tensor = ctx.min_eq_const
    rate_const: Tensor = ctx.rate_const
    molar_mass: Tensor = ctx.mineral_molar_mass
    ssa: Tensor = ctx.ssa
    aux_factor: Tensor | None = ctx.aux_factor
    has_aux: bool = ctx.has_aux
    num_aq: int = ctx.num_aq

    # Mixed precision: use compute_dtype for RT solve, dtype for convergence
    compute_dtype = compute_dtype or dtype

    # Cast fused eval inputs to compute_dtype for RT solve
    if compute_dtype != dtype:
        c0b = c0b.to(compute_dtype)
        # The fused eval will be called with compute_dtype tensors
        # We need to ensure the fused eval works with compute_dtype

    c_after_rt: Tensor = zone._newton_solve(
        fused, c0b, verbose=verbose, compute_dtype=compute_dtype
    )

    # ---- speciation (warm-started from the previous step's x_free) ----
    new_cache: Tensor | None = zone._last_x_free
    if not zone.do_speciation or eq.num_free == 0:
        c_after_eq: Tensor = c_after_rt.to(dtype)
    else:
        n_free: int = eq.num_free
        if (
            zone._last_x_free is None
            or zone._last_x_free.ndim != 2
            or zone._last_x_free.shape[0] != batch
            or zone._last_x_free.shape[1] != n_free
        ):
            initial_x: Tensor = torch.zeros(batch, n_free, dtype=compute_dtype, device=device)
        else:
            initial_x = zone._last_x_free.to(dtype=compute_dtype, device=device)
        x_free: Tensor = eq.x_free_solve(
            c_after_rt.to(compute_dtype), initial_x, raise_on_failure=True, verbose=verbose
        )
        new_cache = x_free.detach().clone()
        c_after_eq = eq.conc_func(x_free.to(dtype))

    # ---- export bookkeeping ----
    tot_moles_after_eq: Tensor = get_tot_moles(c_after_eq, h.state, depth, num_aq)
    mass_in: Tensor = cin_b * h.q_in.unsqueeze(-1)

    def setmz(x: Tensor) -> Tensor:
        return torch.cat(
            [x[..., :num_aq], torch.zeros_like(x[..., num_aq:])], dim=-1
        )

    tot_moles_out_internal: Tensor = setmz(c_after_eq * h.q_internal.unsqueeze(-1))
    tot_moles_ext: Tensor = h.q_external.unsqueeze(-1) * cin_b
    tot_moles_out: Tensor = setmz(tot_moles_ext + tot_moles_out_internal)

    total_q_out: Tensor = h.q_internal + h.q_external  # (B,)
    frac_lat: Tensor = torch.where(
        total_q_out != 0.0,
        h.lat_flux_ext / total_q_out,
        torch.zeros_like(total_q_out),
    )
    frac_vert: Tensor = torch.where(
        total_q_out != 0.0, h.vert_flux_ext / total_q_out, torch.zeros_like(total_q_out)
    )
    mask: Tensor = total_q_out > 1e-6

    zero_conc_t: Tensor = torch.full_like(tot_moles_out, ZERO_CONC)
    lat_mass: Tensor = torch.where(mask.unsqueeze(-1), tot_moles_out * frac_lat.unsqueeze(-1), zero_conc_t)
    vert_mass: Tensor = torch.where(mask.unsqueeze(-1), tot_moles_out * frac_vert.unsqueeze(-1), zero_conc_t)

    lat_conc: Tensor = lat_mass.clone()
    vert_conc: Tensor = vert_mass.clone()
    # Hoisted from a per-zone-step ``torch.arange`` (Phase 3.3): the index range is
    # a loop-invariant (``num_aq`` is fixed per network), so it is cached per
    # (num_aq, device) instead of re-allocated on every step.
    aq_range = _aq_range(num_aq, device)
    q_lat: Tensor = h.lat_flux_ext
    q_vert: Tensor = h.vert_flux_ext
    lat_conc[:, aq_range] = torch.where(
        q_lat.abs().unsqueeze(-1) <= 1e-6,
        torch.full((batch, num_aq), ZERO_CONC, dtype=dtype, device=device),
        lat_mass[:, aq_range] / q_lat.unsqueeze(-1),
    )
    vert_conc[:, aq_range] = torch.where(
        q_vert.abs().unsqueeze(-1) <= 1e-6,
        torch.full((batch, num_aq), ZERO_CONC, dtype=dtype, device=device),
        vert_mass[:, aq_range] / q_vert.unsqueeze(-1),
    )

    if zone.do_reactions and has_aux:
        mineral_rates, _ = _reaction_rate_rust(
            c_after_eq,
            v0=h.state,
            depth=depth,
            num_aq=num_aq,
            num_min=num_min,
            num_spec=num_spec,
            min_start=min_start,
            stoich=stoich,
            monod_mat=monod_mat,
            inhib_mat=inhib_mat,
            tst_stoich=tst_stoich,
            tst_dep=tst_dep,
            min_eq_const=eq_const,
            rate_const=rate_const,
            mineral_molar_mass=molar_mass,
            ssa=ssa,
            aux_factor=aux_factor,
            has_aux=True,
            zone_name=zone_name,
            minerals_only=True,
        )
    else:
        mineral_rates = torch.zeros(batch, num_min, dtype=dtype, device=device)

    return (
        _restore(c_after_eq, leading),
        _restore(tot_moles_after_eq, leading),
        _restore(mass_in, leading),
        _restore(cin_b, leading),
        _restore(lat_conc, leading),
        _restore(vert_conc, leading),
        _restore(lat_mass, leading),
        _restore(vert_mass, leading),
        _restore(mineral_rates, leading),
        new_cache,
    )


# ---------------------------------------------------------------------- #
# RtZone
# ---------------------------------------------------------------------- #


@dataclass(eq=False)
class RtZone:
    """One soil (reactive-transport) zone (mirrors the Rust ``RtZone`` pyclass).

    Zones are dataclasses: the tunable constructor parameters are fields
    (``network``, ``parameters``, the two flags, the name) and the
    network-derived members (``monod``, ``tst``, ``eq``, ``aux``, ``misc``)
    plus the speciation warm-start cache are ``init=False`` fields filled in
    :meth:`__post_init__`.  ``eq=False`` keeps identity equality/hashing
    (zones are never value-compared).

    Attributes:
        network:       the reaction network.
        parameters:    physical + kinetic parameters (``RtParameters``).
        do_reactions:  whether mineral kinetics are active.
        do_speciation: whether equilibrium re-speciation runs after the solve.
        name:          zone name.
        monod:         per-mineral Monod/inhibition matrices.
        tst:           per-mineral TST matrices.
        eq:            equilibrium-speciation matrices.
        aux:           per-mineral auxiliary kinetic parameters (or None).
        misc:          static per-network arrays.
        _last_x_free:  speciation warm-start cache from the previous step.
    """

    network: ReactionNetwork
    parameters: RtParameters
    do_reactions: bool = True
    do_speciation: bool = True
    name: str = "unnamed"
    monod: MonodParameters = field(init=False)
    tst: TstParameters = field(init=False)
    eq: EquilibriumParameters = field(init=False)
    aux: MineralParameters | None = field(init=False)
    misc: MiscData = field(init=False)
    _last_x_free: Tensor | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.do_reactions = bool(self.do_reactions)
        self.do_speciation = bool(self.do_speciation)
        self.monod: MonodParameters = self.network.monod_params
        self.tst: TstParameters = self.network.tst_params
        self.eq: EquilibriumParameters = self.network.equilibrium_parameters
        self.aux: MineralParameters | None = self.parameters.mineral_params
        self.misc: MiscData = MiscData(
            mineral_stoichiometry=self.network.mineral_stoichiometry,
            species_mobility=self.network.transport_mask,
            mineral_molar_mass=self.network.mineral_molar_masses,
            rate_const=self.network.rate_consts,
        )

    @property
    def dimensions(self) -> ZoneDimensions:
        """ZoneDimensions (porosity, depth, passive water storage)."""
        return self.parameters.dimensions

    @property
    def default_depth(self) -> float | Tensor:
        """Default zone depth [mm] (``parameters.dimensions.depth``)."""
        return self.parameters.dimensions.depth

    @property
    def q_in_field(self) -> str:
        """Hydro forcing field that drives this zone's transport (``q_in``)."""
        return "q_in"

    @property
    def all_species(self) -> list[str]:
        """All species names in transportation order."""
        return self.network.species_names

    @property
    def mineral_species(self) -> list[str]:
        """Mineral species names."""
        return self.network.mineral_species_names

    def _num_aq(self) -> int:
        return self.network.num_aqueous_species

    def _min_start(self) -> int:
        """Mineral-row start index: soil minerals sit after the aqueous rows."""
        return self._num_aq()

    def build_fused_eval(
        self,
        d: RtForcing,
        c_0: Tensor,
        dt_days: Scalar,
    ) -> tuple[FusedEval, Tensor]:
        """Build the fused kinetic ``(residual, Jacobian)`` closure for one step.

        Exposes the exact callable :meth:`step` hands to the Newton-to-LM solver,
        together with the flattened initial iterate, so callers (e.g. solver
        microbenchmarks) can time the kinetic solve in isolation without
        duplicating the constant casting / hydro normalisation done in
        :func:`_zone_step`.

        Args:
            d:       reactive-transport forcing for this step/zone.
            c_0:     ``(num_spec,)`` or ``(B, num_spec)`` initial concentrations.
            dt_days: time step length [days].

        Returns:
            ``(fused, c0b)`` where ``fused(conc) -> (residual, Jacobian)`` is the
            batched fused evaluation and ``c0b`` is the ``(B, num_spec)`` initial
            iterate.  Both may be passed straight to
            :func:`potions.solvers.newton_fused_then_lm`.
        """
        ctx: _FusedEvalContext = _build_fused_eval(
            c_0=c_0,
            d=d,
            dt=dt_days,
            network=self.network,
            monod=self.monod,
            tst=self.tst,
            aux=self.aux,
            misc=self.misc,
            depth=self.dimensions.depth,
            min_start=self._num_aq(),
            q_in_field="q_in",
            do_reactions=self.do_reactions,
        )
        return ctx.fused, ctx.c0b

    # ---- diagnostics (mirror the Rust pymethods; 1-D in, 1-D out) ----

    def monod_rate(self, chms: Tensor) -> Tensor:
        """Per-mineral Monod rate at ``chms`` (Rust ``monod_rate``)."""
        b, leading = _flatten_batch(chms)
        return _restore(self.monod.rate(b), leading)

    def tst_rate(self, chms: Tensor) -> Tensor:
        """Per-mineral TST rate at ``chms`` (Rust ``tst_rate``)."""
        b, leading = _flatten_batch(chms)
        return _restore(self.tst.rate(b), leading)

    def aux_factor(self, d: RtForcing) -> Tensor:
        """Combined soil-water/temperature auxiliary factor (Rust ``aux_factor``)."""
        if self.aux is None:
            raise ValueError("Tried to call `aux_factor` without kinetic parameters")
        s_w: Scalar = d.s_w
        temp: Scalar = d.hydro_forc.temp if d.hydro_forc is not None else 20.0
        return self.aux.factor(s_w, temp).squeeze()

    def get_tot_moles(self, chms: Tensor, d: RtForcing) -> Tensor:
        """Concentration vector to total-moles storage (Rust ``get_tot_moles``)."""
        b, leading = _flatten_batch(chms)
        h = _normalise_hydro(d, b.shape[0], b.dtype, b.device)
        out = get_tot_moles(b, h.state, self.dimensions.depth, self._num_aq())
        return _restore(out, leading)
    
    def to_array(self) -> Tensor:
        """Serialize the zone parameters into a flat parameter array."""
        return self.parameters.to_array()

    @classmethod
    def from_array(
        cls,
        arr: Tensor,
        network: ReactionNetwork,
        do_reactions: bool,
        do_speciation: bool,
        name: str,
        natural_scales: bool = True,
    ) -> RtZone:
        """Build an :class:`RtZone` from a flat parameter array.

        Args:
            arr:            ``(3 + 4*num_min,)`` packed parameter array.
            network:        the reaction network for the zone.
            do_reactions:   enable kinetic reactions.
            do_speciation:  enable equilibrium speciation.
            name:           zone name.
            natural_scales: unpack ``ssa`` from log10 scale. Defaults to True.

        Returns:
            A new, fully parameterized :class:`RtZone`.
        """
        params: RtParameters = RtParameters.from_array(arr, natural_scales=natural_scales)
        return cls(
            network=network,
            parameters=params,
            do_reactions=do_reactions,
            do_speciation=do_speciation,
            name=name,
        )

    def moles_to_conc(self, tot_moles: Tensor, d: RtForcing) -> Tensor:
        """Total-moles storage to concentration (Rust ``moles_to_conc``)."""
        b, leading = _flatten_batch(tot_moles)
        h = _normalise_hydro(d, b.shape[0], b.dtype, b.device)
        out = moles_to_conc(b, h.state, self.dimensions.depth, self._num_aq())
        return _restore(out, leading)

    def transport_rate(self, chms: Tensor, d: RtForcing) -> Tensor:
        """Transport contribution ``q_in/v0 * (c_in - chms)`` on aqueous rows.

        Uses ``d.hydro_step.q_in`` for ``RtZone`` (Rust ``transport_rate_rust``);
        when ``|state| < 1e-6`` the whole vector is the zero-concentration sentinel.
        """
        b, leading = _flatten_batch(chms)
        h = _normalise_hydro(d, b.shape[0], b.dtype, b.device)
        cin_b = _flatten_batch(d.conc_in.to(dtype=b.dtype, device=b.device))[0]
        f: Tensor = torch.where(
            h.state.abs() < 1e-6,
            torch.tensor(ZERO_CONC, dtype=b.dtype, device=b.device),
            h.q_in / h.state,
        )
        out: Tensor = f.unsqueeze(-1) * (cin_b - b)
        out[:, self._num_aq() :] = 0.0
        return _restore(out, leading)

    def reaction_rate(
        self, chms: Tensor, d: RtForcing, minerals_only: bool = False
    ) -> Tensor:
        """Kinetic reaction rate (Rust ``reaction_rate``).

        Includes the double-conversion quirk: the input is divided by ``state``
        again for aqueous rows inside the rate evaluation.
        """
        if not self.do_reactions:
            raise ValueError(
                f"Trying to calculate reaction rate on zone '{self.name}'"
                " without kinetic properties"
            )
        b, leading = _flatten_batch(chms)
        h = _normalise_hydro(d, b.shape[0], b.dtype, b.device)
        assert self.aux is not None, "do_reactions requires aux parameters"
        ssa = self.aux.ssa.to(dtype=b.dtype, device=b.device)
        aux_factor = self.aux.factor(h.s_w, h.temp).to(dtype=b.dtype, device=b.device)
        num_min: int = self.network.num_minerals
        mineral_rates, all_species = _reaction_rate_rust(
            b,
            v0=h.state,
            depth=self.dimensions.depth,
            num_aq=self._num_aq(),
            num_min=num_min,
            num_spec=self.network.num_species,
            min_start=self._num_aq(),
            stoich=self.network.mineral_stoichiometry,
            monod_mat=self.monod.monod_mat,
            inhib_mat=self.monod.inhib_mat,
            tst_stoich=self.tst.stoich,
            tst_dep=self.tst.dep,
            min_eq_const=self.tst.min_eq_const,
            rate_const=self.misc.rate_const,
            mineral_molar_mass=self.misc.mineral_molar_mass,
            ssa=ssa,
            aux_factor=aux_factor,
            has_aux=True,
            zone_name=self.name,
            minerals_only=minerals_only,
        )
        if minerals_only:
            return _restore(mineral_rates, leading)
        assert all_species is not None
        return _restore(all_species, leading)

    def mass_balance_ode(self, chms: Tensor, d: RtForcing) -> Tensor:
        """Transport + reaction ODE right-hand side (Rust ``mass_balance_ode``)."""
        transport: Tensor = self.transport_rate(chms, d)
        if not self.do_reactions:
            return transport
        return transport + self.reaction_rate(chms, d, minerals_only=False)

    def residual_function(
        self, c_0: Tensor, conc: Tensor, d: RtForcing, dt_days: Scalar
    ) -> Tensor:
        """Residual ``(c_0 - conc) + dt * mass_balance_ode(conc)`` (Rust API)."""
        b0, leading = _flatten_batch(c_0)
        cb, _ = _flatten_batch(conc)
        dts: Tensor = _coerce_dt(dt_days, b0.dtype, b0.device)
        masstx: Tensor = self.mass_balance_ode(cb, d)
        out: Tensor = (b0 - cb) + dts * masstx
        return _restore(out, leading)

    def jacobian_residual_function(
        self, c_0: Tensor, conc: Tensor, d: RtForcing, dt_days: Scalar
    ) -> Tensor:
        """Analytic Jacobian ``-I + dt * dODE/dconc`` (Rust API)."""
        cb, leading = _flatten_batch(conc)
        h = _normalise_hydro(d, cb.shape[0], cb.dtype, cb.device)
        dts: Tensor = _coerce_dt(dt_days, cb.dtype, cb.device)
        # ``_get_cast`` (id-keyed) instead of a per-call ``t.to(...)`` closure:
        # the sources are zone-owned frozen tensors with stable identity, so
        # each cast is built once per (source, dtype, device) and the
        # downstream caches (``_REACTION_CONSTS_CACHE`` / ``_STOICH_CACHE``)
        # stay bounded even when this is called repeatedly with a varying
        # dtype.
        if self.aux is None:
            ssa = _zero_tensor(self.network.num_minerals, cb.dtype, cb.device)
            aux_factor: Tensor | None = torch.zeros(
                cb.shape[0], self.network.num_minerals, dtype=cb.dtype, device=cb.device
            )
        else:
            ssa = _get_cast(self.aux.ssa, cb.dtype, cb.device)
            aux_factor = self.aux.factor(h.s_w, h.temp).to(dtype=cb.dtype, device=cb.device)
        terms = _reaction_terms(
            conc=cb,
            v0=h.state,
            depth=self.dimensions.depth,
            monod_mat=_get_cast(self.monod.monod_mat, cb.dtype, cb.device),
            inhib_mat=_get_cast(self.monod.inhib_mat, cb.dtype, cb.device),
            tst_stoich=_get_cast(self.tst.stoich, cb.dtype, cb.device),
            tst_dep=_get_cast(self.tst.dep, cb.dtype, cb.device),
            min_eq_const=_get_cast(self.tst.min_eq_const, cb.dtype, cb.device),
            rate_const=_get_cast(self.misc.rate_const, cb.dtype, cb.device),
            mineral_molar_mass=_get_cast(self.misc.mineral_molar_mass, cb.dtype, cb.device),
            ssa=ssa,
            aux_factor=aux_factor,
            has_aux=self.aux is not None,
            num_aq=self._num_aq(),
            num_min=self.network.num_minerals,
            num_spec=self.network.num_species,
            do_reactions=self.do_reactions,
        )
        j_ode: Tensor = _assemble_ode_jacobian(
            terms,
            _get_stoich(self.network.mineral_stoichiometry, cb.dtype, cb.device),
            num_aq=self._num_aq(),
            num_spec=self.network.num_species,
            depth=self.dimensions.depth,
            v0=h.state,
            q_in=h.q_in,
            do_reactions=self.do_reactions,
        )
        eye: Tensor = torch.eye(self.network.num_species, dtype=cb.dtype, device=cb.device)
        out = -eye.unsqueeze(0) + dts * j_ode
        if leading == ():
            return out.squeeze(0)
        return out.reshape(*leading, self.network.num_species, self.network.num_species)

    def _newton_solve(
        self,
        fused: FusedEval,
        c0b: Tensor,
        *,
        verbose: bool = False,
        compute_dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Newton-to-LM solve of the zone's implicit step (Rust ``step`` core).

        Wraps :func:`~potions.solvers.newton_fused_then_lm` with
        ``raise_on_failure=True``: a batch lane that fails to converge
        (Newton exhausted and LM stalled) is a hard error, matching the
        fail-loud scalar reference.  The shared :func:`_zone_step` driver
        calls this so each zone class keeps its own explicit solve entry
        point.
        """
        return newton_fused_then_lm(
            fused,
            c0b,
            raise_on_failure=True,
            verbose=verbose,
            compute_dtype=compute_dtype,
        )

    def step(
        self,
        c_0: Tensor,
        d: RtForcing,
        dt_days: Scalar,
        verbose: bool = False,
        compute_dtype: torch.dtype | None = None,
    ) -> RtStep:
        """Advance the zone one time step (Rust ``step`` pymethod).

        Args:
            c_0:     (num_spec,) or (B, num_spec) initial concentrations.
            d:       reactive-transport forcing for this step/zone.
            dt_days: time step length [days].
            verbose: print solver progress.
            compute_dtype: Optional dtype for the RT solve internals (e.g.,
                float32 compute with float64 convergence checks).

        Returns:
            An :class:`RtStep` with the leading dims of ``c_0`` preserved.
        """
        c_after_eq, total_moles, mass_in, cin_b, lat_conc, vert_conc, lat_mass, vert_mass, mineral_rates, new_cache = _zone_step(
            self,
            c_0=c_0,
            d=d,
            dt=dt_days,
            aux=self.aux,
            depth=self.default_depth,
            verbose=verbose,
            compute_dtype=compute_dtype,
        )
        self._last_x_free = new_cache
        return RtStep(
            state=c_after_eq,
            total_moles=total_moles,
            conc_in=cin_b,
            mass_in=mass_in,
            lat_conc=lat_conc,
            vert_conc=vert_conc,
            lat_mass=lat_mass,
            vert_mass=vert_mass,
            mineral_rates=mineral_rates,
        )

    def step_with_params(
        self,
        c_0: Tensor,
        d: RtForcing,
        dt_days: Scalar,
        aux: MineralParameters | None,
        depth: float | Tensor,
        verbose: bool = False,
        compute_dtype: torch.dtype | None = None,
    ) -> RtStep:
        """Advance the zone one step with *per-lane* parameters.

        Identical to :meth:`step` but the mineral auxiliary parameters
        ``aux`` and the zone ``depth`` are supplied explicitly, so one zone
        object can drive a batch of independent parameter sets in a single
        batched call (each lane uses its own depth / soil-water-specific
        surface area).  The shared network (monod / TST / equilibrium / misc)
        and the per-zone, cross-lane speciation warm-start are reused from the
        instance.

        Args:
            c_0:     (num_spec,) or (batch_size, num_spec) initial
                concentrations.
            d:       reactive-transport forcing for this step/zone (fields may
                be ``(batch_size,)`` / ``(batch_size, num_spec)`` tensors for
                batched lanes).
            dt_days: time step length [days].
            aux:     per-lane mineral auxiliary parameters
                (``(batch_size, num_min)`` fields) or ``None`` when reactions
                are disabled for the zone.
            depth:   per-lane zone depth [mm] (scalar or ``(batch_size,)``
                tensor).
            verbose: print solver progress.
            compute_dtype: Optional dtype for the RT solve internals (e.g.,
                float32 compute with float64 convergence checks).

        Returns:
            An :class:`RtStep` with the leading dims of ``c_0`` preserved.
        """
        c_after_eq, total_moles, mass_in, cin_b, lat_conc, vert_conc, lat_mass, vert_mass, mineral_rates, new_cache = _zone_step(
            self,
            c_0=c_0,
            d=d,
            dt=dt_days,
            aux=aux,
            depth=depth,
            verbose=verbose,
            compute_dtype=compute_dtype,
        )
        self._last_x_free = new_cache
        return RtStep(
            state=c_after_eq,
            total_moles=total_moles,
            conc_in=cin_b,
            mass_in=mass_in,
            lat_conc=lat_conc,
            vert_conc=vert_conc,
            lat_mass=lat_mass,
            vert_mass=vert_mass,
            mineral_rates=mineral_rates,
        )


# ---------------------------------------------------------------------- #
# RiverZone
# ---------------------------------------------------------------------- #


@dataclass(eq=False)
class RiverZone:
    """One river (reactive-transport) zone (mirrors the Rust ``RiverZone`` pyclass).

    Same stepping machinery as :class:`RtZone` but the transport is driven by
    the internal water flux ``q_internal()`` and the mineral rows sit at the
    *end* of the species vector (``num_species - num_minerals``).  The ``num_aq + i``
    hardcode in the kinetic derivative is kept verbatim for reference parity.

    Zones are dataclasses: the tunable constructor parameters are fields
    (``network``, ``parameters``, the two flags, the name) and the
    network-derived members (``monod``, ``tst``, ``eq``, ``aux``, ``misc``)
    plus the speciation warm-start cache are ``init=False`` fields filled in
    :meth:`__post_init__`.  ``eq=False`` keeps identity equality/hashing
    (zones are never value-compared).

    Attributes:
        network:       the reaction network.
        parameters:    physical + kinetic parameters (``RiverParameters``).
        do_reactions:  whether mineral kinetics are active.
        do_speciation: whether equilibrium re-speciation runs after the solve.
        name:          zone name (default ``"river"``).
        monod:         per-mineral Monod/inhibition matrices.
        tst:           per-mineral TST matrices.
        eq:            equilibrium-speciation matrices.
        aux:           per-mineral auxiliary kinetic parameters (or None).
        misc:          static per-network arrays.
        _last_x_free:  speciation warm-start cache from the previous step.
    """

    network: ReactionNetwork
    parameters: RiverParameters
    do_reactions: bool = True
    do_speciation: bool = True
    name: str = "river"
    monod: MonodParameters = field(init=False)
    tst: TstParameters = field(init=False)
    eq: EquilibriumParameters = field(init=False)
    aux: MineralParameters | None = field(init=False)
    misc: MiscData = field(init=False)
    _last_x_free: Tensor | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.do_reactions = bool(self.do_reactions)
        self.do_speciation = bool(self.do_speciation)
        self.monod: MonodParameters = self.network.monod_params
        self.tst: TstParameters = self.network.tst_params
        self.eq: EquilibriumParameters = self.network.equilibrium_parameters
        self.aux: MineralParameters | None = self.parameters.mineral_params
        self.misc: MiscData = MiscData(
            mineral_stoichiometry=self.network.mineral_stoichiometry,
            species_mobility=self.network.transport_mask,
            mineral_molar_mass=self.network.mineral_molar_masses,
            rate_const=self.network.rate_consts,
        )

    @property
    def dimensions(self) -> RiverDimensions:
        """RiverDimensions (bed depth, passive water storage)."""
        return self.parameters.dimensions

    @property
    def default_depth(self) -> float:
        """Default river-bed depth [mm] (``parameters.dimensions.bed_depth``)."""
        return self.parameters.dimensions.bed_depth

    @property
    def q_in_field(self) -> str:
        """Hydro forcing field that drives this zone's transport (``q_internal``)."""
        return "q_internal"

    @property
    def all_species(self) -> list[str]:
        """All species names in transportation order."""
        return self.network.species_names

    @property
    def mineral_species(self) -> list[str]:
        """Mineral species names."""
        return self.network.mineral_species_names

    # ---- diagnostics (mirror the Rust pymethods; 1-D in, 1-D out) ----

    def monod_rate(self, chms: Tensor) -> Tensor:
        """Per-mineral Monod rate at ``chms`` (Rust ``monod_rate``)."""
        b, leading = _flatten_batch(chms)
        return _restore(self.monod.rate(b), leading)

    def tst_rate(self, chms: Tensor) -> Tensor:
        """Per-mineral TST rate at ``chms`` (Rust ``tst_rate``)."""
        b, leading = _flatten_batch(chms)
        return _restore(self.tst.rate(b), leading)

    def aux_factor(self, d: RtForcing) -> Tensor:
        """Combined soil-water/temperature auxiliary factor (Rust ``aux_factor``)."""
        if self.aux is None:
            raise ValueError("Tried to call `aux_factor` without kinetic parameters")
        s_w: Scalar = d.s_w
        temp: Scalar = d.hydro_forc.temp if d.hydro_forc is not None else 20.0
        return self.aux.factor(s_w, temp).squeeze()

    def get_tot_moles(self, chms: Tensor, d: RtForcing) -> Tensor:
        """Concentration vector to total-moles storage (Rust ``get_tot_moles``)."""
        b, leading = _flatten_batch(chms)
        h = _normalise_hydro(d, b.shape[0], b.dtype, b.device)
        out = get_tot_moles(b, h.state, self.dimensions.bed_depth, self.network.num_aqueous_species)
        return _restore(out, leading)

    def to_array(self) -> Tensor:
        """Serialize the zone parameters into a flat parameter array."""
        return self.parameters.to_array()

    @classmethod
    def from_array(
        cls,
        arr: Tensor,
        network: ReactionNetwork,
        do_reactions: bool,
        do_speciation: bool,
        name: str,
        natural_scales: bool = True,
    ) -> RiverZone:
        """Build a :class:`RiverZone` from a flat parameter array.

        Args:
            arr:            ``(2 + 4*num_min,)`` packed parameter array.
            network:        the reaction network for the zone.
            do_reactions:   enable kinetic reactions.
            do_speciation:  enable equilibrium speciation.
            name:           zone name.
            natural_scales: unpack ``ssa`` from log10 scale. Defaults to True.

        Returns:
            A new, fully parameterized :class:`RiverZone`.
        """
        params: RiverParameters = RiverParameters.from_array(arr, natural_scales=natural_scales)
        return cls(
            network=network,
            parameters=params,
            do_reactions=do_reactions,
            do_speciation=do_speciation,
            name=name,
        )

    def moles_to_conc(self, tot_moles: Tensor, d: RtForcing) -> Tensor:
        """Total-moles storage to concentration (Rust ``moles_to_conc``)."""
        b, leading = _flatten_batch(tot_moles)
        h = _normalise_hydro(d, b.shape[0], b.dtype, b.device)
        out = moles_to_conc(b, h.state, self.dimensions.bed_depth, self.network.num_aqueous_species)
        return _restore(out, leading)

    def transport_rate(self, chms: Tensor, d: RtForcing) -> Tensor:
        """Transport contribution driven by ``d.hydro_step.q_internal`` (Rust API)."""
        b, leading = _flatten_batch(chms)
        h = _normalise_hydro(d, b.shape[0], b.dtype, b.device)
        cin_b = _flatten_batch(d.conc_in.to(dtype=b.dtype, device=b.device))[0]
        f: Tensor = torch.where(
            h.state.abs() < 1e-6,
            torch.tensor(ZERO_CONC, dtype=b.dtype, device=b.device),
            h.q_internal / h.state,
        )
        out: Tensor = f.unsqueeze(-1) * (cin_b - b)
        out[:, self.network.num_aqueous_species :] = 0.0
        return _restore(out, leading)

    def reaction_rate(
        self, chms: Tensor, d: RtForcing, minerals_only: bool = False
    ) -> Tensor:
        """Kinetic reaction rate (Rust ``reaction_rate``; double-conversion quirk)."""
        if not self.do_reactions:
            raise ValueError(
                f"Trying to calculate reaction rate on zone '{self.name}'"
                " without kinetic properties"
            )
        b, leading = _flatten_batch(chms)
        h = _normalise_hydro(d, b.shape[0], b.dtype, b.device)
        assert self.aux is not None, "do_reactions requires aux parameters"
        ssa = self.aux.ssa.to(dtype=b.dtype, device=b.device)
        aux_factor = self.aux.factor(h.s_w, h.temp).to(dtype=b.dtype, device=b.device)
        num_min: int = self.network.num_minerals
        mineral_rates, all_species = _reaction_rate_rust(
            b,
            v0=h.state,
            depth=self.dimensions.bed_depth,
            num_aq=self.network.num_aqueous_species,
            num_min=num_min,
            num_spec=self.network.num_species,
            min_start=self._min_start(),
            stoich=self.network.mineral_stoichiometry,
            monod_mat=self.monod.monod_mat,
            inhib_mat=self.monod.inhib_mat,
            tst_stoich=self.tst.stoich,
            tst_dep=self.tst.dep,
            min_eq_const=self.tst.min_eq_const,
            rate_const=self.misc.rate_const,
            mineral_molar_mass=self.misc.mineral_molar_mass,
            ssa=ssa,
            aux_factor=aux_factor,
            has_aux=True,
            zone_name=self.name,
            minerals_only=minerals_only,
        )
        if minerals_only:
            return _restore(mineral_rates, leading)
        assert all_species is not None
        return _restore(all_species, leading)

    def mass_balance_ode(self, chms: Tensor, d: RtForcing) -> Tensor:
        """Transport + reaction ODE right-hand side (Rust ``mass_balance_ode``)."""
        transport: Tensor = self.transport_rate(chms, d)
        if not self.do_reactions:
            return transport
        return transport + self.reaction_rate(chms, d, minerals_only=False)

    def _newton_solve(
        self,
        fused: FusedEval,
        c0b: Tensor,
        *,
        verbose: bool = False,
        compute_dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Newton-to-LM solve of the zone's implicit step (Rust ``step`` core).

        Wraps :func:`~potions.solvers.newton_fused_then_lm` with
        ``raise_on_failure=True``: a batch lane that fails to converge
        (Newton exhausted and LM stalled) is a hard error, matching the
        fail-loud scalar reference.  The shared :func:`_zone_step` driver
        calls this so each zone class keeps its own explicit solve entry
        point.
        """
        return newton_fused_then_lm(
            fused,
            c0b,
            raise_on_failure=True,
            verbose=verbose,
            compute_dtype=compute_dtype,
        )

    def step(
        self,
        c_0: Tensor,
        d: RtForcing,
        dt_days: Scalar,
        verbose: bool = False,
        compute_dtype: torch.dtype | None = None,
    ) -> RtStep:
        """Advance the river zone one time step (Rust ``step`` pymethod).

        Args:
            c_0:     (num_spec,) or (B, num_spec) initial concentrations.
            d:       reactive-transport forcing for this step/zone.
            dt_days: time step length [days].
            verbose: print solver progress.
            compute_dtype: Optional dtype for the RT solve internals (e.g.,
                float32 compute with float64 convergence checks).

        Returns:
            An :class:`RtStep` with the leading dims of ``c_0`` preserved.
        """
        c_after_eq, total_moles, mass_in, cin_b, lat_conc, vert_conc, lat_mass, vert_mass, mineral_rates, new_cache = _zone_step(
            self,
            c_0=c_0,
            d=d,
            dt=dt_days,
            aux=self.aux,
            depth=self.default_depth,
            verbose=verbose,
            compute_dtype=compute_dtype,
        )
        self._last_x_free = new_cache
        return RtStep(
            state=c_after_eq,
            total_moles=total_moles,
            conc_in=cin_b,
            mass_in=mass_in,
            lat_conc=lat_conc,
            vert_conc=vert_conc,
            lat_mass=lat_mass,
            vert_mass=vert_mass,
            mineral_rates=mineral_rates,
        )

    def _min_start(self) -> int:
        """Mineral-row start index: river minerals sit at the *end* of the
        species vector (``num_species - num_minerals``)."""
        return self.network.num_species - self.network.num_minerals