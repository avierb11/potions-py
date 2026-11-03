"""Solver-only microbenchmarks for the batched roots in ``potions.solvers``.

``profile_batched.py`` measures the whole forward pass and ``bench_zone.py``
measures one *zone step* (solver + surrounding machinery).  This script goes one
level deeper and times the **bare numerical solvers** on realistic model data,
so an algorithm change can be attributed without confounders:

- ``rt``:          the fused kinetic Newton-to-LM solve on the 4-species East
                   River network.  The residual/Jacobian closure comes from the
                   public :meth:`potions.rt_zone.RtZone.build_fused_eval`, then
                   ``newton_fused_then_lm`` is timed directly.
- ``rt-batch``:    the same fused solve with a *different parameter set per
                   instance* (each zone's mineral kinetic block drawn around the
                   real zone values) under one shared forcing — the
                   ensemble/calibration pattern.  Exercises the
                   ``RtParameters.from_batched_array`` layout the batched model
                   uses.
- ``speciation``:  ``EquilibriumParameters.x_free_solve`` (Newton with analytic
                   Jacobian, then LM), warm-started from a neighbouring
                   composition (not the exact root).
- ``hydro-seq``:   sequential time-stepping of a single ``SurfaceZone`` over real
                   per-day forcing — the model forward-pass pattern.  One root
                   ``_newton_storage_solve`` per day, each step warm-started
                   from the previous.  The instance count is the number of days
                   solved, capped by the length of the real forcing record.
- ``hydro-batch``: many independent zones (a different ``SurfaceZone`` parameter
                   set per instance) under one shared forcing, solved in a single
                   vectorised ``step`` — the high-throughput
                   ensemble/calibration pattern (e.g. 1M zones in ~2 ms on GPU).
- ``hydro``:       legacy probe: one real ``SurfaceZone``, *many simultaneous
                   forcings/initial states* on a single parameter set, retained
                   for comparison.
- ``synthetic``:   an isolated **linear** system ``f(X) = J X - b`` with a random
                   SPD ``J`` of tunable dimension / condition number.  Pure
                   linear solves make Newton + Armijo converge in one step and
                   never reach LM, so this isolates the batched LU solve
                   (``lu_factor`` / ``lu_solve``) cost from the model.

Independent of the probe, each cell solves the requested number of **instances**
(the model runs / parameter sets / day steps that are solved in parallel).  The
default sweep is powers of ten up to one million so the total wall time can be
compared directly across sizes (``--instances 1,10,100,1000,...``).

Governing equations (all mirrored from ``math.rs`` by ``solvers.py``):

- Newton: solve ``J(x) s = f(x)`` (LU with partial pivoting), then Armijo
  backtracking ``x <- x - alpha s`` accepting the largest ``alpha`` with
  ``||f(x - alpha s)||^2 <= (1 - c alpha) ||f(x)||^2``, ``c = 1e-4``.
- LM fallback: damped normal equations with adaptive ``lambda``; instances
  whose Newton pass stalled/NaN'd resume from Newton's best iterate.
- Hydro storage: damped automatic-differentiation Newton for the
  implicit-Euler mass balance (autograd derivative of the residual,
  sign-preserving derivative floor, storage clamped non-negative, plus an
  Armijo-style backtracking line search on |f| — required because the
  kinked ``k*s**alpha`` baseflow laws make the undamped step 2-cycle).

Probes are built to be genuinely **off-solution** so the measured cost reflects a
real solve rather than a converged no-op:

- the ``rt`` / ``speciation`` zones are chosen to have ``do_reactions`` and
  ``do_speciation`` enabled (never a snow zone), and each instance starts from a
  spun-up state driven by a *different* forcing with jittered inflow/chemistry;
- ``hydro`` uses real East River precipitation/temperature/PET for the probe
  zone, seeded from a spin-up storage trajectory, one real day per instance.

Each cell reports the **total** wall time of the whole call (one call solves all
``n`` instances at once), the per-instance time (``total / n``), the number of
fused evaluations per solve, the final component-wise max absolute residual, and
the percentage of instances that fall through to Levenberg-Marquardt.  A cell
whose ``rt`` solve converges at the initial guess (``evals <= 1``) is flagged
``[TRIVIAL]`` so a meaningless run is never mistaken for a result.

Examples::

    python scripts/bench_solvers.py                # rt + speciation + hydro-seq + hydro-batch
    python scripts/bench_solvers.py --instances 1,10,100,1000 --solves 100
    python scripts/bench_solvers.py --device cpu --dtype float64
    python scripts/bench_solvers.py --synthetic --dim 32 --cond 1e6
    python scripts/bench_solvers.py --zone shallow_hs     # probe another zone
    python scripts/bench_solvers.py --perturb 5           # push starts further off-solution
    python scripts/bench_solvers.py --solvers rt --stress 16   # widen the rt network
    python scripts/bench_solvers.py --solvers rt-batch --instances 1024   # many RT parameter sets
    python scripts/bench_solvers.py --solvers hydro --instances 1,1024    # legacy: one zone, many forcings

The 4-species East River network is arithmetic-trivial, so ``rt`` timings are
flat in the instance count (dispatch-bound).  ``--stress S`` builds an exact
block-diagonal copy of the network with ``S * num_species`` unknowns, which lets
the same timing harness reach the arithmetic-bound regime where cost grows with
the unknown count and the instance count.  ``--stress 1`` (the default) runs the
real network unchanged.
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
import time
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any, Callable, NamedTuple, cast

import numpy as np
import torch
from pandas import DataFrame, DatetimeIndex

SCRIPTS: str = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)

import east_river_forward as fwd  # noqa: E402  (registers showcase paths)
from utils.parameter_ranges import load_input_data  # noqa: E402
from data.raw.east_river import hbv_data, load_all_doc_data  # noqa: E402

import potions as pt  # noqa: E402  # type: ignore[import-not-found]
from potions.common_types import HydroForcing, HydroStep, MiscData, RtForcing  # noqa: E402
from potions.device import default_dtype  # noqa: E402
from potions.equilibrium import EquilibriumParameters  # noqa: E402
from potions.hydro import (  # noqa: E402
    HydrologicZone,
    _hydro_residual,
    _newton_storage_solve,
)
from potions.kinetic_structures import (  # noqa: E402
    MineralParameters,
    MonodParameters,
    PARAMETERS_PER_MINERAL,
    RtParameters,
    TstParameters,
)
from potions.model import HydrologicalModel  # noqa: E402
from potions.model_batched import (  # noqa: E402  # type: ignore[import-not-found]
    BatchedConfig,
    Geometry,
    _build_hydro_zones,
    _forcing_source_tensors,
    _hydro_zone_blocks,
    _per_lane_forcing,
    _rt_blocks,
)
from potions.reaction_network import ReactionNetwork  # noqa: E402
from potions.rt_zone import RtZone, _build_fused_eval  # noqa: E402
from potions.solvers import (  # noqa: E402
    FusedEval,
    _FAILED,
    _newton_fused_core,
    newton_analytic_then_lm,
    newton_fused_then_lm,
)

Tensor = torch.Tensor
DEFAULT_INSTANCES: tuple[int, ...] = (1, 10, 100, 1_000, 10_000, 100_000, 1_000_000)
DEFAULT_SOLVERS: tuple[str, ...] = ("rt", "speciation", "hydro-seq", "hydro-batch")


class SolveSpec(NamedTuple):
    """A single solver probe.

    Attributes:
        solve:    zero-argument timed call reaching the solver once on the full
                  batch (returns the root, which the caller may inspect).
        diagnose: one-shot instrumented call returning
                  ``(evals_per_solve, max_abs_residual, pct_lm)``; any
                  unavailable metric is ``nan``.
    """

    solve: Callable[[], Tensor]
    diagnose: Callable[[], tuple[float, float, float]]


class Row(NamedTuple):
    """One printed result row."""

    solver: str
    instances: int
    device: str
    dtype: str
    median_us: float
    evals: float
    max_res: float
    pct_lm: float
    stress: int


def _sync(device: torch.device) -> None:
    """Block on the device's work queue (no-op on CPU)."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def _timed(
    solve: Callable[[], object],
    device: torch.device,
    solves: int,
    warmup: int,
) -> float:
    """Return the median wall time [seconds] of ``solves`` device-synced calls."""
    for _ in range(warmup):
        solve()
    _sync(device)
    samples: list[float] = []
    for _ in range(solves):
        _sync(device)
        t0: float = time.perf_counter()
        solve()
        _sync(device)
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def _max_abs_residual(fused: FusedEval, root: Tensor) -> float:
    """Component-wise ``max|residual(root)|`` (NaN when an instance failed)."""
    residual, _ = fused(root)
    return float(residual.abs().amax())


def _pct_lm(fused: FusedEval, x0: Tensor) -> float:
    """Percentage of instances whose Newton pass stalls and falls through to LM.

    Uses the private masked solve machine directly: the public
    ``newton_fused_then_lm`` hides the per-instance Newton status, and
    reproducing the fall-through rule here would duplicate it.
    """
    outcome = _newton_fused_core(fused, x0, False)
    failed: Tensor = (outcome.status == _FAILED).to(torch.float64)
    return float(failed.mean()) * 100.0


def make_fused_spec(fused: FusedEval, c0b: Tensor, perturb: float) -> SolveSpec:
    """Probe for a prebuilt fused kinetic ``(residual, Jacobian)`` closure.

    Args:
        fused:   ``fused(conc) -> (residual, Jacobian)``.
        c0b:     ``(B, num_spec)`` initial iterate.
        perturb: multiplies the starting iterate (1.0 uses it as given).

    Returns:
        A :class:`SolveSpec` timing ``newton_fused_then_lm(fused, x0)``.
    """
    x0: Tensor = c0b * perturb

    def solve() -> Tensor:
        return newton_fused_then_lm(fused, x0)

    def diagnose() -> tuple[float, float, float]:
        counter: list[int] = [0]

        def counting(x: Tensor) -> tuple[Tensor, Tensor]:
            counter[0] += 1
            return fused(x)

        root: Tensor = newton_fused_then_lm(counting, x0)
        return float(counter[0]), _max_abs_residual(fused, root), _pct_lm(fused, x0)

    return SolveSpec(solve=solve, diagnose=diagnose)


def make_rt_spec(
    rt_zone: RtZone,
    d: RtForcing,
    c_0: Tensor,
    dt_days: float,
    perturb: float,
) -> SolveSpec:
    """Probe for the fused kinetic Newton-to-LM solve of one ``RtZone``.

    Args:
        rt_zone:  a reacting :class:`RtZone`.
        d:        reactive-transport forcing (``conc_in`` may be ``(num_spec,)``).
        c_0:      ``(B, num_spec)`` initial concentrations.
        dt_days:  step length [days].
        perturb:  multiplies the ``(B, num_spec)`` starting iterate (1.0 uses the
                  zone's own initial concentration, larger pushes off-solution).

    Returns:
        A :class:`SolveSpec` timing ``newton_fused_then_lm(fused, x0)``.
    """
    fused, c0b = rt_zone.build_fused_eval(d, c_0, dt_days)
    return make_fused_spec(fused, c0b, perturb)


def make_rt_batch_spec(
    network: ReactionNetwork,
    rt_zone: RtZone,
    instances: int,
    dt_days: float,
    perturb: float,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator,
) -> SolveSpec:
    """Probe for batched RT parameters: many zones, one forcing (ensemble/calibration).

    Builds ``instances`` parameter sets of :class:`RtParameters` by randomising
    the mineral kinetic block around the chosen zone's real values (broadcasting
    the exact ``RtParameters.from_batched_array`` layout the batched model uses),
    then times one ``build_fused_eval`` Newton-to-LM solve across all instances
    under a shared forcing.

    Args:
        network:   the reaction network.
        rt_zone:   the base zone whose real parameter set is the sampling
                   centre (its name fixes the zone config looked up).
        instances: number of zones / parameter sets to solve in parallel.
        dt_days:   step length [days].
        perturb:   multiplies the ``(instances, num_spec)`` starting iterate (1.0
                   uses the trained initial concentrations, larger pushes
                   further off-solution).
        dtype:     compute dtype.
        device:    compute device.
        generator: random generator for parameter sampling.

    Returns:
        A :class:`SolveSpec` timing one batched fused RT solve over ``instances``
        zones.
    """
    num_min: int = int(network.num_minerals)
    aux = rt_zone.aux

    def sample(lo: float, hi: float) -> Tensor:
        return lo + (hi - lo) * torch.rand(
            instances, 1, dtype=dtype, device=device, generator=generator
        )

    n_cols: int = 3 + PARAMETERS_PER_MINERAL * num_min
    params_arr: Tensor = torch.empty(instances, n_cols, dtype=dtype, device=device)
    params_arr[:, 0] = float(rt_zone.dimensions.porosity)
    params_arr[:, 1] = float(rt_zone.dimensions.depth)
    params_arr[:, 2] = float(rt_zone.dimensions.passive_water_storage)
    for m in range(num_min):
        base = 3 + PARAMETERS_PER_MINERAL * m
        if aux is not None:
            sw_thr: float = float(aux.sw_threshold.reshape(-1)[m])
            sw_exp: float = float(aux.sw_exp.reshape(-1)[m])
            q10: float = float(aux.q_10.reshape(-1)[m])
            ssa: float = float(aux.ssa.reshape(-1)[m])
        else:
            sw_thr, sw_exp, q10, ssa = 0.5, 1.5, 3.0, 10.0
        params_arr[:, base + 0] = 0.8 * sw_thr * sample(0.5, 1.5).reshape(-1)
        params_arr[:, base + 1] = sw_exp * sample(0.5, 1.5).reshape(-1)
        params_arr[:, base + 2] = q10 * sample(0.8, 1.2).reshape(-1)
        params_arr[:, base + 3] = math.log10(ssa) + (
            0.5 * torch.randn(instances, dtype=dtype, device=device, generator=generator)
        )

    params_b: RtParameters = RtParameters.from_batched_array(params_arr, natural_scales=True)
    rt_b: RtZone = RtZone(
        network,
        parameters=params_b,
        do_reactions=rt_zone.do_reactions,
        do_speciation=rt_zone.do_speciation,
        name=rt_zone.name,
    )

    num_spec: int = int(network.num_species)
    c_0: Tensor = (
        torch.full(
            (instances, num_spec), 1e-3 * perturb, dtype=dtype, device=device
        )
        * (
            1.0
            + 0.05
            * torch.randn(
                instances, num_spec, dtype=dtype, device=device, generator=generator
            )
        )
    ).clamp_min(1e-12)
    p: float = 5.0
    d: RtForcing = RtForcing(
        conc_in=torch.full((num_spec,), 1e-3, dtype=dtype, device=device),
        hydro_step=HydroStep(
            state=torch.full((instances,), 25.0, dtype=dtype, device=device),
            forc_flux=torch.zeros(instances, dtype=dtype, device=device),
            lat_flux=torch.zeros(instances, dtype=dtype, device=device),
            vert_flux=torch.zeros(instances, dtype=dtype, device=device),
            vap_flux=torch.zeros(instances, dtype=dtype, device=device),
            q_in=(p + 1.0) * torch.ones(instances, dtype=dtype, device=device),
            lat_flux_ext=torch.zeros(instances, dtype=dtype, device=device),
            vert_flux_ext=torch.zeros(instances, dtype=dtype, device=device),
        ),
        hydro_forc=HydroForcing(precip=p, temp=10.0, pet=2.5, q_in=p + 1.0),
        s_w=torch.full((instances,), 0.5, dtype=dtype, device=device),
        z_w=torch.zeros(instances, dtype=dtype, device=device),
    )
    fused, c0b = rt_b.build_fused_eval(d, c_0, dt_days)
    return make_fused_spec(fused, c0b, 1.0)


def _tile_matrices(mat2d: Tensor, stress: int, dims: tuple[int, int, int, int]) -> Tensor:
    """Block-diagonalise an ``(num_min, num_species)`` matrix ``stress`` times.

    Species are laid out block-major within each class: all aqueous species,
    then all minerals, then all remaining (exchange) species — matching the
    layout the kinetic kernels hardcode (minerals at ``[num_aq : num_aq+num_min]``).
    """
    naq0, nm0, nex0, ns0 = dims
    ns: int = stress * ns0
    nm: int = stress * nm0
    naq: int = stress * naq0
    out: Tensor = torch.zeros(nm, ns, dtype=mat2d.dtype, device=mat2d.device)
    for s in range(stress):
        for i in range(ns0):
            if i < naq0:
                col: int = s * naq0 + i
            elif i < naq0 + nm0:
                col = naq + s * nm0 + (i - naq0)
            else:
                col = naq + nm + s * nex0 + (i - naq0 - nm0)
            out[s * nm0 : (s + 1) * nm0, col] = mat2d[:, i]
    return out


def _repeat_last(x: Tensor, stress: int) -> Tensor:
    """Repeat a per-mineral vector along its trailing (mineral) axis."""
    reps: tuple[int, ...] = (1,) * (x.ndim - 1) + (stress,)
    return x.repeat(*reps)


def _tile_species(x: Tensor, stress: int, dims: tuple[int, int, int, int]) -> Tensor:
    """Replicate a species vector per block, preserving the tiled layout."""
    naq0, nm0, _nex0, _ns0 = dims
    reps: tuple[int, ...] = (1,) * (x.ndim - 1) + (stress,)
    aq: Tensor = x[..., :naq0].repeat(*reps)
    mn: Tensor = x[..., naq0 : naq0 + nm0].repeat(*reps)
    ex: Tensor = x[..., naq0 + nm0 :].repeat(*reps)
    return torch.cat([aq, mn, ex], dim=-1)


def make_stressed_rt_problem(
    rt_zone: RtZone,
    d: RtForcing,
    c_0: Tensor,
    dt_days: float,
    stress: int,
) -> tuple[FusedEval, Tensor]:
    """Widen the kinetic solve to a ``stress``-fold block-diagonal network.

    Replicates the zone's reaction network ``stress`` times inside one zone, so
    the fused Newton/LM solve operates on ``stress * num_species`` unknowns and
    the batched LU cost grows accordingly.  Blocks are exact independent copies,
    so the tiled problem has the same per-block solution as the base and lets
    the benchmark reach the arithmetic-bound regime the 4-species East River
    network never exercises.

    Args:
        rt_zone:  the base reacting zone.
        d:        base forcing (``conc_in`` is replicated per block).
        c_0:      base ``(B, num_spec)`` initial concentrations.
        dt_days:  step length [days].
        stress:   number of network copies (``1`` returns the base problem).

    Returns:
        ``(fused, c0b)`` for the widened system.
    """
    net = rt_zone.network
    ns0: int = int(net.num_species)
    naq0: int = int(net.num_aqueous_species)
    nm0: int = int(net.num_minerals)
    nex0: int = ns0 - naq0 - nm0
    dims: tuple[int, int, int, int] = (naq0, nm0, nex0, ns0)

    if stress <= 1:
        return rt_zone.build_fused_eval(d, c_0, dt_days)

    ns: int = stress * ns0
    naq: int = stress * naq0
    nm: int = stress * nm0
    dtype: torch.dtype = c_0.dtype
    device: torch.device = c_0.device

    stoich0: Tensor = net.mineral_stoichiometry.to(dtype=dtype, device=device)
    stoich: Tensor = torch.zeros(ns, nm, dtype=dtype, device=device)
    for s in range(stress):
        stoich[s * naq0 : (s + 1) * naq0, s * nm0 : (s + 1) * nm0] = stoich0[:naq0, :]

    monod: MonodParameters = MonodParameters(
        monod_mat=_tile_matrices(rt_zone.monod.monod_mat, stress, dims),
        inhib_mat=_tile_matrices(rt_zone.monod.inhib_mat, stress, dims),
    )
    tst: TstParameters = TstParameters(
        stoich=_tile_matrices(rt_zone.tst.stoich, stress, dims),
        dep=_tile_matrices(rt_zone.tst.dep, stress, dims),
        min_eq_const=_repeat_last(
            rt_zone.tst.min_eq_const.to(dtype=dtype, device=device), stress
        ),
    )
    aux: MineralParameters | None = None
    if rt_zone.aux is not None:
        aux = MineralParameters(
            sw_threshold=_repeat_last(rt_zone.aux.sw_threshold.to(dtype=dtype, device=device), stress),
            sw_exp=_repeat_last(rt_zone.aux.sw_exp.to(dtype=dtype, device=device), stress),
            q_10=_repeat_last(rt_zone.aux.q_10.to(dtype=dtype, device=device), stress),
            ssa=_repeat_last(rt_zone.aux.ssa.to(dtype=dtype, device=device), stress),
        )
    misc: MiscData = MiscData(
        mineral_stoichiometry=stoich,
        species_mobility=_tile_species(rt_zone.misc.species_mobility, stress, dims),
        mineral_molar_mass=_repeat_last(
            rt_zone.misc.mineral_molar_mass.to(dtype=dtype, device=device), stress
        ),
        rate_const=_repeat_last(
            rt_zone.misc.rate_const.to(dtype=dtype, device=device), stress
        ),
    )
    network: Any = SimpleNamespace(
        num_species=ns,
        num_aqueous_species=naq,
        num_minerals=nm,
        mineral_stoichiometry=stoich,
    )
    d_t: RtForcing = RtForcing(
        conc_in=_tile_species(d.conc_in, stress, dims),
        hydro_step=d.hydro_step,
        hydro_forc=d.hydro_forc,
        s_w=d.s_w,
        z_w=d.z_w,
    )
    ctx = _build_fused_eval(
        c_0=_tile_species(c_0, stress, dims),
        d=d_t,
        dt=dt_days,
        network=network,
        monod=monod,
        tst=tst,
        aux=aux,
        misc=misc,
        depth=rt_zone.dimensions.depth,
        min_start=naq,
        q_in_field="q_in",
        do_reactions=rt_zone.do_reactions,
    )
    return ctx.fused, ctx.c0b


def make_speciation_spec(
    eq: EquilibriumParameters,
    chms: Tensor,
    x0: Tensor,
    perturb: float,
) -> SolveSpec:
    """Probe for the equilibrium speciation solve (``x_free_solve``).

    Args:
        eq:      equilibrium matrices for the network.
        chms:    ``(B, num_species)`` input concentrations.
        x0:      ``(B, num_free)`` warm start (usually a converged solve).
        perturb: multiplies the warm start (1.0 = realistic 1-2 iteration solve).

    Returns:
        A :class:`SolveSpec` timing ``eq.x_free_solve``.  The residual/Jacobian
        pair is reproduced with the same public callbacks ``x_free_solve`` uses
        so the evaluation count is accurate; the LM fraction is not exposed by
        the public entry point and is reported as ``nan``.
    """
    xn: Tensor = x0 * perturb

    def solve() -> Tensor:
        return eq.x_free_solve(chms, xn)

    def diagnose() -> tuple[float, float, float]:
        counter: list[int] = [0]
        total: Tensor = eq.total.to(dtype=chms.dtype, device=chms.device)
        c_tot: Tensor = chms @ total.T
        if eq.charge_row is not None:
            c_tot = c_tot.clone()
            c_tot[:, eq.charge_row] = 0.0

        def counting(x: Tensor) -> Tensor:
            counter[0] += 1
            return eq.residual(x, c_tot)

        root: Tensor = newton_analytic_then_lm(
            counting, eq.jacobian_residual, xn, raise_on_failure=False
        )
        residual: Tensor = eq.residual(root, c_tot)
        return float(counter[0]), float(residual.abs().amax()), float("nan")

    return SolveSpec(solve=solve, diagnose=diagnose)


def make_hydro_spec(
    zone: HydrologicZone,
    s_0: Tensor,
    d: HydroForcing,
    dt: float,
    perturb: float,
) -> SolveSpec:
    """Probe for the implicit-Euler storage root (damped AD Newton).

    Args:
        zone:    a hydrology zone (e.g. :class:`SurfaceZone`).
        s_0:     ``(B,)`` storage target for the step.
        d:       forcing for the step (scalar or broadcastable fields).
        dt:      step length [days].
        perturb: multiplies the starting storage guess (1.0 = the target ``s_0``).

    Returns:
        A :class:`SolveSpec` timing ``_newton_storage_solve``.  Hydro never
        enters LM, so the LM fraction is 0.  The reported ``evals`` counts
        residual evaluations *including* the solver's ``no_grad`` backtracking
        acceptance evals (1-2 per Newton trip for well-behaved residuals).
    """

    def f(s: Tensor) -> Tensor:
        return _hydro_residual(zone, s, s_0, d, dt)

    x0: Tensor = s_0 * perturb

    def solve() -> Tensor:
        return _newton_storage_solve(f, x0)

    def diagnose() -> tuple[float, float, float]:
        counter: list[int] = [0]

        def counting(s: Tensor) -> Tensor:
            counter[0] += 1
            return f(s)

        root: Tensor = _newton_storage_solve(counting, x0)
        return float(counter[0]), float(f(root).abs().amax()), 0.0

    return SolveSpec(solve=solve, diagnose=diagnose)


def make_hydro_seq_spec(
    zone: HydrologicZone,
    s_0: Tensor,
    forcings: list[HydroForcing],
    dt: float,
    perturb: float,
) -> SolveSpec:
    """Probe for sequential time-stepping: one zone, many forcings (the model loop).

    This is the pattern the forward pass actually runs: a single zone is stepped
    forward in time, each step reusing the previous state as its warm start.

    Args:
        zone:      a hydrology zone (e.g. :class:`SurfaceZone`).
        s_0:       initial storage (scalar).
        forcings:  real per-day forcing for each step.
        dt:        step length [days].
        perturb:   multiplies the initial storage (1.0 = the real initial state).

    Returns:
        A :class:`SolveSpec` timing the full ``len(forcings)``-step sequence; the
        caller sets the row's instance count to ``len(forcings)`` so the
        per-instance time is the wall time of one daily storage step.  ``evals``
        is per-day and includes the backtracking acceptance evals.
    """
    def solve() -> Tensor:
        s: Tensor = s_0 * perturb
        for d in forcings:
            s = cast(Tensor, zone.step(s, d, dt).state)
        return s

    def diagnose() -> tuple[float, float, float]:
        counter: list[int] = [0]
        s: Tensor = s_0 * perturb
        prev: Tensor = s
        for dd in forcings:
            prev = s

            def counting(x: Tensor) -> Tensor:
                counter[0] += 1
                return cast(Tensor, _hydro_residual(zone, x, prev, dd, dt))

            s = _newton_storage_solve(counting, s)
        residual: Tensor = _hydro_residual(zone, s, prev, forcings[-1], dt)
        return float(counter[0]) / len(forcings), float(residual.abs().amax()), 0.0

    return SolveSpec(solve=solve, diagnose=diagnose)


def make_hydro_batch_spec(
    instances: int,
    dt: float,
    perturb: float,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator,
) -> SolveSpec:
    """Probe for batched zone parameters: many zones, one forcing (ensemble/calibration).

    Mirrors the notebook pattern: ``instances`` independent :class:`SurfaceZone`
    objects with randomised parameters (draws from the East River parameter
    ranges), stepped once under a single shared forcing.  The whole call is one
    vectorised ``zone.step``.

    Args:
        instances: number of zones (independent parameter sets).
        dt:        step length [days].
        perturb:   multiplies the initial storage (1.0 = the default 25 mm).
        dtype:     compute dtype.
        device:    compute device.
        generator: random generator for parameter sampling.

    Returns:
        A :class:`SolveSpec` timing one batched ``SurfaceZone.step`` over
        ``instances`` zones; ``evals`` counts residual evaluations including
        the backtracking acceptance evals.
    """
    def sample(lo: float, hi: float) -> Tensor:
        return lo + (hi - lo) * torch.rand(
            instances, dtype=dtype, device=device, generator=generator
        )

    zone: HydrologicZone = pt.SurfaceZone(
        fc=sample(100.0, 1000.0),
        lp=sample(0.05, 1.0),
        beta=sample(0.05, 5.0),
        k0=sample(0.0, 1.0),
        thr=sample(0.0, 1000.0),
    )
    s_0: Tensor = torch.full(
        (instances,), 25.0 * perturb, dtype=dtype, device=device
    )
    d: HydroForcing = HydroForcing(precip=5.0, temp=10.0, pet=2.5, q_in=5.0)

    def solve() -> Tensor:
        return cast(Tensor, zone.step(s_0, d, dt).state)

    def diagnose() -> tuple[float, float, float]:
        counter: list[int] = [0]

        def counting(s: Tensor) -> Tensor:
            counter[0] += 1
            return cast(Tensor, _hydro_residual(zone, s, s_0, d, dt))

        root: Tensor = _newton_storage_solve(counting, s_0)
        residual: Tensor = _hydro_residual(zone, root, s_0, d, dt)
        return float(counter[0]), float(residual.abs().amax()), 0.0

    return SolveSpec(solve=solve, diagnose=diagnose)


def _random_spd(
    instances: int,
    dim: int,
    cond: float,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator,
) -> Tensor:
    """Random symmetric positive-definite ``(B, dim, dim)`` with condition ``cond``.

    Builds ``J = Q diag(e) Q^T`` from a random orthogonal ``Q`` and eigenvalues
    log-spaced from 1 down to ``1/cond``, so ``cond(J) = cond`` exactly while
    ``||J|| ~ 1`` keeps the solved residual near the dtype's ulp (the solver's
    convergence bar is *absolute*), which matters for float32 runs.
    """
    a: Tensor = torch.randn(instances, dim, dim, dtype=dtype, device=device, generator=generator)
    q, _ = torch.linalg.qr(a)
    evals: Tensor = torch.logspace(0.0, -math.log10(cond), dim, dtype=dtype, device=device)
    return q @ torch.diag(evals) @ q.transpose(-1, -2)


def make_synthetic_spec(
    instances: int,
    dim: int,
    cond: float,
    dtype: torch.dtype,
    device: torch.device,
    perturb: float,
    seed: int,
) -> SolveSpec:
    """Probe for an isolated linear system ``f(X) = J X - b`` with SPD ``J``.

    A linear residual makes Newton + Armijo finish in one step (the analytic
    Jacobian is exact and constant), so the timing isolates the batched
    ``lu_factor``/``lu_solve`` cost; LM is never reached.

    Args:
        instances: number of independent systems ``B``.
        dim:       system dimension (residual/Jacobian are ``(B, dim)`` / ``(B, dim, dim)``).
        cond:      condition number of ``J``.
        dtype:     tensor dtype.
        device:    compute device.
        perturb:   scales the starting iterate ``0.5 * x_true`` (linear: iteration
                   count is unchanged, but the first residual magnitude is).
        seed:      RNG seed for ``J`` / ``b`` / ``x_true``.

    Returns:
        A :class:`SolveSpec` timing ``newton_fused_then_lm`` on the linear system.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    cpu: torch.device = torch.device("cpu")
    j_mat: Tensor = _random_spd(instances, dim, cond, dtype, cpu, generator).to(device=device)
    x_true: Tensor = torch.randn(
        instances, dim, dtype=dtype, device=cpu, generator=generator
    ).to(device=device)
    b: Tensor = torch.bmm(j_mat, x_true.unsqueeze(-1)).squeeze(-1)

    def fused(x: Tensor) -> tuple[Tensor, Tensor]:
        residual: Tensor = torch.bmm(j_mat, x.unsqueeze(-1)).squeeze(-1) - b
        return residual, j_mat

    x0: Tensor = 0.5 * x_true * perturb

    def solve() -> Tensor:
        return newton_fused_then_lm(fused, x0)

    def diagnose() -> tuple[float, float, float]:
        counter: list[int] = [0]

        def counting(x: Tensor) -> tuple[Tensor, Tensor]:
            counter[0] += 1
            return fused(x)

        root: Tensor = newton_fused_then_lm(counting, x0)
        return float(counter[0]), _max_abs_residual(fused, root), _pct_lm(fused, x0)

    return SolveSpec(solve=solve, diagnose=diagnose)


def _fmt(value: float, width: int, prec: int = 1) -> str:
    """Format a metric compactly; ``-`` marks a value the solver does not report."""
    return "-".rjust(width) if math.isnan(value) else f"{value:.{prec}f}".rjust(width)


def _choose_zone(
    model_type: type[pt.Model],
    zone_configs: Mapping[str, object],
    override: str | None,
) -> str:
    """Pick the zone the reactive solver probes run on.

    Defaults to the first zone with both reactions and speciation enabled.  The
    benchmark used ``get_zone_names()[0]``, which is a snow zone with
    ``do_reactions=False`` -- so its "rt" solve was reaction-free, transport-free
    and trivially converged at the initial guess.

    Args:
        model_type:   the model class (its zone names fix the search order).
        zone_configs: per-zone configs exposing ``do_reactions`` /
                      ``do_speciation``.
        override:     explicit zone name from the CLI, or ``None``.

    Returns:
        The chosen zone name.
    """
    names: list[str] = model_type.get_zone_names()
    if override is not None:
        if override not in names:
            raise SystemExit(f"unknown zone {override!r}; choose from {names}")
        return override
    for name in names:
        cfg = zone_configs[name]
        if bool(getattr(cfg, "do_reactions")) and bool(getattr(cfg, "do_speciation")):
            return name
    for name in names:
        if bool(getattr(zone_configs[name], "do_reactions")):
            return name
    return names[0]


def _build_context(args: argparse.Namespace) -> dict[str, object]:
    """Load the East River config and build the shared zones / forcing objects."""
    inputs = load_input_data(
        pt.HbvLateralModel, test_wys=[2017, 2020], val_wy=2018, site="EastRiver"
    )
    site_data: DataFrame = hbv_data.load_site_data("EastRiver")
    doc_df: DataFrame = (
        load_all_doc_data()
        .loc[lambda d: d.station == "er_phf0", "npoc"]
        .rename("DOC")
        .to_frame()
        / 1000.0
        / 12.01
    )
    doc_df = doc_df.groupby(doc_df.index).mean()
    dates: DatetimeIndex = cast(DatetimeIndex, site_data.index)
    window: DatetimeIndex = fwd.choose_window(
        dates, wy=None, start=None, end=None, days=args.days, full=False
    )
    cfg: BatchedConfig = cast(
        BatchedConfig,
        fwd.build_window_config(inputs.cal_config, site_data, doc_df, window),
    )
    device: torch.device = torch.device(args._device)
    dtype: torch.dtype = args._dtype

    model_type: type[pt.Model] = cfg["model_type"]
    zone_configs = cfg["zone_configs"]
    network = cfg["network"]
    num_hydro: int = model_type.get_num_hydro_parameters()
    num_rt: int = model_type.get_num_rt_parameters(zone_configs, network)

    mid: list[float] = []
    for nm in inputs.bounds:
        lo, hi = inputs.bounds[nm]
        mid.append(float(0.5 * (lo + hi)))
    arr: np.ndarray = np.asarray(mid, dtype=float).reshape(1, -1)
    hydro_arr: np.ndarray = arr[:, :num_hydro]
    rt_arr: np.ndarray = arr[:, num_hydro : num_hydro + num_rt]

    blocks, scales = _hydro_zone_blocks(model_type, hydro_arr, device, dtype)
    zones: list[HydrologicZone] = _build_hydro_zones(model_type, blocks)
    geo_model: HydrologicalModel = HydrologicalModel(structure=model_type.structure)
    surface_index: int = int(geo_model.surface_zone_ids[0])
    surface_zone: HydrologicZone = zones[surface_index]

    # Real per-day forcing for the probe zone (``(1, T, Z)``; batch size 1 here).
    geometry: Geometry = Geometry(
        lat_mat=geo_model.lat_mat.to(dtype=dtype, device=device),
        vert_mat=geo_model.vert_mat.to(dtype=dtype, device=device),
        size_mat=geo_model._get_size_mat().to(dtype=dtype, device=device),
        river_ids=geo_model.get_river_zone_ids(),
        surface_ids=geo_model.surface_zone_ids,
        zone_names=geo_model.zone_names(),
    )
    all_p, all_t, all_pet = _forcing_source_tensors(
        cfg["forc"], len(geometry.surface_ids), device, dtype
    )
    zone_precip, zone_temp, zone_pet = _per_lane_forcing(
        geometry, all_p, all_t, all_pet, scales, dtype
    )

    # Real precipitation chemistry entering the surface zone (num_steps, num_spec).
    precip_conc: np.ndarray = np.asarray(cfg["precip_conc"], dtype=np.float64)
    conc_series: Tensor = torch.tensor(
        precip_conc[surface_index], dtype=dtype, device=device
    )

    rt_pars = _rt_blocks(model_type, network, zone_configs, rt_arr, device, dtype)
    zname: str = _choose_zone(model_type, zone_configs, args.zone)
    rt_index: int = model_type.get_zone_names().index(zname)
    rt_zone = RtZone(
        network,
        parameters=rt_pars[rt_index],
        do_reactions=zone_configs[zname].do_reactions,
        do_speciation=zone_configs[zname].do_speciation,
        name=zname,
    )
    return {
        "network": network,
        "rt_zone": rt_zone,
        "surface_zone": surface_zone,
        "zone_precip": zone_precip,
        "zone_temp": zone_temp,
        "zone_pet": zone_pet,
        "conc_series": conc_series,
        "rt_index": rt_index,
        "surface_index": surface_index,
        "dtype": dtype,
        "device": device,
    }


def _zero(batch: int, dtype: torch.dtype, device: torch.device) -> Tensor:
    """A ``(batch,)`` zero tensor (shared hydro flux field placeholder)."""
    return torch.zeros(batch, dtype=dtype, device=device)


def _rt_forcing(
    conc_daily: Tensor,
    precip_daily: Tensor,
    temp_daily: Tensor,
    idx: Tensor,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator,
) -> RtForcing:
    """Reactive-transport forcing for one step, drawn from real series.

    Instance ``k`` receives day ``idx[k]`` of the real East River precipitation
    chemistry / precipitation / temperature, with a per-instance inflow scale
    and soil-moisture state.  The inflow floor keeps transport active on dry
    days, so the batched Newton solve does real work in every instance (unlike
    the old snow-zone probe, which had ``do_reactions=False`` and zero inflow).

    Args:
        conc_daily:   ``(T, num_spec)`` precipitation chemistry [mmol/L].
        precip_daily: ``(T,)`` precipitation [mm].
        temp_daily:   ``(T,)`` temperature [C].
        idx:          ``(B,)`` day index per instance.
        dtype:        tensor dtype.
        device:       compute device.
        generator:    RNG for the per-instance inflow scale / moisture.

    Returns:
        An :class:`RtForcing` whose fields are all ``(B,)`` / ``(B, num_spec)``.
    """
    batch: int = int(idx.shape[0])
    num_spec: int = int(conc_daily.shape[1])
    z: Tensor = _zero(batch, dtype, device)
    scale: Tensor = 0.5 + torch.rand(batch, dtype=dtype, device=device, generator=generator)
    q_in: Tensor = (precip_daily[idx] + 1.0) * scale  # nonzero inflow => transport matters
    conc_in: Tensor = (conc_daily[idx] * scale.unsqueeze(-1)).clamp_min(0.0)
    s_w: Tensor = 0.35 + 0.25 * torch.rand(
        batch, dtype=dtype, device=device, generator=generator
    )
    hs: HydroStep = HydroStep(
        state=10.0 + 30.0 * torch.rand(batch, dtype=dtype, device=device, generator=generator),
        forc_flux=z,
        lat_flux=z,
        vert_flux=z,
        vap_flux=z,
        q_in=q_in,
        lat_flux_ext=z,
        vert_flux_ext=z,
    )
    return RtForcing(
        conc_in=conc_in,
        hydro_step=hs,
        hydro_forc=HydroForcing(precip=precip_daily[idx], temp=temp_daily[idx]),
        s_w=s_w,
        z_w=z,
    )


def _rt_warm_start(
    rt_zone: RtZone,
    conc_daily: Tensor,
    precip_daily: Tensor,
    temp_daily: Tensor,
    num_spec: int,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator,
    spinup: int,
) -> Tensor:
    """Spin the reacting zone up over the first ``spinup`` real days.

    The timed solve then uses *later* days with different chemistry, so the
    returned state is a genuine (non-equilibrium) warm start rather than the
    next step's exact root.
    """
    days: int = int(conc_daily.shape[0])
    spinup = max(1, min(spinup, days))
    c: Tensor = torch.full((1, num_spec), 1e-3, dtype=dtype, device=device)
    idx: Tensor = torch.arange(spinup, device=device)
    for t in range(spinup):
        d: RtForcing = _rt_forcing(
            conc_daily, precip_daily, temp_daily, idx[t : t + 1], dtype, device, generator
        )
        c = cast(Tensor, rt_zone.step(c, d, 1.0).state)
    return c


def _hydro_trajectory(
    zone: HydrologicZone,
    zone_precip: Tensor,
    zone_temp: Tensor,
    zone_pet: Tensor,
    zone_index: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Tensor, list[HydroForcing]]:
    """Run real East River forcing through ``zone`` to get a storage trajectory.

    Args:
        zone:        hydrology zone to spin up.
        zone_precip: ``(1, T, Z)`` per-zone precipitation [mm].
        zone_temp:   ``(1, T, Z)`` per-zone temperature [C].
        zone_pet:    ``(1, T, Z)`` per-zone PET [mm].
        zone_index:  flat zone index to slice the forcing at.
        dtype:       tensor dtype.
        device:      compute device.

    Returns:
        ``(before, forcings)`` where ``before[t]`` is the ``(1,)`` storage at the
        start of day ``t`` and ``forcings[t]`` is that day's forcing.  Both have
        length ``T``.
    """
    num_days: int = int(zone_precip.shape[1])
    s: Tensor = torch.as_tensor(
        zone.default_init_state(), dtype=dtype, device=device
    ).reshape(1)
    before: list[Tensor] = []
    forcings: list[HydroForcing] = []
    for t in range(num_days):
        d: HydroForcing = HydroForcing(
            precip=zone_precip[0, t, zone_index],
            temp=zone_temp[0, t, zone_index],
            pet=zone_pet[0, t, zone_index],
            q_in=torch.zeros(1, dtype=dtype, device=device),
        )
        before.append(s)
        forcings.append(d)
        s = cast(Tensor, zone.step(s, d, 1.0).state)
    return torch.cat(before), forcings


def run(args: argparse.Namespace) -> int:
    """Build every requested solver probe and time each (solver, batch) cell."""
    device: torch.device = (
        torch.device("cuda")
        if args.device == "auto" and torch.cuda.is_available()
        else torch.device(args.device)
    )
    dtype: torch.dtype = (
        default_dtype(device)
        if args.dtype == "auto"
        else (torch.float32 if args.dtype == "float32" else torch.float64)
    )
    args._device = str(device)
    args._dtype = dtype

    solvers: list[str] = [s.strip() for s in args.solvers.split(",") if s.strip()]
    if args.synthetic and "synthetic" not in solvers:
        solvers.append("synthetic")
    instances_to_run: list[int] = [
        int(n) for n in args.instances.split(",") if n.strip()
    ]
    print(
        f"== solver bench ==  device: {device}  dtype: {dtype}  "
        f"solves: {args.solves}  warmup: {args.warmup}  "
        f"perturb: {args.perturb}  stress: {args.stress}  instances: {instances_to_run}"
    )
    ctx = _build_context(args)
    network: ReactionNetwork = cast(ReactionNetwork, ctx["network"])
    rt_zone: RtZone = cast(RtZone, ctx["rt_zone"])
    surface_zone: HydrologicZone = cast(HydrologicZone, ctx["surface_zone"])
    num_spec: int = int(network.num_species)  # type: ignore[attr-defined]
    zone_precip: Tensor = cast(Tensor, ctx["zone_precip"])
    zone_temp: Tensor = cast(Tensor, ctx["zone_temp"])
    zone_pet: Tensor = cast(Tensor, ctx["zone_pet"])
    conc_series: Tensor = cast(Tensor, ctx["conc_series"])
    rt_index: int = cast(int, ctx["rt_index"])
    surface_index: int = cast(int, ctx["surface_index"])
    dt_days: float = 1.0

    generator: torch.Generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed))

    p_rt: Tensor = zone_precip[0, :, rt_index]
    t_rt: Tensor = zone_temp[0, :, rt_index]
    num_rt_days: int = int(p_rt.shape[0])

    # Off-solution warm starts, built once and reused across every timed batch.
    c_warm: Tensor = _rt_warm_start(
        rt_zone,
        conc_series,
        p_rt,
        t_rt,
        num_spec,
        dtype,
        device,
        generator,
        int(args.spinup),
    )
    s_hist, hydro_forcings = _hydro_trajectory(
        surface_zone, zone_precip, zone_temp, zone_pet, surface_index, dtype, device
    )
    p_day: Tensor = zone_precip[0, :, surface_index]
    t_day: Tensor = zone_temp[0, :, surface_index]
    pet_day: Tensor = zone_pet[0, :, surface_index]
    num_hydro_days: int = int(s_hist.shape[0])

    eq = rt_zone.eq
    n_free: int = int(eq.num_free)
    print(
        f"   rt zone: {rt_zone.name}  (reactions={rt_zone.do_reactions}, "
        f"speciation={rt_zone.do_speciation})  num_spec={num_spec}  "
        f"spinup={args.spinup}  window={num_rt_days}d"
    )

    rows: list[Row] = []
    for solver in solvers:
        for instances in instances_to_run:
            cell_n: int = instances
            spec: SolveSpec
            if solver == "rt":
                # Days after the spin-up window, so the warm start is never the
                # exact root of the timed forcing.
                span: int = max(1, num_rt_days - int(args.spinup))
                idx: Tensor = int(args.spinup) + (
                    torch.arange(instances, device=device) % span
                )
                c_0: Tensor = (
                    c_warm.expand(instances, num_spec)
                    * (
                        1.0
                        + 0.05
                        * torch.randn(
                            instances,
                            num_spec,
                            dtype=dtype,
                            device=device,
                            generator=generator,
                        )
                    )
                ).clamp_min(1e-12)
                d_rt: RtForcing = _rt_forcing(
                    conc_series, p_rt, t_rt, idx, dtype, device, generator
                )
                if int(args.stress) > 1:
                    fused, c0b = make_stressed_rt_problem(
                        rt_zone, d_rt, c_0, dt_days, int(args.stress)
                    )
                    spec = make_fused_spec(fused, c0b, args.perturb)
                else:
                    spec = make_rt_spec(rt_zone, d_rt, c_0, dt_days, args.perturb)
            elif solver == "rt-batch":
                spec = make_rt_batch_spec(
                    network,
                    rt_zone,
                    instances,
                    dt_days,
                    args.perturb,
                    dtype,
                    device,
                    generator,
                )
            elif solver == "speciation":
                if n_free == 0:
                    continue
                chms_prev: Tensor = c_warm.expand(instances, num_spec).clone()
                zeros: Tensor = torch.zeros(
                    instances, n_free, dtype=dtype, device=device
                )
                x_warm: Tensor = eq.x_free_solve(chms_prev, zeros).detach()
                scale: Tensor = 1.0 + 0.1 * torch.randn(
                    instances, 1, dtype=dtype, device=device, generator=generator
                )
                chms: Tensor = (chms_prev * scale).clamp_min(1e-12)
                spec = make_speciation_spec(eq, chms, x_warm, args.perturb)
            elif solver == "hydro":
                hidx: Tensor = torch.arange(instances, device=device) % num_hydro_days
                s_0: Tensor = s_hist[hidx]
                d_hydro: HydroForcing = HydroForcing(
                    precip=p_day[hidx],
                    temp=t_day[hidx],
                    pet=pet_day[hidx],
                    q_in=torch.zeros(instances, dtype=dtype, device=device),
                )
                spec = make_hydro_spec(surface_zone, s_0, d_hydro, dt_days, args.perturb)
            elif solver == "hydro-seq":
                # One zone stepped day by day: the cell size is bounded by the
                # real forcing record (sq sweep sizes only restate the full month).
                cell_n = min(instances, len(hydro_forcings))
                spec = make_hydro_seq_spec(
                    surface_zone,
                    s_hist[0],
                    list(hydro_forcings[:cell_n]),
                    dt_days,
                    args.perturb,
                )
            elif solver == "hydro-batch":
                spec = make_hydro_batch_spec(
                    instances, dt_days, args.perturb, dtype, device, generator
                )
            elif solver == "synthetic":
                spec = make_synthetic_spec(
                    instances, args.dim, args.cond, dtype, device, args.perturb, args.seed
                )
            else:
                raise SystemExit(f"unknown solver '{solver}'")

            stress_cell: int = int(args.stress) if solver == "rt" else 1
            median_us: float = _timed(spec.solve, device, args.solves, args.warmup) * 1e6
            evals: float
            max_res: float
            pct_lm: float
            evals, max_res, pct_lm = spec.diagnose()
            rows.append(
                Row(
                    solver=solver,
                    instances=cell_n,
                    device=str(device),
                    dtype=str(dtype).replace("torch.", ""),
                    median_us=median_us,
                    evals=evals,
                    max_res=max_res,
                    pct_lm=pct_lm,
                    stress=stress_cell,
                )
            )
            trivial: bool = solver in ("rt", "rt-batch") and not math.isnan(evals) and evals <= 1.0
            flag: str = "  [TRIVIAL: converged at the initial guess]" if trivial else ""
            print(
                f"   {solver:<11} n={cell_n:<7} "
                f"total={_fmt(median_us, 11)} us  "
                f"per-instance={_fmt(median_us / cell_n, 10, 2)} us  "
                f"evals={_fmt(evals, 7, 2)}  "
                f"max|res|={_fmt(max_res, 10, 2)}  "
                f"%LM={_fmt(pct_lm, 6, 1)}  x{stress_cell}{flag}"
            )

    print("\n== summary ==")
    header: str = (
        f"{'solver':<11} {'n':>8} {'x':>3}  {'device':<6} {'dtype':<8} "
        f"{'total_us':>12} {'us/inst':>11} {'evals':>7} {'max|res|':>11} {'%LM':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r.solver:<11} {r.instances:>8} {r.stress:>3}  "
            f"{r.device:<6} {r.dtype:<8} "
            f"{_fmt(r.median_us, 12)} {_fmt(r.median_us / r.instances, 11, 3)} "
            f"{_fmt(r.evals, 7, 2)} "
            f"{_fmt(r.max_res, 11, 2)} {_fmt(r.pct_lm, 6, 1)}"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--solvers",
        default=",".join(DEFAULT_SOLVERS),
        help=f"comma-separated subset of {', '.join(DEFAULT_SOLVERS)},hydro,rt-batch,synthetic",
    )
    parser.add_argument(
        "--instances",
        default=",".join(str(n) for n in DEFAULT_INSTANCES),
        help=(
            "comma-separated instance counts (how many zones / parameter sets are "
            "solved in parallel); powers of 10 up to 1,000,000 by default"
        ),
    )
    parser.add_argument("--device", default="auto", help="cpu | cuda | auto")
    parser.add_argument("--dtype", default="auto", help="float32 | float64 | auto")
    parser.add_argument("--solves", type=int, default=200, help="timed solves per cell")
    parser.add_argument("--warmup", type=int, default=10, help="untimed warmup solves")
    parser.add_argument(
        "--perturb",
        type=float,
        default=1.0,
        help="multiplier on the starting iterate (1.0 = realistic solve)",
    )
    parser.add_argument(
        "--zone",
        default=None,
        help="zone name for the reactive probes (default: first reacting/speciating zone)",
    )
    parser.add_argument(
        "--spinup",
        type=int,
        default=5,
        help="days of real forcing used to build the off-solution warm start",
    )
    parser.add_argument(
        "--stress",
        type=int,
        default=1,
        help=(
            "network copies for the rt probe: builds a block-diagonal zone with "
            "stress*num_species unknowns (1 = the real 4-species East River network)"
        ),
    )
    parser.add_argument("--days", type=int, default=30, help="config window days")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--synthetic", action="store_true", help="add the synthetic solver")
    parser.add_argument("--dim", type=int, default=16, help="synthetic dimension")
    parser.add_argument("--cond", type=float, default=1e3, help="synthetic condition number")
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
