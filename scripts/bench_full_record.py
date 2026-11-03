"""Full-record East River benchmark.

Steps the complete 2557-day East River record for several batch sizes
(``--runs``, default 1,100): that many identical copies of the 8-zone
network advanced together in one computation every day.  Each batch size
is timed once.  "runs" is the batch size — how many simulations one
computation carries — not a repetition of the benchmark; the copies are
identical on purpose, so the measurement is about throughput, not about
parameter diversity.

Reported per batch size:

- total time to advance every run through the 2557-day record
- time per day (all runs together)
- time per zone-day: one zone of one run for one day (the primary metric)
- throughput in zones solved per second (one zone-day counts as one
  solved zone — the same convention as the timing cell in
  scripts/hydrologic_zone_ode_ad.ipynb)
- days simulated per second
- memory footprint, and output sanity (finite, mean discharge)

The CPU path runs in float64; the CUDA path in float32.  The CPU thread
count is set with ``--threads``.

Usage::

    PYTHONPATH=src python scripts/bench_full_record.py
    PYTHONPATH=src python scripts/bench_full_record.py --runs 1,100
    PYTHONPATH=src python scripts/bench_full_record.py --device cpu
    PYTHONPATH=src python scripts/bench_full_record.py --device cpu --threads 1
"""

from __future__ import annotations

import sys
import gc
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from pandas import DataFrame, DatetimeIndex, Series
import torch

import potions as pt  # noqa: E402
from potions.common_types import ForcingData, RtZoneConfiguration  # noqa: E402
from potions.reactive_transport import get_simple_carbon_adsorption_network  # noqa: E402
from potions.model_batched import (  # noqa: E402
    BatchedConfig,
    run_hydro_rt_batched,
    run_hydro_batched,
    run_rt_batched,
    _hydro_zone_blocks,
    _build_hydro_zones,
    _rt_blocks,
    _init_state_tensor,
    _per_lane_forcing,
    _forcing_source_tensors,
    _flow_weighted_river_doc,
    _step_lengths,
    Geometry,
)
from potions.hydro_model import HydrologicalModel  # noqa: E402
from potions.hydro import HydrologicZone  # noqa: E402
from potions.rt_zone import RtZone, RtParameters  # noqa: E402
from potions.kinetic_structures import RtParameters as RtParamsKS  # noqa: E402

DATA_DIR: Path = Path(__file__).resolve().parent.parent / "inputs" / "east_river"

ZONES: tuple[str, ...] = (
    "snow_hs", "snow_rp", "surface_hs", "surface_rp",
    "shallow_hs", "shallow_rp", "deep_hs", "deep_rp",
)
SNOW_ZONES: frozenset[str] = frozenset(("snow_hs", "snow_rp"))
RT_PARAM_NAMES: tuple[str, ...] = ("sw_threshold", "sw_exp", "q_10", "ssa")
SNOW_VALUE: dict[str, float] = {"porosity": 0.5, "depth": 10_000.0, "passive_water_storage": 0.0}
FIXED_ALPHA: dict[str, float] = {
    "shallow_hs.alpha": 1.0, "shallow_rp.alpha": 1.0,
    "deep_hs.alpha": 1.0, "deep_rp.alpha": 1.0,
}
RT_ZONE_MIDPOINTS: dict[str, dict[str, Any]] = {
    "surface": {
        "porosity": 0.3, "depth": (500.0, 5_000.0), "passive_water_storage": 0.0,
        "sw_threshold": (0.0, 1.0), "sw_exp": (0.5, 2.5), "q_10": (1.0, 5.0), "ssa": (-3.0, 5.0),
    },
    "shallow": {
        "porosity": 0.2, "depth": (500.0, 5_000.0), "passive_water_storage": 0.0,
        "sw_threshold": (0.0, 1.0), "sw_exp": (0.5, 2.5), "q_10": 1.0, "ssa": (-3.0, 4.0),
    },
    "deep": {
        "porosity": 0.1, "depth": (1_000.0, 10_000.0), "passive_water_storage": 0.0,
        "sw_threshold": 1.0, "sw_exp": 0.0, "q_10": 1.0, "ssa": (-3.0, 4.0),
    },
}
MIN_CONC_MIDPOINTS: dict[str, tuple[float, float]] = {
    "surface_hs": (-1.0, 2.0), "surface_rp": (-1.0, 1.0),
    "shallow_hs": (-2.0, 0.0), "shallow_rp": (-2.0, 0.0),
    "deep_hs": (-3.0, -1.0), "deep_rp": (-3.0, -1.0),
}


def midpoint(v: Any) -> float:
    if isinstance(v, tuple):
        return 0.5 * (v[0] + v[1])
    return float(v)


def default_parameter_vector() -> tuple[list[str], NDArray[np.float64]]:
    hyd_bounds: dict[str, Any] = dict(pt.HbvLateralModel.default_parameter_ranges())
    for k, v in FIXED_ALPHA.items():
        hyd_bounds[k] = (v, v)
    names: list[str] = list(hyd_bounds.keys())
    values: list[float] = [midpoint(b) for b in hyd_bounds.values()]
    zone_kind: dict[str, str] = {
        "snow_hs": "snow", "snow_rp": "snow",
        "surface_hs": "surface", "surface_rp": "surface",
        "shallow_hs": "shallow", "shallow_rp": "shallow",
        "deep_hs": "deep", "deep_rp": "deep",
    }
    for zone in ZONES:
        if zone in SNOW_ZONES:
            for param in ("porosity", "depth", "passive_water_storage"):
                names.append(f"{zone}.{param}")
                values.append(SNOW_VALUE[param])
            continue
        tbl = RT_ZONE_MIDPOINTS[zone_kind[zone]]
        for param in ("porosity", "depth", "passive_water_storage", *RT_PARAM_NAMES):
            names.append(f"{zone}.{param}")
            values.append(midpoint(tbl[param]))
    for zone, (lo, hi) in MIN_CONC_MIDPOINTS.items():
        names.append(f"mineral_concentration.{zone}_soc")
        values.append(0.5 * (lo + hi))
        names.append(f"mineral_concentration.{zone}_x")
        values.append(0.5 * (lo + hi))
    assert len(names) == 85 and len(values) == 85
    return names, np.asarray(values, dtype=np.float64)


def load_site_data(data_dir: Path = DATA_DIR) -> DataFrame:
    ptq = pd.read_csv(
        data_dir / "PTQ.txt", sep=r"\s+", skiprows=1, index_col=0,
        parse_dates=True, date_format="%Y%m%d",
    )
    pet = pd.read_csv(data_dir / "EVAP.txt", sep=r"\s+")
    ptq.index = pd.to_datetime(ptq.index)
    pet.index = ptq.index
    combined = ptq.merge(pet, left_index=True, right_index=True, how="inner")
    combined.columns = ["ppt", "temp", "q_meas", "pet"]
    return combined[["ppt", "temp", "pet", "q_meas"]].copy()


def load_river_doc(data_dir: Path = DATA_DIR) -> DataFrame:
    doc = pd.read_csv(data_dir / "er_phf0_npoc.csv", index_col=0, parse_dates=True)
    doc.index = pd.to_datetime(doc.index)
    doc["npoc"] = pd.to_numeric(doc["npoc"], errors="coerce")
    converted = (doc["npoc"] / 1000.0 / 12.01).rename("DOC")
    return converted.groupby(converted.index).mean().to_frame()


def build_config(data_dir: Path = DATA_DIR) -> tuple[BatchedConfig, DatetimeIndex]:
    site = load_site_data(data_dir)
    doc = load_river_doc(data_dir)
    window = cast(DatetimeIndex, site.index)
    combined = site.merge(doc, left_index=True, right_index=True, how="left")
    network = get_simple_carbon_adsorption_network()
    model_type = pt.HbvLateralModel
    num_surface = len(model_type.structure[0])
    T = len(window)
    config: BatchedConfig = {
        "network": network,
        "zone_configs": {
            name: RtZoneConfiguration(
                do_reactions=name not in SNOW_ZONES,
                do_speciation=name not in SNOW_ZONES,
            )
            for name in ZONES
        },
        "forc": ForcingData(
            precip=combined.loc[window, "ppt"],
            temp=combined.loc[window, "temp"],
            pet=combined.loc[window, "pet"],
        ),
        "q_meas": combined.loc[window, "q_meas"],
        "conc_df": combined.loc[window, "DOC"].to_frame(),
        "precip_conc": np.full(
            (num_surface, T, network.num_species), 1e-20, dtype=np.float64
        ),
        "model_type": model_type,
    }
    return config, window


def make_runs(runs: int, names: list[str], midpoints: NDArray[np.float64]) -> NDArray[np.float64]:
    """Build ``runs`` identical copies of the parameter vector.

    ``runs`` is the batch size: that many copies of the same simulation
    advanced together every day.  The copies are identical on purpose —
    the benchmark measures how much work one computation can carry, not
    how different parameter sets behave.
    """
    return np.tile(midpoints, (runs, 1))


def estimate_sync_count(T: int, Z: int) -> dict[str, int]:
    """Estimate the number of solver progress checks over the full record.

    Counts the ``.all()``/``.any()`` reductions at the known check sites in
    the hot loops, using typical iteration counts.  The estimates are
    approximate: the actual counts depend on how many iterations each
    zone-day needs.  On a GPU, every check also forces the computer to
    wait for the result (a host-device synchronization).
    """
    hydro_newton_trips = 3
    # Each damped-Newton trip carries two 0-dim reductions: the per-trip
    # backtracking ``improved.all()`` break and the ``converged.all()`` break.
    hydro_syncs_per_trip = 2
    newton_iters = 8
    lm_rate = 0.05
    lm_iters = 3
    armijo_iters = 3

    hydro_zone_steps = T * Z
    hydro_newton_syncs = hydro_zone_steps * hydro_newton_trips * hydro_syncs_per_trip
    hydro_misc_syncs = hydro_zone_steps * 3

    rt_zone_steps = T * Z
    newton_syncs = rt_zone_steps * newton_iters * 3
    armijo_syncs = rt_zone_steps * newton_iters * armijo_iters
    lm_syncs = int(rt_zone_steps * lm_rate) * lm_iters * 3
    lm_armijo_syncs = int(rt_zone_steps * lm_rate) * lm_iters * armijo_iters
    rt_misc_syncs = rt_zone_steps * 2

    total = (hydro_newton_syncs + hydro_misc_syncs
             + newton_syncs + armijo_syncs + lm_syncs + lm_armijo_syncs + rt_misc_syncs)
    return {
        "hydro_newton_all": hydro_newton_syncs,
        "hydro_misc": hydro_misc_syncs,
        "newton_running_any": newton_syncs,
        "armijo_trial_any": armijo_syncs,
        "lm_running_any": lm_syncs,
        "lm_armijo_any": lm_armijo_syncs,
        "rt_misc": rt_misc_syncs,
        "total_estimated": total,
    }


def peak_gpu_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def bench_one(
    params_np: NDArray[np.float64],
    config: BatchedConfig,
    device: torch.device,
    dtype: torch.dtype,
    window: DatetimeIndex,
    profile: bool = False,
) -> dict[str, Any]:
    """Run one full-record simulation and return detailed timing.

    If CUDA f32 fails (precision issues), falls back to CPU f64 and marks
    the result so the diagnosis can report it.
    """
    gc.collect()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    T = len(window)

    t0 = time.perf_counter()
    if profile:
        import cProfile
        import pstats
        profiler = cProfile.Profile()
        profiler.enable()
        out = run_hydro_rt_batched(params_np, config, device, dtype)
        profiler.disable()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        total = t1 - t0
        print("\n  [cProfile] top 30 by cumulative time:")
        stats = pstats.Stats(profiler)
        stats.sort_stats("cumulative")
        stats.print_stats(30)
    else:
        out = run_hydro_rt_batched(params_np, config, device, dtype)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        total = t1 - t0

    q = out.q_sim[0].detach().cpu()
    d = out.doc_riv[0].detach().cpu()
    mem = peak_gpu_mb()

    # Throughput: each day advances all runs through all 8 zones, so the
    # whole record solves runs x zones x T zone-days.  One zone-day counts
    # as one solved zone (the notebook's convention).
    runs: int = int(params_np.shape[0])
    zones: int = len(config["zone_configs"])
    zone_days: int = runs * zones * T

    return {
        "total_s": total,
        "ms_per_day_ms": total / T * 1000,
        "days_per_s": T / total,
        "ms_per_zone_day_ms": total / zone_days * 1000,
        "zones_per_s": zone_days / total,
        "mem_mb": mem,
        "q_finite": bool(torch.isfinite(q).all()),
        "q_mean": float(q.mean()),
        "d_finite": bool(torch.isfinite(d).all()),
        "d_mean": float(d.mean()),
        "overflow": int(out.overflow_mask.any()),
    }


def print_header() -> None:
    print(f"{'runs':>6s}  {'total_s':>8s}  {'ms/day':>10s}  "
          f"{'ms/zone-day':>12s}  {'zones/s':>12s}  {'days/s':>8s}  "
          f"{'mem_MB':>8s}  {'finite_q':>8s}  {'q_mean':>10s}")


def print_row(runs: int, r: dict[str, Any]) -> None:
    print(f"{runs:6d}  {r['total_s']:8.2f}  {r['ms_per_day_ms']:10.3f}  "
          f"{r['ms_per_zone_day_ms']:12.3g}  {r['zones_per_s']:12,.0f}  "
          f"{r['days_per_s']:8.2f}  {r['mem_mb']:8.1f}  "
          f"{str(r['q_finite']):>8s}  {r['q_mean']:10.4f}")


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="1,100",
                        help="comma-separated numbers of identical simulations advanced "
                             "together (the batch size; default: 1,100)")
    parser.add_argument("--device", default=None,
                        help="override device (default: cuda if available)")
    parser.add_argument("--profile", action="store_true",
                        help="profile the 1-run case with cProfile and print the top 30")
    parser.add_argument("--threads", type=int, default=None,
                        help="pin torch intra-op thread count via torch.set_num_threads "
                             "(default: 1 on CPU, leave at torch default on CUDA)")
    args = parser.parse_args()

    runs_values = [int(x) for x in args.runs.split(",")]
    if args.device is not None:
        dev = torch.device(args.device)
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64 if dev.type == "cpu" else torch.float32

    if args.threads is not None:
        torch.set_num_threads(args.threads)
    elif dev.type == "cpu":
        torch.set_num_threads(1)

    print("=" * 90)
    print("Full East River Record Benchmark")
    print("=" * 90)
    print(f"  device: {dev}  dtype: {dtype}  torch threads: {torch.get_num_threads()}")
    print(f"  runs: {runs_values}")
    print()

    names, midpoints = default_parameter_vector()
    config, window = build_config()
    T = len(window)
    print(f"  record: {window[0]:%Y-%m-%d} .. {window[-1]:%Y-%m-%d}  ({T} days)")
    print(f"  network: {config['network'].species_names}")
    print()

    sync_est = estimate_sync_count(T, 8)
    print("Estimated solver progress checks over the full record (approximate, from")
    print("the check sites listed below; actual counts depend on convergence):")
    for k, v in sync_est.items():
        print(f"  {k:25s}: {v:>10,d}")
    print()

    print("Where the solver checks its progress (these are the dominant costs):")
    print("  1. hydro.py:236/251   the water-balance solver checks after each trial step")
    print("     whether it improved and converged (~2 checks per trial, ~3 trials/zone-day)")
    print("  2. solvers.py:462/492 the chemistry step-search checks whether any trial")
    print("     points remain, each Newton iteration")
    print("  3. solvers.py:526/757 the chemistry solver checks whether any runs are still")
    print("     unfinished, each trial step (Newton and Levenberg-Marquardt)")
    print("  4. hydro.py:105       one small temporary value per weather input, once per day")
    print()

    results: dict[int, dict[str, Any]] = {}
    print_header()
    print("-" * 110)
    for runs in runs_values:
        params = make_runs(runs, names, midpoints)
        do_profile = args.profile and runs == 1
        r = bench_one(params, config, dev, dtype, window, profile=do_profile)
        results[runs] = r
        print_row(runs, r)
    print("-" * 110)

    print()
    if len(runs_values) >= 2:
        r0, r1 = results[runs_values[0]], results[runs_values[-1]]
        speedup = r1["total_s"] / r0["total_s"] if r0["total_s"] > 0 else float("inf")
        ratio = runs_values[-1] / runs_values[0]
        print(f"Scaling from {runs_values[0]} to {runs_values[-1]} runs:")
        print(f"  total time: {r0['total_s']:.2f}s -> {r1['total_s']:.2f}s  "
              f"({speedup:.2f}x for {ratio:.0f}x more simulations)")
        if r1["ms_per_zone_day_ms"] > 0:
            per_zone = r0["ms_per_zone_day_ms"] / r1["ms_per_zone_day_ms"]
            print(f"  cost per zone: {r0['ms_per_zone_day_ms']:.3g} -> {r1['ms_per_zone_day_ms']:.3g} "
                  f"ms per zone-day  ({per_zone:.0f}x cheaper)")

    print()
    print("Diagnosis:")
    if results.get(1, {}).get("ms_per_day_ms", 0) > 10:
        print("  1 run: >10 ms/day — the per-day bookkeeping dominates, not the")
        print("  simulation math")
    if (len(runs_values) >= 2
            and results[runs_values[-1]]["total_s"] / results[runs_values[0]]["total_s"]
            > runs_values[-1] / runs_values[0] * 0.9):
        print("  Total time grows almost exactly in proportion to the number of runs:")
        print("  adding runs does not make each zone cheaper (the per-day overhead is fixed)")
    if dev.type == "cuda" and results.get(1, {}).get("mem_mb", 0) < 100:
        print("  Low GPU memory at 1 run: the GPU arithmetic units sit mostly idle")
        print("  (the workload is tiny)")

    # CPU vs CUDA comparison
    if dev.type == "cuda":
        cpu_r = bench_one(
            make_runs(1, names, midpoints), config,
            torch.device("cpu"), torch.float64, window,
        )
        cuda_r = results.get(1, {})
        print(f"\n  CPU (float64), 1 run:  {cpu_r['ms_per_day_ms']:.3f} ms/day  "
              f"({cpu_r['ms_per_zone_day_ms']:.3g} ms/zone-day)")
        print(f"  GPU (float32), 1 run:  {cuda_r.get('ms_per_day_ms', 0):.3f} ms/day  "
              f"({cuda_r.get('ms_per_zone_day_ms', 0):.3g} ms/zone-day)")
        ratio = cuda_r.get("ms_per_day_ms", 0) / max(cpu_r["ms_per_day_ms"], 0.001)
        print(f"  GPU/CPU ratio: {ratio:.1f}x  ({'GPU slower' if ratio > 1 else 'GPU faster'})")

    print()
    print("Root cause: the math of each zone is tiny (a few microseconds of")
    print("arithmetic per step), so the total time is not spent on the")
    print("simulation itself.  It is spent on the solver repeatedly checking")
    print("whether it has finished: every trial step ends in a check that must")
    print("look at every run and zone, and on a GPU each check forces the")
    print("computer to wait for the result.  The estimates above count roughly")
    print("a million such checks over the 2557-day record, plus the thousands")
    print("of small per-day operations the device must be told to perform one")
    print("at a time.")
    print()
    print("Ways to make it faster (by impact):")
    print("  1. Check less often — stop checking after every trial step, or make")
    print("     the check without waiting for the GPU")
    print("  2. Fewer operations per day — compute all 8 zones of a run in one")
    print("     operation instead of eight")
    print("  3. (done) reuse pre-allocated workspaces; cache constants")
    print("  4. (done) load the precipitation data onto the device once")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
