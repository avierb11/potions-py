"""Profiling script for the batched ``K``-lane East River runner.

Measures end-to-end wall time of :func:`potions.run_hydro_rt_batched` over a
ladder of lane counts ``K`` (1, 2, 4, ..., ``--kmax``, default hundreds) against
the **full East River record** (all water years, ~2557 days) and reports the
throughput in **simulations/sec** at each batch size.  Use ``--start``/``--end``
or ``--days`` to target a sub-window instead.

Run on CUDA with float32 by default (the project precision policy in
``potions.device``); pass ``--device cpu`` for the float64 CPU path.  The
sequential scalar baseline (``utils.calibration.arr_to_results``) costs roughly
0.015 s/day/lane, i.e. ~40 s per full-record lane, so by default it is measured
for a single lane only (``--scalar-lanes 1``) to give the speedup context
without dominating the benchmark.

Usage::

    python scripts/profile_batched.py                    # GPU f32, K ladder to 256
    python scripts/profile_batched.py --kmax 512         # even deeper ladder
    python scripts/profile_batched.py --device cpu       # CPU f64 configuration
    python scripts/profile_batched.py --profile-k 256    # cProfile a K=256 run
    python scripts/profile_batched.py --out times.csv    # save the table

Parameter lanes are drawn from the inner 60% of every bound (the same
``aim_mid`` sampling as ``parity_check_batched.py``) so lanes stay clear of the
storage-overflow regime that trips the scalar path; any batched lane that does
touch the zone maximum water volume is counted (its ``overflow_mask`` bit) and
reported rather than aborting the run.
"""

from __future__ import annotations

import argparse
import cProfile
import os
import pstats
import sys
import time
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from pandas import DataFrame, DatetimeIndex

SCRIPTS: str = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)

import east_river_forward as fwd  # noqa: E402  (registers showcase paths)

import potions as pt  # noqa: E402  # type: ignore[import-not-found]
from potions.device import default_dtype  # noqa: E402
from potions.model_batched import BatchedConfig  # noqa: E402  # type: ignore[import-not-found]
from utils.calibration import arr_to_results  # noqa: E402  # type: ignore[import-not-found]
from utils.parameter_ranges import load_input_data  # noqa: E402  # type: ignore[import-not-found]
from data.raw.east_river import hbv_data, load_all_doc_data  # noqa: E402  # type: ignore[import-not-found]


def build_config(args: argparse.Namespace) -> tuple[DatetimeIndex, BatchedConfig]:
    """Assemble the East River calibration config for the requested window.

    Defaults to the **full record** (all water years in the site data); the CLI
    ``--start``/``--end`` or ``--days`` flags override that choice.
    """
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
    window: DatetimeIndex
    if args.start is not None and args.end is not None:
        window = fwd.choose_window(dates, wy=None, start=args.start, end=args.end, days=None)
    elif args.days is not None:
        window = fwd.choose_window(dates, wy=None, start=None, end=None, days=args.days)
    else:
        window = fwd.choose_window(dates, wy=None, start=None, end=None, days=None, full=True)
    return window, cast(
        BatchedConfig,
        fwd.build_window_config(inputs.cal_config, site_data, doc_df, window),
    )


def load_bounds() -> dict[str, tuple[float, float]]:
    """Load the parameter bounds dict (inner-60% sampling needs the ranges)."""
    inputs = load_input_data(
        pt.HbvLateralModel, test_wys=[2017, 2020], val_wy=2018, site="EastRiver"
    )
    return inputs.bounds


def aim_mid(k: int, bounds: dict[str, tuple[float, float]]) -> np.ndarray:
    """Draw ``k`` parameter vectors inside the inner 60% of every bound."""
    out: list[float] = []
    for nm in bounds:
        lo: float
        hi: float
        lo, hi = bounds[nm]
        span: float = hi - lo
        out.append(float(np.random.uniform(lo + 0.2 * span, lo + 0.8 * span)))
    return np.stack([np.asarray(out, dtype=float) for _ in range(k)], axis=0)


@dataclass(frozen=True)
class Timing:
    """Wall-time measurement for one lane count."""

    k: int
    batched_best: float
    batched_median: float
    scalar: float
    overflow: int


def warmup(params: np.ndarray, cfg: BatchedConfig, device: torch.device, dtype: torch.dtype) -> None:
    """Run one batched forward to initialize the runtime (CUDA context, kernels)."""
    pt.run_hydro_rt_batched(params, cfg, device, dtype)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_batched(
    params: np.ndarray,
    cfg: BatchedConfig,
    reps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[float, float, int]:
    """Return ``(best, median, overflow_count)`` for repeated batched runs.

    The median is robust to runtime-contention outliers (OpenMP scheduling on
    CPU, GPU clock/thermal state).  On CUDA the clock is synchronized after each
    run so launched kernels are fully accounted for.  ``overflow_count`` is the
    number of lanes whose storage touched the zone maximum water volume (the
    ``overflow_mask`` bits).
    """
    samples: list[float] = []
    overflow: int = 0
    for _ in range(reps):
        t0: float = time.perf_counter()
        out: pt.BatchedRunOutput = pt.run_hydro_rt_batched(
            params, cfg, device, dtype
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        samples.append(time.perf_counter() - t0)
        overflow = max(overflow, int(torch.sum(out.overflow_mask).item()))
    samples_sorted: list[float] = sorted(samples)
    mid: int = reps // 2
    median: float = (
        samples_sorted[mid]
        if reps % 2 == 1
        else 0.5 * (samples_sorted[mid - 1] + samples_sorted[mid])
    )
    return samples_sorted[0], median, overflow


def time_scalar(params: np.ndarray, cfg: BatchedConfig) -> float:
    """Run ``arr_to_results`` lane-by-lane; return total wall seconds."""
    t0: float = time.perf_counter()
    for k in range(int(params.shape[0])):
        res = arr_to_results(params[k], cfg, hydro_criteria=None, verbose=False)  # type: ignore[arg-type]
        if isinstance(res, pt.HydroModelResults):
            print(f"      lane {k}: hydrology-only result (no RT completion)")
    return time.perf_counter() - t0


def build_ladder(kmax: int) -> list[int]:
    """Lane counts to profile: 1, 2, 4, ..., kmax."""
    out: list[int] = []
    k: int = 1
    while k <= kmax:
        out.append(k)
        k *= 2
    return out


def profile_single_k(
    k: int, params: np.ndarray, cfg: BatchedConfig, device: torch.device, dtype: torch.dtype
) -> None:
    """cProfile one batched run and print the cumulative-time report."""
    print(f"\n== cProfile of batched K={k} ({device}/{dtype}) (top 30 cumulative) ==")
    pr: cProfile.Profile = cProfile.Profile()
    pr.enable()
    pt.run_hydro_rt_batched(params, cfg, device, dtype)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    pr.disable()
    pstats.Stats(pr).sort_stats("cumulative").print_stats(30)


def format_table(timings: list[Timing]) -> str:
    """Render the timing table as monospace text."""
    header: str = (
        f"{'K':>3}  {'wall (s)':>10}  {'sims/s':>10}  {'per-sim (ms)':>12}  "
        f"{'scalar (s)':>11}  {'speedup':>8}  {'ovf':>4}"
    )
    lines: list[str] = [header, "-" * len(header)]
    for t in timings:
        throughput: float = t.k / t.batched_median
        per_sim_ms: float = 1000.0 * t.batched_median / t.k
        scalar_txt: str = f"{t.scalar:>11.4f}" if t.scalar else f"{'--':>11}"
        speedup_txt: str = f"{t.scalar / t.batched_median:>8.2f}" if t.scalar else f"{'--':>8}"
        lines.append(
            f"{t.k:>3}  {t.batched_median:>10.3f}  {throughput:>10.2f}  "
            f"{per_sim_ms:>12.2f}  {scalar_txt}  {speedup_txt}  {t.overflow:>4}"
        )
    return "\n".join(lines)


def resolve_device(args: argparse.Namespace) -> tuple[torch.device, torch.dtype]:
    """Resolve the profiling device/dtype from CLI flags.

    ``--device`` selects CUDA (default when available, else CPU).  ``--precision
    auto`` (the default) applies the project policy — float32 on CUDA, float64 on
    CPU (``potions.device.default_dtype``).
    """
    if args.device == "cuda" and not torch.cuda.is_available():
        print("   cuda requested but unavailable; falling back to cpu")
        args.device = "cpu"
    device: torch.device = torch.device(args.device)
    dtype: torch.dtype
    if args.precision == "auto":
        dtype = default_dtype(device)
    elif args.precision == "f32":
        dtype = torch.float32
    else:
        dtype = torch.float64
    return device, dtype


def run(args: argparse.Namespace) -> int:
    """Build config, sample lanes, time the K ladder and print results."""
    device: torch.device
    dtype: torch.dtype
    device, dtype = resolve_device(args)

    print("== Building East River window config ==")
    window: DatetimeIndex
    cfg: BatchedConfig
    window, cfg = build_config(args)
    print(f"   window {window[0]} .. {window[-1]}  ({len(window)} days)")
    print(f"   device: {device}  dtype: {dtype}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ladder: list[int] = (
        [int(s) for s in args.sizes.split(",")] if args.sizes else build_ladder(args.kmax)
    )
    # One shared lane pool per K (drawn once per K so lane identity is stable).
    lanes_pool: np.ndarray = aim_mid(args.kmax, load_bounds())

    # Warm the runtime (CUDA context / cuBLAS handles / kernel caches) once.
    warmup(lanes_pool[:8], cfg, device, dtype)
    print("   warmup run done")

    print("\n== Timing ==")
    timings: list[Timing] = []
    for k in ladder:
        params: np.ndarray = lanes_pool[:k]
        best, median, overflow = time_batched(params, cfg, args.reps, device, dtype)
        scalar: float = 0.0
        if args.full_scalar or (args.scalar_lanes is not None and k <= args.scalar_lanes):
            scalar = time_scalar(params, cfg)
        timings.append(
            Timing(k=k, batched_best=best, batched_median=median, scalar=scalar, overflow=overflow)
        )
        ovf_note: str = f"  overflow={overflow}" if overflow else ""
        print(
            f"   K={k:>3}:  median={median:.3f}s  {k / median:>8.2f} sims/s  "
            f"(best {best:.3f}s)  scalar={scalar:.4f}s{ovf_note}"
        )

    print(f"\n{format_table(timings)}")

    if args.out:
        rows: list[dict[str, Any]] = [
            {
                "k": t.k,
                "batched_best_s": t.batched_best,
                "batched_median_s": t.batched_median,
                "scalar_s": t.scalar,
                "per_lane_ms": 1000.0 * t.batched_best / t.k,
                "throughput_lanes_per_s": t.k / t.batched_best,
                "overflow": t.overflow,
            }
            for t in timings
        ]
        DataFrame(rows).to_csv(args.out, index=False)
        print(f"\n   table saved: {args.out}")

    if args.profile_k is not None:
        params: np.ndarray = lanes_pool[: args.profile_k]
        profile_single_k(args.profile_k, params, cfg, device, dtype)

    return 0


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Profile the batched K-lane East River runner."
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="compute device (cuda default when available; scalar baseline is CPU)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["auto", "f32", "f64"],
        default="auto",
        help="dtype policy: auto = f32 on cuda, f64 on cpu (project default)",
    )
    parser.add_argument(
        "--sizes", type=str, default=None,
        help="comma-separated K ladder (overrides --kmax); e.g. 32,256"
    )
    parser.add_argument(
        "--kmax", type=int, default=256, help="largest lane count (power of 2)"
    )
    parser.add_argument("--reps", type=int, default=3, help="batched repeats per K")
    parser.add_argument(
        "--start", type=str, default=None, help="ISO start date of a sub-window"
    )
    parser.add_argument(
        "--end", type=str, default=None, help="ISO end date of a sub-window"
    )
    parser.add_argument(
        "--days", type=int, default=None, help="trailing days of a sub-window"
    )
    parser.add_argument("--scalar-lanes", type=int, default=1,
        help="run the ~0.015 s/day/lane scalar baseline only for K <= this (0 disables)")
    parser.add_argument(
        "--full-scalar",
        action="store_true",
        help="run the scalar baseline at every K (very slow on the full record)",
    )
    parser.add_argument("--profile-k", type=int, default=None, help="cProfile one K")
    parser.add_argument("--seed", type=int, default=7, help="RNG seed")
    parser.add_argument("--out", type=str, default=None, help="CSV output path")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())