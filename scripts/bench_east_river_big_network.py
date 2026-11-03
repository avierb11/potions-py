"""Throughput benchmark for the East River big-network batched runner.

Measures *model evaluations per second* of the batched multi-lane runner
``potions.model_batched.run_hydro_rt_batched`` for the large-network East
River case: the lateral three-layer structure (6 zones) with the 11-species
SOC / SOC-lz / calcite / XDOC network, over the full 3287-day record
(730-day spin-up + 2557 days of forcing), on the CPU with ALL cores
(float64).  No reduced-thread or GPU variants are run; the measurement is
defined for the full machine.

A *model evaluation* is one full forward run — hydrology plus reactive
transport over the whole record — of the model for one parameter vector.
A *batch size* of N means the batched runner carries N such evaluations in
one computation, one per lane; following the convention of
``bench_full_record.py``, the copies are *identical* on purpose, so the
measurement is about throughput, not about parameter diversity.

For each batch size the script times one full ``run_hydro_rt_batched``
call and reports:

- wall time for the whole batch
- model evaluations per second (batch size / wall time)
- zones solved per second and days simulated per second (batch-wide totals)
- wall time per single evaluation and per simulated day
- peak memory footprint (resident set size)
- output sanity: finite values, mean discharge, overflow lanes

The batched runner also scores per-lane objective functions after the
forward pass (a small Python/pandas loop that scales linearly with the batch
size).  The script times that scoring step on lane 0 after each run and
reports the estimated total, so the headline evaluations/s can be attributed
between the batched simulation and the per-lane scoring.

Usage (from the repository root)::

    .venv-east-river/bin/python scripts/bench_east_river_big_network.py \
        [--batch-sizes 1,10,100,1000] [--days N] [--input CSV] [--params CSV]
"""

from __future__ import annotations

import os

# This machine is shared (a vLLM service on the GPU plus CPU bursts).  The
# benchmark is defined to run on the CPU with ALL cores: pin the BLAS /
# OpenMP pools to the full core count before numpy / torch load, and keep
# CUDA disabled so nothing touches the GPU.
_ALL_CORES: int = int(os.cpu_count() or 1)
os.environ["OMP_NUM_THREADS"] = str(_ALL_CORES)
os.environ["OPENBLAS_NUM_THREADS"] = str(_ALL_CORES)
os.environ["MKL_NUM_THREADS"] = str(_ALL_CORES)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import sys
import time
from pathlib import Path
from typing import Final

import numpy as np
import torch
from numpy.typing import NDArray
from pandas import DataFrame, Series

import potions
from potions import ForcingData, ReactionNetwork, RtZoneConfiguration
from potions.hydro_model import DEFAULT_OBJECTIVE_FUNCTIONS as HYDRO_OBJECTIVE_FUNCTIONS
from potions.model_batched import BatchedConfig, BatchedRunOutput, run_hydro_rt_batched
from potions.objective_functions import DEFAULT_OBJECTIVE_FUNCTIONS as RT_OBJECTIVE_FUNCTIONS

SCRIPTS: Final[Path] = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from east_river_big_network import (  # noqa: E402
    DEFAULT_INPUT,
    DEFAULT_PARAMS_CSV,
    PRECIP_CONC_FILL,
    SPINUP_DAYS,
    ZONE_ORDER,
    build_full_record,
    build_network,
    build_zone_configs,
    describe_record,
    load_final_params,
    load_site_csv,
    remap_parameter_set,
)


# -------------------------------------------------------------------------- #
# Batched setup
# -------------------------------------------------------------------------- #


def build_batched_params(
    model_vector: NDArray[np.float64],
    init_state_vector: NDArray[np.float64],
    network: ReactionNetwork,
    zone_configs: dict[str, RtZoneConfiguration],
) -> NDArray[np.float64]:
    """Expand the forward script's parameter pair onto the batched layout.

    The batched runner expects one flat vector: the hydrological block, the
    RT zone blocks, and an init-state section carrying one log10 mineral
    concentration per mineral of every reacting zone (in zone order)
    followed by one log10 total-site pool per speciating zone.  The forward
    script fixes every mineral at 1.0 mol/L (log10 = 0) and carries only the
    speciating-zone site pools, so the mineral entries are zeros.

    Args:
        model_vector: the 85-value flat model vector (``Model.from_array``
            layout: hydrology + RT zone blocks).
        init_state_vector: the per-speciating-zone log10 total-site pools
            (showcase ``initial_state.<zone>_x`` values).
        network: the reaction network (fixes the number of minerals).
        zone_configs: per-zone RT flags (fixes the number of reacting zones).

    Returns:
        The flat ``(num_hydro + num_rt_zone + num_init)``-value batched
        parameter vector.
    """
    num_react: int = sum(1 for c in zone_configs.values() if c.do_reactions)
    mineral_init: NDArray[np.float64] = np.zeros(
        num_react * network.num_minerals, dtype=np.float64
    )
    return np.concatenate([model_vector, mineral_init, init_state_vector])


def build_config(full: DataFrame, network: ReactionNetwork) -> BatchedConfig:
    """Assemble the batched config for the big-network case.

    Mirrors the forward script's scalar setup: one forcing series
    broadcast to both surface zones, the measured streamflow and river
    concentrations for objective scoring, and trace (1e-20 mol/L)
    precipitation chemistry for every species and surface zone.

    Args:
        full: the (possibly pilot-truncated) full record frame.
        network: the reaction network (fixes the species count).

    Returns:
        The :class:`BatchedConfig` for ``LateralThreeLayerModel``.
    """
    precip_conc: NDArray[np.float64] = np.full(
        (2, len(full), network.num_species), PRECIP_CONC_FILL, dtype=np.float64
    )
    return BatchedConfig(
        network=network,
        zone_configs=build_zone_configs(),
        forc=ForcingData(precip=full["ppt"], temp=full["temp"], pet=full["pet"]),
        q_meas=full["q_meas"],
        conc_df=DataFrame({"DOC": full["DOC"], "Ca++": full["Ca++"]}),
        precip_conc=precip_conc,
        model_type=potions.LateralThreeLayerModel,
    )


# -------------------------------------------------------------------------- #
# Measured run
# -------------------------------------------------------------------------- #


def peak_rss_mib() -> float:
    """Peak resident set size of this process in MiB (Linux: KB in the
    ``ru_maxrss`` resource counter)."""
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def measure_scoring_overhead(
    out: BatchedRunOutput, config: BatchedConfig
) -> float:
    """Time the per-lane objective scoring on lane 0 (seconds per lane).

    The batched runner, after the forward pass, scores each lane's streamflow
    and river DOC with the standard objective functions in a Python loop;
    this replicates that loop for lane 0 so the scoring cost that grows
    linearly with the batch size can be separated from the batched simulation
    cost.
    """
    doc_meas: Series = config["conc_df"]["DOC"]
    q_meas: Series = config["q_meas"]
    meas_q: torch.Tensor = torch.tensor(q_meas.to_numpy(dtype=float))
    doc_sim: Series = Series(
        out.doc_riv[0].detach().cpu().numpy(),
        dtype=float,
        index=doc_meas.index,
    )
    t0: float = time.perf_counter()
    for name, fn in HYDRO_OBJECTIVE_FUNCTIONS:
        fn(meas_q, out.q_sim[0])
    for name, fn in RT_OBJECTIVE_FUNCTIONS:
        fn(doc_meas, doc_sim)
    return time.perf_counter() - t0


def run_batch(
    one: NDArray[np.float64],
    config: BatchedConfig,
    batch_size: int,
    num_days: int,
    num_zones: int,
) -> dict[str, float]:
    """Run the batched runner once for ``batch_size`` identical copies and time it.

    Args:
        one: the single flat batched parameter vector.
        config: the batched config.
        batch_size: number of identical evaluations carried together, one per
            lane.
        num_days: length of the record in days.
        num_zones: number of model zones.

    Returns:
        Measured statistics for the run (see the printed table).
    """
    batch: NDArray[np.float64] = np.tile(one, (batch_size, 1))
    t0: float = time.perf_counter()
    out: BatchedRunOutput = run_hydro_rt_batched(
        batch,
        config,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    elapsed: float = time.perf_counter() - t0
    scoring_per_lane: float = measure_scoring_overhead(out, config)

    q: torch.Tensor = out.q_sim
    doc: torch.Tensor = out.doc_riv
    return {
        "batch_size": batch_size,
        "time_s": elapsed,
        "evals_per_s": batch_size / elapsed,
        "zones_per_s": batch_size * num_days * num_zones / elapsed,
        "days_per_s": batch_size * num_days / elapsed,
        "s_per_eval": elapsed / batch_size,
        "s_per_day": elapsed / num_days,
        "peak_mib": peak_rss_mib(),
        "scoring_s": scoring_per_lane * batch_size,
        "finite_q": bool(torch.isfinite(q).all()),
        "finite_doc": bool(torch.isfinite(doc).all()),
        "mean_q": float(q.mean()),
        "overflow_lanes": int(out.overflow_mask.sum()),
    }


# -------------------------------------------------------------------------- #
# Driver
# -------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv: argument list (defaults to ``sys.argv[1:]``).

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="East River big-network throughput benchmark "
        "(model evaluations per second at a batch size, CPU, all cores)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="forcing + observation CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=DEFAULT_PARAMS_CSV,
        help="parameter CSV to remap (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,10,100,1000",
        help="comma-separated batch sizes (default: %(default)s)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=0,
        help="pilot: truncate the forcing window to the first N days "
        "(0 = full record)",
    )
    return parser.parse_args(argv)


def print_row(r: dict[str, float]) -> None:
    """Print one batch-size result row (aligned with the printed header)."""
    ok: str = "ok" if (r["finite_q"] and r["finite_doc"]) else "NOT-FINITE"
    print(
        f"  batch_size={int(r['batch_size']):5d}  time={r['time_s']:8.1f}s  "
        f"evals/s={r['evals_per_s']:9.4f}  zones/s={r['zones_per_s']:10.1f}  "
        f"days/s={r['days_per_s']:7.3f}  s/eval={r['s_per_eval']:7.2f}  "
        f"s/day={r['s_per_day']:8.4f}  peak={r['peak_mib']:7.0f}MiB  "
        f"mean_q={r['mean_q']:7.4f}  overflow={int(r['overflow_lanes'])}  "
        f"[{ok}]",
        flush=True,
    )
    print(
        f"           (of which per-lane objective scoring ~ "
        f"{r['scoring_s']:.1f} s, {100.0 * r['scoring_s'] / r['time_s']:.1f}% "
        f"of the wall time)",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark for every requested batch size.

    Args:
        argv: command line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 on success).
    """
    args: argparse.Namespace = parse_args(argv)
    batch_sizes: list[int] = [
        int(v) for v in args.batch_sizes.split(",") if v.strip()
    ]

    # All cores, as requested: PyTorch's OpenMP/MKL pool otherwise settles
    # on the physical core count; pin it explicitly before any tensor work.
    torch.set_num_threads(_ALL_CORES)
    print(f"CPU: {os.cpu_count()} cores (torch intra-op = "
          f"{torch.get_num_threads()})")
    if torch.get_num_threads() != _ALL_CORES:
        raise RuntimeError(
            f"torch intra-op threads = {torch.get_num_threads()}, "
            f"expected {_ALL_CORES} (all cores)"
        )

    print(f"[1/4] Loading forcing + observations: {args.input}")
    raw: DataFrame = load_site_csv(args.input)
    full_record: DataFrame = build_full_record(raw)
    if args.days > 0:
        full: DataFrame = full_record.iloc[: SPINUP_DAYS + args.days]
        print(f"  PILOT: forcing window truncated to the first {args.days} days")
    else:
        full = full_record
    describe_record(full, SPINUP_DAYS)

    print("[2/4] Assembling the reaction network + parameter set")
    network: ReactionNetwork = build_network()
    params: dict[str, float] = load_final_params(args.params)
    model_vector, init_state_vector = remap_parameter_set(params)
    zone_configs: dict[str, RtZoneConfiguration] = build_zone_configs()
    one: NDArray[np.float64] = build_batched_params(
        model_vector, init_state_vector, network, zone_configs
    )
    print(
        f"  {network.num_species} species; batched flat vector = "
        f"{one.shape[0]} values "
        f"(19 hydro + 66 RT + 16 init-state)"
    )

    print(f"[3/4] Building the batched config ({len(full)} days)")
    config: BatchedConfig = build_config(full, network)
    num_days: int = len(full)
    num_zones: int = len(ZONE_ORDER)

    print(f"[4/4] Batched runs on CPU (all {_ALL_CORES} cores, float64)")
    print(
        f"  {'batch_size':>16s}  {'time':>9s}  {'evals/s':>9s}  {'zones/s':>10s}  "
        f"{'days/s':>7s}  {'s/eval':>7s}  {'s/day':>8s}  {'peak':>9s}  "
        f"{'mean_q':>7s}  {'overflow':>8s}  status"
    )
    results: list[dict[str, float]] = []
    for batch_size in batch_sizes:
        print(f"  batch_size={batch_size} ...", flush=True)
        r: dict[str, float] = run_batch(one, config, batch_size, num_days, num_zones)
        print_row(r)
        results.append(r)

    print("\nSummary (model evaluations per second, identical copies):")
    for r in results:
        sim_only: float = r["time_s"] - r["scoring_s"]
        forward_evals: float = (
            r["batch_size"] / sim_only if sim_only > 0.0 else float("nan")
        )
        print(
            f"  batch_size={int(r['batch_size']):5d}: "
            f"{r['evals_per_s']:.4f} evals/s overall; "
            f"forward-only estimate {forward_evals:.4f} evals/s "
            f"(one evaluation = {num_days} days x {num_zones} zones)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
