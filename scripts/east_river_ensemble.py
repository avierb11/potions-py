"""Run the full East River simulation as a parallel ensemble of parameter sets.

Every row of the input is one parameter set: the hydrological parameters
(HBV / groundwater zones of :class:`potions.HbvLateralModel`) plus the
reactive-transport parameters for the simple dissolved-organic-carbon network
``DOC / SOC(s) / X- / XDOC``).  All rows are forwarded together through the
batched hydrology + reactive-transport runner (``potions.run_hydro_rt_batched``
— one batched call, not a Python loop), over the full multi-year East River
record (2014-10-01 .. 2021-09-30, ~2557 daily steps), and the per-row
streamflow / river-DOC time series come back as one ensemble.

The input is a :class:`pandas.DataFrame` whose columns are parameter names
(the 85 calibration-order columns; see :func:`param_matrix`).  Any column a
row does not supply falls back to the midpoint of that parameter's
calibration bound, so a row can specify a few hydrological values and a few
reactive-transport values and let everything else default.  ``--sample N``
draws ``N`` random sets uniformly inside the bounds.

Usage from Python::

    import east_river_ensemble as ens
    res = ens.run_ensemble(param_sets)   # DataFrame of parameter sets
    res.q_sim      # (batch_size, T) streamflow [mm/day]
    res.doc_riv    # (batch_size, T) river DOC [mmol/L]
    res.rt_objectives
    ens.save_ensemble(res, "my_ensemble.pkl")
    res = ens.load_ensemble("my_ensemble.pkl")

Usage from the command line::

    python scripts/east_river_ensemble.py --csv my_param_sets.csv
    python scripts/east_river_ensemble.py --sample 50 --seed 1
    python scripts/east_river_ensemble.py --csv ps.csv --device cuda --dtype float32
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from pandas import DataFrame, DatetimeIndex, Series
import torch

# The showcase ships the loaders and parameter bounds; reach it while
# ``import potions`` still resolves to the (editable) potions-py install.
SHOWCASE_DIR: str = "/home/andrew/Documents/Research/Projects/potions-showcase"
sys.path.insert(0, SHOWCASE_DIR)
os.chdir(SHOWCASE_DIR)

import potions as pt  # noqa: E402  (resolves to potions-py)  # type: ignore[import-not-found]
from utils.parameter_ranges import load_input_data  # noqa: E402  # type: ignore[import-not-found]
from data.raw.east_river import hbv_data, load_all_doc_data  # noqa: E402  # type: ignore[import-not-found]

import east_river_forward as fwd  # noqa: E402  (registers the showcase config helpers)

OUTPUT_DIR: str = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "outputs")
)

#: Hydro + RT parameter count for the East River / HbvLateralModel layout.
NUM_PARAMS: int = 85

#: Species whose concentration is tracked as the river-DOC output.
DOC_SPECIES: str = "DOC"


@dataclass(frozen=True)
class EnsembleResults:
    """Ensemble of simulation results, one entry per input parameter set.

    Attributes:
        params:           canonical ``(batch_size, NUM_PARAMS)`` parameter
            frame, in calibration-order columns (missing columns back-filled to
            the bound midpoints).  Index matches the successful input rows.
        window:           the simulated dates (full multi-year record by
            default).
        q_sim:            ``(batch_size, T)`` simulated streamflow [mm/day].
        doc_riv:          ``(batch_size, T)`` flow-weighted river DOC [mmol/L].
        hydro_objectives: per-set hydrology objective scores (KSE/NSE/...).
        rt_objectives:    per-set river-DOC objective scores.
        overflow:         ``(batch_size,)`` ``True`` for sets whose water
            storage touched a zone's maximum volume (those rows hit the
            capacity limit; the run itself completes).
        failed:           input row positions whose reactive-transport solve
            did not converge (Newton + Levenberg-Marquardt).  Such rows are
            dropped from the ensemble; the run itself completes.
    """

    params: DataFrame
    window: DatetimeIndex
    q_sim: NDArray[np.float64]
    doc_riv: NDArray[np.float64]
    hydro_objectives: DataFrame
    rt_objectives: DataFrame
    overflow: NDArray[np.bool_]
    failed: list[int] = field(default_factory=list)

    @property
    def n_sets(self) -> int:
        """Number of parameter sets in the ensemble."""
        return int(self.params.shape[0])


def save_ensemble(results: EnsembleResults, path: str) -> None:
    """Persist an ensemble to ``path`` as a portable pickle.

    The payload is a plain dict of the results' fields (DataFrames, numpy
    arrays, a DatetimeIndex), so the pickle loads in any process without
    needing this module.
    """
    with open(path, "wb") as f:
        pickle.dump(
            {
                "params": results.params,
                "window": results.window,
                "q_sim": results.q_sim,
                "doc_riv": results.doc_riv,
                "hydro_objectives": results.hydro_objectives,
                "rt_objectives": results.rt_objectives,
                "overflow": results.overflow,
                "failed": results.failed,
            },
            f,
        )


def load_ensemble(path: str) -> EnsembleResults:
    """Load an ensemble saved by :func:`save_ensemble`."""
    with open(path, "rb") as f:
        data: dict[str, object] = pickle.load(f)
    return EnsembleResults(**data)  # type: ignore[arg-type]


def river_doc_series() -> DataFrame:
    """Load and unit-convert the measured river-DOC record (mmol/L).

    The raw ``npoc`` concentration is ng/kg -> mg/L -> mmol/L by ``/1000``
    (mg) then ``/12.01`` (molar mass of carbon).
    """
    doc_df: DataFrame = (
        load_all_doc_data()
        .loc[lambda d: d.station == "er_phf0", "npoc"]
        .rename("DOC")
        .to_frame()
        / 1000.0
        / 12.01
    )
    return doc_df.groupby(doc_df.index).mean()


def calibration_inputs() -> tuple[dict[str, tuple[float, float]], dict[str, object]]:
    """Load the showcase calibration bounds and the base model config."""
    inputs = load_input_data(
        pt.HbvLateralModel, test_wys=[2017, 2020], val_wy=2018, site="EastRiver"
    )
    return inputs.bounds, inputs.cal_config


def build_full_config() -> tuple[dict[str, tuple[float, float]], Any, DatetimeIndex]:
    """Assemble the full-record East River config and its date window."""
    bounds, base_cfg = calibration_inputs()
    site_data: DataFrame = hbv_data.load_site_data("EastRiver")
    window: DatetimeIndex = cast(DatetimeIndex, site_data.index)
    cfg: Any = fwd.build_window_config(base_cfg, site_data, river_doc_series(), window)
    return bounds, cfg, window


def param_matrix(
    param_sets: DataFrame, bounds: dict[str, tuple[float, float]]
) -> NDArray[np.float64]:
    """Map a DataFrame of parameter sets to a ``(batch_size, NUM_PARAMS)`` array.

    Columns are matched by name (calibration-order names such as
    ``surface_hs.k0``, ``surface_hs.q_10``, ``deep_rp.depth``).  Any column
    the frame does not provide is filled with the midpoint of its calibration
    bound, so a row can rerun with just a handful of changed parameters.

    Args:
        param_sets: one row per parameter set; columns are parameter names.
        bounds:     parameter-name -> (min, max) bounds in calibration order.

    Returns:
        The ``(batch_size, NUM_PARAMS)`` float64 array in calibration column
        order.
    """
    names: list[str] = list(bounds.keys())
    midpoints: Series = Series(
        [0.5 * (lo + hi) for lo, hi in bounds.values()], index=names, dtype=float
    )
    canonical: DataFrame = param_sets.reindex(columns=names).fillna(midpoints)
    return canonical[names].to_numpy(dtype=np.float64)


def sample_parameters(n: int, bounds: dict[str, tuple[float, float]], seed: int) -> DataFrame:
    """Draw ``n`` random parameter sets uniformly inside every bound."""
    rng: np.random.Generator = np.random.default_rng(seed)
    names: list[str] = list(bounds.keys())
    lows: NDArray[np.float64] = np.asarray([lo for lo, _ in bounds.values()])
    highs: NDArray[np.float64] = np.asarray([hi for _, hi in bounds.values()])
    draws: NDArray[np.float64] = lows + rng.random((n, len(names))) * (highs - lows)
    return DataFrame(draws, columns=names)


def _bisect_rows(
    params: NDArray[np.float64],
    cfg: Any,
    device: torch.device,
    dtype: torch.dtype,
    init_aq_conc: float,
    lo: int,
    hi: int,
) -> tuple[list[tuple[int, Any]], list[int]]:
    """Run rows ``[lo, hi)``, bisecting sub-batches whose solve fails.

    Some parameter sets (especially extreme random draws) make the
    reactive-transport Newton + LM solve fail, which aborts the whole batched
    call.  Instead of dying, the failing sub-batch is split in half and each
    half retried, so the convergent rows still run batched and only the
    genuinely unsolvable rows are dropped.

    Args:
        params:       ``(batch_size, NUM_PARAMS)`` parameter array.
        cfg:          batched config.
        device:       compute device.
        dtype:        compute dtype.
        init_aq_conc: initial concentration of non-DOC aqueous species.
        lo:           first row index.
        hi:           one past the last row index.

    Returns:
        ``(slices, failed)`` where ``slices`` is a list of ``(start, output)``
        in row order and ``failed`` holds the dropped row positions.
    """
    if lo >= hi:
        return [], []
    try:
        out: Any = pt.run_hydro_rt_batched(
            params[lo:hi], cfg, device, dtype, init_aq_conc=init_aq_conc
        )
        return [(lo, out)], []
    except RuntimeError:
        if hi - lo == 1:
            return [], [lo]
        mid: int = (lo + hi) // 2
        wins_lo, fails_lo = _bisect_rows(params, cfg, device, dtype, init_aq_conc, lo, mid)
        wins_hi, fails_hi = _bisect_rows(params, cfg, device, dtype, init_aq_conc, mid, hi)
        return wins_lo + wins_hi, fails_lo + fails_hi


def _concat_slices(slices: list[tuple[int, Any]]) -> Any:
    """Stitch bisection outputs back into one :class:`BatchedRunOutput`."""
    if not slices:
        raise RuntimeError("no parameter set converged")
    return pt.BatchedRunOutput(
        q_sim=torch.cat([out.q_sim for _, out in slices], dim=0),
        doc_riv=torch.cat([out.doc_riv for _, out in slices], dim=0),
        hydro_objectives=[o for _, out in slices for o in out.hydro_objectives],
        rt_doc_objectives=[o for _, out in slices for o in out.rt_doc_objectives],
        overflow_mask=torch.cat([out.overflow_mask for _, out in slices], dim=0),
    )


def run_ensemble(
    param_sets: DataFrame,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    init_aq_conc: float = 1e-3,
) -> EnsembleResults:
    """Run the full-record East River simulation for every parameter set.

    All parameter sets are solved together in one batched call over the whole
    multi-year record, so every row shares the same forcing and the ensemble
    of streamflow / river-DOC series is directly comparable across rows.

    Args:
        param_sets:     one row per parameter set; columns are parameter
            names (arbitrary subset allowed, missing ones default to the
            midpoints).
        device:         compute device (defaults to CPU; pass CUDA to opt
            into the GPU, which is faster for big ensembles but contends with
            whatever else is running on the card).
        dtype:          compute dtype (defaults to ``float64`` on CPU and
            ``float32`` on CUDA).
        init_aq_conc:   initial concentration of non-DOC aqueous species.

    Returns:
        An :class:`EnsembleResults` with ``(batch_size, T)`` streamflow and
        river-DOC series, per-set objectives and overflow flags.  Rows whose
        reactive-transport solve does not converge are dropped and listed in
        ``failed``.
    """
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.float32 if device.type == "cuda" else torch.float64

    bounds, cfg, window = build_full_config()
    params: NDArray[np.float64] = param_matrix(param_sets, bounds)
    batch_size: int = int(params.shape[0])
    T: int = len(window)

    started: float = time.perf_counter()
    try:
        out = pt.run_hydro_rt_batched(params, cfg, device, dtype, init_aq_conc=init_aq_conc)
        failed: list[int] = []
    except RuntimeError:
        print(
            f"   batch solve failed; bisecting over the {batch_size} rows ..."
        )
        wins: list[tuple[int, Any]]
        wins, failed = _bisect_rows(
            params, cfg, device, dtype, init_aq_conc, 0, batch_size
        )
        out = _concat_slices(wins)
    wall: float = time.perf_counter() - started

    num_good: int = int(out.q_sim.shape[0])
    labels_: pd.Index = (
        param_sets.index
        if not param_sets.index.duplicated().any()
        else pd.RangeIndex(batch_size)
    )
    failed_labels: list[Any] = [labels_[i] for i in failed]
    good: NDArray[np.int64] = np.array(
        [i for i in range(batch_size) if i not in set(failed)]
    )
    params_good: NDArray[np.float64] = params[good]

    q_sim: NDArray[np.float64] = out.q_sim.detach().cpu().numpy()
    doc_riv: NDArray[np.float64] = out.doc_riv.detach().cpu().numpy()
    assert q_sim.shape == (num_good, T) and doc_riv.shape == (num_good, T)

    index: pd.Index = labels_[good]
    canonical: DataFrame = DataFrame(
        params_good, index=index, columns=list(bounds.keys())
    )
    hydro_obj: DataFrame = DataFrame(out.hydro_objectives, index=index)
    rt_obj: DataFrame = DataFrame(out.rt_doc_objectives, index=index)

    results: EnsembleResults = EnsembleResults(
        params=canonical,
        window=window,
        q_sim=q_sim,
        doc_riv=doc_riv,
        hydro_objectives=hydro_obj,
        rt_objectives=rt_obj,
        overflow=out.overflow_mask.detach().cpu().numpy().astype(bool),
        failed=failed_labels,
    )
    if failed_labels:
        print(f"   dropped {len(failed_labels)} non-converging row(s): {failed_labels}")
    print(
        f"ran {num_good}/{batch_size} parameter set(s) over {T} days "
        f"on {device} ({dtype}) in "
        f"{wall / max(num_good, 1) * 1000:.0f} ms/set total {wall:.1f}s"
    )
    return results


def _default_output_path() -> str:
    """A timestamped ensemble pickle in the outputs directory."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return os.path.join(OUTPUT_DIR, f"east_river_ensemble_{datetime.now():%Y%m%d_%H%M%S}.pkl")


def main() -> int:
    """Command-line entry point: sample or load parameter sets, run, persist."""
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--csv", default=None, help="CSV of parameter sets "
                     "(one row per set, any subset of the 85 parameter columns)")
    src.add_argument("--sample", type=int, default=None, help="draw N random "
                     "parameter sets uniformly inside the calibration bounds")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for --sample")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                        help="compute device (default: cpu; pass cuda to opt into the GPU)")
    parser.add_argument("--dtype", default=None, choices=["float32", "float64"],
                        help="compute dtype (default: float32 on cuda, float64 on cpu)")
    parser.add_argument("--out", default=None, help="pickle output path")
    args: argparse.Namespace = parser.parse_args()

    if args.csv is None and args.sample is None:
        parser.error("provide --csv PATH or --sample N")

    bounds, _ = calibration_inputs()
    if args.csv is not None:
        param_sets: DataFrame = pd.read_csv(args.csv)
        print(f"loaded {len(param_sets)} parameter set(s) from {args.csv}")
    else:
        param_sets = sample_parameters(args.sample, bounds, args.seed)
        print(f"sampled {len(param_sets)} parameter set(s) inside the bounds")

    device: torch.device = torch.device(args.device)
    dtype: torch.dtype | None = torch.float32 if args.dtype == "float32" else (
        torch.float64 if args.dtype == "float64" else None
    )

    results: EnsembleResults = run_ensemble(param_sets, device=device, dtype=dtype)

    # ---- Report ----
    print("\n== Ensemble summary ==")
    print(f"   sets:         {results.n_sets}")
    print(f"   window:       {results.window[0]:%Y-%m-%d} .. {results.window[-1]:%Y-%m-%d}  ({len(results.window)} days)")
    q: NDArray[np.float64] = results.q_sim
    d: NDArray[np.float64] = results.doc_riv
    print(f"   streamflow:   mean={float(q.mean()):.4f}  min={float(q.min(axis=1).mean()):.4f}  max={float(q.max()):.4f} mm/day")
    print(f"   river DOC:    mean={float(d.mean()):.5f}  min={float(d.min()):.5f}  max={float(d.max()):.5f} mmol/L")
    print(f"   overflow sets: {int(results.overflow.sum())}/{results.n_sets}")
    print(f"   hydro objectives (first set): {results.hydro_objectives.iloc[0].to_dict()}")
    print(f"   RT objectives   (first set): {results.rt_objectives.iloc[0].to_dict()}")

    out_path: str = args.out or _default_output_path()
    save_ensemble(results, out_path)
    print(f"\n   ensemble saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())