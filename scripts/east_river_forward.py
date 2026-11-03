"""Run a single forward hydrology + reactive-transport pass over the East
River dataset, using the Python reimplementation (potions-py) as a drop-in
replacement for the compiled ``potions`` package.

This is a *sanity check*, not a calibration and not a numerical golden run.
It drives the potions-showcase pipeline end-to-end (data loader, parameter
bounds, the showcase ``arr_to_results``) against potions-py and asserts that
the outputs are numerically sane (finite, non-negative streamflow, finite,
positive river DOC) rather than that they are correct to the golden binaries.

The parameter vector is sourced from a real differential-evolution output row
so the model exercises realistic hydrology / reactive-transport parameters;
any parameter not present in that row (the ``_x`` exchange-site initials under
the simple carbon-adsorption network) falls back to the midpoint of its bound.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from typing import cast

import numpy as np
import pandas as pd
from pandas import DataFrame, DatetimeIndex, Series, Timestamp

# The showcase ships the data loaders and the canonical
# ``arr_to_results`` machinery. Add it to the front of ``sys.path`` so that
# ``data`` / ``utils`` resolve there while ``import potions`` still resolves
# to the (editable) potions-py installation.
SHOWCASE_DIR: str = "/home/andrew/Documents/Research/Projects/potions-showcase"
sys.path.insert(0, SHOWCASE_DIR)
os.chdir(SHOWCASE_DIR)

import potions as pt  # noqa: E402  (resolves to potions-py)  # type: ignore[import-not-found]
import potions.objective_functions as obj  # noqa: E402  # type: ignore[import-not-found]
from utils.calibration import arr_to_results  # noqa: E402  # type: ignore[import-not-found]
from utils.parameter_ranges import (  # noqa: E402  # type: ignore[import-not-found]
    CalibrationData,
    get_precip_conc,
    load_input_data,
    _get_wy,
)
from data.raw.east_river import hbv_data, load_all_doc_data  # noqa: E402  # type: ignore[import-not-found]

DE_OUTPUT_CSV: str = os.path.join(
    SHOWCASE_DIR,
    "data/modeling/outputs/east_river/differential_evolution_outputs/output_20260901_202618.csv",
)
OUTPUT_DIR: str = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "outputs")
)


def pick_param_vector(
    bounds: dict[str, tuple[float, float]],
    csv_path: str | None,
    use_midpoint: bool,
) -> tuple[np.ndarray, int, str]:
    """Build a parameter vector in ``bounds`` order for the East River run.

    For every bound name, the value is taken from the referenced
    differential-evolution output CSV row when that name is present and
    finite; otherwise it falls back to the midpoint of the bound (used for
    the ``_x`` exchange-site initials, which the CSV layout does not carry).

    Args:
        bounds: mapping of parameter name to ``(min, max)`` bound, in order.
        csv_path: path to a DE output CSV, or ``None`` to use all midpoints.
        use_midpoint: if ``True`` ignore the CSV and use every bound midpoint.

    Returns:
        The parameter array, the number of values sourced from the CSV, and
        a short provenance label.
    """
    names: list[str] = list(bounds.keys())
    values: list[float] = []
    matched: int = 0

    row: Series | None = None
    if csv_path is not None and not use_midpoint:
        df: DataFrame = pd.read_csv(csv_path)
        needs: list[str] = ["q_nse_cal", "doc_nse_cal", "q_nse_val", "doc_nse_val"]
        obj_cols: set[str] = {
            f"{p}_{m}_{per}"
            for p in ("q", "doc", "m_doc")
            for m in ("nse", "log_nse", "kge")
            for per in ("cal", "val")
        }
        min_cols: list[str] = [c for c in df.columns if c.endswith("_soc")]
        valid: np.ndarray = np.ones(len(df), dtype=bool)
        for c in needs:
            vals = pd.to_numeric(df[c], errors="coerce").to_numpy()
            valid &= np.isfinite(vals) & (vals >= -1.5) & (vals <= 1.1)
        mv: np.ndarray = df[min_cols].apply(pd.to_numeric, errors="coerce").to_numpy()
        valid &= np.all(np.isfinite(mv), axis=1) & np.all((mv >= -8) & (mv <= 8), axis=1)
        cand: np.ndarray = np.where(valid)[0]
        row = df.iloc[int(cand[0])]

    for nm in names:
        if row is not None and nm in row.index and np.isfinite(row[nm]):
            values.append(float(row[nm]))
            matched += 1
        else:
            lo, hi = bounds[nm]
            values.append(0.5 * (lo + hi))

    src: str = "all-midpoints" if (row is None) else f"CSV row {csv_path} (matched {matched}/{len(names)})"
    return np.asarray(values, dtype=float), matched, src


def build_window_config(
    base: CalibrationData,
    site_data: DataFrame,
    doc_df: DataFrame,
    window: DatetimeIndex,
) -> CalibrationData:
    """Assemble a length-consistent calibration config over a single window.

    The showcase ``load_input_data`` passes a full-record ``forc`` but
    period-sliced ``q_meas`` / ``conc_df``. For a *clean* equal-length
    sanity window we slice the forcing, the measured streamflow, the river
    DOC, and the precipitation chemistry all to the same dates so the
    hydrology and reactive-transport objectives align positionally.

    Args:
        base: the config returned by ``load_input_data`` (source of the
            network, zone configs, and model type).
        site_data: the East River site dataset (``ppt``/``temp``/``pet``/``q_meas``).
        doc_df: the river DOC concentration series (mmol/L), index-aligned to
            the site data where measurements exist.
        window: the dates to simulate (must be present in ``site_data``).

    Returns:
        A new ``CalibrationData`` whose forcing, streamflow, DOC and
        precipitation-chemistry all share the length of ``window``.
    """
    comb: DataFrame = site_data.merge(
        doc_df, left_index=True, right_index=True, how="left"
    )

    window = window.intersection(cast(DatetimeIndex, site_data.index))
    forc: pt.ForcingData = pt.ForcingData(
        precip=site_data.loc[window, "ppt"],
        temp=site_data.loc[window, "temp"],
        pet=site_data.loc[window, "pet"],
    )
    q_meas: Series = site_data.loc[window, "q_meas"]
    conc_df: DataFrame = comb.loc[window, "DOC"].to_frame()
    precip_conc: np.ndarray = get_precip_conc(
        len(pt.HbvLateralModel.structure[0]), len(window), base["network"].num_species
    )

    return {
        "network": base["network"],
        "zone_configs": base["zone_configs"],
        "forc": forc,
        "q_meas": q_meas,
        "conc_df": conc_df,
        "precip_conc": precip_conc,
        "model_type": base["model_type"],
    }


def choose_window(
    dates: DatetimeIndex,
    wy: int | None,
    start: str | None,
    end: str | None,
    days: int | None,
    full: bool = False,
) -> DatetimeIndex:
    """Select the simulation date window from the command-line options.

    Exactly one of ``wy``, ``start``/``end``, ``days``, or ``full`` should be
    given. With no option the last complete water year in the record is
    returned.

    Args:
        dates: the full daily index of the site data.
        wy: a water year to isolate, e.g. ``2016``.
        start: an ISO start date string.
        end: an ISO end date string.
        days: a positive number of trailing days to run.
        full: run the entire record.

    Returns:
        The sorted ``DatetimeIndex`` of dates in the window.
    """
    if full:
        return dates
    if wy is not None:
        wy_arr: np.ndarray = np.array([_get_wy(d) for d in dates])
        mask: np.ndarray = wy_arr == wy
    elif start is not None and end is not None:
        lo: Timestamp = Timestamp(start)
        hi: Timestamp = Timestamp(end)
        mask = (dates >= lo) & (dates <= hi)
    elif days is not None:
        idx: np.ndarray = np.arange(len(dates))
        mask = np.zeros(len(dates), dtype=bool)
        mask[np.arange(len(dates))[-days:] if days > 0 else []] = True
    else:
        wy_arr = np.array([_get_wy(d) for d in dates])
        last_wy: int = int(min(wy_arr[wy_arr > wy_arr.min() + 1].max(), wy_arr.max()))
        mask = wy_arr == last_wy

    return dates[mask]


def run(args: argparse.Namespace) -> int:
    """Execute the forward East River run and report a sanity report."""
    # ---- Load the calibration inputs (showcase pipeline) ---- #
    print("== Loading East River calibration inputs ...")
    t0: float = time.perf_counter()
    inputs = load_input_data(
        pt.HbvLateralModel, test_wys=[2017, 2020], val_wy=2018, site="EastRiver"
    )
    print(f"   loaded inputs in {time.perf_counter() - t0:.1f}s")
    print(f"   num bounds: {len(inputs.bounds)}")

    # ---- Parameter vector ---- #
    arr, matched, src = pick_param_vector(
        inputs.bounds,
        args.csv,
        args.midpoint,
    )
    print("== Parameter vector ==")
    print(f"   {src}")

    # ---- Site data + window ---- #
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

    window: DatetimeIndex = choose_window(
        cast(DatetimeIndex, site_data.index),
        args.wy,
        args.start,
        args.end,
        args.days,
        full=args.full,
    )
    print("== Simulation window ==")
    print(f"   {window[0]} .. {window[-1]}  ({len(window)} days)")

    cfg: CalibrationData = build_window_config(inputs.cal_config, site_data, doc_df, window)

    # ---- Run ---- #
    print("== Running hydrology + reactive transport (potions-py) ...")
    print(f"   hydro zones: {pt.HbvLateralModel.get_zone_names()}")
    print(f"   network:     {cfg['network'].species_order}  (exchange={cfg['network'].has_exchange})")

    t1: float = time.perf_counter()
    res = arr_to_results(arr, cfg, hydro_criteria=None, verbose=args.verbose)
    run_secs: float = time.perf_counter() - t1

    if isinstance(res, pt.HydroModelResults):
        print("\n   ! Reactive transport was not run (hydrology criteria path).")
        return 1

    hydro: pt.HydroModelResults = res.hydro
    rt: pt.RtModelResults = res.reactive_transport

    q_sim: Series = hydro.simulation["sim_streamflow_mmd"]
    doc_sim: Series = rt.simulation["DOC_riv"]
    hydro_obj: dict[str, float] = {
        str(k): float(v) for k, v in hydro.objective_functions.items()
    }
    if rt.objective_functions is None:
        raise ValueError("reactive-transport objectives are None")
    rt_doc_obj: dict[str, float] = {
        str(k): float(rt.objective_functions[k]["DOC"])
        for k in rt.objective_functions.columns.to_list()
    }

    # ---- Sanity assertions ---- #
    print("\n== Sanity report ==")
    print(f"   run duration:        {run_secs:.1f}s  ({run_secs / len(window):.3f}s/day)")
    print(f"   streamflow series:   {q_sim.shape}  finite={bool(np.isfinite(q_sim.to_numpy(dtype=float)).all())}  "
          f"min={float(q_sim.min()):.4f}  max={float(q_sim.max()):.4f}  mean={float(q_sim.mean()):.4f}")
    print(f"   river DOC series:    {doc_sim.shape}  finite={bool(np.isfinite(doc_sim.to_numpy(dtype=float)).all())}  "
          f"min={float(doc_sim.min()):.5f}  max={float(doc_sim.max()):.5f}  mean={float(doc_sim.mean()):.5f}")
    print(f"   hydro objectives:    {hydro_obj}")
    print(f"   RT DOC objectives:   {rt_doc_obj}")
    print(f"   DOC measurement mean: {float(cfg['conc_df']['DOC'].mean()):.5f} mmol/L")

    assert np.isfinite(q_sim.to_numpy(dtype=float)).all(), "non-finite streamflow"
    assert float(q_sim.min()) >= -1e-6, "negative streamflow"
    assert np.isfinite(q_sim.to_numpy(dtype=float)).mean() > 0, "zero-mean streamflow"

    doc_arr: np.ndarray = doc_sim.to_numpy(dtype=float)
    assert np.isfinite(doc_arr).all(), "non-finite river DOC"
    assert float(np.nanmin(doc_arr)) > 0.0, "river DOC not positive"
    assert float(np.nanmean(doc_arr)) > 0.0, "zero-mean river DOC"

    for k, v in hydro_obj.items():
        assert np.isfinite(v), f"non-finite hydro objective {k}"
    for k, v in rt_doc_obj.items():
        assert np.isfinite(v), f"non-finite RT objective {k}"

    # ---- Persist ---- #
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tag: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    window_tag: str = f"{window[0]:%Y%m%d}_to_{window[-1]:%Y%m%d}"
    out_pkl: str = os.path.join(OUTPUT_DIR, f"east_river_{window_tag}_{tag}.pkl")
    import pickle

    with open(out_pkl, "wb") as f:
        pickle.dump(
            {
                "window": window.to_list(),
                "n_days": len(window),
                "run_seconds": run_secs,
                "param_source": src,
                "n_params": int(arr.size),
                "q_sim": q_sim,
                "doc_sim": doc_sim,
                "hydro_objectives": hydro_obj,
                "rt_doc_objectives": rt_doc_obj,
                "param_series": pd.Series(arr, index=list(inputs.bounds.keys())),
            },
            f,
        )
    print(f"\n   all sanity checks PASSED")
    print(f"   results saved: {out_pkl}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--wy", type=int, default=None, help="run a single water year, e.g. 2016")
    g.add_argument("--start", default=None, help="ISO start date, e.g. 2016-01-01 (with --end)")
    g.add_argument("--days", type=int, default=None, help="run the trailing N days")
    g.add_argument("--full", action="store_true", help="run the entire record")
    parser.add_argument("--end", default=None, help="ISO end date, e.g. 2016-12-31 (with --start)")
    parser.add_argument("--csv", default=DE_OUTPUT_CSV, help="DE output CSV for the parameter row")
    parser.add_argument("--midpoint", action="store_true", help="use all bound midpoints, no CSV")
    parser.add_argument(
        "--verbose", action="store_true", help="print per-zone / per-step run diagnostics"
    )
    args = parser.parse_args()
    if args.start is not None and args.end is None:
        parser.error("--start requires --end")
    if args.end is not None and args.start is None:
        parser.error("--end requires --start")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
