"""Parity checks for the batched ``K``-lane East River runner.

Two levels of verification against independent references:

1. ``K = 1``: the batched runner with a single lane must reproduce the saved
   scalar forward ``.pkl`` byte-for-byte (streamflow within ``1e-9`` relative,
   river DOC within ``1e-9`` relative, objective dicts within ``1e-9``
   absolute).
2. ``K = 3``: three lanes must agree with three *independent* scalar
   ``arr_to_results`` runs (the showcase pipeline itself), comparing the full
   ``sim_streamflow_mmd`` / ``DOC_riv`` series and the objective dicts.

A lane whose storage touches the zone maximum water volume aborts the scalar
path with an explicit ``WaterVolumeError`` while the batched path merely sets
its ``overflow_mask`` bit; such lanes are skipped from the scalar comparison
and reported.  Random lanes are drawn from the inner 60% of each parameter
bound to stay clear of the overflow regime.
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys
import time
from typing import Any, cast

import numpy as np
import torch
from pandas import DataFrame, DatetimeIndex

# The bitwise f64 contract is defined for the CPU path; the package is
# 100% eager (no ``torch.compile``), so no env override is needed here.

SCRIPTS: str = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)

import east_river_forward as fwd  # noqa: E402  (registers showcas paths)

import potions as pt  # noqa: E402  # type: ignore[import-not-found]
from potions.model_batched import BatchedConfig  # noqa: E402  # type: ignore[import-not-found]
from utils.calibration import arr_to_results  # noqa: E402  # type: ignore[import-not-found]
from utils.parameter_ranges import load_input_data  # noqa: E402  # type: ignore[import-not-found]
from data.raw.east_river import hbv_data, load_all_doc_data  # noqa: E402  # type: ignore[import-not-found]

#: Newest saved scalar forward result (window 2021-09-01 -> 2021-09-30).
def latest_oracle() -> str:
    """Return the newest matching forward-run ``.pkl`` in the outputs dir."""
    candidates: list[str] = glob.glob(
        os.path.join(fwd.OUTPUT_DIR, "east_river_20210901_to_20210930_*.pkl")
    )
    if not candidates:
        raise FileNotFoundError(
            "no east_river_20210901_to_20210930_*.pkl oracle found; run "
            "east_river_forward.py --start 2021-09-01 --end 2021-09-30 first"
        )
    return max(candidates, key=os.path.getmtime)

#: Relative/absolute tolerance for every compared quantity.
TOL_REL: float = 1e-9
TOL_ABS: float = 1e-9


def report(name: str, ok: bool, detail: str) -> bool:
    """Print one check result and return whether it passed."""
    print(f"   [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


def max_abs_rel(a: np.ndarray, b: np.ndarray) -> float:
    """Max relative difference (masked where b is 0 or NaN)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    finite: np.ndarray = np.isfinite(a) & np.isfinite(b)
    if not finite.any():
        return 0.0
    scale: np.ndarray = np.abs(b[finite])
    scale = np.where(scale > 0.0, scale, 1.0)
    return float(np.max(np.abs(a[finite] - b[finite]) / scale))


def build_config(window: DatetimeIndex) -> BatchedConfig:
    """Reassemble the calibration config for ``window`` (forward-script path).

    The showcase ``CalibrationData`` and the batched :class:`BatchedConfig`
    share the same keys, so the loaded record is reused as-is.
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
    return cast(
        BatchedConfig,
        fwd.build_window_config(inputs.cal_config, site_data, doc_df, window),
    )


def aim_mid(arr: np.ndarray, bounds: dict[str, tuple[float, float]]) -> np.ndarray:
    """Draw one parameter vector inside the inner 60% of every bound."""
    names: list[str] = list(bounds.keys())
    out: list[float] = []
    for nm in names:
        lo: float
        hi: float
        lo, hi = bounds[nm]
        span: float = hi - lo
        out.append(float(np.random.uniform(lo + 0.2 * span, lo + 0.8 * span)))
    return np.asarray(out, dtype=float)


def scalars_for(
    params: np.ndarray, cfg: BatchedConfig
) -> list[dict[str, Any]]:
    """Run one independent scalar ``arr_to_results`` per lane."""
    out: list[dict[str, Any]] = []
    for k in range(int(params.shape[0])):
        res = arr_to_results(params[k], cfg, hydro_criteria=None, verbose=False)  # type: ignore[arg-type]
        if isinstance(res, pt.HydroModelResults):
            raise RuntimeError(f"lane {k}: scalar run returned hydrology-only results")
        hydro: pt.HydroModelResults = res.hydro
        rt: pt.RtModelResults = res.reactive_transport
        hydro_obj: dict[str, float] = {
            str(kk): float(vv) for kk, vv in hydro.objective_functions.items()
        }
        rt_doc_obj: dict[str, float] = {
            str(kk): float(rt.objective_functions[kk]["DOC"])  # type: ignore[index]
            for kk in rt.objective_functions.columns.to_list()  # type: ignore[attr-defined]
        }
        out.append(
            {
                "q_sim": hydro.simulation["sim_streamflow_mmd"].to_numpy(dtype=float),
                "doc_riv": rt.simulation["DOC_riv"].to_numpy(dtype=float),
                "hydro_obj": hydro_obj,
                "rt_doc_obj": rt_doc_obj,
            }
        )
    return out


def check_k1(batched: pt.BatchedRunOutput, oracle: dict[str, Any]) -> bool:
    """Verify the single-lane run against the saved scalar ``.pkl``."""
    q_ref: np.ndarray = oracle["q_sim"].to_numpy(dtype=float)  # type: ignore[union-attr]
    doc_ref: np.ndarray = oracle["doc_sim"].to_numpy(dtype=float)  # type: ignore[union-attr]
    q_sim: np.ndarray = batched.q_sim[0].detach().cpu().numpy()
    doc_riv: np.ndarray = batched.doc_riv[0].detach().cpu().numpy()

    ok: bool = True
    ok &= report(
        "q_sim series", max_abs_rel(q_sim, q_ref) <= TOL_REL,
        f"max-abs-rel={max_abs_rel(q_sim, q_ref):.3e} (n={q_ref.size})",
    )
    ok &= report(
        "q_sim mean", abs(float(q_sim.mean()) - float(q_ref.mean())) <= TOL_ABS,
        f"batched={float(q_sim.mean()):.9f} oracle={float(q_ref.mean()):.9f}",
    )
    ok &= report(
        "DOC_riv series", max_abs_rel(doc_riv, doc_ref) <= TOL_REL,
        f"max-abs-rel={max_abs_rel(doc_riv, doc_ref):.3e} (n={doc_ref.size})",
    )
    ok &= report(
        "DOC_riv mean", abs(float(doc_riv.mean()) - float(doc_ref.mean())) <= TOL_ABS,
        f"batched={float(doc_riv.mean()):.12e} oracle={float(doc_ref.mean()):.12e}",
    )

    for name, bv, ov in (
        ("hydro", batched.hydro_objectives[0], oracle["hydro_objectives"]),
        ("rt_doc", batched.rt_doc_objectives[0], oracle["rt_doc_objectives"]),
    ):
        assert isinstance(ov, dict)
        for key in ov:
            ok &= report(
                f"{name} objective {key}",
                abs(float(bv[key]) - float(ov[key])) <= TOL_ABS,
                f"batched={float(bv[key]):.8f} oracle={float(ov[key]):.8f}",
            )

    ok &= report(
        "overflow_mask[0]",
        bool(not bool(batched.overflow_mask[0].item())),
        "expected False",
    )
    return ok


def check_k3(
    batched: pt.BatchedRunOutput,
    scalar_res: list[dict[str, Any]],
    overflow: list[bool],
) -> bool:
    """Verify every lane against its independent scalar run."""
    ok: bool = True
    for k, (s, ovfl) in enumerate(zip(scalar_res, overflow, strict=True)):
        tag: str = f"lane {k}"
        if ovfl:
            print(f"   [SKIP] {tag}: scalar WaterVolumeError (batched overflow_mask set)")
            continue
        q_sim: np.ndarray = batched.q_sim[k].detach().cpu().numpy()
        doc_riv: np.ndarray = batched.doc_riv[k].detach().cpu().numpy()
        ok &= report(
            f"{tag} q_sim", max_abs_rel(q_sim, s["q_sim"]) <= TOL_REL,
            f"max-abs-rel={max_abs_rel(q_sim, s['q_sim']):.3e}",
        )
        ok &= report(
            f"{tag} doc_riv", max_abs_rel(doc_riv, s["doc_riv"]) <= TOL_REL,
            f"max-abs-rel={max_abs_rel(doc_riv, s['doc_riv']):.3e}",
        )
        for name in ("hydro_obj", "rt_doc_obj"):
            for key in s[name]:  # type: ignore[index]
                diff: float = (
                    abs(
                        float(batched.hydro_objectives[k][key])
                        - float(s[name][key])  # type: ignore[index]
                    )
                    if name == "hydro_obj"
                    else abs(
                        float(batched.rt_doc_objectives[k][key])
                        - float(s[name][key])  # type: ignore[index]
                    )
                )
                ok &= report(f"{tag} {name} {key}", diff <= TOL_ABS, f"diff={diff:.3e}")
    return ok


def run(args: argparse.Namespace) -> int:
    """Load the oracle, rebuild the window config, run both parity stages."""
    with open(latest_oracle(), "rb") as f:
        oracle: dict[str, Any] = pickle.load(f)

    window: DatetimeIndex = DatetimeIndex([str(d) for d in oracle["window"]])  # type: ignore[arg-type]
    print("== Building East River window config ==")
    cfg: BatchedConfig = build_config(window)
    print(f"   window {window[0]} .. {window[-1]}  ({len(window)} days)")

    arr: np.ndarray = oracle["param_series"].to_numpy()  # type: ignore[union-attr]
    assert arr.shape[0] == oracle["n_params"]  # type: ignore[arg-type]

    # ---- Stage 1: K = 1 vs oracle ---- #
    print("\n== Stage 1: K = 1 batched run vs saved scalar pkl ==")
    t0: float = time.perf_counter()
    batched1: pt.BatchedRunOutput = pt.run_hydro_rt_batched(
        arr.reshape(1, -1), cfg, torch.device("cpu"), torch.float64
    )
    print(f"   batched K=1 wall: {time.perf_counter() - t0:.3f}s")
    ok1: bool = check_k1(batched1, oracle)

    # ---- Stage 2: K = 3 vs per-lane scalar runs ---- #
    print("\n== Stage 2: K = 3 batched run vs 3 independent scalar runs ==")
    torch.manual_seed(args.seed)
    inputs = load_input_data(
        pt.HbvLateralModel, test_wys=[2017, 2020], val_wy=2018, site="EastRiver"
    )
    lanes: list[np.ndarray] = [arr] + [
        aim_mid(arr, inputs.bounds) for _ in range(2)
    ]
    params: np.ndarray = np.stack(lanes, axis=0)

    t1: float = time.perf_counter()
    batched3: pt.BatchedRunOutput = pt.run_hydro_rt_batched(
        params, cfg, torch.device("cpu"), torch.float64
    )
    t_batched: float = time.perf_counter() - t1
    print(f"   batched K=3 wall: {t_batched:.3f}s ({t_batched / 3:.3f}s/lane)")

    scalar_res: list[dict[str, Any]] = []
    overflow_lanes: list[bool] = []
    t2: float = time.perf_counter()
    for k in range(3):
        try:
            scalar_res.append(scalars_for(params[k : k + 1], cfg)[0])
            overflow_lanes.append(False)
        except pt.WaterVolumeError:
            overflow_lanes.append(True)
            scalar_res.append({})
        except Exception:
            raise
    t_scalar: float = time.perf_counter() - t2
    print(f"   scalar K=3 wall: {t_scalar:.3f}s ({t_scalar / 3:.3f}s/lane)")

    ok2: bool = check_k3(batched3, scalar_res, overflow_lanes)

    # ---- Summary ---- #
    print("\n== Parity summary ==")
    print(f"   Stage 1 (K=1 vs pkl):      {'PASS' if ok1 else 'FAIL'}")
    print(f"   Stage 2 (K=3 vs scalar):   {'PASS' if ok2 else 'FAIL'}")
    ok_all: bool = ok1 and ok2
    print(f"   overall: {'PASS' if ok_all else 'FAIL'}")
    return 0 if ok_all else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="random lane seed")
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())