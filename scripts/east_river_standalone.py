"""Standalone East River hydrology + reactive-transport forward run.

Runs the full East River record (2014-10-01 .. 2021-09-30, ~2557 days) with
the simple dissolved-organic-carbon network ``DOC / SOC(s) / X- / XDOC`` and
saves the simulated streamflow and river-DOC series plus the objective scores
to a file.

Everything the run needs sits in this repository:

- ``inputs/east_river/PTQ.txt`` / ``EVAP.txt``  -- daily precipitation,
  temperature, measured streamflow and potential evapotranspiration (copied
  verbatim from the potions-showcase ``hbv_data`` package).
- ``inputs/east_river/er_phf0_npoc.csv``        -- measured river DOC for the
  Pumphouse station (copied verbatim from the showcase water-chemistry data).

No code or data outside this repository is read: the reaction network comes
from :func:`potions.reactive_transport.get_simple_carbon_adsorption_network`,
the hydrology zone structure from :class:`potions.HbvLateralModel`, and the
remaining model definition (which zones react, the 85 parameter names and
their default midpoints) is embedded below, mirroring the showcase's
``utils/parameter_ranges.py`` tables.

Usage::

    PYTHONPATH=src python scripts/east_river_standalone.py
    PYTHONPATH=src python scripts/east_river_standalone.py --params my_params.csv
    PYTHONPATH=src python scripts/east_river_standalone.py --out outputs/my_run.csv
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from datetime import datetime
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
from potions.model_batched import BatchedConfig, run_hydro_rt_batched  # noqa: E402

DATA_DIR: Path = Path(__file__).resolve().parent.parent / "inputs" / "east_river"
OUTPUT_DIR: Path = Path(__file__).resolve().parent / "outputs"

#: Flat evaluation order of the 8 HbvLateralModel zones.
ZONES: tuple[str, ...] = (
    "snow_hs", "snow_rp", "surface_hs", "surface_rp",
    "shallow_hs", "shallow_rp", "deep_hs", "deep_rp",
)

#: Zones with snow (non-reacting, non-speciating) behaviour.
SNOW_ZONES: frozenset[str] = frozenset(("snow_hs", "snow_rp"))

#: Which zones run reactions + speciation (mirrors showcase ``HBV_LATERAL_CONFIGS``).
def zone_config() -> dict[str, RtZoneConfiguration]:
    """Per-zone ``(do_reactions, do_speciation)`` flags for the East River."""
    return {
        name: RtZoneConfiguration(
            do_reactions=name not in SNOW_ZONES,
            do_speciation=name not in SNOW_ZONES,
        )
        for name in ZONES
    }


#: Fixed porous-medium values for the snow zones (showcase ``const_zone_bounds``).
SNOW_VALUE: dict[str, float] = {"porosity": 0.5, "depth": 10_000.0, "passive_water_storage": 0.0}

#: Midpoints of the reacting-zone parameter bounds (showcase zone tables):
#: value = fixed constant, ``(lo, hi)`` = calibratable with that bound range.
RT_ZONE_MIDPOINTS: dict[str, dict[str, float | tuple[float, float]]] = {
    "surface": {
        "porosity": 0.3,
        "depth": (500.0, 5_000.0),
        "passive_water_storage": 0.0,
        "sw_threshold": (0.0, 1.0),
        "sw_exp": (0.5, 2.5),
        "q_10": (1.0, 5.0),
        "ssa": (-3.0, 5.0),
    },
    "shallow": {
        "porosity": 0.2,
        "depth": (500.0, 5_000.0),
        "passive_water_storage": 0.0,
        "sw_threshold": (0.0, 1.0),
        "sw_exp": (0.5, 2.5),
        "q_10": 1.0,
        "ssa": (-3.0, 4.0),
    },
    "deep": {
        "porosity": 0.1,
        "depth": (1_000.0, 10_000.0),
        "passive_water_storage": 0.0,
        "sw_threshold": 1.0,
        "sw_exp": 0.0,
        "q_10": 1.0,
        "ssa": (-3.0, 4.0),
    },
}

#: Midpoints of the log10 mineral-concentration / exchange-site bounds
#: (showcase ``HBV_LATERAL_MIN_BOUNDS``), per reacting zone.
MIN_CONC_MIDPOINTS: dict[str, tuple[float, float]] = {
    "surface_hs": (-1.0, 2.0),
    "surface_rp": (-1.0, 1.0),
    "shallow_hs": (-2.0, 0.0),
    "shallow_rp": (-2.0, 0.0),
    "deep_hs": (-3.0, -1.0),
    "deep_rp": (-3.0, -1.0),
}

#: Hydrology zones whose alphas are fixed (showcase calibration convenience).
FIXED_ALPHA: dict[str, float] = {
    "shallow_hs.alpha": 1.0,
    "shallow_rp.alpha": 1.0,
    "deep_hs.alpha": 1.0,
    "deep_rp.alpha": 1.0,
}

#: Per-mineral RT parameter order (sw_threshold, sw_exp, q_10, ssa).
RT_PARAM_NAMES: tuple[str, ...] = ("sw_threshold", "sw_exp", "q_10", "ssa")

DOC_SPECIES: str = "DOC"


def load_site_data(data_dir: Path = DATA_DIR) -> DataFrame:
    """Load the daily forcing + measured streamflow from the copied files.

    Replicates the showcase ``hbv_data.load_site_data`` exactly: ``PTQ.txt``
    holds ``(date, ppt, temp, q_meas)`` and ``EVAP.txt`` holds daily ``pet``,
    merged on their (shared) date index.

    Args:
        data_dir: directory containing ``PTQ.txt`` and ``EVAP.txt``.

    Returns:
        A DataFrame indexed by day with ``ppt``, ``temp``, ``pet``,
        ``q_meas`` columns.
    """
    ptq: DataFrame = pd.read_csv(
        data_dir / "PTQ.txt",
        sep=r"\s+",
        skiprows=1,
        index_col=0,
        parse_dates=True,
        date_format="%Y%m%d",
    )
    pet: DataFrame = pd.read_csv(data_dir / "EVAP.txt", sep=r"\s+")
    ptq.index = pd.to_datetime(ptq.index)
    pet.index = ptq.index
    combined: DataFrame = ptq.merge(pet, left_index=True, right_index=True, how="inner")
    combined.rename(
        columns={
            combined.columns[0]: "ppt",
            combined.columns[1]: "temp",
            combined.columns[2]: "q_meas",
            combined.columns[3]: "pet",
        },
        inplace=True,
    )
    return combined[["ppt", "temp", "pet", "q_meas"]].copy()


def load_river_doc(data_dir: Path = DATA_DIR) -> DataFrame:
    """Load the measured river-DOC record for Pumphouse (mmol/L).

    The raw ``npoc`` values are ng/kg concentrations; ``/1000`` gives mg/L and
    ``/12.01`` converts with the molar mass of carbon to mmol/L.  Duplicate
    dates are averaged.

    Args:
        data_dir: directory containing ``er_phf0_npoc.csv``.

    Returns:
        A ``DOC`` series in mmol/L indexed by date.
    """
    doc: DataFrame = pd.read_csv(data_dir / "er_phf0_npoc.csv", index_col=0, parse_dates=True)
    doc.index = pd.to_datetime(doc.index)
    doc["npoc"] = pd.to_numeric(doc["npoc"], errors="coerce")
    converted: Series = (doc["npoc"] / 1000.0 / 12.01).rename("DOC")
    return converted.groupby(converted.index).mean().to_frame()


def build_config(data_dir: Path = DATA_DIR) -> tuple[BatchedConfig, DatetimeIndex]:
    """Assemble the batched-run config over the full East River record.

    All series (forcing, measured streamflow, river DOC) are aligned to the
    full 2014-10-01 .. 2021-09-30 record; the precipitation chemistry is the
    constant 1e-20 fill the showcase uses.

    Returns:
        The :class:`BatchedConfig` and the simulation date window.
    """
    site: DataFrame = load_site_data(data_dir)
    doc: DataFrame = load_river_doc(data_dir)
    window: DatetimeIndex = cast(DatetimeIndex, site.index)
    combined: DataFrame = site.merge(doc, left_index=True, right_index=True, how="left")

    network = get_simple_carbon_adsorption_network()
    model_type: type[pt.Model] = pt.HbvLateralModel
    num_surface: int = len(model_type.structure[0])
    num_species: int = network.num_species
    T: int = len(window)

    config: BatchedConfig = {
        "network": network,
        "zone_configs": zone_config(),
        "forc": ForcingData(
            precip=combined.loc[window, "ppt"],
            temp=combined.loc[window, "temp"],
            pet=combined.loc[window, "pet"],
        ),
        "q_meas": combined.loc[window, "q_meas"],
        "conc_df": combined.loc[window, "DOC"].to_frame(),
        "precip_conc": np.full(
            (num_surface, T, num_species), fill_value=1e-20, dtype=np.float64
        ),
        "model_type": model_type,
    }
    return config, window


def midpoint(value: float | tuple[float, float]) -> float:
    """The centre of a bound pair, or the fixed value itself."""
    if isinstance(value, tuple):
        return 0.5 * (value[0] + value[1])
    return float(value)


def default_parameter_vector() -> tuple[list[str], NDArray[np.float64]]:
    """The 85 calibration-order parameter names and their midpoint values.

    Name and value ordering follows the batched runner's layout: the 25
    hydrological columns (``HbvLateralModel.default_parameter_ranges()`` with
    the fixed alphas), then the 48 per-zone RT columns (3 porous-medium values
    per zone, plus ``sw_threshold / sw_exp / q_10 / ssa`` per reacting zone),
    then the 12 init-state columns (log10 mineral and exchange-site
    concentrations per reacting zone).  ``ssa`` and ``k`` are on their log10
    natural scales, exactly as in the showcase calibration.

    Returns:
        ``(names, midpoints)`` -- a length-85 ordered list and vector.
    """
    hyd_bounds: dict[str, tuple[float, float]] = dict(pt.HbvLateralModel.default_parameter_ranges())
    for name, value in FIXED_ALPHA.items():
        hyd_bounds[name] = (value, value)
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
        table: dict[str, float | tuple[float, float]] = RT_ZONE_MIDPOINTS[zone_kind[zone]]
        for param in ("porosity", "depth", "passive_water_storage", *RT_PARAM_NAMES):
            names.append(f"{zone}.{param}")
            values.append(midpoint(table[param]))

    for zone, (lo, hi) in MIN_CONC_MIDPOINTS.items():
        names.append(f"mineral_concentration.{zone}_soc")
        values.append(0.5 * (lo + hi))
        names.append(f"mineral_concentration.{zone}_x")
        values.append(0.5 * (lo + hi))

    assert len(names) == 85 and len(values) == 85, "expected the 85-parameter East River layout"
    return names, np.asarray(values, dtype=np.float64)


def resolve_parameters(param_sets: DataFrame | None) -> tuple[list[str], NDArray[np.float64]]:
    """Build the parameter vector, filling missing columns with midpoints.

    Args:
        param_sets: optional one-row DataFrame whose columns are parameter
            names; omitted columns use the built-in midpoints.

    Returns:
        ``(names, vector)`` in calibration order.
    """
    names, midpoints = default_parameter_vector()
    if param_sets is None:
        return names, midpoints
    filled: DataFrame = param_sets.reindex(columns=names).fillna(
        Series(midpoints, index=names)
    )
    return names, filled[names].to_numpy(dtype=np.float64).ravel()


def run_forward(
    params: NDArray[np.float64],
    config: BatchedConfig,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> dict[str, Any]:
    """Run the model forwards for one parameter set over the whole record.

    Uses CUDA float32 by default (the potions-py default device/dtype policy);
    pass ``device="cpu", dtype=torch.float64`` for the exact scalar-parity path.

    Args:
        params: the ``(85,)`` parameter vector in calibration order.
        config: the batched config.
        device: target device.
        dtype: target dtype (float32 on CUDA, float64 on CPU).

    Returns:
        A dict of the simulated streamflow and river-DOC series (indexed by
        date), the objective-score dicts, and the timings.
    """
    dev: torch.device = torch.device(device)
    started: float = time.perf_counter()
    out = run_hydro_rt_batched(params.reshape(1, -1), config, dev, dtype)
    wall: float = time.perf_counter() - started

    window: DatetimeIndex = cast(DatetimeIndex, config["forc"].precip.index)  # type: ignore[union-attr]
    q_sim: Series = Series(out.q_sim[0].detach().cpu().numpy(), index=window, name="q_sim")
    doc_sim: Series = Series(out.doc_riv[0].detach().cpu().numpy(), index=window, name="doc_riv")
    return {
        "window": window,
        "q_sim": q_sim,
        "doc_riv": doc_sim,
        "hydro_objectives": out.hydro_objectives[0],
        "rt_doc_objectives": out.rt_doc_objectives[0],
        "run_seconds": wall,
    }


def save_outputs(results: dict[str, Any], base: Path, tag: str) -> None:
    """Persist the run to a CSV (daily series) and a pickle (everything)."""
    base.parent.mkdir(parents=True, exist_ok=True)
    csv_path: Path = base.with_name(f"east_river_{tag}.csv")
    pkl_path: Path = base.with_name(f"east_river_{tag}.pkl")
    table: DataFrame = pd.concat([results["q_sim"], results["doc_riv"]], axis=1)
    table.to_csv(csv_path, float_format="%.6f")
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)
    print(f"   daily series: {csv_path}")
    print(f"   full payload: {pkl_path}")


def main() -> int:
    """Forward-run the East River model and save the outputs."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", default=None,
                        help="CSV of a parameter set (any subset of the 85 columns; "
                             "missing values use the built-in midpoints)")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"],
                        help="run device (default: cuda)")
    parser.add_argument("--out", default=None, help="output file base name (default: outputs/east_river_<timestamp>)")
    args: argparse.Namespace = parser.parse_args()

    dev: torch.device = torch.device(args.device)
    dtype: torch.dtype = torch.float64 if dev.type == "cpu" else torch.float32

    print("== Standalone East River forward run (potions-py) ==")
    print(f"   device: {dev}  dtype: {dtype}")
    config, window = build_config()
    print(f"   window: {window[0]:%Y-%m-%d} .. {window[-1]:%Y-%m-%d}  ({len(window)} days)")
    print(f"   network: {config['network'].species_names}  (exchange={config['network'].has_exchange})")

    param_sets: DataFrame | None = (
        pd.read_csv(args.params) if args.params is not None else None
    )
    names, params = resolve_parameters(param_sets)
    if param_sets is not None:
        print(f"   parameter set loaded from {args.params}")

    results: dict[str, Any] = run_forward(params, config, device=dev, dtype=dtype)

    print("== Results ==")
    q: Series = results["q_sim"]
    d: Series = results["doc_riv"]
    print(f"   run duration: {results['run_seconds']:.1f}s  ({results['run_seconds']/len(window):.3f}s/day)")
    print(f"   streamflow:   finite={bool(np.isfinite(q).all())}  min={float(q.min()):.4f}  "
          f"max={float(q.max()):.4f}  mean={float(q.mean()):.4f} mm/day")
    print(f"   river DOC:    finite={bool(np.isfinite(d).all())}  min={float(d.min()):.5f}  "
          f"max={float(d.max()):.5f}  mean={float(d.mean()):.5f} mmol/L")
    print(f"   hydro objectives: {results['hydro_objectives']}")
    print(f"   RT DOC objectives: {results['rt_doc_objectives']}")

    base: Path = Path(args.out) if args.out is not None else OUTPUT_DIR / "east_river"
    save_outputs(results, base, datetime.now().strftime("%Y%m%d_%H%M%S"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())