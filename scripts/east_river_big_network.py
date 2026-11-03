"""Forward run of the East River large-network reactive-transport model.

Runs the lateral three-layer hydrologic structure (snow, surface, and
groundwater layers, each with a hillslope and a riparian column) together
with the large carbon network — SOC(s) -> DOC, SOC-lz(s) + DOC -> CO2(aq),
Calcite(s) -> Ca++ + CO3--, and the XDOC adsorption equilibrium — forward
over the full record with the parameter set calibrated in the showcase
project.

The forcing + observation series in ``inputs/east_river_series.csv`` carry
daily forcing from 2014-10-01 (water years 2015-2021).  Because no forcing
exists for water years 2013-2014, the record is spun up with a two-year
prefix: the first two water years of the forcing are repeated, shifted two
years earlier (water years 2013-2014), and all measured values in that
prefix are set to NaN, exactly as in the showcase calibration pipeline.
The full record therefore runs 2012-10-01 through 2021-09-30 (3287 days).
Each measured series is scored over its own available window, because the
objective functions skip the NaN spin-up steps.

The calibrated parameter set (83 values: 19 hydrological, per-zone water
dimensions and mineral kinetics, plus four log10 exchange-site totals) is
remapped onto this package's 85-value layout: each zone's
``(max_water, passive_water_storage)`` pair becomes
``(porosity=1.0, depth=max_water, passive)``.  With porosity 1.0 and zero
passive storage this preserves the reference model's water volume
(``max_water_volume = porosity * depth - passive``) and its mineral
scaling (by zone depth) exactly.

Outputs (default ``outputs/east_river_big_network/``):

- ``simulation_riv.csv``: flow-weighted river concentrations for every
  network species, the derived DIC, and the simulated/measured streamflow,
  over the full record.
- ``comparison.csv``: daily simulated vs measured streamflow, DOC, DIC, and
  calcium over the forcing window (water years 2015-2021).
- ``objective_functions.csv``: the standard performance metrics per series.
- ``sim_vs_obs.png``: observed-vs-simulated panels for the four series.

Usage (from the repository root)::

    .venv-east-river/bin/python scripts/east_river_big_network.py \
        [--input CSV] [--params CSV] [--output-dir DIR] [--days N] [--no-plot]

``--days N`` truncates the record to its first N days, for a short pilot
run to estimate throughput; ``--no-plot`` skips the figure.
"""

from __future__ import annotations

import os

# This machine is shared (a vLLM service on the GPU plus CPU bursts).  Pin
# the BLAS thread pools and disable CUDA before numpy / torch load.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import time
from pathlib import Path
from typing import Callable, Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame, Series
from scipy.optimize import root_scalar

from potions import (
    ChemicalDatabase,
    ChemicalState,
    ForcingData,
    LateralThreeLayerModel,
    ReactionNetwork,
    RtZoneConfiguration,
)

# -------------------------------------------------------------------------- #
# Constants
# -------------------------------------------------------------------------- #

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

DEFAULT_INPUT: Final[Path] = REPO_ROOT / "inputs" / "east_river_series.csv"
DEFAULT_PARAMS_CSV: Final[Path] = REPO_ROOT / "scripts" / "final_params.csv"
DEFAULT_OUTPUT_DIR: Final[Path] = REPO_ROOT / "outputs" / "east_river_big_network"

#: Network species, in the network's internal order: primary aqueous,
#: secondary (speciation) species, minerals, then exchange species.
PRIMARY_AQUEOUS: Final[tuple[str, ...]] = ("DOC", "Ca++", "HCO3-", "H+")
SECONDARY: Final[tuple[str, ...]] = ("CO2(aq)", "CO3--")
MINERALS: Final[tuple[str, ...]] = ("SOC(s)", "SOC-lz(s)", "Calcite(s)")
#: Database kinetics labels paired with :data:`MINERALS` (Monod "test" for
#: the SOC species, TST "default" for calcite — the showcase's selection).
MINERAL_KINETICS_LABELS: Final[tuple[str, ...]] = ("test", "test", "default")
EXCHANGE: Final[tuple[str, ...]] = ("XDOC",)

#: Derived dissolved inorganic carbon (showcase convention): the sum of the
#: aqueous carbonate species in the river series.
DIC_NAME: Final[str] = "DIC"
DIC_COMPONENTS: Final[tuple[str, ...]] = ("CO2(aq)", "HCO3-", "CO3--")

#: Flat zone order of the lateral three-layer model.
ZONE_ORDER: Final[tuple[str, ...]] = (
    "snow_hs",
    "snow_rp",
    "surface_hs",
    "surface_rp",
    "ground_hs",
    "ground_rp",
)
#: Zones that run mineral kinetics and speciation (the reactive zones).
REACTIVE_ZONES: Final[frozenset[str]] = frozenset(
    {"surface_hs", "surface_rp", "ground_hs", "ground_rp"}
)
#: Zones that carry the calibrated exchange-site pools, in the order the
#: showcase parameter set stores them.
SPECIATION_ZONES: Final[tuple[str, ...]] = (
    "surface_hs",
    "surface_rp",
    "ground_hs",
    "ground_rp",
)

#: Number of days in the two-year spin-up prefix (2 x 365, showcase
#: convention).
SPINUP_DAYS: Final[int] = 730
#: First day of the forcing record (the start of water year 2015).
RECORD_START: Final[str] = "2014-10-01"

# Initial-state conventions (showcase ``arr_to_results``).
INIT_AQ_CONC: Final[float] = 1e-3
MINERAL_CONCENTRATION: Final[float] = 1.0
#: Precipitation chemistry fill value (showcase ``get_precip_conc``).
PRECIP_CONC_FILL: Final[float] = 1e-20

#: The 19 hydrological values, in the showcase (and this package's)
#: ``from_array`` order: the zone parameters in flat structure order, then
#: the single proportion parameter.
OLD_HYDRO_NAMES: Final[tuple[str, ...]] = (
    "snow_hs.tt",
    "snow_hs.fmax",
    "snow_rp.tt",
    "snow_rp.fmax",
    "surface_hs.fc",
    "surface_hs.lp",
    "surface_hs.beta",
    "surface_hs.k0",
    "surface_hs.thr",
    "surface_rp.fc",
    "surface_rp.lp",
    "surface_rp.beta",
    "surface_rp.k0",
    "surface_rp.thr",
    "ground_hs.k",
    "ground_hs.alpha",
    "ground_rp.k",
    "ground_rp.alpha",
    "proportion.1",
)
#: Mineral kinetic parameters per zone, in the showcase CSV's order.  The
#: last entry (the CSV's "reactivity") is the log10 specific surface area,
#: which this package's ``MineralParameters`` decodes with ``10 **`` when
#: ``natural_scales`` is set.
OLD_MINERAL_PARAM_NAMES: Final[tuple[str, ...]] = (
    "sw_threshold",
    "sw_exp",
    "q_10",
    "reactivity",
)
#: Old per-zone water dimension names (max water volume, passive storage).
OLD_DIM_NAMES: Final[tuple[str, ...]] = ("max_water", "passive_water_storage")
#: Number of exchange-site (initial state) values in the showcase set: one
#: log10 total-site pool per speciation zone.
NUM_INIT_STATE_PARAMS: Final[int] = len(SPECIATION_ZONES)


# -------------------------------------------------------------------------- #
# Forcing + measurement record
# -------------------------------------------------------------------------- #


def load_site_csv(path: Path) -> DataFrame:
    """Load the forcing + observation CSV with a daily datetime index.

    Args:
        path: path of the CSV with columns ``date, ppt, temp, pet, q_meas,
            doc_molL, dic_molL, ca_molL``.

    Returns:
        The frame indexed by ascending, unique dates.

    Raises:
        ValueError: if any dates are duplicated.
    """
    raw: DataFrame = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    raw = raw.sort_index()
    if raw.index.duplicated().any():
        raise ValueError(f"Duplicate dates in {path}")
    return raw


def build_full_record(raw: DataFrame) -> DataFrame:
    """Assemble the full record with the two-year spin-up prefix.

    Reindexes the raw series onto the daily forcing window
    (2014-10-01 through the last day with forcing; measurements outside
    that window are dropped, matching the showcase left-merge), then
    prepends a copy of the first two water years shifted two years earlier
    with every measured value set to NaN — the showcase spin-up.

    Args:
        raw: raw CSV frame from :func:`load_site_csv`.

    Returns:
        The full record (730-day spin-up + forcing window) with columns
        ``ppt``, ``temp``, ``pet``, ``q_meas``, ``DOC``, ``DIC``, ``Ca++``.

    Raises:
        ValueError: if any forcing column has a gap inside the record
            window.
    """
    forcing_end: pd.Timestamp = raw.loc[raw["ppt"].notna()].index.max()
    record_index: pd.DatetimeIndex = pd.date_range(
        RECORD_START, forcing_end, freq="D"
    )
    comb: DataFrame = raw.reindex(record_index).rename(
        columns={"doc_molL": "DOC", "dic_molL": "DIC", "ca_molL": "Ca++"}
    )
    for column in ("ppt", "temp", "pet", "q_meas"):
        missing: int = int(comb[column].isna().sum())
        if missing > 0:
            raise ValueError(
                f"Forcing column '{column}' has {missing} missing day(s) "
                f"inside the {RECORD_START} -> {forcing_end.date()} window"
            )

    spinup: DataFrame = comb.iloc[0:SPINUP_DAYS].copy()
    spinup.index = spinup.index - pd.Timedelta(days=SPINUP_DAYS)
    for column in ("q_meas", "DOC", "DIC", "Ca++"):
        spinup[column] = np.nan
    return pd.concat([spinup, comb])


def describe_record(full: DataFrame, spinup_days: int) -> None:
    """Print the record's extent and each measured series' available window."""
    forcing_days: int = len(full) - spinup_days
    print(f"Record: {full.index.min().date()} -> {full.index.max().date()} "
          f"({len(full)} days; spin-up {spinup_days} days + "
          f"{forcing_days} days of forcing)")
    for column in ("q_meas", "DOC", "DIC", "Ca++"):
        available: Series = full[column].dropna()
        if available.empty:
            print(f"  {column:8s}: no observations in record window")
        else:
            print(
                f"  {column:8s}: {len(available)} values, "
                f"{available.index.min().date()} -> {available.index.max().date()}"
            )


# -------------------------------------------------------------------------- #
# Reaction network
# -------------------------------------------------------------------------- #


def build_network() -> ReactionNetwork:
    """Assemble the SOC / SOC-lz / calcite network (showcase
    ``get_soc_calcite_network``).

    Draws the species and their kinetics from the default chemical
    database: primary aqueous DOC, Ca++, HCO3-, H+; secondary CO2(aq) and
    CO3-- (the speciation partners of the DOC-consuming and calcite
    reactions); the three kinetic minerals (Monod for the SOC species, TST
    for calcite); and the XDOC adsorption exchange.

    Returns:
        The assembled network (11 species in total).
    """
    db: ChemicalDatabase = ChemicalDatabase.load_default()
    return ReactionNetwork(
        primary_aqueous=db.get_primary_aqueous_species(list(PRIMARY_AQUEOUS)),
        secondary=db.get_secondary_species(list(SECONDARY)),
        mineral=db.get_mineral_species(list(MINERALS)),
        mineral_kinetics=db.get_mineral_reactions(
            list(MINERALS), labels=list(MINERAL_KINETICS_LABELS)
        ),
        exchange_species=db.get_exchange_reactions(list(EXCHANGE)),
    )


# -------------------------------------------------------------------------- #
# Parameter set
# -------------------------------------------------------------------------- #


def load_final_params(path: Path) -> dict[str, float]:
    """Read the showcase's final calibrated parameter CSV.

    Args:
        path: the ``final_params.csv`` produced by the showcase
            calibration (one value per named parameter).

    Returns:
        Parameter name to value, as floats.
    """
    df: DataFrame = pd.read_csv(path, index_col=0)
    if df.shape[1] != 1:
        raise ValueError(
            f"Expected a single value column in {path}, got {df.shape[1]}"
        )
    values: NDArray[np.float64] = df.iloc[:, 0].to_numpy(dtype=np.float64)
    return {str(name): float(value) for name, value in zip(df.index, values)}


def remap_parameter_set(
    params: dict[str, float],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Remap the showcase's 83-value set onto this package's 85-value layout.

    The hydrological block (19 values) is unchanged.  Each zone's old
    ``(max_water, passive_water_storage)`` dimension pair is rewritten as
    this package's ``(porosity=1.0, depth=max_water, passive)`` triplet,
    which preserves the maximum water volume and the mineral concentration
    scale of the reference model (porosity 1.0 with zero passive storage
    leaves the moisture fraction and the mineral scaling invariant).  The
    per-mineral kinetic blocks and the trailing four log10 exchange-site
    totals are carried over unchanged.

    Args:
        params: named showcase parameters from :func:`load_final_params`.

    Returns:
        ``(model_vector, init_state_vector)``: the 85-value flat model
        parameter vector (``Model.from_array`` layout) and the four log10
        total exchange-site concentrations (one per speciation zone, in
        :data:`SPECIATION_ZONES` order).

    Raises:
        KeyError: if the CSV is missing any expected parameter name.
        ValueError: if the CSV carries unexpected parameter names or the
            remapped model vector has the wrong length.
    """
    expected: set[str] = set(OLD_HYDRO_NAMES)
    for zone in ZONE_ORDER:
        expected.update(f"{zone}.{name}" for name in OLD_DIM_NAMES)
        if zone in REACTIVE_ZONES:
            expected.update(
                f"{zone}.{mineral}.{name}"
                for mineral in MINERALS
                for name in OLD_MINERAL_PARAM_NAMES
            )
    expected.update(f"initial_state.{zone}_x" for zone in SPECIATION_ZONES)
    unexpected: set[str] = set(params) - expected
    if unexpected:
        raise ValueError(f"Unexpected parameter names in the set: {sorted(unexpected)}")

    parts: list[NDArray[np.float64]] = [
        np.array([params[name] for name in OLD_HYDRO_NAMES], dtype=np.float64)
    ]
    for zone in ZONE_ORDER:
        max_water: float = params[f"{zone}.{OLD_DIM_NAMES[0]}"]
        passive: float = params[f"{zone}.{OLD_DIM_NAMES[1]}"]
        parts.append(np.array([1.0, max_water, passive], dtype=np.float64))
        if zone in REACTIVE_ZONES:
            mineral_block: list[float] = []
            for mineral in MINERALS:
                mineral_block.extend(
                    params[f"{zone}.{mineral}.{name}"] for name in OLD_MINERAL_PARAM_NAMES
                )
            parts.append(np.array(mineral_block, dtype=np.float64))
    model_vector: NDArray[np.float64] = np.concatenate(parts)

    init_state_vector: NDArray[np.float64] = np.array(
        [params[f"initial_state.{zone}_x"] for zone in SPECIATION_ZONES],
        dtype=np.float64,
    )
    if model_vector.shape[0] != 85:
        raise ValueError(
            f"Remapped model vector has {model_vector.shape[0]} values, expected 85"
        )
    return model_vector, init_state_vector


# -------------------------------------------------------------------------- #
# Initial states
# -------------------------------------------------------------------------- #


def build_zone_configs() -> dict[str, RtZoneConfiguration]:
    """Per-zone reactive-transport configuration for the big network.

    Both snow zones are inert (no mineral kinetics, no speciation); each of
    the four surface and ground zones runs the three-mineral kinetics and
    the XDOC speciation equilibrium.
    """
    return {
        "snow_hs": RtZoneConfiguration(do_reactions=False, do_speciation=False),
        "snow_rp": RtZoneConfiguration(do_reactions=False, do_speciation=False),
        "surface_hs": RtZoneConfiguration(do_reactions=True, do_speciation=True),
        "surface_rp": RtZoneConfiguration(do_reactions=True, do_speciation=True),
        "ground_hs": RtZoneConfiguration(do_reactions=True, do_speciation=True),
        "ground_rp": RtZoneConfiguration(do_reactions=True, do_speciation=True),
    }


def build_initial_states(
    network: ReactionNetwork,
    log10_site_totals: NDArray[np.float64],
    mean_doc: float,
) -> dict[str, ChemicalState]:
    """Build the per-zone initial chemical states (showcase ``arr_to_results``).

    Every zone starts with the mean measured DOC and a trace 1e-3 mol/L for
    all other aqueous species.  The reactive zones start at the fixed
    mineral concentration (1.0 mol/L).  The speciation zones additionally
    partition the calibrated total exchange-site pool between free ``X-``
    and adsorbed ``XDOC`` at the XDOC equilibrium, using a small
    root-find on the exchange residual (the showcase mechanism).

    Args:
        network: the reaction network (fixes the species and the exchange
            equilibrium constant).
        log10_site_totals: total exchange-site pool per speciation zone,
            log10 mol/L, in :data:`SPECIATION_ZONES` order.
        mean_doc: mean measured DOC concentration [mol/L] over the record.

    Returns:
        Per-zone :class:`ChemicalState`, keyed by zone name in
        :data:`ZONE_ORDER`.
    """
    if len(log10_site_totals) != NUM_INIT_STATE_PARAMS:
        raise ValueError(
            f"Expected {NUM_INIT_STATE_PARAMS} log10 site totals, "
            f"got {len(log10_site_totals)}"
        )
    k_eq: float = 10.0 ** float(network.exchange_species[0].log10_k_eq)

    def exchange_residual(x_conc: float, tot_sites: float) -> float:
        """Exchange-equilibrium residual for a candidate ``X-`` concentration."""
        return k_eq * (tot_sites - x_conc) - x_conc * mean_doc

    states: dict[str, ChemicalState] = {}
    site_index: int = 0
    for zone in ZONE_ORDER:
        state: ChemicalState = ChemicalState()
        for primary in network.primary_names:
            state.primary[primary] = mean_doc if primary == "DOC" else INIT_AQ_CONC
        for secondary in network.secondary_names:
            state.secondary[secondary] = INIT_AQ_CONC
        if zone in REACTIVE_ZONES:
            for mineral in network.mineral_names:
                state.mineral[mineral] = MINERAL_CONCENTRATION
        if zone in SPECIATION_ZONES:
            tot_sites: float = 10.0 ** float(log10_site_totals[site_index])
            site_index += 1
            x_free: float = float(
                root_scalar(
                    lambda x: exchange_residual(x, tot_sites), x0=1e-5
                ).root
            )
            state.exchange["X-"] = x_free
            state.exchange["XDOC"] = tot_sites - x_free
        states[zone] = state
    return states


# -------------------------------------------------------------------------- #
# Objective functions (pandas-native mirrors of the golden formulas)
# -------------------------------------------------------------------------- #


def _kge(meas: Series, sim: Series) -> float:
    """Kling-Gupta efficiency (golden ``kge`` formula, NaN-skipping)."""
    r: float = float(sim.corr(meas, method="pearson"))
    mean_o: float = float(meas.mean())
    std_o: float = float(meas.std())
    mean_p: float = float(sim.mean())
    std_p: float = float(sim.std())
    alpha: float = std_p / std_o if std_o > 0.0 else float("nan")
    beta: float = mean_p / mean_o if mean_o != 0.0 else float("nan")
    return float(1.0 - ((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2) ** 0.5)


def _log_kge(meas: Series, sim: Series) -> float:
    """Kling-Gupta efficiency of the log10(x + 0.1) series."""
    return _kge(np.log10(meas + 0.1), np.log10(sim + 0.1))


def _nse(meas: Series, sim: Series) -> float:
    """Nash-Sutcliffe efficiency (NaN-skipping sums)."""
    numer: float = float(((sim - meas) ** 2).sum())
    denom: float = float(((meas - meas.mean()) ** 2).sum())
    if denom == 0.0:
        return float("nan")
    return 1.0 - numer / denom


def _log_nse(meas: Series, sim: Series) -> float:
    """Nash-Sutcliffe efficiency of the log10(x + 0.1) series."""
    return _nse(np.log10(meas + 0.1), np.log10(sim + 0.1))


def _bias(meas: Series, sim: Series) -> float:
    """Relative bias of the simulation against the measurements."""
    mean_o: float = float(meas.mean())
    if mean_o == 0.0:
        return float("nan")
    return float((sim - meas).mean() / mean_o)


def _spearman_rho(meas: Series, sim: Series) -> float:
    """Spearman rank correlation."""
    return float(meas.corr(sim, method="spearman"))


def _r_squared(meas: Series, sim: Series) -> float:
    """Pearson correlation squared."""
    return float(meas.corr(sim, method="pearson") ** 2)


#: Pandas-native objective functions for the RT run, with the golden names
#: and formulas.  These accept plain ``pandas.Series`` (the packaged
#: defaults are tensor-backed and do not run on the Series outputs).
RT_OBJECTIVE_FUNCTIONS: Final[
    list[tuple[str, Callable[[Series, Series], float]]]
] = [
    ("kge", _kge),
    ("log_kge", _log_kge),
    ("nse", _nse),
    ("log_nse", _log_nse),
    ("bias", _bias),
    ("spearman_rho", _spearman_rho),
    ("r_squared", _r_squared),
]


# -------------------------------------------------------------------------- #
# Outputs
# -------------------------------------------------------------------------- #


def simulated_dic(rt_simulation: DataFrame) -> Series:
    """Derived DIC river series: the sum of the aqueous carbonate series."""
    dic: Series = rt_simulation[f"{DIC_COMPONENTS[0]}_riv"]
    for component in DIC_COMPONENTS[1:]:
        dic = dic + rt_simulation[f"{component}_riv"]
    return dic


def write_outputs(
    output_dir: Path,
    full: DataFrame,
    network: ReactionNetwork,
    hydro: DataFrame,
    rt_simulation: DataFrame,
    dic_sim: Series,
    metrics: DataFrame,
    make_figure: bool,
) -> None:
    """Write the simulation CSVs, the metrics table, and the figure.

    Args:
        output_dir: directory to write into (created if needed).
        full: the full record with the measured series.
        network: the reaction network (fixes the species list).
        hydro: the hydrology simulation frame.
        rt_simulation: the reactive-transport simulation frame.
        dic_sim: derived DIC river series aligned to the full record.
        metrics: performance-metric table (one row per scored series).
        make_figure: whether to write ``sim_vs_obs.png``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    riv_columns: list[str] = [f"{species}_riv" for species in network.species_names]
    sim_riv: DataFrame = rt_simulation[riv_columns].copy()
    sim_riv[DIC_NAME] = dic_sim
    sim_riv["sim_streamflow_mmd"] = hydro["sim_streamflow_mmd"]
    sim_riv["meas_streamflow_mmd"] = hydro["meas_streamflow_mmd"]
    sim_riv.to_csv(output_dir / "simulation_riv.csv", index_label="date")

    record: DataFrame = full.loc[full.index >= RECORD_START]
    comparison: DataFrame = DataFrame(index=record.index)
    comparison["q_meas"] = record["q_meas"]
    comparison["q_sim"] = hydro.loc[record.index, "sim_streamflow_mmd"]
    comparison["DOC_meas"] = record["DOC"]
    comparison["DOC_sim"] = rt_simulation.loc[record.index, "DOC_riv"]
    comparison["DIC_meas"] = record["DIC"]
    comparison["DIC_sim"] = dic_sim.loc[record.index]
    comparison["Ca_meas"] = record["Ca++"]
    comparison["Ca_sim"] = rt_simulation.loc[record.index, "Ca++_riv"]
    comparison.to_csv(output_dir / "comparison.csv", index_label="date")

    metrics.to_csv(output_dir / "objective_functions.csv", index_label="series")

    if make_figure:
        make_plot(output_dir / "sim_vs_obs.png", comparison)


def make_plot(path: Path, comparison: DataFrame) -> None:
    """Write the observed-vs-simulated figure (four log-scale panels).

    Args:
        path: output PNG path.
        comparison: daily measured vs simulated series over the forcing
            window; NaN steps are skipped by the plotter.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    panels: list[tuple[str, str, str]] = [
        ("q_meas", "q_sim", "Streamflow [mm/day]"),
        ("DOC_meas", "DOC_sim", "DOC [mol/L]"),
        ("DIC_meas", "DIC_sim", "DIC [mol/L]"),
        ("Ca_meas", "Ca_sim", "Ca++ [mol/L]"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), sharex=True)
    for ax, (obs_col, sim_col, title) in zip(axes.ravel(), panels):
        ax.plot(
            comparison.index,
            comparison[obs_col],
            color="0.45",
            linewidth=0.8,
            label="observed",
        )
        ax.plot(
            comparison.index,
            comparison[sim_col],
            color="tab:blue",
            linewidth=0.8,
            label="simulated",
        )
        ax.set_yscale("log")
        # matplotlib >= 3.10 plots datetimes on the linear scale; pick date
        # tick formatting rather than the removed "date" scale.
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


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
        description="East River large-network (SOC/SOC-lz/calcite) forward run."
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
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory for the output files (default: %(default)s)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=0,
        help="pilot: truncate the record to its first N days (0 = full record)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="skip the observed-vs-simulated figure",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the full simulation and write the outputs.

    Args:
        argv: command line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 on success).
    """
    args: argparse.Namespace = parse_args(argv)
    output_dir: Path = args.output_dir

    print(f"[1/6] Loading forcing + observations: {args.input}")
    raw: DataFrame = load_site_csv(args.input)
    full_record: DataFrame = build_full_record(raw)
    # The mean measured DOC (used for the initial states) is taken over the
    # full record, not a truncated pilot: the observations start a year into
    # the forcing window, so any early pilot slice would be all-NaN.
    mean_doc: float = float(full_record["DOC"].mean())
    if args.days > 0:
        full: DataFrame = full_record.iloc[: SPINUP_DAYS + args.days]
        print(f"  PILOT: forcing window truncated to the first {args.days} days")
    else:
        full = full_record
    describe_record(full, SPINUP_DAYS)

    print("[2/6] Assembling the reaction network")
    network: ReactionNetwork = build_network()
    print(f"  {network.num_species} species: {', '.join(network.species_names)}")

    print(f"[3/6] Loading the calibrated parameter set: {args.params}")
    params: dict[str, float] = load_final_params(args.params)
    model_vector, init_state_vector = remap_parameter_set(params)
    print(
        f"  remapped {len(params)}-value showcase set -> "
        f"{model_vector.shape[0]}-value model vector + "
        f"{init_state_vector.shape[0]} initial-state values"
    )

    print(f"[4/6] Building the model (mean measured DOC = {mean_doc:.3e} mol/L)")
    zone_config: dict[str, RtZoneConfiguration] = build_zone_configs()
    model: LateralThreeLayerModel = LateralThreeLayerModel.from_array(
        model_vector,
        network=network,
        config=zone_config,
        natural_scales=True,
    )
    initial_states: dict[str, ChemicalState] = build_initial_states(
        network, init_state_vector, mean_doc
    )
    precip_conc: NDArray[np.float64] = np.full(
        (2, len(full), network.num_species), PRECIP_CONC_FILL, dtype=np.float64
    )
    forcing: ForcingData = ForcingData(
        precip=full["ppt"], temp=full["temp"], pet=full["pet"]
    )

    num_steps: int = len(full)
    num_zones: int = model.get_num_zones()

    print(f"[5/6] Hydrology run: {num_steps} days, {num_zones} zones")
    start: float = time.perf_counter()
    hydro_res = model.run_hydro_model(forcing, meas_streamflow=full["q_meas"])
    hydro_seconds: float = time.perf_counter() - start
    hydro_df: DataFrame = hydro_res.simulation
    print(
        f"  hydrology done in {hydro_seconds:.1f} s "
        f"({num_steps / hydro_seconds:.1f} days/s, "
        f"{num_steps * num_zones / hydro_seconds:.1f} zones solved per s)"
    )
    print(f"  streamflow metrics: "
          f"{', '.join(f'{k}={v:.4f}' for k, v in hydro_res.objective_functions.items())}")

    print(f"[6/6] Reactive-transport run: {num_steps} days x {num_zones} zones")
    meas_river_conc: DataFrame = DataFrame(
        {"DOC": full["DOC"], "Ca++": full["Ca++"]}
    )
    rt_start: float = time.perf_counter()
    rt_res = model.run_rt_model(
        hydro_res,
        precip_conc=precip_conc,
        initial_state=initial_states,
        meas_river_conc=meas_river_conc,
        objective_functions=list(RT_OBJECTIVE_FUNCTIONS),
        verbose=True,
    )
    rt_seconds: float = time.perf_counter() - rt_start
    rt_df: DataFrame = rt_res.simulation
    zones_per_second: float = num_steps * num_zones / rt_seconds
    print(
        f"  reactive transport done in {rt_seconds:.1f} s ({zones_per_second:.1f} "
        f"zones solved per s)"
    )

    dic_sim: Series = simulated_dic(rt_df)
    dic_metrics: dict[str, float] = {
        name: func(full["DIC"], dic_sim) for name, func in RT_OBJECTIVE_FUNCTIONS
    }

    metric_names: list[str] = [name for name, _ in RT_OBJECTIVE_FUNCTIONS]
    metrics: DataFrame = DataFrame(
        index=["streamflow", "DOC", DIC_NAME, "Ca++"], columns=metric_names
    )
    metrics.loc["streamflow"] = hydro_res.objective_functions
    if rt_res.objective_functions is not None:
        for species in rt_res.objective_functions.index:
            metrics.loc[str(species)] = rt_res.objective_functions.loc[species]
    metrics.loc[DIC_NAME] = pd.Series(dic_metrics)
    metrics = metrics[metric_names]

    print("\nObjective functions (each series scored over its own available window):")
    print(metrics.to_string(float_format=lambda v: f"{v:.4f}"))

    write_outputs(
        output_dir,
        full,
        network,
        hydro_df,
        rt_df,
        dic_sim,
        metrics,
        make_figure=not args.no_plot,
    )
    print(f"\nOutputs written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
