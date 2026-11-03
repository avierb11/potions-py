"""Model orchestration: hydrology + reactive-transport driver (golden ``model.py``).

This module adds the two orchestration layers on top of the validated
:class:`~potions.hydro_model.HydrologicalModel` and
:class:`~potions.rt_zone.RtZone` / :class:`~potions.rt_zone.RiverZone`
step machinery:

* :class:`ReactiveTransportModel` wraps an ``(RtZone, RiverZone)`` map and a
  :class:`~potions.reaction_network.ReactionNetwork` behind the hydrology
  model.  It builds the per-(zone, step) reactive-transport forcing
  (:meth:`construct_rt_forcing_matrix`), advances the whole multi-zone system
  forward one day at a time with the golden Gauss-Seidel zone coupling
  (:meth:`step_rt_model`), and assembles the per-species / per-zone output
  ``DataFrame`` plus the flow-weighted river concentrations and the species
  mixing metrics (:meth:`run_rt_model`).

* :class:`Model` is the calibration-facing subclass: it derives parameter
  counts and zone names from the ``structure`` class attribute, flattens /
  restores full parameter vectors (``to_array`` / ``from_array`` /
  ``hydro_from_series`` / ``rt_params_from_array``), and runs parameter
  batches (:meth:`run_batch`).

The Gauss-Seidel coupling reproduces the Rust ``run_zone_steps`` exactly:

* Zones are stepped in flat (evaluation) order.
* *Surface* zones keep the precipitation concentration as their incoming
  concentration ``conc_in`` (no coupling step).
* Every other zone receives ``mass_in = lat_mat[i] @ lat_mass +
  vert_mat[i] @ vert_mass`` from the zones already solved this step; divided by
  ``tot_water_in = q_in + forc_flux`` when ``tot_water_in > 1e-6`` and taken as
  the all-zero vector otherwise.
* Each zone's ``lat_mass`` / ``vert_mass`` are accumulated so downstream zones
  can consume them.

The outlet chemistry uses the same flow-weighting as the reference: the
concentration at the simulated streamflow is the flux-weighted mean of the
per river-zone concentrations, ``sum_n (q_lat_n / q_sim) * c_n``, with the
river zones being the last zone of every layer.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from typing import Any, Callable, ClassVar, Iterable, cast

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from pandas import DataFrame, Series
from torch import Tensor

from .common_types import (
    BatchResults,
    ChemicalState,
    ForcingData,
    HydroModelResults,
    HydroStep,
    ModelResults,
    RtForcing,
    RtModelResults,
    RtStep,
    RtZoneConfiguration,
    Scalar,
)
from .hydro import HydrologicZone
from .hydro_model import HydrologicalModel, LayerT
from .kinetic_structures import PARAMETERS_PER_MINERAL, RtParameters
from .objective_functions import DEFAULT_OBJECTIVE_FUNCTIONS
from .reaction_network import ReactionNetwork
from .rt_zone import RiverZone, RtZone, WaterVolumeError

__all__ = [
    "ReactiveTransportModel",
    "Model",
    "calculate_moisture_fraction",
    "calculate_water_table_depth",
    "get_hydro_steps",
    "WaterVolumeError",
]


# ---------------------------------------------------------------------- #
# Shared helpers (golden ``reactive_transport/rt_zone.py``)
# ---------------------------------------------------------------------- #


def calculate_moisture_fraction(
    zone_params: dict[str, RtZone],
    sim_res: DataFrame,
    raise_err: bool = True,
) -> NDArray[np.float64]:
    """Soil-water content ``(storage + passive) / (v_max + passive)`` per zone.

    Mirrors the golden helper: rows are zones (dict order), columns are time
    steps, and a zone whose simulated storage reaches its maximum water volume
    raises :class:`WaterVolumeError`.

    Args:
        zone_params: reactive-transport zone map keyed by zone name.
        sim_res:     hydrological simulation ``DataFrame`` (``s_<zone>`` cols).
        raise_err:   raise :class:`WaterVolumeError` on exceeded volume.

    Returns:
        ``(num_zones, num_steps)`` moisture-fraction array.
    """
    sw_vals: NDArray[np.float64] = np.zeros((len(zone_params), len(sim_res)))
    for i, (zone_name, zone) in enumerate(zone_params.items()):
        col_name: str = f"s_{zone_name}"
        max_water_volume: float = zone.parameters.dimensions.max_water_volume
        storage: Series = sim_res[col_name]

        if storage.max() >= max_water_volume:
            if raise_err:
                raise WaterVolumeError(
                    f"Water volume error in zone '{zone_name}': Maximum volume "
                    f"is {round(max_water_volume, 1)} mm but maximum simulated "
                    f"storage is {round(float(storage.max()), 1)} mm. Increase "
                    "porosity or volume"
                )
            continue

        sw_vals[i] = (
            storage.to_numpy() + zone.parameters.dimensions.passive_water_storage
        ) / (max_water_volume + zone.parameters.dimensions.passive_water_storage)

    return sw_vals


def calculate_water_table_depth(
    zone_params: dict[str, RtZone],
    sim_res: DataFrame,
) -> NDArray[np.float64]:
    """Water-table depth ``depth - (storage + passive) / porosity`` per zone.

    Args:
        zone_params: reactive-transport zone map keyed by zone name.
        sim_res:     hydrological simulation ``DataFrame`` (``s_<zone>`` cols).

    Returns:
        ``(num_zones, num_steps)`` water-table-depth array [mm].
    """
    zw_vals: NDArray[np.float64] = np.zeros((len(zone_params), len(sim_res)))
    for i, (zone_name, zone) in enumerate(zone_params.items()):
        col_name: str = f"s_{zone_name}"
        storage: Series = sim_res[col_name]
        zw_vals[i] = (
            zone.parameters.dimensions.depth
            - (storage.to_numpy() + zone.parameters.dimensions.passive_water_storage)
            / zone.parameters.dimensions.porosity
        )
        if zw_vals[i].min() < 0:
            raise WaterVolumeError("Illegal water volume")

    return zw_vals


def get_hydro_steps(sim_res: DataFrame) -> NDArray[np.object_]:
    """Build per-(zone, step) :class:`~potions.common_types.HydroStep` records.

    Each row holds the hydrological fluxes of one zone across all time steps,
    read from the ``s_<zone>`` / ``q_<flux>_<zone>`` simulation columns.

    Args:
        sim_res: hydrological simulation ``DataFrame``.

    Returns:
        ``(num_zones, num_steps)`` object array of :class:`HydroStep`.
    """
    zone_names: list[str] = [
        str(c).replace("s_", "") for c in sim_res.columns if str(c).startswith("s_")
    ]
    hydro_steps: NDArray[np.object_] = np.empty(
        (len(zone_names), len(sim_res)), dtype=object
    )
    for i, z in enumerate(zone_names):
        state: NDArray[np.float64] = sim_res[f"s_{z}"].to_numpy()
        q_forc: NDArray[np.float64] = sim_res[f"q_forc_{z}"].to_numpy()
        q_vap: NDArray[np.float64] = sim_res[f"q_vap_{z}"].to_numpy()
        q_lat: NDArray[np.float64] = sim_res[f"q_lat_{z}"].to_numpy()
        q_lat_ext: NDArray[np.float64] = sim_res[f"q_lat_ext_{z}"].to_numpy()
        q_vert: NDArray[np.float64] = sim_res[f"q_vert_{z}"].to_numpy()
        q_vert_ext: NDArray[np.float64] = sim_res[f"q_vert_ext_{z}"].to_numpy()
        q_in: NDArray[np.float64] = sim_res[f"q_in_{z}"].to_numpy()

        for j in range(len(sim_res)):
            hydro_steps[i, j] = HydroStep(
                state=float(state[j]),
                forc_flux=float(q_forc[j]),
                lat_flux=float(q_lat[j]),
                vert_flux=float(q_vert[j]),
                vap_flux=float(q_vap[j]),
                q_in=float(q_in[j]),
                lat_flux_ext=float(q_lat_ext[j]),
                vert_flux_ext=float(q_vert_ext[j]),
            )
    return hydro_steps


def _as_float(x: Scalar) -> float:
    """Coerce a python / numpy scalar or 0-dim tensor to ``float``."""
    if isinstance(x, (int, float)):
        return float(x)
    return float(x.item())


def _to_series(x: Tensor) -> Series:
    """Convert a tensor to a float ``pandas.Series`` (ddof-1 semantics)."""
    return pd.Series(x.detach().cpu().numpy(), dtype=float)


def _adapt_objective_functions(
    objective_functions: list[tuple[str, Callable[[Series, Series], float]]],
) -> list[tuple[str, Callable[[Tensor, Tensor], float]]]:
    """Wrap pandas-Series objective functions so they accept tensors.

    :meth:`~potions.hydro_model.HydrologicalModel.run_hydro_model` evaluates
    objectives as ``fn(meas_tensor, sim_tensor)``; the RT objective functions
    are written against pandas ``Series``.  This adapter bridges the two.
    """

    def make_wrapper(
        fn: Callable[[Series, Series], float],
    ) -> Callable[[Tensor, Tensor], float]:
        def wrapped(meas_t: Tensor, sim_t: Tensor) -> float:
            return float(fn(_to_series(meas_t), _to_series(sim_t)))

        return wrapped

    return [(name, make_wrapper(fn)) for name, fn in objective_functions]


def _override_zones(
    structure: list[LayerT], zones: dict[str, HydrologicZone] | None
) -> list[LayerT]:
    """Return ``structure`` with any same-named zones replaced by ``zones``."""
    if zones is None:
        return structure
    out: list[LayerT] = []
    for layer in structure:
        out.append([zones.get(zone.name, zone) for zone in layer])
    return out


# ---------------------------------------------------------------------- #
# ReactiveTransportModel
# ---------------------------------------------------------------------- #


class ReactiveTransportModel(HydrologicalModel):
    """A hydrological model with reactive-transport soil and river zones.

    Subclasses (see :class:`Model`) provide a class-level ``structure``; a
    ``ReactiveTransportModel`` can also be constructed directly from an
    explicit ``structure``.

    Args:
        structure: nested ``(layer -> zones)`` hydrology structure.
        zones:     optional per-name zone overrides for ``structure``.
        scales:    optional relative surface-zone areas (sum to 1).
        network:   the :class:`ReactionNetwork` for the RT simulation.
        rt_zones:  per-name soil :class:`RtZone` map (must cover ``structure``).
        river_zone: optional :class:`RiverZone` for the outlet chemistry.
        verbose:   print extra diagnostics in ``__init__``.
    """

    def __init__(
        self,
        structure: list[LayerT] | None = None,
        zones: dict[str, HydrologicZone] | None = None,
        scales: Iterable[float] | None = None,
        network: ReactionNetwork | None = None,
        rt_zones: dict[str, RtZone] | None = None,
        river_zone: RiverZone | None = None,
        verbose: bool = False,
    ) -> None:
        if structure is None:
            structure = []
        super().__init__(structure=_override_zones(structure, zones), scales=scales, verbose=verbose)

        # Make sure that all reactive-transport arguments are present together.
        if (network is not None) == (rt_zones is not None):
            if verbose:
                print("Using reactive transport model")
        else:
            raise ValueError(
                "When doing reactive transport simulation, you must pass both "
                "`network` and `rt_zones` parameters, not just one of them"
            )

        if (network is not None) and (rt_zones is not None):
            self._rt_zones: dict[str, RtZone] = rt_zones
            self._rt_params: dict[str, RtParameters] = {
                zone_name: zone.parameters for zone_name, zone in rt_zones.items()
            }
            self._has_rt: bool = True
            self._network: ReactionNetwork = network

            # Make sure the RT zone map exactly covers the hydrology zones.
            model_zone_names: set[str] = set(self.zone_names())
            rt_zone_names: set[str] = set(rt_zones.keys())
            if model_zone_names != rt_zone_names:
                raise ValueError(
                    f"Incorrect structure for `rt_params`. Expected dictionary "
                    f"with keys {sorted(model_zone_names)}, got keys "
                    f"{sorted(rt_zone_names)}"
                )

            rt_configuration: dict[str, RtZoneConfiguration] = {
                zone_name: RtZoneConfiguration(
                    do_reactions=zone.do_reactions, do_speciation=zone.do_speciation
                )
                for zone_name, zone in rt_zones.items()
            }
            self._zone_configs: dict[str, RtZoneConfiguration] = {
                zone_name: rt_configuration.get(
                    zone_name, RtZoneConfiguration(do_reactions=True, do_speciation=True)
                )
                for zone_name in self.zone_names()
            }
        else:
            self._has_rt = False
            self._rt_zones = {}
            self._rt_params = {}
            self._zone_configs = {}

        self._river_zone: RiverZone | None = river_zone

    # ---- properties -------------------------------------------------- #

    @property
    def reaction_network(self) -> ReactionNetwork | None:
        """The reaction network, or ``None`` for hydrology-only models."""
        if self._has_rt:
            return self._network
        print("This model does not have reactive transport capabilities")
        return None

    @property
    def network(self) -> ReactionNetwork:
        """The reaction network (raises for hydrology-only models)."""
        if self._has_rt:
            return self._network
        raise ValueError("Model does not have reactive-transport capabilities")

    @property
    def rt_zones(self) -> dict[str, RtZone]:
        """Per-zone soil reactive-transport zone map."""
        return self._rt_zones

    @property
    def has_river_zone(self) -> bool:
        """True when the model carries an outlet :class:`RiverZone`."""
        return self._river_zone is not None

    @property
    def river_zone(self) -> RiverZone:
        """The outlet :class:`RiverZone` (raises when absent)."""
        if self._river_zone is not None:
            return self._river_zone
        raise ValueError("No river zone in this model")

    @property
    def zone_configurations(self) -> dict[str, RtZoneConfiguration]:
        """Per-zone ``(do_reactions, do_speciation)`` flags."""
        return self._zone_configs

    # ---- parameter plumbing (RT part) -------------------------------- #

    def rt_to_array(self) -> NDArray[np.float64]:
        """Flatten the RT zone + river parameters into a single array."""
        components: list[NDArray[np.float64]] = []
        for zone_name in self.zone_names():
            components.append(self.rt_zones[zone_name].to_array().detach().numpy())
        if self.has_river_zone:
            components.append(self.river_zone.to_array().detach().numpy())
        return np.concatenate(components)

    def hydro_to_array(self) -> NDArray[np.float64]:
        """Flatten the hydrological zone + size (proportion) parameters."""
        param_list: list[float] = []
        for zone in self.flat_model:
            param_list += [float(v) for v in zone.param_list()]
        if len(self.scales) > 1:
            param_list += self.scales[:-1]
        return np.array(param_list, dtype=np.float64)

    def to_array(self) -> NDArray[np.float64]:
        """Flatten the combined hydro + RT parameter vector."""
        return np.concatenate([self.hydro_to_array(), self.rt_to_array()])

    # ---- forcing ------------------------------------------------------ #

    def construct_rt_forcing_matrix(
        self,
        forc: ForcingData | Iterable[ForcingData],
        hydro_sim_df: DataFrame,
        precip_conc: NDArray[np.float64],
        model_rt_params: dict[str, RtZone],
    ) -> NDArray[np.object_]:
        """Build the per-(zone, step) reactive-transport forcing matrix.

        ``precip_conc`` has shape ``(num_surface_zones, num_steps, num_species)``
        and gives the precipitation chemistry of every surface zone at every
        step; it is written into the incoming concentration of the matching
        surface-zone forcing records.  Soil-water content and water-table depth
        are derived from the hydrological simulation.

        Args:
            forc: hydrological forcing (single series or one per surface zone).
            hydro_sim_df: hydrological simulation ``DataFrame``.
            precip_conc: ``(surface_zone, step, species)`` precipitation chemistry.
            model_rt_params: per-zone :class:`RtZone` map.

        Returns:
            ``(num_zones, num_steps)`` object array of :class:`RtForcing`.
        """
        sw_vals: NDArray[np.float64] = calculate_moisture_fraction(model_rt_params, hydro_sim_df)
        zw_vals: NDArray[np.float64] = calculate_water_table_depth(model_rt_params, hydro_sim_df)
        hydro_steps: NDArray[np.object_] = get_hydro_steps(hydro_sim_df)
        hydro_forcing: list[list[HydroStep]] = self.construct_hydro_forcing_matrix(
            forc
        )

        num_zones: int = self.num_zones
        num_steps: int = len(hydro_sim_df)

        rt_forcings: NDArray[np.object_] = np.empty((num_zones, num_steps), dtype=object)

        for i in range(num_zones):
            for j in range(num_steps):
                rt_forcings[i, j] = RtForcing(
                    conc_in=torch.zeros(self.network.num_species, dtype=torch.float64),
                    hydro_step=hydro_steps[i, j],
                    hydro_forc=hydro_forcing[j][i],
                    s_w=float(sw_vals[i, j]),
                    z_w=float(zw_vals[i, j]),
                )

        # Insert the precipitation chemistry for every surface zone.
        if precip_conc.shape[0] != len(self.surface_zone_ids):
            raise ValueError(
                f"The first dimension of precipitation chemistry must be the "
                f"same as the number of surface zones: {self.num_surface_zones}, "
                f"not {precip_conc.shape[0]}"
            )
        if precip_conc.shape[1] != len(hydro_sim_df):
            raise ValueError(
                f"The second dimension of precipitation chemistry must be the "
                f"same as the number of simulation steps: {len(hydro_sim_df)}, "
                f"not {precip_conc.shape[1]}"
            )

        for i, zone_data in enumerate(precip_conc):
            for j, species_conc in enumerate(zone_data):
                rt_forcings[i, j] = replace(
                    rt_forcings[i, j],
                    conc_in=torch.tensor(species_conc, dtype=torch.float64),
                )

        return rt_forcings

    # ---- forward step ------------------------------------------------- #

    def _failed_repro_context(
        self,
        index: int,
        zone: RtZone,
        day_index: int,
        dt_days: float,
        s_i: Tensor,
        d_i: RtForcing,
        err: BaseException,
    ) -> dict[str, Any]:
        """Build a self-describing context dict for a failed zone step."""
        summary: str = (
            f"reactive-transport step failure: zone '{zone.name}' (index {index}) "
            f"at step {day_index} (dt={dt_days}d) | "
            f"do_reactions={zone.do_reactions}, do_speciation={zone.do_speciation} | "
            f"{type(err).__name__}: {err}"
        )
        return {
            "model_type": type(self).__name__,
            "zone_name": zone.name,
            "zone_index": index,
            "do_reactions": zone.do_reactions,
            "do_speciation": zone.do_speciation,
            "species_names": zone.all_species,
            "step_index": day_index,
            "dt_days": float(dt_days),
            "state_s_i": s_i.detach().cpu().numpy().astype(float).tolist(),
            "conc_in": d_i.conc_in.detach().cpu().numpy().astype(float).tolist(),
            "zone_parameter_array": zone.to_array().detach().numpy().astype(float).tolist(),
            "exception_type": type(err).__name__,
            "exception": str(err),
            "summary": summary,
        }

    def step_rt_model(
        self,
        model: dict[str, RtZone],
        state: Tensor,
        ds: Iterable[RtForcing],
        dt_days: float,
        verbose: bool = False,
        failed_dir: str | None = None,
        step_index: int = 0,
    ) -> list[RtStep]:
        """Advance every zone one RT step with Gauss-Seidel coupling.

        Mirrors the Rust ``run_zone_steps`` driver: a single flat sweep over
        the zones in evaluation order, each zone's incoming concentration
        computed from the upstream masses already produced this step (surface
        zones keep their precipitation concentration).

        Args:
            model: per-zone :class:`RtZone` map.
            state: ``(num_zones, num_species)`` current concentrations.
            ds:    one :class:`RtForcing` per zone for this step.
            dt_days: step length [days].
            verbose: print per-zone step diagnostics.
            failed_dir: when set, write a JSON repro file on failure.
            step_index: current step index (diagnostics only).

        Returns:
            One :class:`RtStep` per zone, in flat zone order.
        """
        zones: list[RtZone] = [model[name] for name in self.zone_names()]
        ds_list: list[RtForcing] = list(ds)
        n_zones: int = len(zones)
        if int(state.shape[0]) != n_zones:
            raise ValueError("`states` first dimension must equal the number of zones")
        if len(ds_list) != n_zones:
            raise ValueError("number of forcings must equal the number of zones")
        if n_zones == 0:
            return []

        n_spec: int = int(state.shape[1])
        surface: list[bool] = [i in self.surface_zone_ids for i in range(n_zones)]
        lat_mat: Tensor = self.lat_mat
        vert_mat: Tensor = self.vert_mat
        # Per-zone, per-species outgoing-mass accumulators (Gauss-Seidel coupling).
        # Row ``k`` records the lateral/vertical solute export of zone ``k`` once it
        # has been solved this step, so a downstream zone ``i`` aggregates over the
        # rows already filled: ``lat_mass.T @ lat_mat[i]`` (golden
        # ``run_zone_steps``). ``lat_mat`` / ``vert_mat`` are ``(n_zones, n_zones)``
        # connectivity rows, so the matmul yields the ``(n_spec,)`` mass arriving at
        # zone ``i``.
        lat_mass: Tensor = torch.zeros(n_zones, n_spec, dtype=torch.float64)
        vert_mass: Tensor = torch.zeros(n_zones, n_spec, dtype=torch.float64)

        out: list[RtStep] = []
        for i, zone in enumerate(zones):
            d: RtForcing = ds_list[i]
            assert d.hydro_step is not None, "RtForcing requires a hydro_step"
            tot_water_in: float = _as_float(d.hydro_step.q_in) + _as_float(
                d.hydro_step.forc_flux
            )
            conc_in: Tensor = d.conc_in
            if not surface[i]:
                mass_in: Tensor = (
                    lat_mass.T @ lat_mat[i] + vert_mass.T @ vert_mat[i]
                )
                if tot_water_in > 1e-6:
                    conc_in = mass_in / tot_water_in
                else:
                    conc_in = torch.zeros(n_spec, dtype=mass_in.dtype, device=mass_in.device)

            d_step: RtForcing = replace(d, conc_in=conc_in)
            s_i: Tensor = state[i].clone()

            try:
                step: RtStep = zone.step(s_i, d_step, dt_days, verbose=verbose)
            except Exception as err:
                d_repro: RtForcing = replace(d, conc_in=conc_in)
                context: dict[str, Any] = self._failed_repro_context(
                    i, zone, step_index, dt_days, s_i, d_repro, err
                )
                if failed_dir is not None:
                    os.makedirs(failed_dir, exist_ok=True)
                    with open(
                        os.path.join(failed_dir, f"step_{step_index}_zone_{i}.json"),
                        "w",
                    ) as f:
                        json.dump(context, f, default=str)
                print(f"Failed on zone {i} ('{zone.name}') step {step_index}: {err}")
                raise err

            lat_mass[i] = step.lat_mass
            vert_mass[i] = step.vert_mass
            out.append(step)

            if verbose:
                print(f"Incoming concentration to zone {i}: {conc_in}")
                print(f"Lateral mass transfer outwards: {step.lat_mass}")
                print(f"Vertical mass transfer outwards: {step.vert_mass}")
                print(f"Finished zone '{zone.name}', final concentration: {step.state}")

        return out

    # ---- full run ----------------------------------------------------- #

    def run_rt_model(
        self,
        # Forcing data
        hydro_res: HydroModelResults,
        precip_conc: NDArray[np.float64],
        # Model states and parameters
        initial_state: dict[str, ChemicalState],
        # Model-performance related data
        meas_river_conc: DataFrame | None = None,
        objective_functions: list[
            tuple[str, Callable[[Series, Series], float]]
        ] = DEFAULT_OBJECTIVE_FUNCTIONS,
        # Optional diagnostic data
        verbose: bool = False,
        return_partial: bool = False,
        failed_dir: str | None = None,
    ) -> RtModelResults:
        """Run the reactive-transport simulation forward over the hydro results.

        Args:
            hydro_res: output of a prior :meth:`run_both_models` hydrology run.
            precip_conc: ``(num_surface_zones, num_steps, num_species)``
                precipitation chemistry per surface zone and step.
            initial_state: per-zone :class:`ChemicalState` initial concs.
            meas_river_conc: observed river concentrations keyed by species for
                the objective-function metrics.
            objective_functions: ``(name, callable(meas, sim) -> float)`` list,
                evaluated per measured species against the ``<spec>_riv`` series.
            verbose: print step diagnostics.
            return_partial: return partial results on failure instead of raising.
            failed_dir: write JSON repro files for failed steps here.

        Returns:
            :class:`RtModelResults` with the per-species output frame, the
            objective-function frame (or ``None``) and per-zone forcing/steps.
        """
        # ==== Set up the forcing ==== #
        zones_list: list[RtZone] = list(self.rt_zones.values())
        hydro_sim_df: DataFrame = hydro_res.simulation
        forc: ForcingData | list[ForcingData] = hydro_res.forcing

        rt_forcing: NDArray[np.object_] = self.construct_rt_forcing_matrix(
            hydro_sim_df=hydro_sim_df,
            forc=forc,
            precip_conc=precip_conc,
            model_rt_params=self.rt_zones,
        )

        dates: pd.DatetimeIndex = cast(pd.DatetimeIndex, hydro_sim_df.index)
        num_steps: int = dates.size
        time_step_days: list[float] = [1.0] + [
            float((dates[i + 1] - dates[i]).total_seconds()) / 86_400.0
            for i in range(num_steps - 1)
        ]

        # ==== Construct the initial concentrations ==== #
        contains_minerals: bool = self.network.num_minerals > 0
        zone_initial_states: list[np.ndarray] = []
        for zone_name in self.zone_names():
            do_reactions: bool = self.rt_zones[zone_name].do_reactions
            do_speciation: bool = self.rt_zones[zone_name].do_speciation
            zone_conc: ChemicalState
            if zone_name in initial_state:
                zone_conc = initial_state[zone_name]
            else:
                zone_conc = ChemicalState()

            zone_state: dict[str, float] = {}

            # Primary aqueous species (default 1e-3 mol/L).
            for prim_aq in self.network.primary_aqueous:
                if prim_aq.name in zone_conc.primary:
                    zone_state[prim_aq.name] = zone_conc.primary[prim_aq.name]
                else:
                    zone_state[prim_aq.name] = 1e-3

            # Secondary aqueous species (default 1e-3 mol/L).
            for sec in self.network.secondary:
                if sec.name in zone_conc.secondary:
                    zone_state[sec.name] = zone_conc.secondary[sec.name]
                else:
                    zone_state[sec.name] = 1e-3

            # Mineral species (default 1e-20 mol/L when not reacting).
            for min_val in self.network.mineral:
                if min_val.name in zone_conc.mineral:
                    zone_state[min_val.name] = zone_conc.mineral[min_val.name]
                else:
                    if do_reactions and contains_minerals:
                        raise ValueError(
                            f"Missing initial concentration of mineral "
                            f"{min_val.name} for reactive transport simulation"
                        )
                    zone_state[min_val.name] = 1e-20

            # Equilibrium adsorption species (exchange site "X-" is required
            # when speciation is on).
            if self.network.has_exchange:
                if "X-" not in zone_conc.exchange and do_speciation:
                    raise ValueError(
                        f"You must specify the initial exchange site "
                        f"concentrations ('X-') for zone {zone_name} or turn "
                        "speciation off"
                    )
                if do_speciation:
                    zone_state["X-"] = zone_conc.exchange["X-"]
                else:
                    zone_state["X-"] = 1e-20

                for exch in self.network.exchange_species:
                    if exch.name in zone_conc.exchange:
                        zone_state[exch.name] = zone_conc.exchange[exch.name]
                    else:
                        zone_state[exch.name] = 1e-6

            zone_init_conc_list: list[float] = []
            for spec_name in self.network.species_order:
                zone_init_conc_list.append(zone_state[spec_name])
            zone_initial_states.append(np.array(zone_init_conc_list, dtype=np.float64))

        state: Tensor = torch.tensor(zone_initial_states, dtype=torch.float64)
        if verbose:
            print("Initial state: ")
            for zone_name, zone_conc in zip(self.zone_names(), state, strict=True):
                print(f"{zone_name}: {zone_conc}")

        # ==== Run the model forwards ==== #
        steps: list[list[RtStep]] = []
        old_states: list[Tensor] = []
        new_states: list[Tensor] = []

        final_completed: int = num_steps
        i: int = 0
        try:
            for i in range(num_steps):
                ds_for_step: list[RtForcing] = rt_forcing[:, i].tolist()
                dt_days: float = time_step_days[i]
                if dt_days < 1e-6:
                    raise ValueError(f"Time step on step {i} is too small: {dt_days}")

                old_states.append(state.clone())
                if verbose:
                    print(f"Starting step {i}")

                step: list[RtStep] = self.step_rt_model(
                    self.rt_zones,
                    state,
                    ds_for_step,
                    dt_days,
                    verbose=verbose,
                    failed_dir=failed_dir,
                    step_index=i,
                )
                steps.append(step)
                new_state: Tensor = torch.stack([s.state for s in step])
                new_states.append(new_state.clone())
                state = new_state

                if verbose:
                    print(f"Finished step {i}")
        except Exception as e:
            if return_partial:
                final_completed = i
            else:
                raise e

        # ==== Construct outputs ==== #
        species_names: list[str] = zones_list[0].all_species
        mineral_names: list[str] = zones_list[0].mineral_species
        zone_names: list[str] = self.zone_names()
        data_cols: dict[str, NDArray[np.float64]] = {}

        for zone_id, zone_name in enumerate(zone_names):
            for species_id, species_name in enumerate(species_names):
                conc: NDArray[np.float64] = np.empty(num_steps, dtype=np.float64)
                mass_in: NDArray[np.float64] = np.empty(num_steps, dtype=np.float64)
                lat_mass: NDArray[np.float64] = np.empty(num_steps, dtype=np.float64)
                lat_conc: NDArray[np.float64] = np.empty(num_steps, dtype=np.float64)
                vert_mass: NDArray[np.float64] = np.empty(num_steps, dtype=np.float64)
                vert_conc: NDArray[np.float64] = np.empty(num_steps, dtype=np.float64)
                tot_moles: NDArray[np.float64] = np.empty(num_steps, dtype=np.float64)
                for t in range(final_completed):
                    single_step: RtStep = steps[t][zone_id]
                    conc[t] = float(single_step.state[species_id])
                    mass_in[t] = float(single_step.mass_in[species_id])
                    lat_mass[t] = float(single_step.lat_mass[species_id])
                    lat_conc[t] = float(single_step.lat_conc[species_id])
                    vert_mass[t] = float(single_step.vert_mass[species_id])
                    vert_conc[t] = float(single_step.vert_conc[species_id])
                    tot_moles[t] = float(single_step.total_moles[species_id])

                data_cols[f"{species_name}_{zone_name}"] = conc
                data_cols[f"{species_name}_{zone_name}_mass_in"] = mass_in
                data_cols[f"{species_name}_{zone_name}_lat_mass"] = lat_mass
                data_cols[f"{species_name}_{zone_name}_lat_conc"] = lat_conc
                data_cols[f"{species_name}_{zone_name}_vert_mass"] = vert_mass
                data_cols[f"{species_name}_{zone_name}_vert_conc"] = vert_conc
                data_cols[f"{species_name}_{zone_name}_total_moles"] = tot_moles

            # Mineral reaction rates (immobile species rows carry no flux data).
            for species_id, species_name in enumerate(mineral_names):
                conc = np.empty(num_steps, dtype=np.float64)
                for t in range(final_completed):
                    conc[t] = float(steps[t][zone_id].mineral_rates[species_id])
                data_cols[f"{species_name}_rate_{zone_name}"] = conc

        res_df: DataFrame = DataFrame(data=data_cols, index=dates)

        # ==== Quantify the river concentrations ==== #
        river_zone_ids: list[int] = self.get_river_zone_ids()
        river_zone_names: list[str] = [self.zone_names()[i] for i in river_zone_ids]
        # Weight each river zone's lateral outflow by the area fraction of the
        # column it discharges from (``size_mat @ scales``), so the mixing
        # fractions are consistent with the area-weighted streamflow and the
        # area factor cancels in ``q_components / q_sim`` (golden
        # reactive_transport_model behaviour).
        zone_scale_arr: NDArray[np.float64] = self.zone_scales.detach().cpu().numpy()
        q_components: dict[str, Series] = {
            n: hydro_sim_df[f"q_lat_{n}"] * float(zone_scale_arr[river_zone_ids[k]])
            for k, n in enumerate(river_zone_names)
        }
        q_sim: Series = hydro_sim_df["sim_streamflow_mmd"]
        flow_fractions: dict[str, Series] = {
            name: val / q_sim for name, val in q_components.items()
        }

        for spec_name in self.network.species_names:
            comp_concs: list[Series] = [
                res_df[f"{spec_name}_{n}"] for n in river_zone_names
            ]
            conc_comps: list[Series] = [
                flow_fractions[n] * comp_concs[i]
                for i, n in enumerate(river_zone_names)
            ]
            weighted_comps: Series = conc_comps[0]
            for cc in conc_comps[1:]:
                weighted_comps = weighted_comps + cc
            res_df[f"{spec_name}_riv"] = weighted_comps

        # ==== Calculate the objective functions ==== #
        obj_df: DataFrame | None
        if meas_river_conc is not None:
            test_species: list[str] = list(meas_river_conc.columns)
            obj_df_cols: list[str] = [x[0] for x in objective_functions]
            data = np.full((len(test_species), len(obj_df_cols)), fill_value=np.nan)

            for j, (_obj_name, obj_func) in enumerate(objective_functions):
                for i, spec in enumerate(test_species):
                    sim_col_name: str = f"{spec}_riv"
                    c_meas: Series = meas_river_conc[spec]
                    c_sim: Series = res_df[sim_col_name]
                    data[i, j] = float(obj_func(c_meas, c_sim))

            obj_df = DataFrame(columns=obj_df_cols, index=test_species, data=data)
        else:
            obj_df = None

        return RtModelResults(
            simulation=res_df,
            objective_functions=obj_df,
            rt_forcings=rt_forcing,
            steps=steps,
            other={"old_states": old_states, "new_states": new_states},
        )

    # ---- combined run ------------------------------------------------- #

    def run_both_models(
        self,
        # Hydrology
        forc: ForcingData | list[ForcingData],
        # Reactive transport
        precip_conc: NDArray[np.float64],
        init_conc: dict[str, ChemicalState],
        # Optional data
        init_hydro_state: Tensor | None = None,
        meas_streamflow: Series | None = None,
        meas_river_conc: DataFrame | None = None,
        objective_functions: (
            list[tuple[str, Callable[[Series, Series], float]]] | None
        ) = None,
        verbose: bool = False,
    ) -> ModelResults:
        """Run hydrology and then reactive transport, returning both results.

        Args:
            forc: hydrological forcing (single series or one per surface zone).
            precip_conc: ``(surface_zone, step, species)`` precipitation chemistry.
            init_conc: per-zone :class:`ChemicalState` initial concentrations.
            init_hydro_state: optional initial hydrology storage per zone.
            meas_streamflow: observed streamflow series for hydro objective metrics.
            meas_river_conc: observed river concentrations for RT objectives.
            objective_functions: RT-style ``(name, pandas-callable)`` objective
                list, forwarded to the hydrology run through a tensor adapter.
            verbose: print run diagnostics.

        Returns:
            :class:`ModelResults` bundling the hydro and RT results.
        """
        if verbose:
            print("Starting hydrologic simulation")

        hydro_res: HydroModelResults = self.run_hydro_model(
            forc=forc,
            init_state=init_hydro_state,
            meas_streamflow=meas_streamflow,
            objective_functions=(
                None
                if objective_functions is None
                else _adapt_objective_functions(objective_functions)
            ),
        )

        if verbose:
            print("Finished hydrologic model simulation")
            print("Starting reactive transport simulation")

        rt_res: RtModelResults = self.run_rt_model(
            hydro_res=hydro_res,
            precip_conc=precip_conc,
            initial_state=init_conc,
            meas_river_conc=meas_river_conc,
            verbose=verbose,
            return_partial=False,
            failed_dir=None,
        )

        return ModelResults(hydro=hydro_res, reactive_transport=rt_res)


# ---------------------------------------------------------------------- #
# Model (structure-based calibration engine)
# ---------------------------------------------------------------------- #


class Model(ReactiveTransportModel):
    """Abstract base class for defining and running hydrologic models.

    Subclasses provide a class-level ``structure`` (a list of layers, each a
    list of :class:`~potions.hydro.HydrologicZone` instances):

    Example:
        >>> class MySimpleModel(Model):
        ...     structure = [[SnowZone()], [SurfaceZone()]]
        >>> model = MySimpleModel()
    """

    #: Nested ``(layer -> zones)`` structure of the model (set on subclasses).
    structure: ClassVar[list[LayerT]]

    def __init__(
        self,
        zones: dict[str, HydrologicZone] | None = None,
        scales: list[float] | None = None,
        network: ReactionNetwork | None = None,
        rt_zones: dict[str, RtZone] | None = None,
        river_zone: RiverZone | None = None,
        verbose: bool = False,
    ) -> None:
        """Initialize the model engine (see :class:`ReactiveTransportModel`)."""
        super().__init__(
            structure=self.structure,
            zones=zones,
            scales=scales,
            network=network,
            rt_zones=rt_zones,
            river_zone=river_zone,
            verbose=verbose,
        )

    # ---- parameter counts -------------------------------------------- #

    @classmethod
    def get_zone_names(cls) -> list[str]:
        """Zone names in flat (evaluation) order from the class structure."""
        return [zone.name for layer in cls.structure for zone in layer]

    @classmethod
    def get_num_zones(cls) -> int:
        """Total number of zones in the class structure."""
        return sum(len(layer) for layer in cls.structure)

    @classmethod
    def get_num_zone_parameters(cls) -> int:
        """Total number of tunable zone parameters."""
        num_zone_params: int = 0
        for layer in cls.structure:
            for zone in layer:
                num_zone_params += len(zone.param_list())
        return num_zone_params

    @classmethod
    def get_num_size_parameters(cls) -> int:
        """Number of size (proportion) parameters = surface zones minus one."""
        return len(cls.structure[0]) - 1

    @classmethod
    def get_num_hydro_parameters(cls) -> int:
        """Total number of optimizable hydrological parameters."""
        num_params: int = 0
        for layer in cls.structure:
            for zone in layer:
                num_params += zone.num_parameters()
        return num_params + cls.get_num_size_parameters()

    @classmethod
    def default_parameter_ranges(cls) -> dict[str, tuple[float, float]]:
        """Default calibration ranges for the hydrological parameters.

        Mirrors the golden ``HydrologicalModel.default_parameter_ranges``: for
        every zone in ``cls.structure`` the zone's own
        :meth:`~potions.hydro.HydrologicZone.default_parameter_range` bounds are
        collected and keyed ``"<zone_name>.<param_name>"``, followed by the
        surface-zone area fractions ``"proportion.<i>"`` (fixed to ``(0.1, 0.9)``
        in the reference).  Lapse-rate ranges are not included (the reference
        keeps them disabled).

        Returns:
            Ordered mapping of parameter name to ``(min, max)`` bound.
        """
        param_ranges: dict[str, tuple[float, float]] = {}
        for layer in cls.structure:
            for zone in layer:
                zone_range: dict[str, tuple[float, float]] = zone.default_parameter_range()
                for param_name in zone.parameter_names():
                    param_ranges[f"{zone.name}.{param_name}"] = zone_range[param_name]

        for i, _ in enumerate(cls.structure[0][:-1]):
            param_ranges[f"proportion.{i + 1}"] = (0.1, 0.9)

        return param_ranges

    # ---- serialization ----------------------------------------------- #

    def num_hydro_parameters(self) -> int:
        """The total number of optimizable hydrological parameters."""
        return len(self.hydro_to_array())

    def hydro_parameter_names(self) -> list[str]:
        """Ordered parameter names (``<zone>.<param>`` then ``proportion.N``)."""
        param_names: list[str] = []
        for zone in self.flat_model:
            for param in zone.parameter_names():
                param_names.append(f"{zone.name}.{param}")
        for i in range(self.num_surface_zones - 1):
            param_names.append(f"proportion.{i + 1}")
        return param_names

    def to_series(self) -> Series:
        """Parameter vector as a :class:`Series` keyed by parameter name."""
        return pd.Series(self.hydro_to_array(), index=self.hydro_parameter_names())

    @classmethod
    def hydro_from_dict(cls, params: dict[str, Any]) -> Model:
        """Build a hydrology-only model from a ``{name: value}`` parameter dict.

        Names use the ``<zone>.<param>`` convention of
        :meth:`hydro_parameter_names` (e.g. ``"soil.fc"``), plus
        ``proportion.N`` keys for the surface-zone area fractions.
        """
        default_zones: dict[str, HydrologicZone] = {
            zone.name: zone for layer in cls.structure for zone in layer
        }
        zone_params_list: list[tuple[str, float]] = [
            (key, float(val))
            for key, val in params.items()
            if str(key).split(".")[0] in default_zones
        ]
        proportion_params_list: list[float] = [
            float(val)
            for key, val in params.items()
            if str(key).startswith("proportion.")
        ]

        new_zones: dict[str, HydrologicZone] = {}
        for zone_name, default_zone in default_zones.items():
            kw: dict[str, float] = {
                str(key).split(".", 1)[1]: val
                for key, val in zone_params_list
                if str(key).startswith(f"{zone_name}.")
            }
            new_zones[zone_name] = type(default_zone)(**kw, name=zone_name)

        scales: list[float] = list(proportion_params_list)
        scales.append(1.0 - sum(scales))

        return cls(zones=new_zones, scales=scales)

    @classmethod
    def hydro_from_series(cls, series: Series) -> Model:
        """Build a hydrology-only model from a ``{name: value}`` parameter Series."""
        return cls.hydro_from_dict(series.to_dict())

    @classmethod
    def hydro_from_array(
        cls, arr: NDArray[np.float64], latent: bool = False, natural_scales: bool = True
    ) -> Model:
        """Build a hydrology-only model from a flat rectangular parameter vector.

        Args:
            arr: flat parameter vector (zone params first, then proportions).
            latent: treat the proportion parameters as latent fractions.
            natural_scales: unpack ``k`` (ground zones) from log10 scale.

        Returns:
            A hydrology-only :class:`Model` instance.
        """
        num_zone_params: int = cls.get_num_zone_parameters()
        num_size_params: int = cls.get_num_size_parameters()

        zone_params: NDArray[np.float64] = arr[:num_zone_params]
        size_params: list[float] = arr[
            num_zone_params : num_zone_params + num_size_params
        ].tolist()

        if latent:
            fractions: list[float] = []
            remainder: float = 1.0
            for size in size_params:
                fraction: float = size / remainder
                fractions.append(fraction)
                remainder -= size
            size_params = fractions

        size_params.append(1.0 - sum(size_params))

        new_zones: dict[str, HydrologicZone] = {}
        for layer in cls.structure:
            for zone in layer:
                ps: NDArray[np.float64]
                ps, zone_params = (
                    zone_params[: zone.num_parameters()],
                    zone_params[zone.num_parameters() :],
                )
                new_zones[zone.name] = cls._from_array_with_scales(
                    zone, ps, natural_scales
                )

        return cls(zones=new_zones, scales=size_params)

    @staticmethod
    def _from_array_with_scales(
        zone: HydrologicZone,
        ps: NDArray[np.float64],
        natural_scales: bool,
    ) -> HydrologicZone:
        """Call ``type(zone).from_array`` passing ``natural_scales`` if accepted.

        The concrete ``from_array`` constructors build zones without a ``name``
        argument, so the rebuilt zone inherits the class-default name; the
        structure override (``_override_zones``) matches zones by name, so the
        class-default name would silently drop the parameterized zone and keep
        the structure template.  The template name is therefore re-applied to
        the reconstructed zone before it is returned.
        """
        zone_type: type[HydrologicZone] = type(zone)
        arr_t: Tensor = torch.tensor(ps, dtype=torch.float64)
        zone_method = zone_type.from_array
        new_zone: HydrologicZone
        try:
            new_zone = zone_method(arr_t, natural_scales=natural_scales)  # type: ignore[call-arg]
        except TypeError as first_err:
            try:
                new_zone = zone_method(arr_t)  # type: ignore[call-arg]
            except TypeError:
                raise first_err

        # Reproduce the structure template's identity so ``_override_zones``
        # can find this zone by name (the ``from_array`` builders do not set
        # it).
        new_zone.name = zone.name
        return new_zone

    # ---- RT parameter plumbing --------------------------------------- #

    @classmethod
    def get_num_rt_parameters(
        cls,
        config: dict[str, RtZoneConfiguration],
        network: ReactionNetwork,
    ) -> int:
        """Number of RT zone parameters in the flat layout for the network.

        Mirrors the width consumed by :meth:`rt_params_from_array`: each
        non-river zone contributes the three water-dimension values plus one
        kinetic block per network mineral when the zone reacts.

        Args:
            config:  per-zone ``(do_reactions, do_speciation)`` flags (a
                ``"river"`` entry, if present, is not counted — the river
                parameters are carried separately by :meth:`from_array`).
            network: the reaction network (fixes the number of minerals;
                without it the count would silently be wrong for
                multi-mineral networks).

        Returns:
            The total number of RT zone parameters.
        """
        num_minerals: int = network.num_minerals
        num_params: int = 0
        for zone_name, zone in config.items():
            if zone_name == "river":
                continue
            num_params += (
                3 + num_minerals * PARAMETERS_PER_MINERAL * zone.do_reactions
            )
        return num_params

    @classmethod
    def rt_params_from_array(
        cls,
        arr: NDArray[np.float64],
        network: ReactionNetwork,
        configs: dict[str, RtZoneConfiguration],
        verbose: bool = False,
        natural_scales: bool = True,
    ) -> dict[str, RtParameters]:
        """Split a flat RT parameter vector into per-zone :class:`RtParameters`.

        Args:
            arr: flat RT parameter vector (3 zone dims + per-reacting-mineral
                blocks per zone, in zone order).
            network: the reaction network (fixes ``num_minerals``).
            configs: per-zone ``(do_reactions, do_speciation)`` flags.
            verbose: print debugging information.
            natural_scales: unpack ``ssa`` from log10 scale.

        Returns:
            Per-zone :class:`RtParameters` keyed by zone name.
        """
        zone_names: list[str] = cls.get_zone_names()
        num_minerals: int = network.num_minerals
        params_per_zone: list[int] = [
            3 + z.do_reactions * num_minerals * PARAMETERS_PER_MINERAL
            for zone_name, z in configs.items()
            if zone_name != "river"
        ]

        if len(arr) != sum(params_per_zone):
            raise ValueError(
                f"Incorrect number of parameters passed to "
                f"`Model.rt_params_from_array`, expected {sum(params_per_zone)}, "
                f"received {len(arr)}"
            )

        params_list: list[NDArray[np.float64]] = []
        rest: NDArray[np.float64] = arr
        for np_i in params_per_zone:
            params_list.append(rest[:np_i])
            rest = rest[np_i:]

        zone_params: dict[str, RtParameters] = {}
        for row_i, zone_name_i in zip(params_list, zone_names, strict=True):
            zone_params[zone_name_i] = RtParameters.from_array(
                torch.tensor(row_i, dtype=torch.float64),
                natural_scales=natural_scales,
            )
        return zone_params

    # ---- full model serialization ------------------------------------ #

    @classmethod
    def from_array(
        cls,
        arr: NDArray[np.float64],
        network: ReactionNetwork,
        config: dict[str, RtZoneConfiguration],
        verbose: bool = False,
        natural_scales: bool = True,
    ) -> Model:
        """Create a fully-parameterized model from one flat parameter vector.

        The vector stacks the hydrological zone + proportion parameters first,
        then the RT zone + mineral parameters, then the river parameters.

        Args:
            arr: flat parameter vector.
            network: the reaction network for the RT zones.
            config: per-zone ``(do_reactions, do_speciation)`` flags (may
                include a ``"river"`` entry).
            verbose: print debugging information.
            natural_scales: unpack ``k`` / ``ssa`` from log10 scale.

        Returns:
            A new, fully-parameterized :class:`Model` instance.
        """
        num_zones: int = cls.get_num_zones()
        num_hydro_params: int = cls.get_num_hydro_parameters()
        num_dim_params: int = 3 * num_zones
        num_minerals: int = network.num_minerals
        num_min_params: int = 0
        num_river_params: int = 0
        for zone_name, zone in config.items():
            if zone_name == "river":
                num_river_params += (
                    2 + num_minerals * PARAMETERS_PER_MINERAL * zone.do_reactions
                )
            else:
                num_min_params += (
                    num_minerals * PARAMETERS_PER_MINERAL * zone.do_reactions
                )
        num_expected: int = (
            num_hydro_params + num_dim_params + num_min_params + num_river_params
        )
        num_rt_zone_params: int = num_dim_params + num_min_params

        if arr.size != num_expected:
            raise ValueError(
                f"Incorrect number of parameters for creating RT model. "
                f"Expected {num_expected}, received {arr.size}, with "
                f"{num_hydro_params} hydrological parameters, {num_dim_params} "
                f"zone parameters, {num_min_params} mineral parameters, and "
                f"{num_river_params} river parameters"
            )

        hydro_params_arr: NDArray[np.float64] = arr[0:num_hydro_params]
        rt_zone_params_arr: NDArray[np.float64] = arr[
            num_hydro_params : num_hydro_params + num_rt_zone_params
        ]
        river_params_arr: NDArray[np.float64] | None = None
        if num_river_params > 0:
            river_params_arr = arr[-num_river_params:]

        hydro_mod: Model = cls.hydro_from_array(
            hydro_params_arr, natural_scales=natural_scales
        )
        rt_params: dict[str, RtParameters] = cls.rt_params_from_array(
            rt_zone_params_arr,
            network,
            config,
            verbose=verbose,
            natural_scales=natural_scales,
        )

        rt_zones: dict[str, RtZone] = {}
        for zone_name in rt_params:
            cfg: RtZoneConfiguration = config[zone_name]
            rt_zones[zone_name] = RtZone(
                network,
                parameters=rt_params[zone_name],
                do_reactions=cfg.do_reactions,
                do_speciation=cfg.do_speciation,
                name=zone_name,
            )

        river_zone: RiverZone | None = None
        if river_params_arr is not None:
            river_config: RtZoneConfiguration = config["river"]
            river_zone = RiverZone.from_array(
                torch.tensor(river_params_arr, dtype=torch.float64),
                network=network,
                do_reactions=river_config.do_reactions,
                do_speciation=river_config.do_speciation,
                name="river",
            )

        return cls(
            zones=hydro_mod.hydro_zones,
            scales=hydro_mod.scales,
            network=network,
            rt_zones=rt_zones,
            river_zone=river_zone,
        )

    # ---- batch runs ------------------------------------------------ #

    @classmethod
    def run_batch(
        cls,
        params: DataFrame,
        forc: ForcingData | list[ForcingData],
        init_state: Tensor | None = None,
        meas_streamflow: Series | None = None,
        num_threads: int = -1,
        write_time_series_results: bool = False,
        threshold_function: Callable[[HydroModelResults], bool] | None = None,
        output_dir: str = "batch_results",
    ) -> BatchResults:
        """Run a batch of hydrological simulations, one parameter set per row.

        Unlike the golden implementation this runs sequentially (no process
        pool) so results are deterministic and obvious; ``num_threads`` is
        accepted for API compatibility but ignored.

        Args:
            params: ``DataFrame`` whose columns are parameter names and whose
                rows are parameter sets (see :meth:`hydro_from_series`).
            forc: hydrological forcing.
            init_state: optional initial storage per zone.
            meas_streamflow: observed streamflow for the objective metrics.
            num_threads: ignored (sequential execution).
            write_time_series_results: save accepted simulations to CSVs.
            threshold_function: keep a run only when it returns True.
            output_dir: directory for saved time series.

        Returns:
            :class:`BatchResults` with the per-run simulations and the stacked
            objective-function frame.
        """
        results: list[tuple[int, HydroModelResults | None]] = []
        for i, row in params.iterrows():
            try:
                run_res: HydroModelResults = (
                    cls.hydro_from_series(row)
                    .run_hydro_model(forc=forc, init_state=init_state, meas_streamflow=meas_streamflow)
                )
                results.append((i, run_res))
                if write_time_series_results and (
                    threshold_function is None or threshold_function(run_res)
                ):
                    os.makedirs(output_dir, exist_ok=True)
                    run_res.simulation.to_csv(os.path.join(output_dir, f"{i}.csv"))
            except Exception as e:
                print(f"Failed on run {i} with error: {e}")
                results.append((i, None))

        simulations: dict[int, HydroModelResults] = {
            i: res for i, res in results if res is not None
        }
        obj_series: list[Series] = [res.objective_functions for res in simulations.values()]
        if obj_series:
            objective_functions: DataFrame = pd.concat(obj_series, axis=1).T
        else:
            objective_functions = pd.DataFrame()

        return BatchResults(
            simulations=simulations, objective_functions=objective_functions
        )