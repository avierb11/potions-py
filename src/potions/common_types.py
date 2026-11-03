"""Common data types shared by the hydrology and reactive-transport model.

These are pure records (frozen dataclasses) mirroring the ``common_types.rs``
Rust structs of the reference ``potions`` project.  Scalar fields are typed as
``float | torch.Tensor`` so the same records work for the scalar reference path
(plain floats) and the batched GPU path (zero-dim or ``(B,)`` tensors).

Units follow the reference implementation:
- mm of water for all fluxes/states/storages (``mm_water``),
- degrees Celsius for temperature,
- mol/L for aqueous concentrations (``molar``),
- mmol over a unit area for moles (``moles``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, TypeAlias, TypedDict

import torch
from pandas import DataFrame, Series

Tensor: TypeAlias = torch.Tensor
Scalar: TypeAlias = float | torch.Tensor

#: Sentinel "zero" concentration used to keep logarithms finite.
ZERO_CONC: float = 1e-20


@dataclass(frozen=True)
class ForcingData:
    """Meteorological forcing time series for a single location.

    Attributes:
        precip: precipitation [mm/day], indexed by timestamp.
        temp:   air temperature [deg C].
        pet:    potential evapotranspiration [mm/day].
    """

    precip: Series
    temp: Series
    pet: Series


@dataclass(frozen=True)
class LapseRateParameters:
    """Temperature/precipitation lapse-rate corrections applied to forcing data.

    Attributes:
        temp_factor: degrees Celsius gained per 100 m of elevation drop.
        precip_factor: fractional precipitation gain per 100 m of elevation rise.
    """

    temp_factor: Scalar = 0.0
    precip_factor: Scalar = 0.0


@dataclass(frozen=True)
class HydroForcing:
    """Meteorological + upstream inflow forcing applied during water-balance steps.

    Attributes:
        precip: precipitation entering the zone [mm].
        temp:   air temperature [degrees Celsius].
        pet:    potential evapotranspiration [mm].
        q_in:   upstream water inflow after merging forcing fluxes [mm].
    """

    precip: Scalar = 0.0
    temp: Scalar = 0.0
    pet: Scalar = 0.0
    q_in: Scalar = 0.0


@dataclass(frozen=True)
class HydroStep:
    """One hydrological time step for a single zone.

    Attributes:
        state:        water storage at the end of the step [mm].
        forc_flux:    forcing flux (precip that stays in the zone, e.g. snow) [mm].
        lat_flux:     lateral subsurface outflow [mm].
        vert_flux:    vertical (deep percolation) outflow [mm].
        vap_flux:     evapotranspiration loss [mm].
        q_in:         inflow to the step [mm].
        lat_flux_ext: external lateral flux bypassing the zone [mm].
        vert_flux_ext: external vertical flux bypassing the zone [mm].
    """

    state: Scalar
    forc_flux: Scalar
    lat_flux: Scalar
    vert_flux: Scalar
    vap_flux: Scalar
    q_in: Scalar
    lat_flux_ext: Scalar
    vert_flux_ext: Scalar

    @property
    def q_internal(self) -> Scalar:
        """Total water exiting the zone while carrying dissolved solutes [mm]."""
        return self.lat_flux + self.vert_flux

    @property
    def q_external(self) -> Scalar:
        """Total water that bypasses (does not interact with) the zone [mm].

        The result can be negative when locally-generated water exceeds external
        bypass; the reference implementation keeps this exact definition.
        """
        return self.lat_flux_ext + self.vert_flux_ext - self.q_internal

    @property
    def total_water_out(self) -> Scalar:
        """Total rate of water exiting the zone (internal + vapor) [mm]."""
        return self.lat_flux + self.vert_flux + self.vap_flux


@dataclass(frozen=True)
class RtForcing:
    """Reactive-transport forcing (one zone, one time step).

    Attributes:
        conc_in:    incoming species concentration vector [mol/L] (``num_spec``,).
        hydro_step: hydrology step that produced the fluxes for this zone.
        hydro_forc: meteorology for this zone/step.
        s_w:        soil-water content in the zone's reactive layer [-].
        z_w:        total depth of water in the zone's reactive layer [-].
    """

    conc_in: Tensor = field(default_factory=lambda: torch.zeros(0))
    hydro_step: HydroStep | None = None
    hydro_forc: HydroForcing | None = None
    s_w: Scalar = 0.0
    z_w: Scalar = 0.0

    @property
    def q_out(self) -> Scalar:
        """Total water leaving the zone carrying solutes (internal + external) [mm]."""
        assert self.hydro_step is not None, "RtForcing requires a hydro_step"
        return self.hydro_step.lat_flux_ext + self.hydro_step.vert_flux_ext


@dataclass(frozen=True)
class RtStep:
    """One reactive-transport time step for a single zone.

    All species-vector fields have length ``num_spec`` and are torch tensors.

    Attributes:
        state:          species concentrations after the step [mol/L].
        total_moles:    total moles of each species in the zone [mmol].
        conc_in:        incoming concentration used this step [mol/L].
        mass_in:        incoming solute mass this step [mmol].
        lat_conc:       lateral outflow concentration [mol/L].
        vert_conc:      vertical outflow concentration [mol/L].
        lat_mass:       lateral solute export [mmol].
        vert_mass:      vertical solute export [mmol].
        mineral_rates:  mineral reaction rates for each mineral [num_min].
    """

    state: Tensor
    total_moles: Tensor
    conc_in: Tensor
    mass_in: Tensor
    lat_conc: Tensor
    vert_conc: Tensor
    lat_mass: Tensor
    vert_mass: Tensor
    mineral_rates: Tensor


@dataclass
class MiscData:
    """Static per-network arrays consumed by kinetic evaluation.

    Attributes:
        mineral_stoichiometry: (num_spec, num_min) mineral dissolution stoichiometry.
        species_mobility:      (num_spec,) True for aqueous (mobile) species.
        mineral_molar_mass:    (num_min,) molar masses [g/mol].
        rate_const:            (num_min,) kinetic rate constants [1/cm^2/day].
    """

    mineral_stoichiometry: Tensor
    species_mobility: Tensor
    mineral_molar_mass: Tensor
    rate_const: Tensor

    def get_mineral_stoichiometry(self) -> Tensor:
        """Return the mineral stoichiometry matrix."""
        return self.mineral_stoichiometry

    def get_species_mobility(self) -> Tensor:
        """Return the species-mobility boolean vector."""
        return self.species_mobility

    def get_mineral_molar_mass(self) -> Tensor:
        """Return the per-mineral molar masses."""
        return self.mineral_molar_mass

    def get_rate_const(self) -> Tensor:
        """Return the per-mineral kinetic rate constants."""
        return self.rate_const


@dataclass(frozen=True)
class HydroModelStep:
    """Results of one hydrological time step for the whole (multi-zone) model.

    Field order corresponds to the flattened (evaluation-ordered) zone list;
    each tensor has shape ``(B, num_zones)`` (or just ``(num_zones,)`` when the
    step was driven with a single-lane state vector).

    Attributes:
        state:          new water storage per zone [mm].
        forc_flux:      forcing flux retained per zone (snow accumulation) [mm].
        vap_flux:       evaporative loss per zone [mm].
        lat_flux:       lateral outflow per zone, carries solutes [mm].
        vert_flux:      vertical outflow per zone, carries solutes [mm].
        q_in:           inflow used by each zone [mm].
        lat_flux_ext:   total lateral outflow per zone (bypass + solute-carrying) [mm].
        vert_flux_ext:  total vertical outflow per zone [mm].
    """

    state: Tensor
    forc_flux: Tensor
    vap_flux: Tensor
    lat_flux: Tensor
    vert_flux: Tensor
    q_in: Tensor
    lat_flux_ext: Tensor
    vert_flux_ext: Tensor


@dataclass(frozen=True)
class HydroModelResults:
    """Results of a full hydrologic model run.

    Attributes:
        forcing:             input forcing (single series or one per surface zone).
        simulation:          DataFrame of states and fluxes, one row per time step
            and ``8 * num_zones`` columns (``s_<z>``, ``q_forc_<z>``, ``q_vap_<z>``,
            ``q_lat_<z>``, ``q_vert_<z>``, ``q_in_<z>``, ``q_lat_ext_<z>``,
            ``q_vert_ext_<z>``) plus ``sim_streamflow_mmd`` (and, when provided,
            ``meas_streamflow_mmd``).
        objective_functions: named performance metrics vs measured streamflow.
    """

    forcing: ForcingData | list[ForcingData]
    simulation: DataFrame
    objective_functions: Series


@dataclass(frozen=True)
class ChemicalState:
    """Initial chemical state of a zone, partitioned by species type.

    Attributes:
        mineral:  mineral species concentrations ``{name: mol/L}``.
        exchange: exchange species concentrations (incl. ``"X-"``) ``{name: mol/L}``.
        primary:  primary aqueous species concentrations ``{name: mol/L}``.
        secondary: secondary aqueous species concentrations ``{name: mol/L}``.
    """

    mineral: dict[str, float] = field(default_factory=dict)
    exchange: dict[str, float] = field(default_factory=dict)
    primary: dict[str, float] = field(default_factory=dict)
    secondary: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RtZoneConfiguration:
    """Per-zone reactive-transport behavioural flags."""

    do_reactions: bool
    do_speciation: bool


@dataclass
class RtModelResults:
    """Results of a full reactive-transport model run.

    Attributes:
        simulation: DataFrame indexed by timestamps. Per species per zone:
            ``{species}_{zone}``, ``_mass_in``, ``_lat_mass``, ``_lat_conc``,
            ``_vert_mass``, ``_vert_conc``, ``_total_moles``; per mineral:
            ``{mineral}_rate_{zone}``; plus flow-weighted river columns
            ``{species}_riv``.
        objective_functions: DataFrame (index = measured species, columns =
            objective names) vs ``meas_river_conc``, or ``None``.
        rt_forcings: per-(zone, step) :class:`RtForcing` object array.
        steps: per-step list of per-zone :class:`RtStep`.
        other: diagnostic dict (e.g. old/new states).
    """

    simulation: DataFrame
    objective_functions: DataFrame | None
    rt_forcings: Any
    steps: list[list[RtStep]]
    other: dict[str, Any]


@dataclass
class ModelResults:
    """Combined results of a hydrology + reactive-transport run."""

    hydro: HydroModelResults
    reactive_transport: RtModelResults


class BatchParams(TypedDict):
    """Per-batch-run persistence settings (golden ``BatchParams``)."""

    output_dir: str
    threshold_function: Callable[[HydroModelResults], bool] | None
    save_results: bool


class BatchResults(TypedDict):
    """Results of a parallel batch calibration run (golden ``BatchResults``)."""

    simulations: dict[int, HydroModelResults]
    objective_functions: DataFrame