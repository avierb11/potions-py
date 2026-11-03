"""Batched (multi-lane) hydrology + reactive-transport forward runs.

The scalar calibration pipeline (``arr_to_results``) builds one ``Model`` per
parameter set and runs the zone-by-zone Python orchestrators
``HydrologicalModel.step_hydro_model`` and
``ReactiveTransportModel.step_rt_model``.  Both loops encode the *batch
dimension as zones* (a ``(B,)`` state) and flatten the flux accumulation back
to 1-D, so they cannot carry independent parameter sets as an extra leading
batch-size axis.

This module provides a self-contained *batched runner* that mirrors the scalar
semantics lane-by-lane while reusing the already-batched zone kernels
(``HydrologicZone.step``, :meth:`RtZone.step_with_params` and the vectorised
speciation solve in ``potions.equilibrium``).  A run with batch size N
reproduces the saved scalar forward exactly when those N parameter sets are
run individually through the scalar pipeline.

Layout conventions (East River / :class:`HbvLateralModel`):

* the flat (evaluation) zone order is ``snow_hs, snow_rp, surface_hs,
  surface_rp, shallow_hs, shallow_rp, deep_hs, deep_rp`` (river zones are the
  last zone of every layer, ``[1, 3, 5, 7]`` here);
* a parameter vector stacks ``[hydro + proportion][RT-zone][init-state]``; per
  zone the RT block is ``(porosity, depth, passive)`` followed by
  ``PARAMETERS_PER_MINERAL`` entries per reacting mineral;
* the initial-state entries are consumed in zone order: one log10 mineral
  concentration per reacting zone plus one log10 exchange-site total per
  speciating zone;
* forcing is distributed to every zone per lane via
  ``size_mat @ diag(scales_k)`` (and its row-normalized relative variant for
  temperature), exactly as in ``HydrologicalModel._get_forc_mat``;
* the streamflow is the catchment-mean outlet: each river zone's lateral
  outflow is weighted by the per-lane area fraction of the column it
  discharges from (``size_mat @ scales``), exactly as in the scalar
  ``run_hydro_model`` and river-concentration fixes.

The runner is *honest* about its concurrency ceiling: reactive transport is a
Gauss-Seidel sweep over the zones per step (each a batched Newton / LM solve
across all lanes), so throughput gains are sublinear in the batch size.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict, cast

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame, DatetimeIndex, Series
import torch
from torch import Tensor

from .common_types import (
    ForcingData,
    HydroForcing,
    HydroStep,
    RtForcing,
    RtZoneConfiguration,
    Scalar,
)
from .hydro import (
    GroundZone,
    GroundZoneB,
    HydrologicZone,
    set_hydro_dtype,
)
from .hydro_model import (
    DEFAULT_OBJECTIVE_FUNCTIONS as HYDRO_OBJECTIVE_FUNCTIONS,
    HydrologicalModel,
)
from .kinetic_structures import PARAMETERS_PER_MINERAL, RtParameters
from .model import Model
from .objective_functions import DEFAULT_OBJECTIVE_FUNCTIONS
from .reaction_network import ReactionNetwork
from .rt_zone import RtZone

#: Sentinel "zero" concentration used for non-reacting mineral rows.
ZERO_CONC: float = 1e-20


def _as_tensor(x: Scalar) -> Tensor:
    """Coerce a possibly-scalar physics output to a plain tensor."""
    return torch.as_tensor(x)


class BatchedConfig(TypedDict):
    """Everything needed to forward-run the model for a batch of parameter sets.

    Mirrors the showcase ``CalibrationData`` typed dict so a loaded config can
    be passed straight through.

    Attributes:
        network: the reaction network fixing the species / kinetics.
        zone_configs: per-zone ``(do_reactions, do_speciation)`` flags.
        forc: single forcing series, or one per surface zone.
        q_meas: measured streamflow series (objective scoring).
        conc_df: measured river concentrations (RT objective scoring).
        precip_conc: ``(num_surface, num_steps, num_species)`` precipitation
            chemistry.
        model_type: the :class:`~potions.model.Model` subclass to emulate.
    """

    network: ReactionNetwork
    zone_configs: dict[str, RtZoneConfiguration]
    forc: ForcingData | list[ForcingData]
    q_meas: Series
    conc_df: DataFrame
    precip_conc: NDArray[np.float64]
    model_type: type[Model]


@dataclass(frozen=True)
class Geometry:
    """Static connectivity geometry shared by all lanes of a run.

    Attributes:
        lat_mat:      ``(num_zones, num_zones)`` lateral adjacency rows.
        vert_mat:     ``(num_zones, num_zones)`` vertical adjacency rows.
        size_mat:     ``(num_zones, num_surface)`` surface-zone contribution map.
        river_ids:    flat indices of the river (layer-tail) zones.
        surface_ids:  flat indices of the surface zones.
        zone_names:   flat zone names in evaluation order.
    """

    lat_mat: Tensor
    vert_mat: Tensor
    size_mat: Tensor
    river_ids: list[int]
    surface_ids: list[int]
    zone_names: list[str]


@dataclass(frozen=True)
class BatchedRunOutput:
    """Per-lane simulation results for a batched forward run.

    Attributes:
        q_sim:             ``(batch_size, num_steps)`` streamflow [mm/day].
        doc_riv:           ``(batch_size, num_steps)`` flow-weighted river DOC
            [mmol/L].
        hydro_objectives:  per-lane hydro objective dicts (same keys/order as
            the scalar hydrology run).
        rt_doc_objectives: per-lane RT DOC objective dicts (same keys/order as
            the scalar ``DOC_riv`` scoring).
        overflow_mask:     ``(batch_size,)`` ``True`` for lanes whose storage
            touched the zone maximum water volume (the scalar path would abort
            those runs with a ``WaterVolumeError``; the batched path keeps
            going).
    """

    q_sim: Tensor
    doc_riv: Tensor
    hydro_objectives: list[dict[str, float]]
    rt_doc_objectives: list[dict[str, float]]
    overflow_mask: Tensor


# ---------------------------------------------------------------------- #
# Parameter decomposition                                                #
# ---------------------------------------------------------------------- #


def _step_lengths(dates: DatetimeIndex, rt_style: bool = True) -> list[float]:
    """Return the duration (days) of each step in a batched run.

    For hydrology, the step length is the calendar difference between
    consecutive dates.  For reactive transport, the first step uses the
    hydrology step length, and subsequent steps use the difference to the
    previous date (since the hydro step is the limiting factor).

    Args:
        dates: the date index of the simulation.
        rt_style: if True, return step lengths for the RT runner (which
            steps at the same rate as hydrology but the first step uses the
            hydrology step length).

    Returns:
        List of step lengths in days.
    """
    if len(dates) < 2:
        return []
    lengths: list[float] = [
        float((dates[i] - dates[i - 1]).days) for i in range(1, len(dates))
    ]
    if rt_style:
        return [lengths[0]] + lengths  # first step uses hydrology length
    return lengths + [lengths[-1]]  # last step length is repeated


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #



def _soil_water(
    state: Tensor,
    v: Tensor,
    depth: Tensor,
) -> Tensor:
    """Compute soil water content from total storage.

    Args:
        state: (B,) or (B, Z) total storage in the zone.
        v: (Z,) or (B, Z) water volume capacity.
        depth: (Z,) or (B, Z) zone depth.

    Returns:
        Soil water content [-] as ratio of storage to capacity.
    """
    # s_w = state / (v + passive_water_storage)
    # But passive is already included in v (v = porosity * depth - passive)
    # So s_w = state / v
    # Avoid division by zero
    v_safe = v.clamp_min(1e-6)
    return state / v_safe.unsqueeze(-1) if state.dim() > v.dim() else state / v_safe


def _build_hydro_step(hydro: dict[str, Tensor], t: int, i: int) -> HydroStep:
    """Build a HydroStep from the hydro dict for a given timestep and zone."""
    return HydroStep(
        state=hydro["state"][:, t, i],
        forc_flux=hydro["forc_flux"][:, t, i],
        lat_flux=hydro["lat_flux"][:, t, i],
        vert_flux=hydro["vert_flux"][:, t, i],
        vap_flux=hydro["vap_flux"][:, t, i],
        q_in=hydro["q_in"][:, t, i],
        lat_flux_ext=hydro["lat_flux_ext"][:, t, i],
        vert_flux_ext=hydro["vert_flux_ext"][:, t, i],
    )


def _build_rt_forcing(
    cin_sources: Tensor,
    hydro_step: HydroStep,
    zone_forc: HydroForcing,
    s_w: Tensor,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> RtForcing:
    """Build an RtForcing object."""
    return RtForcing(
        conc_in=cin_sources,
        hydro_step=hydro_step,
        hydro_forc=zone_forc,
        s_w=s_w,
        z_w=torch.zeros(batch_size, dtype=dtype, device=device),
    )


def _compute_cin_sources(
    is_surface: bool,
    i: int,
    surface_idx: dict[int, int],
    batch_size: int,
    ns: int,
    precip_conc_gpu: Tensor,
    lat_mass: Tensor,
    vert_mass: Tensor,
    lat_mat: Tensor,
    vert_mat: Tensor,
    hydro: dict[str, Tensor],
    t: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """Compute incoming concentration sources for a zone.

    Surface zones (the first-layer zones, ``geometry.surface_ids``) receive the
    precipitation chemistry directly; every other zone mixes the lateral and
    vertical solute mass delivered by the zones that were solved earlier in the
    Gauss-Seidel sweep, normalised by the incoming water volume.  Mirrors the
    scalar ``step_rt_model`` (``reactive_transport_model.py``): the ``conc_in``
    of a non-surface zone is ``mass_in / tot_water_in`` when the incoming water
    exceeds ``1e-6`` else an all-zero concentration.

    Args:
        is_surface:    whether zone ``i`` is a surface (precip-fed) zone.
        i:             flat zone index.
        surface_idx:   flat zone index -> row of ``precip_conc_gpu``.
        batch_size:    number of lanes in the batch.
        ns:            number of species.
        precip_conc_gpu: ``(num_surface, num_steps, num_species)`` precipitation
            chemistry on the compute device.
        lat_mass:      ``(batch_size, num_zones, num_species)`` lateral mass that
            each zone exports once solved this step.
        vert_mass:     ``(batch_size, num_zones, num_species)`` vertical mass
            exports.
        lat_mat:       ``(num_zones, num_zones)`` lateral connectivity rows.
        vert_mat:      ``(num_zones, num_zones)`` vertical connectivity rows.
        hydro:         ``(batch_size, num_steps, num_zones)`` hydrology outputs.
        t:             simulation step index.
        dtype:         compute dtype.
        device:        compute device.

    Returns:
        ``(batch_size, num_species)`` incoming concentrations for zone ``i`` at
        step ``t``.
    """
    if is_surface:
        return precip_conc_gpu[surface_idx[i], t].expand(batch_size, ns)
    mass_in: Tensor = (
        lat_mass.transpose(1, 2) @ lat_mat[i]
        + vert_mass.transpose(1, 2) @ vert_mat[i]
    )
    tot_water_in: Tensor = hydro["q_in"][:, t, i] + hydro["forc_flux"][:, t, i]
    return torch.where(
        tot_water_in.unsqueeze(-1) > 1e-6,
        mass_in / tot_water_in.unsqueeze(-1),
        torch.zeros(batch_size, ns, dtype=dtype, device=device),
    )


def run_rt_batched(
    geometry: Geometry,
    rt_zones: list[RtZone],
    hydro: dict[str, Tensor],
    zone_precip: Tensor,
    zone_temp: Tensor,
    zone_pet: Tensor,
    precip_conc: NDArray[np.float64],
    init_state: Tensor,
    dates: DatetimeIndex,
    device: torch.device,
    dtype: torch.dtype,
    compute_dtype: torch.dtype | None = None,
) -> tuple[Tensor, Tensor]:
    """Run the multi-lane reactive-transport forward (Gauss-Seidel coupling).

    Mirrors ``ReactiveTransportModel.run_rt_model`` / ``step_rt_model`` with
    lane-batched species states.  For every zone ``i`` of every step the
    incoming mass is aggregated over the upstream export rows already solved:
    ``lat_mass.transpose(1, 2) @ lat_mat[i]`` yields the
    ``(batch_size, num_species)`` incoming mass per lane (the scalar
    ``(num_zones, num_species)`` form is the batch-size-1 special case).
    Kinetic steps pass per-lane depth / auxiliary parameters explicitly so one
    zone object serves all lanes while keeping its per-zone, cross-lane
    speciation warm-start.

    Args:
        geometry:    connectivity (``lat_mat`` / ``vert_mat`` / surface ids).
        rt_zones:    per-lane reactive-transport zones (flat order).
        hydro:       hydrology outputs (``(batch_size, T, Z)`` arrays).
        zone_precip: ``(batch_size, T, Z)`` per-zone precipitation (hydro forcing).
        zone_temp:   ``(batch_size, T, Z)`` per-zone temperature.
        zone_pet:    ``(batch_size, T, Z)`` per-zone PET.
        precip_conc: ``(num_surface, T, num_spec)`` surface precipitation chemistry.
        init_state:  ``(batch_size, Z, num_spec)`` initial concentrations.
        dates:       simulation timestamps.
        device:      compute device.
        dtype:       compute dtype.
        compute_dtype: Optional dtype for the RT solve internals (e.g.,
            float32 compute with float64 convergence checks).

    Returns:
        ``(state (T, batch_size, Z, num_spec), overflow (batch_size, Z))``
        where ``overflow[lane, zone]`` is true when that lane saturated that
        zone (the scalar path would raise a ``WaterVolumeError`` there).
    """
    batch_size: int = int(init_state.shape[0])
    Z: int = int(init_state.shape[1])
    ns: int = int(init_state.shape[2])
    num_steps: int = int(precip_conc.shape[1])
    dt_days: list[float] = _step_lengths(dates, rt_style=True)

    lat_mat: Tensor = geometry.lat_mat.to(dtype=dtype, device=device)
    vert_mat: Tensor = geometry.vert_mat.to(dtype=dtype, device=device)
    is_surface: list[bool] = [i in geometry.surface_ids for i in range(Z)]

    lat_mass: Tensor = torch.zeros(batch_size, Z, ns, dtype=dtype, device=device)
    vert_mass: Tensor = torch.zeros(batch_size, Z, ns, dtype=dtype, device=device)

    state: Tensor = init_state.clone()
    # Pre-allocate state_out to avoid per-step clone + stack
    state_out: Tensor = torch.empty(
        num_steps, batch_size, Z, ns, dtype=dtype, device=device
    )
    overflow: Tensor = torch.zeros(batch_size, Z, dtype=torch.bool, device=device)

    per_zone_v: list[Tensor] = []
    per_zone_passive: list[Tensor] = []
    per_zone_depth: list[Tensor] = []
    for zone in rt_zones:
        dims = zone.parameters.dimensions
        porosity: Tensor = torch.as_tensor(dims.porosity, dtype=dtype, device=device)
        depth: Tensor = torch.as_tensor(dims.depth, dtype=dtype, device=device)
        passive: Tensor = torch.as_tensor(
            dims.passive_water_storage, dtype=dtype, device=device
        )
        per_zone_v.append(porosity * depth - passive)
        per_zone_passive.append(passive)
        per_zone_depth.append(depth)

    # Pre-convert precip_conc to a device tensor once (avoids per-step transfer).
    precip_conc_gpu: Tensor = torch.tensor(
        precip_conc, dtype=dtype, device=device
    )  # (num_surface, num_steps, ns)

    # Map each surface (first-layer) flat zone index to its ``precip_conc`` row.
    surface_idx: dict[int, int] = {
        zone_index: s for s, zone_index in enumerate(geometry.surface_ids)
    }

    # Same no-autograd guarantee as ``run_hydro_batched``: the RT step writes
    # ``state[:, i]`` / ``lat_mass[:, i]`` in place, so run under
    # ``inference_mode`` to skip version-counter tracking and state the intent.
    with torch.inference_mode():
        for t in range(num_steps):
            dt: float = dt_days[t]

            # Gauss-Seidel sweep: zones are solved in flat (evaluation) order so
            # downstream zones see this step's exports of upstream zones.
            for i in range(Z):
                zone = rt_zones[i]
                hs = _build_hydro_step(hydro, t, i)
                cin_sources = _compute_cin_sources(
                    is_surface[i],
                    i,
                    surface_idx,
                    batch_size,
                    ns,
                    precip_conc_gpu,
                    lat_mass,
                    vert_mass,
                    lat_mat,
                    vert_mat,
                    hydro,
                    t,
                    dtype,
                    device,
                )
                s_w = _soil_water(
                    hydro["state"][:, t, i].unsqueeze(-1),
                    per_zone_v[i],
                    per_zone_passive[i],
                ).squeeze(-1)
                zone_forc = HydroForcing(
                    precip=zone_precip[:, t, i],
                    temp=zone_temp[:, t, i],
                    pet=zone_pet[:, t, i],
                    q_in=torch.zeros(batch_size, dtype=dtype, device=device),
                )
                d: RtForcing = _build_rt_forcing(
                    cin_sources, hs, zone_forc, s_w, batch_size, dtype, device
                )
                step = zone.step_with_params(
                    c_0=state[:, i].clone(),
                    d=d,
                    dt_days=dt,
                    aux=zone.parameters.mineral_params,
                    depth=per_zone_depth[i],
                    compute_dtype=compute_dtype,
                )
                state[:, i] = step.state
                lat_mass[:, i] = step.lat_mass
                vert_mass[:, i] = step.vert_mass

            state_out[t] = state  # Direct write, no clone

            for i in range(Z):
                overflow[:, i] |= hydro["state"][:, t, i] >= per_zone_v[i]

    return state_out, overflow


def run_hydro_rt_batched(
    params: NDArray[np.float64],
    config: BatchedConfig,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
    init_aq_conc: float = 1e-3,
    compute_dtype: torch.dtype | None = None,
) -> BatchedRunOutput:
    """Run hydrology + reactive transport for a batch of parameter sets.

    Args:
        params:       (batch_size, num_parameters) flat parameter array: the
            hydrological block, the RT zone blocks (one block per flat zone),
            and the init-state section (width and column order per the
            module docstring).
        config:       loaded calibration config (network, zone configs, forcing,
            measurements, model type).
        device:       compute device.
        dtype:        compute dtype (defaults to torch.float64 for exact
            scalar parity).
        init_aq_conc: initial concentration of non-DOC aqueous species.
        compute_dtype: Optional dtype for the RT solve internals (e.g.,
            float32 compute with float64 convergence checks).

    Returns:
        A :class: with per-lane streamflow, river DOC and the
        scalar-equivalent objective dicts.
    """
    model_type: type[Model] = config["model_type"]
    network: ReactionNetwork = config["network"]
    zone_configs: dict[str, RtZoneConfiguration] = config["zone_configs"]

    set_hydro_dtype(dtype)

    batch_size: int = int(params.shape[0])
    num_hydro: int = model_type.get_num_hydro_parameters()
    num_rt_zone: int = model_type.get_num_rt_parameters(zone_configs, network)
    num_react: int = sum(1 for c in zone_configs.values() if c.do_reactions)
    num_spec_zones: int = sum(1 for c in zone_configs.values() if c.do_speciation)
    num_init: int = (
        network.num_minerals * num_react + num_spec_zones * int(network.has_exchange)
    )

    if params.shape[1] != num_hydro + num_rt_zone + num_init:
        raise ValueError(
            f"Parameter width {params.shape[1]} does not match the model layout "
            f"({num_hydro} hydro + {num_rt_zone} RT zone + {num_init} init-state)"
        )

    hydro_arr: NDArray[np.float64] = params[:, :num_hydro]
    rt_arr: NDArray[np.float64] = params[:, num_hydro : num_hydro + num_rt_zone]
    init_arr: NDArray[np.float64] = params[:, num_hydro + num_rt_zone :]

    blocks, scales = _hydro_zone_blocks(model_type, hydro_arr, device, dtype)
    zones: list[HydrologicZone] = _build_hydro_zones(model_type, blocks)
    rt_pars: list[RtParameters] = _rt_blocks(
        model_type, network, zone_configs, rt_arr, device, dtype
    )
    rt_zones: list[RtZone] = [
        RtZone(
            network,
            parameters=rtp,
            do_reactions=zone_configs[name].do_reactions,
            do_speciation=zone_configs[name].do_speciation,
            name=name,
        )
        for name, rtp in zip(model_type.get_zone_names(), rt_pars, strict=True)
    ]

    mean_doc: float = float(config["conc_df"]["DOC"].mean())
    init_state: Tensor = _init_state_tensor(
        model_type, network, zone_configs, init_arr, mean_doc, init_aq_conc, device, dtype
    )

    # Geometry from a hydrology-only model over the same structure.
    geo_model: HydrologicalModel = HydrologicalModel(structure=model_type.structure)
    geometry: Geometry = Geometry(
        lat_mat=geo_model.lat_mat.to(dtype=dtype, device=device),
        vert_mat=geo_model.vert_mat.to(dtype=dtype, device=device),
        size_mat=geo_model._get_size_mat().to(dtype=dtype, device=device),
        river_ids=geo_model.get_river_zone_ids(),
        surface_ids=geo_model.surface_zone_ids,
        zone_names=geo_model.zone_names(),
    )

    all_precip, all_temp, all_pet = _forcing_source_tensors(
        config["forc"], len(geometry.surface_ids), device, dtype
    )
    zone_precip, zone_temp, zone_pet = _per_lane_forcing(
        geometry, all_precip, all_temp, all_pet, scales, dtype
    )

    forc: ForcingData | list[ForcingData] = config["forc"]
    dates: DatetimeIndex = cast(
        DatetimeIndex,
        forc.precip.index if isinstance(forc, ForcingData) else forc[0].precip.index,
    )

    init_hydro: Tensor = torch.tensor(
        [z.default_init_state() for z in geo_model.flat_model],
        dtype=dtype,
        device=device,
    )
    hydro: dict[str, Tensor] = run_hydro_batched(
        geometry,
        zones,
        zone_precip,
        zone_temp,
        zone_pet,
        dates,
        init_hydro,
        device,
        dtype,
    )
    state_t, overflow = run_rt_batched(
        geometry,
        rt_zones,
        hydro,
        zone_precip,
        zone_temp,
        zone_pet,
        config["precip_conc"],
        init_state,
        dates,
        device,
        dtype,
        compute_dtype,
    )

    # ---- Per-lane outputs ---- #
    lat_flux: Tensor = hydro["lat_flux"]  # (batch_size, T, Z)
    # Per-lane area fractions: each zone inherits the fraction of the surface
    # column it drains (``size_mat @ scales``), mirroring the scalar
    # ``HydrologicalModel.zone_scales``.
    zone_scales: Tensor = scales @ geometry.size_mat.T  # (batch_size, num_zones)
    # Catchment-mean outlet: weight each river zone's lateral outflow by the
    # area fraction of the column it discharges from, so the streamflow is a
    # catchment-mean depth rather than an unweighted sum of zone fluxes
    # (mirrors the scalar ``run_hydro_model`` fix).
    q_sim: Tensor = (
        (lat_flux * zone_scales.unsqueeze(1))[:, :, geometry.river_ids].sum(dim=-1)
    )

    doc_idx: int = int(network.species_names.index("DOC"))
    river_scales: Tensor = zone_scales[:, geometry.river_ids]  # (batch_size, num_river)
    doc_riv: Tensor = torch.stack(
        [
            _flow_weighted_river_doc(
                state_t[t, :, :, :],
                lat_flux[:, t, :],
                geometry.river_ids,
                doc_idx,
                river_scales,
            )
            for t in range(int(state_t.shape[0]))
        ],
        dim=1,
    )

    # ---- Per-lane objective functions ---- #
    meas_q: Tensor = torch.tensor(
        config["q_meas"].to_numpy(dtype=float), dtype=dtype, device=device
    )
    doc_meas: Series = config["conc_df"]["DOC"]

    hydro_objectives: list[dict[str, float]] = []
    rt_doc_objectives: list[dict[str, float]] = []
    for lane in range(batch_size):
        hydro_objectives.append(
            {
                name: float(fn(meas_q, q_sim[lane]))
                for name, fn in HYDRO_OBJECTIVE_FUNCTIONS
            }
        )
        doc_sim: Series = Series(
            doc_riv[lane].detach().cpu().numpy(),
            dtype=float,
            index=doc_meas.index,
        )
        rt_doc_objectives.append(
            {name: float(fn(doc_meas, doc_sim)) for name, fn in DEFAULT_OBJECTIVE_FUNCTIONS}
        )

    return BatchedRunOutput(
        q_sim=q_sim.detach(),
        doc_riv=doc_riv.detach(),
        hydro_objectives=hydro_objectives,
        rt_doc_objectives=rt_doc_objectives,
        overflow_mask=overflow.any(dim=1),
    )


def _flow_weighted_river_doc(
    state_t: Tensor,
    lat_flux: Tensor,
    river_ids: list[int],
    doc_idx: int,
    river_scales: Tensor | None = None,
) -> Tensor:
    """Flow-weighted river DOC ``sum_r (q_lat_r / q_sim) * DOC_r`` per lane.

    Args:
        state_t:      ``(batch_size, num_zones, num_species)`` concentrations at one
            step.
        lat_flux:     ``(batch_size, num_zones)`` internal lateral fluxes at that step.
        river_ids:    flat indices of the river zones.
        doc_idx:      index of the DOC species.
        river_scales: optional ``(batch_size, len(river_ids))`` per-lane area-fraction
            weights for the river zones (``size_mat @ scales`` restricted to
            the river zones).  When given, each river zone's lateral outflow
            is weighted by its lane's area fraction before the mixing
            fractions are formed, mirroring the scalar ``model.py``
            river-concentration fix (the area factor cancels against the
            weighted streamflow in the fraction).

    Returns:
        ``(batch_size,)`` river DOC concentrations for the step.
    """
    if river_scales is not None:
        w: Tensor = lat_flux[:, river_ids] * river_scales
    else:
        w = lat_flux[:, river_ids]
    q_sim: Tensor = w.sum(dim=1).unsqueeze(-1)
    frac: Tensor = w / q_sim
    doc: Tensor = state_t[:, river_ids, doc_idx]
    return (frac * doc).sum(dim=1)


def _build_hydro_zones(
    model_type: type[Model], blocks: list[Tensor]
) -> list[HydrologicZone]:
    """Build per-lane :class:`HydrologicZone` objects from batched blocks.

    Each returned zone holds ``(batch_size,)`` parameter tensors so one zone
    object steps all lanes together.  The ground-zone ``k`` drainage coefficient
    is unpacked
    from its log10 natural scale, matching ``HydrologicZone.from_array`` (and
    therefore ``Model.hydro_from_array`` ``natural_scales=True``).

    Args:
        model_type: the hydrology ``Model`` subclass (fixes zone order/types).
        blocks:     per-flat-zone ``(batch_size, num_parameters)`` blocks (from
            :func:`_hydro_zone_blocks`).

    Returns:
        Zones in flat (structure) order, one per input block.
    """
    flat: list[HydrologicZone] = [
        proto for layer in model_type.structure for proto in layer
    ]
    zones: list[HydrologicZone] = []
    for proto, block in zip(flat, blocks, strict=True):
        ztype: type[HydrologicZone] = type(proto)
        kwargs: dict[str, Tensor] = {}
        for j, pname in enumerate(ztype.parameter_names()):
            # Clone the block column: ``block[:, j]`` is a *view* of the shared
            # ``(batch_size, num_parameters)`` block, so give each zone
            # parameter its own storage (no aliasing of the shared block across
            # zone objects).
            col: Tensor = block[:, j].clone()
            if isinstance(proto, (GroundZone, GroundZoneB)) and pname == "k":
                col = 10.0 ** col
            kwargs[pname] = col
        zones.append(cast(Any, ztype)(**kwargs, name=proto.name))
    return zones


def _rt_blocks(
    model_type: type[Model],
    network: ReactionNetwork,
    zone_configs: dict[str, RtZoneConfiguration],
    arr: NDArray[np.float64],
    device: torch.device,
    dtype: torch.dtype,
) -> list[RtParameters]:
    """Split a batched RT parameter array into per-zone :class:`RtParameters`.

    Mirrors ``Model.rt_params_from_array`` with a leading batch axis: every zone
    consumes ``3 + PARAMETERS_PER_MINERAL * num_minerals`` columns when it
    reacts (dimensions first, then the per-mineral kinetic blocks), else just
    the 3 dimensions.  Mineral ``ssa`` values are in log10 and unpacked via
    ``RtParameters.from_batched_array(natural_scales=True)``.

    Args:
        model_type:  the ``Model`` subclass (fixes zone names/order).
        network:     the reaction network (fixes ``num_minerals``).
        zone_configs: per-zone ``(do_reactions, do_speciation)`` flags.
        arr:         ``(batch_size, num_rt_zone_parameters)`` RT columns.
        device:      compute device.
        dtype:       compute dtype.

    Returns:
        One :class:`RtParameters` per flat zone, in zone order.
    """
    num_minerals: int = network.num_minerals
    zone_names: list[str] = model_type.get_zone_names()
    arr_t: Tensor = torch.as_tensor(arr, dtype=dtype, device=device)

    blocks: list[RtParameters] = []
    lower: int = 0
    for name in zone_names:
        width: int = (
            3 + PARAMETERS_PER_MINERAL * num_minerals
            if zone_configs[name].do_reactions
            else 3
        )
        block: Tensor = arr_t[:, lower : lower + width]
        blocks.append(RtParameters.from_batched_array(block, natural_scales=True))
        lower += width
    return blocks


def _init_state_tensor(
    model_type: type[Model],
    network: ReactionNetwork,
    zone_configs: dict[str, RtZoneConfiguration],
    arr: NDArray[np.float64],
    mean_doc: float,
    init_aq_conc: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Build the ``(batch_size, num_zones, num_species)`` initial RT state.

    Mirrors the scalar ``Model.run_rt_model`` initial-concentration defaults
    (``model.py``): primary aqueous species start at ``init_aq_conc`` except
    DOC which starts at the observed mean ``mean_doc``; minerals start at
    ``10 ** log10_init`` for reacting zones and at :data:`ZERO_CONC` otherwise;
    the ``X-`` exchange site starts at ``10 ** log10_init`` for speciating zones
    and at :data:`ZERO_CONC` otherwise; exchange species start at ``1e-6``.

    The init-state columns of the parameter vector are consumed in the same
    order ``run_hydro_rt_batched`` counts them: first one log10 mineral
    concentration per mineral of every *reacting* zone (in zone order), then
    one log10 exchange-site total per *speciating* zone.

    Args:
        model_type:   the ``Model`` subclass (fixes zone names/order).
        network:      the reaction network (fixes species ordering/sizes).
        zone_configs: per-zone ``(do_reactions, do_speciation)`` flags.
        arr:          ``(batch_size, num_init_columns)`` init-state columns.
        mean_doc:     observed mean river DOC [mmol/L] (DOC initial conc).
        init_aq_conc: initial concentration of the other primary/secondary
            aqueous species.
        device:       compute device.
        dtype:        compute dtype.

    Returns:
        ``(batch_size, num_zones, num_species)`` initial concentrations.
    """
    zone_names: list[str] = model_type.get_zone_names()
    batch_size: int = int(arr.shape[0])
    nz: int = len(zone_names)
    ns: int = network.num_species
    idx: dict[str, int] = {name: i for i, name in enumerate(network.species_names)}

    state: Tensor = torch.full(
        (batch_size, nz, ns), ZERO_CONC, dtype=dtype, device=device
    )

    for prim in network.primary_aqueous:
        col: int = idx[prim.name]
        if prim.name == "DOC":
            state[:, :, col] = mean_doc
        else:
            state[:, :, col] = init_aq_conc
    for sec in network.secondary:
        state[:, :, idx[sec.name]] = init_aq_conc
    for exch in network.exchange_species:
        state[:, :, idx[exch.name]] = 1e-6

    arr_t: Tensor = torch.as_tensor(arr, dtype=dtype, device=device)
    col_ptr: int = 0
    for z, name in enumerate(zone_names):
        if zone_configs[name].do_reactions:
            for mineral_name in network.mineral_names:
                state[:, z, idx[mineral_name]] = 10.0 ** arr_t[:, col_ptr]
                col_ptr += 1
    x_site: int | None = idx.get("X-")
    if x_site is not None and network.has_exchange:
        for z, name in enumerate(zone_names):
            if zone_configs[name].do_speciation:
                state[:, z, x_site] = 10.0 ** arr_t[:, col_ptr]
                col_ptr += 1
    return state


def _forcing_source_tensors(
    forc: ForcingData | list[ForcingData],
    num_surface: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor]:
    """Convert the forcing series to device tensors, one column per surface zone.

    Exactly like ``HydrologicalModel.construct_hydro_forcing_matrix``: a single
    :class:`ForcingData` is broadcast to all surface zones; otherwise ``forc``
    must hold exactly ``num_surface`` entries.  Returns the ``(num_steps,
    num_surface)`` precip / temp / PET source matrices per surface zone.

    Args:
        forc:        a single forcing series (broadcast) or one per surface zone.
        num_surface: number of surface (first-layer) zones.
        device:      compute device.
        dtype:       compute dtype.

    Returns:
        ``(all_precip, all_temp, all_pet)`` each ``(num_steps, num_surface)``.
    """
    forcing_data: list[ForcingData]
    if isinstance(forc, ForcingData):
        forcing_data = [forc] * num_surface
    else:
        forcing_data = list(forc)
        if len(forcing_data) == 1:
            forcing_data = forcing_data * num_surface
        elif len(forcing_data) != num_surface:
            raise ValueError(
                f"The number of forcing data series must be either 1 or "
                f"{num_surface}, not {len(forcing_data)}"
            )

    def _source(attr: str) -> Tensor:
        rows: list[list[float]] = [
            [float(v) for v in getattr(fd, attr).to_numpy()] for fd in forcing_data
        ]
        return torch.tensor(rows, dtype=dtype, device=device).T

    return _source("precip"), _source("temp"), _source("pet")


def _per_lane_forcing(
    geometry: Geometry,
    all_precip: Tensor,
    all_temp: Tensor,
    all_pet: Tensor,
    scales: Tensor,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor]:
    """Distribute the surface forcing to every zone, per lane.

    For each lane the per-zone series are ``all_sources @ (size_mat @
    diag(scales[lane])).T``; temperature additionally uses the row-normalized
    relative matrix (mirrors ``_get_forc_mat(relative=True)``).

    Args:
        geometry:  connectivity geometry (``size_mat`` ``(num_zones, num_surface)``).
        all_precip: ``(num_steps, num_surface)`` precip source matrix.
        all_temp:   ``(num_steps, num_surface)`` temperature source matrix.
        all_pet:    ``(num_steps, num_surface)`` PET source matrix.
        scales:     ``(batch_size, num_surface)`` per-lane surface-zone areas.
        dtype:      compute dtype.

    Returns:
        ``(zone_precip, zone_temp, zone_pet)``, each of shape
        ``(batch_size, num_steps, num_zones)``.
    """
    batch_size: int = int(scales.shape[0])
    num_steps: int = int(all_precip.shape[0])
    num_zones: int = int(geometry.size_mat.shape[0])
    device: torch.device = all_precip.device

    zone_precip: Tensor = torch.empty(
        batch_size, num_steps, num_zones, dtype=dtype, device=device
    )
    zone_temp: Tensor = torch.empty(
        batch_size, num_steps, num_zones, dtype=dtype, device=device
    )
    zone_pet: Tensor = torch.empty(
        batch_size, num_steps, num_zones, dtype=dtype, device=device
    )

    eye: Tensor = torch.eye(scales.shape[1], dtype=dtype, device=device)
    for lane in range(batch_size):
        mat_lane: Tensor = geometry.size_mat @ (eye * scales[lane])
        row_sums: Tensor = mat_lane.sum(dim=1)
        relative_mat: Tensor = torch.zeros_like(mat_lane)
        non_zero: Tensor = row_sums != 0
        relative_mat[non_zero] = (
            mat_lane[non_zero] / row_sums[non_zero].unsqueeze(1)
        )

        zone_precip[lane] = all_precip @ mat_lane.T
        zone_temp[lane] = all_temp @ relative_mat.T
        zone_pet[lane] = all_pet @ mat_lane.T

    return zone_precip, zone_temp, zone_pet


def run_hydro_batched(
    geometry: Geometry,
    zones: list[HydrologicZone],
    zone_precip: Tensor,
    zone_temp: Tensor,
    zone_pet: Tensor,
    dates: DatetimeIndex,
    init_state: Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Tensor]:
    """Run the multi-lane hydrology forward for all zones simultaneously.

    Mirrors ``HydrologicalModel.run_hydro_model`` / ``step_hydro_model`` with
    lanes carried as the leading batch axis.  Zones are stepped in flat order;
    the incoming flux to zone ``i`` at each step is the sum of the external
    lateral/vertical fluxes of the upstream (already solved) zones, exactly as
    in the scalar ``step_hydro_model``.

    Args:
        geometry:    connectivity (``lat_mat`` / ``vert_mat``).
        zones:       per-lane hydrologic zones (flat order).
        zone_precip: ``(batch_size, num_steps, num_zones)`` per-zone precipitation.
        zone_temp:   ``(batch_size, num_steps, num_zones)`` per-zone temperature.
        zone_pet:    ``(batch_size, num_steps, num_zones)`` per-zone PET.
        dates:       simulation timestamps (drives the step lengths).
        init_state:  ``(num_zones,)`` or ``(batch_size, num_zones)`` initial
            storage.
        device:      compute device.
        dtype:       compute dtype.

    Returns:
        A dict of ``(batch_size, num_steps, num_zones)`` tensors: ``state``,
        ``forc_flux``, ``vap_flux``, ``lat_flux``, ``vert_flux``, ``q_in``,
        ``lat_flux_ext``, ``vert_flux_ext``.
    """
    batch_size: int = int(zone_precip.shape[0])
    num_steps: int = int(zone_precip.shape[1])
    num_zones: int = int(zone_precip.shape[2])

    if init_state.dim() == 1:
        state: Tensor = (
            init_state.to(dtype=dtype, device=device)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .clone()
        )
    else:
        state = init_state.to(dtype=dtype, device=device).clone()

    delta_ts: list[float] = _step_lengths(dates, rt_style=False)
    if not delta_ts:
        delta_ts = [1.0] * num_steps

    lat_mat: Tensor = geometry.lat_mat.to(dtype=dtype, device=device)
    vert_mat: Tensor = geometry.vert_mat.to(dtype=dtype, device=device)

    # Phase 3.1: pre-allocated per-(lane, step, zone) output slabs, matching
    # the original ``torch.stack`` layout (the per-zone list was stacked with
    # ``dim=1`` into ``(batch_size, Z)`` per step and the per-step slabs with
    # ``dim=1`` into ``(batch_size, T, Z)``, so the leading axis is the batch
    # lane).  The step loop writes straight into ``out_*[:, t, i]`` instead of
    # appending per-zone tensors to Python lists and ``torch.stack``ing them
    # (8 stacks /step plus the per-zone ``q_in`` matvec rebuilds).  The
    # per-step zone-export accumulators are two dense ``(num_zones,
    # batch_size)`` buffers, so the upstream ``q_in`` is a single matvec
    # against a slice — no list, no ``torch.stack``.
    def _out() -> Tensor:
        return torch.empty(
            batch_size, num_steps, num_zones, dtype=dtype, device=device
        )

    out_state: Tensor = _out()
    out_forc: Tensor = _out()
    out_vap: Tensor = _out()
    out_lat: Tensor = _out()
    out_vert: Tensor = _out()
    out_qin: Tensor = _out()
    out_lat_ext: Tensor = _out()
    out_vert_ext: Tensor = _out()
    lat_ext_step: Tensor = torch.empty(
        num_zones, batch_size, dtype=dtype, device=device
    )
    vert_ext_step: Tensor = torch.empty(
        num_zones, batch_size, dtype=dtype, device=device
    )

    for t in range(num_steps):
        dt: float = delta_ts[t]

        for i in range(num_zones):
            if i > 0:
                # Sum over the already-solved upstream zones: a single dense
                # matvec (``lat_ext_step[:i]`` is ``(i, batch_size)``) replaces the old
                # ``torch.stack(accum[:i])`` + reshape.
                q_in: Tensor = (
                    lat_ext_step[:i].transpose(0, 1) @ lat_mat[i, :i]
                    + vert_ext_step[:i].transpose(0, 1) @ vert_mat[i, :i]
                )
            else:
                q_in = torch.zeros(batch_size, dtype=dtype, device=device)

            d: HydroForcing = HydroForcing(
                precip=zone_precip[:, t, i],
                temp=zone_temp[:, t, i],
                pet=zone_pet[:, t, i],
                q_in=q_in,
            )
            step = zones[i].step(state[:, i], d, dt)

            s_new: Tensor = _as_tensor(step.state)
            lat_ext: Tensor = _as_tensor(step.lat_flux_ext)
            vert_ext: Tensor = _as_tensor(step.vert_flux_ext)

            out_state[:, t, i] = s_new
            out_forc[:, t, i] = _as_tensor(step.forc_flux)
            out_vap[:, t, i] = _as_tensor(step.vap_flux)
            out_lat[:, t, i] = _as_tensor(step.lat_flux)
            out_vert[:, t, i] = _as_tensor(step.vert_flux)
            out_qin[:, t, i] = q_in
            out_lat_ext[:, t, i] = lat_ext
            out_vert_ext[:, t, i] = vert_ext

            state[:, i] = s_new
            lat_ext_step[i, :] = lat_ext
            vert_ext_step[i, :] = vert_ext

    return {
        "state": out_state,
        "forc_flux": out_forc,
        "vap_flux": out_vap,
        "lat_flux": out_lat,
        "vert_flux": out_vert,
        "q_in": out_qin,
        "lat_flux_ext": out_lat_ext,
        "vert_flux_ext": out_vert_ext,
    }


# ---------------------------------------------------------------------- #
# Batched construction helpers                                           #
# ---------------------------------------------------------------------- #


def _hydro_zone_blocks(
    model_type: type[Model],
    arr: NDArray[np.float64],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[Tensor], Tensor]:
    """Split a batched hydrological parameter array into per-zone blocks.

    The hydrological section of a parameter vector stacks every zone's
    ``num_parameters()`` values in flat (structure) order, followed by the
    ``num_surface - 1`` relative-proportion columns.  Each zone block is kept
    as a ``(batch_size, num_parameters)`` tensor (raw, log10 scales intact); the
    proportions are recombined into a ``(batch_size, num_surface)`` per-lane scale
    vector whose last column is ``1 - sum(others)``.

    Args:
        model_type: the hydrology ``Model`` subclass (fixes the layout).
        arr:        ``(batch_size, num_hydro_parameters)`` hydrological columns.
        device:     compute device.
        dtype:      compute dtype.

    Returns:
        ``(blocks, scales)`` where ``blocks[i]`` is the ``(batch_size, num_parameters)``
        block of the ``i``-th flat zone and ``scales`` is ``(batch_size, num_surface)``.
    """
    num_zone_params: int = model_type.get_num_zone_parameters()
    num_size_params: int = model_type.get_num_size_parameters()
    arr_t: Tensor = torch.as_tensor(arr, dtype=dtype, device=device)

    blocks: list[Tensor] = []
    lower: int = 0
    for layer in model_type.structure:
        for proto in layer:
            n: int = proto.num_parameters()
            blocks.append(arr_t[:, lower : lower + n])
            lower += n

    size_params: Tensor = arr_t[
        :, num_zone_params : num_zone_params + num_size_params
    ]
    last_scale: Tensor = 1.0 - size_params.sum(dim=1, keepdim=True)
    scales: Tensor = torch.cat([size_params, last_scale], dim=1)
    return blocks, scales