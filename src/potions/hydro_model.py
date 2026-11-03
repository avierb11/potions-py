"""NetworkX-free batchable hydrological model (port of the golden ``hydro_model``).

The golden reference constructs a ``networkx.DiGraph`` describing lateral /
vertical zone connectivity and derives adjacency matrices from it via
``get_connection_matrices_with_river_row``.  ``networkx`` is **not** a
dependency of this port, so the two connectivity matrices are instead built
directly from the nested ``structure`` with the exact same wiring rules:

* **Lateral (within a layer, towards the river):** zone ``j`` of layer ``l``
  discharges laterally into zone ``j + 1`` of the same layer.  The last zone of
  each layer is a *river* zone: its lateral outflow contributes to streamflow
  and it is flagged in the augmented ``lat_mat`` river row.
* **Vertical (between adjacent layers):** when both layers hold the same number
  of zones each zone percolates straight down; when the lower layer holds a
  single zone, every zone of the upper layer drains into it (the golden
  aggregation rule).  Other layer-count combinations get no vertical edges
  (a warning is printed), matching the golden behaviour.
* **Forcing:** forcing is distributed to *every* zone through the golden
  forcing matrices ``precip_mat = pet_mat = get_forc_mat(scales)`` and
  ``temp_mat = get_forc_mat(scales, relative=True)``, i.e.
  ``size_mat @ diag(scales)`` where ``size_mat`` maps each zone to the surface
  zones that contribute vertically to it. A single forcing series is
  broadcast, otherwise one series per surface zone is required.

The public surface mirrors the golden: :class:`HydrologicalModel` wraps a list
of :class:`Layer` objects, and a full forward run returns
:class:`~potions.common_types.HydroModelResults` whose ``simulation``
``DataFrame`` holds ``8 * num_zones`` columns (``s_<z>``, ``q_forc_<z>``,
``q_vap_<z>``, ``q_lat_<z>``, ``q_vert_<z>``, ``q_in_<z>``, ``q_lat_ext_<z>``,
``q_vert_ext_<z>``) plus ``sim_streamflow_mmd``.
"""

from __future__ import annotations

from typing import Callable, Iterable, Iterator, TypeAlias

import pandas as pd
import torch
from torch import Tensor

from .common_types import (
    ForcingData,
    HydroForcing,
    HydroModelResults,
    HydroModelStep,
    Scalar,
)
from .hydro import HydrologicZone

#: Columns stored per zone in the output DataFrame (golden
#: ``OUTPUT_COLUMNS_PER_ZONE``).
OUTPUT_COLUMNS_PER_ZONE: int = 8


LayerT: TypeAlias = list[HydrologicZone]


def _t(x: Scalar) -> Tensor:
    """Coerce a python scalar to a 0-dim float64 tensor; pass tensors through."""
    if isinstance(x, (int, float)):
        return torch.tensor(float(x), dtype=torch.float64)
    return x.detach()


def _broadcast(x: Scalar, shape: tuple[int, ...], dtype: torch.dtype) -> Tensor:
    """Coerce a python scalar / tensor to a tensor of ``shape`` (broadcast)."""
    t: Tensor = _t(x)
    if t.shape == shape:
        return t.to(dtype=dtype)
    return t.expand(shape).to(dtype=dtype)


class Layer:
    """A single hydrologic layer (a homogeneous band of zones)."""

    zones: list[HydrologicZone]

    def __init__(self, zones: list[HydrologicZone]) -> None:
        self.zones = list(zones)

    def __len__(self) -> int:
        return len(self.zones)

    def __iter__(self) -> Iterator[HydrologicZone]:
        return iter(self.zones)

    def __getitem__(self, idx: int) -> HydrologicZone:
        return self.zones[idx]


class HydrologicalModel:
    """A networkx-free hydrological model made of zone ``layers``.

    Args:
        structure: Nested list of zones; each inner list is a layer evaluated
            from the surface downwards. The first layer holds the surface
            zones.
        scales: optional relative areas of the surface zones (must sum to 1).
            Defaults to equal scales.
        verbose: print per-step diagnostics. Defaults to False.
    """

    def __init__(
        self,
        structure: list[LayerT],
        scales: Iterable[float] | None = None,
        verbose: bool = False,
    ) -> None:
        if structure == []:
            raise ValueError(
                "Model structure is empty, make sure you define a model "
                "structure class property describing your model"
            )

        self.__structure: list[LayerT] = structure

        # Flat, evaluation-ordered zone list.
        self.__flat: list[HydrologicZone] = [
            zone for layer in structure for zone in layer
        ]

        num_surface: int = len(self.__structure[0])
        if scales is None:
            self.__scales: list[float] = [1 / num_surface for _ in range(num_surface)]
        else:
            scales_list: list[float] = list(scales)
            if len(scales_list) != num_surface:
                raise ValueError(
                    f"The number of scales ({len(scales_list)}) must equal "
                    f"the number of surface zones ({num_surface})"
                )
            if abs(sum(scales_list) - 1) > 1e-3:
                raise ValueError(
                    "Scales must sum to 1.0 to maintain water balance"
                )
            self.__scales = scales_list

        self.verbose: bool = verbose

        # Networkx-free connectivity matrices.
        self.__lat_mat, self.__vert_mat = self._build_adjacency_matrices()

        # Forcing distribution matrices (golden ``get_forc_mat`` /
        # ``get_size_mat``): precip_mat = pet_mat = size_mat @ diag(scales),
        # temp_mat = the same with each row normalized to sum to 1.
        self.__precip_mat: Tensor = self._get_forc_mat(self.__scales, relative=False)
        self.__pet_mat: Tensor = self.__precip_mat
        self.__temp_mat: Tensor = self._get_forc_mat(self.__scales, relative=True)

    # ------------------------------------------------------------------ #
    # Properties / iteration                                              #
    # ------------------------------------------------------------------ #

    @property
    def structure(self) -> list[LayerT]:
        return self.__structure

    @property
    def flat_model(self) -> list[HydrologicZone]:
        return self.__flat

    @property
    def hydro_zones(self) -> dict[str, HydrologicZone]:
        """Zone instances keyed by name (golden ``hydro_zones``)."""
        return {zone.name: zone for zone in self.__flat}

    @property
    def scales(self) -> list[float]:
        return self.__scales

    @property
    def zone_scales(self) -> torch.Tensor:
        """Per-zone area fractions (golden ``zone_scales``).

        Propagates the surface-zone area fractions down to every zone: a zone
        inherits the fraction of the surface column it drains, i.e.
        ``size_mat @ scales`` (a length-``num_zones`` vector in evaluation
        order).  Used to weight river-zone lateral outflows so the outlet is a
        catchment-mean depth rather than an unweighted sum of zone fluxes.
        """
        return self._get_size_mat() @ torch.tensor(self.scales, dtype=torch.float64)

    @property
    def num_surface_zones(self) -> int:
        """Number of surface zones (zones in the first layer)."""
        return len(self.__structure[0])

    @property
    def surface_zone_ids(self) -> list[int]:
        """Flat indices of the surface zones (first layer; no vertical inflow).

        These zones receive precipitation chemistry as their incoming
        reactive-transport concentration instead of an upstream mass flux.
        """
        return list(range(self.num_surface_zones))

    @property
    def num_zones(self) -> int:
        return len(self.__flat)

    @property
    def lat_mat(self) -> Tensor:
        """(n+1) x n lateral adjacency matrix incl. the river outflow row."""
        return self.__lat_mat

    @property
    def vert_mat(self) -> Tensor:
        """n x n vertical adjacency matrix."""
        return self.__vert_mat

    def zone_names(self) -> list[str]:
        return [zone.name for zone in self.__flat]

    def get_zone_names(self) -> list[str]:
        return self.zone_names()

    def __len__(self) -> int:
        return len(self.__flat)

    def __iter__(self) -> Iterator[HydrologicZone]:
        return iter(self.__flat)

    def __getitem__(self, i: int) -> HydrologicZone:
        return self.__flat[i]

    # ------------------------------------------------------------------ #
    # Connectivity (networkx-free)                                        #
    # ------------------------------------------------------------------ #

    def _build_adjacency_matrices(self) -> tuple[Tensor, Tensor]:
        """Build the lateral & vertical adjacency matrices from ``structure``.

        Returns:
            tuple[Tensor, Tensor]: ``(lat_mat, vert_mat)`` where ``lat_mat``
                has shape ``(n + 1, n)`` (the last row flags river zones) and
                ``vert_mat`` has shape ``(n, n)``.
        """
        num_zones: int = self.num_zones

        lat: torch.Tensor = torch.zeros((num_zones + 1, num_zones), dtype=torch.float64)
        vert: torch.Tensor = torch.zeros((num_zones, num_zones), dtype=torch.float64)

        # Flattened index of each (layer_idx, zone_idx).
        index_of: dict[tuple[int, int], int] = {}
        counter: int = 0
        for l_idx, layer in enumerate(self.__structure):
            for z_idx in range(len(layer)):
                index_of[(l_idx, z_idx)] = counter
                counter += 1

        # 1. Lateral connections within each layer (zone j -> zone j + 1,
        #    towards the river).  The last zone of each layer is a river zone.
        for l_idx, layer in enumerate(self.__structure):
            for z_idx in range(len(layer)):
                if z_idx < len(layer) - 1:
                    src: int = index_of[(l_idx, z_idx)]
                    dst: int = index_of[(l_idx, z_idx + 1)]
                    lat[dst, src] = 1.0
                else:
                    # River zone: outflow row marks the river.
                    lat[num_zones, index_of[(l_idx, z_idx)]] = 1.0

        # 2. Vertical connections between adjacent layers.
        for l_idx in range(len(self.__structure) - 1):
            cur: LayerT = self.__structure[l_idx]
            nxt: LayerT = self.__structure[l_idx + 1]

            if len(cur) == len(nxt):
                # Same number of zones: each flows straight down.
                for z_idx in range(len(cur)):
                    vert[
                        index_of[(l_idx + 1, z_idx)],
                        index_of[(l_idx, z_idx)],
                    ] = 1.0
            elif len(nxt) == 1:
                # All upper zones flow into the single aggregated zone below.
                dst_agg: int = index_of[(l_idx + 1, 0)]
                for z_idx in range(len(cur)):
                    vert[dst_agg, index_of[(l_idx, z_idx)]] = 1.0
            else:
                if self.verbose:
                    print(
                        f"Warning: Layer {l_idx} has {len(cur)} zones, but "
                        f"Layer {l_idx + 1} has {len(nxt)} zones; no vertical "
                        "connections added for this layer pair as it violates rules."
                    )

        return lat, vert

    def get_river_zone_ids(self) -> list[int]:
        """Indices (flat order) of the zones discharging to the river.

        By convention these are the last zones of each layer.
        """
        river_zones: list[int] = []
        current: int = 0
        for layer in self.__structure:
            if len(layer) > 0:
                river_zones.append(current + len(layer) - 1)
            current += len(layer)
        return river_zones

    def _get_size_mat(self) -> Tensor:
        """Map every zone to the surface zones that contribute to it vertically.

        Rows of ``vert_mat`` with (nearly) zero sum identify the surface zones
        (no vertical inflow); each such row is a unit vector on its own
        surface index, and every other row is the sum of its vertical parents'
        rows (golden ``get_size_mat``).

        Returns:
            Tensor: ``(num_zones, num_surface)`` binary contribution matrix.
        """
        vert: Tensor = self.__vert_mat
        index_rows: list[int] = [
            i for i, row in enumerate(vert) if float(row.sum()) < 1e-12
        ]
        mat: torch.Tensor = torch.zeros(
            (self.num_zones, len(index_rows)), dtype=torch.float64
        )
        for i, x_i in enumerate(index_rows):
            mat[x_i, i] = 1.0
        for c_i in range(self.num_zones):
            if c_i in index_rows:
                continue
            mat[c_i] = sum(
                mat[i]
                for i, row_i in enumerate(vert[c_i])
                if float(row_i) != 0.0
            )
        return mat

    def _get_forc_mat(self, sizes: list[float], relative: bool = False) -> Tensor:
        """Combine surface-zone sizes with connectivity (golden ``get_forc_mat``).

        Args:
            sizes: relative area of each surface zone.
            relative: if True, normalize each matrix row to sum to 1 (used for
                intensive variables like temperature). Defaults to False.

        Returns:
            Tensor: ``(num_zones, num_surface)`` forcing distribution matrix.
        """
        conn_mat: torch.Tensor = self._get_size_mat()
        s_arr: torch.Tensor = torch.tensor(sizes, dtype=torch.float64)
        mat: torch.Tensor = conn_mat @ torch.diag(s_arr)

        if relative:
            row_sums: torch.Tensor = mat.sum(dim=1)
            relative_mat: torch.Tensor = torch.zeros_like(mat)
            non_zero: torch.Tensor = row_sums != 0
            relative_mat[non_zero] = mat[non_zero] / row_sums[non_zero].unsqueeze(1)
            return relative_mat
        return mat

    def construct_hydro_forcing_matrix(
        self,
        forc: ForcingData | Iterable[ForcingData],
    ) -> list[list[HydroForcing]]:
        """Distribute per-surface-zone forcing to every zone for all time steps.

        Mirrors the golden ``construct_hydro_forcing_matrix``: per-zone forcing
        series are ``all_sources @ mat.T`` where ``all_sources`` is the
        ``(num_steps, num_surface)`` matrix of the raw surface-zone series and
        ``mat`` is the corresponding forcing matrix (precip/pet use the
        non-relative matrix; temp the relative one).

        Args:
            forc: A single :class:`ForcingData` (broadcast to all surface
                zones) or a list with one entry per surface zone.

        Returns:
            list[list[HydroForcing]]: per-step, per-zone forcing, ordered
                ``[step][flat_zone_idx]``, with ``q_in`` left at 0.0.
        """
        # Resolve the per-surface-zone source forcing list.
        forcing_data: list[ForcingData]
        if isinstance(forc, ForcingData):
            forcing_data = [forc] * self.num_surface_zones
        else:
            forcing_data = list(forc)
            if len(forcing_data) != 1:
                if len(forcing_data) != self.num_surface_zones:
                    raise ValueError(
                        f"The number of forcing data series must be either 1 or "
                        f"{self.num_surface_zones}, not {len(forcing_data)}"
                    )

        num_forcing_sources_expected: int = int(self.__precip_mat.shape[1])
        if len(forcing_data) != num_forcing_sources_expected:
            raise ValueError(
                f"Model expects {num_forcing_sources_expected} ForcingData "
                f"objects, but {len(forcing_data)} were provided."
            )
        num_steps: int = len(forcing_data[0].precip)
        for i, fd in enumerate(forcing_data):
            if len(fd.precip) != num_steps or len(fd.temp) != num_steps or len(fd.pet) != num_steps:
                raise ValueError(
                    f"ForcingData at index {i} has series lengths inconsistent "
                    f"with num_steps ({num_steps})."
                )

        # Source matrices: (num_steps, num_surface).
        all_precip_sources: torch.Tensor = torch.tensor(
            [[float(p) for p in fd.precip.to_numpy()] for fd in forcing_data],
            dtype=torch.float64,
        ).T
        all_temp_sources: torch.Tensor = torch.tensor(
            [[float(t) for t in fd.temp.to_numpy()] for fd in forcing_data],
            dtype=torch.float64,
        ).T
        all_pet_sources: torch.Tensor = torch.tensor(
            [[float(p) for p in fd.pet.to_numpy()] for fd in forcing_data],
            dtype=torch.float64,
        ).T

        # Distribute to every zone: (num_steps, num_zones).
        zone_precip: torch.Tensor = all_precip_sources @ self.__precip_mat.T
        zone_temp: torch.Tensor = all_temp_sources @ self.__temp_mat.T
        zone_pet: torch.Tensor = all_pet_sources @ self.__pet_mat.T

        out: list[list[HydroForcing]] = []
        for t in range(num_steps):
            step_forcing: list[HydroForcing] = []
            for j in range(self.num_zones):
                step_forcing.append(
                    HydroForcing(
                        precip=float(zone_precip[t, j]),
                        temp=float(zone_temp[t, j]),
                        pet=float(zone_pet[t, j]),
                        q_in=0.0,
                    )
                )
            out.append(step_forcing)
        return out

    # ------------------------------------------------------------------ #
    # Single step                                                         #
    # ------------------------------------------------------------------ #

    def step_hydro_model(
        self,
        state: Tensor,
        ds: Iterable[HydroForcing],
        dt: Scalar,
        check_water_balance: bool = False,
    ) -> HydroModelStep:
        """Advances all zones by a single time step.

        Args:
            state: current storage of all zones, shape ``(num_zones,)`` or
                ``(B, num_zones)`` [mm].
            ds:    one :class:`HydroForcing` per zone for this step.
            dt:    step length [days].

        Returns:
            HydroModelStep: per-zone new states and fluxes (batched tensors,
                shape ``(B, num_zones)`` or ``(num_zones,)``).
        """
        ds_list: list[HydroForcing] = list(ds)
        num_zones: int = self.num_zones
        if len(ds_list) != num_zones:
            raise ValueError(
                f"Expected {num_zones} forcing objects, got {len(ds_list)}"
            )

        batched: bool = state.dim() >= 2
        B: int = state.shape[0] if batched else 1

        s0: torch.Tensor = (
            state.reshape(B, -1).clone()
            if batched
            else state.reshape(1, -1).clone()
        )
        s_new: torch.Tensor = s0.clone()

        new_states: list[torch.Tensor] = []
        forc_fluxes: list[torch.Tensor] = []
        vap_fluxes: list[torch.Tensor] = []
        lat_fluxes: list[torch.Tensor] = []
        vert_fluxes: list[torch.Tensor] = []
        q_ins: list[torch.Tensor] = []
        lat_ext_fluxes: list[torch.Tensor] = []
        vert_ext_fluxes: list[torch.Tensor] = []

        lat_ext_accum: list[torch.Tensor] = []
        vert_ext_accum: list[torch.Tensor] = []

        for i, (zone, d_i) in enumerate(zip(self.__flat, ds_list, strict=True)):
            # Incoming flux to zone i from upstream zones (flattened order).
            if i > 0:
                q_in_lat: torch.Tensor = (
                    self.__lat_mat[i, :i]
                    .reshape(1, -1)
                    .to(dtype=s0.dtype, device=s0.device)
                    @ torch.stack(lat_ext_accum[:i], dim=0)
                ).reshape(-1)
                q_in_vert: torch.Tensor = (
                    self.__vert_mat[i, :i]
                    .reshape(1, -1)
                    .to(dtype=s0.dtype, device=s0.device)
                    @ torch.stack(vert_ext_accum[:i], dim=0)
                ).reshape(-1)
            else:
                q_in_lat = torch.zeros(B, dtype=s0.dtype, device=s0.device)
                q_in_vert = torch.zeros(B, dtype=s0.dtype, device=s0.device)

            q_in: torch.Tensor = q_in_lat + q_in_vert

            d_i_zone: HydroForcing = HydroForcing(
                precip=_broadcast(d_i.precip, s0[:, i].shape, s0.dtype),
                temp=_broadcast(d_i.temp, s0[:, i].shape, s0.dtype),
                pet=_broadcast(d_i.pet, s0[:, i].shape, s0.dtype),
                q_in=q_in,
            )

            step_res = zone.step(s_new[:, i], d_i_zone, dt)

            new_states.append(_t(step_res.state))
            forc_fluxes.append(_t(step_res.forc_flux))
            vap_fluxes.append(_t(step_res.vap_flux))
            lat_fluxes.append(_t(step_res.lat_flux))
            vert_fluxes.append(_t(step_res.vert_flux))
            q_ins.append(_t(q_in))
            lat_ext_fluxes.append(_t(step_res.lat_flux_ext))
            vert_ext_fluxes.append(_t(step_res.vert_flux_ext))

            s_new[:, i] = step_res.state

            lat_ext_accum.append(_t(step_res.lat_flux_ext).detach())
            vert_ext_accum.append(_t(step_res.vert_flux_ext).detach())

        def _stack(xs: list[torch.Tensor]) -> torch.Tensor:
            return torch.stack(xs, dim=-1)

        st: torch.Tensor = _stack(new_states)
        return HydroModelStep(
            state=st,
            forc_flux=_stack(forc_fluxes),
            vap_flux=_stack(vap_fluxes),
            lat_flux=_stack(lat_fluxes),
            vert_flux=_stack(vert_fluxes),
            q_in=_stack(q_ins),
            lat_flux_ext=_stack(lat_ext_fluxes),
            vert_flux_ext=_stack(vert_ext_fluxes),
        )

    # ------------------------------------------------------------------ #
    # Full run                                                            #
    # ------------------------------------------------------------------ #

    def run_hydro_model(
        self,
        forc: ForcingData | list[ForcingData],
        init_state: torch.Tensor | None = None,
        meas_streamflow: torch.Tensor | pd.Series | None = None,
        objective_functions: Iterable[
            tuple[str, Callable[[Tensor, Tensor], float]]
        ] | None = None,
        dt: Scalar = 1.0,
        check_water_balance: bool = False,
    ) -> HydroModelResults:
        """Runs a forward simulation of the hydrologic model.

        Args:
            forc: A single :class:`ForcingData` (broadcast to all surface
                zones) or a list with one per surface zone.
            init_state: optional initial storage per zone. Defaults to the
                default init state of each zone.
            meas_streamflow: observed streamflow series for objective metrics.
            objective_functions: named objective functions
                ``(name, callable(sim, meas) -> float)``. Defaults to named
                NSE / KGE metrics when ``meas_streamflow`` is provided.
            dt: step length [days]. Defaults to 1.0.

        Returns:
            HydroModelResults: per-step state/flux ``DataFrame``
                (``8 * num_zones`` columns + ``sim_streamflow_mmd``) and the
                objective-function series.
        """
        if init_state is None:
            init_state = torch.tensor(
                [
                    zone.default_init_state()
                    for zone in self.__flat
                ],
                dtype=torch.float64,
            )
        state: torch.Tensor = init_state.clone().reshape(-1)

        # Dates / step sizes (golden: day-differences between consecutive
        # forcing timestamps, last dt repeated).
        dates = (
            forc.precip.index
            if isinstance(forc, ForcingData)
            else forc[0].precip.index
        )
        num_steps: int = len(dates)
        if isinstance(dates, pd.DatetimeIndex) and num_steps > 1:
            delta_ts: list[float] = [
                float((dates[i] - dates[i - 1]).days) for i in range(1, num_steps)
            ]
            delta_ts.append(delta_ts[-1])
        else:
            delta_ts = [float(dt)] * num_steps
        if any(t <= 1e-5 for t in delta_ts):
            raise ValueError(
                "Invalid step size in hydrologic simulation, make sure all "
                "simulation points are >= 1e-5 days (~1 second)"
            )

        # Per-step, per-zone forcing from the forcing matrix.
        forcing_matrix = self.construct_hydro_forcing_matrix(forc)

        states: list[torch.Tensor] = []
        forc_fluxes: list[torch.Tensor] = []
        vap_fluxes: list[torch.Tensor] = []
        lat_fluxes: list[torch.Tensor] = []
        vert_fluxes: list[torch.Tensor] = []
        q_ins: list[torch.Tensor] = []
        lat_ext_fluxes: list[torch.Tensor] = []
        vert_ext_fluxes: list[torch.Tensor] = []

        river_ids: list[int] = self.get_river_zone_ids()

        for t in range(num_steps):
            step_res: HydroModelStep = self.step_hydro_model(
                state,
                forcing_matrix[t],
                delta_ts[t],
                check_water_balance=check_water_balance,
            )

            state = step_res.state.reshape(-1)
            states.append(step_res.state.reshape(-1))
            forc_fluxes.append(step_res.forc_flux.reshape(-1))
            vap_fluxes.append(step_res.vap_flux.reshape(-1))
            lat_fluxes.append(step_res.lat_flux.reshape(-1))
            vert_fluxes.append(step_res.vert_flux.reshape(-1))
            q_ins.append(step_res.q_in.reshape(-1))
            lat_ext_fluxes.append(step_res.lat_flux_ext.reshape(-1))
            vert_ext_fluxes.append(step_res.vert_flux_ext.reshape(-1))

        def _stack(xs: list[torch.Tensor]) -> torch.Tensor:
            return torch.stack(xs, dim=0)

        # Streamflow: catchment-mean outlet depth (mm/d). Each river zone's
        # internal lateral flux is weighted by the area fraction of the column
        # it discharges from (``size_mat @ scales``), so the outlet is a
        # catchment mean rather than an unweighted sum of intensive zone fluxes
        # (golden ``run_hydro_model`` behaviour).
        lat: torch.Tensor = _stack(lat_fluxes)
        zone_scale: torch.Tensor = self.zone_scales.to(lat.device)
        sim_streamflow: torch.Tensor = (lat * zone_scale)[:, river_ids].sum(dim=1)

        # Measured streamflow tensor (num_steps,).
        meas_t: torch.Tensor | None
        if meas_streamflow is not None:
            meas_t = (
                meas_streamflow
                if isinstance(meas_streamflow, torch.Tensor)
                else torch.tensor(meas_streamflow.to_numpy(), dtype=torch.float64)
            )
        else:
            meas_t = None

        # DataFrame: per-zone columns in golden order.
        data: dict[str, torch.Tensor] = {}
        for i, name in enumerate(self.zone_names()):
            data[f"s_{name}"] = _stack(states)[:, i]
            data[f"q_forc_{name}"] = _stack(forc_fluxes)[:, i]
            data[f"q_vap_{name}"] = _stack(vap_fluxes)[:, i]
            data[f"q_lat_{name}"] = _stack(lat_fluxes)[:, i]
            data[f"q_vert_{name}"] = _stack(vert_fluxes)[:, i]
            data[f"q_in_{name}"] = _stack(q_ins)[:, i]
            data[f"q_lat_ext_{name}"] = _stack(lat_ext_fluxes)[:, i]
            data[f"q_vert_ext_{name}"] = _stack(vert_ext_fluxes)[:, i]
        data["sim_streamflow_mmd"] = sim_streamflow

        df = pd.DataFrame({k: v.detach().numpy() for k, v in data.items()}, index=dates)

        # Streamflow-derived columns (golden ``run_hydro_model`` behaviour).
        if meas_t is not None:
            df["meas_streamflow_mmd"] = meas_t.detach().numpy()
        for river_id in river_ids:
            river_name: str = self.zone_names()[river_id]
            df[f"prop_q_{river_name}"] = df[f"q_lat_{river_name}"] / df["sim_streamflow_mmd"]

        # ---- Objectives ---- #
        obj_vals: dict[str, float] = {}
        if meas_t is not None:
            if objective_functions is None:
                funcs: list[tuple[str, Callable[[Tensor, Tensor], float]]] = list(DEFAULT_OBJECTIVE_FUNCTIONS)
            else:
                funcs = list(objective_functions)
            for name, fn in funcs:
                obj_vals[name] = float(fn(meas_t, sim_streamflow))
        objective_functions_ser: pd.Series = pd.Series(obj_vals)

        return HydroModelResults(
            forcing=forc,
            simulation=df,
            objective_functions=objective_functions_ser,
        )


# ---------------------------------------------------------------------- #
# Default objective metrics (golden ``objective_functions`` names/formulas).
# ---------------------------------------------------------------------- #


def _to_series(x: torch.Tensor) -> pd.Series:
    """Convert a 1-D tensor to a float ``pandas.Series`` (ddof-1 semantics)."""
    return pd.Series(x.detach().cpu().numpy(), dtype=float)


def _mean_std(y: pd.Series) -> tuple[float, float]:
    """Pandas mean / (ddof=1) standard deviation of a series."""
    return float(y.mean()), float(y.std())


def kge(y_obs: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Kling-Gupta efficiency (golden ``kge``)."""
    obs: pd.Series = _to_series(y_obs)
    pred: pd.Series = _to_series(y_pred)
    r: float = float(pred.corr(obs, method="pearson"))
    mean_o: float
    std_o: float
    mean_o, std_o = _mean_std(obs)
    mean_p: float
    std_p: float
    mean_p, std_p = _mean_std(pred)
    alpha: float = std_p / std_o if std_o > 0.0 else float("nan")
    beta: float = mean_p / mean_o if mean_o != 0.0 else float("nan")
    return float(1.0 - ((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2) ** 0.5)


def log_kge(y_obs: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Kling-Gupta efficiency of the log-transformed data (golden ``log_kge``)."""
    return kge(
        torch.log10(y_obs + 0.1),
        torch.log10(y_pred + 0.1),
    )


def nse(y_obs: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Nash-Sutcliffe efficiency (golden ``nse``)."""
    obs: pd.Series = _to_series(y_obs)
    pred: pd.Series = _to_series(y_pred)
    numer: float = float(((pred - obs) ** 2).sum())
    denom: float = float(((obs - obs.mean()) ** 2).sum())
    if denom == 0.0:
        return float("nan")
    return 1.0 - numer / denom


def log_nse(y_obs: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Nash-Sutcliffe efficiency of the log-transformed data (golden ``log_nse``)."""
    return nse(
        torch.log10(y_obs + 0.1),
        torch.log10(y_pred + 0.1),
    )


def bias(y_obs: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Bias of the simulated streamflow (golden ``bias``)."""
    obs: pd.Series = _to_series(y_obs)
    pred: pd.Series = _to_series(y_pred)
    mean_o: float = float(obs.mean())
    if mean_o == 0.0:
        return float("nan")
    return float((pred - obs).mean() / mean_o)


def spearman_rho(y_obs: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Spearman rank correlation (golden ``spearman_rho``)."""
    obs: pd.Series = _to_series(y_obs)
    pred: pd.Series = _to_series(y_pred)
    return float(obs.corr(pred, method="spearman"))


def r_squared(y_obs: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Pearson correlation squared (golden ``r_squared``)."""
    obs: pd.Series = _to_series(y_obs)
    pred: pd.Series = _to_series(y_pred)
    return float(obs.corr(pred, method="pearson") ** 2)


#: Default objective functions (names match the golden reference).
DEFAULT_OBJECTIVE_FUNCTIONS: list[
    tuple[str, Callable[[Tensor, Tensor], float]]
] = [
    ("kge", kge),
    ("log_kge", log_kge),
    ("nse", nse),
    ("log_nse", log_nse),
    ("bias", bias),
    ("spearman_rho", spearman_rho),
    ("r_squared", r_squared),
]
