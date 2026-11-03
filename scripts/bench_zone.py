"""Zone-level microbenchmarks: isolate hydrology and reactive-transport cost.

Unlike the full-record ``profile_batched.py`` (which mixes device transfers,
cross-zone coupling and the whole pipeline), these benches time **one zone**
stepping over a short forcing window, so a single kernel / solver can be
measured in isolation:

- ``--zone hydro``: a single :class:`SurfaceZone` storage solve
  (one damped AD-Newton root find per day) over the forcing series;
- ``--zone rt``:    a single reacting :class:`RtZone` step (fused Newton/LM
  solve + equilibrium re-speciation) over the same window;

and reports ``us/step`` for each configuration so improvements to a solver can
be attributed without a full run.  Sweep ``device`` / ``dtype`` to compare:

    python scripts/bench_zone.py --zone hydro --days 60
    python scripts/bench_zone.py --zone rt --days 60
    python scripts/bench_zone.py --zone both --device cuda --days 2557

The window defaults to the full record like ``profile_batched.py``; ``--days``
limits it (2557-day windows are fine for one zone).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, cast

import numpy as np
import torch
from pandas import DataFrame, DatetimeIndex, Series

SCRIPTS: str = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)

import east_river_forward as fwd  # noqa: E402  (registers showcase paths)
from utils.parameter_ranges import load_input_data  # noqa: E402
from data.raw.east_river import hbv_data, load_all_doc_data  # noqa: E402

import potions as pt  # noqa: E402  # type: ignore[import-not-found]
from potions.common_types import HydroForcing, HydroStep, RtForcing  # noqa: E402
from potions.device import default_dtype  # noqa: E402
from potions.model import HydrologicalModel  # noqa: E402
from potions.model_batched import (  # noqa: E402  # type: ignore[import-not-found]
    BatchedConfig,
    Geometry,
    _build_hydro_zones,
    _hydro_zone_blocks,
    _per_lane_forcing,
    _forcing_source_tensors,
    _rt_blocks,
)
from potions.rt_zone import RtZone  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zone", choices=["hydro", "rt", "both"], default="both")
    parser.add_argument("--days", type=int, default=0, help="window days (0 = full record)")
    parser.add_argument("--device", default="auto", help="cpu | cuda | auto")
    parser.add_argument("--dtype", default="auto", help="float32 | float64 | auto")
    parser.add_argument("--reps", type=int, default=3, help="repeated timing passes")
    args = parser.parse_args()

    device: torch.device = (
        torch.device("cuda")
        if args.device == "auto" and torch.cuda.is_available()
        else torch.device(args.device)
    )
    dtype: torch.dtype = default_dtype() if args.device == "auto" else (
        torch.float32 if args.dtype == "float32" else torch.float64
    )
    if args.device == "auto" and args.dtype == "float64":
        dtype = torch.float64

    # ---- config (full record unless --days) ---- #
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
    dates: DatetimeIndex = cast(DatetimeIndex, site_data.index)
    window: DatetimeIndex = fwd.choose_window(
        dates, wy=None, start=None, end=None, days=(args.days or None), full=args.days == 0
    )
    cfg: BatchedConfig = cast(
        BatchedConfig,
        fwd.build_window_config(inputs.cal_config, site_data, doc_df, window),
    )
    num_steps: int = len(window)
    print(
        f"== zone bench ==  window {window[0]} .. {window[-1]}"
        f"  ({num_steps} days)  device: {device}  dtype: {dtype}"
    )

    model_type: type[pt.Model] = cfg["model_type"]
    zone_configs: dict[str, Any] = cfg["zone_configs"]
    network = cfg["network"]
    num_hydro: int = model_type.get_num_hydro_parameters()
    num_rt: int = model_type.get_num_rt_parameters(zone_configs, network)

    # One lane sampled from the mid of every hydro + RT bound (avoids overflow).
    mid: list[float] = []
    for nm in inputs.bounds:
        lo, hi = inputs.bounds[nm]
        mid.append(float(0.5 * (lo + hi)))
    arr: np.ndarray = np.asarray(mid, dtype=float).reshape(1, -1)
    hydro_arr: np.ndarray = arr[:, :num_hydro]
    rt_arr: np.ndarray = arr[:, num_hydro : num_hydro + num_rt]

    # Shared geometry + real per-day forcing for the probe zones (single lane).
    blocks, scales = _hydro_zone_blocks(model_type, hydro_arr, device, dtype)
    geo = HydrologicalModel(structure=model_type.structure)
    surface_idx: int = int(geo.surface_zone_ids[0])
    all_p, all_t, all_pet = _forcing_source_tensors(
        cfg["forc"], len(geo.surface_zone_ids), device, dtype
    )
    zp, zt, zpet = _per_lane_forcing(
        Geometry(
            lat_mat=geo.lat_mat.to(dtype=dtype, device=device),
            vert_mat=geo.vert_mat.to(dtype=dtype, device=device),
            size_mat=geo._get_size_mat().to(dtype=dtype, device=device),
            river_ids=geo.get_river_zone_ids(),
            surface_ids=geo.surface_zone_ids,
            zone_names=geo.zone_names(),
        ),
        all_p, all_t, all_pet, scales, dtype,
    )

    if args.zone in ("hydro", "both"):
        print("\n== hydrology: single SurfaceZone step ==")
        zones = _build_hydro_zones(model_type, blocks)
        zone = zones[surface_idx]
        s_0: torch.Tensor = torch.full((1,), 25.0, dtype=dtype, device=device)
        d_hydro: list[HydroForcing] = [
            HydroForcing(
                precip=zp[0, t, surface_idx],
                temp=zt[0, t, surface_idx],
                pet=zpet[0, t, surface_idx],
                q_in=torch.zeros(1, dtype=dtype, device=device),
            )
            for t in range(num_steps)
        ]
        t0: float = time.perf_counter()
        s: torch.Tensor = s_0.clone()
        for _rep in range(args.reps):
            for d in d_hydro:
                s = cast(torch.Tensor, zone.step(s, d).state)
        wall: float = time.perf_counter() - t0
        per_us: float = wall / (num_steps * args.reps) * 1e6
        print(f"   {per_us:9.1f} us/step   ({num_steps} steps, {args.reps} reps)")

    if args.zone in ("rt", "both"):
        print("\n== reactive transport: single reacting RtZone step ==")
        rt_pars = _rt_blocks(model_type, network, zone_configs, rt_arr, device, dtype)
        zone_names: list[str] = model_type.get_zone_names()
        zname: str = next(
            (
                n
                for n in zone_names
                if zone_configs[n].do_reactions and zone_configs[n].do_speciation
            ),
            zone_names[0],
        )
        rt_index: int = zone_names.index(zname)
        rt_zone = RtZone(
            network,
            parameters=rt_pars[rt_index],
            do_reactions=zone_configs[zname].do_reactions,
            do_speciation=zone_configs[zname].do_speciation,
            name=zname,
        )
        print(
            f"   zone: {zname}  (reactions={rt_zone.do_reactions}, "
            f"speciation={rt_zone.do_speciation})"
        )
        ns: int = int(network.num_species)
        conc_series: torch.Tensor = torch.tensor(
            np.asarray(cfg["precip_conc"], dtype=float)[surface_idx],
            dtype=dtype,
            device=device,
        )
        spinup: int = min(5, num_steps)

        def make_d(t: int) -> RtForcing:
            """Real-chemistry, nonzero-inflow forcing for day ``t``."""
            p: torch.Tensor = zp[0, t, rt_index]
            return RtForcing(
                conc_in=conc_series[t],
                hydro_step=HydroStep(
                    state=torch.full((1,), 25.0, dtype=dtype, device=device),
                    forc_flux=torch.zeros(1, dtype=dtype, device=device),
                    lat_flux=torch.zeros(1, dtype=dtype, device=device),
                    vert_flux=torch.zeros(1, dtype=dtype, device=device),
                    vap_flux=torch.zeros(1, dtype=dtype, device=device),
                    q_in=(p + 1.0).reshape(1),
                    lat_flux_ext=torch.zeros(1, dtype=dtype, device=device),
                    vert_flux_ext=torch.zeros(1, dtype=dtype, device=device),
                ),
                hydro_forc=HydroForcing(precip=p, temp=zt[0, t, rt_index]),
                s_w=torch.full((1,), 0.5, dtype=dtype, device=device),
                z_w=torch.zeros(1, dtype=dtype, device=device),
            )

        dt: torch.Tensor = torch.as_tensor(1.0, dtype=dtype, device=device)
        # Off-solution warm start from earlier (different-forcing) days.
        c: torch.Tensor = torch.full((1, ns), 1e-3, dtype=dtype, device=device)
        for t in range(spinup):
            c = cast(torch.Tensor, rt_zone.step(c, make_d(t), dt).state)
        d_rt: list[RtForcing] = [make_d(t) for t in range(spinup, num_steps)]
        t0 = time.perf_counter()
        for _rep in range(args.reps):
            for d in d_rt:
                c = cast(torch.Tensor, rt_zone.step(c, d, dt).state)
        wall = time.perf_counter() - t0
        per_us = wall / (max(1, len(d_rt)) * args.reps) * 1e6
        print(f"   {per_us:9.1f} us/step   ({len(d_rt)} steps, {args.reps} reps)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())