"""A/B check: analytic hydro derivative vs the legacy autograd path.

Runs the full East River record (batch size 1, float64, single thread)
**twice** with identical forcing / parameters and compares the results
bit-for-bit
(NaN-aware: a lane matches when the values are equal or both NaN):

1. **analytic** — the current code: ``HydrologicZone.step`` feeds the damped
   Newton driver a ``(residual, analytic_derivative)`` pair, so no
   ``torch.autograd.grad`` is ever called on the hot path;
2. **autograd** — the same code with ``potions.hydro._newton_storage_solve``
   monkey-patched to unwrap any ``(r, dr)`` pair to the bare residual
   ``r``.  The driver's first-trip mode probe then classifies ``f`` as the
   legacy automatic-differentiation path and differentiates ``r`` with
   ``torch.autograd.grad`` on every trip — the same numerics as the
   pre-Phase-2 driver (whose body is byte-identical to the current one).

The comparison covers:

* every per-day, per-zone hydro state and flux recorded by
  ``model_batched.run_hydro_batched`` (``state``, ``forc_flux``,
  ``vap_flux``, ``lat_flux``, ``vert_flux``, ``q_in``, ``lat_flux_ext``,
  ``vert_flux_ext``) — the direct output of the Newton driver, and
* the full-pipeline ``q_sim`` / ``doc_riv`` / overflow mask of
  ``model_batched.run_hydro_rt_batched`` — the end-to-end simulation.

If the analytic derivative matches autograd bit-for-bit (as pinned by
``scripts/parity_hydro_derivative.py`` at the single-evaluation level), the
Newton iterates of the two modes must be identical and every tensor above
must match lane-for-lane.  A mismatch localizes a residual difference
between the two residual-construction paths or a driver-body difference.

Usage::

    PYTHONPATH=src .venv/bin/python scripts/ab_hydro_derivative.py
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pandas import DatetimeIndex
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_full_record as bench  # noqa: E402

import potions as pt  # noqa: E402
from potions.hydro import (  # noqa: E402
    NEWTON_MAX_ITER,
    HydrologicZone,
    _newton_storage_solve as _orig_solve,
)
from potions.hydro_model import HydrologicalModel  # noqa: E402
from potions.model_batched import (  # noqa: E402
    Geometry,
    _build_hydro_zones,
    _forcing_source_tensors,
    _hydro_zone_blocks,
    _per_lane_forcing,
    run_hydro_batched,
    run_hydro_rt_batched,
)

HydroInputs = tuple[
    object,  # BatchedConfig
    list[HydrologicZone],
    torch.Tensor,  # zone_precip
    torch.Tensor,  # zone_temp
    torch.Tensor,  # zone_pet
    DatetimeIndex,  # window
    torch.Tensor,  # init_hydro
    torch.device,
    torch.dtype,
    Geometry,
]


def _nan_equal(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Elementwise equality that treats two NaNs as equal."""
    return (a == b) | (a.isnan() & b.isnan())


def _report(name: str, a: torch.Tensor, b: torch.Tensor) -> int:
    """Report the day/zone locations of any lane where ``a`` and ``b`` differ."""
    bad: torch.Tensor = ~_nan_equal(a, b)
    n: int = int(bad.sum())
    if n:
        print(f"  MISMATCH ({n}/{a.numel()}) in {name}")
        days: list[int] = bad.any(dim=1).nonzero().squeeze(-1).tolist()[:5]
        for t in days:
            zs: list[int] = bad[0, t].nonzero().squeeze(-1).tolist()
            print(f"    day {t}: zones {zs}")
            for z in zs[:4]:
                print(
                    f"      zone {z}: analytic={a[0, t, z].item():+.10e}  "
                    f"autograd={b[0, t, z].item():+.10e}"
                )
    return n


def build_hydro_inputs(batch_size: int) -> HydroInputs:
    """Rebuild the exact hydro inputs ``run_hydro_rt_batched`` would build.

    Mirrors the setup block of ``model_batched.run_hydro_rt_batched`` (that
    many identical midpoint parameter sets) so the A/B runs feed
    ``run_hydro_batched`` the same zones / geometry / forcing as the
    production call.
    """
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float64

    model_type = pt.HbvLateralModel
    names, midpoints = bench.default_parameter_vector()
    config, window = bench.build_config()
    params: NDArray[np.float64] = bench.make_params(batch_size, names, midpoints)

    num_hydro: int = model_type.get_num_hydro_parameters()
    hydro_arr: NDArray[np.float64] = params[:, :num_hydro]

    geo_model: HydrologicalModel = HydrologicalModel(structure=model_type.structure)
    geometry: Geometry = Geometry(
        lat_mat=geo_model.lat_mat.to(dtype=dtype, device=device),
        vert_mat=geo_model.vert_mat.to(dtype=dtype, device=device),
        size_mat=geo_model._get_size_mat().to(dtype=dtype, device=device),
        river_ids=geo_model.get_river_zone_ids(),
        surface_ids=geo_model.surface_zone_ids,
        zone_names=geo_model.zone_names(),
    )

    blocks, scales = _hydro_zone_blocks(model_type, hydro_arr, device, dtype)
    zones: list[HydrologicZone] = _build_hydro_zones(model_type, blocks)

    all_precip, all_temp, all_pet = _forcing_source_tensors(
        config["forc"], len(geometry.surface_ids), device, dtype
    )
    zone_precip, zone_temp, zone_pet = _per_lane_forcing(
        geometry, all_precip, all_temp, all_pet, scales, dtype
    )

    init_hydro: torch.Tensor = torch.tensor(
        [z.default_init_state() for z in geo_model.flat_model],
        dtype=dtype,
        device=device,
    )

    return (
        config,
        zones,
        zone_precip,
        zone_temp,
        zone_pet,
        window,
        init_hydro,
        device,
        dtype,
        geometry,
    )


def _ad_wrapped_solve(
    f: object,
    s_0: torch.Tensor,
    max_iter: int = NEWTON_MAX_ITER,
    f_value: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Driver wrapper that forces the legacy autograd path.

    Unwraps any ``(r, dr)`` pair returned by ``f`` down to the bare residual
    before handing it to the real driver; the driver's first-trip mode probe
    then classifies it as the autograd path (bare tensor return) and runs
    ``torch.autograd.grad`` on every trip, exactly as the pre-Phase-2 code.
    ``f_value`` is accepted for signature parity with the real driver and
    forwarded, but is inert in this mode: the unwrapped ``f`` is already a
    bare (value-only) residual, so the driver's candidate evaluations use
    ``f`` directly (``analytic`` is False and ``f_value`` is never consulted).
    """
    def _unwrapped(s: torch.Tensor) -> torch.Tensor:
        out = f(s)  # type: ignore[operator]
        return out[0] if isinstance(out, tuple) else out

    return _orig_solve(_unwrapped, s_0, max_iter, f_value=f_value)


def _set_mode(mode: str) -> None:
    """Switch ``potions.hydro._newton_storage_solve`` between the two modes."""
    import potions.hydro as hydro_mod

    if mode == "autograd":
        hydro_mod._newton_storage_solve = _ad_wrapped_solve  # type: ignore[assignment]
    else:
        hydro_mod._newton_storage_solve = _orig_solve


def run_hydro_once(
    mode: str,
    zones: list[HydrologicZone],
    zone_precip: torch.Tensor,
    zone_temp: torch.Tensor,
    zone_pet: torch.Tensor,
    window: DatetimeIndex,
    init_hydro: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    geometry: Geometry,
) -> tuple[dict[str, torch.Tensor], float]:
    """Run the batched hydro forward once in the given derivative mode."""
    _set_mode(mode)
    t0: float = time.perf_counter()
    out: dict[str, torch.Tensor] = run_hydro_batched(
        geometry,
        zones,
        zone_precip,
        zone_temp,
        zone_pet,
        window,
        init_hydro,
        device,
        dtype,
    )
    wall: float = time.perf_counter() - t0
    return out, wall


def run_full_once(
    mode: str,
    config: object,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[object, float]:
    """Run the full hydro + RT pipeline once in the given derivative mode."""
    _set_mode(mode)
    t0: float = time.perf_counter()
    out = run_hydro_rt_batched(
        bench.make_params(batch_size, *bench.default_parameter_vector()),
        config,  # type: ignore[arg-type]
        device,
        dtype,
    )
    wall: float = time.perf_counter() - t0
    return out, wall


def main() -> int:
    torch.set_num_threads(1)
    batch_size: int = 1
    print("=" * 84)
    print(
        "A/B: analytic hydro derivative vs legacy autograd path "
        f"(batch size {batch_size}, f64, 1 thread)"
    )
    print("=" * 84)

    (
        config,
        zones,
        zone_precip,
        zone_temp,
        zone_pet,
        window,
        init_hydro,
        device,
        dtype,
        geometry,
    ) = build_hydro_inputs(batch_size)
    T: int = int(zone_precip.shape[1])
    Z: int = int(zone_precip.shape[2])
    print(f"  record: {window[0]:%Y-%m-%d} .. {window[-1]:%Y-%m-%d}  ({T} days, {Z} zones)")
    print()

    # ---------------- hydro-only A/B ---------------- #
    print("[1] hydro-only: per-day zone states and fluxes")
    hydro_a, wall_a = run_hydro_once(
        "analytic", zones, zone_precip, zone_temp, zone_pet,
        window, init_hydro, device, dtype, geometry,
    )
    hydro_g, wall_g = run_hydro_once(
        "autograd", zones, zone_precip, zone_temp, zone_pet,
        window, init_hydro, device, dtype, geometry,
    )
    print(f"  wall: analytic={wall_a:.2f}s ({wall_a / T * 1000:.3f} ms/day)   "
          f"autograd={wall_g:.2f}s ({wall_g / T * 1000:.3f} ms/day)")
    total_bad: int = 0
    for name in ("state", "forc_flux", "vap_flux", "lat_flux",
                 "vert_flux", "q_in", "lat_flux_ext", "vert_flux_ext"):
        total_bad += _report(name, hydro_a[name].detach().cpu(),
                             hydro_g[name].detach().cpu())
    print(f"  hydro mismatches: {total_bad}")
    print()

    # ---------------- full-record A/B ---------------- #
    print("[2] full record: q_sim / doc_riv / overflow (analytic vs autograd driver)")
    out_a, wall_fa = run_full_once("analytic", config, batch_size, device, dtype)
    out_g, wall_fg = run_full_once("autograd", config, batch_size, device, dtype)
    print(f"  wall: analytic={wall_fa:.2f}s ({wall_fa / T * 1000:.3f} ms/day)   "
          f"autograd={wall_fg:.2f}s ({wall_fg / T * 1000:.3f} ms/day)")

    full_bad: int = 0
    full_bad += _report("q_sim", out_a.q_sim.detach().cpu(), out_g.q_sim.detach().cpu())
    full_bad += _report("doc_riv", out_a.doc_riv.detach().cpu(), out_g.doc_riv.detach().cpu())
    ov_bad: int = int((out_a.overflow_mask != out_g.overflow_mask).sum())
    full_bad += ov_bad
    if ov_bad:
        print(f"  MISMATCH ({ov_bad}) overflow_mask")

    qa: torch.Tensor = out_a.q_sim[0].detach().cpu()
    qg: torch.Tensor = out_g.q_sim[0].detach().cpu()
    da: torch.Tensor = out_a.doc_riv[0].detach().cpu()
    dg: torch.Tensor = out_g.doc_riv[0].detach().cpu()
    print(f"  q_mean: analytic={qa.mean():.6f}  autograd={qg.mean():.6f}")
    print(f"  doc_mean: analytic={da.mean():.6f}  autograd={dg.mean():.6f}")
    print(f"  full-record mismatches: {full_bad}")
    print()

    print("=" * 84)
    if total_bad + full_bad == 0:
        print("PASS: analytic and autograd paths are bit-identical across the")
        print("      full record — every per-day zone state/flux and the final")
        print("      q_sim / doc_riv match lane-for-lane.")
    else:
        print(f"FAIL: {total_bad + full_bad} mismatched lane(s) across the record")
    print("=" * 84)
    return 1 if (total_bad + full_bad) else 0


if __name__ == "__main__":
    raise SystemExit(main())
