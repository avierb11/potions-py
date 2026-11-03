"""Tests for the hydrology zones and the damped AD-Newton storage solver in :mod:`potions.hydro`.

The zone flux laws are simple algebraic expressions, so the tests assert the
branch conditions directly (snow above/below the threshold temperature, ET
limited by ``fc*lp``, baseflow thresholding, ...).  The stepping tests check the
implicit-Euler contract: the returned storage must satisfy the residual to
within tolerance, storage is clamped non-negative, and batched lanes stay
independent.  The solver tests include a regression for the kinked
``k*s**alpha`` baseflow law, on which the undamped notebook Newton 2-cycles.
"""

from __future__ import annotations

import torch
import pytest

from potions import hydro
from potions.common_types import HydroForcing, HydroStep
from potions.hydro import (
    GroundZone,
    GroundZoneB,
    SnowZone,
    SurfaceZone,
)


def forcing(
    precip: float = 0.0,
    temp: float = 0.0,
    pet: float = 0.0,
    q_in: float = 0.0,
) -> HydroForcing:
    """Build a scalar :class:`HydroForcing` for the algebraic flux tests."""
    return HydroForcing(precip=precip, temp=temp, pet=pet, q_in=q_in)


class TestNewtonStorageSolve:
    def test_linear_converges_to_root(self) -> None:
        s: torch.Tensor = hydro._newton_storage_solve(
            lambda x: x - 7.0, torch.tensor([1.0], dtype=torch.float64)
        )
        torch.testing.assert_close(s, torch.tensor([7.0], dtype=torch.float64), atol=1e-6, rtol=1e-6)

    def test_quadratic_converges_in_few_trips(self) -> None:
        # Newton is quadratic: x0=3 -> 2.1667 -> 2.0064 -> 2.0000104 ->
        # 2.0000000013, with the residual below the f64 bar on the 5th trip.
        # The smooth quadratic accepts every full (alpha=1) step, so each
        # trip costs one residual eval plus one backtracking acceptance eval.
        # Count only the trips: the trip's residual eval is the sole
        # gradient-carrying call (the acceptance evals run under no_grad).
        trips: list[int] = [0]
        evals: list[int] = [0]

        def counting(x: torch.Tensor) -> torch.Tensor:
            evals[0] += 1
            if bool(x.requires_grad):
                trips[0] += 1
            return x * x - 4.0

        s: torch.Tensor = hydro._newton_storage_solve(
            counting, torch.tensor([3.0], dtype=torch.float64)
        )
        torch.testing.assert_close(s, torch.tensor([2.0], dtype=torch.float64), atol=1e-6, rtol=1e-6)
        assert trips[0] <= 5
        assert evals[0] <= 2 * trips[0]

    def test_kinked_residual_does_not_limit_cycle(self) -> None:
        # Regression: the GroundZoneB baseflow law ``k * s**alpha`` with
        # ``0 < alpha < 1`` is kinked at ``s = 0`` (infinite analytic
        # one-sided slope, zero autograd subgradient on the clamped
        # boundary).  The undamped notebook Newton 2-cycles on exactly this
        # residual (s = 1.0344 <-> 0, trips 12-19 of the failing run) and
        # exhausts the budget; backtracking must damp it to the root.
        k: float = 10.0 ** 0.4
        s_0: float = 1.0343889801223856  # s0 + dt*q_in for the failing step

        def f(x: torch.Tensor) -> torch.Tensor:
            return s_0 - x - k * x.clamp_min(0.0) ** 0.4

        s: torch.Tensor = hydro._newton_storage_solve(
            f, torch.tensor([0.6344], dtype=torch.float64)
        )
        # The root sits just below 0.1 (f(0.1) ~= -0.066, f(0.08) ~= +0.040),
        # and the carried-out state must meet the f64 absolute bar.
        assert 0.05 < float(s) < 0.15
        assert float(f(s).abs()) <= hydro.NEWTON_ATOL_F64

    def test_lanes_are_independent(self) -> None:
        # One shared residual, a different root per lane: each lane must
        # converge to its own target without cross-talk.
        targets: torch.Tensor = torch.tensor([1.0, 7.0, 42.0], dtype=torch.float64)
        s: torch.Tensor = hydro._newton_storage_solve(
            lambda x: x - targets, torch.zeros(3, dtype=torch.float64)
        )
        torch.testing.assert_close(s, targets, atol=1e-9, rtol=1e-12)

    def test_tolerance_bars(self) -> None:
        atol: float
        rtol: float
        atol, rtol = hydro._solver_tols(torch.float64)
        assert (atol, rtol) == (hydro.NEWTON_ATOL_F64, 0.0)
        atol, rtol = hydro._solver_tols(torch.float32)
        # f32 clamps the absolute bar above the ulp floor of mm-scale storage.
        assert atol == max(1e-6, 64.0 * float(torch.finfo(torch.float32).eps) * hydro._STORAGE_MM_SCALE)
        assert rtol == hydro.NEWTON_RTOL_F32

    def test_f32_storage_scale_converges(self) -> None:
        # An implicit-Euler-style linear residual at storage scale in f32:
        # f(x) = (s0 - x) + dt*(qin - k*x) has the analytic root 30/1.1.
        s0: torch.Tensor = torch.tensor([25.0], dtype=torch.float32)
        atol: float
        rtol: float
        atol, rtol = hydro._solver_tols(torch.float32)

        def f(x: torch.Tensor) -> torch.Tensor:
            return (s0 - x) + (5.0 - 0.1 * x)

        s: torch.Tensor = hydro._newton_storage_solve(f, torch.tensor([10.0], dtype=torch.float32))
        torch.testing.assert_close(s, torch.tensor([30.0 / 1.1], dtype=torch.float32), atol=1e-4, rtol=1e-6)
        # The carried-out state satisfies the dtype-aware two-term bar.
        assert float(f(s).abs()) <= atol + rtol * float(s)

    def test_nonconverging_raises(self) -> None:
        # A constant residual has a zero derivative; the floored step keeps
        # the iterate stuck, so the trip budget is exhausted and the driver
        # raises (no silent last-iterate return).
        with pytest.raises(hydro.ScalarRootFindingError):
            hydro._newton_storage_solve(
                lambda x: x * 0.0 + 1.0, torch.tensor([1.0], dtype=torch.float64)
            )

    def test_nan_residual_raises(self) -> None:
        # NaN never satisfies the convergence bar (NaN comparisons are False),
        # so a NaN-producing residual also exhausts the budget and raises.
        with pytest.raises(hydro.ScalarRootFindingError):
            hydro._newton_storage_solve(
                lambda x: x * 0.0 + float("nan"), torch.tensor([1.0], dtype=torch.float64)
            )


class TestSnowZone:
    def test_snowfall_below_threshold(self) -> None:
        zone: SnowZone = SnowZone(tt=0.0, fmax=2.0)
        flux: torch.Tensor = zone.forc_flux(0.0, forcing(precip=5.0, temp=-3.0))
        assert float(flux) == 5.0

    def test_no_snowfall_above_threshold(self) -> None:
        zone: SnowZone = SnowZone(tt=0.0, fmax=2.0)
        flux: torch.Tensor = zone.forc_flux(0.0, forcing(precip=5.0, temp=3.0))
        assert float(flux) == 0.0

    def test_melt_capped_by_storage(self) -> None:
        zone: SnowZone = SnowZone(tt=0.0, fmax=2.0)
        # 2.0 mm/K * 3 K = 6 mm potential melt, but only 4 mm is stored.
        flux: torch.Tensor = zone.vert_flux(4.0, forcing(temp=3.0))
        assert float(flux) == 4.0

    def test_melt_uncapped(self) -> None:
        zone: SnowZone = SnowZone(tt=0.0, fmax=2.0)
        flux: torch.Tensor = zone.vert_flux(100.0, forcing(temp=3.0))
        assert float(flux) == 6.0

    def test_ext_flux_adds_rain_on_snow(self) -> None:
        zone: SnowZone = SnowZone(tt=0.0, fmax=2.0)
        flux: torch.Tensor = zone.vert_flux_ext(100.0, forcing(precip=1.5, temp=3.0))
        # melt 6.0 + rain 1.5
        assert float(flux) == pytest.approx(7.5)

    def test_from_array_and_metadata(self) -> None:
        zone: SnowZone = SnowZone.from_array(torch.tensor([-1.0, 3.0], dtype=torch.float64))
        assert zone.tt == -1.0
        assert zone.fmax == 3.0
        assert SnowZone.parameter_names() == ["tt", "fmax"]
        assert SnowZone.num_parameters() == 2
        assert SnowZone.base_name() == "snow"

    def test_from_array_wrong_size_raises(self) -> None:
        with pytest.raises(ValueError):
            SnowZone.from_array(torch.zeros(3, dtype=torch.float64))

    def test_step_conserves_and_solves_residual(self) -> None:
        zone: SnowZone = SnowZone(tt=0.0, fmax=1.0)
        # Cold: all precipitation accumulates, nothing leaves.
        d: HydroForcing = forcing(precip=10.0, temp=-5.0)
        res: HydroStep = zone.step(2.0, d, dt=1.0)
        assert float(res.state) == pytest.approx(12.0)
        assert float(res.vert_flux) == 0.0
        assert float(zone.residual(res.state, 2.0, d, 1.0)) == pytest.approx(0.0, abs=1e-6)


class TestSurfaceZone:
    def test_vap_limited_by_fc_lp(self) -> None:
        zone: SurfaceZone = SurfaceZone(fc=100.0, lp=0.5)
        # s / (fc*lp) = 25/50 = 0.5, so ET = 0.5 * pet.
        assert float(zone.vap_flux(25.0, forcing(pet=4.0))) == pytest.approx(2.0)
        # Saturated: ET = pet.
        assert float(zone.vap_flux(1000.0, forcing(pet=4.0))) == pytest.approx(4.0)

    def test_lat_flux_threshold(self) -> None:
        zone: SurfaceZone = SurfaceZone(k0=0.5, thr=10.0)
        assert float(zone.lat_flux(5.0, forcing())) == 0.0
        assert float(zone.lat_flux(30.0, forcing())) == pytest.approx(10.0)

    def test_vert_flux_uses_q_in(self) -> None:
        zone: SurfaceZone = SurfaceZone(fc=100.0, beta=1.0)
        # (s/fc)**1 = 0.5, so percolation = 0.5 * q_in.
        assert float(zone.vert_flux(50.0, forcing(q_in=8.0))) == pytest.approx(4.0)

    def test_mass_balance(self) -> None:
        zone: SurfaceZone = SurfaceZone(fc=100.0, lp=0.5, beta=1.0, k0=0.0, thr=0.0)
        d: HydroForcing = forcing(pet=2.0, q_in=8.0)
        expected: float = 8.0 - 2.0 - 4.0  # q_in - vap - vert (lat flux k0=0)
        assert float(zone.mass_balance(50.0, d)) == pytest.approx(expected)

    def test_from_array_roundtrip(self) -> None:
        arr: torch.Tensor = torch.tensor([200.0, 0.4, 2.0, 0.3, 5.0], dtype=torch.float64)
        zone: SurfaceZone = SurfaceZone.from_array(arr)
        assert (zone.fc, zone.lp, zone.beta, zone.k0, zone.thr) == (200.0, 0.4, 2.0, 0.3, 5.0)

    def test_step_nonnegative_storage(self) -> None:
        zone: SurfaceZone = SurfaceZone(fc=100.0, lp=0.5)
        # Huge PET drain relative to storage; the root is clamped to zero.
        res: HydroStep = zone.step(0.1, forcing(pet=100.0), dt=1.0)
        assert float(res.state) >= 0.0


class TestGroundZone:
    def test_baseflow_below_epsilon_is_zero(self) -> None:
        zone: GroundZone = GroundZone(k=0.1, alpha=1.0)
        assert float(zone.lat_flux(0.0, forcing())) == 0.0

    def test_baseflow_power_law(self) -> None:
        zone: GroundZone = GroundZone(k=0.1, alpha=2.0)
        assert float(zone.lat_flux(3.0, forcing())) == pytest.approx(0.9)

    def test_percolation_capped(self) -> None:
        zone: GroundZone = GroundZone(perc=2.0)
        assert float(zone.vert_flux(10.0, forcing())) == 2.0
        assert float(zone.vert_flux(1.0, forcing())) == 1.0

    def test_mass_balance(self) -> None:
        zone: GroundZone = GroundZone(k=0.1, alpha=1.0, perc=1.0)
        d: HydroForcing = forcing(q_in=5.0)
        assert float(zone.mass_balance(3.0, d)) == pytest.approx(5.0 - 1.0 - 0.3)

    def test_from_array_natural_scales(self) -> None:
        arr: torch.Tensor = torch.tensor([-2.0, 1.5, 3.0], dtype=torch.float64)
        zone: GroundZone = GroundZone.from_array(arr, natural_scales=True)
        assert zone.k == pytest.approx(10.0**-2.0)
        assert zone.alpha == 1.5
        assert zone.perc == 3.0

    def test_step_residual_is_satisfied(self) -> None:
        zone: GroundZone = GroundZone(k=0.05, alpha=1.0, perc=1.0)
        d: HydroForcing = forcing(q_in=4.0)
        res: HydroStep = zone.step(10.0, d, dt=1.0)
        assert float(zone.residual(res.state, 10.0, d, 1.0)) == pytest.approx(0.0, abs=1e-6)
        assert float(res.state) >= 0.0


class TestGroundZoneB:
    def test_vert_flux_is_zero(self) -> None:
        zone: GroundZoneB = GroundZoneB(k=0.1, alpha=1.0)
        assert float(zone.vert_flux(100.0, forcing())) == 0.0

    def test_from_array(self) -> None:
        arr: torch.Tensor = torch.tensor([-3.0, 2.0], dtype=torch.float64)
        zone: GroundZoneB = GroundZoneB.from_array(arr)
        assert zone.k == pytest.approx(1e-3)
        assert zone.alpha == 2.0
        assert GroundZoneB.num_parameters() == 2


class TestBatchedZones:
    def test_surface_zone_broadcast(self) -> None:
        zone: SurfaceZone = SurfaceZone(fc=100.0, lp=0.5, beta=1.0, k0=0.0, thr=0.0)
        s: torch.Tensor = torch.tensor([25.0, 50.0, 100.0], dtype=torch.float64)
        vap: torch.Tensor = zone.vap_flux(s, forcing(pet=4.0))
        # s/(fc*lp) = 0.5, 1.0, 2.0 -> capped at 1.0
        torch.testing.assert_close(
            vap, torch.tensor([2.0, 4.0, 4.0], dtype=torch.float64)
        )

    def test_step_accepts_tensor_state(self) -> None:
        zone: SnowZone = SnowZone(tt=0.0, fmax=1.0)
        s_0: torch.Tensor = torch.zeros(4, dtype=torch.float64)
        d: HydroForcing = forcing(precip=1.0, temp=-1.0)
        res: HydroStep = zone.step(s_0, d, dt=1.0)
        assert torch.as_tensor(res.state).shape == (4,)
        torch.testing.assert_close(
            res.state, torch.ones(4, dtype=torch.float64), atol=1e-6, rtol=1e-6
        )

    def test_analytic_candidate_evals_skip_derivative(self) -> None:
        # B1: in the analytic (pair) mode the driver's Armijo *candidate*
        # evaluations run the residual-only ``f_value`` companion, so a single
        # step makes strictly more ``_hydro_residual`` calls than
        # ``_hydro_residual_deriv`` calls — each trip's derivative is computed
        # once, but candidates re-evaluate the residual without the derivative.
        # Without B1 every candidate would call the full ``(r, dr)`` ``f`` and
        # the two counts would be equal.
        orig_residual = hydro._hydro_residual
        orig_deriv = hydro._hydro_residual_deriv
        counts: dict[str, int] = {"residual": 0, "deriv": 0}

        def counting_residual(
            zone: hydro.HydrologicZone,
            s: hydro.Scalar,
            s_0: hydro.Scalar,
            d: HydroForcing,
            dt: hydro.Scalar,
        ) -> torch.Tensor:
            counts["residual"] += 1
            return orig_residual(zone, s, s_0, d, dt)

        def counting_deriv(
            zone: hydro.HydrologicZone,
            s: hydro.Scalar,
            s_0: hydro.Scalar,
            d: HydroForcing,
            dt: hydro.Scalar,
        ) -> torch.Tensor:
            counts["deriv"] += 1
            return orig_deriv(zone, s, s_0, d, dt)

        hydro._hydro_residual = counting_residual
        hydro._hydro_residual_deriv = counting_deriv
        try:
            zone: SurfaceZone = SurfaceZone(fc=100.0, lp=0.5, beta=1.0, k0=0.1, thr=10.0)
            zone.step(50.0, forcing(precip=5.0, pet=3.0, q_in=1.0), dt=1.0)
        finally:
            hydro._hydro_residual = orig_residual
            hydro._hydro_residual_deriv = orig_deriv
        # >= 1 trip (residual + derivative) and >= 1 candidate (residual only).
        assert counts["deriv"] >= 1
        assert counts["residual"] >= counts["deriv"] + 1

    def test_per_lane_line_search_matches_standalone_lanes(self) -> None:
        # B2: the Armijo step-halving is per lane, so a batched step with mixed
        # per-lane states / forcings must be bit-identical (lane-for-lane) to
        # each lane stepped on its own as a one-lane batch.  A shared scalar
        # ``alpha`` would let one lane's acceptance gate the whole batch's
        # halving and perturb the other lanes' candidates, breaking this.
        torch.manual_seed(0)
        zone: SurfaceZone = SurfaceZone(fc=100.0, lp=0.5, beta=1.0, k0=0.1, thr=10.0)
        # Lanes at / near the flux-law kinks (s = thr = 10 quickflow kink,
        # s = fc*lp = 50 ET-saturation kink) plus a well-conditioned lane, with
        # distinct per-lane forcings so each lane accepts at its own pace.
        s0: torch.Tensor = torch.tensor([5.0, 10.0, 50.0, 80.0], dtype=torch.float64)
        d: HydroForcing = HydroForcing(
            precip=torch.tensor([5.0, 2.0, 8.0, 1.0], dtype=torch.float64),
            temp=torch.zeros(4, dtype=torch.float64),
            pet=torch.tensor([3.0, 1.0, 4.0, 2.0], dtype=torch.float64),
            q_in=torch.tensor([1.0, 2.0, 0.5, 3.0], dtype=torch.float64),
        )
        batched: HydroStep = zone.step(s0, d)
        refs: list[torch.Tensor] = []
        for i in range(s0.shape[0]):
            lane: HydroForcing = HydroForcing(
                precip=d.precip[i : i + 1],
                temp=d.temp[i : i + 1],
                pet=d.pet[i : i + 1],
                q_in=d.q_in[i : i + 1],
            )
            refs.append(zone.step(s0[i : i + 1], lane).state)
        expected: torch.Tensor = torch.cat(refs)
        assert torch.equal(batched.state, expected)


class TestBaseZone:
    def test_inert_fluxes_are_zero(self) -> None:
        zone: hydro.HydrologicZone = hydro.HydrologicZone()
        zone.name = "inert"
        d: HydroForcing = forcing(precip=1.0, temp=1.0, pet=1.0, q_in=1.0)
        assert float(zone.forc_flux(1.0, d)) == 0.0
        assert float(zone.vap_flux(1.0, d)) == 0.0
        assert float(zone.lat_flux(1.0, d)) == 0.0
        assert float(zone.vert_flux(1.0, d)) == 0.0
        # The inert zone has no fluxes of its own but still passes ``q_in``
        # through its mass balance, so the storage gains ``dt * q_in``.
        res: HydroStep = zone.step(3.0, d, dt=1.0)
        assert float(res.state) == pytest.approx(3.0 + float(d.q_in))

    def test_introspection_defaults(self) -> None:
        assert hydro.HydrologicZone.parameter_names() == []
        assert hydro.HydrologicZone.num_parameters() == 0
        assert hydro.HydrologicZone.base_name() == "unnamed"
        assert hydro.HydrologicZone.default_init_state() == 0.0
        assert hydro.HydrologicZone().param_list() == []

    def test_from_dict_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError):
            hydro.HydrologicZone.from_dict({})
