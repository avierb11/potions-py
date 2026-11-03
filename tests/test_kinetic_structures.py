"""Tests for :mod:`potions.kinetic_structures`.

Covers the two rate laws (Monod with inhibition, and transition-state theory),
the soil-water / temperature activity factors, and the parameter containers'
packing / unpacking conventions (including the log10-scaled specific surface
area).
"""

from __future__ import annotations

import torch
import pytest

from potions.kinetic_structures import (
    PARAMETERS_PER_MINERAL,
    MineralAuxParams,
    MineralParameters,
    MonodParameters,
    RiverDimensions,
    RiverParameters,
    RtParameters,
    TstParameters,
    ZoneDimensions,
    monod_inhib_rates,
    tst_rates,
)

NAN: float = float("nan")


class TestMonodInhibRates:
    def test_missing_terms_are_neutral(self) -> None:
        monod: torch.Tensor = torch.tensor([[1.0, NAN]], dtype=torch.float64)
        inhib: torch.Tensor = torch.tensor([[NAN, NAN]], dtype=torch.float64)
        chms: torch.Tensor = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
        rate: torch.Tensor = monod_inhib_rates(monod, inhib, chms)
        # c/(K+c) = 0.5 on the defined term, inhibition absent -> 0.5.
        torch.testing.assert_close(rate, torch.tensor([[0.5]], dtype=torch.float64))

    def test_inhibition_reduces_rate(self) -> None:
        monod: torch.Tensor = torch.tensor([[1.0]], dtype=torch.float64)
        inhib: torch.Tensor = torch.tensor([[1.0]], dtype=torch.float64)
        chms: torch.Tensor = torch.tensor([[1.0]], dtype=torch.float64)
        rate: torch.Tensor = monod_inhib_rates(monod, inhib, chms)
        # (1/2) * (1/2) = 0.25
        torch.testing.assert_close(rate, torch.tensor([[0.25]], dtype=torch.float64))

    def test_zero_row_disables_mineral(self) -> None:
        monod: torch.Tensor = torch.zeros(1, 2, dtype=torch.float64)
        inhib: torch.Tensor = torch.zeros(1, 2, dtype=torch.float64)
        chms: torch.Tensor = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
        rate: torch.Tensor = monod_inhib_rates(monod, inhib, chms)
        assert float(rate[0, 0]) == 0.0

    def test_empty_mineral_network(self) -> None:
        rate: torch.Tensor = monod_inhib_rates(
            torch.zeros(0, 2, dtype=torch.float64),
            torch.zeros(0, 2, dtype=torch.float64),
            torch.ones(3, 2, dtype=torch.float64),
        )
        assert rate.shape == (3, 0)

    def test_batch_lanes(self) -> None:
        monod: torch.Tensor = torch.tensor([[1.0]], dtype=torch.float64)
        inhib: torch.Tensor = torch.tensor([[NAN]], dtype=torch.float64)
        chms: torch.Tensor = torch.tensor([[1.0], [3.0]], dtype=torch.float64)
        rate: torch.Tensor = monod_inhib_rates(monod, inhib, chms)
        torch.testing.assert_close(
            rate, torch.tensor([[0.5], [0.75]], dtype=torch.float64)
        )


class TestTstRates:
    def test_undersaturated_dissolves(self) -> None:
        stoich: torch.Tensor = torch.zeros(1, 1, dtype=torch.float64)
        dep: torch.Tensor = torch.ones(1, 1, dtype=torch.float64)
        k: torch.Tensor = torch.tensor([2.0], dtype=torch.float64)
        chms: torch.Tensor = torch.tensor([[2.0]], dtype=torch.float64)
        rate: torch.Tensor = tst_rates(stoich, dep, k, chms)
        # D=2, Q=1, K=2 -> 2*(1 - 0.5) = 1.0
        torch.testing.assert_close(rate, torch.tensor([[1.0]], dtype=torch.float64))

    def test_supersaturated_precipitates(self) -> None:
        stoich: torch.Tensor = torch.ones(1, 1, dtype=torch.float64)
        dep: torch.Tensor = torch.ones(1, 1, dtype=torch.float64)
        k: torch.Tensor = torch.tensor([1.0], dtype=torch.float64)
        chms: torch.Tensor = torch.tensor([[10.0]], dtype=torch.float64)
        rate: torch.Tensor = tst_rates(stoich, dep, k, chms)
        # D=10, Q=10, K=1 -> 10*(1 - 10) = -90
        torch.testing.assert_close(rate, torch.tensor([[-90.0]], dtype=torch.float64))

    def test_empty_mineral_network(self) -> None:
        rate: torch.Tensor = tst_rates(
            torch.zeros(0, 1, dtype=torch.float64),
            torch.zeros(0, 1, dtype=torch.float64),
            torch.zeros(0, dtype=torch.float64),
            torch.ones(2, 1, dtype=torch.float64),
        )
        assert rate.shape == (2, 0)


class TestMonodParameters:
    def test_rate_uses_matrix_convention(self) -> None:
        params: MonodParameters = MonodParameters(
            monod_mat=torch.tensor([[1.0]], dtype=torch.float64),
            inhib_mat=torch.tensor([[NAN]], dtype=torch.float64),
        )
        assert params.num_minerals == 1
        assert params.num_species == 1
        rate: torch.Tensor = params.rate(torch.tensor([[1.0]], dtype=torch.float64))
        torch.testing.assert_close(rate, torch.tensor([[0.5]], dtype=torch.float64))


class TestTstParameters:
    def test_exponents_and_rates_agree(self) -> None:
        params: TstParameters = TstParameters(
            stoich=torch.zeros(1, 1, dtype=torch.float64),
            dep=torch.ones(1, 1, dtype=torch.float64),
            min_eq_const=torch.tensor([2.0], dtype=torch.float64),
        )
        chms: torch.Tensor = torch.tensor([[2.0]], dtype=torch.float64)
        q, e_q = params.calculate_solubility_product(chms)
        d, e_d = params.calculate_dependence_term(chms)
        torch.testing.assert_close(q, torch.ones(1, 1, dtype=torch.float64))
        torch.testing.assert_close(d, torch.full((1, 1), 2.0, dtype=torch.float64))
        torch.testing.assert_close(
            params.rates_from_exponents(e_d, e_q), params.rate(chms)
        )


class TestMineralParameters:
    def test_soil_water_factor_branches(self) -> None:
        params: MineralParameters = MineralParameters(
            sw_threshold=torch.tensor([0.5], dtype=torch.float64),
            sw_exp=torch.tensor([1.0], dtype=torch.float64),
            q_10=torch.tensor([2.0], dtype=torch.float64),
            ssa=torch.tensor([1.0], dtype=torch.float64),
        )
        # Above threshold: (1 - 0.75)/(1 - 0.5) = 0.5
        assert float(params.soil_water_factor(0.75)[0]) == pytest.approx(0.5)
        # Below threshold: 0.25/0.5 = 0.5
        assert float(params.soil_water_factor(0.25)[0]) == pytest.approx(0.5)

    def test_temperature_factor_q10(self) -> None:
        params: MineralParameters = MineralParameters(
            sw_threshold=torch.tensor([0.5], dtype=torch.float64),
            sw_exp=torch.tensor([1.0], dtype=torch.float64),
            q_10=torch.tensor([2.0], dtype=torch.float64),
            ssa=torch.tensor([1.0], dtype=torch.float64),
        )
        assert float(params.temperature_factor(20.0)[0]) == pytest.approx(1.0)
        assert float(params.temperature_factor(30.0)[0]) == pytest.approx(2.0)

    def test_factor_is_product(self) -> None:
        params: MineralParameters = MineralParameters(
            sw_threshold=torch.tensor([0.5], dtype=torch.float64),
            sw_exp=torch.tensor([1.0], dtype=torch.float64),
            q_10=torch.tensor([2.0], dtype=torch.float64),
            ssa=torch.tensor([1.0], dtype=torch.float64),
        )
        combined: torch.Tensor = params.factor(0.75, 30.0)
        expected: torch.Tensor = params.soil_water_factor(0.75) * params.temperature_factor(30.0)
        torch.testing.assert_close(combined, expected)

    def test_batched_factor_shape(self) -> None:
        params: MineralParameters = MineralParameters(
            sw_threshold=torch.full((3,), 0.5, dtype=torch.float64),
            sw_exp=torch.ones(3, dtype=torch.float64),
            q_10=torch.full((3,), 2.0, dtype=torch.float64),
            ssa=torch.ones(3, dtype=torch.float64),
        )
        s_w: torch.Tensor = torch.tensor([0.25, 0.75], dtype=torch.float64)
        out: torch.Tensor = params.factor(s_w, 20.0)
        assert out.shape == (2, 3)

    def test_from_array_natural_scales_false_roundtrips(self) -> None:
        original: MineralParameters = MineralParameters(
            sw_threshold=torch.tensor([0.3, 0.6], dtype=torch.float64),
            sw_exp=torch.tensor([1.0, 2.0], dtype=torch.float64),
            q_10=torch.tensor([2.0, 3.0], dtype=torch.float64),
            ssa=torch.tensor([4.0, 5.0], dtype=torch.float64),
        )
        packed: torch.Tensor = original.to_array()
        assert packed.shape == (PARAMETERS_PER_MINERAL * 2,)
        restored: MineralParameters = MineralParameters.from_array(
            packed, natural_scales=False
        )
        for field in ("sw_threshold", "sw_exp", "q_10", "ssa"):
            torch.testing.assert_close(
                getattr(restored, field), getattr(original, field)
            )

    def test_from_array_wrong_size_raises(self) -> None:
        with pytest.raises(ValueError):
            MineralParameters.from_array(torch.zeros(5, dtype=torch.float64))

    def test_from_batched_array_applies_ssa_exponent(self) -> None:
        arr: torch.Tensor = torch.tensor(
            [[0.5, 1.0, 2.0, 3.0], [0.6, 1.5, 3.0, 4.0]], dtype=torch.float64
        )
        params: MineralParameters = MineralParameters.from_batched_array(arr)
        assert params.sw_threshold.shape == (2, 1)
        torch.testing.assert_close(
            params.ssa, torch.tensor([[1000.0], [10000.0]], dtype=torch.float64)
        )

    def test_from_mineral_parameters(self) -> None:
        params: MineralParameters = MineralParameters.from_mineral_parameters(
            [
                MineralAuxParams(sw_threshold=0.1, sw_exp=1.0, q_10=2.0, ssa=3.0),
                MineralAuxParams(sw_threshold=0.2, sw_exp=1.5, q_10=4.0, ssa=5.0),
            ]
        )
        torch.testing.assert_close(
            params.sw_threshold, torch.tensor([0.1, 0.2], dtype=torch.float64)
        )
        torch.testing.assert_close(
            params.ssa, torch.tensor([3.0, 5.0], dtype=torch.float64)
        )


class TestMineralAuxParams:
    def test_array_roundtrip(self) -> None:
        aux: MineralAuxParams = MineralAuxParams(0.1, 1.5, 2.0, 3.0)
        arr: torch.Tensor = aux.to_array()
        assert arr.shape == (PARAMETERS_PER_MINERAL,)
        restored: MineralAuxParams = MineralAuxParams.from_array(arr)
        assert restored == aux

    def test_from_array_wrong_size_raises(self) -> None:
        with pytest.raises(ValueError):
            MineralAuxParams.from_array(torch.zeros(3, dtype=torch.float64))


class TestZoneDimensions:
    def test_max_water_volume(self) -> None:
        dims: ZoneDimensions = ZoneDimensions(porosity=0.4, depth=100.0, passive_water_storage=5.0)
        assert dims.max_water_volume == pytest.approx(0.4 * 100.0 - 5.0)

    def test_array_roundtrip(self) -> None:
        dims: ZoneDimensions = ZoneDimensions(porosity=0.4, depth=100.0, passive_water_storage=5.0)
        restored: ZoneDimensions = ZoneDimensions.from_array(dims.to_array())
        assert restored == dims

    def test_from_array_wrong_size_raises(self) -> None:
        with pytest.raises(ValueError):
            ZoneDimensions.from_array(torch.zeros(2, dtype=torch.float64))


class TestRtParameters:
    def test_from_batched_array_dimensions_only(self) -> None:
        arr: torch.Tensor = torch.tensor(
            [[0.4, 100.0, 5.0], [0.5, 200.0, 10.0]], dtype=torch.float64
        )
        params: RtParameters = RtParameters.from_batched_array(arr)
        assert params.mineral_params is None
        torch.testing.assert_close(
            params.dimensions.depth, torch.tensor([100.0, 200.0], dtype=torch.float64)
        )

    def test_from_batched_array_with_minerals(self) -> None:
        # one mineral -> 3 + 4 = 7 columns
        arr: torch.Tensor = torch.tensor(
            [[0.4, 100.0, 5.0, 0.5, 1.0, 2.0, 1.0]], dtype=torch.float64
        )
        params: RtParameters = RtParameters.from_batched_array(arr)
        assert params.mineral_params is not None
        assert params.mineral_params.num_minerals == 1
        torch.testing.assert_close(
            params.mineral_params.ssa, torch.tensor([[10.0]], dtype=torch.float64)
        )

    def test_from_batched_array_bad_shape_raises(self) -> None:
        with pytest.raises(ValueError):
            RtParameters.from_batched_array(torch.zeros(2, 2, dtype=torch.float64))

    def test_to_array_dimensions_only(self) -> None:
        params: RtParameters = RtParameters(
            dimensions=ZoneDimensions(porosity=0.4, depth=100.0, passive_water_storage=5.0)
        )
        torch.testing.assert_close(
            params.to_array(), torch.tensor([0.4, 100.0, 5.0], dtype=torch.float64)
        )


class TestRiverDimensions:
    def test_num_parameters(self) -> None:
        assert RiverDimensions.num_parameters() == 2

    def test_array_roundtrip(self) -> None:
        dims: RiverDimensions = RiverDimensions(bed_depth=12.0, passive_water_storage=3.0)
        restored: RiverDimensions = RiverDimensions.from_array(dims.to_array())
        assert restored == dims

    def test_from_array_wrong_size_raises(self) -> None:
        with pytest.raises(ValueError):
            RiverDimensions.from_array(torch.zeros(3, dtype=torch.float64))

    def test_river_parameters_without_minerals(self) -> None:
        params: RiverParameters = RiverParameters(
            dimensions=RiverDimensions(bed_depth=12.0, passive_water_storage=3.0)
        )
        restored: RiverParameters = RiverParameters.from_array(params.to_array())
        assert restored.mineral_params is None
        assert restored.dimensions == params.dimensions
