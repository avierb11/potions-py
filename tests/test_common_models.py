"""Tests for the pre-built model structures in :mod:`potions.common_models`.

The structures are part of the public contract (the East River showcase selects
``HbvLateralModel`` and the calibration layout depends on flat zone order), so
the zone names/types and their parameter counts are pinned here.
"""

from __future__ import annotations

from potions.common_models import (
    HbvLateralModel,
    HbvModel,
    HbvNonlinearModel,
    LateralThreeLayerModel,
    ThreeLayerModel,
)
from potions.hydro import GroundZone, GroundZoneB, SnowZone, SurfaceZone
from potions.model import Model


class TestHbvModel:
    def test_structure(self) -> None:
        assert HbvModel.get_zone_names() == ["snow", "surface", "shallow", "deep"]
        layers = HbvModel.structure
        assert [type(layer[0]) for layer in layers] == [
            SnowZone,
            SurfaceZone,
            GroundZone,
            GroundZoneB,
        ]

    def test_repr(self) -> None:
        instance: HbvModel = HbvModel()
        text: str = repr(instance)
        assert text.startswith("HbvModel(")
        assert "\ttt=" in text
        assert "\tk2=" in text


class TestHbvLateralModel:
    def test_zone_names_and_order(self) -> None:
        assert HbvLateralModel.get_zone_names() == [
            "snow_hs",
            "snow_rp",
            "surface_hs",
            "surface_rp",
            "shallow_hs",
            "shallow_rp",
            "deep_hs",
            "deep_rp",
        ]

    def test_two_columns(self) -> None:
        assert HbvLateralModel.get_num_zones() == 8
        for layer in HbvLateralModel.structure:
            assert len(layer) == 2
            assert layer[0].name.endswith("_hs")
            assert layer[1].name.endswith("_rp")

    def test_hydro_parameter_count(self) -> None:
        expected: int = (
            sum(zone.num_parameters() for layer in HbvLateralModel.structure for zone in layer)
            + HbvLateralModel.get_num_size_parameters()
        )
        assert HbvLateralModel.get_num_hydro_parameters() == expected


class TestOtherModels:
    def test_hbv_nonlinear_same_shape_as_hbv(self) -> None:
        assert HbvNonlinearModel.get_zone_names() == HbvModel.get_zone_names()

    def test_three_layer(self) -> None:
        assert ThreeLayerModel.get_zone_names() == ["snow", "surface", "ground"]
        assert type(ThreeLayerModel.structure[-1][0]) is GroundZoneB

    def test_lateral_three_layer(self) -> None:
        assert LateralThreeLayerModel.get_zone_names() == [
            "snow_hs",
            "snow_rp",
            "surface_hs",
            "surface_rp",
            "ground_hs",
            "ground_rp",
        ]
        assert LateralThreeLayerModel.get_num_zones() == 6


class TestModelSubclassing:
    def test_all_are_models(self) -> None:
        for model_type in (
            HbvModel,
            HbvLateralModel,
            HbvNonlinearModel,
            ThreeLayerModel,
            LateralThreeLayerModel,
        ):
            assert issubclass(model_type, Model)
