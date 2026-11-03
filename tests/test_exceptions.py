"""Tests for the exception hierarchy in :mod:`potions.exceptions`.

Downstream calibration code catches these by type and reads the constructor-
captured fields, so the hierarchy *and* the rich-field constructors are part of
the public contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from potions.common_types import HydroForcing, RtForcing
from potions.exceptions import (
    HydrologyError,
    HydrologyNumericalError,
    OptimizationError,
    PotionsError,
    ReactiveTransportError,
    RtNumericalError,
)


class TestHierarchy:
    def test_hydrology_error_is_potions_error(self) -> None:
        assert issubclass(HydrologyError, PotionsError)

    def test_reactive_transport_error_is_potions_error(self) -> None:
        assert issubclass(ReactiveTransportError, PotionsError)

    def test_numerical_errors_subclass_their_family(self) -> None:
        assert issubclass(HydrologyNumericalError, HydrologyError)
        assert issubclass(RtNumericalError, ReactiveTransportError)

    def test_optimization_error_is_plain_exception(self) -> None:
        assert issubclass(OptimizationError, Exception)
        assert not issubclass(OptimizationError, PotionsError)


class TestHydrologyNumericalError:
    def test_captures_constructor_fields(self) -> None:
        err: HydrologyNumericalError = HydrologyNumericalError(
            type,
            object(),
            np.array([1.0, 2.0]),
            3.0,
            HydroForcing(precip=1.0),
            "boom",
        )
        assert err.model_type is type
        assert err.state == 3.0
        assert err.parameters.tolist() == [1.0, 2.0]
        assert err.hydro_forcing.precip == 1.0
        assert str(err) == "boom"


class TestRtNumericalError:
    def test_captures_constructor_fields(self) -> None:
        math_err: OptimizationError = OptimizationError("solver stalled")
        err: RtNumericalError = RtNumericalError(
            "iteration",
            type,
            object(),
            np.array([0.1]),
            np.array([1.0, 2.0]),
            RtForcing(),
            math_err,
            "rt boom",
        )
        assert err.error_type == "iteration"
        assert err.state.tolist() == [1.0, 2.0]
        assert err.math_err is math_err
        assert str(err) == "rt boom"

    @pytest.mark.parametrize("kind", ["iteration", "linear_system", "other"])
    def test_accepts_each_error_type(self, kind: str) -> None:
        err: RtNumericalError = RtNumericalError(
            kind,  # type: ignore[arg-type]
            type,
            object(),
            np.zeros(1),
            np.zeros(1),
            RtForcing(),
            OptimizationError(),
        )
        assert err.error_type == kind
