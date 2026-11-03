"""Tests for :mod:`potions.objective_functions`.

The metrics are thin pandas wrappers, so the tests pin the identities that make
them useful (a perfect simulation scores 1.0 / 0.0 regardless of scale), the
direction of the bias and efficiency scores, and the percentile masking helpers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from potions import objective_functions as obj


def series(values: list[float]) -> pd.Series:
    """Build a float ``Series`` from a list."""
    return pd.Series(np.asarray(values, dtype=float))


class TestKge:
    def test_perfect_match_is_one(self) -> None:
        obs: pd.Series = series([1.0, 2.0, 3.0, 4.0, 5.0])
        assert obj.kge(obs, obs.copy()) == pytest.approx(1.0)

    def test_mean_bias_penalised(self) -> None:
        obs: pd.Series = series([1.0, 2.0, 3.0, 4.0, 5.0])
        # r=1, alpha=1 (shift preserves std), beta=4/3 -> 1 - 1/3.
        assert obj.kge(obs, obs + 1.0) == pytest.approx(1.0 - 1.0 / 3.0)

    def test_variance_bias_penalised(self) -> None:
        obs: pd.Series = series([1.0, 2.0, 3.0, 4.0, 5.0])
        assert obj.kge(obs, obs * 2.0) < 1.0


class TestNse:
    def test_perfect_match_is_one(self) -> None:
        obs: pd.Series = series([1.0, 2.0, 3.0, 4.0, 5.0])
        assert obj.nse(obs, obs.copy()) == pytest.approx(1.0)

    def test_mean_predictor_is_zero(self) -> None:
        obs: pd.Series = series([1.0, 2.0, 3.0, 4.0, 5.0])
        pred: pd.Series = series([3.0] * 5)
        assert obj.nse(obs, pred) == pytest.approx(0.0)

    def test_negative_for_poor_prediction(self) -> None:
        obs: pd.Series = series([1.0, 2.0, 3.0, 4.0, 5.0])
        assert obj.nse(obs, series([5.0, 4.0, 3.0, 2.0, 1.0])) < 0.0


class TestLogMetrics:
    def test_log_kge_perfect_match(self) -> None:
        obs: pd.Series = series([1.0, 2.0, 3.0, 4.0, 5.0])
        assert obj.log_kge(obs, obs.copy()) == pytest.approx(1.0)

    def test_log_kge_detects_scale_error(self) -> None:
        obs: pd.Series = series([1.0, 2.0, 3.0, 4.0, 5.0])
        assert obj.log_kge(obs, obs * 10.0) < 1.0

    def test_log_nse_perfect_match(self) -> None:
        obs: pd.Series = series([1.0, 2.0, 3.0, 4.0, 5.0])
        assert obj.log_nse(obs, obs.copy()) == pytest.approx(1.0)


class TestBiasAndCorrelation:
    def test_bias_zero_for_perfect(self) -> None:
        obs: pd.Series = series([1.0, 2.0, 3.0, 4.0, 5.0])
        assert obj.bias(obs, obs.copy()) == pytest.approx(0.0)

    def test_bias_positive_for_overprediction(self) -> None:
        obs: pd.Series = series([1.0, 2.0, 3.0, 4.0, 5.0])
        assert obj.bias(obs, obs * 2.0) == pytest.approx(1.0)

    def test_r_squared_perfect(self) -> None:
        obs: pd.Series = series([1.0, 2.0, 3.0, 4.0, 5.0])
        assert obj.r_squared(obs, obs.copy()) == pytest.approx(1.0)

    def test_spearman_monotone_transform(self) -> None:
        obs: pd.Series = series([1.0, 2.0, 3.0, 4.0, 5.0])
        assert obj.spearman_rho(obs, obs**3) == pytest.approx(1.0)


class TestFlowPercentiles:
    def test_low_flow_uses_bottom_fraction(self) -> None:
        obs: pd.Series = series([1.0, 2.0, 3.0, 4.0, 100.0])
        # Bottom 40% is {1, 2}; a perfect fit on those points scores 1.0.
        assert obj.objective_low_flow(obj.nse, 0.4, obs, obs.copy()) == pytest.approx(1.0)

    def test_high_flow_uses_top_fraction(self) -> None:
        obs: pd.Series = series([1.0, 2.0, 3.0, 4.0, 100.0])
        assert obj.objective_high_flow(obj.nse, 0.4, obs, obs.copy()) == pytest.approx(1.0)

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
    def test_invalid_low_percentile_raises(self, bad: float) -> None:
        obs: pd.Series = series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="Percentile"):
            obj.objective_low_flow(obj.nse, bad, obs, obs.copy())

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
    def test_invalid_high_percentile_raises(self, bad: float) -> None:
        obs: pd.Series = series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="Percentile"):
            obj.objective_high_flow(obj.nse, bad, obs, obs.copy())


class TestRegistry:
    def test_default_registry_names_and_callables(self) -> None:
        names: list[str] = [name for name, _ in obj.DEFAULT_OBJECTIVE_FUNCTIONS]
        assert names == [
            "kge",
            "log_kge",
            "nse",
            "log_nse",
            "bias",
            "spearman_rho",
            "r_squared",
        ]
        assert all(callable(fn) for _, fn in obj.DEFAULT_OBJECTIVE_FUNCTIONS)
