"""Smoke tests for :mod:`potions.model_batched`.

A full ``run_hydro_rt_batched`` call is driven on a tiny hand-built East River
config (2 surface zones, 8 zones total, short forcing window) so the batched
runner is exercised end-to-end without any external data files.  The checks are
structural: output shapes, finite values, objective registry keys, and that
duplicating a parameter lane reproduces the same lane (the batched path has no
cross-lane coupling).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from pandas import DataFrame, DatetimeIndex, Series

import potions as pt
from potions.common_types import ForcingData, RtZoneConfiguration
from potions.model_batched import BatchedConfig, BatchedRunOutput, _step_lengths
from potions.reactive_transport.common_networks import (
    get_simple_carbon_adsorption_network,
)

F64: torch.dtype = torch.float64
F32: torch.dtype = torch.float32
CPU: torch.device = torch.device("cpu")

#: Forcing window length (kept short: the runner is Python-orchestration bound).
NUM_STEPS: int = 6


def _reacting(zone_name: str) -> bool:
    """Whether a zone reacts speciates in this smoke config (deep/shallow soil)."""
    return zone_name.startswith("shallow") or zone_name.startswith("deep")


def build_config_and_params() -> tuple[BatchedConfig, np.ndarray]:
    """Build a minimal valid East River config plus a ``(2, width)`` parameter array."""
    network = get_simple_carbon_adsorption_network()
    model_type = pt.HbvLateralModel
    dates: DatetimeIndex = DatetimeIndex([f"2021-09-0{i}" for i in range(1, NUM_STEPS + 1)])
    forcing: ForcingData = ForcingData(
        precip=Series([1.0, 3.0, 0.0, 5.0, 2.0, 4.0], index=dates),
        temp=Series([10.0, 12.0, 8.0, 15.0, 11.0, 9.0], index=dates),
        pet=Series([0.5, 0.8, 0.2, 1.0, 0.6, 0.4], index=dates),
    )
    zone_configs: dict[str, RtZoneConfiguration] = {
        name: RtZoneConfiguration(
            do_reactions=_reacting(name), do_speciation=_reacting(name)
        )
        for name in model_type.get_zone_names()
    }

    num_hydro: int = model_type.get_num_hydro_parameters()
    num_rt: int = sum(
        3 + (4 if zone_configs[name].do_reactions else 0)
        for name in model_type.get_zone_names()
    )
    num_react: int = sum(1 for c in zone_configs.values() if c.do_reactions)
    num_spec_zones: int = sum(1 for c in zone_configs.values() if c.do_speciation)
    num_init: int = network.num_minerals * num_react + num_spec_zones * int(
        network.has_exchange
    )

    params: np.ndarray = np.zeros((2, num_hydro + num_rt + num_init), dtype=np.float64)
    params[:, :num_hydro] = 0.4
    lower: int = num_hydro
    for name in model_type.get_zone_names():
        if zone_configs[name].do_reactions:
            params[:, lower : lower + 3] = np.array([0.4, 300.0, 5.0])
            params[:, lower + 3 : lower + 7] = np.array([0.3, 1.0, 2.0, 0.0])
            lower += 7
        else:
            params[:, lower : lower + 3] = np.array([0.4, 300.0, 5.0])
            lower += 3
    params[:, num_hydro + num_rt :] = 0.0

    num_surface: int = len(model_type.structure[0])
    precip_conc: np.ndarray = np.zeros(
        (num_surface, NUM_STEPS, network.num_species)
    )
    precip_conc[:, :, 0] = 1e-3
    q_meas: Series = Series([0.5, 1.5, 0.2, 2.5, 1.0, 2.0], index=dates)
    conc_df: DataFrame = DataFrame(
        {"DOC": [0.05, 0.15, 0.02, 0.25, 0.1, 0.2]}, index=dates
    )
    config: BatchedConfig = BatchedConfig(
        network=network,
        zone_configs=zone_configs,
        forc=forcing,
        q_meas=q_meas,
        conc_df=conc_df,
        precip_conc=precip_conc,
        model_type=model_type,
    )
    return config, params


@pytest.fixture(scope="module")
def run_output() -> tuple[BatchedRunOutput, BatchedConfig, np.ndarray]:
    """One end-to-end batched run reused by the structural assertions."""
    config, params = build_config_and_params()
    out: BatchedRunOutput = pt.run_hydro_rt_batched(params, config, CPU, F64)
    return out, config, params


class TestRunHydroRtBatched:
    def test_output_shapes(self, run_output) -> None:
        out, _config, _params = run_output
        assert out.q_sim.shape == (2, NUM_STEPS)
        assert out.doc_riv.shape == (2, NUM_STEPS)
        assert out.overflow_mask.shape == (2,)

    def test_outputs_finite(self, run_output) -> None:
        out, _config, _params = run_output
        assert torch.isfinite(out.q_sim).all()
        assert torch.isfinite(out.doc_riv).all()

    def test_objective_registry_keys(self, run_output) -> None:
        out, _config, _params = run_output
        for hydro_obj, rt_obj in zip(
            out.hydro_objectives, out.rt_doc_objectives, strict=True
        ):
            assert list(hydro_obj.keys()) == [
                "kge",
                "log_kge",
                "nse",
                "log_nse",
                "bias",
                "spearman_rho",
                "r_squared",
            ]
            assert list(rt_obj.keys()) == list(hydro_obj.keys())

    def test_identical_lanes_are_identical(self, run_output) -> None:
        out, _config, _params = run_output
        torch.testing.assert_close(out.q_sim[0], out.q_sim[1])
        torch.testing.assert_close(out.doc_riv[0], out.doc_riv[1])

    def test_parameter_width_mismatch_raises(self) -> None:
        config, params = build_config_and_params()
        with pytest.raises(ValueError, match="does not match the model layout"):
            pt.run_hydro_rt_batched(params[:, :-1], config, CPU, F64)

    def test_k1_matches_k2_lane(self) -> None:
        config, params = build_config_and_params()
        out1: BatchedRunOutput = pt.run_hydro_rt_batched(params[:1], config, CPU, F64)
        out2: BatchedRunOutput = pt.run_hydro_rt_batched(params, config, CPU, F64)
        torch.testing.assert_close(out1.q_sim[0], out2.q_sim[0])
        torch.testing.assert_close(out1.doc_riv[0], out2.doc_riv[0])

    def test_float32_compute_stays_close_to_float64(self) -> None:
        config, params = build_config_and_params()
        out64: BatchedRunOutput = pt.run_hydro_rt_batched(params, config, CPU, F64)
        out32: BatchedRunOutput = pt.run_hydro_rt_batched(
            params, config, CPU, F64, compute_dtype=F32
        )
        torch.testing.assert_close(
            out32.q_sim, out64.q_sim, atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(
            out32.doc_riv, out64.doc_riv, atol=1e-6, rtol=1e-6
        )
        torch.testing.assert_close(out32.overflow_mask, out64.overflow_mask)
        assert ['kge', 'log_kge', 'nse', 'log_nse', 'bias', 'spearman_rho', 'r_squared'] == list(
            out32.rt_doc_objectives[0].keys()
        )


class TestStepLengths:
    def test_hydro_rule_last_step_repeats(self) -> None:
        dates: DatetimeIndex = DatetimeIndex(["2021-01-01", "2021-01-02", "2021-01-04"])
        dt: list[float] = _step_lengths(dates, rt_style=False)
        assert dt == [1.0, 2.0, 2.0]

    def test_rt_rule_leads_with_one(self) -> None:
        dates: DatetimeIndex = DatetimeIndex(["2021-01-01", "2021-01-02", "2021-01-04"])
        dt: list[float] = _step_lengths(dates, rt_style=True)
        assert dt[0] == 1.0
        assert dt[1:] == [1.0, 2.0]
