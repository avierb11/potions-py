"""Tests for the plain record types in :mod:`potions.common_types`."""

from __future__ import annotations

import torch
import pytest

from potions.common_types import (
    ChemicalState,
    ForcingData,
    HydroForcing,
    HydroStep,
    LapseRateParameters,
    MiscData,
    RtForcing,
    ZERO_CONC,
)


def step(
    lat: float = 0.0,
    vert: float = 0.0,
    lat_ext: float = 0.0,
    vert_ext: float = 0.0,
    vap: float = 0.0,
) -> HydroStep:
    """Build a :class:`HydroStep` with the export fluxes under test."""
    return HydroStep(
        state=1.0,
        forc_flux=0.0,
        lat_flux=lat,
        vert_flux=vert,
        vap_flux=vap,
        q_in=0.0,
        lat_flux_ext=lat_ext,
        vert_flux_ext=vert_ext,
    )


class TestHydroStep:
    def test_q_internal_is_lateral_plus_vertical(self) -> None:
        assert step(lat=2.0, vert=3.0).q_internal == 5.0

    def test_q_external_definition(self) -> None:
        # lat_ext + vert_ext - (lat + vert)
        assert step(lat=1.0, vert=2.0, lat_ext=10.0, vert_ext=20.0).q_external == 27.0

    def test_q_external_can_be_negative(self) -> None:
        assert step(lat=5.0, vert=5.0, lat_ext=1.0, vert_ext=1.0).q_external == -8.0

    def test_total_water_out_includes_vapor(self) -> None:
        assert step(lat=1.0, vert=2.0, vap=3.0).total_water_out == 6.0


class TestRtForcing:
    def test_q_out_requires_hydro_step(self) -> None:
        d: RtForcing = RtForcing()
        with pytest.raises(AssertionError):
            _ = d.q_out

    def test_q_out_uses_external_export(self) -> None:
        d: RtForcing = RtForcing(
            conc_in=torch.zeros(2),
            hydro_step=step(lat_ext=4.0, vert_ext=6.0),
        )
        assert d.q_out == 10.0

    def test_conc_in_default_is_empty(self) -> None:
        assert RtForcing().conc_in.numel() == 0


class TestDefaultRecords:
    def test_lapse_rate_defaults(self) -> None:
        params: LapseRateParameters = LapseRateParameters()
        assert params.temp_factor == 0.0
        assert params.precip_factor == 0.0

    def test_hydro_forcing_defaults(self) -> None:
        d: HydroForcing = HydroForcing()
        assert (d.precip, d.temp, d.pet, d.q_in) == (0.0, 0.0, 0.0, 0.0)

    def test_chemical_state_defaults_are_empty_dicts(self) -> None:
        state: ChemicalState = ChemicalState()
        assert state.mineral == {}
        assert state.exchange == {}
        assert state.primary == {}
        assert state.secondary == {}

    def test_zero_conc_sentinel(self) -> None:
        assert ZERO_CONC == 1e-20

    def test_forcing_data_holds_series(self) -> None:
        import pandas as pd

        fd: ForcingData = ForcingData(
            precip=pd.Series([1.0]), temp=pd.Series([2.0]), pet=pd.Series([3.0])
        )
        assert float(fd.precip.iloc[0]) == 1.0


class TestMiscData:
    def test_getters(self) -> None:
        data: MiscData = MiscData(
            mineral_stoichiometry=torch.ones(3, 1),
            species_mobility=torch.tensor([True, False]),
            mineral_molar_mass=torch.tensor([1.0]),
            rate_const=torch.tensor([2.0]),
        )
        assert data.get_mineral_stoichiometry().shape == (3, 1)
        assert data.get_species_mobility().dtype == torch.bool
        assert float(data.get_mineral_molar_mass()[0]) == 1.0
        assert float(data.get_rate_const()[0]) == 2.0
