"""Tests for :mod:`potions.rt_zone`.

These cover the public reactive-transport zone API on the simple
carbon/adsorption network: transport, kinetic diagnostics, the analytic
residual/Jacobian, the fused ``(residual, Jacobian)`` closure that feeds the
Newton solver, and a full (speciation + kinetics) step.  ``RiverZone`` is
exercised through its differing transport field (``q_internal``) and its
``bed_depth`` mineral scale.
"""

from __future__ import annotations

import pytest
import torch

from potions.common_types import HydroForcing, HydroStep, RtForcing, RtStep
from potions.kinetic_structures import RiverParameters, RtParameters
from potions.reaction_network import ReactionNetwork
from potions.reactive_transport.common_networks import (
    get_simple_carbon_adsorption_network,
)
from potions.rt_zone import RiverZone, RtZone

F64: torch.dtype = torch.float64


def make_hydro_step(
    *,
    state: float = 100.0,
    lat_flux: float = 2.0,
    vert_flux: float = 1.0,
    q_in: float = 3.0,
) -> HydroStep:
    """Build a scalar :class:`HydroStep` with internally consistent fluxes."""
    return HydroStep(
        state=torch.tensor(state, dtype=F64),
        forc_flux=torch.tensor(5.0, dtype=F64),
        lat_flux=torch.tensor(lat_flux, dtype=F64),
        vert_flux=torch.tensor(vert_flux, dtype=F64),
        vap_flux=torch.tensor(0.0, dtype=F64),
        q_in=torch.tensor(q_in, dtype=F64),
        lat_flux_ext=torch.tensor(lat_flux, dtype=F64),
        vert_flux_ext=torch.tensor(vert_flux, dtype=F64),
    )


def make_forcing(
    num_species: int, *, conc_in_doc: float = 2e-3, s_w: float = 0.4
) -> RtForcing:
    """Build a scalar :class:`RtForcing` with only DOC entering the zone."""
    conc_in: torch.Tensor = torch.zeros(num_species, dtype=F64)
    conc_in[0] = conc_in_doc
    return RtForcing(
        conc_in=conc_in,
        hydro_step=make_hydro_step(),
        hydro_forc=HydroForcing(precip=1.0, temp=15.0),
        s_w=s_w,
        z_w=0.0,
    )


def make_params() -> RtParameters:
    """Zone parameters: dims (porosity, depth, passive) + one mineral block."""
    packed: torch.Tensor = torch.tensor(
        [0.4, 300.0, 5.0, 0.3, 1.0, 2.0, 0.0], dtype=F64
    )
    return RtParameters.from_array(packed, natural_scales=True)


def make_zone(
    network: ReactionNetwork, *, do_reactions: bool = True, do_speciation: bool = True
) -> RtZone:
    """Build an :class:`RtZone` on ``network`` with :func:`make_params`."""
    return RtZone(
        network,
        make_params(),
        do_reactions=do_reactions,
        do_speciation=do_speciation,
        name="test_zone",
    )


def initial_conc(num_species: int) -> torch.Tensor:
    """A plausible non-trivial initial concentration vector."""
    return torch.tensor([1e-3, 1.0, 1e-6, 0.05][:num_species], dtype=F64)


class TestConstruction:
    def test_introspection(self, carbon_network: ReactionNetwork) -> None:
        zone: RtZone = make_zone(carbon_network)
        assert zone.name == "test_zone"
        assert zone.all_species == carbon_network.species_names
        assert zone.mineral_species == carbon_network.mineral_species_names
        assert zone.dimensions.depth == 300.0
        assert zone.dimensions.porosity == 0.4
        assert zone.aux is not None
        assert zone.aux.ssa.item() == 1.0

    def test_to_array_roundtrip(self, carbon_network: ReactionNetwork) -> None:
        zone: RtZone = make_zone(carbon_network)
        packed: torch.Tensor = zone.to_array()
        assert packed.shape == (7,)
        rebuilt: RtZone = RtZone.from_array(
            packed,
            carbon_network,
            do_reactions=True,
            do_speciation=True,
            name="rebuilt",
            natural_scales=False,
        )
        torch.testing.assert_close(rebuilt.to_array(), packed)

    def test_no_minerals_has_none_aux(self, carbon_network: ReactionNetwork) -> None:
        zone: RtZone = RtZone(
            carbon_network,
            RtParameters.from_array(torch.tensor([0.4, 300.0, 5.0], dtype=F64)),
            do_reactions=False,
            do_speciation=False,
            name="inert",
        )
        assert zone.aux is None


class TestDiagnostics:
    def test_monod_and_tst_rate_shapes(self, carbon_network: ReactionNetwork) -> None:
        zone: RtZone = make_zone(carbon_network)
        conc: torch.Tensor = initial_conc(carbon_network.num_species)
        assert zone.monod_rate(conc).shape == (carbon_network.num_minerals,)
        assert zone.tst_rate(conc).shape == (carbon_network.num_minerals,)

    def test_aux_factor(self, carbon_network: ReactionNetwork) -> None:
        zone: RtZone = make_zone(carbon_network)
        factor: torch.Tensor = zone.aux_factor(make_forcing(carbon_network.num_species))
        assert factor.ndim == 0
        assert torch.isfinite(factor)

    def test_aux_factor_without_params_raises(self, carbon_network: ReactionNetwork) -> None:
        zone: RtZone = RtZone(
            carbon_network,
            RtParameters.from_array(torch.tensor([0.4, 300.0, 5.0], dtype=F64)),
            do_reactions=False,
            do_speciation=False,
            name="inert",
        )
        with pytest.raises(ValueError, match="without kinetic parameters"):
            zone.aux_factor(make_forcing(carbon_network.num_species))

    def test_tot_moles_conc_roundtrip(self, carbon_network: ReactionNetwork) -> None:
        zone: RtZone = make_zone(carbon_network)
        conc: torch.Tensor = initial_conc(carbon_network.num_species)
        d: RtForcing = make_forcing(carbon_network.num_species)
        moles: torch.Tensor = zone.get_tot_moles(conc, d)
        back: torch.Tensor = zone.moles_to_conc(moles, d)
        torch.testing.assert_close(back, conc)

    def test_batched_leading_dim_preserved(self, carbon_network: ReactionNetwork) -> None:
        zone: RtZone = make_zone(carbon_network)
        conc: torch.Tensor = initial_conc(carbon_network.num_species)
        batch: torch.Tensor = conc.unsqueeze(0).expand(3, -1).contiguous()
        d: RtForcing = make_forcing(carbon_network.num_species)
        assert zone.mass_balance_ode(batch, d).shape == (3, carbon_network.num_species)


class TestTransportRate:
    def test_aqueous_row_uses_q_in_over_state(self, carbon_network: ReactionNetwork) -> None:
        zone: RtZone = make_zone(carbon_network)
        conc: torch.Tensor = initial_conc(carbon_network.num_species)
        d: RtForcing = make_forcing(carbon_network.num_species, conc_in_doc=2e-3)
        out: torch.Tensor = zone.transport_rate(conc, d)
        # q_in / state = 3 / 100, times (c_in - c) on the DOC row only.
        expected: float = (3.0 / 100.0) * (2e-3 - 1e-3)
        assert out[0].item() == expected
        assert bool((out[1:] == 0.0).all())

    def test_zero_state_sentinel(self, carbon_network: ReactionNetwork) -> None:
        zone: RtZone = make_zone(carbon_network)
        d: RtForcing = RtForcing(
            conc_in=torch.ones(carbon_network.num_species, dtype=F64),
            hydro_step=make_hydro_step(state=0.0),
            hydro_forc=HydroForcing(precip=1.0, temp=15.0),
            s_w=0.4,
        )
        out: torch.Tensor = zone.transport_rate(
            initial_conc(carbon_network.num_species), d
        )
        # The aqueous row uses the ZERO_CONC sentinel (~1e-20), not exactly 0.
        assert out[0].abs().item() < 1e-18
        assert bool((out[1:] == 0.0).all())


class TestResidualAndJacobian:
    def test_residual_at_initial_is_dt_times_ode(self, carbon_network: ReactionNetwork) -> None:
        zone: RtZone = make_zone(carbon_network)
        conc: torch.Tensor = initial_conc(carbon_network.num_species)
        d: RtForcing = make_forcing(carbon_network.num_species)
        residual: torch.Tensor = zone.residual_function(conc, conc, d, 1.0)
        ode: torch.Tensor = zone.mass_balance_ode(conc, d)
        torch.testing.assert_close(residual, ode)

    def test_jacobian_shape_and_diagonal(self, carbon_network: ReactionNetwork) -> None:
        zone: RtZone = make_zone(carbon_network)
        conc: torch.Tensor = initial_conc(carbon_network.num_species)
        d: RtForcing = make_forcing(carbon_network.num_species)
        jac: torch.Tensor = zone.jacobian_residual_function(conc, conc, d, 1.0)
        n: int = carbon_network.num_species
        assert jac.shape == (n, n)
        # -I from the (c_0 - conc) term is always present on the diagonal.
        assert bool((jac.diagonal() < 0.0).all())

    def test_fused_closure_matches_split_paths(self, carbon_network: ReactionNetwork) -> None:
        zone: RtZone = make_zone(carbon_network)
        conc: torch.Tensor = initial_conc(carbon_network.num_species)
        d: RtForcing = make_forcing(carbon_network.num_species)
        fused, c0b = zone.build_fused_eval(d, conc, 1.0)
        residual, jac = fused(c0b)
        torch.testing.assert_close(
            residual.squeeze(0), zone.residual_function(conc, conc, d, 1.0)
        )
        torch.testing.assert_close(
            jac.squeeze(0), zone.jacobian_residual_function(conc, conc, d, 1.0)
        )


class TestReactionGating:
    def test_reaction_rate_requires_reactions(self, carbon_network: ReactionNetwork) -> None:
        zone: RtZone = make_zone(carbon_network, do_reactions=False, do_speciation=False)
        d: RtForcing = make_forcing(carbon_network.num_species)
        with pytest.raises(ValueError, match="without kinetic properties"):
            zone.reaction_rate(initial_conc(carbon_network.num_species), d)

    def test_ode_equals_transport_without_reactions(
        self, carbon_network: ReactionNetwork
    ) -> None:
        zone: RtZone = make_zone(carbon_network, do_reactions=False, do_speciation=False)
        conc: torch.Tensor = initial_conc(carbon_network.num_species)
        d: RtForcing = make_forcing(carbon_network.num_species)
        assert torch.equal(zone.mass_balance_ode(conc, d), zone.transport_rate(conc, d))

    def test_mass_balance_ode_includes_reactions(self, carbon_network: ReactionNetwork) -> None:
        zone: RtZone = make_zone(carbon_network)
        conc: torch.Tensor = initial_conc(carbon_network.num_species)
        d: RtForcing = make_forcing(carbon_network.num_species)
        ode: torch.Tensor = zone.mass_balance_ode(conc, d)
        with_rxn: torch.Tensor = zone.transport_rate(conc, d) + zone.reaction_rate(conc, d)
        torch.testing.assert_close(ode, with_rxn)


class TestStep:
    def test_step_returns_finite_rtstep(self, carbon_network: ReactionNetwork) -> None:
        zone: RtZone = make_zone(carbon_network)
        n: int = carbon_network.num_species
        d: RtForcing = make_forcing(n)
        step: RtStep = zone.step(initial_conc(n), d, 1.0)
        for field in (
            step.state,
            step.total_moles,
            step.conc_in,
            step.mass_in,
            step.lat_conc,
            step.vert_conc,
            step.lat_mass,
            step.vert_mass,
        ):
            assert field.shape == (n,)
            assert torch.isfinite(field).all()
        assert step.mineral_rates.shape == (carbon_network.num_minerals,)

    def test_step_batched(self, carbon_network: ReactionNetwork) -> None:
        zone: RtZone = make_zone(carbon_network)
        n: int = carbon_network.num_species
        conc: torch.Tensor = initial_conc(n).unsqueeze(0).expand(2, -1).contiguous()
        step: RtStep = zone.step(conc, make_forcing(n), 1.0)
        assert step.state.shape == (2, n)
        assert step.mineral_rates.shape == (2, carbon_network.num_minerals)

    def test_step_with_params_batched(self, carbon_network: ReactionNetwork) -> None:
        n: int = carbon_network.num_species
        rows: torch.Tensor = torch.tensor(
            [
                [0.4, 300.0, 5.0, 0.3, 1.0, 2.0, 0.0],
                [0.45, 200.0, 2.0, 0.25, 1.5, 1.5, 0.0],
                [0.30, 400.0, 10.0, 0.35, 0.8, 2.5, 0.0],
            ],
            dtype=F64,
        )
        batched: RtParameters = RtParameters.from_batched_array(rows, natural_scales=True)
        zone: RtZone = RtZone(carbon_network, batched, name="batched")
        conc: torch.Tensor = initial_conc(n).unsqueeze(0).expand(3, -1).contiguous()
        d: RtForcing = make_forcing(n)
        d = RtForcing(
            conc_in=d.conc_in.expand(3, n).contiguous(),
            hydro_step=make_hydro_step(),
            hydro_forc=d.hydro_forc,
            s_w=0.4,
            z_w=0.0,
        )
        step: RtStep = zone.step_with_params(
            conc, d, 1.0, aux=batched.mineral_params, depth=batched.dimensions.depth
        )
        assert step.state.shape == (3, n)
        assert step.mineral_rates.shape == (3, carbon_network.num_minerals)
        assert torch.isfinite(step.state).all()
        # Different per-lane mineral parameters must yield different rates.
        assert not torch.allclose(step.mineral_rates[0], step.mineral_rates[1])


class TestRiverZone:
    def make_river(self, network: ReactionNetwork) -> RiverZone:
        packed: torch.Tensor = torch.tensor(
            [300.0, 5.0, 0.3, 1.0, 2.0, 0.0], dtype=F64
        )
        params: RiverParameters = RiverParameters.from_array(packed, natural_scales=True)
        return RiverZone(network, params, name="river")

    def test_construction(self, carbon_network: ReactionNetwork) -> None:
        zone: RiverZone = self.make_river(carbon_network)
        assert zone.name == "river"
        assert zone.dimensions.bed_depth == 300.0
        assert zone.all_species == carbon_network.species_names

    def test_tot_moles_uses_bed_depth(self, carbon_network: ReactionNetwork) -> None:
        zone: RiverZone = self.make_river(carbon_network)
        n: int = carbon_network.num_species
        conc: torch.Tensor = initial_conc(n)
        d: RtForcing = make_forcing(n)
        moles: torch.Tensor = zone.get_tot_moles(conc, d)
        back: torch.Tensor = zone.moles_to_conc(moles, d)
        torch.testing.assert_close(back, conc)

    def test_transport_uses_q_internal(self, carbon_network: ReactionNetwork) -> None:
        zone: RiverZone = self.make_river(carbon_network)
        n: int = carbon_network.num_species
        conc: torch.Tensor = initial_conc(n)
        d: RtForcing = make_forcing(n, conc_in_doc=2e-3)
        out: torch.Tensor = zone.transport_rate(conc, d)
        q_internal: float = 3.0  # lat_flux + vert_flux = 2 + 1
        expected: float = (q_internal / 100.0) * (2e-3 - 1e-3)
        assert out[0].item() == expected

    def test_step(self, carbon_network: ReactionNetwork) -> None:
        zone: RiverZone = self.make_river(carbon_network)
        n: int = carbon_network.num_species
        step: RtStep = zone.step(initial_conc(n), make_forcing(n), 1.0)
        assert step.state.shape == (n,)
        assert torch.isfinite(step.state).all()
