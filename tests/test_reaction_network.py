"""Tests for :mod:`potions.reaction_network` and the network factories.

The matrices themselves are checked for structural consistency (shapes line up
with the species counts, the mobility mask marks exactly the aqueous rows, the
conservation matrix has one row per total species); the species ordering is
pinned because it is the authority the Rust core uses.
"""

from __future__ import annotations

import torch
import pytest

from potions.reaction_network import ReactionNetwork
from potions.reactive_transport.common_networks import (
    get_carbon_network,
    get_chloride_network,
    get_simple_carbon_network,
)


class TestSimpleCarbonAdsorptionNetwork:
    def test_species_order(self, carbon_network: ReactionNetwork) -> None:
        assert carbon_network.species_order == ["DOC", "SOC(s)", "X-", "XDOC"]

    def test_species_types(self, carbon_network: ReactionNetwork) -> None:
        assert carbon_network.species_types == {
            "DOC": "primary",
            "SOC(s)": "mineral",
            "X-": "exchange",
            "XDOC": "exchange",
        }

    def test_counts(self, carbon_network: ReactionNetwork) -> None:
        assert carbon_network.num_species == 4
        assert carbon_network.num_aqueous_species == 1
        assert carbon_network.num_secondary == 0
        assert carbon_network.num_minerals == 1
        assert carbon_network.has_exchange
        assert carbon_network.num_exchange_species == 2
        assert carbon_network.num_mineral_parameters == 5

    def test_index_groups(self, carbon_network: ReactionNetwork) -> None:
        assert carbon_network.primary_ids == [0]
        assert carbon_network.secondary_ids == []
        assert carbon_network.mineral_ids == [1]
        assert carbon_network.exchange_ids == [2, 3]

    def test_names(self, carbon_network: ReactionNetwork) -> None:
        assert carbon_network.primary_names == ["DOC"]
        assert carbon_network.secondary_names == []
        assert carbon_network.mineral_names == ["SOC(s)"]
        assert carbon_network.exchange_names == ["X-", "XDOC"]
        assert carbon_network.total_species == ["DOC", "SOC(s)", "X-"]

    def test_species_indices(self, carbon_network: ReactionNetwork) -> None:
        assert carbon_network.species_indices == {
            "DOC": 0,
            "SOC(s)": 1,
            "X-": 2,
            "XDOC": 3,
        }

    def test_species_dataframes(self, carbon_network: ReactionNetwork) -> None:
        species = carbon_network.species
        assert list(species.columns) == ["name", "type"]
        assert list(species["name"]) == carbon_network.species_order
        equilibrium = carbon_network.equilibrium_species
        assert "mineral" not in set(equilibrium["type"])
        assert list(carbon_network.kinetic_species["name"]) == carbon_network.species_order

    def test_mobility_mask(self, carbon_network: ReactionNetwork) -> None:
        mask: torch.Tensor = carbon_network.transport_mask
        assert mask.dtype == torch.bool
        assert mask.tolist() == [True, False, False, False]
        assert torch.equal(carbon_network.mobile_mask, mask)

    def test_charges_length(self, carbon_network: ReactionNetwork) -> None:
        assert carbon_network.charges.shape == (carbon_network.num_species,)

    def test_matrix_shapes(self, carbon_network: ReactionNetwork) -> None:
        ns: int = carbon_network.num_species
        nm: int = carbon_network.num_minerals
        assert carbon_network.mineral_stoichiometry.shape == (ns, nm)
        assert carbon_network.monod_params.monod_mat.shape == (nm, ns)
        assert carbon_network.tst_params.stoich.shape == (nm, ns)
        assert carbon_network.rate_consts.shape == (nm,)
        assert carbon_network.mineral_molar_masses.shape == (nm,)

    def test_equilibrium_parameters_consistency(self, carbon_network: ReactionNetwork) -> None:
        eq = carbon_network.equilibrium_parameters
        assert eq.total.shape[0] == len(carbon_network.total_species)
        assert eq.total_names == tuple(carbon_network.total_species)
        assert eq.stoich.shape == (carbon_network.num_secondary + 1, carbon_network.num_species)
        assert eq.stoich_null_space.shape[0] == carbon_network.num_species
        assert eq.num_species == carbon_network.num_species

    def test_default_aqueous_initial_state(self, carbon_network: ReactionNetwork) -> None:
        state: torch.Tensor = carbon_network.get_default_aqueous_initial_state()
        assert state.shape == (carbon_network.num_aqueous_species,)
        assert bool((state == 1e-6).all())


class TestCarbonNetwork:
    def test_full_carbon_order(self) -> None:
        net: ReactionNetwork = get_carbon_network()
        assert net.species_order == [
            "DOC",
            "HCO3-",
            "H+",
            "CO2(aq)",
            "CO3--",
            "SOC(s)",
            "X-",
            "XDOC",
        ]
        assert net.num_secondary == 2
        assert net.num_aqueous_species == 5

    def test_charge_vector_has_nonzero_hydrogen(self) -> None:
        net: ReactionNetwork = get_carbon_network()
        h_index: int = net.species_order.index("H+")
        assert float(net.charges[h_index]) == 1.0


class TestSimpleCarbonNetwork:
    def test_no_exchange(self) -> None:
        net: ReactionNetwork = get_simple_carbon_network()
        assert not net.has_exchange
        assert net.num_exchange_species == 0
        assert net.exchange_names == ["X-"]  # the accessor always leads with X-


class TestChlorideNetwork:
    def test_transport_only(self) -> None:
        net: ReactionNetwork = get_chloride_network()
        assert net.species_order == ["Cl-"]
        assert net.num_aqueous_species == 1
        assert net.num_minerals == 0
        assert not net.has_exchange
        assert net.get_default_aqueous_initial_state().shape == (1,)

    def test_empty_mineral_matrices(self) -> None:
        net: ReactionNetwork = get_chloride_network()
        assert net.mineral_stoichiometry.shape == (1, 0)
        assert net.monod_params.monod_mat.shape == (0, 1)
        assert net.rate_consts.shape == (0,)
