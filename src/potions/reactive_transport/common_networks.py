"""Common reaction-network factory functions.

Verbatim port of the reference (compiled) ``potions`` package's
``reactive_transport/common_networks.py``.  Each factory builds a
:class:`~potions.reaction_network.ReactionNetwork` over a documented set of
species drawn from the default chemical database
(:meth:`~potions.database.ChemicalDatabase.load_default`).

The species sets are the same as the reference implementation; only the import
paths differ (this project loads the database and species dataclasses from the
pure-Python ``potions`` modules rather than the compiled ``potions.core``).
"""

from __future__ import annotations

from ..database import (
    ChemicalDatabase,
    ExchangeReaction,
    MineralSpecies,
    MineralKineticData,
    PrimaryAqueousSpecies,
    SecondarySpecies,
)
from ..reaction_network import ReactionNetwork

__all__: list[str] = [
    "get_carbon_network",
    "get_simple_carbon_network",
    "get_simple_carbon_adsorption_network",
    "get_chloride_network",
]


def get_carbon_network() -> ReactionNetwork:
    """Full carbonate system with DOC, dissolved inorganic carbon, and SOC(s).

    Primary aqueous species: ``DOC``, ``HCO3-``, ``H+``.  Secondary (equilibrium)
    species: ``CO2(aq)``, ``CO3--``.  Mineral: ``SOC(s)`` with the ``test``
    kinetic label.  Exchange: ``XDOC`` adsorption.
    """
    db: ChemicalDatabase = ChemicalDatabase.load_default()
    primary_species_names: list[str] = ["DOC", "HCO3-", "H+"]
    secondary_species_names: list[str] = ["CO2(aq)", "CO3--"]
    mineral_species_names: list[str] = ["SOC(s)"]
    mineral_kinetics_names: list[str] = ["test"]
    exchange_species_names: list[str] = ["XDOC"]

    primary_species: list[PrimaryAqueousSpecies] = db.get_primary_aqueous_species(
        primary_species_names
    )
    secondary_species: list[SecondarySpecies] = db.get_secondary_species(
        secondary_species_names
    )
    mineral_species: list[MineralSpecies] = db.get_mineral_species(mineral_species_names)
    min_kin: MineralKineticData = db.get_mineral_reactions(
        mineral_species_names, labels=mineral_kinetics_names
    )
    exchange_species: list[ExchangeReaction] = db.get_exchange_reactions(
        exchange_species_names
    )

    network: ReactionNetwork = ReactionNetwork(
        primary_aqueous=primary_species,
        mineral=mineral_species,
        mineral_kinetics=min_kin,
        secondary=secondary_species,
        exchange_species=exchange_species,
    )
    return network


def get_simple_carbon_network() -> ReactionNetwork:
    """Minimal carbon network: DOC + SOC(s) "test" kinetics, no equilibrium/adsorption.

    Primary aqueous: ``DOC``.  No secondary species, no exchange.  Mineral:
    ``SOC(s)`` with the ``test`` kinetic label.
    """
    db: ChemicalDatabase = ChemicalDatabase.load_default()
    primary_species_names: list[str] = ["DOC"]
    secondary_species_names: list[str] = []
    mineral_species_names: list[str] = ["SOC(s)"]
    mineral_kinetics_names: list[str] = ["test"]
    exchange_species_names: list[str] = []

    primary_species: list[PrimaryAqueousSpecies] = db.get_primary_aqueous_species(
        primary_species_names
    )
    secondary_species: list[SecondarySpecies] = db.get_secondary_species(
        secondary_species_names
    )
    mineral_species: list[MineralSpecies] = db.get_mineral_species(mineral_species_names)
    min_kin: MineralKineticData = db.get_mineral_reactions(
        mineral_species_names, labels=mineral_kinetics_names
    )
    exchange_species: list[ExchangeReaction] = db.get_exchange_reactions(
        exchange_species_names
    )

    network: ReactionNetwork = ReactionNetwork(
        primary_aqueous=primary_species,
        mineral=mineral_species,
        mineral_kinetics=min_kin,
        secondary=secondary_species,
        exchange_species=exchange_species,
    )
    return network


def get_simple_carbon_adsorption_network() -> ReactionNetwork:
    """Simple carbon network with SOC(s) kinetics *and* XDOC adsorption.

    This is the network used by the East River / Coal Creek showcase workflow.

    Primary aqueous: ``DOC``.  No secondary species.  Mineral: ``SOC(s)`` with
    the ``test`` kinetic label.  Exchange: ``XDOC`` (ionic adsorption of DOC on
    the ``X-`` exchange site).
    """
    db: ChemicalDatabase = ChemicalDatabase.load_default()
    primary_species_names: list[str] = ["DOC"]
    secondary_species_names: list[str] = []
    mineral_species_names: list[str] = ["SOC(s)"]
    mineral_kinetics_names: list[str] = ["test"]
    exchange_species_names: list[str] = ["XDOC"]

    primary_species: list[PrimaryAqueousSpecies] = db.get_primary_aqueous_species(
        primary_species_names
    )
    secondary_species: list[SecondarySpecies] = db.get_secondary_species(
        secondary_species_names
    )
    mineral_species: list[MineralSpecies] = db.get_mineral_species(mineral_species_names)
    min_kin: MineralKineticData = db.get_mineral_reactions(
        mineral_species_names, labels=mineral_kinetics_names
    )
    exchange_species: list[ExchangeReaction] = db.get_exchange_reactions(
        exchange_species_names
    )

    network: ReactionNetwork = ReactionNetwork(
        primary_aqueous=primary_species,
        mineral=mineral_species,
        mineral_kinetics=min_kin,
        secondary=secondary_species,
        exchange_species=exchange_species,
    )
    return network


def get_chloride_network() -> ReactionNetwork:
    """Transport-only chloride network (no minerals, no reactions, no exchange).

    Primary aqueous: ``Cl-``.  Useful for pure-advection/dispersion checks.
    """
    db: ChemicalDatabase = ChemicalDatabase.load_default()
    primary_species_names: list[str] = ["Cl-"]
    primary_species: list[PrimaryAqueousSpecies] = db.get_primary_aqueous_species(
        primary_species_names
    )
    min_kin: MineralKineticData = db.get_mineral_reactions([], labels=[])
    network: ReactionNetwork = ReactionNetwork(
        primary_aqueous=primary_species,
        mineral=[],
        mineral_kinetics=min_kin,
        secondary=[],
        exchange_species=[],
    )
    return network
