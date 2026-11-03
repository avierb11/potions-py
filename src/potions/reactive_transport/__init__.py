"""Reactive-transport helpers: network factory functions.

This subpackage mirrors the layout of the reference (compiled) ``potions``
package, where common reaction networks are constructed from
``potions.reactive_transport.common_networks``.  The showcase and other
downstream code imports that exact path, so we expose the same factory
functions here, built on top of :class:`~potions.database.ChemicalDatabase`
and :class:`~potions.reaction_network.ReactionNetwork`.
"""

from .common_networks import (
    get_carbon_network,
    get_chloride_network,
    get_simple_carbon_adsorption_network,
    get_simple_carbon_network,
)

__all__: list[str] = [
    "get_carbon_network",
    "get_chloride_network",
    "get_simple_carbon_adsorption_network",
    "get_simple_carbon_network",
]
