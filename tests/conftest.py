"""Shared pytest fixtures for the potions test suite.

The suite exercises the numerical kernels on CPU float64 (the golden-parity
precision) unless a fixture explicitly asks for something else, so results are
deterministic and independent of whether a GPU happens to be present.  The
package is 100% eager (no ``torch.compile``), so the CPU-f64 path is
bit-identical to the scalar reference by construction.
"""

from __future__ import annotations

import torch
import pytest

from potions.reaction_network import ReactionNetwork


@pytest.fixture
def cpu_f64() -> tuple[torch.device, torch.dtype]:
    """The (device, dtype) pair used by the parity / scalar-oracle runs."""
    return torch.device("cpu"), torch.float64


@pytest.fixture(scope="session")
def carbon_network() -> ReactionNetwork:
    """The ``get_simple_carbon_adsorption_network`` network (session-scoped).

    Building the network loads the default chemical database and runs scipy's
    null-space / pinv construction, so it is cached for the whole session.
    """
    from potions.reactive_transport.common_networks import (
        get_simple_carbon_adsorption_network,
    )

    return get_simple_carbon_adsorption_network()
