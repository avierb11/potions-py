"""Reaction-network construction mirroring the reference Rust ``reaction_network.rs``.

A :class:`ReactionNetwork` bundles the species lists (primary aqueous, secondary,
mineral, exchange sites) and the selected kinetic rate laws, then derives every
matrix the reactive-transport solver needs.  The matrices are torch tensors
(float64 CPU, ``requires_grad=False``) built **once** per network; the batched
solver casts them to the target device/dtype as needed.

The species ordering is fixed and matches the Rust core:

``[primary..., secondary..., mineral..., (X-, exchange...)]``

The Rust core is the authority for matrix *contents*.  In particular the
``tst`` tensor matrices are swapped relative to their natural names (see the
``kinetic_structures`` module docstring), whereas the legacy ``potions`` Python
wrapper (``src/potions/reactive_transport/reaction_network.py``) produces the
opposite layout and is **not** what the notebooks/golden tests execute.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import numpy.typing as npt
import torch
from pandas import DataFrame

from .database import (
    ExchangeReaction,
    MineralKineticData,
    MineralSpecies,
    PrimaryAqueousSpecies,
    SecondarySpecies,
)
from .equilibrium import (
    EquilibriumParameters,
    compute_stoich_null_space,
)
from .kinetic_structures import MonodParameters, TstParameters

Tensor: TypeAlias = torch.Tensor
FloatArray: TypeAlias = npt.NDArray[np.float64]


def species_table(
    primary_aqueous: list[PrimaryAqueousSpecies],
    secondary: list[SecondarySpecies],
    mineral: list[MineralSpecies],
    exchange_species: list[ExchangeReaction],
) -> tuple[list[str], list[str]]:
    """Build the ordered ``(names, types)`` lists laid out as in the Rust core.

    Order: primary, secondary, mineral, then (when any exchange reaction is
    present) the ``X-`` site followed by each exchange species.

    Args:
        primary_aqueous: primary aqueous species in network order.
        secondary:       secondary species in network order.
        mineral:         mineral species in network order.
        exchange_species: exchange reactions; ``X-`` is prepended when non-empty.

    Returns:
        ``(names, types)`` where ``types[i]`` describes ``names[i]``.
    """
    names: list[str] = [x.name for x in primary_aqueous]
    names += [x.name for x in secondary]
    names += [x.name for x in mineral]
    types: list[str] = ["primary"] * len(primary_aqueous)
    types += ["secondary"] * len(secondary)
    types += ["mineral"] * len(mineral)
    if exchange_species:
        names.append("X-")
        names += [x.name for x in exchange_species]
        types.append("exchange")
        types += ["exchange"] * len(exchange_species)
    return names, types


def charges_vector(
    primary_aqueous: list[PrimaryAqueousSpecies],
    secondary: list[SecondarySpecies],
    mineral: list[MineralSpecies],
    exchange_species: list[ExchangeReaction],
    has_exchange: bool,
) -> FloatArray:
    """Build the per-species charge vector in species order.

    Minerals and the ``X-`` site carry charge ``0.0``; exchange species use
    their own charge (mirrors the Rust ``charges`` getter).
    """
    charges: list[float] = [x.charge for x in primary_aqueous]
    charges += [x.charge for x in secondary]
    charges += [0.0] * len(mineral)
    if has_exchange:
        charges.append(0.0)
        charges += [x.charge for x in exchange_species]
    return np.asarray(charges, dtype=np.float64)


def transport_mask(num_species: int, num_aqueous_species: int) -> FloatArray:
    """Boolean mobility mask: True for aqueous, False for mineral/X-/exchange."""
    mask = np.ones(num_species, dtype=bool)
    mask[num_aqueous_species:] = False
    return mask


def mineral_stoich_matrix(
    mineral: list[MineralSpecies], species_names: list[str]
) -> FloatArray:
    """(num_species, num_minerals) mineral dissolution stoichiometry."""
    n_spec = len(species_names)
    matrix = np.zeros((n_spec, len(mineral)), dtype=np.float64)
    for col, m in enumerate(mineral):
        for row, spec in enumerate(species_names):
            matrix[row, col] = m.stoichiometry.get(spec, 0.0)
    return matrix


def monod_inhib_matrices(
    mineral_kinetics: MineralKineticData,
    mineral: list[MineralSpecies],
    species_names: list[str],
) -> tuple[FloatArray, FloatArray]:
    """(num_minerals, num_species) monod / inhibition matrices.

    Entries are ``NaN`` where the reaction defines no term for the species; a
    mineral without a monod reaction gets an all-zero row (Rust keeps ``0.0``).
    """
    n_spec = len(species_names)
    monod_np = np.full((len(mineral), n_spec), np.nan)
    inhib_np = np.full((len(mineral), n_spec), np.nan)
    for row, m in enumerate(mineral):
        rxn = mineral_kinetics.monod_reactions.get(m.name)
        if rxn is None:
            monod_np[row, :] = 0.0
            inhib_np[row, :] = 0.0
            continue
        for col, spec in enumerate(species_names):
            monod_np[row, col] = rxn.monod_terms.get(spec, np.nan)
            inhib_np[row, col] = rxn.inhib_terms.get(spec, np.nan)
    return monod_np, inhib_np


def tst_matrices(
    mineral_kinetics: MineralKineticData,
    mineral: list[MineralSpecies],
    species_names: list[str],
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """TST matrices with the Rust core's swapped layout.

    Returns:
        ``(stoich, dep, min_eq_const)`` where:
        - ``stoich[i, j]`` = TST reaction ``dependence`` of mineral ``i`` on
          species ``j`` (0 where the dependence map lacks the species);
        - ``dep[i, j]``    = mineral ``i``'s stoichiometry for species ``j``,
          masked to the species present in its dependence map;
        - ``min_eq_const[i]`` = ``10 ** eq_consts[1]`` for TST minerals else ``1.0``.
    """
    n_spec = len(species_names)
    stoich_np = np.zeros((len(mineral), n_spec), dtype=np.float64)
    dep_np = np.zeros((len(mineral), n_spec), dtype=np.float64)
    min_eq_const = np.ones(len(mineral), dtype=np.float64)
    for row, m in enumerate(mineral):
        rxn = mineral_kinetics.tst_reactions.get(m.name)
        if rxn is None:
            continue
        min_eq_const[row] = 10.0**m.eq_consts[1]
        for col, spec in enumerate(species_names):
            if spec in rxn.dependence:
                stoich_np[row, col] = rxn.dependence[spec]
                dep_np[row, col] = m.stoichiometry.get(spec, 0.0)
    return stoich_np, dep_np, min_eq_const


def rate_const_vector(
    mineral_kinetics: MineralKineticData, mineral: list[MineralSpecies]
) -> FloatArray:
    """(num_minerals,) linear rate constants, searched monod-then-tst (Rust order)."""
    log10_rate_consts: list[float] = []
    for m in mineral:
        if m.name in mineral_kinetics.monod_reactions:
            log10_rate_consts.append(mineral_kinetics.monod_reactions[m.name].rate_constant)
        elif m.name in mineral_kinetics.tst_reactions:
            log10_rate_consts.append(mineral_kinetics.tst_reactions[m.name].rate_constant)
        else:
            raise KeyError(f"Cannot find mineral {m.name}")
    return 10.0 ** np.asarray(log10_rate_consts, dtype=np.float64)


def equilibrium_stoich_matrix(
    secondary: list[SecondarySpecies],
    exchange_species: list[ExchangeReaction],
    species_names: list[str],
) -> tuple[FloatArray, FloatArray]:
    """Equilibrium stoich matrix + log10 eq constants (secondaries, then exchange).

    Returns:
        ``(stoich, log_eq_consts)`` with shape ``(num_reactions, num_species)``
        and ``(num_reactions,)`` respectively.
    """
    n_spec = len(species_names)
    rows: list[FloatArray] = []
    log_eq: list[float] = []
    for spec in secondary:
        row = np.zeros(n_spec, dtype=np.float64)
        for col, name in enumerate(species_names):
            row[col] = spec.stoichiometry.get(name, 0.0)
        rows.append(row)
        log_eq.append(spec.eq_consts[1])
    for exch in exchange_species:
        row = np.zeros(n_spec, dtype=np.float64)
        for col, name in enumerate(species_names):
            row[col] = exch.stoichiometry.get(name, 0.0)
        rows.append(row)
        log_eq.append(exch.log10_k_eq)
    stoich = np.asarray(rows, dtype=np.float64).reshape(-1, n_spec)
    return stoich, np.asarray(log_eq, dtype=np.float64)


def total_conservation_matrix(
    total_species_names: list[str],
    species_names: list[str],
    species_ids: dict[str, int],
    secondary: list[SecondarySpecies],
    exchange_species: list[ExchangeReaction],
    charges: FloatArray,
) -> FloatArray:
    """(num_total, num_species) mass/charge conservation matrix.

    Each row corresponds to a total species (primary + mineral + ``X-``).  For
    ``H+`` the row is the charge vector; otherwise it has ``1.0`` on the total
    species' own column plus its stoichiometric coefficients on the species that
    release it (the Rust construction, with no absolute value).
    """
    n_spec = len(species_names)
    rows: list[FloatArray] = []
    for tot in total_species_names:
        if tot == "H+":
            rows.append(charges.copy())
            continue
        row = np.zeros(n_spec, dtype=np.float64)
        row[species_ids[tot]] = 1.0
        for sec in secondary:
            if tot in sec.stoichiometry:
                row[species_ids[sec.name]] = sec.stoichiometry[tot]
        for exch in exchange_species:
            if tot in exch.stoichiometry:
                row[species_ids[exch.name]] = exch.stoichiometry[tot]
        rows.append(row)
    return np.asarray(rows, dtype=np.float64).reshape(-1, n_spec)


class ReactionNetwork:
    """A complete reactive-transport network: species + kinetics + matrices.

    Mirrors the Rust ``ReactionNetwork`` pyclass.  Construct from the ordered
    species lists and the selected kinetic data; the derived matrices are
    precomputed once and exposed as torch tensors / immutable dataclasses.

    Args:
        primary_aqueous: primary aqueous species, in order.
        mineral:         mineral species, in order.
        secondary:       secondary species, in order.
        mineral_kinetics: selected kinetic rate laws (``MineralKineticData``).
        exchange_species: exchange reactions, in order (empty when none).
    """

    def __init__(
        self,
        primary_aqueous: list[PrimaryAqueousSpecies],
        mineral: list[MineralSpecies],
        secondary: list[SecondarySpecies],
        mineral_kinetics: MineralKineticData,
        exchange_species: list[ExchangeReaction] | None = None,
    ) -> None:
        self.primary_aqueous: list[PrimaryAqueousSpecies] = primary_aqueous
        self.mineral: list[MineralSpecies] = mineral
        self.secondary: list[SecondarySpecies] = secondary
        self.mineral_kinetics: MineralKineticData = mineral_kinetics
        self.exchange_species: list[ExchangeReaction] = exchange_species or []

        names, types = species_table(
            primary_aqueous, secondary, mineral, self.exchange_species
        )
        self._names: list[str] = names
        self._types: list[str] = types
        self._species_indices: dict[str, int] = {name: i for i, name in enumerate(names)}
        self._has_exchange: bool = bool(self.exchange_species)

        self._charges: FloatArray = charges_vector(
            primary_aqueous, secondary, mineral, self.exchange_species, self._has_exchange
        )
        self._mineral_stoich: FloatArray = mineral_stoich_matrix(mineral, names)
        self._monod_np, self._inhib_np = monod_inhib_matrices(
            mineral_kinetics, mineral, names
        )
        self._tst_stoich, self._tst_dep, self._min_eq_const = tst_matrices(
            mineral_kinetics, mineral, names
        )
        self._rate_consts: FloatArray = rate_const_vector(mineral_kinetics, mineral)

        # Precompute the numpy-backed *constant* network tensors exactly once.
        # The constant-tensor properties below return these stored views rather
        # than rebuilding ``torch.tensor(...)`` on every access: in the batched
        # RT hot path (``_build_fused_eval``) the stoichiometry is read on every
        # zone step, and ``_get_stoich`` keys its cache by ``id`` of this base
        # tensor.  If each access produced a fresh object, the ``id`` would never
        # be reused and the cache would grow by one entry per step (a leak).
        # Stable identity keeps the cache bounded and also removes the per-step
        # CPU->GPU materialisation of the float64 base.
        self._mineral_stoich_t: torch.Tensor = torch.tensor(
            self._mineral_stoich, dtype=torch.float64
        )
        self._mineral_molar_mass_t: torch.Tensor = torch.tensor(
            [m.molar_mass for m in self.mineral], dtype=torch.float64
        )
        self._rate_consts_t: torch.Tensor = torch.tensor(
            self._rate_consts, dtype=torch.float64
        )
        self._monod_np_t: torch.Tensor = torch.tensor(self._monod_np, dtype=torch.float64)
        self._inhib_np_t: torch.Tensor = torch.tensor(self._inhib_np, dtype=torch.float64)
        self._tst_stoich_t: torch.Tensor = torch.tensor(self._tst_stoich, dtype=torch.float64)
        self._tst_dep_t: torch.Tensor = torch.tensor(self._tst_dep, dtype=torch.float64)
        self._min_eq_const_t: torch.Tensor = torch.tensor(
            self._min_eq_const, dtype=torch.float64
        )
        self._charges_t: torch.Tensor = torch.tensor(self._charges, dtype=torch.float64)

        eq_stoich, eq_consts = equilibrium_stoich_matrix(
            secondary, self.exchange_species, names
        )
        self._eq_stoich: FloatArray = eq_stoich
        self._log_eq_consts: FloatArray = eq_consts
        self._total_mat: FloatArray = total_conservation_matrix(
            self.total_species,
            names,
            self._species_indices,
            secondary,
            self.exchange_species,
            self._charges,
        )

    # ------------------------------------------------------------------ #
    # Species bookkeeping
    # ------------------------------------------------------------------ #

    @property
    def species_order(self) -> list[str]:
        """Species names in solve order."""
        return self._names

    @property
    def species_names(self) -> list[str]:
        """Alias of :attr:`species_order` (matches the Rust getter)."""
        return self._names

    @property
    def species_types(self) -> dict[str, str]:
        """Map ``species name -> type`` (primary/secondary/mineral/exchange)."""
        return dict(zip(self._names, self._types))

    @property
    def species(self) -> DataFrame:
        """``name``/``type`` species DataFrame (API parity with the wrapper)."""
        return DataFrame({"name": self._names, "type": self._types})

    @property
    def equilibrium_species(self) -> DataFrame:
        """Species DataFrame restricted to aqueous + exchange (non-mineral)."""
        df = self.species
        return df.loc[df["type"] != "mineral"].reset_index(drop=True)

    @property
    def kinetic_species(self) -> DataFrame:
        """The full species DataFrame (Rust returns the whole table here)."""
        return self.species

    @property
    def has_exchange(self) -> bool:
        """Whether any cation-exchange species are present."""
        return self._has_exchange

    @property
    def primary_names(self) -> list[str]:
        """Primary aqueous species names in order."""
        return [x.name for x in self.primary_aqueous]

    @property
    def secondary_names(self) -> list[str]:
        """Secondary species names in order."""
        return [x.name for x in self.secondary]

    @property
    def mineral_names(self) -> list[str]:
        """Mineral species names in order."""
        return [x.name for x in self.mineral]

    @property
    def mineral_species_names(self) -> list[str]:
        """Alias of :attr:`mineral_names` (matches the Rust getter)."""
        return self.mineral_names

    @property
    def exchange_names(self) -> list[str]:
        """Exchange site names: ``["X-", ...]`` when exchange is present."""
        names: list[str] = ["X-"]
        names += [x.name for x in self.exchange_species]
        return names

    @property
    def total_species(self) -> list[str]:
        """Total-species names: primary + mineral + ``X-`` (no secondaries)."""
        total = self.primary_names + self.mineral_names
        if self._has_exchange:
            total.append("X-")
        return total

    @property
    def species_indices(self) -> dict[str, int]:
        """Map ``species name -> column index`` in the matrices."""
        return dict(self._species_indices)

    @property
    def primary_ids(self) -> list[int]:
        """Indices of the primary aqueous species."""
        return list(range(len(self.primary_aqueous)))

    @property
    def secondary_ids(self) -> list[int]:
        """Indices of the secondary species."""
        return list(range(len(self.primary_aqueous), len(self.primary_aqueous) + len(self.secondary)))

    @property
    def mineral_ids(self) -> list[int]:
        """Indices of the mineral species."""
        return list(
            range(
                len(self.primary_aqueous) + len(self.secondary),
                len(self.primary_aqueous) + len(self.secondary) + len(self.mineral),
            )
        )

    @property
    def exchange_ids(self) -> list[int]:
        """Indices of the ``X-`` site and exchange species."""
        return list(
            range(
                len(self.primary_aqueous) + len(self.secondary) + len(self.mineral),
                self.num_species,
            )
        )

    # ------------------------------------------------------------------ #
    # Sizes
    # ------------------------------------------------------------------ #

    @property
    def num_species(self) -> int:
        """Total number of species."""
        return len(self._names)

    @property
    def num_aqueous_species(self) -> int:
        """Number of aqueous species (primary + secondary)."""
        return len(self.primary_aqueous) + len(self.secondary)

    @property
    def num_secondary(self) -> int:
        """Number of secondary species."""
        return len(self.secondary)

    @property
    def num_minerals(self) -> int:
        """Number of minerals."""
        return len(self.mineral)

    @property
    def num_exchange_species(self) -> int:
        """Number of exchange sites (``0`` or ``1 + len(exchange)``)."""
        return 1 + len(self.exchange_species) if self._has_exchange else 0

    @property
    def num_total_species(self) -> int:
        """Number of total-species rows (primary + mineral + ``X-``)."""
        return len(self.primary_aqueous) + len(self.mineral) + self.num_exchange_species

    @property
    def num_mineral_parameters(self) -> int:
        """Number of per-mineral kinetic parameters (5 in the reference)."""
        return 5 * self.num_minerals

    # ------------------------------------------------------------------ #
    # Constant tensors used by the solver
    # ------------------------------------------------------------------ #

    @property
    def charges(self) -> Tensor:
        """(num_species,) charge vector (a stable, shared tensor)."""
        return self._charges_t

    @property
    def transport_mask(self) -> Tensor:
        """(num_species,) bool mobility mask (True for aqueous species)."""
        return torch.tensor(transport_mask(self.num_species, self.num_aqueous_species))

    @property
    def mobile_mask(self) -> Tensor:
        """Alias of :attr:`transport_mask` (matches ``mobile_mask_rust``)."""
        return self.transport_mask

    @property
    def mineral_stoichiometry(self) -> Tensor:
        """(num_species, num_minerals) mineral dissolution stoichiometry.

        Returns the same stored tensor on every call (stable object identity)
        so callers can cache by identity / ``id`` without unbounded growth.
        """
        return self._mineral_stoich_t

    @property
    def mineral_molar_masses(self) -> Tensor:
        """(num_minerals,) molar masses [g/mol] (a stable, shared tensor)."""
        return self._mineral_molar_mass_t

    @property
    def rate_consts(self) -> Tensor:
        """(num_minerals,) linear rate constants (a stable, shared tensor)."""
        return self._rate_consts_t

    @property
    def monod_params(self) -> MonodParameters:
        """Monod/inhibition constant matrices (``NaN`` for missing terms).

        Wraps the stored constant tensors so per-step access reuses the same
        base buffers instead of re-materialising them.
        """
        return MonodParameters(
            monod_mat=self._monod_np_t,
            inhib_mat=self._inhib_np_t,
        )

    @property
    def tst_params(self) -> TstParameters:
        """TST matrices in the reference (swapped) layout, see ``kinetic_structures``.

        Wraps the stored constant tensors (see :attr:`monod_params`).
        """
        return TstParameters(
            stoich=self._tst_stoich_t,
            dep=self._tst_dep_t,
            min_eq_const=self._min_eq_const_t,
        )

    @property
    def equilibrium_parameters(self) -> EquilibriumParameters:
        """Equilibrium matrices incl. scipy null space + particular solution."""
        null, x_particular = compute_stoich_null_space(self._eq_stoich, self._log_eq_consts)
        return EquilibriumParameters(
            stoich=torch.tensor(self._eq_stoich, dtype=torch.float64),
            log_eq_consts=torch.tensor(self._log_eq_consts, dtype=torch.float64),
            total=torch.tensor(self._total_mat, dtype=torch.float64),
            stoich_null_space=torch.tensor(null, dtype=torch.float64),
            x_particular=torch.tensor(x_particular, dtype=torch.float64),
            total_names=tuple(self.total_species),
        )

    def get_default_aqueous_initial_state(self, init_conc: float = 1e-6) -> Tensor:
        """Initial aqueous concentrations, all ``init_conc`` (Rust default 1e-6)."""
        return torch.full(
            (self.num_aqueous_species,), float(init_conc), dtype=torch.float64
        )