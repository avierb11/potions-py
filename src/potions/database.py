"""Chemical database records mirroring the reference Rust ``database.rs``.

The database is a static description of every aqueous species, mineral, exchange
reaction and kinetic rate law known to the model.  It is parsed once from
``data/default_database.json`` (identical to the Rust ``include_str!`` copy) into
immutable dataclasses.

**Logarithm convention (critical):** every value that is intrinsically a
logarithm is *stored as-is* in the database and only exponentiated by the
reaction network.  Specifically:

- ``TstReaction.rate_constant`` / ``MonodReaction.rate_constant`` hold the
  base-10 *logarithm* of the rate constant at 25 degC (see
  ``ReactionNetwork.rate_consts``, which applies ``10 ** log10_k``).
- ``SecondarySpecies.eq_consts`` are ``log10 K`` values per temperature bin;
  ``ReactionNetwork`` uses ``eq_consts[1]`` (the 25 degC bin).
- ``ExchangeReaction.log10_k_eq`` is already named as a log10 quantity.

Nothing in this module converts log10 -> linear; that lives in
``reaction_network.py`` to mirror the Rust separation exactly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from typing import Any, TypeAlias

#: A paired kinetic reaction, mirroring Rust's ``MineralKineticReaction`` enum.
MineralKineticReaction: TypeAlias = "TstReaction | MonodReaction"


@dataclass(frozen=True)
class PrimaryAqueousSpecies:
    """A primary (master) aqueous species.

    Attributes:
        name:         species name, e.g. ``"DOC"`` or ``"H+"``.
        molar_mass:   molar mass [g/mol].
        charge:       integer-form charge [-].
        dh_size_param: Debye-Hueckel ion-size parameter [angstrom].
    """

    name: str
    molar_mass: float
    charge: float
    dh_size_param: float


@dataclass(frozen=True)
class SecondarySpecies:
    """A secondary species defined by equilibrium reaction(s).

    Attributes:
        name:         species name, e.g. ``"CO2(aq)"``.
        stoichiometry: map ``species name -> coefficient`` of the equilibrium
            reaction producing this species from the primary species.
        eq_consts:    base-10 ``log10 K`` per temperature bin; index 1 (25 degC).
        dh_size_param: Debye-Hueckel ion-size parameter [angstrom].
        charge:       integer-form charge [-].
        molar_mass:   molar mass [g/mol].
    """

    name: str
    stoichiometry: dict[str, float]
    eq_consts: tuple[float, ...]
    dh_size_param: float
    charge: float
    molar_mass: float


@dataclass(frozen=True)
class MineralSpecies:
    """A solid mineral species that participates in kinetic reactions.

    Attributes:
        name:         mineral name, e.g. ``"SOC(s)"``.
        molar_mass:   molar mass [g/mol].
        stoichiometry: map ``species name -> coefficient`` of the mineral's
            dissolution stoichiometry (its own mineral entry is ``-1``).
        eq_consts:    base-10 ``log10 K`` per temperature bin; index 1 (25 degC).
        molar_volume: molar volume [cm^3/mol].
    """

    name: str
    molar_mass: float
    stoichiometry: dict[str, float]
    eq_consts: tuple[float, ...]
    molar_volume: float


@dataclass(frozen=True)
class TstReaction:
    """Transition-state-theory kinetic rate law for a dissolving mineral.

    Attributes:
        mineral_name: name of the mineral this rate law applies to.
        label:        user-supplied label selecting this rate law from the DB.
        rate_constant: base-10 *logarithm* of the rate constant at 25 degC.
        dependence:   map ``species name -> exponent``; used as the ``Q``
            (solubility-product) exponent matrix by the reference model.
    """

    mineral_name: str
    label: str
    rate_constant: float
    dependence: dict[str, float]


@dataclass(frozen=True)
class MonodReaction:
    """Monod kinetics (Michaelis-Menten substrate + inhibition terms).

    Attributes:
        mineral_name: name of the mineral this rate law applies to.
        label:        user-supplied label selecting this rate law from the DB.
        rate_constant: base-10 *logarithm* of the rate constant at 25 degC.
        monod_terms:  map ``species name -> half-saturation constant K``.
        inhib_terms:  map ``species name -> inhibition constant K``.
    """

    mineral_name: str
    label: str
    rate_constant: float
    monod_terms: dict[str, float]
    inhib_terms: dict[str, float]


@dataclass(frozen=True)
class MineralKineticData:
    """The kinetic rate laws selected for the minerals in a reaction network.

    Attributes:
        tst_reactions:   map ``mineral name -> TstReaction``.
        monod_reactions: map ``mineral name -> MonodReaction``.
    """

    tst_reactions: dict[str, TstReaction]
    monod_reactions: dict[str, MonodReaction]


@dataclass(frozen=True)
class ExchangeReaction:
    """A cation-exchange reaction involving the ``X-`` site.

    Attributes:
        name:        exchange species name, e.g. ``"XDOC"``.
        stoichiometry: map ``species name -> coefficient`` (includes ``X-``).
        log10_k_eq:  base-10 logarithm of the exchange equilibrium constant.
        charge:      integer-form charge [-].
    """

    name: str
    stoichiometry: dict[str, float]
    log10_k_eq: float
    charge: float


def _require(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    """Return ``mapping[key]`` raising a ValueError with a useful message."""
    try:
        value: Any = mapping[key]
    except KeyError as exc:
        raise ValueError(f"database entry missing required key {key!r}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"database entry {key!r} must be an object, got {type(value).__name__}")
    return value


def _optional_float(data: dict[str, Any], key: str) -> float:
    """Return a float field that may be absent, defaulting to ``0.0``."""
    value: Any = data.get(key, 0.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key!r} must be a number, got {value!r}")
    return float(value)


def _tst_reaction(data: dict[str, Any]) -> TstReaction:
    """Parse a raw JSON object into a :class:`TstReaction`."""
    return TstReaction(
        mineral_name=str(data["mineral_name"]),
        label=str(data["label"]),
        rate_constant=float(data["rate_constant"]),
        dependence={str(k): float(v) for k, v in data.get("dependence", {}).items()},
    )


def _monod_reaction(data: dict[str, Any]) -> MonodReaction:
    """Parse a raw JSON object into a :class:`MonodReaction`."""
    return MonodReaction(
        mineral_name=str(data["mineral_name"]),
        label=str(data["label"]),
        rate_constant=float(data["rate_constant"]),
        monod_terms={str(k): float(v) for k, v in data.get("monod_terms", {}).items()},
        inhib_terms={str(k): float(v) for k, v in data.get("inhib_terms", {}).items()},
    )


def _nested_reactions(
    data: dict[str, Any],
    parse: Any,
) -> dict[str, dict[str, Any]]:
    """Parse the ``mineral -> label -> reaction`` nested JSON structure."""
    return {str(name): {str(lbl): parse(rxn) for lbl, rxn in _require(data, name).items()} for name in data}


@dataclass(frozen=True)
class ChemicalDatabase:
    """The complete chemical database, mirroring Rust ``ChemicalDatabase``.

    Every mapping is keyed by the relevant species/reaction name.

    Attributes:
        primary_species:   map ``name -> PrimaryAqueousSpecies``.
        secondary_species: map ``name -> SecondarySpecies``.
        mineral_species:   map ``name -> MineralSpecies``.
        exchange_reactions: map ``name -> ExchangeReaction``.
        tst_reactions:     map ``mineral -> (label -> TstReaction)``.
        monod_reactions:   map ``mineral -> (label -> MonodReaction)``.
    """

    primary_species: dict[str, PrimaryAqueousSpecies]
    secondary_species: dict[str, SecondarySpecies]
    mineral_species: dict[str, MineralSpecies]
    exchange_reactions: dict[str, ExchangeReaction]
    tst_reactions: dict[str, dict[str, TstReaction]]
    monod_reactions: dict[str, dict[str, MonodReaction]]

    @staticmethod
    def load_default() -> ChemicalDatabase:
        """Load the default database shipped inside the package.

        The embedded JSON is byte-identical to the Rust ``include_str!("default_database.json")``.
        """
        raw: str = resources.files("potions.data").joinpath("default_database.json").read_text()
        data: dict[str, Any] = json.loads(raw)
        return ChemicalDatabase.from_dict(data)

    @staticmethod
    def from_file(file_path: str) -> ChemicalDatabase:
        """Load a database from a JSON file (Rust ``ChemicalDatabase.from_file``)."""
        from pathlib import Path

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Database file at '{file_path}' does not exist")
        try:
            data: dict[str, Any] = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError("Failed to deconstruct database") from exc
        return ChemicalDatabase.from_dict(data)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> ChemicalDatabase:
        """Parse a raw JSON object (dict) into a :class:`ChemicalDatabase`."""
        primary: dict[str, PrimaryAqueousSpecies] = {}
        for name, entry in data.get("primary_species", {}).items():
            obj: dict[str, Any] = _require(data["primary_species"], name)
            primary[str(name)] = PrimaryAqueousSpecies(
                name=str(name),
                molar_mass=_optional_float(obj, "molar_mass"),
                charge=_optional_float(obj, "charge"),
                dh_size_param=_optional_float(obj, "dh_size_param"),
            )

        secondary: dict[str, SecondarySpecies] = {}
        for name, entry in data.get("secondary_species", {}).items():
            obj = _require(data["secondary_species"], name)
            eq_consts = obj.get("eq_consts", [])
            secondary[str(name)] = SecondarySpecies(
                name=str(name),
                stoichiometry={
                    str(k): float(v) for k, v in obj.get("stoichiometry", {}).items()
                },
                eq_consts=tuple(float(x) for x in eq_consts),
                dh_size_param=_optional_float(obj, "dh_size_param"),
                charge=_optional_float(obj, "charge"),
                molar_mass=_optional_float(obj, "molar_mass"),
            )

        minerals: dict[str, MineralSpecies] = {}
        for name, entry in data.get("mineral_species", {}).items():
            obj = _require(data["mineral_species"], name)
            eq_consts = obj.get("eq_consts", [])
            minerals[str(name)] = MineralSpecies(
                name=str(name),
                molar_mass=_optional_float(obj, "molar_mass"),
                stoichiometry={
                    str(k): float(v) for k, v in obj.get("stoichiometry", {}).items()
                },
                eq_consts=tuple(float(x) for x in eq_consts),
                molar_volume=_optional_float(obj, "molar_volume"),
            )

        exchange: dict[str, ExchangeReaction] = {}
        for name, entry in data.get("exchange_reactions", {}).items():
            obj = _require(data["exchange_reactions"], name)
            exchange[str(name)] = ExchangeReaction(
                name=str(name),
                stoichiometry={
                    str(k): float(v) for k, v in obj.get("stoichiometry", {}).items()
                },
                log10_k_eq=_optional_float(obj, "log10_k_eq"),
                charge=_optional_float(obj, "charge"),
            )

        tst: dict[str, dict[str, TstReaction]] = _nested_reactions(
            data.get("tst_reactions", {}), _tst_reaction
        )
        monod: dict[str, dict[str, MonodReaction]] = _nested_reactions(
            data.get("monod_reactions", {}), _monod_reaction
        )

        return ChemicalDatabase(
            primary_species=primary,
            secondary_species=secondary,
            mineral_species=minerals,
            exchange_reactions=exchange,
            tst_reactions=tst,
            monod_reactions=monod,
        )

    def get_primary_aqueous_species(self, primary_names: list[str]) -> list[PrimaryAqueousSpecies]:
        """Return the requested primary species in order (Rust ``get_primary_aqueous_species``)."""
        species: list[PrimaryAqueousSpecies] = []
        for name in primary_names:
            if name not in self.primary_species:
                raise ValueError(f"Database does not contain primary species '{name}'")
            species.append(self.primary_species[name])
        return species

    def get_mineral_species(self, mineral_names: list[str]) -> list[MineralSpecies]:
        """Return the requested mineral species in order (Rust ``get_mineral_species``)."""
        species: list[MineralSpecies] = []
        for name in mineral_names:
            if name not in self.mineral_species:
                raise ValueError(f"Database does not contain mineral species '{name}'")
            species.append(self.mineral_species[name])
        return species

    def get_secondary_species(self, secondary_names: list[str]) -> list[SecondarySpecies]:
        """Return the requested secondary species in order (Rust ``get_secondary_species``)."""
        species: list[SecondarySpecies] = []
        for name in secondary_names:
            if name not in self.secondary_species:
                raise ValueError(f"Database does not contain secondary species '{name}'")
            species.append(self.secondary_species[name])
        return species

    def get_exchange_reactions(self, species_names: list[str]) -> list[ExchangeReaction]:
        """Return the requested exchange reactions (Rust ``get_exchange_reactions``)."""
        species: list[ExchangeReaction] = []
        for name in species_names:
            if name not in self.exchange_reactions:
                raise ValueError(f"Database does not contain exchange reaction '{name}'")
            species.append(self.exchange_reactions[name])
        return species

    def get_single_mineral_reaction(
        self, mineral: str, label: str
    ) -> tuple[str, MineralKineticReaction]:
        """Return ``(reaction_kind, reaction)`` for one mineral + label.

        ``reaction_kind`` is ``"tst"`` or ``"monod"``.  TST reactions are checked
        first, exactly matching the Rust lookup order.
        """
        if mineral in self.tst_reactions:
            lbls: dict[str, TstReaction] = self.tst_reactions[mineral]
            if label in lbls:
                return "tst", lbls[label]
            raise KeyError(f"Mineral '{mineral}' has no reaction labeled '{label}'")
        if mineral in self.monod_reactions:
            monod_lbls: dict[str, MonodReaction] = self.monod_reactions[mineral]
            if label in monod_lbls:
                return "monod", monod_lbls[label]
            raise KeyError(f"Mineral '{mineral}' has no reaction labeled '{label}'")
        raise KeyError(f"Chemical database does not contain kinetic data for '{mineral}'")

    def get_mineral_reactions(
        self, mineral_names: list[str], labels: list[str]
    ) -> MineralKineticData:
        """Return the kinetic data for paired minerals + labels (Rust ``get_mineral_reactions``)."""
        if len(mineral_names) != len(labels):
            raise ValueError("Must pass the same number of minerals and labels")
        tst_rxns: dict[str, TstReaction] = {}
        monod_rxns: dict[str, MonodReaction] = {}
        for min_name, lbl in zip(mineral_names, labels):
            _kind, rxn = self.get_single_mineral_reaction(min_name, lbl)
            if isinstance(rxn, TstReaction):
                tst_rxns[min_name] = rxn
            else:
                monod_rxns[min_name] = rxn
        return MineralKineticData(tst_reactions=tst_rxns, monod_reactions=monod_rxns)