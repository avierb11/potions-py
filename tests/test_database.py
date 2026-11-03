"""Tests for :mod:`potions.database` (JSON chemical database loading + lookups)."""

from __future__ import annotations

import json

import pytest

from potions.database import (
    ChemicalDatabase,
    ExchangeReaction,
    MineralKineticData,
    MineralSpecies,
    MonodReaction,
    PrimaryAqueousSpecies,
    SecondarySpecies,
    TstReaction,
)


@pytest.fixture(scope="module")
def db() -> ChemicalDatabase:
    """The bundled default chemical database (loaded once)."""
    return ChemicalDatabase.load_default()


class TestLoadDefault:
    def test_contains_expected_species(self, db: ChemicalDatabase) -> None:
        assert "DOC" in db.primary_species
        assert "HCO3-" in db.primary_species
        assert "H+" in db.primary_species
        assert "SOC(s)" in db.mineral_species
        assert "CO2(aq)" in db.secondary_species
        assert "XDOC" in db.exchange_reactions

    def test_species_dataclass_fields(self, db: ChemicalDatabase) -> None:
        doc: PrimaryAqueousSpecies = db.primary_species["DOC"]
        assert isinstance(doc, PrimaryAqueousSpecies)
        assert doc.name == "DOC"
        assert doc.molar_mass > 0.0

    def test_mineral_lookup(self, db: ChemicalDatabase) -> None:
        soc: MineralSpecies = db.mineral_species["SOC(s)"]
        assert isinstance(soc, MineralSpecies)
        assert soc.name == "SOC(s)"
        assert soc.eq_consts


class TestLookups:
    def test_get_primary_order_preserved(self, db: ChemicalDatabase) -> None:
        names: list[str] = ["HCO3-", "DOC", "H+"]
        out: list[PrimaryAqueousSpecies] = db.get_primary_aqueous_species(names)
        assert [s.name for s in out] == names

    def test_get_primary_missing_raises(self, db: ChemicalDatabase) -> None:
        with pytest.raises(ValueError, match="primary species"):
            db.get_primary_aqueous_species(["NOT_A_SPECIES"])

    def test_get_mineral_missing_raises(self, db: ChemicalDatabase) -> None:
        with pytest.raises(ValueError, match="mineral species"):
            db.get_mineral_species(["NOT_A_MINERAL"])

    def test_get_secondary_missing_raises(self, db: ChemicalDatabase) -> None:
        with pytest.raises(ValueError, match="secondary species"):
            db.get_secondary_species(["NOT_A_SPECIES"])

    def test_get_exchange_missing_raises(self, db: ChemicalDatabase) -> None:
        with pytest.raises(ValueError, match="exchange reaction"):
            db.get_exchange_reactions(["NOT_A_SPECIES"])

    def test_get_secondary_species(self, db: ChemicalDatabase) -> None:
        out: list[SecondarySpecies] = db.get_secondary_species(["CO2(aq)", "CO3--"])
        assert [s.name for s in out] == ["CO2(aq)", "CO3--"]

    def test_get_exchange_reactions(self, db: ChemicalDatabase) -> None:
        out: list[ExchangeReaction] = db.get_exchange_reactions(["XDOC"])
        assert out[0].name == "XDOC"
        assert "X-" in out[0].stoichiometry


class TestMineralReactions:
    def test_single_monod_reaction(self, db: ChemicalDatabase) -> None:
        kind: str
        rxn: TstReaction | MonodReaction
        kind, rxn = db.get_single_mineral_reaction("SOC(s)", "test")
        # In the bundled database ``SOC(s)``/"test" is a Monod reaction, so the
        # TST map is checked first and misses, then the Monod map hits.
        assert kind == "monod"
        assert isinstance(rxn, MonodReaction)
        assert rxn.mineral_name == "SOC(s)"
        assert rxn.label == "test"
        assert rxn.monod_terms

    def test_single_unknown_mineral_raises(self, db: ChemicalDatabase) -> None:
        with pytest.raises(KeyError, match="does not contain kinetic data"):
            db.get_single_mineral_reaction("UNOBTAINIUM", "test")

    def test_single_unknown_label_raises(self, db: ChemicalDatabase) -> None:
        with pytest.raises(KeyError, match="no reaction labeled"):
            db.get_single_mineral_reaction("SOC(s)", "nonexistent")

    def test_get_mineral_reactions(self, db: ChemicalDatabase) -> None:
        data: MineralKineticData = db.get_mineral_reactions(["SOC(s)"], ["test"])
        assert "SOC(s)" in data.monod_reactions
        assert data.tst_reactions == {}

    def test_get_mineral_reactions_length_mismatch_raises(self, db: ChemicalDatabase) -> None:
        with pytest.raises(ValueError, match="same number"):
            db.get_mineral_reactions(["SOC(s)"], [])


class TestFromDict:
    def test_minimal_database(self) -> None:
        raw: dict = {
            "primary_species": {"A": {"molar_mass": 10.0, "charge": -1.0}},
            "secondary_species": {
                "B": {"stoichiometry": {"A": 2.0}, "eq_consts": [1.0, 2.0]}
            },
            "mineral_species": {"M": {"molar_mass": 20.0}},
            "exchange_reactions": {},
            "tst_reactions": {},
            "monod_reactions": {},
        }
        db: ChemicalDatabase = ChemicalDatabase.from_dict(raw)
        assert db.primary_species["A"].molar_mass == 10.0
        assert db.secondary_species["B"].eq_consts == (1.0, 2.0)
        assert db.mineral_species["M"].molar_mass == 20.0

    def test_missing_required_field_raises(self) -> None:
        raw: dict = {"primary_species": {"A": 5.0}}  # not an object
        with pytest.raises(ValueError, match="must be an object"):
            ChemicalDatabase.from_dict(raw)

    def test_non_numeric_field_raises(self) -> None:
        raw: dict = {"primary_species": {"A": {"charge": "plus"}}}
        with pytest.raises(ValueError, match="must be a number"):
            ChemicalDatabase.from_dict(raw)


class TestFromFile:
    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            ChemicalDatabase.from_file("/nonexistent/path/db.json")

    def test_roundtrip_via_file(self, tmp_path) -> None:
        raw: dict = {"primary_species": {"A": {"molar_mass": 1.0}}}
        path = tmp_path / "db.json"
        path.write_text(json.dumps(raw))
        db: ChemicalDatabase = ChemicalDatabase.from_file(str(path))
        assert db.primary_species["A"].name == "A"

    def test_invalid_json_raises(self, tmp_path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{not json")
        with pytest.raises(ValueError, match="Failed to deconstruct"):
            ChemicalDatabase.from_file(str(path))
