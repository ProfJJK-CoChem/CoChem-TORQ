"""Identity compatibility checks; no electronic-structure results are supplied."""

import hashlib
import json

import pytest
from pydantic import ValidationError

from cochem_torq.ecosystem import (
    AtomIdentity,
    ConformerHandoff,
    HandoffSource,
    MethodProvenance,
    MoleculeHandoff,
)


def record(tmp_path):
    source = tmp_path / "explicit-geometry.xyz"
    source.write_text(
        "2\nExplicit input only; no calculated observables\nH 0 0 -0.37\nH 0 0 0.37\n"
    )
    molecule = MoleculeHandoff(
        molecule_id="explicit-input",
        atoms=(
            AtomIdentity(atom_id="a", symbol="H"),
            AtomIdentity(atom_id="b", symbol="H"),
        ),
        geometry=((0.0, 0.0, -0.37), (0.0, 0.0, 0.37)),
        geometry_unit="angstrom",
        charge=0,
        multiplicity=1,
    )
    return ConformerHandoff(
        conformer_id="explicit-input",
        molecule=molecule,
        source=HandoffSource(
            producer="explicit_import",
            artifact_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            source_schema="XYZ",
            source_record=str(source),
        ),
        source_method=None,
        energy=None,
        source_convergence="unknown",
        quality_flags=("explicit_input_only",),
    )


def test_outer_label_preserves_legacy_molecule_digest_and_missing_field_load(tmp_path):
    original = record(tmp_path)
    payload = original.molecule.model_dump(mode="json")
    expected = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    assert original.molecule.geometry_sha256 == expected
    assert original.serialization_profile == "cochem.sorted-json/1"
    legacy = original.model_dump(mode="json")
    legacy.pop("serialization_profile")
    loaded = ConformerHandoff.model_validate(legacy)
    assert loaded.molecule.geometry_sha256 == expected
    assert loaded.serialization_profile == "cochem.sorted-json/1"
    assert loaded.energy is None


def test_source_recipe_label_does_not_change_existing_digest_payload():
    # Explicit protocol metadata only; this record supplies no calculated energy.
    payload = dict(
        recipe_id="unexecuted-protocol",
        engine="PySCF",
        engine_version="2.14.0",
        method="HF",
        basis="sto-3g",
        parameters={"scope": "identity arithmetic only", "tolerance": 1e-8},
    )
    expected = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    method = MethodProvenance.from_recipe(**payload)
    assert method.recipe_sha256 == expected
    assert method.serialization_profile == "cochem.sorted-json/1"
    assert "serialization_profile" not in method.model_dump(mode="json")
    assert (
        MethodProvenance.model_validate({**payload, "recipe_sha256": expected})
        == method
    )


def test_profile_cannot_claim_rfc8785_for_legacy_digest(tmp_path):
    altered = record(tmp_path).model_dump(mode="json")
    altered["serialization_profile"] = "RFC8785"
    with pytest.raises(ValidationError, match="serialization_profile"):
        ConformerHandoff.model_validate(altered)
