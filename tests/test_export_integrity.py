"""Export integrity using real serialization and analytic/mathematical inputs.

No quantum-program output is impersonated. The AtomicResult fixture is the
explicit analytic hydrogenic 1s eigenenergy (infinite nuclear mass), evaluated
here and labelled as an analytic result; it is not an engine integration test.
"""

import copy
import json
import math
import xml.etree.ElementTree as ET

import h5py
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import zstandard as zstd
from qcelemental.models import AtomicResult

from Libraries.cochem_torq_export import (
    KraitchmanZPVEWarning,
    TorqExporter,
    calculate_kraitchman_coords,
    canonical_json_dumps,
    export_qcschema,
    export_torq_result_bundle,
    generate_pgopher_skeleton,
)
from Libraries.cochem_torq_spcat import PickettSPCATRunner


@pytest.fixture
def analytic_hydrogen_result():
    """Compute the nonrelativistic hydrogen 1s eigenenergy in atomic units."""
    nuclear_charge, principal_n = 1, 1
    eigenenergy = -(nuclear_charge**2) / (2 * principal_n**2)
    return {
        "schema_name": "qcschema_output",
        "schema_version": 1,
        "molecule": {
            "symbols": ["H"],
            "geometry": [0.0, 0.0, 0.0],
            "molecular_charge": 0,
            "molecular_multiplicity": 2,
            "fix_com": True,
            "fix_orientation": True,
        },
        "driver": "energy",
        "model": {"method": "exact-nonrelativistic-hydrogenic-1s", "basis": None},
        "properties": {"return_energy": eigenenergy},
        "return_result": eigenenergy,
        "success": True,
        "provenance": {
            "creator": "TORQ analytic validation",
            "version": "1",
            "routine": "tests.test_export_integrity.analytic_hydrogen_result",
        },
        "extras": {
            "model_assumptions": "infinite nuclear mass; no relativistic corrections"
        },
    }


def test_atomic_result_roundtrip_validated_by_reference_model(
    tmp_path, analytic_hydrogen_result
):
    output = tmp_path / "hydrogen.json"
    export_qcschema(analytic_hydrogen_result, str(output))
    payload = json.loads(output.read_text())
    validated = AtomicResult(**payload)
    assert validated.return_result == -0.5
    assert validated.provenance.creator == "TORQ analytic validation"
    assert set(payload["properties"]) == {"return_energy"}


@pytest.mark.parametrize(
    "field", ["return_result", "provenance", "molecule", "properties"]
)
def test_incomplete_atomic_result_rejected_before_writing(
    tmp_path, analytic_hydrogen_result, field
):
    result = copy.deepcopy(analytic_hydrogen_result)
    del result[field]
    output = tmp_path / "existing.json"
    output.write_text("preserve existing export")
    with pytest.raises(ValueError, match="Complete AtomicResult"):
        export_qcschema(result, str(output))
    assert output.read_text() == "preserve existing export"


@pytest.mark.parametrize(
    "field", ["molecular_charge", "molecular_multiplicity", "geometry"]
)
def test_molecule_metadata_not_inferred(tmp_path, analytic_hydrogen_result, field):
    del analytic_hydrogen_result["molecule"][field]
    with pytest.raises(ValueError, match=f"molecule.{field}"):
        export_qcschema(analytic_hydrogen_result, str(tmp_path / "out.json"))


def test_missing_energy_not_zero(tmp_path, analytic_hydrogen_result):
    analytic_hydrogen_result["properties"] = {}
    with pytest.raises(ValueError, match="return_energy"):
        export_qcschema(analytic_hydrogen_result, str(tmp_path / "out.json"))


def test_contradictory_energy_rejected(tmp_path, analytic_hydrogen_result):
    analytic_hydrogen_result["return_result"] = 0.0
    with pytest.raises(ValueError, match="must match"):
        export_qcschema(analytic_hydrogen_result, str(tmp_path / "out.json"))


def test_invalid_atomic_geometry_rejected_by_reference_schema(
    tmp_path, analytic_hydrogen_result
):
    analytic_hydrogen_result["molecule"]["geometry"] = [0.0, 0.0]
    with pytest.raises(Exception):
        export_qcschema(analytic_hydrogen_result, str(tmp_path / "out.json"))
    assert not (tmp_path / "out.json").exists()


def test_composite_bundle_has_own_identity_and_preserves_absence(tmp_path):
    result = {"harmonic": {"status": "unavailable", "frequencies_cm1": None}}
    output = tmp_path / "bundle.json"
    export_torq_result_bundle(result, str(output))
    payload = json.loads(output.read_text())
    assert payload["schema_name"] == "cochem_torq_result_bundle"
    assert payload["result"] == result
    assert "return_energy" not in payload


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_results_cannot_be_serialized(value):
    with pytest.raises(ValueError):
        canonical_json_dumps({"mathematical_input": value})


def test_pgopher_missing_properties_rejected(tmp_path):
    # Empty storage schema tests metadata handling; it is not a spectral catalog.
    parquet = tmp_path / "empty-schema.parquet"
    pq.write_table(pa.table({"frequency": pa.array([], type=pa.float64())}), parquet)
    with pytest.raises(ValueError, match="Explicit rotational constants"):
        generate_pgopher_skeleton(parquet)
    with pytest.raises(ValueError, match="Explicit principal-axis dipoles"):
        generate_pgopher_skeleton(parquet, rotational_constants=(3.0, 2.0, 1.0))
    assert not list(tmp_path.glob("*.pgo"))


def test_pgopher_explicit_zero_dipole_preserved_without_ground_state_claim(tmp_path):
    parquet = tmp_path / "empty-schema.parquet"
    pq.write_table(pa.table({"frequency": pa.array([], type=pa.float64())}), parquet)
    output = generate_pgopher_skeleton(
        parquet,
        molecule_name="mathematical_serialization_input",
        rotational_constants=(3.0, 2.0, 1.0),
        dipoles={"mu_a": 0.0, "mu_b": 1.0, "mu_c": 0.0},
    )
    root = ET.parse(output).getroot()
    assert root.find(".//AsymmetricTop").attrib["Name"] == "unspecified"
    values = {
        item.attrib["Name"]: float(item.attrib["Value"])
        for item in root.findall(".//Parameter")
    }
    assert values["mu_a"] == 0.0
    assert values["mu_b"] == 1.0


def test_kraitchman_missing_moments_and_singular_inputs_rejected():
    with pytest.raises(ValueError, match="missing principal moment"):
        calculate_kraitchman_coords({"Ia": 1}, (1, 2, 3), 20, 1)
    with pytest.raises(ValueError, match="near-symmetric denominator"):
        calculate_kraitchman_coords((10, 10, 20), (10.1, 10.1, 20.2), 20, 1)


def test_kraitchman_imaginary_coordinate_kept_unavailable():
    with pytest.warns(KraitchmanZPVEWarning, match="coordinate unavailable"):
        result = calculate_kraitchman_coords((10, 25, 30), (11.5, 25.1, 30.1), 60, 1)
    assert result["radicands"]["a"] < 0
    assert result["coords"]["a"] is None
    assert result["costain_errors"]["a"] is None
    assert result["quality_flags"]["a"] == "imaginary_coordinate"


def test_tensor_metadata_does_not_claim_undetermined_symmetry(tmp_path):
    exporter = TorqExporter(str(tmp_path))
    metadata = exporter._generate_metadata("mathematical_input", {"values": [1.0]})
    assert "symmetry_group" not in metadata
    assert "LAM_TRIGGER_REQUIRED" not in metadata


def test_tensor_export_verification_detects_recompressed_payload_tampering(tmp_path):
    h5_path = tmp_path / "mathematical_data.h5"
    with h5py.File(h5_path, "w") as handle:
        handle.create_dataset("integer_sequence", data=[1, 2, 3])
    exporter = TorqExporter(str(tmp_path))
    output = exporter.export_tensor_to_zstd(str(h5_path))
    assert exporter.verify_export(output)[0] is True
    from pathlib import Path

    path = Path(output)
    payload = json.loads(zstd.ZstdDecompressor().decompress(path.read_bytes()))
    payload["tensor_data"]["integer_sequence"][0] = 99
    path.write_bytes(zstd.ZstdCompressor().compress(json.dumps(payload).encode()))
    assert exporter.verify_export(output) == (False, None)


def test_spcat_explicit_missing_executable_does_not_select_another(tmp_path):
    with pytest.raises(FileNotFoundError, match="Requested SPCAT executable missing"):
        PickettSPCATRunner(tmp_path / "not-installed")


@pytest.mark.parametrize("contents", ["", "\n", "invalid short catalog row\n"])
def test_spcat_malformed_catalog_cannot_become_empty_success(tmp_path, contents):
    path = tmp_path / "invalid.cat"
    path.write_text(contents)
    with pytest.raises(ValueError, match="no supported transitions|line 1"):
        PickettSPCATRunner().parse_cat_file(path)


def test_legacy_spcat_writers_do_not_invent_uncertainty_or_partition(tmp_path):
    runner = PickettSPCATRunner()
    with pytest.raises(NotImplementedError, match="unvalidated"):
        runner.write_var_file(tmp_path / "rejected.var", None)
    with pytest.raises(NotImplementedError, match="partition function"):
        runner.write_int_file(tmp_path / "rejected.int", None)
    assert not list(tmp_path.iterdir())
