"""Actual CC0 publisher records plus explicitly damaged copies for parser checks.

No fabricated quantum output or synthetic reference value is used. Source pins
come from the retained full publisher archive/member acquisition receipts.
Mutations test format/integrity handling and are never publication references.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from cochem_torq.qm9_sources import (
    QM9SourceError,
    parse_qm9_xyz,
    validate_qm9_publisher,
)

SAMPLES = (
    Path(__file__).resolve().parents[1] / "benchmarks/published-values/qm9-samples"
)
INDEX = json.loads((SAMPLES / "source-index.json").read_text())
FILES = {entry["path"]: entry for entry in INDEX["files"]}
XYZ_FILES = tuple(name for name in FILES if name.endswith(".xyz"))
WATER = "dsgdb9nsd_000003.xyz"


def _source(name: str = WATER) -> bytes:
    payload = (SAMPLES / name).read_bytes()
    assert len(payload) == FILES[name]["bytes"]
    assert hashlib.sha256(payload).hexdigest() == FILES[name]["sha256"]
    return payload


def _parse(payload: bytes):
    # Explicitly mutated inputs get their own digests for grammar checks; a
    # mutated digest is never represented as the original publisher's digest.
    return parse_qm9_xyz(payload, expected_sha256=hashlib.sha256(payload).hexdigest())


def _mutated_field(line: int, field: int, value: str) -> bytes:
    rows = _source().decode().split("\n")
    fields = rows[line - 1].split()
    fields[field] = value
    rows[line - 1] = "\t".join(fields)
    return "\n".join(rows).encode()


def _publisher(metadata: bytes | None = None, readme: bytes | None = None):
    metadata = _source("publisher-metadata.json") if metadata is None else metadata
    readme = _source("publisher-readme.txt") if readme is None else readme
    return validate_qm9_publisher(
        metadata,
        readme,
        expected_metadata_sha256=hashlib.sha256(metadata).hexdigest(),
        expected_readme_sha256=hashlib.sha256(readme).hexdigest(),
    )


@pytest.mark.parametrize("name", XYZ_FILES)
def test_all_twenty_original_archive_members_retain_exact_numeric_locators(name):
    assert len(XYZ_FILES) == 20
    payload = _source(name)
    record = parse_qm9_xyz(payload, expected_sha256=FILES[name]["sha256"])
    assert record.source_text_utf8.encode() == payload
    assert record.method == "B3LYP" and record.basis == "6-31G(2df,p)"
    assert len(record.properties) == 15
    rows = payload.decode().split("\n")
    numbers = list(record.properties) + list(record.frequencies)
    for atom in record.atoms:
        numbers.extend(atom.coordinates_angstrom)
        numbers.append(atom.mulliken_charge_e)
    for number in numbers:
        line = rows[number.source_line_number - 1]
        assert number.source_excerpt_utf8 == line
        assert line[number.source_column_number - 1 :].startswith(number.literal)
        assert line.split()[number.source_field_index] == number.literal
        assert number.standard_uncertainty is None
        assert number.source_sha256 == record.source_sha256
    assert record.unreported_protocol.atomic_masses is None
    assert record.unreported_protocol.isotope_numbers is None
    assert record.unreported_protocol.electronic_state is None
    assert record.unreported_protocol.spin_multiplicity is None
    assert record.unreported_protocol.gaussian_b3lyp_variant is None
    assert record.unreported_protocol.harmonic_mode_assignments is None


@pytest.mark.parametrize(
    "name, composition, rotation, frequencies",
    [
        (
            "dsgdb9nsd_000001.xyz",
            "CHHHH",
            ("157.7118", "157.70997", "157.70699"),
            (
                "1341.307",
                "1341.3284",
                "1341.365",
                "1562.6731",
                "1562.7453",
                "3038.3205",
                "3151.6034",
                "3151.6788",
                "3151.7078",
            ),
        ),
        (
            "dsgdb9nsd_000002.xyz",
            "NHHH",
            ("293.60975", "293.54111", "191.39397"),
            (
                "1103.8733",
                "1684.1158",
                "1684.3072",
                "3458.7145",
                "3575.2414",
                "3575.3343",
            ),
        ),
        (
            WATER,
            "OHH",
            ("799.58812", "437.90386", "282.94545"),
            ("1671.4222", "3803.6305", "3907.698"),
        ),
        (
            "dsgdb9nsd_000006.xyz",
            "COHH",
            ("285.48839", "38.9823", "34.29892"),
            (
                "1201.9838",
                "1272.5136",
                "1544.4691",
                "1846.4364",
                "2882.3231",
                "2929.819",
            ),
        ),
    ],
)
def test_named_small_molecules_match_the_actual_published_fields(
    name, composition, rotation, frequencies
):
    record = _parse(_source(name))
    assert "".join(atom.element for atom in record.atoms) == composition
    assert tuple(record.datum(axis).literal for axis in ("A", "B", "C")) == rotation
    assert tuple(item.literal for item in record.frequencies) == frequencies
    assert all(record.datum(axis).unit == "GHz" for axis in ("A", "B", "C"))
    assert all(item.unit == "cm^-1" for item in record.frequencies)


def test_water_geometry_identity_and_thermochemical_energy_are_not_replaced():
    record = _parse(_source())
    assert tuple(x.literal for x in record.atoms[0].coordinates_angstrom) == (
        "-0.0343604951",
        "0.9775395708",
        "0.0076015923",
    )
    assert record.molecule_id_literal == "3"
    assert record.smiles == ("O", "O")
    assert record.inchi == ("InChI=1S/H2O/h1H2", "InChI=1S/H2O/h1H2")
    assert record.datum("U0").literal == "-76.404702"
    assert record.datum("zpve").literal == "0.021375"
    with pytest.raises(KeyError):
        record.datum("total_electronic_energy")
    proxy = record.derive_u0_minus_zpve_proxy()
    assert proxy.decimal_literal == "-76.426077"
    assert proxy.quantity == "derived_U0_minus_zpve_proxy"
    assert proxy.operands == (record.datum("U0"), record.datum("zpve"))
    assert "not a published total electronic energy" in proxy.interpretation
    with pytest.raises(FrozenInstanceError):
        record.molecule_id = 42


def test_actual_linear_acetylene_raw_zero_is_an_undefined_axis_not_finite_a():
    record = _parse(_source("dsgdb9nsd_000004.xyz"))
    assert record.datum("A").literal == "0."
    assert record.datum("A").value == 0.0
    assert record.undefined_rotational_axes == ("A",)
    assert "zero_rotational_constants_mark_undefined_axes" in record.warnings
    assert len(record.frequencies) == 3 * len(record.atoms) - 5


def test_derived_proxy_does_not_publish_nonfinite_subtraction():
    rows = _source().decode().split("\n")
    properties = rows[1].split()
    properties[11:13] = ["1e308", "-1e308"]
    rows[1] = "\t".join(properties)
    record = _parse("\n".join(rows).encode())
    with pytest.raises(QM9SourceError, match="derived proxy"):
        record.derive_u0_minus_zpve_proxy()


def test_signed_zero_and_negative_damaged_copy_modes_are_preserved_not_fabricated():
    record = _parse(_mutated_field(6, 0, "-1671.4222"))
    assert record.frequencies[0].literal == "-1671.4222"
    assert record.frequencies[0].value == -1671.4222
    assert "negative_frequencies_in_source" in record.warnings
    record = _parse(_mutated_field(6, 0, "-0."))
    assert record.frequencies[0].literal == "-0."
    assert "zero_frequencies_in_source" in record.warnings
    assert _parse(_mutated_field(2, 2, "-799.58812")).datum("A").value == -799.58812


def test_numeric_record_is_compatible_with_exact_automated_extraction_tokens():
    record = _parse(_source())
    numeric = re.compile(
        r"(?<![\w.])[+\-\N{MINUS SIGN}]?(?:\d+(?:\.\d*)?|\.\d+)"
        r"(?:[eEdD][+\-\N{MINUS SIGN}]?\d+)?(?![\w.])"
    )
    for name, index in (("A", 1), ("B", 2), ("C", 3)):
        datum = record.datum(name)
        extracted = datum.extraction_record()
        assert extracted["source_value_token_index"] == index
        assert numeric.findall(datum.source_excerpt_utf8)[index] == datum.literal
    for rank, datum in enumerate(record.frequencies):
        assert datum.source_line_number == 6
        assert datum.source_value_token_index == rank
        assert numeric.findall(datum.source_excerpt_utf8)[rank] == datum.literal


@pytest.mark.parametrize("literal", ["NaN", "inf", "1e999", "1e-999", "1*^3", ""])
def test_malformed_nonfinite_or_unrepresentable_source_numbers_are_rejected(literal):
    with pytest.raises(QM9SourceError):
        _parse(_mutated_field(3, 1, literal))


@pytest.mark.parametrize(
    "line,field,replacement",
    [
        (2, 0, "substituted"),
        (2, 1, "0"),
        (2, 1, "133886"),
        (3, 0, "Ne"),
        (3, 0, "18O"),
        (8, 0, "missing-InChI"),
    ],
)
def test_wrong_dataset_identity_element_or_identity_row_is_rejected(
    line, field, replacement
):
    with pytest.raises(QM9SourceError):
        _parse(_mutated_field(line, field, replacement))


def test_swapped_inchi_formula_is_rejected_even_if_both_identity_strings_agree():
    rows = _source().decode().split("\n")
    rows[7] = "InChI=1S/CH4/h1H4\tInChI=1S/CH4/h1H4"
    with pytest.raises(QM9SourceError, match="InChI formula"):
        _parse("\n".join(rows).encode())


@pytest.mark.parametrize("identity", ["InChI=1S/H2.O", "InChI=1S/H3O/p-1"])
def test_complex_formula_or_proton_layer_is_retained_with_uninterpreted_warning(
    identity,
):
    rows = _source().decode().split("\n")
    rows[7] = f"{identity}\t{identity}"
    record = _parse("\n".join(rows).encode())
    assert record.inchi == (identity, identity)
    assert "complex_or_proton_layer_inchi_formula_not_interpreted" in record.warnings
    assert record.unreported_protocol.electronic_state is None
    assert record.unreported_protocol.isotope_numbers is None


@pytest.mark.parametrize("damage", ["count", "property", "atom", "mode", "tail"])
def test_incomplete_or_extra_source_rows_never_get_missing_value_defaults(damage):
    rows = _source().decode().split("\n")
    if damage == "count":
        rows[0] = "4"
    elif damage == "property":
        rows[1] = "\t".join(rows[1].split()[:-1])
    elif damage == "atom":
        rows[2] = "\t".join(rows[2].split()[:-1])
    elif damage == "mode":
        rows[5] = "\t".join(rows[5].split()[:-1])
    else:
        rows.append("extra source row")
    with pytest.raises(QM9SourceError):
        _parse("\n".join(rows).encode())


def test_coincident_atom_coordinates_and_invalid_utf8_are_rejected():
    rows = _source().decode().split("\n")
    fields = rows[3].split()
    fields[1:4] = rows[2].split()[1:4]
    rows[3] = "\t".join(fields)
    with pytest.raises(QM9SourceError, match="Coincident"):
        _parse("\n".join(rows).encode())
    with pytest.raises(QM9SourceError, match="UTF-8"):
        _parse(_source() + b"\xff")


def test_actual_publisher_metadata_and_readme_establish_protocol_and_reuse():
    evidence = _publisher()
    assert evidence.paper_doi == "10.1038/sdata.2014.22"
    assert evidence.dataset_doi == "10.6084/m9.figshare.978904_D12"
    assert evidence.reuse_license == "CC0"
    assert evidence.archive_bytes == 86144227
    assert evidence.archive_md5 == "ad1ebd51ee7f5b3a6e32e974e5d54012"
    assert evidence.method == "B3LYP" and evidence.basis == "6-31G(2df,p)"


@pytest.mark.parametrize(
    "key,value",
    [
        ("id", 978904),
        ("doi", "10.6084/m9.figshare.other"),
        ("resource_doi", "10.1038/other"),
    ],
)
def test_wrong_publisher_article_or_paper_is_not_silently_accepted(key, value):
    metadata = json.loads(_source("publisher-metadata.json"))
    metadata[key] = value
    with pytest.raises(QM9SourceError, match="Publisher identity"):
        _publisher(json.dumps(metadata).encode())


@pytest.mark.parametrize(
    "key,value",
    [
        ("size", 0),
        ("computed_md5", "0" * 32),
        ("download_url", "https://example.com/file"),
    ],
)
def test_wrong_original_archive_descriptor_cannot_verify_member_values(key, value):
    metadata = json.loads(_source("publisher-metadata.json"))
    metadata["files"][0][key] = value
    with pytest.raises(QM9SourceError, match="archive descriptor"):
        _publisher(json.dumps(metadata).encode())


def test_duplicate_metadata_keys_changed_reuse_and_changed_units_are_rejected():
    payload = _source("publisher-metadata.json").rstrip()
    with pytest.raises(QM9SourceError, match="duplicate"):
        _publisher(payload[:-1] + b', "id":1057646}')
    metadata = json.loads(payload)
    metadata["license"]["name"] = "unverified"
    with pytest.raises(QM9SourceError, match="CC0"):
        _publisher(json.dumps(metadata).encode())
    changed = _source("publisher-readme.txt").replace(b"GHz", b"MHz")
    with pytest.raises(QM9SourceError, match="property/unit"):
        _publisher(readme=changed)


def test_byte_mismatch_rejects_even_a_well_formed_source_and_readme():
    with pytest.raises(QM9SourceError, match="SHA-256"):
        parse_qm9_xyz(_source(), expected_sha256="0" * 64)
    with pytest.raises(QM9SourceError, match="SHA-256"):
        validate_qm9_publisher(
            _source("publisher-metadata.json"),
            _source("publisher-readme.txt") + b" ",
            expected_metadata_sha256=FILES["publisher-metadata.json"]["sha256"],
            expected_readme_sha256=FILES["publisher-readme.txt"]["sha256"],
        )
