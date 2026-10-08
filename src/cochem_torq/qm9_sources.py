"""Read original QM9 extended XYZ records without inventing scientific metadata.

The format and property definitions come from the publisher's README for
Ramakrishnan et al., Scientific Data (2014), DOI 10.1038/sdata.2014.22.
Byte identity, format integrity, publisher metadata and predictive accuracy are
separate concerns. These readers do not qualify a calculation or authenticate
a publication merely because the caller supplies a matching digest.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
from typing import Any, NoReturn

QM9_PAPER_DOI = "10.1038/sdata.2014.22"
QM9_DATASET_DOI = "10.6084/m9.figshare.978904_D12"
QM9_METHOD = "B3LYP"
QM9_BASIS = "6-31G(2df,p)"

# The two leading property fields are the literal tag and molecule identifier.
# Units retain the publisher's convention, including the thermochemical fields.
_PROPERTIES = (
    ("A", "GHz", "Rotational constant A"),
    ("B", "GHz", "Rotational constant B"),
    ("C", "GHz", "Rotational constant C"),
    ("mu", "Debye", "Dipole moment"),
    ("alpha", "Bohr^3", "Isotropic polarizability"),
    ("homo", "Hartree", "Energy of Highest occupied molecular orbital (HOMO)"),
    ("lumo", "Hartree", "Energy of Lowest occupied molecular orbital (LUMO)"),
    ("gap", "Hartree", "Gap, difference between LUMO and HOMO"),
    ("r2", "Bohr^2", "Electronic spatial extent"),
    ("zpve", "Hartree", "Zero point vibrational energy"),
    ("U0", "Hartree", "Internal energy at 0 K"),
    ("U", "Hartree", "Internal energy at 298.15 K"),
    ("H", "Hartree", "Enthalpy at 298.15 K"),
    ("G", "Hartree", "Free energy at 298.15 K"),
    ("Cv", "cal/(mol K)", "Heat capacity at 298.15 K"),
)
_NUMBER_BODY = (
    r"[+\-\N{MINUS SIGN}]?(?:\d+(?:\.\d*)?|\.\d+)"
    r"(?:[eEdD][+\-\N{MINUS SIGN}]?\d+)?"
)
_NUMBER = re.compile(_NUMBER_BODY)
_NUMERIC_TOKEN = re.compile(r"(?<![\w.])" + _NUMBER_BODY + r"(?![\w.])")
_FIELD = re.compile(r"\S+")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class QM9SourceError(ValueError):
    """An original source is incomplete, malformed or differs from its pin."""


@dataclass(frozen=True, slots=True)
class QM9NumericDatum:
    """One literal number with its original line and exact text locator.

    Lines and character columns are one-based. Field and numeric-token indices
    are zero-based. Numeric indices match the comparison module's numeric-token
    grammar; for the properties line, the molecule identifier has index zero.
    Printing precision is not a physical uncertainty estimate.
    """

    name: str
    value: float
    literal: str
    unit: str
    source_sha256: str
    source_line_number: int
    source_column_number: int
    source_field_index: int
    source_value_token_index: int
    source_excerpt_utf8: str
    standard_uncertainty: None = None

    def extraction_record(self) -> dict[str, Any]:
        """Return source fields for an explicit AutomatedDatumExtraction."""
        return {
            "raw_source_format": "qm9_extended_xyz",
            "source_sha256": self.source_sha256,
            "source_line_number": self.source_line_number,
            "source_value_literal": self.literal,
            "source_value_token_index": self.source_value_token_index,
            "source_excerpt_utf8": self.source_excerpt_utf8,
            "source_value": self.value,
            "source_unit": self.unit,
            "source_standard_uncertainty": None,
        }


@dataclass(frozen=True, slots=True)
class QM9Atom:
    element: str
    coordinates_angstrom: tuple[QM9NumericDatum, QM9NumericDatum, QM9NumericDatum]
    mulliken_charge_e: QM9NumericDatum


@dataclass(frozen=True, slots=True)
class QM9UnreportedProtocol:
    """Fields not specified in the acquired original XYZ/README."""

    atomic_masses: None = None
    isotope_numbers: None = None
    electronic_state: None = None
    spin_multiplicity: None = None
    engine: None = None
    engine_version: None = None
    gaussian_b3lyp_variant: None = None
    integration_grid: None = None
    harmonic_mode_assignments: None = None


@dataclass(frozen=True, slots=True)
class QM9DerivedEnergyProxy:
    """Explicit arithmetic on published thermochemical fields, not native energy."""

    value_hartree: float
    decimal_literal: str
    source_sha256: str
    operands: tuple[QM9NumericDatum, QM9NumericDatum]
    operation: str = "U0 - zpve"
    quantity: str = "derived_U0_minus_zpve_proxy"
    interpretation: str = (
        "Derived subtraction of published rounded thermochemical values; "
        "not a published total electronic energy or an independently qualified result."
    )


@dataclass(frozen=True, slots=True)
class QM9Record:
    source_sha256: str
    source_text_utf8: str
    format_tag: str
    molecule_id: int
    molecule_id_literal: str
    atoms: tuple[QM9Atom, ...]
    properties: tuple[QM9NumericDatum, ...]
    frequencies: tuple[QM9NumericDatum, ...]
    smiles: tuple[str, str]
    inchi: tuple[str, str]
    undefined_rotational_axes: tuple[str, ...]
    warnings: tuple[str, ...]
    method: str = QM9_METHOD
    basis: str = QM9_BASIS
    unreported_protocol: QM9UnreportedProtocol = QM9UnreportedProtocol()

    def datum(self, name: str) -> QM9NumericDatum:
        for item in self.properties:
            if item.name == name:
                return item
        raise KeyError(f"QM9 has no published property named {name!r}.")

    def derive_u0_minus_zpve_proxy(self) -> QM9DerivedEnergyProxy:
        """Derive only an explicitly named proxy; U0 itself includes ZPVE."""
        u0, zpve = self.datum("U0"), self.datum("zpve")
        # Preserve printed decimal subtraction beyond the default 28 digits.
        # The extra digits cover the finite binary64 exponent range of operands.
        with localcontext() as context:
            context.prec = len(u0.literal) + len(zpve.literal) + 650
            difference = Decimal(_normalized(u0.literal)) - Decimal(
                _normalized(zpve.literal)
            )
        value = float(difference)
        if not math.isfinite(value) or (value == 0 and difference != 0):
            raise QM9SourceError("The derived proxy cannot overflow or underflow.")
        return QM9DerivedEnergyProxy(
            value, str(difference), self.source_sha256, (u0, zpve)
        )


@dataclass(frozen=True, slots=True)
class QM9PublisherEvidence:
    """SHA-bound publisher descriptors; no scientific qualification is implied."""

    metadata_sha256: str
    readme_sha256: str
    archive_url: str
    archive_bytes: int
    archive_md5: str
    paper_doi: str = QM9_PAPER_DOI
    dataset_doi: str = QM9_DATASET_DOI
    method: str = QM9_METHOD
    basis: str = QM9_BASIS
    reuse_license: str = "CC0"


def _checked_bytes(payload: bytes, expected_sha256: str, limit: int) -> str:
    if type(payload) is not bytes or not 0 < len(payload) <= limit:
        raise QM9SourceError("Supply nonempty bounded immutable source bytes.")
    if not isinstance(expected_sha256, str) or not _SHA256.fullmatch(expected_sha256):
        raise QM9SourceError("Source pins require lowercase SHA-256 hexadecimal.")
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise QM9SourceError("Original source SHA-256 differs from its expected pin.")
    return actual


def _normalized(literal: str) -> str:
    return literal.replace("\N{MINUS SIGN}", "-").replace("D", "E").replace("d", "e")


def _number(
    name: str, unit: str, line: str, line_number: int, field_index: int, sha: str
) -> QM9NumericDatum:
    fields = tuple(_FIELD.finditer(line))
    token = fields[field_index]
    literal = token.group()
    if len(literal) > 64 or not _NUMBER.fullmatch(literal):
        raise QM9SourceError(f"Line {line_number} contains a malformed numeric field.")
    try:
        exact = Decimal(_normalized(literal))
    except InvalidOperation as error:
        raise QM9SourceError(
            "Source number exceeds decimal representation limits."
        ) from error
    value = float(exact)
    if not math.isfinite(value) or (value == 0 and exact != 0):
        raise QM9SourceError("A source number cannot overflow or silently underflow.")
    numeric_tokens = tuple(_NUMERIC_TOKEN.finditer(line))
    index = next(
        (i for i, match in enumerate(numeric_tokens) if match.span() == token.span()),
        None,
    )
    if index is None:
        raise QM9SourceError("The numeric field has no unambiguous source locator.")
    return QM9NumericDatum(
        name,
        value,
        literal,
        unit,
        sha,
        line_number,
        token.start() + 1,
        field_index,
        index,
        line,
    )


def _check_inchi_formula(inchi: str, atoms: list[QM9Atom]) -> str | None:
    """Check only an explicit unambiguous formula, without guessing chemistry.

    Component multiplicities and proton layers need a distinct InChI adapter.
    Their original strings remain available with a warning instead of an
    invented elemental interpretation, charge, isotope or state assignment.
    """
    layers = inchi.split("/")
    formula = layers[1]
    if any(layer.startswith("p") for layer in layers[2:]) or not re.fullmatch(
        r"(?:[CHNOF](?:[1-9][0-9]*)?)+", formula
    ):
        return "complex_or_proton_layer_inchi_formula_not_interpreted"
    reported: dict[str, int] = {}
    for element, count in re.findall(r"([CHNOF])([0-9]*)", formula):
        reported[element] = reported.get(element, 0) + (int(count) if count else 1)
    observed: dict[str, int] = {}
    for atom in atoms:
        observed[atom.element] = observed.get(atom.element, 0) + 1
    if reported != observed:
        raise QM9SourceError("InChI formula differs from the explicit source atoms.")
    return None


def parse_qm9_xyz(source: bytes, *, expected_sha256: str) -> QM9Record:
    """Parse one complete original extended XYZ against an explicit byte pin.

    Signed frequencies and all finite signed properties remain unchanged.
    Negative/zero frequencies are warnings, not fabricated replacement modes.
    A zero rotational constant is retained as a raw undefined-axis marker.
    Dataset method labels describe the source protocol, not the current engine.
    """
    sha = _checked_bytes(source, expected_sha256, 64 * 1024)
    try:
        text = source.decode("utf-8")
    except UnicodeDecodeError as error:
        raise QM9SourceError("QM9 original text must be UTF-8.") from error
    if any(ord(c) < 32 and c not in "\t\r\n" for c in text):
        raise QM9SourceError("Unexpected control character in original source.")
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()  # Preserve one optional original line terminator in text.
    lines = [line.removesuffix("\r") for line in lines]
    if any("\r" in line for line in lines):
        raise QM9SourceError("Malformed source line terminator.")
    if not lines or not re.fullmatch(r"[1-9]\d?", lines[0]):
        raise QM9SourceError("The first line requires an explicit positive atom count.")
    count = int(lines[0])
    if count > 29 or len(lines) != count + 5:
        raise QM9SourceError("Incomplete or extra QM9 atom/property/identity rows.")
    header = lines[1].split()
    if (
        len(header) != 17
        or header[0] not in {"gdb", "gdb9"}
        or not re.fullmatch(r"[1-9]\d{0,5}", header[1])
        or int(header[1]) > 133885
    ):
        raise QM9SourceError("The properties row requires its tag, ID and 15 values.")
    properties = tuple(
        _number(name, unit, lines[1], 2, i + 2, sha)
        for i, (name, unit, _) in enumerate(_PROPERTIES)
    )
    atoms: list[QM9Atom] = []
    for i in range(count):
        line_number = i + 3
        line = lines[i + 2]
        fields = line.split()
        if len(fields) != 5 or fields[0] not in {"H", "C", "N", "O", "F"}:
            raise QM9SourceError(
                "Each atom requires H/C/N/O/F, XYZ and Mulliken charge."
            )
        xyz = tuple(
            _number(axis, "Angstrom", line, line_number, j + 1, sha)
            for j, axis in enumerate(("x", "y", "z"))
        )
        atoms.append(
            QM9Atom(
                fields[0],
                (xyz[0], xyz[1], xyz[2]),
                _number("mulliken_charge", "e", line, line_number, 4, sha),
            )
        )
    if sum(atom.element != "H" for atom in atoms) > 9:
        raise QM9SourceError("The GDB-9 source exceeds nine heavy atoms.")
    positions = [tuple(x.value for x in atom.coordinates_angstrom) for atom in atoms]
    if len(set(positions)) != count:
        raise QM9SourceError(
            "Coincident source atoms cannot form a molecular geometry."
        )
    frequency_fields = lines[count + 2].split()
    allowed_mode_counts = (
        {0} if count == 1 else {1} if count == 2 else {3 * count - 5, 3 * count - 6}
    )
    if len(frequency_fields) not in allowed_mode_counts:
        raise QM9SourceError("Incomplete vibrational row: require 3N-5 or 3N-6 modes.")
    frequencies = tuple(
        _number(f"frequency_rank_{i}", "cm^-1", lines[count + 2], count + 3, i, sha)
        for i in range(len(frequency_fields))
    )
    smiles = lines[count + 3].split()
    inchi = lines[count + 4].split()
    if (
        len(smiles) != 2
        or len(inchi) != 2
        or any(
            not item.startswith("InChI=1S/") or len(item) <= len("InChI=1S/")
            for item in inchi
        )
    ):
        raise QM9SourceError(
            "Retain both original and relaxed SMILES/InChI identities."
        )
    axes = tuple(item.name for item in properties[:3] if item.value == 0)
    warnings: list[str] = []
    for item in inchi:
        warning = _check_inchi_formula(item, atoms)
        if warning is not None and warning not in warnings:
            warnings.append(warning)
    if axes:
        warnings.append("zero_rotational_constants_mark_undefined_axes")
    if any(item.value < 0 for item in properties[:3]):
        warnings.append("negative_rotational_constants_in_source")
    if any(item.value < 0 for item in frequencies):
        warnings.append("negative_frequencies_in_source")
    if any(item.value == 0 for item in frequencies):
        warnings.append("zero_frequencies_in_source")
    if smiles[0] != smiles[1] or inchi[0] != inchi[1]:
        warnings.append("original_and_relaxed_connectivity_strings_differ")
    return QM9Record(
        sha,
        text,
        header[0],
        int(header[1]),
        header[1],
        tuple(atoms),
        properties,
        frequencies,
        (smiles[0], smiles[1]),
        (inchi[0], inchi[1]),
        axes,
        tuple(warnings),
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise QM9SourceError("Publisher metadata has duplicate JSON keys.")
        result[key] = value
    return result


def _reject_json_constant(_: str) -> NoReturn:
    raise QM9SourceError("Publisher metadata cannot contain NaN or infinity.")


def validate_qm9_publisher(
    metadata: bytes,
    readme: bytes,
    *,
    expected_metadata_sha256: str,
    expected_readme_sha256: str,
) -> QM9PublisherEvidence:
    """Bind the actual original dataset descriptor and format definitions.

    These checks do not verify archive members from the descriptor alone. The
    caller must separately preserve the archive/member extraction receipt and
    pins; the archive MD5 identifies its compressed bytes, not an XYZ member.
    """
    metadata_sha = _checked_bytes(metadata, expected_metadata_sha256, 256 * 1024)
    readme_sha = _checked_bytes(readme, expected_readme_sha256, 64 * 1024)
    try:
        value = json.loads(
            metadata.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
        documentation = readme.decode("utf-8")
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QM9SourceError(
            "Invalid original publisher metadata/README encoding."
        ) from error
    if not isinstance(value, dict):
        raise QM9SourceError("Publisher metadata must be a JSON object.")
    license_value = value.get("license")
    if (
        type(value.get("id")) is not int
        or value["id"] != 1057646
        or value.get("doi") != QM9_DATASET_DOI
        or value.get("resource_doi") != QM9_PAPER_DOI
        or not isinstance(license_value, dict)
        or license_value.get("name") != "CC0"
        or license_value.get("url")
        != "https://creativecommons.org/publicdomain/zero/1.0/"
    ):
        raise QM9SourceError("Publisher identity, linked paper or CC0 license differs.")
    files = value.get("files")
    if not isinstance(files, list) or len(files) != 1 or not isinstance(files[0], dict):
        raise QM9SourceError("The original dataset requires one archive descriptor.")
    archive = files[0]
    url = "https://ndownloader.figshare.com/files/3195389"
    md5 = "ad1ebd51ee7f5b3a6e32e974e5d54012"
    if (
        type(archive.get("id")) is not int
        or archive["id"] != 3195389
        or archive.get("name") != "dsgdb9nsd.xyz.tar.bz2"
        or type(archive.get("size")) is not int
        or archive["size"] != 86144227
        or archive.get("download_url") != url
        or archive.get("is_link_only") is not False
        or archive.get("computed_md5") != md5
        or archive.get("supplied_md5") != md5
    ):
        raise QM9SourceError("The original compressed archive descriptor differs.")
    normalized = " ".join(documentation.split())
    required_definitions = (
        "133885 neutral organic molecules",
        "molecular geometries were relaxed and properties calculated at the "
        "DFT/B3LYP/6-31G(2df,p) level of theory.",
        "Element type, coordinate (x,y,z) (Angstrom), and Mulliken partial charge (e)",
        "Frequencies (3na-5 or 3na-6)",
        "lowest frequency < 10i cm^-1",
    )
    if any(item not in normalized for item in required_definitions):
        raise QM9SourceError("Publisher README does not establish the source protocol.")
    for index, (name, unit, description) in enumerate(_PROPERTIES, start=3):
        if f"{index} {name} {unit} {description}" not in normalized:
            raise QM9SourceError("Publisher property/unit definitions differ.")
    return QM9PublisherEvidence(metadata_sha, readme_sha, url, 86144227, md5)
