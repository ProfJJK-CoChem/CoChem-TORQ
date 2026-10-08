"""Strict, file-based BASE/TOPOS boundaries for TORQ.

The sibling modules run in separate environments. This module imports neither
provider and never infers spin, atom mapping, convergence, or a method from a
filename, tier number, XYZ comment, or a successful serialization operation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
from filelock import FileLock
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    field_validator,
    model_validator,
)

from .units import ANGSTROM_BOHR

MAX_ARTIFACT_BYTES = 16 * 1024 * 1024
MAX_ATOMS = 2000
ECOSYSTEM_DIGEST_PROFILE = "cochem.sorted-json/1"
# IUPAC element identifiers, in atomic-number order; no inferred atomic masses.
ELEMENTS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La "
    "Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi "
    "Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt "
    "Ds Rg Cn Nh Fl Mc Lv Ts Og"
).split()
ATOMIC_NUMBERS = {symbol: index for index, symbol in enumerate(ELEMENTS, start=1)}


class HandoffError(ValueError):
    """The boundary lacks valid, consistent, integrity-checked information."""


class StrictRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, revalidate_instances="always"
    )


def _canonical_json(value: Any) -> bytes:
    """Preserve the declared legacy identity bytes used by BASE/TOPOS v1.

    This is not RFC8785. Changing this algorithm requires a versioned contract
    and requalified consumers; adding a profile label must not alter old hashes.
    """
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or not 0 < path.stat().st_size <= MAX_ARTIFACT_BYTES:
        raise HandoffError(
            "The manifest must be a nonempty regular file below the 16 MiB limit"
        )

    def reject_constant(value: str) -> None:
        raise HandoffError(f"Nonfinite JSON value {value} is prohibited")

    def unique_keys(items: list[tuple[str, Any]]) -> dict[str, Any]:
        record: dict[str, Any] = {}
        for key, value in items:
            if key in record:
                raise HandoffError(f"Duplicate JSON key {key!r} is prohibited")
            record[key] = value
        return record

    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=unique_keys,
    )
    if not isinstance(value, dict):
        raise HandoffError("A handoff manifest must be a JSON object")
    try:
        _canonical_json(value)
    except ValueError as exc:
        raise HandoffError("All numeric JSON values must remain finite") from exc
    return value


class AtomIdentity(StrictRecord):
    atom_id: str = Field(
        min_length=1, max_length=120, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$"
    )
    symbol: str
    isotope_mass_number: StrictInt | None = Field(default=None, ge=1)

    @field_validator("symbol")
    @classmethod
    def valid_symbol(cls, value: str) -> str:
        if value not in ATOMIC_NUMBERS:
            raise ValueError(
                "Use a canonical element symbol and an explicit isotope_mass_number"
            )
        return value

    @model_validator(mode="after")
    def plausible_isotope_identifier(self) -> AtomIdentity:
        if (
            self.isotope_mass_number is not None
            and self.isotope_mass_number < ATOMIC_NUMBERS[self.symbol]
        ):
            raise ValueError(
                "An isotope mass number cannot be smaller than its proton count"
            )
        return self


class MoleculeHandoff(StrictRecord):
    """Row order is defined by stable atom IDs, with explicit charge and spin.

    An absent isotope means unspecified isotope, never an average-mass isotope.
    Electron parity only rejects impossible states; it does not identify ground state.
    """

    molecule_id: str = Field(min_length=1, max_length=200)
    atoms: tuple[AtomIdentity, ...] = Field(min_length=1, max_length=MAX_ATOMS)
    geometry: tuple[tuple[float, float, float], ...]
    geometry_unit: Literal["angstrom", "bohr"]
    charge: StrictInt
    multiplicity: StrictInt = Field(ge=1)
    topology_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @field_validator("geometry", mode="before")
    @classmethod
    def numeric_coordinates(cls, value: Any) -> Any:
        if not isinstance(value, (list, tuple, np.ndarray)):
            raise ValueError("Coordinates must be an explicitly supplied N by 3 array")
        for row in value:
            if not isinstance(row, (list, tuple, np.ndarray)):
                raise ValueError("Coordinates must have three numbers per atom")
            for coordinate in row:
                if isinstance(coordinate, (bool, str)) or not isinstance(
                    coordinate, (int, float, np.number)
                ):
                    raise ValueError(
                        "Coordinates must contain numbers, excluding "
                        "booleans and strings"
                    )
        return value

    @model_validator(mode="after")
    def validate_identity_and_state(self) -> MoleculeHandoff:
        ids = [atom.atom_id for atom in self.atoms]
        if len(set(ids)) != len(ids):
            raise ValueError(
                "Stable atom IDs must be unique; provide an explicit row mapping"
            )
        positions = np.asarray(self.geometry, dtype=np.float64)
        if positions.shape != (len(self.atoms), 3) or not np.isfinite(positions).all():
            raise ValueError(
                "The molecule needs exactly one finite Cartesian row per atom"
            )
        if np.max(np.abs(positions)) > 1e6:
            raise ValueError("Coordinates exceed the bounded molecular input domain")
        for index, row in enumerate(positions):
            if np.any(np.linalg.norm(positions[index + 1 :] - row, axis=1) < 1e-10):
                raise ValueError("Coincident atom coordinates are invalid")
        electrons = (
            sum(ATOMIC_NUMBERS[atom.symbol] for atom in self.atoms) - self.charge
        )
        unpaired = self.multiplicity - 1
        if electrons < 0 or unpaired > electrons or (electrons - unpaired) % 2:
            raise ValueError(
                "Charge, electron count, and multiplicity have incompatible parity"
            )
        return self

    def to_application_molecule(self) -> dict[str, Any]:
        """Convert explicit source units to the TORQ application input contract."""
        scale = 1.0 if self.geometry_unit == "bohr" else ANGSTROM_BOHR
        return {
            "symbols": [atom.symbol for atom in self.atoms],
            "geometry_bohr": (np.asarray(self.geometry) * scale).tolist(),
            "charge": self.charge,
            "multiplicity": self.multiplicity,
            "atom_ids": [atom.atom_id for atom in self.atoms],
            "isotopes": [atom.isotope_mass_number for atom in self.atoms],
        }

    @property
    def geometry_sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.model_dump(mode="json"))).hexdigest()


class MethodProvenance(StrictRecord):
    """The actual source recipe, which may differ from the requested TORQ recipe."""

    recipe_id: str = Field(min_length=1, max_length=200)
    engine: str = Field(min_length=1, max_length=120)
    engine_version: str = Field(min_length=1, max_length=120)
    method: str = Field(min_length=1, max_length=200)
    basis: str | None
    parameters: dict[str, Any]
    recipe_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @property
    def serialization_profile(self) -> str:
        """Identity profile; excluded from the existing recipe digest payload."""
        return ECOSYSTEM_DIGEST_PROFILE

    @model_validator(mode="after")
    def complete_recipe_digest(self) -> MethodProvenance:
        for name in ("recipe_id", "engine", "engine_version", "method"):
            if not getattr(self, name).strip():
                raise ValueError(
                    f"{name} must identify an explicitly supplied source recipe"
                )
        payload = self.model_dump(mode="json", exclude={"recipe_sha256"})
        if hashlib.sha256(_canonical_json(payload)).hexdigest() != self.recipe_sha256:
            raise ValueError(
                "recipe_sha256 does not match the full source method record"
            )
        return self

    @classmethod
    def from_recipe(
        cls,
        *,
        recipe_id: str,
        engine: str,
        engine_version: str,
        method: str,
        basis: str | None,
        parameters: dict[str, Any],
    ) -> MethodProvenance:
        payload: dict[str, Any] = dict(
            recipe_id=recipe_id,
            engine=engine,
            engine_version=engine_version,
            method=method,
            basis=basis,
            parameters=parameters,
        )
        return cls(
            **payload,
            recipe_sha256=hashlib.sha256(_canonical_json(payload)).hexdigest(),
        )


class EnergyObservation(StrictRecord):
    value: float = Field(allow_inf_nan=False)
    unit: Literal["hartree"]
    kind: Literal["total_electronic", "relative_electronic"]
    geometry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_id: str | None

    @field_validator("value", mode="before")
    @classmethod
    def physical_numeric_value(cls, value: Any) -> Any:
        if isinstance(value, (bool, str)):
            raise ValueError("Energy must be an explicitly observed finite number")
        return value

    @model_validator(mode="after")
    def relative_reference(self) -> EnergyObservation:
        if self.kind == "relative_electronic" and not self.reference_id:
            raise ValueError("A relative energy requires an explicit reference_id")
        return self


class HandoffSource(StrictRecord):
    producer: Literal["base", "topos", "torq", "explicit_import"]
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_schema: str = Field(min_length=1)
    repository_revision: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    source_record: str = Field(min_length=1)
    manifest_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    metadata: dict[str, Any] = Field(default_factory=dict)
    validation_scope: Literal["structure_and_integrity"] = "structure_and_integrity"


class ConformerHandoff(StrictRecord):
    schema_version: Literal["cochem.torq-conformer-handoff/1"] = (
        "cochem.torq-conformer-handoff/1"
    )
    # Older v1 files omit this field. Their molecule/recipe payload bytes remain
    # identical because this additive declaration lives in the outer contract.
    serialization_profile: Literal["cochem.sorted-json/1"] = "cochem.sorted-json/1"
    conformer_id: str = Field(min_length=1)
    molecule: MoleculeHandoff
    source: HandoffSource
    source_method: MethodProvenance | None
    energy: EnergyObservation | None
    source_convergence: Literal["unknown", "not_converged", "converged"]
    convergence_evidence_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    quality_flags: tuple[str, ...]

    @model_validator(mode="after")
    def energy_is_geometry_bound(self) -> ConformerHandoff:
        if (
            self.source_convergence != "unknown"
            and self.convergence_evidence_sha256 is None
        ):
            raise ValueError(
                "A convergence assertion requires an independently preserved "
                "evidence artifact digest"
            )
        if self.energy is not None:
            if self.source_method is None:
                raise ValueError(
                    "An energy handoff requires its complete source method recipe"
                )
            if self.energy.geometry_sha256 != self.molecule.geometry_sha256:
                raise ValueError(
                    "An energy cannot be attached to a different molecular "
                    "geometry/state"
                )
            if self.energy.source_artifact_sha256 != self.source.artifact_sha256:
                raise ValueError("Energy/source artifact digests disagree")
        return self

    def to_application_molecule(self) -> dict[str, Any]:
        return self.molecule.to_application_molecule()

    def to_application_provenance(self) -> dict[str, Any]:
        return {
            "handoff": self.model_dump(mode="json"),
            "geometry_sha256": self.molecule.geometry_sha256,
            "atom_mapping": [atom.atom_id for atom in self.molecule.atoms],
        }


def parse_xyz(
    text: str,
) -> tuple[tuple[str, ...], tuple[tuple[float, float, float], ...]]:
    """Read exactly one frame; the caller must explicitly specify XYZ units."""
    lines = text.splitlines()
    if not lines or not lines[0].strip().isascii() or not lines[0].strip().isdigit():
        raise HandoffError("XYZ requires an explicit integer atom count")
    count = int(lines[0].strip())
    if not 1 <= count <= MAX_ATOMS or len(lines) != count + 2:
        raise HandoffError("XYZ must contain exactly one complete bounded geometry")
    symbols: list[str] = []
    geometry: list[tuple[float, float, float]] = []
    for line in lines[2:]:
        fields = line.split()
        if len(fields) != 4 or fields[0] not in ATOMIC_NUMBERS:
            raise HandoffError(
                "XYZ rows require a canonical element and three coordinates"
            )
        row = tuple(float(value) for value in fields[1:])
        if not all(math.isfinite(value) for value in row):
            raise HandoffError("XYZ coordinates must be finite")
        symbols.append(fields[0])
        geometry.append((row[0], row[1], row[2]))
    return tuple(symbols), tuple(geometry)


def _molecule_from_xyz(
    text: str,
    *,
    molecule_id: str,
    charge: int,
    multiplicity: int,
    atom_ids: tuple[str, ...] | list[str],
    geometry_unit: Literal["angstrom", "bohr"],
    isotope_mass_numbers: tuple[int | None, ...] | list[int | None] | None = None,
) -> MoleculeHandoff:
    symbols, geometry = parse_xyz(text)
    if len(atom_ids) != len(symbols):
        raise HandoffError("Explicit atom IDs must map every source geometry row")
    isotopes = (
        isotope_mass_numbers
        if isotope_mass_numbers is not None
        else [None] * len(symbols)
    )
    if len(isotopes) != len(symbols):
        raise HandoffError("Isotope identifiers must match the atom mapping")
    atoms = tuple(
        AtomIdentity(atom_id=atom_id, symbol=symbol, isotope_mass_number=isotope)
        for atom_id, symbol, isotope in zip(atom_ids, symbols, isotopes, strict=True)
    )
    return MoleculeHandoff(
        molecule_id=molecule_id,
        atoms=atoms,
        geometry=geometry,
        geometry_unit=geometry_unit,
        charge=charge,
        multiplicity=multiplicity,
    )


def read_topos_handoff(
    path: str | Path,
    *,
    geometry_id: str,
    tier: str,
    molecule_id: str,
    charge: int,
    multiplicity: int,
    atom_ids: tuple[str, ...] | list[str],
    geometry_unit: Literal["angstrom", "bohr"],
    method_provenance: MethodProvenance,
    energy_kind: Literal["total_electronic", "relative_electronic"],
    reference_id: str | None,
    source_convergence: Literal["unknown", "not_converged", "converged"] = "unknown",
    convergence_evidence: str | Path | None = None,
    repository_revision: str | None = None,
    isotope_mass_numbers: tuple[int | None, ...] | list[int | None] | None = None,
    lock_timeout: float = 10.0,
) -> ConformerHandoff:
    """Read a real TOPOS v1 serializer snapshot under its actual writer lock.

    TOPOS omits state, row mapping, units, recipe and convergence from its HDF5.
    These arguments are mandatory external metadata, not defaults. Explicit
    convergence must come from independent producer evidence, not this reader.
    """
    source = Path(path).expanduser().resolve(strict=True)
    evidence_digest = None
    if convergence_evidence is not None:
        evidence = Path(convergence_evidence).expanduser().resolve(strict=True)
        if not evidence.is_file() or evidence.stat().st_size <= 0:
            raise HandoffError(
                "Convergence evidence must identify a nonempty preserved artifact"
            )
        evidence_digest = _sha256(evidence)
    if source_convergence != "unknown" and evidence_digest is None:
        raise HandoffError(
            "An explicit convergence assertion requires convergence_evidence"
        )
    if any(
        not value or "/" in value or value in {".", ".."}
        for value in (geometry_id, tier)
    ):
        raise HandoffError(
            "Geometry and tier IDs must each identify a single HDF5 group"
        )
    if not math.isfinite(lock_timeout) or lock_timeout <= 0:
        raise HandoffError("A positive finite snapshot lock timeout is required")
    with FileLock(str(source) + ".lock", timeout=lock_timeout):
        digest = _sha256(source)
        with h5py.File(source, "r") as database:
            if database.attrs.get("format_version") != "1.0":
                raise HandoffError(
                    "Unsupported TOPOS serializer format; require format_version 1.0"
                )
            key = f"{geometry_id}/{tier}"
            if key not in database or not isinstance(database[key], h5py.Group):
                raise HandoffError("The requested TOPOS geometry/tier record is absent")
            group = database[key]
            if (
                "geometry_xyz" not in group
                or "electronic_energy_hartree" not in group.attrs
            ):
                raise HandoffError("The TOPOS record lacks explicit geometry or energy")
            dataset = group["geometry_xyz"]
            if not isinstance(dataset, h5py.Dataset) or dataset.shape != ():
                raise HandoffError(
                    "TOPOS geometry_xyz must be a single scalar UTF-8 byte string"
                )
            if dataset.nbytes > MAX_ARTIFACT_BYTES:
                raise HandoffError(
                    "The TOPOS geometry exceeds the bounded artifact limit"
                )
            raw = dataset[()]
            if not isinstance(raw, (str, bytes)) or len(raw) > MAX_ARTIFACT_BYTES:
                raise HandoffError("Invalid or oversized TOPOS geometry byte string")
            xyz = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            energy_value = group.attrs["electronic_energy_hartree"]
            if isinstance(energy_value, (bool, np.bool_)) or np.ndim(energy_value) != 0:
                raise HandoffError("TOPOS energy must be a scalar observed number")
            if not isinstance(energy_value, (int, float, np.number)):
                raise HandoffError(
                    "TOPOS energy must be an explicitly numeric observation"
                )
            energy_value = float(energy_value)
        if _sha256(source) != digest:
            raise HandoffError("TOPOS bytes changed during the locked snapshot")
    molecule = _molecule_from_xyz(
        xyz,
        molecule_id=molecule_id,
        charge=charge,
        multiplicity=multiplicity,
        atom_ids=atom_ids,
        geometry_unit=geometry_unit,
        isotope_mass_numbers=isotope_mass_numbers,
    )
    observation = EnergyObservation(
        value=energy_value,
        unit="hartree",
        kind=energy_kind,
        geometry_sha256=molecule.geometry_sha256,
        source_artifact_sha256=digest,
        reference_id=reference_id,
    )
    flags = [
        "source_state_mapping_units_and_recipe_supplied_externally",
        "serialization_does_not_establish_scientific_accuracy",
    ]
    if source_convergence == "unknown":
        flags.append("source_convergence_unavailable")
    return ConformerHandoff(
        conformer_id=geometry_id,
        molecule=molecule,
        source=HandoffSource(
            producer="topos",
            artifact_sha256=digest,
            source_schema="topos.CascadeHDF5Serializer/1.0",
            repository_revision=repository_revision,
            source_record=f"{geometry_id}/{tier}",
        ),
        source_method=method_provenance,
        energy=observation,
        source_convergence=source_convergence,
        convergence_evidence_sha256=evidence_digest,
        quality_flags=tuple(flags),
    )


def read_base_handoff(
    path: str | Path,
    *,
    molecule_id: str,
    charge: int,
    multiplicity: int,
    atom_ids: tuple[str, ...] | list[str],
    isotope_mass_numbers: tuple[int | None, ...] | list[int | None] | None = None,
    repository_revision: str | None = None,
) -> ConformerHandoff:
    """Consume BASE's real ``cochem.module-handoff/1`` geometry artifact.

    This receiver rechecks actual bytes and metadata independently. It neither
    imports BASE in TORQ's environment nor claims BASE execution qualification.
    BASE's manifest intentionally does not carry molecular charge/spin/map.
    """
    manifest = Path(path).expanduser().resolve(strict=True)
    data = _read_json(manifest)
    manifest_digest = _sha256(manifest)
    expected_keys = {
        "schema_version",
        "handoff_id",
        "created_at",
        "module_id",
        "operation",
        "status",
        "capability",
        "artifact",
        "options",
        "scientific_execution_performed",
        "validation_scope",
    }
    if (
        set(data) != expected_keys
        or data["schema_version"] != "cochem.module-handoff/1"
    ):
        raise HandoffError(
            "Require the exact supported BASE module-handoff/1 structure"
        )
    if (
        not isinstance(data["handoff_id"], str)
        or not data["handoff_id"].strip()
        or not isinstance(data["operation"], str)
        or not data["operation"].strip()
        or not isinstance(data["created_at"], str)
        or not isinstance(data["options"], dict)
    ):
        raise HandoffError(
            "BASE identity, operation, timestamp, and options need explicit valid types"
        )
    try:
        timestamp = datetime.fromisoformat(data["created_at"])
    except ValueError as exc:
        raise HandoffError("BASE created_at must be an ISO timestamp") from exc
    if timestamp.tzinfo is None:
        raise HandoffError("BASE created_at must identify its timezone")
    _canonical_json(data["options"])
    if (
        data["module_id"] != "torq"
        or data["status"] != "pending_integration"
        or data["scientific_execution_performed"] is not False
        or data["validation_scope"] != "artifact_structure_and_integrity"
    ):
        raise HandoffError("BASE handoff recipient/status/scope is inconsistent")
    capability = data["capability"]
    if (
        not isinstance(capability, dict)
        or capability.get("module_id") != "torq"
        or capability.get("execution_verified") is not False
        or capability.get("integration_contract") != "cochem.module-handoff/1"
    ):
        raise HandoffError("BASE capability metadata is inconsistent")
    if capability.get("status") not in {
        "not_installed",
        "installed_pending_integration",
        "conflicting_providers",
    }:
        raise HandoffError(
            "BASE discovery cannot claim qualified TORQ execution availability"
        )
    reference = data["artifact"]
    if not isinstance(reference, dict) or reference.get("kind") != "geometry_xyz":
        raise HandoffError(
            "This adapter accepts only a BASE single-geometry XYZ handoff"
        )
    if set(reference) != {
        "kind",
        "filename",
        "source_path",
        "sha256",
        "size_bytes",
        "metadata",
    }:
        raise HandoffError(
            "BASE artifact fields do not match the supported v1 contract"
        )
    if not isinstance(reference["source_path"], str) or not reference["source_path"]:
        raise HandoffError("BASE artifact must retain its explicit source path")
    filename = reference.get("filename")
    if (
        not isinstance(filename, str)
        or Path(filename).name != filename
        or filename in {".", ".."}
    ):
        raise HandoffError(
            "BASE artifact filename must stay inside the handoff package"
        )
    artifact = (manifest.parent / filename).resolve(strict=True)
    if artifact.parent != manifest.parent or not artifact.is_file():
        raise HandoffError(
            "BASE artifact path escapes its package or is not a regular file"
        )
    size = reference.get("size_bytes")
    if (
        type(size) is not int
        or not 0 < size <= MAX_ARTIFACT_BYTES
        or artifact.stat().st_size != size
    ):
        raise HandoffError("BASE artifact size differs from its bounded manifest")
    raw = artifact.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != reference.get("sha256"):
        raise HandoffError("BASE artifact checksum verification failed")
    molecule = _molecule_from_xyz(
        raw.decode("utf-8"),
        molecule_id=molecule_id,
        charge=charge,
        multiplicity=multiplicity,
        atom_ids=atom_ids,
        geometry_unit="angstrom",
        isotope_mass_numbers=isotope_mass_numbers,
    )
    metadata = reference.get("metadata")
    actual = {
        "symbols": [atom.symbol for atom in molecule.atoms],
        "atom_count": len(molecule.atoms),
        "coordinates_unit": "angstrom",
    }
    if metadata != actual:
        raise HandoffError("BASE artifact metadata disagrees with actual XYZ contents")
    if (
        _read_json(manifest) != data
        or _sha256(manifest) != manifest_digest
        or artifact.read_bytes() != raw
    ):
        raise HandoffError("BASE handoff changed during receiver validation")
    return ConformerHandoff(
        conformer_id=data["handoff_id"],
        molecule=molecule,
        source=HandoffSource(
            producer="base",
            artifact_sha256=digest,
            source_schema=data["schema_version"],
            repository_revision=repository_revision,
            source_record=data["handoff_id"],
            manifest_sha256=manifest_digest,
            metadata={
                "operation": data["operation"],
                "options": data["options"],
                "created_at": data["created_at"],
                "capability": capability,
            },
        ),
        source_method=None,
        energy=None,
        source_convergence="unknown",
        quality_flags=(
            "input_geometry_is_not_an_optimized_result",
            "source_charge_spin_and_atom_mapping_supplied_externally",
            "base_handoff_integrity_revalidated_without_provider_import",
        ),
    )


def write_conformer_handoff(handoff: ConformerHandoff, destination: str | Path) -> Path:
    """Publish a strict complete handoff exclusively, without replacing user files."""
    record = ConformerHandoff.model_validate(handoff.model_dump(mode="json"))
    return _publish_json(record.model_dump(mode="json"), destination)


def _publish_json(record: dict[str, Any], destination: str | Path) -> Path:
    target = Path(destination).expanduser().resolve()
    from cochem.core.context import assert_writable_path

    assert_writable_path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(record, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=".torq-handoff-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


def export_base_calculation_result(
    run_directory: str | Path, destination: str | Path
) -> Path:
    """Export a genuine sealed TORQ electronic-energy result to BASE's consumer.

    Both the outer worker shard and the actual parsed native engine artifacts are
    checked. This operation reports observed electronic energy at the verified
    final geometry. It does not assign spectroscopic or experimental accuracy.
    """
    from .artifacts import verify_shard
    from .domain import CalculationRequest, StageResult

    directory = Path(run_directory).expanduser().resolve(strict=True)
    manifest = verify_shard(directory)
    manifest_digest = _sha256(directory / "manifest.json")
    result = _read_json(directory / "result.json")
    if result.get("schema_version") != "cochem.torq.result/1":
        raise HandoffError(
            "A genuine versioned TORQ result is required for BASE export"
        )
    request = CalculationRequest.model_validate(_read_json(directory / "request.json"))
    stages = result.get("stages")
    if not isinstance(stages, dict):
        raise HandoffError("A result must contain its actual typed scientific stages")
    electronic = StageResult.model_validate(stages.get("electronic_structure"))
    geometry = StageResult.model_validate(stages.get("equilibrium_geometry"))
    if electronic.status != "available" or geometry.status != "available":
        raise HandoffError(
            "BASE export requires available electronic and verified geometry stages"
        )
    if electronic.value is None or geometry.value is None:
        raise HandoffError("Scientific stages must retain their actual values")
    observed = electronic.value
    positions = geometry.value
    if observed.get("scf", {}).get("converged") is not True:
        raise HandoffError(
            "Electronic energy export requires genuine converged SCF evidence"
        )
    if observed.get("stability", {}).get("status") != "stable":
        raise HandoffError(
            "Electronic energy export requires established wavefunction stability"
        )
    native = result.get("native_result")
    if not isinstance(native, dict):
        raise HandoffError("TORQ's actual native calculation record is missing")
    native_manifest = directory / "engine" / "manifest.json"
    native_record = _read_json(directory / "engine" / "result.json")
    if (
        not native_manifest.is_file()
        or _sha256(native_manifest) != observed.get("native_manifest_sha256")
        or native.get("manifest_sha256") != observed.get("native_manifest_sha256")
    ):
        raise HandoffError(
            "The electronic stage does not identify the preserved "
            "native engine manifest"
        )
    native_without_manifest = {
        key: value
        for key, value in native.items()
        if key not in {"artifacts", "manifest_path", "manifest_sha256"}
    }
    if native_without_manifest != native_record:
        raise HandoffError(
            "TORQ's parsed native result differs from its sealed engine record"
        )
    for field in (
        "energy_hartree",
        "engine",
        "engine_version",
        "method",
        "scf",
        "stability",
    ):
        if field not in observed or observed[field] != native_record.get(field):
            raise HandoffError(
                f"Electronic stage {field} disagrees with the authentic native result"
            )
    energy = observed["energy_hartree"]
    if (
        isinstance(energy, bool)
        or not isinstance(energy, (int, float))
        or not math.isfinite(energy)
    ):
        raise HandoffError(
            "An explicitly observed finite Hartree electronic energy is required"
        )
    if (
        not isinstance(observed["engine"], str)
        or not observed["engine"].strip()
        or native_record.get("units", {}).get("energy") != "hartree"
        or native_record.get("units", {}).get("geometry") != "bohr"
    ):
        raise HandoffError(
            "The producer engine and actual energy/coordinate units must be explicit"
        )
    geometry_array = np.asarray(positions.get("geometry_bohr"), dtype=np.float64)
    actual_geometry = np.asarray(native_record.get("geometry_bohr"), dtype=np.float64)
    native_molecule = native_record.get("molecule", {})
    if (
        positions.get("symbols") != request.molecule.symbols
        or native_molecule.get("symbols") != request.molecule.symbols
        or native_molecule.get("charge") != request.molecule.charge
        or native_molecule.get("multiplicity") != request.molecule.multiplicity
        or geometry_array.shape != (len(request.molecule.symbols), 3)
        or not np.isfinite(geometry_array).all()
        or not np.array_equal(geometry_array, actual_geometry)
        or positions.get("atom_ids") != request.molecule.atom_ids
    ):
        raise HandoffError(
            "The exported energy and geometry/state/row mapping do not match"
        )
    optimization = positions.get("optimization")
    if (
        not isinstance(optimization, dict)
        or optimization.get("converged") is not True
        or optimization.get("final_gradient_verified") is not True
        or optimization != native_record.get("optimization")
    ):
        raise HandoffError(
            "Geometry export requires actual optimizer and final-gradient evidence"
        )
    payload = {
        "schema_version": "cochem.torq-base-calculation-result/1",
        "quantity": "total_electronic_energy",
        "scope": "observed_electronic_energy_at_verified_final_geometry",
        "converged": True,
        "energy_hartree": energy,
        "engine": observed["engine"],
        "engine_version": observed["engine_version"],
        "method": observed["method"],
        "scientific_recipe": result["recipe"],
        "molecule": {
            **request.molecule.model_dump(mode="json"),
            "geometry_bohr": geometry_array.tolist(),
        },
        "atom_mapping": {
            "source_row_indices": list(range(len(request.molecule.symbols))),
            "atom_ids": request.molecule.atom_ids,
        },
        "scf": observed["scf"],
        "stability": observed["stability"],
        "optimization": optimization,
        "quality_flags": sorted(set(electronic.quality_flags + geometry.quality_flags)),
        "uncertainty": electronic.uncertainty,
        "source_provenance": request.source_provenance,
        "source": {
            "request_id": str(request.request_id),
            "request_sha256": manifest["request_sha256"],
            "recipe_sha256": manifest["recipe_sha256"],
            "shard_manifest_sha256": manifest_digest,
            "result_sha256": _sha256(directory / "result.json"),
            "native_manifest_sha256": _sha256(native_manifest),
        },
    }
    if (
        verify_shard(directory) != manifest
        or _sha256(directory / "manifest.json") != manifest_digest
    ):
        raise HandoffError(
            "TORQ source artifacts changed while preparing the BASE export"
        )
    return _publish_json(payload, destination)


def load_conformer_handoff(path: str | Path) -> ConformerHandoff:
    return ConformerHandoff.model_validate(
        _read_json(Path(path).expanduser().resolve(strict=True))
    )


def module_provider() -> dict[str, str]:
    """BASE discovery metadata only; operation qualification remains explicit."""
    return {
        "module_id": "torq",
        "distribution": "CoChem-TORQ",
        "integration_contract": "cochem.module-handoff/1",
        "conformer_contract": "cochem.torq-conformer-handoff/1",
    }


__all__ = [
    "AtomIdentity",
    "MoleculeHandoff",
    "MethodProvenance",
    "EnergyObservation",
    "HandoffSource",
    "ConformerHandoff",
    "HandoffError",
    "parse_xyz",
    "read_base_handoff",
    "read_topos_handoff",
    "write_conformer_handoff",
    "load_conformer_handoff",
    "export_base_calculation_result",
    "module_provider",
]
