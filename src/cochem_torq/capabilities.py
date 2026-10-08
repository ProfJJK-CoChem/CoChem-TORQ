"""Exact experimental capability routing and authentic restart compatibility.

Local validation establishes an exact numerical integration case, never universal
chemical accuracy. Restart compatibility and preparation do not establish that
an engine has actually consumed a checkpoint or skipped a physical calculation.
"""

from __future__ import annotations

import os
import platform
import re
from builtins import property as computed_property
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import Field, StrictFloat, StrictInt, model_validator

from .domain import Contract, Molecule, canonical_json, digest, read_json

MAX_ARTIFACT_BYTES = 512 * 1024**2
SHA_PATTERN = r"^[0-9a-f]{64}$"
Availability = Literal[
    "documented", "locally_validated", "experimental", "unsupported", "unknown"
]
Purpose = Literal["controlled_validation", "production"]


class CapabilityTuple(Contract):
    method: str = Field(min_length=1, max_length=128)
    basis: str = Field(min_length=1, max_length=128)
    ecp: str = Field(default="none", min_length=1, max_length=128)
    electronic_reference: str = Field(min_length=1, max_length=128)
    property: str = Field(min_length=1, max_length=128)
    derivative: str = Field(min_length=1, max_length=128)
    engine: str = Field(min_length=1, max_length=128)
    engine_version: str = Field(min_length=1, max_length=64)
    optimizer_version: str | None = None
    hardware: str = Field(min_length=1, max_length=128)
    recipe_sha256: str = Field(pattern=SHA_PATTERN)

    @model_validator(mode="after")
    def derivative_identity(self) -> CapabilityTuple:
        routes = {
            "energy": {"none"},
            "gradient": {"analytic"},
            "hessian": {"analytic", "centered_difference_of_analytic_gradient"},
            "dipole": {"density_expectation"},
            "optimization": {"analytic_gradient_optimization"},
        }
        if self.property in routes and self.derivative not in routes[self.property]:
            raise ValueError("The property and exact derivative route disagree.")
        if (self.property == "optimization") != (self.optimizer_version is not None):
            raise ValueError("Optimization must identify its exact optimizer version.")
        return self

    @computed_property
    def identity_sha256(self) -> str:
        return digest(self.model_dump(mode="json"))


class LocalCapabilityEvidence(Contract):
    result_path: str
    result_sha256: str = Field(pattern=SHA_PATTERN)
    manifest_path: str
    manifest_sha256: str = Field(pattern=SHA_PATTERN)
    native_request_sha256: str = Field(pattern=SHA_PATTERN)
    source_code_sha256: str = Field(pattern=SHA_PATTERN)
    adapter_source_sha256: str = Field(pattern=SHA_PATTERN)
    engine_installation_sha256: str = Field(pattern=SHA_PATTERN)
    basis_definition_sha256: str = Field(pattern=SHA_PATTERN)
    capability_sha256: str = Field(pattern=SHA_PATTERN)
    qualification_scope: Literal["exact_native_request_numerical_integration"] = (
        "exact_native_request_numerical_integration"
    )
    chemical_accuracy_established: Literal[False] = False


class CapabilityRecord(Contract):
    schema_version: Literal["cochem.torq.capability/1"] = "cochem.torq.capability/1"
    tuple_definition: CapabilityTuple
    availability: Availability
    reason: str = Field(min_length=1)
    evidence: LocalCapabilityEvidence | None = None

    @model_validator(mode="after")
    def evidence_binding(self) -> CapabilityRecord:
        if self.availability == "locally_validated":
            if (
                self.evidence is None
                or self.evidence.capability_sha256
                != self.tuple_definition.identity_sha256
            ):
                raise ValueError(
                    "Local validation requires evidence for this exact tuple."
                )
        elif self.evidence is not None:
            raise ValueError("Unqualified availability cannot present local evidence.")
        return self


def observed_cpu_hardware() -> str:
    """Record observed OS/architecture without inferring a GPU or benchmark machine."""
    return f"{platform.system().lower()}-{platform.machine().lower()}-cpu"


def _actual_file(path: Path) -> Path:
    path = path.absolute()
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError("Native evidence and restart inputs cannot follow symlinks.")
    if not path.is_file() or not 0 < path.stat().st_size <= MAX_ARTIFACT_BYTES:
        raise ValueError("A bounded, nonempty authentic artifact file is required.")
    return path


def _file_sha256(path: Path) -> str:
    identity = sha256()
    with _actual_file(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            identity.update(block)
    return identity.hexdigest()


def _relative_file(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("Native artifact paths must be explicit relative paths.")
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Native artifact paths must remain inside their bundle.")
    return _actual_file(root / relative)


def _native_bundle(directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    directory = directory.absolute()
    manifest = read_json(_actual_file(directory / "manifest.json"))
    if manifest.get("schema_version") != "cochem-torq.engine-artifacts.v1":
        raise ValueError("An authentic supported native artifact manifest is required.")
    entries = manifest.get("artifacts")
    if (
        not isinstance(entries, list)
        or not entries
        or any(not isinstance(entry, dict) for entry in entries)
        or len({entry.get("path") for entry in entries}) != len(entries)
    ):
        raise ValueError(
            "Native artifact inventory is absent or has duplicate entries."
        )
    for entry in entries:
        path = _relative_file(directory, entry.get("path"))
        if path.stat().st_size != entry.get("size_bytes") or _file_sha256(
            path
        ) != entry.get("sha256"):
            raise ValueError(
                "Native artifact bytes differ from their retained inventory."
            )
    names = {entry["path"] for entry in entries}
    if not {"request.json", "result.json"} <= names:
        raise ValueError(
            "Native request/result/basis/installation provenance is incomplete."
        )
    request = read_json(directory / "request.json")
    result = read_json(directory / "result.json")
    if not isinstance(request, dict) or not isinstance(result, dict):
        raise ValueError("Native request/result evidence requires JSON objects.")
    if (
        result.get("schema_version") != "cochem-torq.pyscf-result.v1"
        or result.get("status") != "complete"
        or result.get("scf", {}).get("converged") is not True
        or result.get("stability", {}).get("status") != "stable"
    ):
        raise ValueError("A completed, converged, stable native result is required.")
    provenance = directory
    if "optimization" in result:
        provenance = _optimization_final_bundle(directory, request, result, entries)
    elif request.get("molecule") != result.get("molecule") or result.get(
        "geometry_bohr"
    ) != request.get("molecule", {}).get("geometry_bohr"):
        raise ValueError("Native single-point geometry/state differs from its request.")
    prefix = "final/" if provenance != directory else ""
    if (
        not {
            f"{prefix}basis-definition.json",
            f"{prefix}engine-installation.json",
        }
        <= names
    ):
        raise ValueError(
            "Native request/result/basis/installation provenance is incomplete."
        )
    if result.get("basis_definition_sha256") != _file_sha256(
        provenance / "basis-definition.json"
    ):
        raise ValueError("Native basis identity does not match the actual definitions.")
    if result.get("engine_installation_sha256") != read_json(
        provenance / "engine-installation.json"
    ).get("digest"):
        raise ValueError("Native installation identity is inconsistent.")
    if request.get("method") != result.get("method") or request.get(
        "settings"
    ) != result.get("settings"):
        raise ValueError("The retained native request and result definitions differ.")
    return request, result


def _optimization_final_bundle(
    directory: Path,
    request: dict[str, Any],
    result: dict[str, Any],
    outer_entries: list[dict[str, Any]],
) -> Path:
    """Bind actual optimizer evidence to its separately sealed final calculation."""
    names = {entry["path"] for entry in outer_entries}
    if (
        not {
            "final/request.json",
            "final/result.json",
            "final/basis-definition.json",
            "final/engine-installation.json",
            "optimization-trajectory.json",
            "geometric.log",
        }
        <= names
    ):
        raise ValueError("Native optimization final provenance is incomplete.")
    final = directory / "final"
    final_request, final_result = _native_bundle(final)
    # Native outer manifests retain each child's raw files, but the current
    # native format omits nested manifest files. Compare their inventories to
    # the parent-sealed bytes rather than inventing a retained manifest digest.
    nested = read_json(final / "manifest.json")["artifacts"]
    expected = sorted(
        ({**entry, "path": f"final/{entry['path']}"} for entry in nested),
        key=lambda entry: entry["path"],
    )
    observed = sorted(
        (entry for entry in outer_entries if entry["path"].startswith("final/")),
        key=lambda entry: entry["path"],
    )
    parsed = dict(result)
    optimization = parsed.pop("optimization")
    if expected != observed or parsed != final_result:
        raise ValueError(
            "Outer optimization differs from its retained final native evidence."
        )
    initial_molecule = dict(request["molecule"])
    final_molecule = dict(final_request["molecule"])
    initial_geometry = initial_molecule.pop("geometry_bohr")
    final_geometry = final_molecule.pop("geometry_bohr")
    if (
        not isinstance(optimization, dict)
        or initial_molecule != final_molecule
        or final_geometry != result["geometry_bohr"]
        or optimization.get("initial_geometry_bohr") != initial_geometry
        or request["method"] != final_request["method"]
        or request["settings"] != final_request["settings"]
        or request["optimization"] != final_request["optimization"]
        or set(final_request["properties"]) != set(request["properties"]) | {"gradient"}
    ):
        raise ValueError(
            "Optimization initial/final geometry/state definitions differ."
        )
    import numpy as np

    parameters = optimization.get("parameters", {})
    required = {
        "maxsteps",
        "convergence_energy",
        "convergence_grms",
        "convergence_gmax",
        "convergence_drms",
        "convergence_dmax",
    }
    trajectory = read_json(directory / "optimization-trajectory.json")
    gradient = np.asarray(result["gradient_hartree_bohr"], dtype=float)
    if (
        optimization.get("engine") != "geomeTRIC"
        or optimization.get("converged") is not True
        or optimization.get("final_gradient_verified") is not True
        or not isinstance(parameters, dict)
        or set(parameters) != required
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value <= 0
            for value in parameters.values()
        )
        or type(parameters["maxsteps"]) is not int
        or parameters["maxsteps"] > 300
        or any(
            parameters.get(name) != value
            for name, value in request["optimization"].items()
        )
        or gradient.shape != (len(result["molecule"]["symbols"]), 3)
        or not np.isfinite(gradient).all()
        or float(np.sqrt(np.mean(gradient * gradient))) > parameters["convergence_grms"]
        or float(np.max(np.abs(gradient))) > parameters["convergence_gmax"]
        or not isinstance(trajectory, list)
        or not trajectory
        or not isinstance(trajectory[-1], dict)
        or optimization.get("evaluations") != len(trajectory)
        or trajectory[-1].get("geometry_bohr") != final_geometry
    ):
        raise ValueError("Actual optimization convergence/trajectory is unestablished.")
    return final


def _match_native_tuple(definition: CapabilityTuple, result: dict[str, Any]) -> None:
    method = result["method"]
    named_method = method["name"] + ("-d4" if method.get("dispersion") == "d4" else "")
    hardware = result["platform"]
    observed = f"{hardware['system'].lower()}-{hardware['machine'].lower()}-cpu"
    if (
        definition.method != named_method
        or definition.basis != method["basis"]
        or definition.ecp != "none"
        or definition.electronic_reference != "restricted_closed_shell"
        or definition.engine != result["engine"]
        or definition.engine_version != result["engine_version"]
        or definition.hardware != observed
        or method.get("reference") != "restricted"
        or method.get("frozen_core") is not False
        or method.get("density_fitting") is not False
    ):
        raise ValueError(
            "The authentic native result does not match the exact capability tuple."
        )
    fields = {
        "energy": "energy_hartree",
        "gradient": "gradient_hartree_bohr",
        "hessian": "hessian_hartree_bohr2",
        "dipole": "dipole_debye",
        "optimization": "optimization",
    }
    if definition.property not in fields:
        raise NotImplementedError(
            "No authentic native qualification adapter exists for this property."
        )
    if result.get(fields[definition.property]) is None:
        raise ValueError(
            "The requested property is absent from the actual native result."
        )
    if (
        definition.property == "hessian"
        and result.get("hessian_evidence", {}).get("derivative")
        != definition.derivative
    ):
        raise ValueError(
            "The actual Hessian derivative route does not match the tuple."
        )
    if definition.property == "optimization":
        optimization = result["optimization"]
        if (
            optimization.get("converged") is not True
            or optimization.get("final_gradient_verified") is not True
            or optimization.get("version") != definition.optimizer_version
        ):
            raise ValueError(
                "The exact optimizer version and convergence are unestablished."
            )
    from .registry import list_method_profiles, profile_capabilities

    matching = [
        profile
        for profile in list_method_profiles()
        if profile["recipe_sha256"] == definition.recipe_sha256
    ]
    if len(matching) != 1 or definition not in tuple(
        record.tuple_definition
        for record in profile_capabilities(matching[0]["id"])
        if record.availability == "experimental"
    ):
        raise ValueError(
            "Local qualification requires a registered exact experimental recipe."
        )
    profile = matching[0]
    if (
        any(
            result["settings"].get(name) != value
            for name, value in profile["numerical"].items()
        )
        or len(result["molecule"]["symbols"]) > profile["max_atoms"]
        or set(result["molecule"]["symbols"]) - set(profile["elements"])
        or result["molecule"]["multiplicity"] != profile["multiplicity"]
        or profile.get("dispersion") == "d4"
        and result.get("dispersion", {}).get("library_version")
        != profile["dispersion_version"]
    ):
        raise ValueError(
            "The actual native settings/domain do not match the exact recipe."
        )


def locally_validate_capability(
    definition: CapabilityTuple, native_directory: str | Path
) -> CapabilityRecord:
    """Bind one genuine retained native case; do not promote neighboring requests."""
    from .application import source_identity

    root = Path(native_directory).absolute()
    request, result = _native_bundle(root)
    _match_native_tuple(definition, result)
    _require_current_native_installation(result)
    adapter_path = Path(__file__).parent / "engines/pyscf_backend.py"
    if result.get("adapter_source_sha256") != _file_sha256(adapter_path):
        raise ValueError("The actual native run belongs to a different adapter source.")
    evidence = LocalCapabilityEvidence(
        result_path=str(root / "result.json"),
        result_sha256=_file_sha256(root / "result.json"),
        manifest_path=str(root / "manifest.json"),
        manifest_sha256=_file_sha256(root / "manifest.json"),
        native_request_sha256=digest(request),
        source_code_sha256=source_identity()["code_sha256"],
        adapter_source_sha256=result["adapter_source_sha256"],
        engine_installation_sha256=result["engine_installation_sha256"],
        basis_definition_sha256=result["basis_definition_sha256"],
        capability_sha256=definition.identity_sha256,
    )
    return CapabilityRecord(
        tuple_definition=definition,
        availability="locally_validated",
        evidence=evidence,
        reason="Exact authentic native numerical-integration case; "
        "chemical accuracy unqualified.",
    )


def authorize_capability(
    record: CapabilityRecord,
    *,
    purpose: Purpose,
    native_request_sha256: str | None = None,
) -> dict[str, Any]:
    """Authorize only the selected exact status/scope, with no method substitutions."""
    record = CapabilityRecord.model_validate(record.model_dump(mode="json"))
    if purpose not in {"controlled_validation", "production"}:
        raise ValueError("Unknown capability routing purpose.")
    if record.availability == "experimental" and purpose == "controlled_validation":
        return {
            "authorized": True,
            "purpose": purpose,
            "availability": "experimental",
            "capability_sha256": record.tuple_definition.identity_sha256,
            "chemical_accuracy_established": False,
        }
    if record.availability != "locally_validated":
        raise ValueError(
            f"{record.availability} tuples do not authorize {purpose} dispatch."
        )
    evidence = record.evidence
    if evidence is None:
        raise ValueError("Local validation requires evidence for this exact tuple.")
    from .application import source_identity

    if evidence.source_code_sha256 != source_identity()["code_sha256"]:
        raise ValueError("Local evidence belongs to a different implementation source.")
    if native_request_sha256 != evidence.native_request_sha256:
        raise ValueError(
            "Local evidence does not cover this exact normalized native request."
        )
    root = Path(evidence.manifest_path).parent
    if (
        Path(evidence.result_path) != root / "result.json"
        or _file_sha256(root / "result.json") != evidence.result_sha256
        or _file_sha256(root / "manifest.json") != evidence.manifest_sha256
    ):
        raise ValueError("Local capability evidence bytes or identities changed.")
    request, result = _native_bundle(root)
    _match_native_tuple(record.tuple_definition, result)
    _require_current_native_installation(result)
    if observed_cpu_hardware() != record.tuple_definition.hardware:
        raise ValueError(
            "Local evidence does not cover the observed current CPU platform."
        )
    if digest(request) != evidence.native_request_sha256:
        raise ValueError("The actual native request changed after local qualification.")
    for field in (
        "adapter_source_sha256",
        "engine_installation_sha256",
        "basis_definition_sha256",
    ):
        if getattr(evidence, field) != result.get(field):
            raise ValueError(
                "Declared local evidence differs from authentic native provenance."
            )
    return {
        "authorized": True,
        "purpose": purpose,
        "availability": "locally_validated",
        "capability_sha256": record.tuple_definition.identity_sha256,
        "qualification_scope": evidence.qualification_scope,
        "chemical_accuracy_established": False,
    }


def _require_current_native_installation(result: dict[str, Any]) -> None:
    from importlib.metadata import version

    from .engines.pyscf_backend import _engine_fingerprint

    if (
        version("pyscf") != result["engine_version"]
        or _engine_fingerprint()["digest"] != result["engine_installation_sha256"]
    ):
        raise ValueError(
            "The observed current native engine installation is incompatible."
        )
    if (
        result.get("dispersion") is not None
        and version("dftd4") != result["dispersion"]["library_version"]
    ):
        raise ValueError(
            "The observed current dispersion installation is incompatible."
        )
    if result.get("optimization") is not None and version("geometric") != result[
        "optimization"
    ].get("version"):
        raise ValueError("The observed current optimizer installation is incompatible.")


class RestartFingerprint(Contract):
    schema_version: Literal["cochem.torq.restart-fingerprint/1"] = (
        "cochem.torq.restart-fingerprint/1"
    )
    geometry_bohr: tuple[tuple[StrictFloat, StrictFloat, StrictFloat], ...]
    symbols: tuple[str, ...]
    atom_ids: tuple[str, ...]
    isotope_symbols: tuple[str, ...]
    charge: StrictInt
    multiplicity: StrictInt
    coordinate_convention: Literal["bohr; atom-major XYZ; input Cartesian frame"] = (
        "bohr; atom-major XYZ; input Cartesian frame"
    )
    electronic_reference: Literal["restricted"]
    basis: str
    basis_definition_sha256: str = Field(pattern=SHA_PATTERN)
    ecp: Literal["none"] = "none"
    frozen_core: Literal[False] = False
    recipe_sha256: str = Field(pattern=SHA_PATTERN)
    engine: Literal["PySCF"]
    engine_version: str
    engine_installation_sha256: str = Field(pattern=SHA_PATTERN)
    adapter_source_sha256: str = Field(pattern=SHA_PATTERN)
    source_code_sha256: str = Field(pattern=SHA_PATTERN)
    method_definition_json: str
    numerical_settings_json: str

    @model_validator(mode="after")
    def atomic_and_numerical_identity(self) -> RestartFingerprint:
        Molecule(
            symbols=list(self.symbols),
            geometry_bohr=[list(row) for row in self.geometry_bohr],
            charge=self.charge,
            multiplicity=self.multiplicity,
            atom_ids=list(self.atom_ids),
        )
        if len(self.isotope_symbols) != len(self.symbols):
            raise ValueError("Restart isotope row identity must match every atom.")
        for symbol, isotope in zip(self.symbols, self.isotope_symbols):
            match = re.fullmatch(r"(?:([0-9]+))?([A-Z][a-z]?)", isotope)
            if (
                match is None
                or match.group(2) != symbol
                or match.group(1) is not None
                and not 1 <= int(match.group(1)) <= 350
            ):
                raise ValueError(
                    "Restart isotope symbols contradict the atomic row map."
                )
        import json

        for name in ("method_definition_json", "numerical_settings_json"):
            contents = getattr(self, name)
            definition = json.loads(contents)
            if (
                not isinstance(definition, dict)
                or canonical_json(definition).decode() != contents
            ):
                raise ValueError(
                    "Method/settings must preserve their exact canonical objects."
                )
        method = json.loads(self.method_definition_json)
        if (
            method.get("basis") != self.basis
            or method.get("reference") != self.electronic_reference
            or method.get("frozen_core") is not False
            or method.get("density_fitting") is not False
        ):
            raise ValueError(
                "Restart method definition contradicts its exact basis/reference."
            )
        return self

    @property
    def identity_sha256(self) -> str:
        return digest(self.model_dump(mode="json"))


class RestartArtifact(Contract):
    schema_version: Literal["cochem.torq.restart-artifact/1"] = (
        "cochem.torq.restart-artifact/1"
    )
    artifact_kind: Literal[
        "wavefunction",
        "hessian",
        "grid",
        "integrals",
        "force_field",
        "model_checkpoint",
    ]
    original_path: str
    sealed_path: str
    artifact_sha256: str = Field(pattern=SHA_PATTERN)
    size_bytes: StrictInt = Field(ge=1, le=MAX_ARTIFACT_BYTES)
    fingerprint: RestartFingerprint
    original_result_sha256: str = Field(pattern=SHA_PATTERN)
    native_manifest_path: str
    native_manifest_sha256: str = Field(pattern=SHA_PATTERN)
    engine_checkpoint_reused: Literal[False] = False


class ReusePolicy(Contract):
    purpose: Literal["initial_guess", "final_result"]
    allowed_initial_guess_changes: tuple[
        Literal["geometry_bohr", "numerical_settings_json"], ...
    ] = ()

    @model_validator(mode="after")
    def exact_final_policy(self) -> ReusePolicy:
        if len(set(self.allowed_initial_guess_changes)) != len(
            self.allowed_initial_guess_changes
        ):
            raise ValueError("Initial-guess change permissions must be distinct.")
        if self.purpose == "final_result" and self.allowed_initial_guess_changes:
            raise ValueError(
                "Final physical result reuse requires an exact fingerprint."
            )
        return self


def restart_fingerprint(
    native_directory: str | Path, *, recipe_sha256: str
) -> RestartFingerprint:
    """Fingerprint genuine final native geometry/state/basis/implementation/settings."""
    from .application import source_identity

    request, result = _native_bundle(Path(native_directory))
    _require_current_native_installation(result)
    if result.get("adapter_source_sha256") != _file_sha256(
        Path(__file__).parent / "engines/pyscf_backend.py"
    ):
        raise ValueError(
            "Restart registration requires the exact current native adapter."
        )
    molecule = result["molecule"]
    return RestartFingerprint(
        geometry_bohr=tuple(tuple(row) for row in result["geometry_bohr"]),
        symbols=tuple(molecule["symbols"]),
        atom_ids=tuple(
            molecule.get("atom_ids")
            or [f"atom-{index}" for index in range(len(molecule["symbols"]))]
        ),
        isotope_symbols=tuple(molecule["isotope_symbols"]),
        charge=molecule["charge"],
        multiplicity=molecule["multiplicity"],
        electronic_reference=result["method"]["reference"],
        basis=result["method"]["basis"],
        basis_definition_sha256=result["basis_definition_sha256"],
        recipe_sha256=recipe_sha256,
        engine=result["engine"],
        engine_version=result["engine_version"],
        engine_installation_sha256=result["engine_installation_sha256"],
        adapter_source_sha256=result["adapter_source_sha256"],
        source_code_sha256=source_identity()["code_sha256"],
        method_definition_json=canonical_json(result["method"]).decode(),
        numerical_settings_json=canonical_json(request["settings"]).decode(),
    )


def _copy_new(
    source: Path, destination: Path, expected_sha256: str, *, mode: int
) -> None:
    source = _actual_file(source)
    destination = destination.absolute()
    if destination.exists() or destination.is_symlink() or destination == source:
        raise FileExistsError(
            "Restart originals and existing files cannot be overwritten."
        )
    if any(parent.is_symlink() for parent in destination.parents):
        raise ValueError("Restart output paths cannot follow symlinks.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    observed = sha256()
    with source.open("rb") as original, destination.open("xb") as output:
        for block in iter(lambda: original.read(1024**2), b""):
            observed.update(block)
            output.write(block)
        output.flush()
        os.fsync(output.fileno())
    if (
        observed.hexdigest() != expected_sha256
        or _file_sha256(source) != expected_sha256
    ):
        raise ValueError(
            "Authentic restart bytes changed while preparing the new copy."
        )
    destination.chmod(mode)


def register_restart_artifact(
    native_directory: str | Path,
    relative_path: str,
    sealed_path: str | Path,
    *,
    artifact_kind: Literal[
        "wavefunction",
        "hessian",
        "grid",
        "integrals",
        "force_field",
        "model_checkpoint",
    ],
    recipe_sha256: str,
) -> RestartArtifact:
    root = Path(native_directory).absolute()
    _native_bundle(root)
    manifest = read_json(root / "manifest.json")
    if relative_path not in {entry["path"] for entry in manifest["artifacts"]}:
        raise ValueError(
            "The restart input is absent from the authentic native inventory."
        )
    original = _relative_file(root, relative_path)
    fingerprint = restart_fingerprint(root, recipe_sha256=recipe_sha256)
    _check_restart_kind(original, artifact_kind, fingerprint=fingerprint)
    identity = _file_sha256(original)
    destination = Path(sealed_path).absolute()
    _copy_new(original, destination, identity, mode=0o440)
    return RestartArtifact(
        artifact_kind=artifact_kind,
        original_path=str(original),
        sealed_path=str(destination),
        artifact_sha256=identity,
        size_bytes=original.stat().st_size,
        fingerprint=fingerprint,
        original_result_sha256=_file_sha256(root / "result.json"),
        native_manifest_path=str(root / "manifest.json"),
        native_manifest_sha256=_file_sha256(root / "manifest.json"),
    )


def _check_restart_kind(
    path: Path, artifact_kind: str, *, fingerprint: RestartFingerprint | None = None
) -> None:
    if artifact_kind == "wavefunction":
        if path.name != "wavefunction.chk":
            raise ValueError(
                "The registered wavefunction must be an actual native checkpoint."
            )
        from pyscf.lib import chkfile

        molecule = chkfile.load_mol(str(path))
        scf = chkfile.load(str(path), "scf")
        if (
            molecule is None
            or not isinstance(scf, dict)
            or not {"mo_coeff", "mo_occ", "mo_energy"} <= set(scf)
        ):
            raise ValueError(
                "The actual checkpoint lacks native molecule/orbital data."
            )
        if fingerprint is not None:
            import json

            import numpy as np

            basis_bytes = (
                json.dumps(molecule._basis, indent=2, sort_keys=True, allow_nan=False)
                + "\n"
            ).encode()
            if (
                tuple(molecule.atom_symbol(index) for index in range(molecule.natm))
                != fingerprint.symbols
                or molecule.charge != fingerprint.charge
                or molecule.spin + 1 != fingerprint.multiplicity
                or not np.allclose(
                    molecule.atom_coords(unit="Bohr"),
                    fingerprint.geometry_bohr,
                    rtol=0,
                    atol=1e-12,
                )
                or sha256(basis_bytes).hexdigest()
                != fingerprint.basis_definition_sha256
            ):
                raise ValueError(
                    "Native checkpoint geometry/state/basis differs from its "
                    "declared final restart fingerprint."
                )
    elif artifact_kind == "hessian":
        if path.name != "hessian-hartree-bohr2.npy":
            raise ValueError("The registered Hessian must be an actual native array.")
        import numpy as np

        data = np.load(path, allow_pickle=False)
        if (
            data.dtype != np.dtype("float64")
            or data.ndim != 2
            or data.shape[0] != data.shape[1]
            or not np.isfinite(data).all()
        ):
            raise ValueError(
                "The native Hessian must be a finite float64 square array."
            )
        if fingerprint is not None and data.shape != (
            3 * len(fingerprint.symbols),
            3 * len(fingerprint.symbols),
        ):
            raise ValueError("Native Hessian dimensions differ from the fingerprint.")
    else:
        raise NotImplementedError(
            "This artifact kind has no qualified native registration adapter."
        )


def assess_restart_reuse(
    artifact: RestartArtifact, target: RestartFingerprint, policy: ReusePolicy
) -> dict[str, Any]:
    artifact = RestartArtifact.model_validate(artifact.model_dump(mode="json"))
    target = RestartFingerprint.model_validate(target.model_dump(mode="json"))
    policy = ReusePolicy.model_validate(policy.model_dump(mode="json"))
    from .application import source_identity

    root = Path(artifact.native_manifest_path).parent
    if (
        _file_sha256(Path(artifact.native_manifest_path))
        != artifact.native_manifest_sha256
        or _file_sha256(root / "result.json") != artifact.original_result_sha256
        or artifact.fingerprint.source_code_sha256 != source_identity()["code_sha256"]
    ):
        raise ValueError("Restart native provenance or current implementation changed.")
    actual = restart_fingerprint(root, recipe_sha256=artifact.fingerprint.recipe_sha256)
    if actual != artifact.fingerprint:
        raise ValueError(
            "Restart fingerprint differs from the authentic native definition."
        )
    if not Path(artifact.original_path).is_relative_to(root):
        raise ValueError(
            "The retained original is outside the native provenance bundle."
        )
    relative = Path(artifact.original_path).relative_to(root).as_posix()
    if relative not in {
        entry["path"] for entry in read_json(root / "manifest.json")["artifacts"]
    }:
        raise ValueError(
            "The retained original is absent from the native provenance bundle."
        )
    _check_restart_kind(
        Path(artifact.original_path), artifact.artifact_kind, fingerprint=actual
    )
    for name in ("original_path", "sealed_path"):
        path = _actual_file(Path(getattr(artifact, name)))
        if (
            path.stat().st_size != artifact.size_bytes
            or _file_sha256(path) != artifact.artifact_sha256
        ):
            raise ValueError(
                "Retained restart bytes changed; the original is "
                "not repaired or substituted."
            )
    if Path(artifact.original_path) == Path(artifact.sealed_path):
        raise ValueError(
            "The authentic original and sealed input must be distinct files."
        )
    if Path(artifact.sealed_path).stat().st_mode & 0o222:
        raise ValueError("The sealed restart input is not read-only.")
    source = artifact.fingerprint.model_dump(mode="json")
    destination = target.model_dump(mode="json")
    changed = tuple(key for key in source if source[key] != destination[key])
    allowed = (
        set(policy.allowed_initial_guess_changes)
        if policy.purpose == "initial_guess"
        else set()
    )
    if set(changed) - allowed:
        raise ValueError(
            "Incompatible restart fingerprint fields: " + ", ".join(changed)
        )
    if policy.purpose == "initial_guess" and artifact.artifact_kind not in {
        "wavefunction",
        "model_checkpoint",
    }:
        raise ValueError(
            "This artifact kind has no implemented initial-guess preparation policy."
        )
    return {
        "schema_version": "cochem.torq.restart-assessment/1",
        "compatible": True,
        "purpose": policy.purpose,
        "changed_fields": list(changed),
        "artifact_sha256": artifact.artifact_sha256,
        "source_fingerprint_sha256": artifact.fingerprint.identity_sha256,
        "target_fingerprint_sha256": target.identity_sha256,
        "new_physical_calculation_required": policy.purpose == "initial_guess",
        "engine_checkpoint_reused": False,
        "chemical_accuracy_established": False,
    }


def prepare_initial_guess(
    artifact: RestartArtifact,
    target: RestartFingerprint,
    output_path: str | Path,
    *,
    policy: ReusePolicy,
) -> dict[str, Any]:
    if policy.purpose != "initial_guess":
        raise ValueError(
            "Prepare a separate working copy only under an initial-guess policy."
        )
    assessment = assess_restart_reuse(artifact, target, policy)
    destination = Path(output_path).absolute()
    if destination == Path(artifact.original_path):
        raise ValueError("An immutable original cannot also be mutable engine output.")
    _copy_new(
        Path(artifact.sealed_path), destination, artifact.artifact_sha256, mode=0o600
    )
    return {
        **assessment,
        "prepared_working_copy": str(destination),
        "engine_checkpoint_reused": False,
    }
