"""Pinned native Pickett consumer; no vendor code, binary or license is bundled.

The bounded deck contract uses integer asymmetric-top quantum numbers, one
spin-free state, explicitly chosen Watson reduction and Ir axes. Native fitting
and mathematical regression are distinguished from independent measurements.
Missing parameter covariance is never presented as exact zero uncertainty.
"""

from __future__ import annotations

import ctypes
import errno
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field, StrictFloat, StrictInt, model_validator
from typing_extensions import Self

from ..domain import PrerequisiteError, canonical_json, digest, read_json
from ..publication import verify_publication_bundle
from ..scientific_contracts import Name, ScientificContract
from .rotational import rigid_rotor_catalog

SUPPORTED_COMMIT = "f5af2ff0f1acb1eb11ecffe7ef3b55d457801f7b"
SUPPORTED_SOURCE_SHA256 = (
    "9ce0f0b761a4094eee4aaf2d32017d80b47c1fca4083861d68c80a3accd46053"
)
_SCHEMA = "cochem.torq.pickett-native-artifacts/1"
_PARAMETERS = {
    "A": (10000, 1.0),
    "B": (20000, 1.0),
    "C": (30000, 1.0),
    "Delta_J": (200, -1.0),
    "Delta_JK": (1100, -1.0),
    "Delta_K": (2000, -1.0),
    "delta_J": (40100, -1.0),
    "delta_K": (41000, -1.0),
    "D_J": (200, -1.0),
    "D_JK": (1100, -1.0),
    "D_K": (2000, -1.0),
    "d1": (40100, 1.0),
    "d2": (50000, 1.0),
}
_QUARTIC = {
    "A": ("Delta_J", "Delta_JK", "Delta_K", "delta_J", "delta_K"),
    "S": ("D_J", "D_JK", "D_K", "d1", "d2"),
}


def _safe(path: str | Path, *, directory: bool = False) -> Path:
    target = Path(path).absolute()
    for component in (target, *target.parents):
        if component.is_symlink():
            raise ValueError("Pickett paths cannot traverse symlinks.")
    if directory and not target.is_dir():
        raise NotADirectoryError(target)
    return target


def _file_sha(path: Path) -> str:
    with path.open("rb") as stream:
        return sha256(stream.read()).hexdigest()


@dataclass(frozen=True)
class PickettInstallation:
    """Externally pinned build provenance, checked against actual retained bytes."""

    manifest_path: Path
    manifest_sha256: str
    source_directory: Path
    source_inventory: tuple[tuple[str, str, int], ...]
    spcat: Path
    spcat_sha256: str
    spfit: Path
    spfit_sha256: str
    compiler: str
    distribution_permission: str

    def verify(self) -> None:
        if _file_sha(_safe(self.manifest_path)) != self.manifest_sha256:
            raise ValueError("Pickett provisioning manifest changed.")
        records = []
        for name, expected, size in self.source_inventory:
            if Path(name).is_absolute() or ".." in Path(name).parts or "\\" in name:
                raise ValueError("Unsafe Pickett source inventory path.")
            path = _safe(self.source_directory / name)
            if (
                not path.is_file()
                or path.stat().st_size != size
                or _file_sha(path) != expected
            ):
                raise ValueError("Pickett source inventory bytes changed.")
            records.append({"path": name, "sha256": expected, "size_bytes": size})
        if digest(records) != SUPPORTED_SOURCE_SHA256:
            raise ValueError("Pickett source inventory is outside the pinned contract.")
        for path, expected in (
            (self.spcat, self.spcat_sha256),
            (self.spfit, self.spfit_sha256),
        ):
            path = _safe(path)
            if (
                not path.is_file()
                or not os.access(path, os.X_OK)
                or _file_sha(path) != expected
            ):
                raise PrerequisiteError(
                    "Actual pinned Pickett executable is unavailable or changed."
                )
            with path.open("rb") as stream:
                if stream.read(4) != b"\x7fELF":
                    raise PrerequisiteError(
                        "This Linux contract requires actual native ELF executables."
                    )


def load_pickett_installation(
    manifest_path: str | Path, *, expected_manifest_sha256: str
) -> PickettInstallation:
    path = _safe(manifest_path)
    if not path.is_file() or _file_sha(path) != expected_manifest_sha256:
        raise PrerequisiteError(
            "Actual independently pinned Pickett provisioning is required."
        )
    record = read_json(path)
    if (
        record.get("schema_version") != "cochem.torq.pickett-provisioning/1"
        or record.get("upstream_url") != "https://github.com/laserkelvin/Pickett"
        or record.get("source_commit") != SUPPORTED_COMMIT
        or record.get("source_inventory_sha256") != SUPPORTED_SOURCE_SHA256
        or record.get("distribution_permission") != "unverified_not_bundled"
        or record.get("binary_redistribution_authorized") is not False
    ):
        raise ValueError("Unsupported Pickett source/license/provisioning contract.")
    source = _safe(record["source_directory"], directory=True)
    pin = PickettInstallation(
        path,
        expected_manifest_sha256,
        source,
        tuple(
            (r["path"], r["sha256"], r["size_bytes"])
            for r in record["source_inventory"]
        ),
        _safe(record["spcat"]["path"]),
        record["spcat"]["sha256"],
        _safe(record["spfit"]["path"]),
        record["spfit"]["sha256"],
        record["compiler"],
        record["distribution_permission"],
    )
    pin.verify()
    return pin


def configured_pickett_installation() -> PickettInstallation:
    path = os.environ.get("COCHEM_TORQ_PICKETT_PROVISIONING_JSON")
    expected = os.environ.get("COCHEM_TORQ_PICKETT_PROVISIONING_SHA256")
    if not path or not expected:
        raise PrerequisiteError(
            "Pickett native tests require explicit source/binary provisioning pins."
        )
    return load_pickett_installation(path, expected_manifest_sha256=expected)


class PickettModel(ScientificContract):
    schema_version: Literal["cochem.torq.pickett-model/1"] = (
        "cochem.torq.pickett-model/1"
    )
    title: str = Field(min_length=1, max_length=50, pattern=r"^[ -~]+$")
    reduction: Literal["A", "S"]
    representation: Literal["Ir"] = "Ir"
    constant_observable: Literal["Be", "B0", "experiment", "mathematical_model"]
    parameter_origin: Literal["ab_initio", "experimental_fit", "mathematical_model"]
    constants_mhz: tuple[StrictFloat, StrictFloat, StrictFloat]
    dipole_abc_debye: tuple[StrictFloat, StrictFloat, StrictFloat]
    quartic_mhz: dict[str, StrictFloat] | None = None
    parameter_covariance_mhz2: tuple[tuple[StrictFloat, ...], ...] | None = None
    covariance_origin: Name | None = None
    source_bundle: str | None = None
    source_manifest_sha256: str | None = Field(default=None, pattern="^[0-9a-f]{64}$")
    parameter_unit: Literal["MHz"] = "MHz"
    dipole_unit: Literal["debye"] = "debye"
    spin_policy: Literal["spin_free_no_nuclear_statistical_restrictions"] = (
        "spin_free_no_nuclear_statistical_restrictions"
    )

    @model_validator(mode="after")
    def convention_and_uncertainty(self) -> Self:
        a, b, c = self.constants_mhz
        if not a >= b >= c > 0:
            raise ValueError(
                "Defined nonlinear constants must follow principal A >= B >= C > 0."
            )
        if not any(value != 0 for value in self.dipole_abc_debye):
            raise ValueError(
                "A zero dipole has no electric-dipole catalog; do not invent lines."
            )
        if self.quartic_mhz is not None and set(self.quartic_mhz) != set(
            _QUARTIC[self.reduction]
        ):
            raise ValueError(
                "A quartic model requires all five parameters "
                "of its explicit reduction."
            )
        if self.parameter_origin == "mathematical_model":
            if (
                self.constant_observable != "mathematical_model"
                or self.source_bundle is not None
            ):
                raise ValueError(
                    "Mathematical parameters cannot be labeled "
                    "molecular Be/B0 or experiment."
                )
        elif self.parameter_origin == "ab_initio":
            if (
                self.constant_observable not in {"Be", "B0"}
                or not self.source_bundle
                or not self.source_manifest_sha256
            ):
                raise ValueError(
                    "Ab initio parameters require their exact "
                    "verified native publication source."
                )
            if self.quartic_mhz is not None:
                raise ValueError(
                    "Ab initio quartic export awaits a validated "
                    "tensor-to-Pickett source contract."
                )
        elif (
            self.constant_observable != "experiment"
            or not self.source_bundle
            or not self.source_manifest_sha256
        ):
            raise ValueError(
                "Experimental fitting requires its exact verified native fit artifact."
            )
        count = 3 + (5 if self.quartic_mhz is not None else 0)
        if (self.parameter_covariance_mhz2 is not None) != (
            self.covariance_origin is not None
        ):
            raise ValueError(
                "Parameter covariance requires explicit provenance; "
                "missing stays absent."
            )
        if self.parameter_covariance_mhz2 is not None:
            cov = np.asarray(self.parameter_covariance_mhz2)
            if cov.shape != (count, count) or not np.allclose(
                cov, cov.T, atol=1e-14, rtol=1e-12
            ):
                raise ValueError(
                    "Covariance must be complete and symmetric "
                    "in exact parameter order."
                )
            try:
                np.linalg.cholesky(cov)
            except np.linalg.LinAlgError as exc:
                raise ValueError(
                    "This native full-covariance export requires "
                    "positive definite covariance."
                ) from exc
        return self

    def names(self) -> tuple[str, ...]:
        return ("A", "B", "C") + (
            _QUARTIC[self.reduction] if self.quartic_mhz is not None else ()
        )

    def values(self) -> tuple[float, ...]:
        return (
            self.constants_mhz
            + tuple(self.quartic_mhz[name] for name in _QUARTIC[self.reduction])
            if self.quartic_mhz is not None
            else self.constants_mhz
        )


class CatalogSettings(ScientificContract):
    temperature_kelvin: StrictFloat = Field(gt=0, le=1000)
    maximum_j: StrictInt = Field(ge=1, le=30)
    maximum_frequency_ghz: StrictFloat = Field(gt=0, le=1e5)
    log10_intensity_cutoff_nm2_mhz: StrictFloat = Field(default=-30.0, ge=-100, le=10)
    species_tag: StrictInt = Field(default=999999, ge=1, le=999999)
    partition_function: StrictFloat = Field(gt=0)
    partition_origin: Name
    wall_seconds: StrictFloat = Field(default=30.0, gt=0, le=600)


def decode_quantum_number(field: str) -> int:
    """Decode the actual pinned two-column codec, including a0=-10 and z9=-269."""
    if len(field) != 2 or field == "**":
        raise ValueError("Unsupported/overflowed native quantum-number field.")
    left, right = field[0], field[1]
    if not right.isascii() or not right.isdigit():
        raise ValueError("Malformed native quantum-number field.")
    digit = int(right)
    if left == " ":
        return digit
    if left == "-":
        return -digit
    if left.isascii() and left.isdigit():
        return int(field)
    if "a" <= left <= "z":
        return -((ord(left) - ord("a") + 1) * 10 + digit)
    if "A" <= left <= "Z":
        return (ord(left) - ord("A") + 10) * 10 + digit
    raise ValueError("Malformed native quantum-number field.")


def encode_quantum_number(value: int) -> str:
    if type(value) is not int or not -269 <= value <= 359:
        raise ValueError("Native two-column quantum numbers span -269 through 359.")
    if -9 <= value <= 99:
        return f"{value:2d}"
    if value < -9:
        tens, digit = divmod(-value, 10)
        return chr(ord("a") + tens - 1) + str(digit)
    tens, digit = divmod(value, 10)
    return chr(ord("A") + tens - 10) + str(digit)


def _quantum_state(values: Sequence[int]) -> None:
    if len(values) != 3:
        raise ValueError("Only integer asymmetric-top J,Ka,Kc states are supported.")
    j, ka, kc = values
    if (
        any(type(v) is not int for v in values)
        or not 0 <= ka <= j <= 359
        or not 0 <= kc <= j
        or ka + kc not in {j, j + 1}
    ):
        raise ValueError("Invalid asymmetric-top quantum-state identity.")


def parse_catalog(raw: bytes, *, covariance_available: bool) -> list[dict[str, Any]]:
    records = []
    identities = set()
    for number, line in enumerate(raw.decode("ascii", errors="strict").splitlines(), 1):
        try:
            if (
                len(line) < 79
                or line[79:].strip()
                or int(line[51:55]) != 303
                or line[61:67].strip()
                or line[73:79].strip()
            ):
                raise ValueError(
                    "Only the pinned 79-column spin-free "
                    "QNFMT=303 contract is supported."
                )
            freq, error, logint, lower = (
                float(line[a:b]) for a, b in ((0, 13), (13, 21), (21, 29), (31, 41))
            )
            if (
                not all(np.isfinite(v) for v in (freq, error, logint, lower))
                or freq <= 0
                or error < 0
                or lower < 0
            ):
                raise ValueError("Invalid/nonfinite native spectral observable.")
            if int(line[29:31]) != 3 or int(line[44:51]) <= 0:
                raise ValueError(
                    "Predicted nonlinear rotor and positive native tag are required."
                )
            upper = tuple(decode_quantum_number(line[i : i + 2]) for i in (55, 57, 59))
            lower_qn = tuple(
                decode_quantum_number(line[i : i + 2]) for i in (67, 69, 71)
            )
            _quantum_state(upper)
            _quantum_state(lower_qn)
            if abs(upper[0] - lower_qn[0]) > 1 or upper[0] == lower_qn[0] == 0:
                raise ValueError("Unsupported electric-dipole delta-J assignment.")
            key = upper + lower_qn
            if key in identities:
                raise ValueError("Duplicate native transition identity.")
            identities.add(key)
            degeneracy = int(line[41:44])
            if degeneracy != 2 * upper[0] + 1:
                raise ValueError("Native degeneracy differs from the spin-free model.")
            records.append(
                {
                    "upper_j_ka_kc": list(upper),
                    "lower_j_ka_kc": list(lower_qn),
                    "frequency_mhz": freq,
                    "native_error_column_mhz": error,
                    "frequency_standard_uncertainty_mhz": error
                    if covariance_available
                    else None,
                    "uncertainty_status": "native_linearized_parameter_covariance"
                    if covariance_available
                    else "unavailable_parameter_covariance_not_supplied",
                    "log10_integrated_intensity_nm2_mhz": logint,
                    "lower_energy_cm1": lower,
                    "upper_degeneracy": degeneracy,
                    "species_tag": int(line[44:51]),
                    "quantum_number_format": 303,
                }
            )
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Native catalog line {number}: {exc}") from exc
    if not records:
        raise ValueError("Native catalog has no supported transitions.")
    return records


def _deck(raw: str) -> bytes:
    lines = raw.splitlines()
    if any(not line.isascii() or len(line) > 79 for line in lines):
        raise ValueError(
            "Every native input line must be ASCII and at most 79 characters."
        )
    # The pinned upstream SPFIT filbak has an 82-byte buffer. Restrict generated
    # lines before execution rather than modifying or bundling vendor code.
    return ("\n".join(lines) + "\n").encode("ascii")


def _parameter_deck(
    model: PickettModel, *, fit_lines: int = 0, rejection_limit: float = 100.0
) -> bytes:
    names = model.names()
    values = model.values()
    signs = np.asarray([_PARAMETERS[name][1] for name in names])
    covariance = (
        np.asarray(model.parameter_covariance_mhz2)
        if model.parameter_covariance_mhz2 is not None
        else None
    )
    if covariance is not None:
        covariance = signs[:, None] * covariance * signs[None, :]
        sigma = np.sqrt(np.diag(covariance))
    else:
        sigma = np.full(len(names), 1e6 if fit_lines else 0.0)
    lines = [
        model.title,
        f"{len(names)} {fit_lines} 20 0 0 {rejection_limit:.8g} 1 1",
        f"{model.reduction.lower()} 1 1 0 30 0 1 1 1 0 0 0 0",
    ]
    for name, value, uncertainty in zip(names, values, sigma):
        pid, sign = _PARAMETERS[name]
        lines.append(f"{pid:8d} {sign * value:.16g} {uncertainty:.10g} /{name}")
    if covariance is not None:
        # The pinned getvar/putvar store a lower factor L with C=L.T@L,
        # normalized by column uncertainties, not numpy's C=L@L.T convention.
        factor = np.linalg.cholesky(covariance[::-1, ::-1]).T[::-1, ::-1]
        normalized = factor / sigma[None, :]
        packed = [
            float(normalized[i, j]) for i in range(len(names)) for j in range(i + 1)
        ]
        lines.extend(
            "".join(f"{v:10.7f}" for v in packed[i : i + 7])
            for i in range(0, len(packed), 7)
        )
    return _deck("\n".join(lines))


def _intensity_deck(model: PickettModel, settings: CatalogSettings) -> bytes:
    return _deck(
        "\n".join(
            [
                model.title,
                f"1 {settings.species_tag} {settings.partition_function:.16g} "
                f"0 {settings.maximum_j} "
                f"{settings.log10_intensity_cutoff_nm2_mhz:g} "
                f"{settings.log10_intensity_cutoff_nm2_mhz:g} "
                f"{settings.maximum_frequency_ghz:g} "
                f"{settings.temperature_kelvin:g}",
                *(
                    f"{index} {value:.16g}"
                    for index, value in enumerate(model.dipole_abc_debye, 1)
                ),
            ]
        )
    )


def _source_model(model: PickettModel) -> None:
    if model.parameter_origin == "mathematical_model":
        return
    if model.parameter_origin == "experimental_fit":
        manifest = verify_native_run(
            model.source_bundle or "",
            expected_manifest_sha256=model.source_manifest_sha256 or "",
        )
        result = read_json(Path(model.source_bundle or "") / "analysis.json")
        if (
            manifest["program"] != "SPFIT"
            or result.get("fit_origin") != "curated_experimental_measurements"
            or result.get("covariance_status") != "full_rank_native_local_linearization"
        ):
            raise ValueError(
                "Experimental constants require a full-rank genuine experimental fit."
            )
        if (
            tuple(result["constants_mhz"]) != model.constants_mhz
            or result["reduction"] != model.reduction
        ):
            raise ValueError(
                "Experimental constants/reduction differ from the native fit."
            )
        if model.parameter_covariance_mhz2 is not None and not np.array_equal(
            np.asarray(model.parameter_covariance_mhz2),
            np.asarray(result["parameter_covariance_mhz2"]),
        ):
            raise ValueError(
                "Experimental covariance differs from the actual native fit."
            )
        return
    bundle = Path(model.source_bundle or "")
    verify_publication_bundle(
        bundle, expected_manifest_sha256=model.source_manifest_sha256 or ""
    )
    result = read_json(bundle / "shard/result.json")
    stage = result["stages"][
        "equilibrium_constants"
        if model.constant_observable == "Be"
        else "ground_state_constants"
    ]
    if (
        stage["status"] != "available"
        or stage["observable"] != model.constant_observable
        or tuple(stage["value"]["constants_mhz"]) != model.constants_mhz
    ):
        raise ValueError(
            "Ab initio constants must match the actual typed available stage."
        )
    native = result.get("native_result") or {}
    dipole = native.get("dipole_debye")
    if dipole is None:
        raise ValueError("Actual native electronic dipole evidence is unavailable.")
    axes = np.asarray(stage["value"]["principal_axes_columns"])
    molecule = result["molecule"]
    if molecule["charge"] != 0:
        raise ValueError(
            "Charged-system dipole origin requires a separately validated contract."
        )
    if not np.allclose(
        np.asarray(dipole) @ axes, model.dipole_abc_debye, atol=1e-12, rtol=1e-12
    ):
        raise ValueError(
            "Principal-axis dipoles differ from actual native density evidence."
        )
    if model.parameter_covariance_mhz2 is not None:
        raise ValueError(
            "Ab initio covariance needs its independent calibrated source contract."
        )
    verify_publication_bundle(
        bundle, expected_manifest_sha256=model.source_manifest_sha256 or ""
    )


def _inventory(root: Path) -> list[dict[str, Any]]:
    result = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Native Pickett artifacts forbid symlinks.")
        if path.is_file() and path != root / "manifest.json":
            result.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": _file_sha(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    return result


def _publish(staging: Path, target: Path) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    if not hasattr(library, "renameat2"):
        raise PrerequisiteError(
            "Immutable native publication requires Linux renameat2."
        )
    rename = library.renameat2
    rename.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(staging), -100, os.fsencode(target), 1):
        err = ctypes.get_errno()
        if err == errno.EEXIST:
            raise FileExistsError("Native Pickett artifacts cannot be overwritten.")
        raise OSError(err, os.strerror(err))
    fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _execute(
    installation: PickettInstallation,
    program: Literal["SPCAT", "SPFIT"],
    inputs: Mapping[str, bytes],
    target: str | Path,
    *,
    wall_seconds: float,
    metadata: Mapping[str, Any],
    analyze: Any,
    retained_sources: Mapping[str, bytes] | None = None,
) -> dict[str, Any]:
    destination = _safe(target)
    _safe(destination.parent)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if destination.exists():
        raise FileExistsError("Native Pickett artifacts cannot be overwritten.")
    installation.verify()
    staging = Path(tempfile.mkdtemp(prefix=".torq-pickett-", dir=destination.parent))
    start = time.monotonic()
    try:
        (staging / "inputs").mkdir(mode=0o700)
        for name, raw in inputs.items():
            (staging / "inputs" / name).write_bytes(raw)
            (staging / name).write_bytes(raw)
        for name, raw in (retained_sources or {}).items():
            relative = Path(name)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or "\\" in name
                or relative.parts[0] != "sources"
            ):
                raise ValueError("Unsafe retained native source path.")
            path = staging / relative
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            path.write_bytes(raw)
        executable = installation.spcat if program == "SPCAT" else installation.spfit
        returncode: int | None = None
        failure: str | None = None
        with (
            (staging / "stdout.log").open("wb") as stdout,
            (staging / "stderr.log").open("wb") as stderr,
        ):
            process = subprocess.Popen(
                [str(executable), "molecule"],
                cwd=staging,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
                env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
            )
            try:
                returncode = process.wait(timeout=wall_seconds)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                returncode = process.wait(timeout=10)
                failure = "Actual native wall-time budget exhausted."
        if returncode != 0:
            failure = failure or f"Actual native {program} exited {returncode}."
        if failure is None:
            try:
                analysis = analyze(staging)
            except (ValueError, OSError) as exc:
                failure = f"Native output verification failed: {exc}"
                analysis = None
        else:
            analysis = None
        installation.verify()
        (staging / "request.json").write_bytes(canonical_json(dict(metadata)) + b"\n")
        (staging / "analysis.json").write_bytes(canonical_json(analysis) + b"\n")
        receipt = {
            "schema_version": _SCHEMA,
            "program": program,
            "source_commit": SUPPORTED_COMMIT,
            "source_inventory_sha256": SUPPORTED_SOURCE_SHA256,
            "provisioning_manifest_sha256": installation.manifest_sha256,
            "executable_sha256": installation.spcat_sha256
            if program == "SPCAT"
            else installation.spfit_sha256,
            "compiler": installation.compiler,
            "distribution_permission": installation.distribution_permission,
            "binary_redistribution_authorized": False,
            "returncode": returncode,
            "status": "failed" if failure else "completed_native_unqualified",
            "failure": failure,
            "elapsed_seconds": time.monotonic() - start,
            "input_sha256": {
                name: sha256(raw).hexdigest() for name, raw in inputs.items()
            },
            "files": _inventory(staging),
            "experimental_accuracy_established": False,
            "identification_ready": False,
        }
        manifest = canonical_json(receipt) + b"\n"
        (staging / "manifest.json").write_bytes(manifest)
        for path in staging.rglob("*"):
            if path.is_file():
                path.chmod(0o600)
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
        _publish(staging, destination)
        return {
            "manifest_sha256": sha256(manifest).hexdigest(),
            "status": receipt["status"],
            "failure": failure,
            "analysis": analysis,
        }
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def verify_native_run(
    directory: str | Path, *, expected_manifest_sha256: str
) -> dict[str, Any]:
    root = _safe(directory, directory=True)
    manifest = _safe(root / "manifest.json")
    if _file_sha(manifest) != expected_manifest_sha256:
        raise ValueError(
            "Native Pickett manifest differs from its independently retained digest."
        )
    payload = read_json(manifest)
    if not isinstance(payload, dict):
        raise ValueError("Native Pickett manifest must be an actual JSON object.")
    record: dict[str, Any] = payload
    if record.get("schema_version") != _SCHEMA or record.get("files") != _inventory(
        root
    ):
        raise ValueError("Native Pickett artifact inventory/hash mismatch.")
    if (
        record.get("experimental_accuracy_established") is not False
        or record.get("identification_ready") is not False
        or record.get("source_commit") != SUPPORTED_COMMIT
    ):
        raise ValueError("Native Pickett receipt cannot invent method qualification.")
    if (
        record.get("program") not in {"SPCAT", "SPFIT"}
        or record.get("source_inventory_sha256") != SUPPORTED_SOURCE_SHA256
    ):
        raise ValueError(
            "Native Pickett program/source is outside the pinned contract."
        )
    for name, expected in record["input_sha256"].items():
        if Path(name).name != name or _file_sha(root / "inputs" / name) != expected:
            raise ValueError("Native input was changed by the calculation.")
    request = read_json(root / "request.json")
    analysis = read_json(root / "analysis.json")
    model = PickettModel.model_validate(request["model"])
    if record["status"] == "completed_native_unqualified":
        if record.get("returncode") != 0 or record.get("failure") is not None:
            raise ValueError(
                "Completed native evidence contradicts its process result."
            )
        if record["program"] == "SPCAT":
            _source_model(model)
            settings = CatalogSettings.model_validate(request["settings"])
            expected_inputs = {
                "molecule.var": _parameter_deck(model),
                "molecule.int": _intensity_deck(model, settings),
            }
            expected_analysis = _catalog_analysis(root, model, settings)
        else:
            if request.get("fit_origin") != (
                "verified_native_prediction_regression_not_experiment"
            ):
                raise ValueError(
                    "Curated experimental fitting is not qualified by this adapter."
                )
            lines = tuple(FitLine.model_validate(line) for line in request["lines"])
            authentic = {}
            for source_id, source in request["native_catalog_sources"].items():
                relative = Path(source["retained_path"])
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or "\\" in source["retained_path"]
                    or len(relative.parts) != 2
                    or relative.parts[0] != "sources"
                ):
                    raise ValueError("Unsafe retained native source path.")
                native_root = root / relative
                native = verify_native_run(
                    native_root,
                    expected_manifest_sha256=source["manifest_sha256"],
                )
                if (
                    native["program"] != "SPCAT"
                    or native["status"] != "completed_native_unqualified"
                ):
                    raise ValueError("Fit sources must be completed native catalogs.")
                authentic[source_id] = read_json(native_root / "analysis.json")["lines"]
            expected_inputs = {
                "molecule.par": _parameter_deck(
                    model,
                    fit_lines=len(lines),
                    rejection_limit=request["rejection_limit"],
                ),
                "molecule.lin": _line_deck(lines, authentic),
            }
            expected_analysis = _fit_output(
                root, model, "verified_native_prediction_regression_not_experiment"
            )
        if record["input_sha256"] != {
            name: sha256(raw).hexdigest() for name, raw in expected_inputs.items()
        }:
            raise ValueError("Native input decks differ from the retained request.")
        if expected_analysis != analysis:
            raise ValueError(
                "Native analysis differs from actual retained output values."
            )
    elif (
        record["status"] != "failed"
        or analysis is not None
        or not record.get("failure")
    ):
        raise ValueError(
            "Failed native evidence must retain an explicit failure and no analysis."
        )
    return record


def _catalog_analysis(
    root: Path, model: PickettModel, settings: CatalogSettings
) -> dict[str, Any]:
    rows = parse_catalog(
        (root / "molecule.cat").read_bytes(),
        covariance_available=model.parameter_covariance_mhz2 is not None,
    )
    if any(
        max(row["upper_j_ka_kc"][0], row["lower_j_ka_kc"][0]) > settings.maximum_j
        or row["species_tag"] != settings.species_tag
        or row["frequency_mhz"] > settings.maximum_frequency_ghz * 1000 + 0.000051
        for row in rows
    ):
        raise ValueError("Native catalog violated its explicit domain.")
    return {
        "lines": rows,
        "line_count": len(rows),
        "parameter_origin": model.parameter_origin,
        "constant_observable": model.constant_observable,
        "reduction": model.reduction,
        "representation": model.representation,
        "partition_function_supplied": settings.partition_function,
        "partition_origin": settings.partition_origin,
        "intensity_unit": "log10(nm^2 MHz)",
        "uncertainty_includes_model_error": False,
        "identification_ready": False,
    }


def run_spcat(
    installation: PickettInstallation,
    model: PickettModel,
    settings: CatalogSettings,
    destination: str | Path,
) -> dict[str, Any]:
    # Copy/revalidate the mutable quartic mapping before generating native decks.
    model = PickettModel.model_validate(model.model_dump(mode="python"))
    _source_model(model)
    var = _parameter_deck(model)
    intensity = _intensity_deck(model, settings)

    return _execute(
        installation,
        "SPCAT",
        {"molecule.var": var, "molecule.int": intensity},
        destination,
        wall_seconds=settings.wall_seconds,
        metadata={
            "model": model.model_dump(mode="json"),
            "settings": settings.model_dump(mode="json"),
        },
        analyze=lambda root: _catalog_analysis(root, model, settings),
    )


class FitLine(ScientificContract):
    upper_j_ka_kc: tuple[StrictInt, StrictInt, StrictInt]
    lower_j_ka_kc: tuple[StrictInt, StrictInt, StrictInt]
    frequency_mhz: StrictFloat = Field(gt=0)
    standard_uncertainty_or_regression_tolerance_mhz: StrictFloat = Field(gt=0)
    source_id: Name
    origin: Literal[
        "curated_experimental_measurement", "verified_native_regression_prediction"
    ]

    @model_validator(mode="after")
    def identity(self) -> Self:
        _quantum_state(self.upper_j_ka_kc)
        _quantum_state(self.lower_j_ka_kc)
        if max(*self.upper_j_ka_kc, *self.lower_j_ka_kc) > 30:
            raise ValueError(
                "This fitting deck is bounded to quantum numbers through 30."
            )
        return self


def _line_deck(
    lines: Sequence[FitLine], authentic: Mapping[str, list[dict[str, Any]]]
) -> bytes:
    keys = set()
    line_deck = []
    for line in lines:
        if line.origin != "verified_native_regression_prediction":
            raise ValueError("Experimental targets require their separate contract.")
        key = line.upper_j_ka_kc + line.lower_j_ka_kc
        if key in keys:
            raise ValueError(
                "Duplicate fit transition identities are not independent observations."
            )
        keys.add(key)
        matches = [
            r
            for r in authentic.get(line.source_id, [])
            if tuple(r["upper_j_ka_kc"]) == line.upper_j_ka_kc
            and tuple(r["lower_j_ka_kc"]) == line.lower_j_ka_kc
            and r["frequency_mhz"] == line.frequency_mhz
        ]
        if len(matches) != 1:
            raise ValueError(
                "Fit regression values must equal the actual verified native catalog."
            )
        quanta = key + (0,) * 6
        line_deck.append(
            "".join(f"{q:3d}" for q in quanta)
            + f" {line.frequency_mhz:.10g} "
            + f"{line.standard_uncertainty_or_regression_tolerance_mhz:.8g} 1"
        )
    return _deck("\n".join(line_deck))


def _fit_output(root: Path, model: PickettModel, origin: str) -> dict[str, Any]:
    stdout = (root / "stdout.log").read_text(encoding="ascii")
    if "FIT COMPLETE" not in stdout:
        raise ValueError("Native SPFIT did not report its actual completion.")
    lines = (root / "molecule.var").read_text(encoding="ascii").splitlines()
    count = len(model.names())
    if int(lines[1].split()[0]) != count:
        raise ValueError("Native fitted parameter inventory changed.")
    values, sigma = [], []
    for name, row in zip(model.names(), lines[3 : 3 + count]):
        numbers = row.split("/", 1)[0].split()
        if len(numbers) != 3 or int(numbers[0]) != _PARAMETERS[name][0]:
            raise ValueError("Native fit parameter identities differ from input.")
        values.append(float(numbers[1]) * _PARAMETERS[name][1])
        sigma.append(float(numbers[2]))
    if not all(np.isfinite(v) for v in values + sigma) or min(sigma) <= 0:
        raise ValueError("Native fitted values/errors are nonfinite or unavailable.")
    packed: list[float] = []
    for line in lines[3 + count :]:
        if len(line) % 10:
            raise ValueError(
                "Native covariance factor has malformed fixed-width fields."
            )
        packed.extend(float(line[i : i + 10]) for i in range(0, len(line), 10))
    if len(packed) != count * (count + 1) // 2 or not np.isfinite(packed).all():
        raise ValueError("Native full covariance factor is missing or malformed.")
    factor = np.zeros((count, count))
    index = 0
    for i in range(count):
        for j in range(i + 1):
            factor[i, j] = packed[index] * sigma[j]
            index += 1
    covariance = factor.T @ factor
    signs = np.asarray([_PARAMETERS[n][1] for n in model.names()])
    covariance = signs[:, None] * covariance * signs[None, :]
    rank = int(np.linalg.matrix_rank(factor))
    fit_text = (root / "molecule.fit").read_text(encoding="ascii")
    rms = re.findall(r"MICROWAVE RMS\s*=\s*([\d.Ee+\-]+)", fit_text)
    if not rms:
        raise ValueError("Native fit has no actual residual diagnostic.")
    final_iteration = fit_text.rsplit("EXP.FREQ.", 1)[-1]
    rejected = len(re.findall(r"NEXT LINE NOT USED IN FIT", final_iteration))
    return {
        "parameter_names": list(model.names()),
        "parameters_mhz": values,
        "constants_mhz": values[:3],
        "parameter_covariance_mhz2": covariance.tolist(),
        "covariance_status": "full_rank_native_local_linearization"
        if rank == count
        else "rank_deficient_native_factor",
        "covariance_rank": rank,
        "reduction": model.reduction,
        "representation": model.representation,
        "fit_origin": origin,
        "native_microwave_rms_mhz": float(rms[-1]),
        "native_final_rejection_count": rejected,
        "uncertainty_includes_model_error": False,
        "experimental_accuracy_established": False,
    }


def run_spfit(
    installation: PickettInstallation,
    model: PickettModel,
    lines: Sequence[FitLine],
    destination: str | Path,
    *,
    wall_seconds: float = 30.0,
    rejection_limit: float = 100.0,
    native_catalog_sources: Mapping[str, tuple[str | Path, str]] | None = None,
) -> dict[str, Any]:
    model = PickettModel.model_validate(model.model_dump(mode="python"))
    lines = tuple(lines)
    if (
        not lines
        or len(lines) > 100000
        or not np.isfinite(wall_seconds)
        or not 0 < wall_seconds <= 600
        or not np.isfinite(rejection_limit)
        or not 0 < rejection_limit <= 1e9
    ):
        raise ValueError(
            "Fit requires actual lines and finite explicit execution/rejection budgets."
        )
    origins = {line.origin for line in lines}
    if len(origins) != 1:
        raise ValueError(
            "Experimental observations and native regression targets cannot be pooled."
        )
    if origins != {"verified_native_regression_prediction"}:
        raise PrerequisiteError(
            "Experimental fitting awaits its curated "
            "measurement import/attestation contract."
        )
    if model.parameter_origin != "mathematical_model" or not native_catalog_sources:
        raise ValueError(
            "Native regression requires an explicit mathematical "
            "model and verified catalogs."
        )
    authentic: dict[str, list[dict[str, Any]]] = {}
    retained_sources = {}
    source_records = {}
    for index, (source_id, (path, expected)) in enumerate(
        native_catalog_sources.items()
    ):
        manifest = verify_native_run(path, expected_manifest_sha256=expected)
        if (
            manifest["program"] != "SPCAT"
            or manifest["status"] != "completed_native_unqualified"
        ):
            raise ValueError(
                "Regression source must be a completed genuine SPCAT calculation."
            )
        relative = f"sources/catalog-{index:04d}"
        source_records[source_id] = {
            "retained_path": relative,
            "manifest_sha256": expected,
        }
        source_root = _safe(path, directory=True)
        raw_manifest = (source_root / "manifest.json").read_bytes()
        if sha256(raw_manifest).hexdigest() != expected:
            raise ValueError("Native source manifest changed during capture.")
        retained_sources[f"{relative}/manifest.json"] = raw_manifest
        for entry in manifest["files"]:
            raw = (source_root / entry["path"]).read_bytes()
            if (
                sha256(raw).hexdigest() != entry["sha256"]
                or len(raw) != entry["size_bytes"]
            ):
                raise ValueError("Native source file changed during capture.")
            retained_sources[f"{relative}/{entry['path']}"] = raw
        verify_native_run(source_root, expected_manifest_sha256=expected)
        authentic[source_id] = json.loads(
            retained_sources[f"{relative}/analysis.json"]
        )["lines"]
    par = _parameter_deck(model, fit_lines=len(lines), rejection_limit=rejection_limit)
    lin = _line_deck(lines, authentic)
    return _execute(
        installation,
        "SPFIT",
        {"molecule.par": par, "molecule.lin": lin},
        destination,
        wall_seconds=wall_seconds,
        metadata={
            "model": model.model_dump(mode="json"),
            "lines": [line.model_dump(mode="json") for line in lines],
            "fit_origin": "verified_native_prediction_regression_not_experiment",
            "rejection_limit": rejection_limit,
            "native_catalog_sources": source_records,
        },
        analyze=lambda root: _fit_output(
            root, model, "verified_native_prediction_regression_not_experiment"
        ),
        retained_sources=retained_sources,
    )


def compare_rigid_rotor_reference(
    catalog: Sequence[Mapping[str, Any]], model: PickettModel, settings: CatalogSettings
) -> dict[str, Any]:
    """Independent finite-J numerical comparison; no experimental accuracy claim."""
    if model.quartic_mhz is not None:
        raise ValueError("Rigid-rotor reference cannot validate distortion parameters.")
    independent = rigid_rotor_catalog(
        model.constants_mhz,
        model.dipole_abc_debye,
        temperature_kelvin=settings.temperature_kelvin,
        J_max=settings.maximum_j,
        constant_observable="Be",
    )
    if not np.isclose(
        independent.partition_function, settings.partition_function, rtol=1e-12, atol=0
    ):
        raise ValueError(
            "Comparison requires identical explicit finite-J partition normalization."
        )
    residuals = []
    for row in catalog:
        possible = [
            line
            for line in independent.lines
            if line.upper_J == row["upper_j_ka_kc"][0]
            and line.lower_J == row["lower_j_ka_kc"][0]
        ]
        if not possible:
            raise ValueError(
                "Native quantum states have no independent finite-J reference."
            )
        closest = min(
            possible, key=lambda line: abs(line.frequency_mhz - row["frequency_mhz"])
        )
        # Keep the actual Pickett legacy conversion coefficient separate from
        # contemporary SI constants rather than silently changing engine output.
        predicted = np.log10(
            4.16231e-5
            * closest.frequency_mhz
            * closest.relative_absorption_weight_debye2
        )
        residuals.append(
            {
                "frequency_residual_mhz": row["frequency_mhz"] - closest.frequency_mhz,
                "log10_integrated_intensity_residual": row[
                    "log10_integrated_intensity_nm2_mhz"
                ]
                - float(predicted),
            }
        )
    return {
        "native_line_count": len(catalog),
        "independent_in_band_line_count": sum(
            line.frequency_mhz <= settings.maximum_frequency_ghz * 1000
            for line in independent.lines
        ),
        "maximum_absolute_frequency_residual_mhz": max(
            abs(r["frequency_residual_mhz"]) for r in residuals
        ),
        "maximum_absolute_log10_integrated_intensity_residual": max(
            abs(r["log10_integrated_intensity_residual"]) for r in residuals
        ),
        "native_frequency_rounding_mhz": 0.0001,
        "native_log10_intensity_rounding": 0.0001,
        "native_legacy_intensity_coefficient": 4.16231e-5,
        "comparison": "nearest energy separation within actual upper/lower J; "
        "not an independent Ka/Kc assignment proof",
        "experimental_accuracy_established": False,
    }
