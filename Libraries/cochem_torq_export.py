"""CoChem-TORQ: Cryptographic Payload Synthesizer & Deliverable Gateway.

Phase 9 (Stages 5.5 - 6.0) Specification
---------------------------------------------------------------------
Aggregates forward predictions, exact physics tensors, PyArrow .parquet catalogs,
and .var/.int files into a unified, cryptographically locked export payload
specifically designed for seamless ingestion by CoChem-SpycFit.

Implements:
1. Kraitchman coordinate calculations with explicit unavailable coordinates
   and rejection of singular inversions.
2. OOM-proof PGOPHER XML skeleton generation using PyArrow parquet metadata.
3. Provenance lock manifest generation with deterministic strict JSON.
4. Deterministic .tar.zst payload bundling with normalized POSIX metadata.
5. Cryptographic payload verification raising CoChemIntegrityError.
6. Legacy TorqExporter, PESStore, and export_qcschema integration.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import os
import tarfile
import warnings
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py  # type: ignore[import-untyped]
import numpy.typing as npt
import pyarrow.parquet as pq
import zstandard as zstd

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(levelname)s: [CoChem-TORQ-Export] %(message)s"
)
logger = logging.getLogger("TorqExport")

ARTIFACTS_DIR = os.environ.get(
    "COCHEM_ARTIFACTS_DIR", str(Path.home() / "cochem_artifacts")
)


# ============================================================================
# Custom Warning & Exception Classes
# ============================================================================


class CoChemIntegrityError(Exception):
    """Raised when cryptographic verification or payload integrity check fails."""


class KraitchmanZPVEWarning(UserWarning):
    """Issued when ZPVE defect causes an imaginary substitution coordinate."""


class KraitchmanSingularityWarning(UserWarning):
    """Issued when near-symmetric top or denominator singularity occurs."""


# ============================================================================
# Canonical JSON & Cryptographic Helpers
# ============================================================================


def canonical_json_dumps(data: Any) -> str:
    """Serialize deterministic strict JSON (not an RFC 8785 implementation).

    Nonfinite numbers are rejected; unavailable values must use JSON null.
    """
    return json.dumps(
        data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def compute_file_sha256(
    file_path: str | Path, chunk_size: int = 8192
) -> tuple[str, int]:
    """Computes streaming SHA-256 digest and byte size of a file.

    :param file_path: Path to the target file.
    :param chunk_size: Chunk size in bytes for streaming read.
    :return: Tuple of (sha256_hex_string, size_in_bytes).
    """
    path = Path(file_path)
    hasher = hashlib.sha256()
    total_size = 0
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
            total_size += len(chunk)
    return hasher.hexdigest(), total_size


# ============================================================================
# Core Phase 9: Kraitchman Coordinate Engine
# ============================================================================


def calculate_kraitchman_coords(
    parent_moments: dict[str, float]
    | tuple[float, float, float]
    | list[float]
    | npt.NDArray[Any],
    substituted_moments: dict[str, float]
    | tuple[float, float, float]
    | list[float]
    | npt.NDArray[Any],
    parent_mass: float,
    delta_m: float,
    singularity_threshold: float = 1e-4,
) -> dict[str, Any]:
    """Derives substitution coordinates and Costain bounds using Kraitchman equations.

    :param parent_moments: Principal moments (Ia, Ib, Ic) of parent (u*A^2).
    :param substituted_moments: Moments (Ia', Ib', Ic') of isotopologue (u*A^2).
    :param parent_mass: Total molecular mass of parent molecule (u).
    :param delta_m: Mass difference of substituted atom m' - m (u).
    :param singularity_threshold: Reject inversion below this denominator threshold.
    :return: Dictionary containing coordinates, Costain errors, radicands, reduced mass.
    """

    def read_moments(values: Any, label: str) -> tuple[float, float, float]:
        if isinstance(values, dict):
            parsed = []
            for axis in ("a", "b", "c"):
                keys = (f"I{axis}", axis, f"I{axis.upper()}")
                present = [values[key] for key in keys if key in values]
                if not present or any(value is None for value in present):
                    raise ValueError(f"{label}: missing principal moment I{axis}")
                if len({float(value) for value in present}) != 1:
                    raise ValueError(f"{label}: conflicting aliases for I{axis}")
                parsed.append(float(present[0]))
        else:
            parsed = [float(value) for value in values]
        if len(parsed) != 3 or not all(math.isfinite(v) and v > 0 for v in parsed):
            raise ValueError(
                f"{label}: three finite positive principal moments required"
            )
        if parsed != sorted(parsed):
            raise ValueError(f"{label}: principal moments must satisfy Ia <= Ib <= Ic")
        return parsed[0], parsed[1], parsed[2]

    i_a, i_b, i_c = read_moments(parent_moments, "parent")
    i_ap, i_bp, i_cp = read_moments(substituted_moments, "substituted")
    if not all(math.isfinite(v) for v in (parent_mass, delta_m, singularity_threshold)):
        raise ValueError("Masses and singularity threshold must be finite")
    if parent_mass <= 0 or parent_mass + delta_m <= 0 or delta_m == 0:
        raise ValueError("Positive total masses and nonzero substitution mass required")
    if singularity_threshold <= 0:
        raise ValueError("Singularity threshold must be positive")

    # Principal moment differences
    d_ia = i_ap - i_a
    d_ib = i_bp - i_b
    d_ic = i_cp - i_c

    # Planar moment differences: Delta P_x = 1/2 (Delta I_y + Delta I_z - Delta I_x)
    d_pa = 0.5 * (d_ib + d_ic - d_ia)
    d_pb = 0.5 * (d_ic + d_ia - d_ib)
    d_pc = 0.5 * (d_ia + d_ib - d_ic)

    # Reduced mass for substitution mu = (M * delta_m) / (M + delta_m)
    mu = (parent_mass * delta_m) / (parent_mass + delta_m)

    # Near-degenerate principal moments do not support this asymmetric-top
    # inversion. Changing the denominator would create a different result.
    def _guard_denom(denom: float, label: str) -> float:
        if abs(denom) < singularity_threshold:
            raise ValueError(
                f"Kraitchman inversion unavailable: near-symmetric denominator "
                f"|{label}|={abs(denom):.6e} < {singularity_threshold}"
            )
        return denom

    d_ab = _guard_denom(i_a - i_b, "Ia - Ib")
    d_ac = _guard_denom(i_a - i_c, "Ia - Ic")
    d_bc = _guard_denom(i_b - i_c, "Ib - Ic")
    d_ba = _guard_denom(i_b - i_a, "Ib - Ia")
    d_ca = _guard_denom(i_c - i_a, "Ic - Ia")
    d_cb = _guard_denom(i_c - i_b, "Ic - Ib")

    # Kraitchman asymmetric top equations (Kraitchman 1953 Eq. 18)
    r_a = (d_pa / mu) * (1.0 + d_pb / d_ab) * (1.0 + d_pc / d_ac)
    r_b = (d_pb / mu) * (1.0 + d_pc / d_bc) * (1.0 + d_pa / d_ba)
    r_c = (d_pc / mu) * (1.0 + d_pa / d_ca) * (1.0 + d_pb / d_cb)

    radicands = {"a": float(r_a), "b": float(r_b), "c": float(r_c)}
    coords: dict[str, float | None] = {}
    costain_errors: dict[str, float | None] = {}
    quality_flags: dict[str, str] = {}

    for axis, r_val in radicands.items():
        if not math.isfinite(r_val):
            raise ValueError(f"Nonfinite Kraitchman radicand for axis {axis}")
        if r_val < 0.0:
            warnings.warn(
                f"Imaginary substitution coordinate for axis {axis} "
                f"(radicand={r_val:.6e}); coordinate unavailable. "
                "Vibrational effects, measurement error or input inconsistency "
                "require investigation.",
                KraitchmanZPVEWarning,
                stacklevel=2,
            )
            coords[axis] = None
            costain_errors[axis] = None
            quality_flags[axis] = "imaginary_coordinate"
            continue
        coord_val = math.sqrt(r_val)
        coords[axis] = coord_val
        quality_flags[axis] = "real_substitution_coordinate"
        # Historical empirical Costain estimate, not calibrated uncertainty.
        costain_errors[axis] = (
            0.0015 / coord_val if coord_val >= 0.15 else math.sqrt(r_val)
        )

    return {
        "coords": coords,
        "costain_errors": costain_errors,
        "costain_errors_kind": "legacy_empirical_estimate_not_calibrated_uncertainty",
        "quality_flags": quality_flags,
        "radicands": radicands,
        "delta_moments": {"a": d_ia, "b": d_ib, "c": d_ic},
        "planar_delta_moments": {"a": d_pa, "b": d_pb, "c": d_pc},
        "reduced_mass": mu,
        "parent_mass": parent_mass,
        "delta_m": delta_m,
    }


# ============================================================================
# Core Phase 9: OOM-Proof PGOPHER XML Skeleton Generator
# ============================================================================


def generate_pgopher_skeleton(
    parquet_path: str | Path,
    json_path: str | Path | None = None,
    output_path: str | Path | None = None,
    molecule_name: str = "Molecule",
    temperature_k: float = 298.15,
    rotational_constants: dict[str, float]
    | tuple[float, float, float]
    | list[float]
    | None = None,
    dipoles: dict[str, float] | tuple[float, float, float] | list[float] | None = None,
) -> str:
    """Inspects PyArrow metadata and generates PGOPHER .pgo XML skeleton.

    :param parquet_path: Path to the PyArrow Parquet catalog file.
    :param json_path: Optional path to internal JSON metadata/result file.
    :param output_path: Destination path for the .pgo file.
    :param molecule_name: Name of the molecule.
    :param temperature_k: Simulation temperature in Kelvin.
    :param rotational_constants: Optional dict/tuple of (A, B, C) in MHz.
    :param dipoles: Optional dict/tuple of (mu_a, mu_b, mu_c) in Debye.
    :return: String path to the generated .pgo file.
    """
    parquet_file = Path(parquet_path)
    if not parquet_file.exists():
        raise FileNotFoundError(f"Parquet catalog file not found: {parquet_path}")

    # OOM-Proof metadata inspection
    pq_metadata: Any = pq.read_metadata(str(parquet_file))  # type: ignore[no-untyped-call]
    num_rows = pq_metadata.num_rows
    num_columns = pq_metadata.num_columns
    column_names = pq_metadata.schema.names

    jdata: dict[str, Any] = {}
    if json_path is not None:
        # An explicitly requested but missing/invalid source is an error.
        with open(json_path, encoding="utf-8") as source:
            jdata = json.load(source)
        if not isinstance(jdata, dict):
            raise ValueError("PGOPHER metadata must be a JSON object")
        molecule_name = str(
            jdata.get("molecule_name", jdata.get("point_id", molecule_name))
        )
        temperature_k = float(
            jdata.get("temperature_k", jdata.get("temperature", temperature_k))
        )

    properties = jdata.get("properties", {})
    if rotational_constants is None:
        rotational_constants = jdata.get(
            "rotational_constants", properties.get("rotational_constants")
        )
    if dipoles is None:
        dipoles = jdata.get(
            "dipoles", jdata.get("dipole_moment", properties.get("dipole_moment"))
        )

    def read_triplet(
        values: Any, names: tuple[tuple[str, ...], ...], label: str
    ) -> tuple[float, float, float]:
        if isinstance(values, dict):
            parsed = []
            for aliases in names:
                present = [values[key] for key in aliases if key in values]
                if not present or any(value is None for value in present):
                    raise ValueError(f"Missing {label} component {aliases[0]}")
                if len({float(value) for value in present}) != 1:
                    raise ValueError(f"Conflicting {label} aliases {aliases}")
                parsed.append(float(present[0]))
        elif isinstance(values, (list, tuple)):
            parsed = [float(value) for value in values]
        else:
            raise ValueError(f"Explicit {label} required; no physical defaults exist")
        if len(parsed) != 3 or not all(math.isfinite(value) for value in parsed):
            raise ValueError(f"Three finite {label} components required")
        return parsed[0], parsed[1], parsed[2]

    a_val, b_val, c_val = read_triplet(
        rotational_constants,
        (("A", "a"), ("B", "b"), ("C", "c")),
        "rotational constants in MHz",
    )
    if not a_val >= b_val >= c_val > 0:
        raise ValueError(
            "Asymmetric-top rotational constants must satisfy A >= B >= C > 0"
        )
    mu_a, mu_b, mu_c = read_triplet(
        dipoles,
        (("mu_a", "a"), ("mu_b", "b"), ("mu_c", "c")),
        "principal-axis dipoles in Debye",
    )
    if not math.isfinite(temperature_k) or temperature_k <= 0:
        raise ValueError("Temperature must be finite and positive")
    # No vibrational state is inferred from a geometry or a set of constants.
    rotational_state = jdata.get("rotational_state", "unspecified")
    if not isinstance(rotational_state, str) or not rotational_state.strip():
        raise ValueError("rotational_state must be a nonempty string")

    # Build PGOPHER XML document
    root = ET.Element("Document", attrib={"Type": "PGopher", "Version": "10.1"})
    species = ET.SubElement(root, "Species", attrib={"Name": molecule_name})
    mol = ET.SubElement(species, "AsymmetricMolecule", attrib={"Name": molecule_name})
    manifold = ET.SubElement(
        mol, "AsymmetricManifold", attrib={"Name": rotational_state}
    )
    top = ET.SubElement(manifold, "AsymmetricTop", attrib={"Name": rotational_state})

    ET.SubElement(top, "Parameter", attrib={"Name": "A", "Value": f"{a_val:.6f}"})
    ET.SubElement(top, "Parameter", attrib={"Name": "B", "Value": f"{b_val:.6f}"})
    ET.SubElement(top, "Parameter", attrib={"Name": "C", "Value": f"{c_val:.6f}"})
    ET.SubElement(top, "Parameter", attrib={"Name": "mu_a", "Value": f"{mu_a:.6f}"})
    ET.SubElement(top, "Parameter", attrib={"Name": "mu_b", "Value": f"{mu_b:.6f}"})
    ET.SubElement(top, "Parameter", attrib={"Name": "mu_c", "Value": f"{mu_c:.6f}"})

    form = ET.SubElement(
        root,
        "Form",
        attrib={"Name": "Form", "Temperature": f"{temperature_k:.2f}", "Units": "MHz"},
    )
    ET.SubElement(
        form,
        "Metadata",
        attrib={
            "NumTransitions": str(num_rows),
            "NumColumns": str(num_columns),
            "Columns": ",".join(column_names),
            "ParquetSource": parquet_file.name,
        },
    )

    if output_path is None:
        target_out = parquet_file.parent / f"{molecule_name}.pgo"
    else:
        target_out = Path(output_path)

    target_out.parent.mkdir(parents=True, exist_ok=True)

    tree = ET.ElementTree(root)
    tree.write(str(target_out), encoding="utf-8", xml_declaration=True)
    logger.info(
        f"Generated PGOPHER skeleton at {target_out} "
        f"(Metadata: {num_rows} rows, {num_columns} cols)"
    )
    return str(target_out)


# ============================================================================
# Core Phase 9: Cryptographic Provenance Lock & Canonical Manifest
# ============================================================================


def lock_provenance_payload(
    target_directory: str | Path,
    output_manifest_path: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
    rotational_constants: dict[str, float]
    | tuple[float, float, float]
    | list[float]
    | None = None,
    dipoles: dict[str, float] | tuple[float, float, float] | list[float] | None = None,
    kraitchman_coords: dict[str, Any] | None = None,
    chunk_size: int = 8192,
) -> dict[str, Any]:
    """Computes streaming SHA-256 checksums and serializes spycfit_manifest.json.

    Bundles rotational constants, dipoles, Kraitchman substitution coordinates,
    and metadata alongside the deliverable manifest.

    :param target_directory: Path to directory containing deliverables.
    :param output_manifest_path: Destination path for the manifest JSON file.
    :param metadata: Additional metadata dictionary to embed.
    :param rotational_constants: Optional (A, B, C) rotational constants.
    :param dipoles: Optional (mu_a, mu_b, mu_c) dipole moments.
    :param kraitchman_coords: Optional Kraitchman coordinates dictionary.
    :param chunk_size: Binary chunk read size in bytes.
    :return: Canonical manifest dictionary.
    """
    target_dir = Path(target_directory)
    if not target_dir.exists():
        raise FileNotFoundError(f"Target directory not found: {target_directory}")

    if output_manifest_path is None:
        manifest_path = target_dir / "spycfit_manifest.json"
    else:
        manifest_path = Path(output_manifest_path)

    file_entries: list[dict[str, Any]] = []
    total_bytes = 0

    all_files = sorted(target_dir.rglob("*"))
    for p in all_files:
        if p.is_file():
            if (
                p.resolve() == manifest_path.resolve()
                or p.name == "spycfit_manifest.json"
            ):
                continue
            if (
                p.name.endswith(".tmp")
                or p.name.endswith(".tar.zst")
                or p.name.endswith(".zip")
            ):
                continue

            rel_path = p.relative_to(target_dir).as_posix()
            sha256_hash, file_size = compute_file_sha256(p, chunk_size=chunk_size)
            file_entries.append(
                {
                    "relative_path": rel_path,
                    "sha256": sha256_hash,
                    "size_bytes": file_size,
                }
            )
            total_bytes += file_size

    file_entries.sort(key=lambda x: str(x["relative_path"]))

    meta_payload = dict(metadata or {})
    if rotational_constants is not None:
        meta_payload["rotational_constants"] = rotational_constants
    if dipoles is not None:
        meta_payload["dipoles"] = dipoles
    if kraitchman_coords is not None:
        meta_payload["kraitchman_coords"] = kraitchman_coords

    manifest: dict[str, Any] = {
        "format": "CoChem-SpycFit-Manifest",
        "schema_version": "1.0.0",
        "generator": "CoChem-TORQ",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "file_count": len(file_entries),
        "total_bytes": total_bytes,
        "metadata": meta_payload,
        "files": file_entries,
    }

    # Deterministic strict JSON serialization
    canonical_json_str = canonical_json_dumps(manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(canonical_json_str, encoding="utf-8")

    logger.info(
        f"Locked provenance payload: {len(file_entries)} files "
        f"({total_bytes} bytes) -> {manifest_path}"
    )
    return manifest


# ============================================================================
# Core Phase 9: Deterministic .tar.zst Payload Bundler
# ============================================================================


def bundle_spycfit_payload(
    manifest_path_or_target_dir: str | Path,
    output_dir: str | Path | None = None,
    project_name: str = "Project",
    compression_level: int = 3,
) -> str:
    """Bundles deliverables into a deterministic .tar.zst archive with POSIX metadata.

    :param manifest_path_or_target_dir: Path to manifest or directory containing files.
    :param output_dir: Destination directory for the .tar.zst archive.
    :param project_name: Project name for archive filename.
    :param compression_level: Zstandard compression level (1-22).
    :return: String path to the created .tar.zst archive.
    """
    input_path = Path(manifest_path_or_target_dir)
    if input_path.is_file():
        target_dir = input_path.parent
    else:
        target_dir = input_path

    if not target_dir.exists():
        raise FileNotFoundError(f"Target directory not found: {target_dir}")

    manifest_file = target_dir / "spycfit_manifest.json"
    if not manifest_file.exists():
        lock_provenance_payload(target_dir, manifest_file)

    dest_dir = Path(output_dir) if output_dir is not None else target_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive_path = dest_dir / f"CoChem_{project_name}_SpycFit_Payload.tar.zst"

    tar_buffer = io.BytesIO()
    with tarfile.open(mode="w", fileobj=tar_buffer) as tar:
        for item in sorted(target_dir.rglob("*")):
            if item.resolve() == archive_path.resolve() or item.name.endswith(
                ".tar.zst"
            ):
                continue

            arcname = item.relative_to(target_dir).as_posix()
            tarinfo = tar.gettarinfo(str(item), arcname=arcname)

            tarinfo.mtime = 0
            tarinfo.uid = 0
            tarinfo.gid = 0
            tarinfo.uname = ""
            tarinfo.gname = ""
            tarinfo.mode = 0o755 if tarinfo.isdir() else 0o644

            if item.is_file():
                with open(item, "rb") as f:
                    tar.addfile(tarinfo, f)
            elif item.is_dir():
                tar.addfile(tarinfo)

    tar_bytes = tar_buffer.getvalue()

    compressor = zstd.ZstdCompressor(level=compression_level)
    compressed_bytes = compressor.compress(tar_bytes)

    archive_path.write_bytes(compressed_bytes)
    logger.info(
        f"Bundled deterministic SpycFit payload: {archive_path} "
        f"({len(compressed_bytes)} bytes)"
    )
    return str(archive_path)


# ============================================================================
# Core Phase 9: Cryptographic Payload Integrity Verification
# ============================================================================


def verify_payload_integrity(
    manifest_or_path: str | Path | dict[str, Any],
    base_dir: str | Path | None = None,
    chunk_size: int = 8192,
) -> bool:
    """Validates cryptographic integrity of a payload directory, manifest, or archive.

    :param manifest_or_path: Path to archive/manifest/directory or manifest dict.
    :param base_dir: Optional base directory if verifying a manifest dictionary.
    :param chunk_size: Chunk size in bytes for streaming SHA-256.
    :return: True if all cryptographic checksums and sizes match perfectly.
    :raises CoChemIntegrityError: If any integrity mismatch is detected.
    """
    p = Path(manifest_or_path) if isinstance(manifest_or_path, str | Path) else None
    if (
        p is not None
        and p.is_file()
        and (p.name.endswith(".tar.zst") or p.name.endswith(".zst"))
    ):
        compressed_bytes = p.read_bytes()
        dctx = zstd.ZstdDecompressor()
        try:
            decompressed_bytes = dctx.decompress(compressed_bytes)
        except Exception as e:
            raise CoChemIntegrityError(
                f"Failed to decompress Zstandard archive {p}: {e}"
            ) from e

        with tarfile.open(fileobj=io.BytesIO(decompressed_bytes), mode="r") as tar:
            try:
                manifest_member = tar.getmember("spycfit_manifest.json")
            except KeyError as err:
                raise CoChemIntegrityError(
                    "Manifest 'spycfit_manifest.json' not found in archive."
                ) from err

            manifest_f = tar.extractfile(manifest_member)
            if manifest_f is None:
                raise CoChemIntegrityError(
                    "Failed to extract 'spycfit_manifest.json' from archive."
                )

            manifest_data = json.loads(manifest_f.read().decode("utf-8"))
            files = manifest_data.get("files", [])

            for entry in files:
                rel_path = entry["relative_path"]
                try:
                    member = tar.getmember(rel_path)
                except KeyError as err:
                    raise CoChemIntegrityError(
                        f"Missing file in archive: {rel_path}"
                    ) from err

                member_f = tar.extractfile(member)
                if member_f is None:
                    raise CoChemIntegrityError(
                        f"Failed to read file in archive: {rel_path}"
                    )

                hasher = hashlib.sha256()
                actual_size = 0
                while chunk := member_f.read(chunk_size):
                    hasher.update(chunk)
                    actual_size += len(chunk)

                expected_size = entry.get("size_bytes")
                expected_sha = entry.get("sha256")

                if actual_size != expected_size:
                    raise CoChemIntegrityError(
                        f"File size mismatch for {rel_path}: "
                        f"expected {expected_size} bytes, got {actual_size} bytes."
                    )
                computed_sha = hasher.hexdigest()
                if computed_sha != expected_sha:
                    raise CoChemIntegrityError(
                        f"SHA-256 hash mismatch for {rel_path}: "
                        f"expected {expected_sha}, got {computed_sha}."
                    )

        logger.info(f"Archive integrity verification passed: {p}")
        return True

    # Case 2: Manifest dictionary or directory or manifest file path
    if isinstance(manifest_or_path, dict):
        manifest_data = manifest_or_path
        target_base = Path(base_dir) if base_dir is not None else Path(".")
    else:
        if p is None:
            raise ValueError("Invalid manifest_or_path argument.")
        if p.is_dir():
            manifest_file = p / "spycfit_manifest.json"
            target_base = p
        else:
            manifest_file = p
            target_base = Path(base_dir) if base_dir is not None else p.parent

        if not manifest_file.exists():
            raise CoChemIntegrityError(f"Manifest file not found: {manifest_file}")

        manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))

    files = manifest_data.get("files", [])
    for entry in files:
        rel_path = entry["relative_path"]
        target_file = target_base / rel_path

        if not target_file.exists():
            raise CoChemIntegrityError(
                f"Missing file in payload: {rel_path} (expected at {target_file})"
            )

        expected_size = entry.get("size_bytes")
        expected_sha = entry.get("sha256")
        actual_size = target_file.stat().st_size

        if actual_size != expected_size:
            raise CoChemIntegrityError(
                f"File size mismatch for {rel_path}: "
                f"expected {expected_size} bytes, got {actual_size} bytes."
            )

        computed_sha, _ = compute_file_sha256(target_file, chunk_size=chunk_size)
        if computed_sha != expected_sha:
            raise CoChemIntegrityError(
                f"SHA-256 hash mismatch for {rel_path}: "
                f"expected {expected_sha}, got {computed_sha}."
            )

    logger.info(f"Payload integrity verification passed: {len(files)} files verified.")
    return True


# ============================================================================
# Retained: TorqExporter, PESStore & export_qcschema
# ============================================================================


class TorqExporter:
    """Manages Zstandard tensor exports and metadata tracking."""

    def __init__(
        self, export_dir: str = "torq_exports", zstd_compression_level: int = 3
    ) -> None:
        self.export_dir = Path(export_dir)
        self.zstd_compression_level = zstd_compression_level
        self.export_dir.mkdir(parents=True, exist_ok=True)

    def _generate_metadata(
        self,
        point_id: str,
        tensor_data: dict[str, Any],
        lam_trigger_required: bool | None = None,
        symmetry_group: str | None = None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "point_id": point_id,
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "data_hash": hashlib.sha256(
                canonical_json_dumps(tensor_data).encode()
            ).hexdigest(),
            "compression_method": "Zstandard",
            "compression_level": self.zstd_compression_level,
        }
        if lam_trigger_required is not None:
            metadata["LAM_TRIGGER_REQUIRED"] = lam_trigger_required
        if symmetry_group is not None:
            metadata["symmetry_group"] = symmetry_group
        return metadata

    def export_tensor_to_zstd(
        self, h5_file_path: str, output_file: str | None = None
    ) -> str:
        """Exports an HDF5 tensor to a Zstandard-compressed file."""
        try:
            with h5py.File(h5_file_path, "r") as f:
                tensor_data: dict[str, Any] = {}

                def read_group(name: str, obj: Any) -> None:
                    if isinstance(obj, h5py.Group):
                        tensor_data[name] = {}
                        for key, value in obj.items():
                            if isinstance(value, h5py.Dataset):
                                val = value[()]
                                if hasattr(val, "tolist"):
                                    val = val.tolist()
                                tensor_data[name][key] = val
                            else:
                                tensor_data[name][key] = str(value.attrs)
                    elif isinstance(obj, h5py.Dataset):
                        val = obj[()]
                        if hasattr(val, "tolist"):
                            val = val.tolist()
                        tensor_data[name] = val

                f.visititems(read_group)
        except Exception as e:
            logger.error(f"Error reading HDF5 file {h5_file_path}: {e}")
            raise

        point_id = Path(h5_file_path).stem.replace("cochem_", "").replace(".h5", "")
        metadata = self._generate_metadata(point_id, tensor_data)

        export_data = {
            "tensor_data": tensor_data,
            "metadata": metadata,
        }

        json_data = json.dumps(export_data, indent=2, allow_nan=False)
        if output_file is None:
            output_file = f"{Path(h5_file_path).stem}.zst"

        compressed_file = self.export_dir / output_file
        try:
            with open(compressed_file, "wb") as f:
                compressor = zstd.ZstdCompressor(level=self.zstd_compression_level)
                compressed_data = compressor.compress(json_data.encode("utf-8"))
                f.write(compressed_data)

            logger.info(
                f"Exported tensor to Zstandard-compressed file: {compressed_file}"
            )
            return str(compressed_file)
        except Exception as e:
            logger.error(f"Error compressing data to Zstandard: {e}")
            raise

    def export_tensor_to_zstd_with_sinc_dvr(
        self, h5_file_path: str, output_file: str | None = None
    ) -> str:
        """Exports an HDF5 tensor with Sinc-DVR data to a Zstandard-compressed file."""
        try:
            with h5py.File(h5_file_path, "r") as f:
                tensor_data: dict[str, Any] = {}

                def read_group(name: str, obj: Any) -> None:
                    if isinstance(obj, h5py.Group):
                        tensor_data[name] = {}
                        for key, value in obj.items():
                            if isinstance(value, h5py.Dataset):
                                val = value[()]
                                if hasattr(val, "tolist"):
                                    val = val.tolist()
                                tensor_data[name][key] = val
                            else:
                                tensor_data[name][key] = str(value.attrs)
                    elif isinstance(obj, h5py.Dataset):
                        val = obj[()]
                        if hasattr(val, "tolist"):
                            val = val.tolist()
                        tensor_data[name] = val

                f.visititems(read_group)
        except Exception as e:
            logger.error(f"Error reading HDF5 file {h5_file_path}: {e}")
            raise

        point_id = Path(h5_file_path).stem.replace("cochem_", "").replace(".h5", "")
        metadata = self._generate_metadata(point_id, tensor_data)

        export_data = {
            "tensor_data": tensor_data,
            "metadata": metadata,
        }

        json_data = json.dumps(export_data, indent=2, allow_nan=False)
        if output_file is None:
            output_file = f"{Path(h5_file_path).stem}_dvr.zst"

        compressed_file = self.export_dir / output_file
        try:
            with open(compressed_file, "wb") as f:
                compressor = zstd.ZstdCompressor(level=self.zstd_compression_level)
                compressed_data = compressor.compress(json_data.encode("utf-8"))
                f.write(compressed_data)

            logger.info(
                f"Exported Sinc-DVR tensor to Zstandard-compressed file: "
                f"{compressed_file}"
            )
            return str(compressed_file)
        except Exception as e:
            logger.error(f"Error compressing data to Zstandard: {e}")
            raise

    def batch_export_to_zstd(
        self, h5_files: list[str], output_dir: str | None = None
    ) -> list[str]:
        """Exports multiple HDF5 tensor files to Zstandard-compressed files."""
        if output_dir:
            self.export_dir = Path(output_dir)
            self.export_dir.mkdir(parents=True, exist_ok=True)

        exported_files: list[str] = []
        for h5_file in h5_files:
            try:
                exported_file = self.export_tensor_to_zstd(h5_file)
                exported_files.append(exported_file)
            except Exception as e:
                logger.error(f"Error exporting {h5_file}: {e}")
                continue

        return exported_files

    def verify_export(
        self, compressed_file_path: str
    ) -> tuple[bool, dict[str, Any] | None]:
        """Verifies the integrity of a compressed export file."""
        try:
            with open(compressed_file_path, "rb") as f:
                decompressor = zstd.ZstdDecompressor()
                decompressed_data = decompressor.decompress(f.read())

            export_data = json.loads(decompressed_data.decode("utf-8"))
            metadata = export_data["metadata"]
            actual_hash = hashlib.sha256(
                canonical_json_dumps(export_data["tensor_data"]).encode("utf-8")
            ).hexdigest()
            if actual_hash != metadata.get("data_hash"):
                raise CoChemIntegrityError(
                    "Tensor payload hash does not match metadata"
                )
            logger.info(f"Verification successful for {compressed_file_path}")
            return True, metadata
        except Exception as e:
            logger.error(f"Verification failed for {compressed_file_path}: {e}")
            return False, None

    def export_to_scribe_daemon(
        self,
        compressed_file_path: str,
        host: str = "127.0.0.1",
        port: int = 5555,
        timeout_ms: int = 2000,
    ) -> bool:
        """Exports compressed tensor to CoChem-SCRIBE daemon via ZeroMQ IPC."""
        logger.info(
            f"Connecting to CoChem-SCRIBE daemon at {host}:{port} "
            f"for {compressed_file_path}"
        )
        file_path = Path(compressed_file_path)
        if not file_path.exists():
            logger.error(f"Compressed export file not found: {compressed_file_path}")
            return False

        try:
            payload = file_path.read_bytes()
            sha256_hash = hashlib.sha256(payload).hexdigest()
            meta_data = {
                "file_name": file_path.name,
                "file_size": len(payload),
                "sha256": sha256_hash,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "ready",
            }

            try:
                import zmq

                ctx: Any = zmq.Context.instance()
                socket = ctx.socket(zmq.REQ)
                socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
                socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
                socket.setsockopt(zmq.LINGER, 0)
                socket.connect(f"tcp://{host}:{port}")

                socket.send_json(meta_data)
                reply = socket.recv_json()
                logger.info(f"CoChem-SCRIBE daemon response: {reply}")
                socket.close()
                return True
            except ImportError as err:
                logger.error("pyzmq module missing; cannot export.")
                raise err
        except Exception as e:
            logger.error(f"Failed to export to CoChem-SCRIBE: {e}")
            raise e


class PESStore:
    """Appends coordinate geometry and energy to a chunked HDF5 database."""

    def __init__(self, h5_filepath: str) -> None:
        self.h5_filepath = h5_filepath

    def append_data(self, step: int, coordinates: list[float], energy: float) -> None:
        """Appends coordinates and energy without scaleoffset per Section 6.4.3."""
        with h5py.File(self.h5_filepath, "a") as f:
            if "coordinates" not in f:
                f.create_dataset(
                    "coordinates",
                    data=[coordinates],
                    maxshape=(None, len(coordinates)),
                    chunks=True,
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                    scaleoffset=None,
                )
            else:
                f["coordinates"].resize(
                    (f["coordinates"].shape[0] + 1, f["coordinates"].shape[1])
                )
                f["coordinates"][-1] = coordinates

            if "energies" not in f:
                f.create_dataset(
                    "energies",
                    data=[energy],
                    maxshape=(None,),
                    chunks=True,
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                    scaleoffset=None,
                )
            else:
                f["energies"].resize((f["energies"].shape[0] + 1,))
                f["energies"][-1] = energy


def export_qcschema(result_dict: dict[str, Any], output_filename: str) -> str:
    """Validate and serialize a complete QCSchema v1 AtomicResult.

    QCSchema geometry and energy use bohr and hartree. The caller must supply
    actual engine provenance and explicit charge/spin; this function does not
    infer a method, convert ambiguous units, or create absent results. Composite
    spectroscopy results belong in :func:`export_torq_result_bundle` instead.
    """
    required = {
        "schema_name",
        "schema_version",
        "molecule",
        "driver",
        "model",
        "properties",
        "return_result",
        "success",
        "provenance",
    }
    missing = required.difference(result_dict)
    if missing:
        raise ValueError(f"Complete AtomicResult required; missing: {sorted(missing)}")
    if (
        result_dict["schema_name"] != "qcschema_output"
        or result_dict["schema_version"] != 1
    ):
        raise ValueError("Only QCSchema AtomicResult schema_version=1 is supported")
    if result_dict["success"] is not True:
        raise ValueError(
            "A failed calculation cannot be exported as a successful AtomicResult"
        )
    for key in ("symbols", "geometry", "molecular_charge", "molecular_multiplicity"):
        if key not in result_dict["molecule"] or result_dict["molecule"][key] is None:
            raise ValueError(f"Explicit molecule.{key} required")
    if not result_dict["molecule"]["symbols"]:
        raise ValueError("AtomicResult requires a nonempty molecule")
    for key in ("creator", "version", "routine"):
        if not result_dict["provenance"].get(key):
            raise ValueError(f"Actual engine provenance.{key} required")
    if not result_dict["model"].get("method"):
        raise ValueError("Actual model.method required")
    # Ensure strict JSON before schema validation can normalize array values.
    canonical_json_dumps(result_dict)
    if result_dict["driver"] == "energy":
        energy = result_dict["properties"].get("return_energy")
        if energy is None:
            raise ValueError("Energy driver requires properties.return_energy")
        if isinstance(energy, bool) or energy != result_dict["return_result"]:
            raise ValueError("Energy return_result must match properties.return_energy")
    try:
        from qcelemental.models import AtomicResult
    except ImportError as exc:
        raise ImportError("QCSchema export requires the qcelemental package") from exc
    validated = AtomicResult(**result_dict)
    # Serialize only explicitly present fields: reference-model defaults may
    # describe molecule metadata, but must not invent optional observables.
    payload = json.loads(validated.json(exclude_unset=True))
    encoded = json.dumps(payload, indent=2, allow_nan=False)
    # Validation completes before touching an existing output file.
    Path(output_filename).write_text(encoded, encoding="utf-8")
    return output_filename


def export_torq_result_bundle(result_dict: dict[str, Any], output_filename: str) -> str:
    """Write a namespaced TORQ bundle, preserving absent values and raw provenance.

    This format is a transparent container, not an AtomicResult or a claim of
    scientific validation. Its digest verifies the supplied payload bytes only.
    """
    payload_json = canonical_json_dumps(result_dict)
    bundle = {
        "schema_name": "cochem_torq_result_bundle",
        "schema_version": 1,
        "payload_sha256": hashlib.sha256(payload_json.encode("utf-8")).hexdigest(),
        "result": result_dict,
    }
    encoded = json.dumps(bundle, indent=2, allow_nan=False)
    Path(output_filename).write_text(encoded, encoding="utf-8")
    return output_filename
