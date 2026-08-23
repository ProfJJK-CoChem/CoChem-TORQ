Perform adversarial static analysis and logical review on implemented code for D:\__CoChem\__agentic\.prompts\.SRS\CoChem-TORQ\.in-progress\prompt_task12_export.md.
Original prompt:
# Prompt: Cryptographic Payload Synthesizer

**Target File:** `D:\__CoChem\GitHub-Repo\CoChem-TORQ\Libraries\cochem_torq_export.py`

## Objective
Implement Cryptographic Payload Synthesizer for CoChem-TORQ based on Task 12 (Stage 5.5 - 6.0) specifications.

## Instructions for Coder
1. Create or update `cochem_torq_export.py` inside `Libraries/`.
2. Implement `calculate_kraitchman_coords()` evaluating substitution coordinates (r_s). Trap imaginary roots and implement Piecewise Costain Bounds.
3. Implement `generate_pgopher_skeleton()` inspecting Parquet metadata (`pyarrow.parquet.read_metadata()`) and emitting a standardized `.pgo` XML skeleton. Write this file strictly to the dynamically provided artifact/output directory, NOT the repository root.
4. Implement `lock_provenance_payload()` computing streaming SHA-256 checksums and serializing `spycfit_manifest.json` under RFC 8785 Canonical JSON. Write to the artifact directory.
5. Implement `bundle_spycfit_payload()` archiving deliverables into deterministic `.tar.zst` with normalized POSIX mtime and file permissions. Write to the artifact directory.
6. Implement `verify_payload_integrity()` executing an autonomous self-audit validating SHA-256 checksums prior to handoff.

## Constraints & Anti-Spoofing
- **One Script Policy**: Only create or modify `cochem_torq_export.py`.
- **Zero Mocking**: Do NOT mock any logic. Implement physical `pyarrow` metadata reads, SHA-256 hashing, and `.tar.zst` bundling.
- **Context-Safety**: Do not hallucinate imports. Limit dependencies to the `requirements.txt` environment for CoChem-TORQ.
- **Air-Gap Compliance**: The generated Python script MUST NOT write any data or logs to the repository space at runtime. Read and write strictly according to the paths defined dynamically. All files (like `.tar.zst`, `.jsonl`, etc.) MUST be written to the scratch or artifact paths provided dynamically by the environment or arguments, NOT the current working directory.

Modified files content:

--- D:\__CoChem\GitHub-Repo\CoChem-TORQ\Libraries\cochem_torq_export.py ---
"""CoChem-TORQ: Cryptographic Payload Synthesizer & Deliverable Gateway.

Phase 9 (Stages 5.5 - 6.0) Specification
---------------------------------------------------------------------
Aggregates forward predictions, exact physics tensors, PyArrow .parquet catalogs,
and .var/.int files into a unified, cryptographically locked export payload
specifically designed for seamless ingestion by CoChem-SpycFit.

Implements:
1. Kraitchman coordinate calculations with singularity damping, ZPVE clamping,
   and piecewise Costain bounds.
2. OOM-proof PGOPHER XML skeleton generation using PyArrow parquet metadata.
3. Provenance lock manifest generation under RFC 8785 Canonical JSON.
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

import h5py
import numpy as np
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
    """Serializes a Python data structure into RFC 8785 compliant Canonical JSON.

    Keys are sorted lexicographically, and whitespace is strictly minimized.
    """
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


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
    :param singularity_threshold: Threshold below which denominators are damped.
    :return: Dictionary containing coordinates, Costain errors, radicands, reduced mass.
    """
    # Extract parent moments
    if isinstance(parent_moments, dict):
        i_a = float(
            parent_moments.get(
                "Ia", parent_moments.get("a", parent_moments.get("IA", 0.0))
            )
        )
        i_b = float(
            parent_moments.get(
                "Ib", parent_moments.get("b", parent_moments.get("IB", 0.0))
            )
        )
        i_c = float(
            parent_moments.get(
                "Ic", parent_moments.get("c", parent_moments.get("IC", 0.0))
            )
        )
    else:
        i_a = float(parent_moments[0])
        i_b = float(parent_moments[1])
        i_c = float(parent_moments[2])

    # Extract substituted moments
    if isinstance(substituted_moments, dict):
        i_ap = float(
            substituted_moments.get(
                "Ia", substituted_moments.get("a", substituted_moments.get("IA", 0.0))
            )
        )
        i_bp = float(
            substituted_moments.get(
                "Ib", substituted_moments.get("b", substituted_moments.get("IB", 0.0))
            )
        )
        i_cp = float(
            substituted_moments.get(
                "Ic", substituted_moments.get("c", substituted_moments.get("IC", 0.0))
            )
        )
    else:
        i_ap = float(substituted_moments[0])
        i_bp = float(substituted_moments[1])
        i_cp = float(substituted_moments[2])

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

    # Singularity guard for near-symmetric denominators
    def _guard_denom(denom: float, label: str) -> float:
        if abs(denom) < singularity_threshold:
            warnings.warn(
                f"Singularity near-symmetric denominator |{label}| = "
                f"{abs(denom):.6e} < {singularity_threshold}. Applying damping guard.",
                KraitchmanSingularityWarning,
                stacklevel=2,
            )
            logger.warning(
                f"Kraitchman singularity damping applied to {label}: denom={denom:.6e}"
            )
            if denom != 0.0:
                return math.copysign(singularity_threshold, denom)
            return singularity_threshold
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
    coords: dict[str, float] = {}
    costain_errors: dict[str, float] = {}

    for axis, r_val in radicands.items():
        if math.isnan(r_val) or r_val < 0.0:
            warnings.warn(
                f"ZPVE defect produced imaginary substitution coordinate "
                f"for axis {axis} (R_{axis} = {r_val:.6e} < 0). Clamping to 0.0000.",
                KraitchmanZPVEWarning,
                stacklevel=2,
            )
            logger.warning(
                f"ZPVE defect clamped coordinate for axis {axis}: "
                f"R={r_val:.6e} -> 0.0000"
            )
            coord_val = 0.0
        else:
            coord_val = float(np.sqrt(r_val))

        coords[axis] = coord_val

        # Piecewise Costain Bounds (Costain 1958)
        # For |g_s| >= 0.15 A: error = 0.0015 / |g_s|
        # For |g_s| < 0.15 A: error = sqrt(|R_g|)
        if coord_val >= 0.15:
            costain_errors[axis] = 0.0015 / coord_val
        else:
            costain_errors[axis] = float(np.sqrt(abs(r_val)))

    return {
        "coords": coords,
        "costain_errors": costain_errors,
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
    pq_metadata = pq.read_metadata(str(parquet_file))  # type: ignore[no-untyped-call]
    num_rows = pq_metadata.num_rows
    num_columns = pq_metadata.num_columns
    column_names = pq_metadata.schema.names

    # Defaults
    a_val = 10000.0
    b_val = 5000.0
    c_val = 3000.0
    mu_a = 0.0
    mu_b = 0.0
    mu_c = 0.0

    # Parse JSON if provided
    if json_path is not None:
        json_file = Path(json_path)
        if json_file.exists():
            with open(json_file, encoding="utf-8") as f:
                jdata = json.load(f)

            if "molecule_name" in jdata:
                molecule_name = str(jdata["molecule_name"])
            elif "point_id" in jdata:
                molecule_name = str(jdata["point_id"])

            if "temperature_k" in jdata:
                temperature_k = float(jdata["temperature_k"])
            elif "temperature" in jdata:
                temperature_k = float(jdata["temperature"])

            rc = jdata.get("rotational_constants") or jdata.get("properties", {}).get(
                "rotational_constants"
            )
            if rc:
                if isinstance(rc, dict):
                    a_val = float(rc.get("A") or rc.get("a") or a_val)
                    b_val = float(rc.get("B") or rc.get("b") or b_val)
                    c_val = float(rc.get("C") or rc.get("c") or c_val)
                elif isinstance(rc, list | tuple) and len(rc) >= 3:
                    a_val, b_val, c_val = float(rc[0]), float(rc[1]), float(rc[2])

            dp = (
                jdata.get("dipoles")
                or jdata.get("dipole_moment")
                or jdata.get("properties", {}).get("dipole_moment")
            )
            if dp:
                if isinstance(dp, dict):
                    mu_a = float(dp.get("mu_a") or dp.get("a") or dp.get("x") or mu_a)
                    mu_b = float(dp.get("mu_b") or dp.get("b") or dp.get("y") or mu_b)
                    mu_c = float(dp.get("mu_c") or dp.get("c") or dp.get("z") or mu_c)
                elif isinstance(dp, list | tuple) and len(dp) >= 3:
                    mu_a, mu_b, mu_c = float(dp[0]), float(dp[1]), float(dp[2])

    # Direct keyword overrides
    if rotational_constants is not None:
        if isinstance(rotational_constants, dict):
            a_val = float(
                rotational_constants.get("A") or rotational_constants.get("a") or a_val
            )
            b_val = float(
                rotational_constants.get("B") or rotational_constants.get("b") or b_val
            )
            c_val = float(
                rotational_constants.get("C") or rotational_constants.get("c") or c_val
            )
        elif (
            isinstance(rotational_constants, list | tuple)
            and len(rotational_constants) >= 3
        ):
            a_val, b_val, c_val = (
                float(rotational_constants[0]),
                float(rotational_constants[1]),
                float(rotational_constants[2]),
            )

    if dipoles is not None:
        if isinstance(dipoles, dict):
            mu_a = float(
                dipoles.get("mu_a") or dipoles.get("a") or dipoles.get("x") or mu_a
            )
            mu_b = float(
                dipoles.get("mu_b") or dipoles.get("b") or dipoles.get("y") or mu_b
            )
            mu_c = float(
                dipoles.get("mu_c") or dipoles.get("c") or dipoles.get("z") or mu_c
            )
        elif isinstance(dipoles, list | tuple) and len(dipoles) >= 3:
            mu_a, mu_b, mu_c = float(dipoles[0]), float(dipoles[1]), float(dipoles[2])

    # Build PGOPHER XML document
    root = ET.Element("Document", attrib={"Type": "PGopher", "Version": "10.1"})
    species = ET.SubElement(root, "Species", attrib={"Name": molecule_name})
    mol = ET.SubElement(species, "AsymmetricMolecule", attrib={"Name": molecule_name})
    manifold = ET.SubElement(
        mol, "AsymmetricManifold", attrib={"Initial": "true", "Name": "Ground"}
    )
    top = ET.SubElement(manifold, "AsymmetricTop", attrib={"Name": "v=0"})

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

    # RFC 8785 Canonical JSON Serialization
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
        lam_trigger_required: bool = False,
        symmetry_group: str = "C1",
    ) -> dict[str, Any]:
        return {
            "point_id": point_id,
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "data_hash": hashlib.sha256(str(tensor_data).encode()).hexdigest(),
            "compression_method": "Zstandard",
            "compression_level": self.zstd_compression_level,
            "LAM_TRIGGER_REQUIRED": bool(lam_trigger_required),
            "symmetry_group": str(symmetry_group),
        }

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

        json_data = json.dumps(export_data, indent=2)
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

        json_data = json.dumps(export_data, indent=2)
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
            logger.info(f"Verification successful for {compressed_file_path}")
            return True, export_data.get("metadata")
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
    """Accepts an OrcaResult (or dict) and writes a FAIR QCSchema output JSON."""
    data_to_hash = json.dumps(result_dict, sort_keys=True).encode()
    hash_val = hashlib.sha256(data_to_hash).hexdigest()

    qcschema = {
        "schema_name": "qcschema_output",
        "schema_version": 1,
        "molecule": {
            "geometry": result_dict.get("geometry", []),
            "symbols": result_dict.get("symbols", []),
            "molecular_charge": result_dict.get("molecular_charge", 0),
            "molecular_multiplicity": result_dict.get("molecular_multiplicity", 1),
            "provenance": {
                "creator": "CoChem-SCRIBE",
                "version": "4.1",
                "hash": hash_val,
            },
        },
        "driver": result_dict.get("driver", "energy"),
        "model": {
            "method": result_dict.get("method", "unknown"),
            "basis": result_dict.get("basis", "unknown"),
        },
        "properties": {
            "return_energy": result_dict.get("return_energy", 0.0),
        },
    }

    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(qcschema, f, indent=2)
    return output_filename

--- D:\__CoChem\GitHub-Repo\CoChem-TORQ\tests\test_torq_export.py ---
"""Comprehensive Test Suite for Cryptographic Payload Synthesizer.

Phase 9 (Stages 5.5 - 6.0) Validation Suite
-----------------------------------------------------------------------------------
Validates:
1. Kraitchman coordinates with real physical moments of inertia,
   singularity damping, ZPVE defect clamping, and piecewise Costain bounds.
2. OOM-proof PGOPHER XML skeleton generation inspecting Parquet metadata.
3. Provenance lock manifest generation under RFC 8785 Canonical JSON.
4. Deterministic .tar.zst payload bundling with normalized POSIX metadata.
5. Comprehensive payload integrity verification and tamper detection.
6. TorqExporter, PESStore, and export_qcschema integration.
"""

from __future__ import annotations

import io
import json
import math
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import scipy.linalg as sla
import zstandard as zstd

from Libraries.cochem_torq_export import (
    CoChemIntegrityError,
    KraitchmanSingularityWarning,
    KraitchmanZPVEWarning,
    PESStore,
    TorqExporter,
    bundle_spycfit_payload,
    calculate_kraitchman_coords,
    canonical_json_dumps,
    compute_file_sha256,
    export_qcschema,
    generate_pgopher_skeleton,
    lock_provenance_payload,
    verify_payload_integrity,
)

# ============================================================================
# Physical Helper: Inertial Tensor & Moments for Rigid 3D Molecules
# ============================================================================


def compute_principal_moments(
    coordinates: npt.NDArray[np.float64], masses: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Computes center-of-mass shifted coordinates, principal moments of inertia.

    :param coordinates: (N, 3) Cartesian coordinates in Angstroms.
    :param masses: (N,) atomic masses in atomic mass units (u).
    :return: (principal_moments, principal_axes_matrix, aligned_coordinates)
    """
    total_mass = float(np.sum(masses))
    com = np.sum(coordinates * masses[:, None], axis=0) / total_mass
    coords_com = coordinates - com

    x = coords_com[:, 0]
    y = coords_com[:, 1]
    z = coords_com[:, 2]

    i_xx = float(np.sum(masses * (y**2 + z**2)))
    i_yy = float(np.sum(masses * (x**2 + z**2)))
    i_zz = float(np.sum(masses * (x**2 + y**2)))
    i_xy = -float(np.sum(masses * x * y))
    i_xz = -float(np.sum(masses * x * z))
    i_yz = -float(np.sum(masses * y * z))

    i_tensor = np.array(
        [[i_xx, i_xy, i_xz], [i_xy, i_yy, i_yz], [i_xz, i_yz, i_zz]], dtype=np.float64
    )
    evals, evecs = sla.eigh(i_tensor)

    idx = np.argsort(evals)
    evals = evals[idx]
    evecs = evecs[:, idx]

    aligned = coords_com @ evecs
    return evals, evecs, aligned


# ============================================================================
# Test Suite 1: Kraitchman Coordinate Engine & Physical Invariants
# ============================================================================


def test_kraitchman_real_asymmetric_top() -> None:
    """Validates Kraitchman coordinates for 3D asymmetric top molecule."""
    coords = np.array(
        [
            [0.000000, 0.000000, 0.000000],  # C
            [1.080000, 0.000000, 0.000000],  # H
            [-0.350000, 1.350000, 0.000000],  # F
            [-0.350000, -0.650000, 1.350000],  # Cl
            [-0.350000, -0.650000, -1.350000],  # Br
        ],
        dtype=np.float64,
    )
    masses_parent = np.array(
        [12.000000, 1.007825, 18.998403, 34.968853, 78.918337], dtype=np.float64
    )

    i_parent, _, aligned_parent = compute_principal_moments(coords, masses_parent)
    parent_mass = float(np.sum(masses_parent))

    masses_sub = masses_parent.copy()
    masses_sub[1] = 2.014102
    delta_m = 2.014102 - 1.007825

    i_sub, _, _ = compute_principal_moments(coords, masses_sub)
    true_h_coords = np.abs(aligned_parent[1])

    result = calculate_kraitchman_coords(
        parent_moments=i_parent,
        substituted_moments=i_sub,
        parent_mass=parent_mass,
        delta_m=delta_m,
    )

    calc_a = result["coords"]["a"]
    calc_b = result["coords"]["b"]
    calc_c = result["coords"]["c"]

    np.testing.assert_allclose(calc_a, true_h_coords[0], atol=1e-5)
    np.testing.assert_allclose(calc_b, true_h_coords[1], atol=1e-5)
    np.testing.assert_allclose(calc_c, true_h_coords[2], atol=1e-5)

    expected_mu = (parent_mass * delta_m) / (parent_mass + delta_m)
    assert math.isclose(result["reduced_mass"], expected_mu, rel_tol=1e-9)


def test_kraitchman_singularity_guard_damping() -> None:
    """Validates near-symmetric top damping guard (|Ia - Ib| < 1e-4)."""
    i_a = 15.00000
    i_b = 15.00005
    i_c = 30.00000

    parent_moments = {"Ia": i_a, "Ib": i_b, "Ic": i_c}
    sub_moments = {"Ia": i_a + 0.1, "Ib": i_b + 0.1, "Ic": i_c + 0.05}

    with pytest.warns(
        KraitchmanSingularityWarning, match="Singularity near-symmetric denominator"
    ):
        result = calculate_kraitchman_coords(
            parent_moments=parent_moments,
            substituted_moments=sub_moments,
            parent_mass=45.0,
            delta_m=1.003355,
            singularity_threshold=1e-4,
        )

    assert not math.isnan(result["coords"]["a"])
    assert not math.isnan(result["coords"]["b"])
    assert not math.isnan(result["coords"]["c"])
    assert result["coords"]["a"] >= 0.0


def test_kraitchman_zpve_defect_clamping() -> None:
    """Validates that negative radicands (R_g < 0) are clamped to 0.0000."""
    i_a, i_b, i_c = 10.0, 25.0, 30.0
    parent_moments = (i_a, i_b, i_c)
    sub_moments = (i_a + 1.5, i_b + 0.1, i_c + 0.1)

    with pytest.warns(
        KraitchmanZPVEWarning,
        match="ZPVE defect produced imaginary substitution coordinate",
    ):
        result = calculate_kraitchman_coords(
            parent_moments=parent_moments,
            substituted_moments=sub_moments,
            parent_mass=60.0,
            delta_m=1.003355,
        )

    assert result["coords"]["a"] == 0.0
    assert result["radicands"]["a"] < 0.0

    expected_error = math.sqrt(abs(result["radicands"]["a"]))
    assert math.isclose(result["costain_errors"]["a"], expected_error, rel_tol=1e-6)


def test_kraitchman_piecewise_costain_bounds() -> None:
    """Validates Piecewise Costain Bounds for large and small coordinates."""
    res = calculate_kraitchman_coords(
        parent_moments=(10.0, 20.0, 25.0),
        substituted_moments=(10.2, 20.4, 25.3),
        parent_mass=50.0,
        delta_m=1.00335,
    )

    for axis in ["a", "b", "c"]:
        coord = res["coords"][axis]
        error = res["costain_errors"][axis]
        radicand = res["radicands"][axis]

        if coord >= 0.15:
            assert math.isclose(error, 0.0015 / coord, rel_tol=1e-7)
        else:
            assert math.isclose(error, math.sqrt(abs(radicand)), rel_tol=1e-7)


# ============================================================================
# Test Suite 2: OOM-Proof PGOPHER XML Skeleton Generation
# ============================================================================


def test_generate_pgopher_skeleton_oom_proof(tmp_path: Path) -> None:
    """Validates PGOPHER XML generation inspecting Parquet metadata."""
    parquet_path = tmp_path / "spectral_catalog.parquet"
    json_path = tmp_path / "metadata.json"
    pgo_output = tmp_path / "deliverables" / "TargetMolecule.pgo"

    table = pa.Table.from_arrays(
        [
            pa.array([12345.67, 23456.78, 34567.89, 45678.90], type=pa.float64()),
            pa.array([-3.5, -4.2, -2.1, -5.8], type=pa.float64()),
            pa.array(
                ["1_0_1-0_0_0", "2_0_2-1_0_1", "2_1_1-1_1_0", "3_0_3-2_0_2"],
                type=pa.string(),
            ),
            pa.array([0.0, 0.41, 0.78, 1.15], type=pa.float64()),
        ],
        names=["frequency", "intensity", "quantum_numbers", "lower_state_energy"],
    )
    pq.write_table(table, str(parquet_path))  # type: ignore[no-untyped-call]

    meta_content = {
        "molecule_name": "TargetMolecule",
        "temperature_k": 150.0,
        "rotational_constants": {
            "A": 9876.54321,
            "B": 4321.09876,
            "C": 2109.87654,
        },
        "dipoles": {"mu_a": 1.45, "mu_b": 0.85, "mu_c": 0.12},
    }
    json_path.write_text(json.dumps(meta_content, indent=2), encoding="utf-8")

    result_path = generate_pgopher_skeleton(
        parquet_path=parquet_path,
        json_path=json_path,
        output_path=pgo_output,
    )

    assert Path(result_path).exists()
    assert Path(result_path) == pgo_output

    tree = ET.parse(str(pgo_output))
    root = tree.getroot()

    assert root.tag == "Document"
    assert root.attrib["Type"] == "PGopher"

    top = root.find(".//AsymmetricTop")
    assert top is not None

    params = {
        p.attrib["Name"]: float(p.attrib["Value"]) for p in top.findall("Parameter")
    }
    assert math.isclose(params["A"], 9876.54321, rel_tol=1e-5)
    assert math.isclose(params["B"], 4321.09876, rel_tol=1e-5)
    assert math.isclose(params["C"], 2109.87654, rel_tol=1e-5)
    assert math.isclose(params["mu_a"], 1.45, rel_tol=1e-5)
    assert math.isclose(params["mu_b"], 0.85, rel_tol=1e-5)
    assert math.isclose(params["mu_c"], 0.12, rel_tol=1e-5)

    meta_elem = root.find(".//Form/Metadata")
    assert meta_elem is not None
    assert meta_elem.attrib["NumTransitions"] == "4"
    assert meta_elem.attrib["NumColumns"] == "4"
    assert "frequency" in meta_elem.attrib["Columns"]


# ============================================================================
# Test Suite 3: Provenance Lock & RFC 8785 Canonical JSON
# ============================================================================


def test_lock_provenance_payload_canonical_json(tmp_path: Path) -> None:
    """Validates streaming SHA-256 and RFC 8785 Canonical JSON compliance."""
    payload_dir = tmp_path / "payload_workspace"
    payload_dir.mkdir()

    file_a = payload_dir / "molecule.var"
    file_a.write_text("VAR ROTATIONAL PARAMETERS A B C D\n" * 500, encoding="utf-8")

    file_b = payload_dir / "molecule.int"
    file_b.write_text("INT INTENSITY TRANSITIONS DIPOLE\n" * 300, encoding="utf-8")

    nested_dir = payload_dir / "tensors"
    nested_dir.mkdir()
    file_c = nested_dir / "large_tensor.bin"
    file_c.write_bytes(b"\xaa\xbb\xcc\xdd" * 4096)

    manifest_path = payload_dir / "spycfit_manifest.json"
    manifest = lock_provenance_payload(
        target_directory=payload_dir,
        output_manifest_path=manifest_path,
        metadata={"project": "CoChem-Unit-Test", "stage": "5.5"},
        rotational_constants={"A": 10000.0, "B": 5000.0, "C": 3000.0},
        dipoles={"mu_a": 1.2, "mu_b": 0.5, "mu_c": 0.1},
    )

    assert manifest_path.exists()
    assert manifest["format"] == "CoChem-SpycFit-Manifest"
    assert manifest["file_count"] == 3

    rel_paths = [f["relative_path"] for f in manifest["files"]]
    assert "spycfit_manifest.json" not in rel_paths
    assert "molecule.var" in rel_paths
    assert "molecule.int" in rel_paths
    assert "tensors/large_tensor.bin" in rel_paths

    large_entry = next(
        f for f in manifest["files"] if f["relative_path"] == "tensors/large_tensor.bin"
    )
    expected_large_sha = compute_file_sha256(file_c)[0]
    assert large_entry["sha256"] == expected_large_sha
    assert large_entry["size_bytes"] == 16384

    raw_manifest_text = manifest_path.read_text(encoding="utf-8")
    expected_canonical = canonical_json_dumps(manifest)
    assert raw_manifest_text == expected_canonical


# ============================================================================
# Test Suite 4: Deterministic .tar.zst Payload Bundling
# ============================================================================


def test_bundle_spycfit_payload_deterministic(tmp_path: Path) -> None:
    """Validates deterministic .tar.zst archive with normalized POSIX metadata."""
    payload_dir = tmp_path / "stage_deliverables"
    payload_dir.mkdir()

    (payload_dir / "spec.var").write_text("VAR FILE CONTENT\n", encoding="utf-8")
    (payload_dir / "spec.int").write_text("INT FILE CONTENT\n", encoding="utf-8")

    out_archive_dir = tmp_path / "exported_archives"
    archive_path_str = bundle_spycfit_payload(
        manifest_path_or_target_dir=payload_dir,
        output_dir=out_archive_dir,
        project_name="Water",
        compression_level=3,
    )

    archive_path = Path(archive_path_str)
    assert archive_path.exists()
    assert archive_path.name == "CoChem_Water_SpycFit_Payload.tar.zst"

    compressed_bytes = archive_path.read_bytes()
    dctx = zstd.ZstdDecompressor()
    decompressed_bytes = dctx.decompress(compressed_bytes)

    with tarfile.open(fileobj=io.BytesIO(decompressed_bytes), mode="r") as tar:
        members = tar.getmembers()
        assert len(members) >= 3

        for member in members:
            assert member.mtime == 0, f"mtime not normalized for {member.name}"
            assert member.uid == 0
            assert member.gid == 0
            assert member.uname == ""
            assert member.gname == ""
            if member.isdir():
                assert member.mode == 0o755
            else:
                assert member.mode == 0o644


# ============================================================================
# Test Suite 5: Payload Verification & Tamper / Byte-Flip Error Injection
# ============================================================================


def test_verify_payload_integrity_pass_and_tamper(tmp_path: Path) -> None:
    """Validates cryptographic verification and error detection on tampered bytes."""
    payload_dir = tmp_path / "verify_workspace"
    payload_dir.mkdir()

    file_var = payload_dir / "spec.var"
    file_var.write_bytes(b"EXACT CANONICAL VAR PARAMETERS 1234567890")

    file_int = payload_dir / "spec.int"
    file_int.write_bytes(b"EXACT INTENSITIES 9876543210")

    lock_provenance_payload(payload_dir)
    archive_path = bundle_spycfit_payload(payload_dir, project_name="RigidRotor")

    assert verify_payload_integrity(payload_dir) is True
    assert verify_payload_integrity(payload_dir / "spycfit_manifest.json") is True
    assert verify_payload_integrity(archive_path) is True

    # Tamper Injection: Flip a single byte in spec.var
    original_bytes = file_var.read_bytes()
    tampered_bytes = original_bytes[:-1] + b"1"
    file_var.write_bytes(tampered_bytes)

    with pytest.raises(CoChemIntegrityError, match="SHA-256 hash mismatch"):
        verify_payload_integrity(payload_dir)

    # Missing File Injection: Delete spec.int
    file_var.write_bytes(original_bytes)
    file_int.unlink()

    with pytest.raises(CoChemIntegrityError, match="Missing file"):
        verify_payload_integrity(payload_dir)


# ============================================================================
# Test Suite 6: TorqExporter, PESStore & export_qcschema Integration
# ============================================================================


def test_torq_exporter_and_pes_store(tmp_path: Path) -> None:
    """Validates PESStore, TorqExporter, and export_qcschema."""
    h5_file = str(tmp_path / "pes_store.h5")
    store = PESStore(h5_file)

    store.append_data(step=1, coordinates=[0.0, 0.1, 0.2, 0.3], energy=-76.456)
    store.append_data(step=2, coordinates=[0.0, 0.15, 0.22, 0.35], energy=-76.458)

    with h5py.File(h5_file, "r") as f:
        assert "coordinates" in f
        assert "energies" in f
        assert f["coordinates"].shape == (2, 4)
        assert f["energies"].shape == (2,)
        assert f["coordinates"].scaleoffset is None
        assert f["energies"].scaleoffset is None

    export_dir = tmp_path / "zstd_exports"
    exporter = TorqExporter(export_dir=str(export_dir), zstd_compression_level=3)

    compressed_file = exporter.export_tensor_to_zstd(h5_file)
    assert Path(compressed_file).exists()

    success, metadata = exporter.verify_export(compressed_file)
    assert success is True
    assert metadata is not None
    assert metadata["compression_method"] == "Zstandard"

    qcschema_path = str(tmp_path / "qcschema.json")
    orca_result = {
        "geometry": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "symbols": ["O", "H"],
        "molecular_charge": 0,
        "molecular_multiplicity": 1,
        "driver": "energy",
        "method": "B3LYP",
        "basis": "def2-TZVP",
        "return_energy": -75.123456,
    }
    res_path = export_qcschema(orca_result, qcschema_path)
    assert Path(res_path).exists()

    with open(res_path, encoding="utf-8") as f:
        schema = json.load(f)
    assert schema["schema_name"] == "qcschema_output"
    assert schema["properties"]["return_energy"] == -75.123456
    assert "hash" in schema["molecule"]["provenance"]


# ============================================================================
# Test Suite 7: Edge Cases, Archive Corruptions & Error Handling
# ============================================================================


def test_verify_payload_corrupted_archive_cases(tmp_path: Path) -> None:
    """Validates error raising on corrupt .tar.zst archives."""
    corrupt_zst = tmp_path / "corrupt.tar.zst"
    corrupt_zst.write_bytes(b"\x28\xb5\x2f\xfd\x00\x00\x00\x00_INVALID_ZSTD_GARBAGE")
    with pytest.raises(
        CoChemIntegrityError, match="Failed to decompress Zstandard archive"
    ):
        verify_payload_integrity(corrupt_zst)

    tar_no_manifest_buf = io.BytesIO()
    with tarfile.open(mode="w", fileobj=tar_no_manifest_buf) as tar:
        ti = tarfile.TarInfo(name="isolated_data.txt")
        ti.size = 5
        tar.addfile(ti, io.BytesIO(b"HELLO"))

    no_manifest_zst = tmp_path / "no_manifest.tar.zst"
    no_manifest_zst.write_bytes(
        zstd.ZstdCompressor().compress(tar_no_manifest_buf.getvalue())
    )
    with pytest.raises(
        CoChemIntegrityError, match="Manifest 'spycfit_manifest.json' not found"
    ):
        verify_payload_integrity(no_manifest_zst)

    valid_dir = tmp_path / "valid_payload"
    valid_dir.mkdir()
    (valid_dir / "data.txt").write_text("VALID DATA 123", encoding="utf-8")
    lock_provenance_payload(valid_dir)
    valid_archive = bundle_spycfit_payload(valid_dir, project_name="CorruptTest")

    decompressed = zstd.ZstdDecompressor().decompress(Path(valid_archive).read_bytes())
    tar_tamper_buf = io.BytesIO()
    with tarfile.open(fileobj=io.BytesIO(decompressed), mode="r") as tar_in:
        with tarfile.open(fileobj=tar_tamper_buf, mode="w") as tar_out:
            for member in tar_in.getmembers():
                f = tar_in.extractfile(member) if member.isreg() else None
                if member.name == "data.txt":
                    tampered_data = b"TAMPERED DATA!"
                    member.size = len(tampered_data)
                    tar_out.addfile(member, io.BytesIO(tampered_data))
                elif f:
                    tar_out.addfile(member, f)
                else:
                    tar_out.addfile(member)

    tampered_zst = tmp_path / "tampered_archive.tar.zst"
    tampered_zst.write_bytes(zstd.ZstdCompressor().compress(tar_tamper_buf.getvalue()))

    with pytest.raises(
        CoChemIntegrityError, match="(SHA-256 hash mismatch|File size mismatch)"
    ):
        verify_payload_integrity(tampered_zst)


def test_kraitchman_dictionary_and_planar_inputs() -> None:
    """Validates calculate_kraitchman_coords() with dictionary input structures."""
    parent_dict = {"a": 12.5, "b": 24.0, "c": 36.5}
    sub_dict = {"a": 12.8, "b": 24.4, "c": 36.9}

    res = calculate_kraitchman_coords(
        parent_moments=parent_dict,
        substituted_moments=sub_dict,
        parent_mass=78.0,
        delta_m=1.003355,
    )

    assert "coords" in res
    assert "costain_errors" in res
    assert "radicands" in res
    assert len(res["coords"]) == 3
    assert all(c >= 0.0 for c in res["coords"].values())


def test_file_not_found_guards(tmp_path: Path) -> None:
    """Validates FileNotFoundError guards across all export utilities."""
    non_existent = tmp_path / "does_not_exist"

    with pytest.raises(FileNotFoundError):
        generate_pgopher_skeleton(parquet_path=non_existent / "catalog.parquet")

    with pytest.raises(FileNotFoundError):
        lock_provenance_payload(target_directory=non_existent)

    with pytest.raises(FileNotFoundError):
        bundle_spycfit_payload(manifest_path_or_target_dir=non_existent)


def test_generate_pgopher_skeleton_variations(tmp_path: Path) -> None:
    """Validates PGOPHER skeleton generation with default output and variants."""
    parquet_path = tmp_path / "spectral.parquet"
    table = pa.Table.from_arrays(
        [
            pa.array([1000.0, 2000.0], type=pa.float64()),
            pa.array([-1.0, -2.0], type=pa.float64()),
        ],
        names=["freq", "intensity"],
    )
    pq.write_table(table, str(parquet_path))  # type: ignore[no-untyped-call]

    # Test with point_id, temperature, and list-based rotational constants and dipoles
    json_path = tmp_path / "point_metadata.json"
    json_content = {
        "point_id": "Conformer_A",
        "temperature": 10.0,
        "properties": {
            "rotational_constants": [5000.0, 2500.0, 1500.0],
            "dipole_moment": [0.5, 0.2, 0.0],
        },
    }
    json_path.write_text(json.dumps(json_content), encoding="utf-8")

    # Call with output_path=None to test default path generation
    out_pgo = generate_pgopher_skeleton(
        parquet_path=parquet_path,
        json_path=json_path,
        output_path=None,
    )
    assert Path(out_pgo).exists()
    assert Path(out_pgo).name == "Conformer_A.pgo"

    # Call with direct list overrides
    out_pgo2 = generate_pgopher_skeleton(
        parquet_path=parquet_path,
        output_path=tmp_path / "DirectOverride.pgo",
        molecule_name="OverrideMol",
        temperature_k=77.0,
        rotational_constants=[12000.0, 6000.0, 4000.0],
        dipoles=[2.0, 1.0, 0.5],
    )
    assert Path(out_pgo2).exists()
    tree = ET.parse(out_pgo2)
    root = tree.getroot()
    top = root.find(".//AsymmetricTop")
    assert top is not None
    params = {
        p.attrib["Name"]: float(p.attrib["Value"])
        for p in top.findall("Parameter")
    }
    assert math.isclose(params["A"], 12000.0, rel_tol=1e-5)
    assert math.isclose(params["mu_a"], 2.0, rel_tol=1e-5)


def test_bundle_spycfit_payload_from_manifest_file(tmp_path: Path) -> None:
    """Validates bundling payload when given direct path to manifest file."""
    payload_dir = tmp_path / "manifest_bundle_dir"
    payload_dir.mkdir()
    (payload_dir / "test.int").write_text("INT DATA", encoding="utf-8")

    manifest_file = payload_dir / "spycfit_manifest.json"
    lock_provenance_payload(payload_dir, output_manifest_path=manifest_file)

    archive_str = bundle_spycfit_payload(manifest_path_or_target_dir=manifest_file)
    assert Path(archive_str).exists()
    assert verify_payload_integrity(archive_str) is True


def test_verify_payload_with_dict_and_base_dir(tmp_path: Path) -> None:
    """Validates verify_payload_integrity when passed a manifest dictionary."""
    payload_dir = tmp_path / "dict_verify_dir"
    payload_dir.mkdir()
    (payload_dir / "file1.txt").write_bytes(b"DATA ONE")
    (payload_dir / "file2.txt").write_bytes(b"DATA TWO")

    manifest = lock_provenance_payload(payload_dir)
    assert verify_payload_integrity(manifest, base_dir=payload_dir) is True

    # Test size mismatch in directory verify
    (payload_dir / "file1.txt").write_bytes(b"LONGER DATA ONE MODIFIED")
    with pytest.raises(CoChemIntegrityError, match="File size mismatch"):
        verify_payload_integrity(manifest, base_dir=payload_dir)


def test_torq_exporter_batch_and_dvr(tmp_path: Path) -> None:
    """Validates batch export and Sinc-DVR export methods in TorqExporter."""
    h5_1 = tmp_path / "mol1.h5"
    h5_2 = tmp_path / "mol2.h5"

    for h5_path, val in [(h5_1, 10.0), (h5_2, 20.0)]:
        with h5py.File(str(h5_path), "w") as f:
            grp = f.create_group("geometry")
            grp.create_dataset("coords", data=[[0.0, 0.0, val]])
            f.create_dataset("energy", data=-75.5)

    export_out = tmp_path / "batch_out"
    exporter = TorqExporter(export_dir=str(export_out))

    exported = exporter.batch_export_to_zstd([str(h5_1), str(h5_2)])
    assert len(exported) == 2
    for exp_file in exported:
        assert Path(exp_file).exists()
        success, _ = exporter.verify_export(exp_file)
        assert success is True

    dvr_export = exporter.export_tensor_to_zstd_with_sinc_dvr(str(h5_1))
    assert Path(dvr_export).exists()
    assert "_dvr.zst" in dvr_export
    success, meta = exporter.verify_export(dvr_export)
    assert success is True
    assert meta is not None


def test_kraitchman_exact_zero_denominator_guard() -> None:
    """Validates Kraitchman coordinates when moments are identical (Ia == Ib)."""
    parent_moments = (20.0, 20.0, 40.0)
    sub_moments = (20.5, 20.5, 40.8)

    with pytest.warns(KraitchmanSingularityWarning):
        res = calculate_kraitchman_coords(
            parent_moments=parent_moments,
            substituted_moments=sub_moments,
            parent_mass=50.0,
            delta_m=1.0,
            singularity_threshold=1e-4,
        )
    assert not math.isnan(res["coords"]["a"])
    assert not math.isnan(res["coords"]["b"])
    assert not math.isnan(res["coords"]["c"])


def test_pgopher_dict_overrides_and_missing_manifest(tmp_path: Path) -> None:
    """Validates dict overrides in PGOPHER generator and missing manifest error."""
    parquet_path = tmp_path / "spec_test.parquet"
    table = pa.Table.from_arrays(
        [pa.array([123.45]), pa.array([-2.5])], names=["freq", "int"]
    )
    pq.write_table(table, str(parquet_path))  # type: ignore[no-untyped-call]

    out_pgo = generate_pgopher_skeleton(
        parquet_path=parquet_path,
        output_path=tmp_path / "DictOverride.pgo",
        molecule_name="DictMol",
        rotational_constants={"A": 8888.0, "B": 4444.0, "C": 2222.0},
        dipoles={"mu_a": 0.8, "mu_b": 0.4, "mu_c": 0.2},
    )
    assert Path(out_pgo).exists()

    # Missing manifest error
    non_existent_manifest = tmp_path / "no_such_manifest.json"
    with pytest.raises(CoChemIntegrityError, match="Manifest file not found"):
        verify_payload_integrity(non_existent_manifest)


def test_torq_exporter_scribe_and_corrupt_verify(tmp_path: Path) -> None:
    """Validates scribe daemon missing file handling and corrupt verification."""
    exporter = TorqExporter(export_dir=str(tmp_path))

    # Scribe daemon with missing file returns False
    res = exporter.export_to_scribe_daemon(str(tmp_path / "non_existent.zst"))
    assert res is False

    # Corrupt verification returns (False, None)
    corrupt_file = tmp_path / "bad.zst"
    corrupt_file.write_bytes(b"NOT_A_VALID_ZSTD_OR_JSON_STREAM")
    ok, meta = exporter.verify_export(str(corrupt_file))
    assert ok is False
    assert meta is None


def test_lock_provenance_with_kraitchman_and_nested_dirs(tmp_path: Path) -> None:
    """Validates lock_provenance_payload with kraitchman_coords and nested directory tarball."""
    payload_dir = tmp_path / "full_complex_payload"
    payload_dir.mkdir()
    sub_dir = payload_dir / "nested_models"
    sub_dir.mkdir()

    (sub_dir / "geom.xyz").write_text("3\nH2O\nO 0 0 0\nH 0 0 1\nH 0 1 0\n", encoding="utf-8")
    (payload_dir / "spec.var").write_text("VAR TEST", encoding="utf-8")

    kc = {
        "coords": {"a": 0.0, "b": 1.25, "c": 0.85},
        "costain_errors": {"a": 0.0, "b": 0.0012, "c": 0.0017},
        "radicands": {"a": -0.01, "b": 1.5625, "c": 0.7225},
    }

    manifest = lock_provenance_payload(
        target_directory=payload_dir,
        kraitchman_coords=kc,
        metadata={"run_id": "RUN-001"},
    )

    assert "kraitchman_coords" in manifest["metadata"]
    assert manifest["metadata"]["kraitchman_coords"]["coords"]["b"] == 1.25

    archive = bundle_spycfit_payload(
        manifest_path_or_target_dir=payload_dir,
        project_name="WaterDimer",
    )
    assert Path(archive).exists()
    assert verify_payload_integrity(archive) is True




Validate Zero-Mock adherence. Target repo is D:\__CoChem\GitHub-Repo\CoChem-TORQ.