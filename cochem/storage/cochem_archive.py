"""Standardized .cochem_archive Container Format & Security Engine.

Provides cryptographic packaging, streaming Zstandard decompression, decompression-bomb
quota defenses, and traversal attack (Zip Slip, NTFS stream, symlink escape) prevention.
"""

import hashlib
import io
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Set
from uuid import UUID, uuid4

import zstandard
from pydantic import ConfigDict, Field, model_validator
from pydantic.main import BaseModel


class ArchiveSecurityError(Exception):
    """Raised when an archive violates sandbox isolation or integrity checks."""


class FileChecksum(BaseModel):
    """Integrity checksum and size metadata for an archive member."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sha256: str = Field(..., pattern=r"^[a-f0-9]{64}$")
    size_bytes: int = Field(..., ge=0)


class ArchiveManifest(BaseModel):
    """Cryptographic manifest describing archive contents and metadata."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default="1.0.0", pattern=r"^\d+\.\d+\.\d+$")
    archive_id: UUID = Field(default_factory=uuid4)
    created_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    cochem_version: str
    files: Dict[str, FileChecksum]
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_core_files(self) -> "ArchiveManifest":
        """Enforce presence of primary data.h5 dataset container."""
        if "data.h5" not in self.files:
            raise ValueError(
                "Archive manifest must contain primary 'data.h5' store."
            )
        return self


class CochemArchive:
    """Packager and secure unpacker for standardized .cochem_archive containers."""

    MAX_EXTRACTION_BYTES: int = 10 * 1024 * 1024 * 1024  # 10 GB limit

    @staticmethod
    def _compute_sha256(path: Path) -> str:
        """Compute SHA-256 digest of file reading in 64 KB blocks."""
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()

    @classmethod
    def pack(
        cls,
        source_dir: Path,
        output_archive: Path,
        cochem_version: str,
        metadata: Optional[Dict[str, Any]] = None,
        cctx_level: int = 6,
    ) -> Path:
        """Package directory containing data.h5 into a standardized .cochem_archive.

        Args:
            source_dir: Source directory containing files to archive.
            output_archive: Destination path for compressed archive.
            cochem_version: Semantic version string of CoChem.
            metadata: Optional arbitrary dictionary metadata.
            cctx_level: Zstandard compression level (default 6).

        Returns:
            Resolved Path to created archive.
        """
        resolved_src = Path(source_dir).resolve()
        data_h5_path = resolved_src / "data.h5"
        if not data_h5_path.exists():
            raise FileNotFoundError(
                f"Source directory missing primary 'data.h5' store: {resolved_src}"
            )

        files_map: Dict[str, FileChecksum] = {}
        for entry in sorted(resolved_src.rglob("*")):
            if entry.is_file() and not entry.is_symlink() and entry.name != "manifest.json":
                rel_posix = entry.relative_to(resolved_src).as_posix()
                sha256 = cls._compute_sha256(entry)
                size_bytes = entry.stat().st_size
                files_map[rel_posix] = FileChecksum(
                    sha256=sha256,
                    size_bytes=size_bytes,
                )

        manifest = ArchiveManifest(
            cochem_version=cochem_version,
            files=files_map,
            metadata=metadata or {},
        )
        manifest_bytes = manifest.model_dump_json(indent=2).encode("utf-8")

        resolved_out = Path(output_archive).resolve()
        resolved_out.parent.mkdir(parents=True, exist_ok=True)

        cctx = zstandard.ZstdCompressor(level=cctx_level)
        mtime_ts = int(manifest.created_utc.timestamp())

        with open(resolved_out, "wb") as f_out:
            with cctx.stream_writer(f_out) as compressor:
                with tarfile.open(fileobj=compressor, mode="w|") as tar:
                    m_info = tarfile.TarInfo(name="manifest.json")
                    m_info.size = len(manifest_bytes)
                    m_info.mtime = mtime_ts
                    m_info.mode = 0o644
                    m_info.uid = 0
                    m_info.gid = 0
                    m_info.uname = ""
                    m_info.gname = ""
                    tar.addfile(m_info, io.BytesIO(manifest_bytes))

                    for rel_path in sorted(files_map.keys()):
                        f_path = resolved_src / rel_path
                        t_info = tarfile.TarInfo(name=rel_path)
                        t_info.size = f_path.stat().st_size
                        t_info.mtime = mtime_ts
                        t_info.mode = 0o644
                        t_info.uid = 0
                        t_info.gid = 0
                        t_info.uname = ""
                        t_info.gname = ""
                        with open(f_path, "rb") as f_data:
                            tar.addfile(t_info, f_data)

        return resolved_out

    @classmethod
    def unpack(cls, archive_path: Path, destination_dir: Path) -> ArchiveManifest:
        """Securely unpack .cochem_archive into destination directory with strict integrity checks.

        Args:
            archive_path: Path to existing .cochem_archive.
            destination_dir: Directory where archive members will be extracted.

        Returns:
            Validated ArchiveManifest.
        """
        resolved_archive = Path(archive_path).resolve()
        if not resolved_archive.exists():
            raise FileNotFoundError(
                f"Archive file not found: {resolved_archive}"
            )

        dest_dir = Path(destination_dir).resolve()
        dest_dir.mkdir(parents=True, exist_ok=True)

        dctx = zstandard.ZstdDecompressor()
        total_extracted_bytes = 0
        extracted_rel_paths: Set[str] = set()

        with open(resolved_archive, "rb") as f_in:
            with dctx.stream_reader(f_in) as decompressor:
                with tarfile.open(fileobj=decompressor, mode="r|") as tar:
                    for member in tar:
                        if (
                            member.islnk()
                            or member.issym()
                            or member.ischr()
                            or member.isblk()
                            or member.isfifo()
                        ):
                            raise ArchiveSecurityError(
                                f"Archive contains forbidden link or special file: '{member.name}'"
                            )

                        norm_name = member.name.replace("\\", "/")
                        if norm_name.startswith("/"):
                            raise ArchiveSecurityError(
                                f"Archive member starts with root slash: '{member.name}'"
                            )
                        if ":" in norm_name:
                            raise ArchiveSecurityError(
                                f"Archive member contains forbidden character ':': '{member.name}'"
                            )

                        parts = [p for p in norm_name.split("/") if p]
                        if ".." in parts:
                            raise ArchiveSecurityError(
                                f"Path traversal detected for member: '{member.name}'"
                            )

                        clean_rel = "/".join(parts)
                        target_path = (dest_dir / clean_rel).resolve()

                        try:
                            target_path.relative_to(dest_dir)
                        except ValueError as err:
                            raise ArchiveSecurityError(
                                f"Path traversal detected for member: '{member.name}'"
                            ) from err

                        total_extracted_bytes += member.size
                        if total_extracted_bytes > cls.MAX_EXTRACTION_BYTES:
                            raise ArchiveSecurityError(
                                f"Extraction exceeded maximum allowable limit ({cls.MAX_EXTRACTION_BYTES} bytes)"
                            )

                        if member.isdir():
                            target_path.mkdir(parents=True, exist_ok=True)
                        elif member.isfile():
                            target_path.parent.mkdir(parents=True, exist_ok=True)
                            f_member = tar.extractfile(member)
                            if f_member is None:
                                raise ArchiveSecurityError(
                                    f"Failed to extract member: '{member.name}'"
                                )
                            with open(target_path, "wb") as f_dest:
                                shutil.copyfileobj(f_member, f_dest)
                            extracted_rel_paths.add(clean_rel)
                        else:
                            raise ArchiveSecurityError(
                                f"Unsupported member type: '{member.name}'"
                            )

        manifest_path = dest_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError("manifest.json not found in archive")

        manifest = ArchiveManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )

        expected_files = set(manifest.files.keys()) | {"manifest.json"}
        if extracted_rel_paths != expected_files:
            missing = expected_files - extracted_rel_paths
            extra = extracted_rel_paths - expected_files
            raise ArchiveSecurityError(
                f"Archive manifest member mismatch. Missing: {missing}, Extra: {extra}"
            )

        for rel_path, csum in manifest.files.items():
            file_disk = dest_dir / rel_path
            if not file_disk.exists():
                raise ArchiveSecurityError(f"Declared file missing: {rel_path}")
            if file_disk.stat().st_size != csum.size_bytes:
                raise ArchiveSecurityError(
                    f"Size mismatch for {rel_path}: expected {csum.size_bytes}, got {file_disk.stat().st_size}"
                )
            actual_sha = cls._compute_sha256(file_disk)
            if actual_sha != csum.sha256:
                raise ArchiveSecurityError(
                    f"SHA-256 mismatch for {rel_path}: expected {csum.sha256}, got {actual_sha}"
                )

        return manifest
