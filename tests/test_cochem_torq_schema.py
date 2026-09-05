"""CoChem-TORQ: Test Suite for Phase 1 Hardware Schema & 5-Whys Gatekeeper.

==========================================================================
Phase 1 (Stage 0.0) Test Suite
------------------------------
Zero-Mock test suite verifying strict Pydantic v2 validation of the Golden Registry
state (cochem_system_config.json), physical hardware constraint enforcement (MPI threads,
GPU VRAM, memory limits, and resolved filesystem paths), air-gap collision assertions,
and 5-Whys root cause analysis trace generation upon schema validation failures.

All tests operate against authentic physical files, directories, and data structures
within pytest `tmp_path`. Zero synthetic mocks permitted.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from Libraries.cochem_torq_schema import (
    TorqHardwareSchema,
    TorqSchemaValidationError,
    format_5_whys_error,
    validate_registry_state,
)


class TestTorqSchema:
    """Test suite for Libraries/cochem_torq_schema.py."""

    def test_torq_hardware_schema_valid(self, tmp_path: Path) -> None:
        """Verify instantiation with fully valid hardware configuration and paths."""
        scratch = tmp_path / "scratch"
        artifacts = tmp_path / "artifacts"
        torq_lib = tmp_path / "lib"
        scratch.mkdir()
        artifacts.mkdir()
        torq_lib.mkdir()

        schema = TorqHardwareSchema(
            mpi_threads=8,
            gpu_vram_gb=16.0,
            gpu_device_ids=[0, 1],
            maxcore_mb=4096,
            scratch_dir=scratch,
            artifacts_dir=artifacts,
            torq_lib_dir=torq_lib,
            cuda_enabled=True,
            strict_airgap=True,
        )
        assert schema.mpi_threads == 8
        assert schema.gpu_vram_gb == 16.0
        assert schema.gpu_device_ids == [0, 1]
        assert schema.maxcore_mb == 4096
        assert schema.scratch_dir == scratch.resolve()
        assert schema.artifacts_dir == artifacts.resolve()
        assert schema.torq_lib_dir == torq_lib.resolve()
        assert schema.cuda_enabled is True
        assert schema.strict_airgap is True

    def test_torq_hardware_schema_invalid_threads_zero_or_negative(self, tmp_path: Path) -> None:
        """Assert threads < 1 fails validation."""
        scratch = tmp_path / "scratch"
        artifacts = tmp_path / "artifacts"
        torq_lib = tmp_path / "lib"
        scratch.mkdir()
        artifacts.mkdir()
        torq_lib.mkdir()

        with pytest.raises(ValidationError):
            TorqHardwareSchema(
                mpi_threads=0,
                scratch_dir=scratch,
                artifacts_dir=artifacts,
                torq_lib_dir=torq_lib,
            )

        with pytest.raises(ValidationError):
            TorqHardwareSchema(
                mpi_threads=-4,
                scratch_dir=scratch,
                artifacts_dir=artifacts,
                torq_lib_dir=torq_lib,
            )

    def test_torq_hardware_schema_thread_upper_limit(self, tmp_path: Path) -> None:
        """Assert threads > 1024 fails validation."""
        scratch = tmp_path / "scratch"
        artifacts = tmp_path / "artifacts"
        torq_lib = tmp_path / "lib"
        scratch.mkdir()
        artifacts.mkdir()
        torq_lib.mkdir()

        with pytest.raises(ValidationError):
            TorqHardwareSchema(
                mpi_threads=2048,
                scratch_dir=scratch,
                artifacts_dir=artifacts,
                torq_lib_dir=torq_lib,
            )

    def test_torq_hardware_schema_invalid_maxcore(self, tmp_path: Path) -> None:
        """Assert maxcore_mb < 256 fails validation."""
        scratch = tmp_path / "scratch"
        artifacts = tmp_path / "artifacts"
        torq_lib = tmp_path / "lib"
        scratch.mkdir()
        artifacts.mkdir()
        torq_lib.mkdir()

        with pytest.raises(ValidationError):
            TorqHardwareSchema(
                maxcore_mb=128,
                scratch_dir=scratch,
                artifacts_dir=artifacts,
                torq_lib_dir=torq_lib,
            )

    def test_torq_hardware_schema_negative_vram(self, tmp_path: Path) -> None:
        """Assert gpu_vram_gb < 0.0 fails validation."""
        scratch = tmp_path / "scratch"
        artifacts = tmp_path / "artifacts"
        torq_lib = tmp_path / "lib"
        scratch.mkdir()
        artifacts.mkdir()
        torq_lib.mkdir()

        with pytest.raises(ValidationError):
            TorqHardwareSchema(
                gpu_vram_gb=-1.0,
                scratch_dir=scratch,
                artifacts_dir=artifacts,
                torq_lib_dir=torq_lib,
            )

    def test_torq_hardware_schema_airgap_collision_identical(self, tmp_path: Path) -> None:
        """Assert scratch_dir == artifacts_dir fails air-gap validation."""
        same_dir = tmp_path / "shared"
        same_dir.mkdir()
        torq_lib = tmp_path / "lib"
        torq_lib.mkdir()

        with pytest.raises(ValidationError, match="Air-gap violation"):
            TorqHardwareSchema(
                scratch_dir=same_dir,
                artifacts_dir=same_dir,
                torq_lib_dir=torq_lib,
                strict_airgap=True,
            )

    def test_torq_hardware_schema_airgap_collision_nested_scratch(self, tmp_path: Path) -> None:
        """Assert scratch_dir inside artifacts_dir fails air-gap validation."""
        parent_dir = tmp_path / "parent_art"
        child_scratch = parent_dir / "child_scratch"
        torq_lib = tmp_path / "lib"
        parent_dir.mkdir()
        child_scratch.mkdir()
        torq_lib.mkdir()

        with pytest.raises(ValidationError, match="Air-gap violation"):
            TorqHardwareSchema(
                scratch_dir=child_scratch,
                artifacts_dir=parent_dir,
                torq_lib_dir=torq_lib,
                strict_airgap=True,
            )

    def test_torq_hardware_schema_airgap_collision_nested_artifacts(self, tmp_path: Path) -> None:
        """Assert artifacts_dir inside scratch_dir fails air-gap validation."""
        parent_scratch = tmp_path / "parent_scratch"
        child_art = parent_scratch / "child_art"
        torq_lib = tmp_path / "lib"
        parent_scratch.mkdir()
        child_art.mkdir()
        torq_lib.mkdir()

        with pytest.raises(ValidationError, match="Air-gap violation"):
            TorqHardwareSchema(
                scratch_dir=parent_scratch,
                artifacts_dir=child_art,
                torq_lib_dir=torq_lib,
                strict_airgap=True,
            )

    def test_torq_hardware_schema_extra_fields_forbidden(self, tmp_path: Path) -> None:
        """Assert arbitrary unknown fields are strictly forbidden."""
        scratch = tmp_path / "scratch"
        artifacts = tmp_path / "artifacts"
        torq_lib = tmp_path / "lib"
        scratch.mkdir()
        artifacts.mkdir()
        torq_lib.mkdir()

        with pytest.raises(ValidationError):
            TorqHardwareSchema(
                scratch_dir=scratch,
                artifacts_dir=artifacts,
                torq_lib_dir=torq_lib,
                unrecognized_field="illegal_value",  # type: ignore[call-arg]
            )

    def test_validate_registry_state_dict(self, tmp_path: Path) -> None:
        """Verify validate_registry_state directly with a configuration dictionary."""
        scratch = tmp_path / "scratch"
        artifacts = tmp_path / "artifacts"
        torq_lib = tmp_path / "lib"
        scratch.mkdir()
        artifacts.mkdir()
        torq_lib.mkdir()

        data = {
            "mpi_threads": 4,
            "gpu_vram_gb": 8.0,
            "gpu_device_ids": [0],
            "maxcore_mb": 2048,
            "scratch_dir": str(scratch),
            "artifacts_dir": str(artifacts),
            "torq_lib_dir": str(torq_lib),
            "cuda_enabled": True,
        }
        schema = validate_registry_state(data)
        assert isinstance(schema, TorqHardwareSchema)
        assert schema.mpi_threads == 4
        assert schema.gpu_vram_gb == 8.0
        assert schema.maxcore_mb == 2048

    def test_validate_registry_state_json_file(self, tmp_path: Path) -> None:
        """Verify validate_registry_state from a real physical cochem_system_config.json file."""
        scratch = tmp_path / "scratch"
        artifacts = tmp_path / "artifacts"
        torq_lib = tmp_path / "lib"
        scratch.mkdir()
        artifacts.mkdir()
        torq_lib.mkdir()

        json_path = tmp_path / "cochem_system_config.json"
        data = {
            "mpi_threads": 16,
            "gpu_vram_gb": 24.0,
            "gpu_device_ids": [0],
            "maxcore_mb": 8192,
            "scratch_dir": str(scratch),
            "artifacts_dir": str(artifacts),
            "torq_lib_dir": str(torq_lib),
            "cuda_enabled": False,
        }
        with open(json_path, "w", encoding="utf-8") as fp:
            json.dump(data, fp)

        schema = validate_registry_state(json_path)
        assert schema.mpi_threads == 16
        assert schema.maxcore_mb == 8192
        assert schema.gpu_vram_gb == 24.0

    def test_validate_registry_state_5_whys_on_missing_file(self, tmp_path: Path) -> None:
        """Verify 5-Whys diagnostic trace generated when config file is missing."""
        missing_file = tmp_path / "non_existent.json"
        with pytest.raises(TorqSchemaValidationError) as exc_info:
            validate_registry_state(missing_file)

        assert exc_info.value.five_whys_trace is not None
        assert "Why 1 (Symptom)" in exc_info.value.five_whys_trace
        assert "Why 2 (Trigger)" in exc_info.value.five_whys_trace
        assert "Why 3 (Boundary)" in exc_info.value.five_whys_trace
        assert "Why 4 (Origin)" in exc_info.value.five_whys_trace
        assert "Why 5 (Architectural Resolution)" in exc_info.value.five_whys_trace

    def test_validate_registry_state_5_whys_on_corrupt_json(self, tmp_path: Path) -> None:
        """Verify 5-Whys diagnostic trace generated when config file contains invalid JSON."""
        corrupt_file = tmp_path / "corrupt_config.json"
        corrupt_file.write_text("{ unquoted_invalid_json: true, }", encoding="utf-8")

        with pytest.raises(TorqSchemaValidationError) as exc_info:
            validate_registry_state(corrupt_file)

        assert exc_info.value.five_whys_trace is not None
        assert "Why 1 (Symptom)" in exc_info.value.five_whys_trace
        assert "Why 5 (Architectural Resolution)" in exc_info.value.five_whys_trace

    def test_validate_registry_state_5_whys_on_invalid_type(self) -> None:
        """Verify 5-Whys diagnostic trace when an invalid type is passed."""
        with pytest.raises(TorqSchemaValidationError) as exc_info:
            validate_registry_state(12345)  # type: ignore[arg-type]

        assert exc_info.value.five_whys_trace is not None
        assert "Why 1 (Symptom)" in exc_info.value.five_whys_trace
        assert "Why 5 (Architectural Resolution)" in exc_info.value.five_whys_trace

    def test_format_5_whys_error(self) -> None:
        """Verify format_5_whys_error output structure and content."""
        trace = format_5_whys_error(
            ValueError("Negative threads"),
            {
                "field": "mpi_threads",
                "value": "-4",
                "rule": "Threads must be >= 1",
                "origin": "test_input",
                "remediation": "Set mpi_threads >= 1",
            },
        )
        assert "=== 5-WHYS ROOT CAUSE ANALYSIS TRACE ===" in trace
        assert "Why 1 (Symptom)" in trace
        assert "Why 2 (Trigger)" in trace
        assert "Why 3 (Boundary)" in trace
        assert "Why 4 (Origin)" in trace
        assert "Why 5 (Architectural Resolution)" in trace
        assert "mpi_threads" in trace
        assert "-4" in trace
        assert "Threads must be >= 1" in trace
        assert "Set mpi_threads >= 1" in trace
