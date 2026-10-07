"""Execute SPCAT with supplied decks and parse supported fixed-width catalogs.

Legacy automatic deck writers are disabled until validated against SPCAT. Missing
engines, failed runs and unsupported quantum-number formats are explicit errors.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

try:
    from .cochem_torq_asymmetric_rotor import RotationalConstants, TransitionRecord
except ImportError:
    from cochem_torq_asymmetric_rotor import RotationalConstants, TransitionRecord


@dataclass(frozen=True)
class SPCATTransitionRecord(TransitionRecord):
    """Catalog record retaining uncertainty and quantum-number convention."""

    uncertainty_mhz: float
    quantum_number_format: int


class PickettSPCATRunner:
    """Run the requested SPCAT binary without substitution of another solver."""

    def __init__(self, spcat_bin_path: Path | str | None = None) -> None:
        if spcat_bin_path is not None:
            candidate = Path(spcat_bin_path).resolve()
            if not candidate.is_file():
                raise FileNotFoundError(
                    f"Requested SPCAT executable missing: {candidate}"
                )
            self.spcat_bin: Path | None = candidate
        else:
            resolved = shutil.which("spcat") or shutil.which("spcat.exe")
            self.spcat_bin = Path(resolved).resolve() if resolved else None

    def write_var_file(self, target_path: Path | str, c: RotationalConstants) -> Path:
        """Reject the legacy writer's unvalidated Hamiltonian and uncertainty deck."""
        raise NotImplementedError(
            "Legacy SPCAT .var generation is unvalidated: supply an explicit "
            "validated parameter deck, including state model and uncertainties."
        )

    def write_int_file(
        self,
        target_path: Path | str,
        c: RotationalConstants,
        temperature_k: float = 298.15,
        freq_min_mhz: float = 0.0,
        freq_max_mhz: float = 200000.0,
    ) -> Path:
        """Reject the legacy writer's invented partition function and invalid format."""
        raise NotImplementedError(
            "Legacy SPCAT .int generation is unvalidated: supply an explicit "
            "validated intensity deck with calculated partition function, "
            "temperature and principal-axis dipoles."
        )

    def run_spcat(
        self, base_name: str, working_dir: Path | str, timeout: float = 30.0
    ) -> Path:
        """Execute supplied decks in a fresh directory; retain execution provenance."""
        if self.spcat_bin is None or not self.spcat_bin.is_file():
            raise FileNotFoundError(
                "SPCAT executable unavailable; no alternate solver selected"
            )
        if Path(base_name).name != base_name or base_name in ("", ".", ".."):
            raise ValueError("SPCAT base_name must be a plain file stem")
        wd = Path(working_dir).resolve()
        inputs = [wd / f"{base_name}.{suffix}" for suffix in ("var", "int")]
        for path in inputs:
            if not path.is_file() or not path.stat().st_size:
                raise FileNotFoundError(
                    f"Explicit nonempty SPCAT input required: {path}"
                )
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("SPCAT timeout must be finite and positive")
        with tempfile.TemporaryDirectory(prefix="spcat_", dir=wd) as task_dir:
            task_path = Path(task_dir)
            for path in inputs:
                shutil.copy2(path, task_path / path.name)
            input_hashes = {
                path.name: hashlib.sha256(
                    (task_path / path.name).read_bytes()
                ).hexdigest()
                for path in inputs
            }
            executable_hash = hashlib.sha256(self.spcat_bin.read_bytes()).hexdigest()
            completed = subprocess.run(
                [str(self.spcat_bin), base_name],
                cwd=task_path,
                check=True,
                timeout=timeout,
                capture_output=True,
            )
            generated = task_path / f"{base_name}.cat"
            if not generated.is_file():
                raise RuntimeError("SPCAT exited without producing a new catalog")
            self.parse_cat_file(generated)
            catalog = wd / generated.name
            shutil.copy2(generated, catalog)
            (wd / f"{base_name}.stdout").write_bytes(completed.stdout)
            (wd / f"{base_name}.stderr").write_bytes(completed.stderr)
            provenance = {
                "engine": "SPCAT",
                "executable": str(self.spcat_bin),
                "executable_sha256": executable_hash,
                "input_sha256": input_hashes,
                "catalog_sha256": hashlib.sha256(catalog.read_bytes()).hexdigest(),
                "returncode": completed.returncode,
                "parser_scope": "QN format 303; integer asymmetric-top J,Ka,Kc",
            }
            (wd / f"{base_name}.provenance.json").write_text(
                json.dumps(provenance, indent=2, allow_nan=False), encoding="utf-8"
            )
        return catalog

    def parse_cat_file(self, cat_path: Path | str) -> list[TransitionRecord]:
        """Parse QNFMT=303 records; reject malformed/unsupported rows with line number.

        This adapter handles integer J,Ka,Kc up to 99. Hyperfine and encoded quantum
        numbers need a separately validated parser. Intensity remains log10 SPCAT
        integrated intensity, not a linear intensity.
        """
        path = Path(cat_path).resolve()
        transitions: list[TransitionRecord] = []
        with path.open(encoding="ascii", errors="strict") as source:
            for line_number, raw in enumerate(source, 1):
                line = raw.rstrip("\r\n")
                if not line.strip():
                    continue
                try:
                    if len(line) < 79:
                        raise ValueError(
                            "truncated fixed-width record; expected 79 columns"
                        )
                    qn_format = int(line[51:55])
                    if qn_format != 303:
                        raise ValueError(
                            f"unsupported quantum-number format {qn_format}"
                        )
                    frequency = float(line[:13])
                    uncertainty = float(line[13:21])
                    log_intensity = float(line[21:29])
                    lower_energy = float(line[31:41])
                    if not all(
                        math.isfinite(v)
                        for v in (frequency, uncertainty, log_intensity, lower_energy)
                    ):
                        raise ValueError("nonfinite catalog observable")
                    if frequency <= 0 or uncertainty < 0 or lower_energy < 0:
                        raise ValueError(
                            "invalid frequency, uncertainty or lower-state energy"
                        )
                    if line[61:67].strip() or line[73:79].strip():
                        raise ValueError("unexpected additional quantum numbers")
                    upper = tuple(int(line[i : i + 2]) for i in (55, 57, 59))
                    lower = tuple(int(line[i : i + 2]) for i in (67, 69, 71))
                    for j, ka, kc in (upper, lower):
                        if not (
                            0 <= ka <= j <= 99
                            and 0 <= kc <= j
                            and ka + kc in (j, j + 1)
                        ):
                            raise ValueError("invalid asymmetric-top quantum numbers")
                    if abs(upper[0] - lower[0]) > 1 or upper[0] == lower[0] == 0:
                        raise ValueError(
                            "unsupported electric-dipole delta-J assignment"
                        )
                    parity = (
                        abs(upper[1] - lower[1]) % 2,
                        abs(upper[2] - lower[2]) % 2,
                    )
                    dipole_type = {(0, 1): "a", (1, 1): "b", (1, 0): "c"}.get(parity)
                    if dipole_type is None:
                        raise ValueError(
                            "unassigned dipole parity; no dipole type inferred"
                        )
                    transitions.append(
                        SPCATTransitionRecord(
                            freq_mhz=frequency,
                            intensity=log_intensity,
                            j_upper=upper[0],
                            ka_upper=upper[1],
                            kc_upper=upper[2],
                            j_lower=lower[0],
                            ka_lower=lower[1],
                            kc_lower=lower[2],
                            e_lower_cm1=lower_energy,
                            dipole_type=dipole_type,
                            uncertainty_mhz=uncertainty,
                            quantum_number_format=qn_format,
                        )
                    )
                except (ValueError, IndexError) as exc:
                    raise ValueError(f"{path.name}: line {line_number}: {exc}") from exc
        if not transitions:
            raise ValueError(f"SPCAT catalog contains no supported transitions: {path}")
        return transitions

    def generate_line_catalog(
        self,
        constants: RotationalConstants | None,
        output_parquet: Path | str,
        working_dir: Path | str | None = None,
        base_name: str = "spcat_run",
    ) -> Path:
        """Run explicitly supplied decks and export supported catalog records.

        Pass ``constants=None`` to select the externally supplied decks. The
        legacy constants-only generation path is unavailable; two contradictory
        sources of Hamiltonian parameters cannot be silently accepted.
        """
        if constants is not None:
            raise ValueError(
                "Pass constants=None to use explicit SPCAT decks; automatic "
                "constants-to-deck generation is unvalidated"
            )
        wd = Path(working_dir) if working_dir else Path(output_parquet).parent
        catalog = self.run_spcat(base_name, wd)
        records = self.parse_cat_file(catalog)
        table = pa.Table.from_pylist([record.__dict__ for record in records])
        metadata = {
            b"engine": b"SPCAT",
            b"intensity_convention": b"log10 integrated intensity, nm^2 MHz",
            b"catalog_sha256": hashlib.sha256(catalog.read_bytes())
            .hexdigest()
            .encode(),
            b"provenance": (wd / f"{base_name}.provenance.json").read_bytes(),
        }
        pq.write_table(table.replace_schema_metadata(metadata), output_parquet)
        return Path(output_parquet)
