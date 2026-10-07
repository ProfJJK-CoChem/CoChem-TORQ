#!/usr/bin/env python3
"""CoChem-TORQ: Standalone ORCA ExtOpt Client Bridge for OET Inference Server.

Mandated by Method Matrix v4 Quick Start QS-1 (Step 2), Section 9B.4, Section 8A.2,
and Section 10.1-10.8 as the standalone client bridge communicating with the OET
server daemon via ORCA %method ProgExt parameters.

Contract Specifications (Method Matrix v4 §10.1-10.8):
- Invocation from ORCA:
    %method
      ProgExt "<PATH>/oet_client"
      Ext_Params "-b localhost:8888"
    end
    ORCA executes: `oet_client <basename>_EXT.extinp.tmp -b localhost:8888`
- Input Contract (§10.2): ORCA writes `<basename>_EXT.extinp.tmp` containing:
    Line 1: `<basename>_EXT.xyz` (standard XYZ in Angstroms)
    Line 2: Charge (integer; e.g. 0)
    Line 3: Multiplicity (integer >= 1; e.g. 1)
    Line 4: NCores (integer >= 1)
    Line 5: do_gradient (0 or 1)
    Line 6: (Optional) point charges file path
- Output Contract (§10.2): `<basename>_EXT.engrad` containing:
    Number of atoms
    Total energy in Eh (Hartree)
    Energy gradient in Eh/bohr (Hartree/bohr) (atom1_x, atom1_y, atom1_z, ...)
- Units & Sign Conventions (§10.3):
    Input coordinates: Angstrom (A)
    Output energy: Hartree (Eh) = E_eV / 27.211386245988
    Output gradient: Eh/bohr = (-Force_eV_per_A) * 0.529177210903 / 27.211386245988
    MANDATORY SIGN FLIP: Gradient = -Force (\\nabla E = -F). ASE/models return
    forces F; ORCA optimizers require the potential energy gradient.
- Persistent Daemon Architecture (§8A.2 & §9B.4):
    Communicates with `oet_server` on TCP socket (default 127.0.0.1:8888) to bypass
    the ~30s model reloading overhead per gradient call during GOAT search.
- Float32 Precision Guard (§9B.4 & §13.1 T1-30min):
    Float32 MLFF potentials operate with a ~4 x 10^-6 Eh noise floor. Pair with
    `! TightOpt` and `%scf TolE 1e-5 end` in ORCA to prevent numerical noise.
- Committee Uncertainty Quantification (§10.8):
    Supports receiving or computing normalized energy uncertainty sigma_E and max
    atomic force uncertainty U_F, writing an uncertainty marker file if exceeding
    eps_E / eps_F.
- Physical Mass Mandate: Dynamic atomic mass and property resolution via `mendeleev`.
- Scientific integrity: authentic socket IPC and explicit missing-engine errors.
  An unavailable daemon never selects an alternate model or generates results.
"""

from __future__ import annotations

import argparse
import datetime
import functools
import json
import logging
import os
import platform
import socket
import sys
import tempfile
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timezone
from pathlib import Path
from typing import Any, Final, Optional, Union

import mendeleev  # type: ignore[import-untyped]
import numpy as np

try:
    from cochem_base.exceptions import OETDaemonConnectionError
    from cochem_base.schemas import OETFallbackAlertManifest
except ImportError:
    from pydantic import BaseModel, ConfigDict, Field

    class OETFallbackAlertManifest(BaseModel):
        model_config = ConfigDict(frozen=True, extra="forbid")
        calculation_base: str
        timestamp: str = Field(
            default_factory=lambda: datetime.datetime.now(timezone.utc).isoformat()
        )
        trigger_event: str
        fallback_calculator: str
        provenance_tag: str = "[E]"
        host_telemetry: dict[str, Any]
        scratch_alert_file: str
        staged_artifact_file: str

    class OETDaemonConnectionError(RuntimeError):
        """Raised when communication with persistent OET server daemon fails."""

        pass


class OETDaemonUnavailableError(RuntimeError):
    """The configured OET daemon or scientific engine is unavailable."""

    pass


class AirGapViolationError(RuntimeError):
    """Raised when writing to static Ring 1 repository files is detected."""

    pass


def emit_fallback_alert(
    calculation_base: str,
    trigger_event: str,
    fallback_calculator: str = "PhysicalOETFallbackCalculator",
    requested_backend: str = "mace_off24m",
    socket_target: str = "127.0.0.1:8888",
    scratch_dir: Path | str | None = None,
    artifacts_dir: Path | str | None = None,
    host_telemetry: dict[str, Any] | None = None,
) -> OETFallbackAlertManifest:
    """Persist a compatibility alert; this does not enable an alternate engine."""
    clean_base = calculation_base
    if clean_base.endswith("_EXT"):
        clean_base = clean_base[:-4]

    # Resolve scratch (Ring 2)
    if scratch_dir is not None:
        s_dir = Path(scratch_dir).resolve()
    elif "COCHEM_SCRATCH" in os.environ:
        s_dir = Path(os.environ["COCHEM_SCRATCH"]).resolve()
    else:
        s_dir = Path(tempfile.gettempdir()) / "cochem_scratch"
    s_dir.mkdir(parents=True, exist_ok=True)

    # Resolve artifacts (Ring 3)
    if artifacts_dir is not None:
        a_dir = Path(artifacts_dir).resolve()
    elif "COCHEM_ARTIFACTS" in os.environ:
        a_dir = Path(os.environ["COCHEM_ARTIFACTS"]).resolve()
    else:
        a_dir = s_dir / "artifacts"
    alerts_dir = a_dir / "alerts"
    alerts_dir.mkdir(parents=True, exist_ok=True)

    # Prohibit writing to Ring 1 static repository paths
    cochem_root = os.environ.get("COCHEM_ROOT")
    if cochem_root:
        root_path = Path(cochem_root).resolve()
        if root_path in s_dir.parents or root_path == s_dir:
            raise AirGapViolationError(
                f"Prohibited write to Ring 1 repository root: {s_dir}"
            )
        if root_path in alerts_dir.parents or root_path == alerts_dir:
            raise AirGapViolationError(
                f"Prohibited write to Ring 1 repository root: {alerts_dir}"
            )

    scratch_alert_file = s_dir / f"{clean_base}_EXT.fallback_alert.json"
    staged_artifact_file = alerts_dir / f"{clean_base}_EXT.fallback_alert.json"
    uncertainty_marker_file = s_dir / f"{clean_base}_EXT.uncertainty_marker"

    now_utc = datetime.datetime.now(timezone.utc).isoformat()
    telemetry = host_telemetry or {
        "platform": platform.platform(),
        "python_version": sys.version,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "timestamp_utc": now_utc,
    }

    alert_payload: dict[str, Any] = {
        "timestamp_utc": now_utc,
        "event": "OET_DAEMON_FALLBACK_TRIGGERED",
        "requested_backend": requested_backend,
        "active_fallback": fallback_calculator,
        "provenance_tag": "[E]",
        "socket_target": socket_target,
        "reason": str(trigger_event),
        "investigator_action_required": True,
        "calculation_base": clean_base,
        "host_telemetry": telemetry,
    }

    manifest = OETFallbackAlertManifest(
        calculation_base=clean_base,
        timestamp=now_utc,
        trigger_event=str(trigger_event),
        fallback_calculator=fallback_calculator,
        provenance_tag="[E]",
        host_telemetry=telemetry,
        scratch_alert_file=str(scratch_alert_file),
        staged_artifact_file=str(staged_artifact_file),
    )

    # Atomic write to Ring 2 scratch via temporary UUID sidecar
    tmp_scratch = s_dir / f".tmp_{uuid.uuid4().hex}"
    tmp_scratch.write_text(json.dumps(alert_payload, indent=2), encoding="utf-8")
    os.replace(tmp_scratch, scratch_alert_file)

    # Atomic write to Ring 3 artifacts via temporary UUID sidecar
    tmp_artifact = alerts_dir / f".tmp_{uuid.uuid4().hex}"
    tmp_artifact.write_text(json.dumps(alert_payload, indent=2), encoding="utf-8")
    os.replace(tmp_artifact, staged_artifact_file)

    # Uncertainty marker file in scratch with provenance tag [E]
    tmp_marker = s_dir / f".tmp_{uuid.uuid4().hex}"
    tmp_marker.write_text(
        f"PROVENANCE_TAG: [E]\n"
        f"EVENT: OET_DAEMON_FALLBACK_TRIGGERED\n"
        f"TRIGGER_EVENT: {trigger_event}\n"
        f"CALCULATION_BASE: {clean_base}\n"
        f"FALLBACK_CALCULATOR: {fallback_calculator}\n"
        f"TIMESTAMP: {now_utc}\n",
        encoding="utf-8",
    )
    os.replace(tmp_marker, uncertainty_marker_file)

    return manifest


# Physical conversion constants (Method Matrix v4 §10.3 & NIST CODATA 2022)
BOHR_TO_ANGSTROM: Final[float] = 0.529177210903
ANGSTROM_TO_BOHR: Final[float] = 1.0 / BOHR_TO_ANGSTROM  # ~1.8897261246257708
HARTREE_TO_EV: Final[float] = 27.211386245988
EV_TO_HARTREE: Final[float] = 1.0 / HARTREE_TO_EV
EH_PER_EV: Final[float] = EV_TO_HARTREE
BOHR_PER_A: Final[float] = ANGSTROM_TO_BOHR
EV_PER_ANG_TO_EH_PER_BOHR: Final[float] = EH_PER_EV / BOHR_PER_A
HARTREE_TO_KCAL_MOL: Final[float] = 627.5094740631
HARTREE_TO_KJ_MOL: Final[float] = 2625.4996394799

# Logger setup
logger = logging.getLogger("cochem.torq.oet_client")


# =============================================================================
# 1. Dynamic Mendeleev Mass and Property Resolution (Mendeleev Mandate)
# =============================================================================


@functools.lru_cache(maxsize=128)
def get_element_atomic_mass(symbol: str) -> float:
    """Resolve a tabulated isotope mass; never return a mixture as one isotope."""
    from Libraries.cochem_isotopes import isotope_mass

    return isotope_mass(symbol)


@functools.lru_cache(maxsize=128)
def get_element_atomic_number(symbol: str) -> int:
    """Retrieve atomic number for an element symbol using Mendeleev."""
    clean_sym = symbol.strip().capitalize()
    try:
        elem = mendeleev.element(clean_sym)
        atomic_num = elem.atomic_number
        if atomic_num is None:
            raise ValueError(f"Mendeleev atomic number is None for '{clean_sym}'")
        return int(atomic_num)
    except Exception as err:
        raise ValueError(
            f"Failed to retrieve atomic number for '{symbol}' via Mendeleev: {err}"
        ) from err


@functools.lru_cache(maxsize=128)
def get_element_symbol(atomic_number: int) -> str:
    """Retrieve element symbol from atomic number using Mendeleev."""
    try:
        elem = mendeleev.element(int(atomic_number))
        sym = elem.symbol
        if sym is None:
            raise ValueError(f"Mendeleev symbol is None for Z={atomic_number}")
        return str(sym)
    except Exception as err:
        raise ValueError(
            f"Failed to get element symbol for Z={atomic_number}: {err}"
        ) from err


@functools.lru_cache(maxsize=128)
def get_element_covalent_radius(symbol: str) -> float:
    """Retrieve the specified tabulated Pyykko radius, failing if unavailable."""
    elem = mendeleev.element(symbol)
    radius = elem.covalent_radius_pyykko
    if radius is None or not np.isfinite(radius) or radius <= 0:
        raise ValueError(f"No tabulated Pyykko covalent radius for {symbol!r}.")
    return float(radius) / 100


@functools.lru_cache(maxsize=128)
def get_element_vdw_radius(symbol: str) -> float:
    """Retrieve an actual database van der Waals radius; no guessed value."""
    elem = mendeleev.element(symbol)
    radius = elem.vdw_radius
    if radius is None or not np.isfinite(radius) or radius <= 0:
        raise ValueError(f"No tabulated van der Waals radius for {symbol!r}.")
    return float(radius) / 100


# =============================================================================
# 2. Data Structures & Configuration Models
# =============================================================================


@dataclass(frozen=True)
class ExtInpData:
    """Parsed data from ORCA `<base>_EXT.extinp.tmp` file per Method Matrix §10.2."""

    xyz_file: Path
    charge: int
    multiplicity: int
    ncores: int
    dograd: bool
    pointcharges_file: Path | None = None


@dataclass(frozen=True)
class EngradResult:
    """Calculated energy and gradient results written to `<base>_EXT.engrad`."""

    num_atoms: int
    energy_eh: float
    gradient_eh_bohr: list[float]
    atom_symbols: list[str]
    coordinates_angstrom: list[tuple[float, float, float]]
    engrad_file: Path
    uncertainty_energy_eh: float | None = None
    uncertainty_force_max: float | None = None
    provenance_tag: str = "[M]"


@dataclass
class OETClientConfig:
    """Configuration options for OET client bridge communication."""

    server_host: str = "127.0.0.1"
    server_port: int = 8888
    timeout_seconds: float = 60.0
    retries: int = 3
    retry_delay_seconds: float = 0.5
    scf_tole: float = 1e-5
    allow_fallback: bool = False
    standalone: bool = False
    fallback_driver: str = "physical"
    device: str = "cpu"
    dtype: str = "float64"
    eps_energy: float | None = None
    eps_force: float | None = None
    uncertainty_marker_file: str | None = None
    verbose: bool = False


# =============================================================================
# 3. File Contract I/O & Formatting (§10.2)
# =============================================================================


def read_extinp(path: str | Path) -> ExtInpData:
    """Parse an ORCA `<base>_EXT.extinp.tmp` external input file.

    Parameters
    ----------
    path : Union[str, Path]
        Path to the `.extinp.tmp` file written by ORCA.

    Returns
    -------
    ExtInpData
        Parsed parameters (XYZ path, charge, multiplicity, ncores, dograd, etc.).
    """
    p = Path(path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"ORCA external input file does not exist: {p}")

    content = p.read_text(encoding="utf-8")
    clean_lines: list[str] = []
    for line in content.splitlines():
        no_comment = line.split("#")[0].strip()
        if no_comment:
            clean_lines.append(no_comment)

    if len(clean_lines) < 5:
        raise ValueError(
            f"Invalid ORCA extinp file {p}: expected at least 5 lines "
            f"(xyz, charge, mult, ncores, dograd), found {len(clean_lines)}"
        )

    xyz_str = clean_lines[0]
    xyz_path = Path(xyz_str)
    if not xyz_path.is_absolute():
        xyz_path = p.parent / xyz_str

    charge = int(clean_lines[1])
    mult = int(clean_lines[2])
    if mult < 1:
        raise ValueError(f"Multiplicity must be >= 1, got {mult}")

    ncores = int(clean_lines[3])
    if ncores < 1:
        raise ValueError(
            "NCores must be positive; the requested value is not replaced."
        )

    dograd_int = int(clean_lines[4])
    if dograd_int not in (0, 1):
        raise ValueError("do_gradient must be exactly 0 or 1.")
    dograd = bool(dograd_int)

    pcfile: Path | None = None
    if len(clean_lines) > 5:
        pc_str = clean_lines[5]
        pc_candidate = Path(pc_str)
        if not pc_candidate.is_absolute():
            pc_candidate = p.parent / pc_str
        if pc_candidate.is_file():
            pcfile = pc_candidate
        else:
            raise FileNotFoundError(
                f"Requested point-charge file is unavailable: {pc_candidate}"
            )

    return ExtInpData(
        xyz_file=xyz_path,
        charge=charge,
        multiplicity=mult,
        ncores=ncores,
        dograd=dograd,
        pointcharges_file=pcfile,
    )


def read_xyz(
    xyz_path: str | Path,
) -> tuple[list[str], list[tuple[float, float, float]]]:
    """Parse standard XYZ file into element symbols and Cartesian coordinates."""
    p = Path(xyz_path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"XYZ coordinate file not found: {p}")

    lines = p.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        raise ValueError(f"Empty XYZ file: {p}")

    try:
        num_atoms = int(lines[0].strip())
    except ValueError as err:
        raise ValueError(
            f"Invalid XYZ header in {p}: first line must be integer count: '{lines[0]}'"
        ) from err
    if num_atoms < 1 or len(lines) != num_atoms + 2:
        raise ValueError("Supply exactly one nonempty XYZ frame with its comment line.")

    symbols: list[str] = []
    coords: list[tuple[float, float, float]] = []

    atom_lines = lines[2 : 2 + num_atoms]
    if len(atom_lines) < num_atoms:
        raise ValueError(
            f"XYZ file {p} declares {num_atoms} atoms but has {len(atom_lines)} lines."
        )

    for idx, line in enumerate(atom_lines, 1):
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(
                f"Malformed coordinate line {idx} in {p}: '{line}' (expected sym x y z)"
            )
        sym = parts[0].strip().capitalize()
        # Validate symbol with Mendeleev
        get_element_atomic_number(sym)
        try:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError as err:
            raise ValueError(
                f"Non-numeric coordinates on line {idx} in {p}: '{line}'"
            ) from err
        if not np.isfinite([x, y, z]).all():
            raise ValueError(f"Nonfinite coordinate on line {idx} in {p}.")
        symbols.append(sym)
        coords.append((x, y, z))

    return symbols, coords


def write_xyz(
    xyz_path: str | Path,
    symbols: Sequence[str],
    coordinates: Sequence[tuple[float, float, float]],
    comment: str = "Generated by CoChem-TORQ oet_client",
) -> Path:
    """Write geometry to a standard XYZ file."""
    p = Path(xyz_path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    n_atoms = len(symbols)
    if len(coordinates) != n_atoms:
        raise ValueError(
            f"Symbol count ({n_atoms}) != coordinate count ({len(coordinates)})"
        )

    lines = [f"{n_atoms}", comment]
    for sym, (x, y, z) in zip(symbols, coordinates):
        lines.append(f"{sym:<3} {x:20.12f} {y:20.12f} {z:20.12f}")

    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def write_engrad(
    engrad_path: str | Path,
    num_atoms: int,
    energy_eh: float,
    gradient_eh_bohr: Sequence[float],
    dograd: bool = True,
) -> Path:
    """Write ORCA `<base>_EXT.engrad` file adhering to Section 10.2 format verbatim."""
    if not dograd:
        raise ValueError(
            "An engrad gradient block requires an actual derivative; "
            "energy-only data cannot supply zero placeholders."
        )
    if type(num_atoms) is not int or num_atoms < 1 or not np.isfinite(energy_eh):
        raise ValueError(
            "Engrad needs an actual nonempty atom count and finite energy."
        )
    if not np.isfinite(np.asarray(gradient_eh_bohr, dtype=float)).all():
        raise ValueError(
            "Nonfinite derivatives cannot be written as scientific results."
        )
    p = Path(engrad_path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "#",
        "# Number of atoms",
        "#",
        f"{num_atoms}",
        "#",
        "# The current total energy in Eh",
        "#",
        f"{energy_eh:20.12f}",
        "#",
        "# The current gradient in Eh/bohr: Atom1X, Atom1Y, Atom1Z, Atom2X, ...",
        "#",
    ]

    if dograd:
        grad_list = list(gradient_eh_bohr)
        expected_size = num_atoms * 3
        if len(grad_list) != expected_size:
            raise ValueError(
                f"Gradient size mismatch: expected {expected_size} components for "
                f"{num_atoms} atoms, got {len(grad_list)}"
            )
        for g_val in grad_list:
            lines.append(f"{g_val:20.12f}")

    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


# =============================================================================
# 4. Units & Sign Conversion Physics (§10.3)
# =============================================================================


def convert_ase_forces_to_orca_gradient(
    forces_ev_per_ang: Sequence[Sequence[float]] | np.ndarray,
) -> list[float]:
    """Convert external atomic forces (eV/Angstrom) to ORCA energy gradient (Eh/bohr).

    MANDATORY SIGN FLIP (Method Matrix v4 §10.3):
    \\nabla E = -F
    g_Eh_a0 = (-F_eV_per_A) * 0.529177210903 / 27.211386245988
    """
    forces_arr = np.asarray(forces_ev_per_ang, dtype=np.float64)
    grad_arr = (-forces_arr) * EV_PER_ANG_TO_EH_PER_BOHR
    return list(grad_arr.flatten())


def convert_orca_gradient_to_ase_forces(
    gradient_eh_bohr: Sequence[float],
) -> list[tuple[float, float, float]]:
    """Convert ORCA gradient (Eh/bohr) back to atomic forces (eV/Angstrom)."""
    grad_arr = np.asarray(gradient_eh_bohr, dtype=np.float64)
    forces_flat = (-grad_arr) / EV_PER_ANG_TO_EH_PER_BOHR
    n_atoms = len(forces_flat) // 3
    forces_reshaped = forces_flat.reshape((n_atoms, 3))
    return [(float(fx), float(fy), float(fz)) for fx, fy, fz in forces_reshaped]


# =============================================================================
# 5. Rejected historical standalone calculator
# =============================================================================


class PhysicalOETFallbackCalculator:
    """Compatibility name for an unavailable, unqualified historical calculator.

    The former atomic reference, pair parameters and charges were invented.
    No physical energy, gradient or uncertainty is produced by this class.
    Configure an actual model server with an explicit scientific recipe instead.
    """

    def __init__(self, eps_dispersion: float = 0.05, k_bond: float = 0.35) -> None:
        self.unavailable_reason = "No qualified standalone OET engine is configured."

    def calculate(self, symbols, coordinates, charge=0, multiplicity=1, dograd=True):
        raise OETDaemonUnavailableError(self.unavailable_reason)


class OETClient:
    """IPC client bridge connecting ORCA ExtOpt with persistent OET inference server.

    Complies with Method Matrix v4 Quick Start QS-1 Step 2, Section 9B.4,
    and Section 10.5-10.8.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8888,
        timeout: float = 60.0,
        retries: int = 3,
        retry_delay: float = 0.5,
        scf_tole: float = 1e-5,
        allow_fallback: bool = False,
        fail_on_fallback: bool = False,
        strict_provenance: bool = False,
        standalone: bool = False,
        socket_path: Optional[Union[str, Path]] = None,
        scratch_dir: Optional[Union[str, Path]] = None,
        artifacts_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.retries = max(1, retries)
        self.retry_delay = max(0.01, retry_delay)
        self.scf_tole = scf_tole
        self.allow_fallback = allow_fallback
        self.fail_on_fallback = fail_on_fallback
        self.strict_provenance = strict_provenance
        self.standalone = standalone
        self.socket_path = Path(socket_path) if socket_path else None
        self.scratch_dir = Path(scratch_dir) if scratch_dir else None
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else None
        self.last_manifest: Optional[OETFallbackAlertManifest] = None

    def format_orca_extopt_input(
        self,
        xyz_filename: str,
        pal: int = 8,
        tight_opt: bool = True,
        scf_tole: float = 1e-5,
        maxen: float = 12.0,
    ) -> str:
        """Generate standard ORCA ExtOpt input block with %method ProgExt parameters."""
        opt_keyword = "TightOpt" if tight_opt else "Opt"
        return f"""! GOAT-EXPLORE ExtOpt {opt_keyword} PAL{pal}
%method
  ProgExt "oet_client"
  Ext_Params "-b {self.host}:{self.port}"
end
%scf
  TolE {scf_tole}
end
%goat
  maxen {maxen:.1f}
  conftemp 298.15
  confdegen auto
end
* xyzfile 0 1 {xyz_filename}
"""

    def send_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send JSON payload to persistent OET server daemon and receive response."""
        req_bytes = (json.dumps(payload) + "\n").encode("utf-8")
        last_error: Exception | None = None

        if self.socket_path is not None:
            if not hasattr(socket, "AF_UNIX"):
                raise OETDaemonConnectionError(
                    f"AF_UNIX not supported on {sys.platform} "
                    f"for domain socket {self.socket_path}"
                )
            if not self.socket_path.exists():
                raise OETDaemonConnectionError(
                    f"Target domain socket does not exist: {self.socket_path}"
                )
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            try:
                sock.connect(str(self.socket_path))
                sock.sendall(req_bytes)

                chunks: list[bytes] = []
                while True:
                    chunk = sock.recv(65536)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    try:
                        raw_combined = b"".join(chunks).decode("utf-8").strip()
                        if raw_combined.endswith("}") or raw_combined.endswith("]"):
                            parsed = json.loads(raw_combined)
                            if isinstance(parsed, dict):
                                return parsed
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue

                raw_data = b"".join(chunks).decode("utf-8").strip()
                if not raw_data:
                    raise ConnectionResetError("Server closed connection without data.")
                resp = json.loads(raw_data)
                if isinstance(resp, dict):
                    return resp
                return {
                    "status": "ERROR",
                    "message": f"Unexpected response type: {type(resp)}",
                }
            except Exception as err:
                raise OETDaemonConnectionError(
                    f"Failed to communicate with domain socket "
                    f"{self.socket_path}: {err}"
                ) from err
            finally:
                try:
                    sock.close()
                except Exception:
                    pass

        for attempt in range(1, self.retries + 1):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            try:
                sock.connect((self.host, self.port))
                sock.sendall(req_bytes)

                chunks = []
                while True:
                    chunk = sock.recv(65536)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    try:
                        raw_combined = b"".join(chunks).decode("utf-8").strip()
                        if raw_combined.endswith("}") or raw_combined.endswith("]"):
                            parsed = json.loads(raw_combined)
                            if isinstance(parsed, dict):
                                return parsed
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue

                raw_data = b"".join(chunks).decode("utf-8").strip()
                if not raw_data:
                    raise ConnectionResetError("Server closed connection without data.")

                resp = json.loads(raw_data)
                if isinstance(resp, dict):
                    return resp
                return {
                    "status": "ERROR",
                    "message": f"Unexpected response type: {type(resp)}",
                }

            except (
                TimeoutError,
                ConnectionRefusedError,
                ConnectionResetError,
                OSError,
            ) as err:
                last_error = err
                logger.warning(
                    "OET socket connection attempt %d/%d to %s:%d failed: %s",
                    attempt,
                    self.retries,
                    self.host,
                    self.port,
                    err,
                )
                if attempt < self.retries:
                    time.sleep(self.retry_delay * (1.5 ** (attempt - 1)))
            finally:
                try:
                    sock.close()
                except Exception:
                    pass

        raise ConnectionRefusedError(
            f"Failed to connect to OET server at {self.host}:{self.port} "
            f"after {self.retries} attempts: {last_error}"
        )

    def calculate_remote(
        self,
        symbols: Sequence[str],
        coordinates: Sequence[tuple[float, float, float]],
        charge: int = 0,
        multiplicity: int = 1,
        ncores: int = 1,
        dograd: bool = True,
        xyz_file: Path | None = None,
        pointcharges_file: Path | None = None,
        calculation_base: str = "calculation",
    ) -> dict[str, Any]:
        """Use the configured OET engine; absence never selects another model."""
        clean_base = calculation_base
        if xyz_file is not None and clean_base == "calculation":
            clean_base = xyz_file.stem
        if clean_base.endswith("_EXT"):
            clean_base = clean_base[:-4]

        if self.standalone:
            raise OETDaemonUnavailableError(
                "Standalone OET execution is unavailable: configure an actual "
                "scientific engine. No generated physical fallback is permitted."
            )
        coordinates_array = np.asarray(coordinates, dtype=float)
        if (
            not symbols
            or coordinates_array.shape != (len(symbols), 3)
            or not np.isfinite(coordinates_array).all()
        ):
            raise ValueError("OET needs an actual nonempty finite [N,3] geometry.")
        if (
            type(charge) is not int
            or type(multiplicity) is not int
            or multiplicity < 1
            or type(ncores) is not int
            or ncores < 1
            or type(dograd) is not bool
        ):
            raise ValueError(
                "OET needs explicit integer state/resources "
                "and a boolean derivative flag."
            )

        payload: dict[str, Any] = {
            "command": "calculate",
            "xyz_file": str(xyz_file.resolve()) if xyz_file else None,
            "symbols": list(symbols),
            "coordinates": [list(c) for c in coordinates],
            "charge": int(charge),
            "mult": int(multiplicity),
            "multiplicity": int(multiplicity),
            "ncores": int(ncores),
            "dograd": int(dograd),
            "pcfile": str(pointcharges_file.resolve()) if pointcharges_file else None,
        }

        try:
            resp = self.send_request(payload)
            return self._normalize_server_response(
                resp, dograd=dograd, n_atoms=len(symbols)
            )
        except (ConnectionRefusedError, OETDaemonConnectionError, OSError) as conn_err:
            raise OETDaemonUnavailableError(
                f"The configured OET server at {self.host}:{self.port} is unavailable. "
                "No replacement engine, energy, gradient or uncertainty was supplied."
            ) from conn_err

    def _normalize_server_response(
        self,
        resp: dict[str, Any],
        dograd: bool = True,
        n_atoms: int = 1,
    ) -> dict[str, Any]:
        """Normalize response dictionary across different OET server versions."""
        if not isinstance(resp, dict):
            raise ValueError("OET response must be an explicit result object.")
        status = str(resp.get("status", "")).upper()
        if status not in ("OK", "SUCCESS"):
            raise RuntimeError(
                "OET server did not report an explicit successful calculation."
            )
        if "energy_Eh" in resp:
            raw_energy = resp["energy_Eh"]
        elif "energy_hartree" in resp:
            raw_energy = resp["energy_hartree"]
        elif "energy" in resp and resp.get("units", {}).get("energy") == "hartree":
            raw_energy = resp["energy"]
        else:
            raise ValueError(
                "OET response requires a real energy with explicit Hartree units."
            )
        if (
            isinstance(raw_energy, bool)
            or not isinstance(raw_energy, (int, float))
            or not np.isfinite(raw_energy)
        ):
            raise ValueError("OET energy must be an actual finite number.")
        gradient = None
        if "gradient_Eh_bohr" in resp:
            gradient = resp["gradient_Eh_bohr"]
        elif "gradients_hartree_bohr" in resp:
            gradient = resp["gradients_hartree_bohr"]
        elif (
            "gradient" in resp
            and resp.get("units", {}).get("gradient") == "hartree/bohr"
        ):
            gradient = resp["gradient"]
        elif "forces" in resp and resp.get("units", {}).get("forces") == "eV/angstrom":
            gradient = convert_ase_forces_to_orca_gradient(resp["forces"])
        elif dograd:
            raise ValueError(
                "The requested OET gradient is unavailable; zero is not a substitute."
            )
        if gradient is not None:
            gradient = np.asarray(gradient, dtype=float).reshape(-1)
            if gradient.shape != (n_atoms * 3,) or not np.isfinite(gradient).all():
                raise ValueError(
                    "OET gradient must contain three finite components per actual atom."
                )
            gradient = gradient.tolist()
        reported_atoms = resp.get("num_atoms", n_atoms)
        if type(reported_atoms) is not int or reported_atoms != n_atoms:
            raise ValueError(
                "OET response atom count disagrees with the actual request."
            )
        u_energy = (
            resp["uncertainty_energy_Eh"]
            if "uncertainty_energy_Eh" in resp
            else resp.get("sigma_E")
        )
        u_force = (
            resp["uncertainty_force_max"]
            if "uncertainty_force_max" in resp
            else resp.get("U_F")
        )
        for uncertainty in (u_energy, u_force):
            if uncertainty is not None and (
                isinstance(uncertainty, bool)
                or not isinstance(uncertainty, (float, int))
                or not np.isfinite(uncertainty)
                or uncertainty < 0
            ):
                raise ValueError(
                    "Provider uncertainty must be finite and nonnegative "
                    "when available."
                )
        return {
            "status": "OK",
            "energy_Eh": float(raw_energy),
            "gradient_Eh_bohr": gradient,
            "num_atoms": n_atoms,
            "uncertainty_energy_Eh": u_energy,
            "uncertainty_force_max": u_force,
            "fallback_active": False,
            "uncertainty_status": "unavailable"
            if u_energy is None and u_force is None
            else "provider_reported_uncalibrated",
            "response_provenance": resp.get("provenance"),
        }


# =============================================================================
# 7. Pipeline Execution Runner
# =============================================================================


def run_oet_client(
    extinp_path: str | Path,
    config: OETClientConfig | None = None,
    client: OETClient | None = None,
) -> EngradResult:
    """Execute complete ORCA ExtOpt calculation step via OET client bridge."""
    cfg = config or OETClientConfig()
    inp_data = read_extinp(extinp_path)

    symbols, coords = read_xyz(inp_data.xyz_file)
    num_atoms = len(symbols)

    active_client = client or OETClient(
        host=cfg.server_host,
        port=cfg.server_port,
        timeout=cfg.timeout_seconds,
        retries=cfg.retries,
        retry_delay=cfg.retry_delay_seconds,
        scf_tole=cfg.scf_tole,
        allow_fallback=cfg.allow_fallback,
        standalone=cfg.standalone,
    )

    resp = active_client.calculate_remote(
        symbols=symbols,
        coordinates=coords,
        charge=inp_data.charge,
        multiplicity=inp_data.multiplicity,
        ncores=inp_data.ncores,
        dograd=inp_data.dograd,
        xyz_file=inp_data.xyz_file,
        pointcharges_file=inp_data.pointcharges_file,
    )

    energy_eh = float(resp["energy_Eh"])
    gradient_eh_bohr = resp.get("gradient_Eh_bohr")
    if gradient_eh_bohr is None:
        raise ValueError("ORCA engrad export requires an actual calculated gradient.")
    gradient_eh_bohr = list(gradient_eh_bohr)
    u_energy = resp.get("uncertainty_energy_Eh")
    u_force = resp.get("uncertainty_force_max")

    if (
        cfg.eps_energy is not None
        and u_energy is not None
        and u_energy > cfg.eps_energy
    ):
        logger.warning(
            "Energy uncertainty %.6e Eh exceeds threshold eps_energy %.6e Eh",
            u_energy,
            cfg.eps_energy,
        )
        if cfg.uncertainty_marker_file:
            msg = f"UNCERTAINTY_EXCEEDED: sigma_E={u_energy} > {cfg.eps_energy}\n"
            Path(cfg.uncertainty_marker_file).write_text(msg, encoding="utf-8")

    if cfg.eps_force is not None and u_force is not None and u_force > cfg.eps_force:
        logger.warning(
            "Force uncertainty %.6e Eh/bohr exceeds threshold eps_force %.6e Eh/bohr",
            u_force,
            cfg.eps_force,
        )
        if cfg.uncertainty_marker_file:
            msg = f"UNCERTAINTY_EXCEEDED: U_F={u_force} > {cfg.eps_force}\n"
            with open(cfg.uncertainty_marker_file, "a", encoding="utf-8") as mf:
                mf.write(msg)

    extinp_p = Path(extinp_path).resolve()
    base_name = extinp_p.name
    if base_name.endswith(".extinp.tmp"):
        base_stem = base_name[: -len(".extinp.tmp")]
    elif base_name.endswith(".tmp"):
        base_stem = base_name[: -len(".tmp")]
    else:
        base_stem = extinp_p.stem

    engrad_file = extinp_p.parent / f"{base_stem}.engrad"
    write_engrad(
        engrad_path=engrad_file,
        num_atoms=num_atoms,
        energy_eh=energy_eh,
        gradient_eh_bohr=gradient_eh_bohr,
        dograd=inp_data.dograd,
    )

    return EngradResult(
        num_atoms=num_atoms,
        energy_eh=energy_eh,
        gradient_eh_bohr=gradient_eh_bohr,
        atom_symbols=symbols,
        coordinates_angstrom=coords,
        engrad_file=engrad_file,
        uncertainty_energy_eh=u_energy,
        uncertainty_force_max=u_force,
        provenance_tag="[M]",
    )


# =============================================================================
# 8. Command-Line Interface (CLI Entrypoint)
# =============================================================================


def build_argument_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser for ORCA ProgExt parameters."""
    parser = argparse.ArgumentParser(
        description="CoChem-TORQ Standalone ORCA ExtOpt Client Bridge (§10.1-10.8)"
    )
    parser.add_argument(
        "extinp",
        nargs="?",
        default=None,
        help="Path to ORCA external input file (<basename>_EXT.extinp.tmp)",
    )
    parser.add_argument(
        "-b",
        "--bind",
        "--server-address",
        default="127.0.0.1:8888",
        help="OET server address 'host:port' (default: '127.0.0.1:8888')",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="OET server host address (default: '127.0.0.1')",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8888,
        help="OET server port number (default: 8888)",
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=float,
        default=60.0,
        help="Socket timeout in seconds (default: 60.0)",
    )
    parser.add_argument(
        "-r",
        "--retries",
        type=int,
        default=3,
        help="Socket connection retry attempts (default: 3)",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=0.5,
        help="Delay between retry attempts in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--scf-tole",
        type=float,
        default=1e-5,
        help="Energy convergence tolerance threshold (default: 1e-5)",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Compatibility flag: unavailable engines always fail explicitly",
    )
    parser.add_argument(
        "--standalone",
        action="store_true",
        help="Historical unsupported mode; returns an explicit unavailable error",
    )
    parser.add_argument(
        "-m",
        "--model",
        "--driver",
        default=None,
        help="Historical driver metadata; never selects a replacement engine",
    )
    parser.add_argument(
        "-d",
        "--device",
        default="cpu",
        help="Compute device ('cpu', 'cuda', default: 'cpu')",
    )
    parser.add_argument(
        "--dtype",
        default="float64",
        choices=["float32", "float64"],
        help="Floating point precision ('float32', 'float64')",
    )
    parser.add_argument(
        "--eps-e",
        type=float,
        default=None,
        help="Energy uncertainty threshold sigma_E in Hartree (Eh) (§10.8)",
    )
    parser.add_argument(
        "--eps-f",
        type=float,
        default=None,
        help="Force uncertainty threshold U_F in Hartree/bohr (Eh/bohr) (§10.8)",
    )
    parser.add_argument(
        "--marker-file",
        default=None,
        help="Path to write uncertainty marker file if thresholds are exceeded",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging output",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="CoChem-TORQ oet_client v4.0.0 (explicit-engine bridge)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI execution entrypoint invoked by ORCA or standalone user."""
    parser = build_argument_parser()
    args, _unknown = parser.parse_known_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    )

    if not args.extinp:
        parser.print_help(sys.stderr)
        return 1

    extinp_path = Path(args.extinp)
    if not extinp_path.is_file():
        sys.stderr.write(f"Error: Input file does not exist: {extinp_path}\n")
        return 1

    server_host = args.host
    server_port = args.port
    if args.bind:
        if ":" in args.bind:
            parts = args.bind.split(":", 1)
            server_host = parts[0]
            try:
                server_port = int(parts[1])
            except ValueError:
                sys.stderr.write(f"Error: Invalid port in --bind '{args.bind}'\n")
                return 1
        else:
            server_host = args.bind

    config = OETClientConfig(
        server_host=server_host,
        server_port=server_port,
        timeout_seconds=args.timeout,
        retries=1 if args.standalone else args.retries,
        retry_delay_seconds=args.retry_delay,
        scf_tole=args.scf_tole,
        allow_fallback=False,
        standalone=args.standalone,
        fallback_driver=args.model,
        device=args.device,
        dtype=args.dtype,
        eps_energy=args.eps_e,
        eps_force=args.eps_f,
        uncertainty_marker_file=args.marker_file,
        verbose=args.verbose,
    )

    try:
        result = run_oet_client(extinp_path=extinp_path, config=config)
        logger.info(
            "Successfully evaluated %d atoms: Energy = %.10f Eh, Output = %s",
            result.num_atoms,
            result.energy_eh,
            result.engrad_file.name,
        )
        return 0
    except Exception as err:
        sys.stderr.write(f"OET Client Critical Error: {err}\n")
        if args.verbose:
            import traceback

            traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
