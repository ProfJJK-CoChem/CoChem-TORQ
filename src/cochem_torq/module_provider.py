"""Canonical, bounded scientific operations for installed BASE consumers.

The module owns rigid-fragment scans, Löwdin-AO and general-element PySCF NAO
Wiberg bond-order analyses, and authentic NBO 7 (ORCA %nbo / standalone GenNBO)
population, Wiberg, and E(2) donor-acceptor analyses. BASE owns installation,
execution containment and UI.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .chem490_vpt2_bridge import (
    export_spyfy_geom_handoff as export_spyfy_geom_handoff,
)
from .chem490_vpt2_bridge import (
    parse_orca_vpt2_rotational_constants as parse_orca_vpt2_rotational_constants,
)
from .chem490_vpt2_bridge import (
    run_relaxed_1d_torsional_scan as run_relaxed_1d_torsional_scan,
)
from .ecosystem import ATOMIC_NUMBERS

_SYMBOL = re.compile(
    r"(?:[1-9][0-9]{0,2}[A-Z][a-z]?|[A-Z][a-z]?(?:-?[1-9][0-9]{0,2})?)\Z"
)
_MAX_ATOMS = 2000
_MAX_INPUT_BYTES = 2_000_000
_OPERATIONS = (
    "geometry_analysis",
    "research_scan",
    "wiberg_lowdin",
    "nbo_analysis",
    "wiberg_nao",
)

_NAO_PROFILE_ACTIVATED = True
_NAO_PROFILE = "cochem.torq.pyscf-general-nao-wiberg/2"
_NAO_DEFAULT_METHOD = {"name": "b3lyp", "basis": "def2-svp"}
_NAO_SUPPORTED_METHODS = {"hf", "rhf", "pbe", "pbe-d4", "b3lyp", "b3lyp-d4"}
_NBO_DEFAULT_METHOD = {"name": "b3lyp", "basis": "def2-svp"}
_NBO_DEFAULT_KEYWORDS = "$NBO BNDIDX ARCHIVE $END"
_NAO_CONTROLLED_EVIDENCE = {
    "id": "torq-nao-general-20261125",
    "phase": "general_element_all_electron_pyscf_nao",
    "actual_cases": ["H2", "H2O", "HF", "N2"],
}


def _provider_version() -> str:
    try:
        return importlib.metadata.version("CoChem-TORQ")
    except importlib.metadata.PackageNotFoundError:
        return "0.1.0"


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _pure_element_symbol(symbol: str) -> str:
    normalized = {"D": "2H", "T": "3H"}.get(symbol, symbol)
    match = re.fullmatch(r"(?:[1-9][0-9]{0,2})?([A-Z][a-z]?)", normalized)
    if match is None or match.group(1) not in ATOMIC_NUMBERS:
        raise ValueError(f"Unsupported element symbol {symbol!r}.")
    return match.group(1)


def _normalize_nao_method(method: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
    """Validate and normalize a PySCF restricted SCF/DFT method for NAO Wiberg."""
    if not isinstance(method, dict) or set(method) != {"name", "basis"}:
        raise ValueError(
            "NAO analysis method must be a mapping with exact keys 'name' and 'basis'."
        )
    raw_name = method.get("name")
    raw_basis = method.get("basis")
    if not isinstance(raw_name, str) or not raw_name.strip():
        raise ValueError("NAO analysis requires an explicit method name.")
    if (
        not isinstance(raw_basis, str)
        or not raw_basis.strip()
        or re.fullmatch(r"[A-Za-z0-9+*(),_. -]{1,100}", raw_basis.strip()) is None
    ):
        raise ValueError("NAO analysis requires a valid explicit basis set name.")

    lower_name = raw_name.strip().lower()
    if lower_name == "mp2" or lower_name == "rmp2":
        raise ValueError(
            "A correlated MP2 density is not available in the retained "
            "restricted SCF checkpoint for NAO analysis."
        )
    if lower_name not in _NAO_SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported PySCF NAO method {raw_name!r}; supported: "
            f"{sorted(_NAO_SUPPORTED_METHODS)}."
        )
    basis_clean = raw_basis.strip()
    dispersion: str | None = None
    backend_name = lower_name
    if backend_name.endswith("-d4"):
        backend_name = backend_name[:-3]
        dispersion = "d4"
    if backend_name == "rhf":
        backend_name = "hf"
    expected_native = {
        "name": backend_name,
        "basis": basis_clean,
        "reference": "restricted",
        "dispersion": dispersion,
        "frozen_core": False,
        "density_fitting": False,
    }
    return {"name": lower_name, "basis": basis_clean}, expected_native


def _nao_algorithm_identity(
    symbols: Sequence[str] | None = None,
    basis: str | None = None,
) -> dict[str, Any]:
    """Bind the live PySCF natural-AO implementation and its AOSHELL configuration."""
    import pyscf
    from pyscf import gto
    from pyscf.data import elements
    from pyscf.lo import nao, orth

    root = Path(pyscf.__file__).resolve(strict=True).parent
    basis_directory = (root / "gto/basis").resolve(strict=True)
    if Path(gto.basis._BASIS_DIR).resolve(strict=True) != basis_directory:
        raise ValueError(
            "The PySCF basis loader no longer selects its pinned basis file directory."
        )
    basis_key = re.sub(r"[^a-z0-9]", "", (basis or "sto-3g").lower())
    if basis is None or basis_key == "sto3g":
        if gto.basis.ALIAS.get("sto3g") != "sto-3g.dat":
            raise ValueError(
                "The requested STO-3G loader no longer selects its pinned basis file."
            )

    sources: dict[str, dict[str, str]] = {}
    for relative, module in (
        ("lo/nao.py", nao),
        ("lo/orth.py", orth),
        ("data/elements.py", elements),
    ):
        source = Path(module.__file__).resolve(strict=True)
        if source != root / relative:
            raise ValueError("The PySCF NAO algorithm module path has changed.")
        sources[relative] = {
            "path": str(source),
            "sha256": _sha256_file(source),
        }
    sto3g_path = (root / "gto/basis/sto-3g.dat").resolve(strict=True)
    if basis is None or basis_key == "sto3g":
        sources["gto/basis/sto-3g.dat"] = {
            "path": str(sto3g_path),
            "sha256": _sha256_file(sto3g_path),
        }

    if symbols is None:
        atomic_numbers = [1, 8]
    else:
        atomic_numbers = sorted(
            {ATOMIC_NUMBERS[_pure_element_symbol(sym)] for sym in symbols}
        )
    configuration: dict[str, list[str]] = {}
    for number in atomic_numbers:
        try:
            actual = tuple(nao.AOSHELL[number])
            default = (
                elements.N_CORE_SHELLS[number],
                elements.N_CORE_VALENCE_SHELLS[number],
            )
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            raise ValueError(
                f"Default AOSHELL configuration is unavailable for Z={number}."
            ) from exc
        if actual != default:
            raise ValueError(
                "This NAO profile requires the unmodified default AOSHELL "
                f"configuration (mismatch at Z={number})."
            )
        configuration[str(number)] = list(actual)

    alias_name = gto.basis.ALIAS.get(basis_key)
    return {
        "profile": _NAO_PROFILE,
        "implementation": "pyscf.lo.nao.nao",
        "engine": "pyscf",
        "engine_version": pyscf.__version__,
        "platform": f"{platform.system().lower()}-cpu",
        "source_files": sources,
        "AOSHELL": configuration,
        "restore": True,
        "basis_loader": {
            "normalized_name": basis_key,
            "alias": alias_name,
            "directory": str(basis_directory),
        },
    }


def _nao_options(options: dict[str, Any], symbols: list[str]) -> dict[str, Any]:
    """Validate explicit general-element PySCF NAO Wiberg analysis options."""
    allowed = {"engine", "method", "charge", "multiplicity", "cores", "memory_mb"}
    if not isinstance(options, dict) or set(options) - allowed:
        raise ValueError(
            "NAO analysis accepts only explicit engine, method, charge, "
            "multiplicity, cores, and memory_mb options."
        )
    if "engine" not in options or options["engine"] != "pyscf":
        raise ValueError(
            "NAO analysis requires an explicit 'engine': 'pyscf' backend option."
        )
    raw_method = options.get("method", dict(_NAO_DEFAULT_METHOD))
    normalized_method, _ = _normalize_nao_method(raw_method)

    charge = options.get("charge", 0)
    multiplicity = options.get("multiplicity", 1)
    if (
        type(charge) is not int
        or abs(charge) > 4
        or type(multiplicity) is not int
        or multiplicity != 1
    ):
        raise ValueError(
            "NAO analysis requires an integer charge (|q| <= 4) and "
            "closed-shell singlet multiplicity (1)."
        )
    if not 2 <= len(symbols) <= 50:
        raise ValueError("NAO analysis supports molecules with 2 to 50 atoms.")
    pure_symbols = [_pure_element_symbol(sym) for sym in symbols]
    total_electrons = sum(ATOMIC_NUMBERS[sym] for sym in pure_symbols) - charge
    if total_electrons <= 0 or total_electrons % 2 != 0:
        raise ValueError(
            "Requested closed-shell singlet NAO state has incompatible electron "
            f"count ({total_electrons} electrons for charge={charge})."
        )
    cores = options.get("cores", 1)
    memory_mb = options.get("memory_mb", 1024)
    if (
        type(cores) is not int
        or not 1 <= cores <= 8
        or type(memory_mb) is not int
        or not 256 <= memory_mb <= 16384
    ):
        raise ValueError(
            "NAO analysis permits 1-8 cores and 256-16384 MiB total memory."
        )
    return {
        "engine": "pyscf",
        "method": normalized_method,
        "charge": charge,
        "multiplicity": multiplicity,
        "cores": cores,
        "memory_mb": memory_mb,
    }


def _is_executable_file(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def _resolve_executable(
    *,
    explicit: str | Path | None = None,
    env_vars: Sequence[str] = (),
    dir_env_pairs: Sequence[tuple[str, Sequence[str]]] = (),
    path_names: Sequence[str] = (),
    fallback_paths: Sequence[Path] = (),
    env: Mapping[str, str] | None = None,
    search_defaults: bool = True,
) -> Path | None:
    environ = os.environ if env is None else env
    if explicit is not None:
        candidate = Path(explicit).expanduser()
        if not _is_executable_file(candidate):
            raise FileNotFoundError(
                f"Explicit executable is absent or not executable: {candidate}"
            )
        return candidate.resolve()

    candidates: list[Path] = []
    for var_name in env_vars:
        val = environ.get(var_name, "").strip()
        if val:
            candidates.append(Path(val).expanduser())
    for dir_var, names in dir_env_pairs:
        dir_val = environ.get(dir_var, "").strip()
        if dir_val:
            base = Path(dir_val).expanduser()
            for name in names:
                candidates.append(base / name)
    path_env = environ.get("PATH", "")
    for name in path_names:
        found = shutil.which(name, path=path_env)
        if found:
            candidates.append(Path(found))
    if search_defaults:
        candidates.extend(fallback_paths)

    for candidate in candidates:
        if _is_executable_file(candidate):
            return candidate.resolve()
    return None


def probe_nbo7(
    *,
    nbo_exe: str | Path | None = None,
    orca_bin: str | Path | None = None,
    gennbo_exe: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    search_defaults: bool = True,
) -> dict[str, Any]:
    """Dynamically probe authentic installed NBO 7, GenNBO, and ORCA binaries."""
    home = Path.home()
    default_nbo_dir = home / "CoChem_Workstation_Direct" / "engines" / "nbo7" / "bin"
    default_orca = (
        home
        / ".local"
        / "share"
        / "cochem-topos"
        / "workstation-20261008"
        / "engines"
        / "orca-native-v1"
        / "orca_6_1_1_linux_x86-64_shared_openmpi418"
        / "orca"
    )

    try:
        resolved_nbo = _resolve_executable(
            explicit=nbo_exe,
            env_vars=("NBOEXE", "COCHEM_NBO7_BIN"),
            dir_env_pairs=(("NBOBIN", ("nbo7.i8.exe", "nbo7.i4.exe")),),
            path_names=("nbo7.i8.exe", "nbo7.i4.exe"),
            fallback_paths=(
                default_nbo_dir / "nbo7.i8.exe",
                default_nbo_dir / "nbo7.i4.exe",
            ),
            env=env,
            search_defaults=search_defaults,
        )
    except FileNotFoundError as exc:
        return {"available": False, "engine": "NBO7", "reason": str(exc)}

    try:
        resolved_orca = _resolve_executable(
            explicit=orca_bin,
            env_vars=("COCHEM_ORCA_BIN", "ORCA_CMD", "ORCA_BIN"),
            dir_env_pairs=(("ORCA_DIR", ("orca",)), ("ORCA_PATH", ("orca",))),
            path_names=("orca",),
            fallback_paths=(default_orca,),
            env=env,
            search_defaults=search_defaults,
        )
    except FileNotFoundError as exc:
        return {"available": False, "engine": "NBO7", "reason": str(exc)}

    # Prefer integer width matching resolved NBOEXE when selecting gennbo
    gennbo_names: tuple[str, ...] = ("gennbo.i8.exe", "gennbo.i4.exe")
    if resolved_nbo is not None and resolved_nbo.name.endswith(".i4.exe"):
        gennbo_names = ("gennbo.i4.exe", "gennbo.i8.exe")

    gennbo_fallbacks: list[Path] = []
    if resolved_nbo is not None:
        for name in gennbo_names:
            gennbo_fallbacks.append(resolved_nbo.parent / name)
    if search_defaults:
        for name in gennbo_names:
            gennbo_fallbacks.append(default_nbo_dir / name)

    try:
        resolved_gennbo = _resolve_executable(
            explicit=gennbo_exe,
            env_vars=(
                ()
                if (
                    resolved_nbo is not None
                    and resolved_nbo.name.endswith(".i8.exe")
                    and (resolved_nbo.parent / "gennbo.i8.exe").is_file()
                )
                else ("COCHEM_GENNBO_BIN", "GENNBO_EXE")
            ),
            dir_env_pairs=(("NBOBIN", gennbo_names),),
            path_names=gennbo_names,
            fallback_paths=tuple(gennbo_fallbacks),
            env=env,
            search_defaults=search_defaults or (resolved_nbo is not None),
        )
    except FileNotFoundError as exc:
        return {"available": False, "engine": "NBO7", "reason": str(exc)}

    modes: list[str] = []
    if resolved_nbo is not None and resolved_orca is not None:
        modes.append("orca_nbo")
    if resolved_nbo is not None and resolved_gennbo is not None:
        modes.append("gennbo_47")

    if resolved_nbo is None:
        return {
            "available": False,
            "engine": "NBO7",
            "modes_available": [],
            "reason": (
                "Native NBO 7 binary (nbo7.i8.exe / nbo7.i4.exe) is not "
                "installed or not executable on this host."
            ),
        }
    if not modes:
        return {
            "available": False,
            "engine": "NBO7",
            "nbo_exe": str(resolved_nbo),
            "modes_available": [],
            "reason": (
                "NBO 7 binary is present, but neither ORCA (for %nbo) nor "
                "standalone GenNBO (gennbo.i8.exe / gennbo.i4.exe) is available."
            ),
        }

    receipt: dict[str, Any] = {
        "available": True,
        "engine": "NBO7",
        "modes_available": modes,
        "nbo_exe": str(resolved_nbo),
        "nbo_exe_sha256": _sha256_file(resolved_nbo),
    }
    if resolved_orca is not None:
        receipt["orca_bin"] = str(resolved_orca)
        receipt["orca_bin_sha256"] = _sha256_file(resolved_orca)
    if resolved_gennbo is not None:
        receipt["gennbo_exe"] = str(resolved_gennbo)
        receipt["gennbo_exe_sha256"] = _sha256_file(resolved_gennbo)
    return receipt


def parse_nbo7_output(output_text: str) -> dict[str, Any]:
    """Verify authentic NBO 7.x banner and parse NPA, Wiberg indices, and E(2)."""
    cite_match = re.search(r"\[NBO\s+(7\.\d+(?:\.\d+)?)\b", output_text)
    banner_match = re.search(r"\*{3,}\s*NBO\s+(7\.\d+(?:\.\d+)?)\s*\*{3,}", output_text)
    if cite_match is not None:
        nbo_version = cite_match.group(1)
    elif banner_match is not None:
        nbo_version = banner_match.group(1)
    else:
        raise RuntimeError(
            "Authentic NBO 7.x banner was not found in calculation output."
        )

    lines = output_text.splitlines()

    # 1. Natural Population Analysis (NPA)
    npa_charges: list[dict[str, Any]] = []
    npa_total: dict[str, float] | None = None
    npa_row_re = re.compile(
        r"^\s*([A-Z][a-z]?)\s+(\d+)\s+"
        r"([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+"
        r"([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s*$"
    )
    npa_tot_re = re.compile(
        r"^\s*\*\s*Total\s*\*\s+"
        r"([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+"
        r"([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s*$"
    )
    in_npa = False
    for line in lines:
        if "Summary of Natural Population Analysis:" in line:
            in_npa = True
            npa_charges = []
            continue
        if not in_npa:
            continue
        row_m = npa_row_re.match(line)
        if row_m is not None:
            atom_no = int(row_m.group(2))
            npa_charges.append(
                {
                    "atom_index": atom_no - 1,
                    "atom_number_1based": atom_no,
                    "symbol": row_m.group(1),
                    "natural_charge": float(row_m.group(3)),
                    "core": float(row_m.group(4)),
                    "valence": float(row_m.group(5)),
                    "rydberg": float(row_m.group(6)),
                    "total": float(row_m.group(7)),
                }
            )
            continue
        tot_m = npa_tot_re.match(line)
        if tot_m is not None:
            npa_total = {
                "charge": float(tot_m.group(1)),
                "core": float(tot_m.group(2)),
                "valence": float(tot_m.group(3)),
                "rydberg": float(tot_m.group(4)),
                "total": float(tot_m.group(5)),
            }
            in_npa = False

    if not npa_charges:
        raise RuntimeError(
            "Failed to parse Summary of Natural Population Analysis from NBO 7 output."
        )

    # 2. Wiberg bond index matrix in the NAO basis (supports multi-block >9 atoms)
    n_atoms = len(npa_charges)
    wiberg_matrix: list[list[float]] = [
        [0.0 for _ in range(n_atoms)] for _ in range(n_atoms)
    ]
    wiberg_found = False
    in_wiberg = False
    col_indices: list[int] = []
    wiberg_row_re = re.compile(r"^\s*(\d+)\.\s+([A-Z][a-z]?)\s+(.*)$")
    for line in lines:
        if "Wiberg bond index matrix in the NAO basis:" in line:
            in_wiberg = True
            wiberg_found = True
            col_indices = []
            continue
        if not in_wiberg:
            continue
        stripped = line.strip()
        if stripped.startswith("Wiberg bond index, Totals by atom"):
            in_wiberg = False
            continue
        if stripped.startswith("Atom "):
            col_indices = [int(tok) - 1 for tok in stripped.split()[1:]]
            continue
        row_m = wiberg_row_re.match(line)
        if row_m is not None and col_indices:
            row_idx = int(row_m.group(1)) - 1
            vals = [float(tok) for tok in row_m.group(3).split()]
            if 0 <= row_idx < n_atoms:
                for col_idx, val in zip(col_indices, vals, strict=False):
                    if 0 <= col_idx < n_atoms:
                        wiberg_matrix[row_idx][col_idx] = val

    wiberg_bonds: list[dict[str, Any]] = []
    if wiberg_found:
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                val = 0.5 * (wiberg_matrix[i][j] + wiberg_matrix[j][i])
                wiberg_bonds.append(
                    {
                        "atom_i": i,
                        "atom_j": j,
                        "symbol_i": npa_charges[i]["symbol"],
                        "symbol_j": npa_charges[j]["symbol"],
                        "value": val,
                    }
                )

    # 3. Second-Order Perturbation Theory E(2) donor-acceptor analysis
    e2_interactions: list[dict[str, Any]] = []
    e2_row_re = re.compile(
        r"^\s*(\d+\.\s+\S.*?)\s+(\d+\.\s+\S.*?)\s+"
        r"([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s*$"
    )
    in_e2 = False
    unit_context = ""
    for line in lines:
        if "SECOND ORDER PERTURBATION THEORY ANALYSIS OF FOCK MATRIX IN NBO BASIS" in line:
            in_e2 = True
            e2_interactions = []
            unit_context = ""
            continue
        if not in_e2:
            continue
        stripped = line.strip()
        if stripped.startswith("NATURAL BOND ORBITALS (Summary)"):
            in_e2 = False
            continue
        if stripped.startswith(("within unit", "from unit")):
            unit_context = stripped
            continue
        e2_m = e2_row_re.match(line)
        if e2_m is not None:
            e2_interactions.append(
                {
                    "donor": " ".join(e2_m.group(1).split()),
                    "acceptor": " ".join(e2_m.group(2).split()),
                    "e2_kcal_mol": float(e2_m.group(3)),
                    "energy_diff_au": float(e2_m.group(4)),
                    "fock_element_au": float(e2_m.group(5)),
                    "unit_context": unit_context,
                }
            )

    return {
        "nbo_version": nbo_version,
        "nbo_banner_verified": True,
        "npa_charges": npa_charges,
        "npa_total": npa_total,
        "wiberg_matrix_present": wiberg_found,
        "wiberg_matrix": wiberg_matrix if wiberg_found else None,
        "wiberg_bonds": wiberg_bonds,
        "e2_interactions": e2_interactions,
    }


def _orca_method_keywords(method: Mapping[str, Any]) -> str:
    """Format validated method/basis into authentic ORCA 6 simple-input keywords."""
    raw_name = str(method.get("name", "")).strip()
    raw_basis = str(method.get("basis", "")).strip()
    if not raw_name or not raw_basis:
        raise ValueError("NBO 7 ORCA mode requires explicit method 'name' and 'basis'.")
    if re.fullmatch(r"[A-Za-z0-9+*(),_./ -]{1,100}", raw_name) is None:
        raise ValueError(f"Invalid ORCA method name {raw_name!r}.")
    if re.fullmatch(r"[A-Za-z0-9+*(),_./ -]{1,100}", raw_basis) is None:
        raise ValueError(f"Invalid ORCA basis set {raw_basis!r}.")
    lower = raw_name.lower()
    if lower in {"hf", "rhf"}:
        method_kw = "HF"
    elif lower == "uhf":
        method_kw = "UHF"
    elif lower.endswith("-d4"):
        method_kw = f"{raw_name[:-3].upper()} D4"
    else:
        method_kw = raw_name.upper()
    return f"! {method_kw} {raw_basis} NBO"


def _ensure_47_bndidx(text_47: str) -> str:
    """Ensure BNDIDX is requested in the $NBO keylist of a working .47 copy."""
    pattern = re.compile(r"(\$NBO\b)(.*?)(\$END)", re.IGNORECASE | re.DOTALL)
    match = pattern.search(text_47)
    if match is None:
        return text_47
    body = match.group(2)
    if re.search(r"\bBNDIDX\b", body, re.IGNORECASE):
        return text_47
    return (
        text_47[: match.start()]
        + f"{match.group(1)} BNDIDX {body.strip()} {match.group(3)}"
        + text_47[match.end() :]
    )


def run_nbo7_analysis(
    *,
    symbols: Sequence[str] | None = None,
    coordinates_angstrom: Sequence[Sequence[float]] | None = None,
    xyz_path: str | Path | None = None,
    archive_47_path: str | Path | None = None,
    method: Mapping[str, Any] | None = None,
    charge: int = 0,
    multiplicity: int = 1,
    cores: int = 1,
    memory_mb: int = 1024,
    nbo_keywords: str = _NBO_DEFAULT_KEYWORDS,
    ensure_bndidx: bool = True,
    work_dir: str | Path | None = None,
    timeout_seconds: float = 300.0,
    nbo_exe: str | Path | None = None,
    orca_bin: str | Path | None = None,
    gennbo_exe: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    search_defaults: bool = True,
) -> dict[str, Any]:
    """Execute authentic NBO 7 analysis via ORCA ``%nbo`` or standalone ``gennbo``.

    Fails closed with ``FileNotFoundError`` if native binaries are missing and
    ``RuntimeError`` if the engine exits non-zero or lacks an authentic ``NBO 7.``
    banner. Never substitutes another population analysis.
    """
    probe = probe_nbo7(
        nbo_exe=nbo_exe,
        orca_bin=orca_bin,
        gennbo_exe=gennbo_exe,
        env=env,
        search_defaults=search_defaults,
    )
    if not probe.get("available"):
        raise FileNotFoundError(
            probe.get("reason", "Native NBO 7 runtime is unavailable.")
        )

    if type(charge) is not int or type(multiplicity) is not int or multiplicity < 1:
        raise ValueError("charge must be an int and multiplicity must be an int >= 1.")
    if type(cores) is not int or not 1 <= cores <= 8:
        raise ValueError("cores must be an integer between 1 and 8.")
    if type(memory_mb) is not int or not 256 <= memory_mb <= 32768:
        raise ValueError("memory_mb must be an integer between 256 and 32768.")

    def _execute_in(run_root: Path) -> dict[str, Any]:
        run_env = dict(os.environ if env is None else env)
        run_env["OMP_NUM_THREADS"] = str(cores)

        if archive_47_path is not None:
            if "gennbo_exe" not in probe:
                raise FileNotFoundError(
                    "Standalone .47 execution requires gennbo.i8.exe / gennbo.i4.exe."
                )
            src_47 = Path(archive_47_path).expanduser().resolve()
            if not src_47.is_file():
                raise FileNotFoundError(f"NBO .47 archive file not found: {src_47}")
            orig_47_sha = _sha256_file(src_47)
            raw_47_text = src_47.read_text(encoding="utf-8", errors="replace")
            work_47_text = (
                _ensure_47_bndidx(raw_47_text) if ensure_bndidx else raw_47_text
            )
            work_47 = run_root / "nbo_job.47"
            work_47.write_text(work_47_text, encoding="utf-8")

            gen_path = Path(probe["gennbo_exe"])
            nbo_path = Path(probe["nbo_exe"])
            # Match i4/i8 integer width between gennbo and nbo7 client executable
            if gen_path.name.endswith(".i4.exe"):
                paired_i4 = gen_path.parent / "nbo7.i4.exe"
                if _is_executable_file(paired_i4):
                    nbo_path = paired_i4.resolve()
            elif gen_path.name.endswith(".i8.exe"):
                paired_i8 = gen_path.parent / "nbo7.i8.exe"
                if _is_executable_file(paired_i8):
                    nbo_path = paired_i8.resolve()
            run_env["NBOEXE"] = str(nbo_path)
            run_env["NBOBIN"] = str(nbo_path.parent)

            proc = subprocess.run(
                [str(gen_path), work_47.name],
                cwd=str(run_root),
                capture_output=True,
                text=True,
                env=run_env,
                timeout=timeout_seconds,
                check=False,
            )
            out_path = run_root / "gennbo.out"
            out_path.write_text(proc.stdout, encoding="utf-8")
            if proc.returncode != 0:
                raise RuntimeError(
                    f"GenNBO exited with rc={proc.returncode}: "
                    f"{(proc.stderr or proc.stdout)[-400:]}"
                )
            parsed = parse_nbo7_output(proc.stdout)
            return {
                "schema_version": "cochem.nbo7-analysis/1",
                "analysis_kind": "nbo_analysis",
                "driver_mode": "gennbo_47",
                "engine": "nbo7",
                "nbo_version": parsed["nbo_version"],
                "nbo_banner_verified": True,
                "nbo_exe": str(nbo_path),
                "nbo_exe_sha256": _sha256_file(nbo_path),
                "driver_binary": str(gen_path),
                "driver_binary_sha256": _sha256_file(gen_path),
                "input_path": str(src_47),
                "input_sha256": orig_47_sha,
                "archive_47_path": str(work_47),
                "archive_47_sha256": _sha256_file(work_47),
                "raw_output_path": str(out_path),
                "raw_output_sha256": _sha256_file(out_path),
                "npa_charges": parsed["npa_charges"],
                "npa_total": parsed["npa_total"],
                "wiberg_matrix": parsed["wiberg_matrix"],
                "wiberg_bonds": parsed["wiberg_bonds"],
                "e2_interactions": parsed["e2_interactions"],
            }

        # ORCA %nbo mode
        if "orca_bin" not in probe:
            raise FileNotFoundError(
                "ORCA binary is required to run NBO 7 from XYZ coordinates."
            )
        if xyz_path is not None:
            parsed_syms, parsed_coords, xyz_sha = read_xyz(Path(xyz_path))
        elif symbols is not None and coordinates_angstrom is not None:
            parsed_syms = [str(s) for s in symbols]
            parsed_coords = [
                [float(v) for v in row] for row in coordinates_angstrom
            ]
            if len(parsed_syms) != len(parsed_coords) or len(parsed_syms) < 2:
                raise ValueError(
                    "symbols and coordinates_angstrom must have equal length >= 2."
                )
            xyz_sha = hashlib.sha256(
                json.dumps(
                    {"symbols": parsed_syms, "coordinates": parsed_coords},
                    sort_keys=True,
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
        else:
            raise ValueError(
                "Provide either archive_47_path, xyz_path, or "
                "(symbols, coordinates_angstrom) for NBO 7 execution."
            )

        pure_syms = [_pure_element_symbol(sym) for sym in parsed_syms]
        n_elec = sum(ATOMIC_NUMBERS[sym] for sym in pure_syms) - charge
        unpaired = multiplicity - 1
        if n_elec <= 0 or unpaired > n_elec or (n_elec - unpaired) % 2 != 0:
            raise ValueError(
                f"Incompatible charge ({charge}) and multiplicity ({multiplicity}) "
                f"for electron count {n_elec}."
            )

        selected_method = dict(_NBO_DEFAULT_METHOD if method is None else method)
        header_line = _orca_method_keywords(selected_method)
        nbo_kw_clean = nbo_keywords.strip()
        if not nbo_kw_clean.upper().startswith("$NBO"):
            raise ValueError("nbo_keywords must begin with '$NBO' and end with '$END'.")

        orca_lines = [
            header_line,
            f"%maxcore {memory_mb}",
        ]
        if cores > 1:
            orca_lines.append(f"%pal nprocs {cores} end")
        orca_lines.extend(
            [
                "%nbo",
                f'  NBOKEYLIST = "{nbo_kw_clean}"',
                "end",
                f"* xyz {charge} {multiplicity}",
            ]
        )
        for sym, pos in zip(pure_syms, parsed_coords, strict=True):
            orca_lines.append(f"{sym:<3s} {pos[0]:16.10f} {pos[1]:16.10f} {pos[2]:16.10f}")
        orca_lines.append("*")

        inp_path = run_root / "orca_nbo.inp"
        inp_text = "\n".join(orca_lines) + "\n"
        inp_path.write_text(inp_text, encoding="utf-8")

        orca_path = Path(probe["orca_bin"])
        nbo_path = Path(probe["nbo_exe"])
        run_env["NBOEXE"] = str(nbo_path)
        run_env["NBOBIN"] = str(nbo_path.parent)
        ld_parts: list[str] = []
        if run_env.get("COCHEM_ORCA_LD_LIBRARY_PATH"):
            ld_parts.append(run_env["COCHEM_ORCA_LD_LIBRARY_PATH"])
        ld_parts.append(str(orca_path.parent))
        if run_env.get("LD_LIBRARY_PATH"):
            ld_parts.append(run_env["LD_LIBRARY_PATH"])
        run_env["LD_LIBRARY_PATH"] = ":".join(part for part in ld_parts if part)

        proc = subprocess.run(
            [str(orca_path), inp_path.name],
            cwd=str(run_root),
            capture_output=True,
            text=True,
            env=run_env,
            timeout=timeout_seconds,
            check=False,
        )
        out_path = run_root / "orca_nbo.out"
        out_path.write_text(proc.stdout, encoding="utf-8")
        if proc.returncode != 0:
            raise RuntimeError(
                f"ORCA+NBO7 exited with rc={proc.returncode}: "
                f"{(proc.stderr or proc.stdout)[-500:]}"
            )
        if "****ORCA TERMINATED NORMALLY****" not in proc.stdout:
            raise RuntimeError(
                "ORCA output lacks '****ORCA TERMINATED NORMALLY****' marker."
            )
        parsed = parse_nbo7_output(proc.stdout)
        archive_file = run_root / "FILE.47"
        archive_sha = _sha256_file(archive_file) if archive_file.is_file() else None
        return {
            "schema_version": "cochem.nbo7-analysis/1",
            "analysis_kind": "nbo_analysis",
            "driver_mode": "orca_nbo",
            "engine": "orca+nbo7",
            "nbo_version": parsed["nbo_version"],
            "nbo_banner_verified": True,
            "method": selected_method,
            "electronic_state": {"charge": charge, "multiplicity": multiplicity},
            "atoms": parsed_syms,
            "coordinates_angstrom": parsed_coords,
            "nbo_exe": str(nbo_path),
            "nbo_exe_sha256": _sha256_file(nbo_path),
            "driver_binary": str(orca_path),
            "driver_binary_sha256": _sha256_file(orca_path),
            "input_sha256": xyz_sha,
            "orca_input_sha256": _sha256_file(inp_path),
            "archive_47_path": str(archive_file) if archive_file.is_file() else None,
            "archive_47_sha256": archive_sha,
            "raw_output_path": str(out_path),
            "raw_output_sha256": _sha256_file(out_path),
            "npa_charges": parsed["npa_charges"],
            "npa_total": parsed["npa_total"],
            "wiberg_matrix": parsed["wiberg_matrix"],
            "wiberg_bonds": parsed["wiberg_bonds"],
            "e2_interactions": parsed["e2_interactions"],
        }

    if work_dir is not None:
        run_root = Path(work_dir).expanduser().resolve()
        run_root.mkdir(parents=True, exist_ok=True)
        return _execute_in(run_root)

    with tempfile.TemporaryDirectory(prefix="torq_nbo7_") as tmp_dir:
        return _execute_in(Path(tmp_dir))


def _source_digest() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def scientific_metadata() -> dict[str, Any]:
    """Observe available runtimes; installation is not accuracy qualification."""
    from cochem_torq.engines.pyscf_backend import PySCFBackend

    try:
        probe = PySCFBackend.probe()
    except (OSError, RuntimeError) as exc:
        probe = {"available": False, "engine": "PySCF", "reason": str(exc)}
    receipts: dict[str, Any] = {}
    if probe.get("available"):
        import pyscf

        source = Path(pyscf.__file__).resolve(strict=True)
        receipts["pyscf"] = {
            "available": True,
            "version": probe["version"],
            "module_file": str(source),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
    nbo_probe = probe_nbo7()
    nbo_receipts: dict[str, Any] = {}
    if nbo_probe.get("available"):
        nbo_receipts["nbo7"] = {
            "available": True,
            "executable_path": nbo_probe["nbo_exe"],
            "sha256": nbo_probe["nbo_exe_sha256"],
            "modes_available": nbo_probe["modes_available"],
        }
        if "orca_bin" in nbo_probe:
            nbo_receipts["orca"] = {
                "available": True,
                "executable_path": nbo_probe["orca_bin"],
                "sha256": nbo_probe["orca_bin_sha256"],
            }
        if "gennbo_exe" in nbo_probe:
            nbo_receipts["gennbo"] = {
                "available": True,
                "executable_path": nbo_probe["gennbo_exe"],
                "sha256": nbo_probe["gennbo_exe_sha256"],
            }

    definitions: dict[str, Any] = {
        "geometry_analysis": {
            "available": True,
            "required_engines": [],
            "supported_engines": [],
            "scope": (
                "Supplied-geometry mass, center of mass, proper "
                "principal-axis alignment, inertia and rigid-rotor "
                "constants; no stationarity or B0 claim."
            ),
        },
        "research_scan": {
            "available": bool(probe.get("available")),
            "required_engines": ["pyscf"],
            "supported_engines": ["pyscf"],
            "runtime_receipts": receipts,
            "engine": probe,
            "definition": PySCFBackend.capabilities(),
            "scope": (
                "Bounded rigid two-fragment mass-COM separation scan using "
                "actual restricted native electronic energies; no "
                "relaxation or counterpoise."
            ),
        },
        "wiberg_lowdin": {
            "available": bool(probe.get("available")),
            "required_engines": ["pyscf"],
            "supported_engines": ["pyscf"],
            "runtime_receipts": receipts,
            "scope": (
                "Wiberg indices and charges in the Löwdin orthogonalized AO"
                " basis from the authenticated converged stable real "
                "restricted SCF density; not NAO or NBO."
            ),
        },
        "nbo_analysis": {
            "available": bool(nbo_probe.get("available")),
            "required_engines": ["nbo7"],
            "supported_engines": ["orca", "gennbo"],
            "modes_available": nbo_probe.get("modes_available", []),
            "default_method": dict(_NBO_DEFAULT_METHOD),
            "runtime_receipts": nbo_receipts,
            "scope": (
                "Authentic NBO 7 (ORCA %nbo with NBOEXE or standalone GenNBO on .47) "
                "Natural Population Analysis charges, NAO Wiberg bond indices, and "
                "second-order perturbation E(2) donor-acceptor energies with raw "
                "SHA-256 provenance."
            ),
        },
    }
    if not probe.get("available"):
        for name in ("research_scan", "wiberg_lowdin"):
            definitions[name]["reason"] = probe.get("reason", "PySCF is unavailable.")
    if not nbo_probe.get("available"):
        definitions["nbo_analysis"]["reason"] = nbo_probe.get(
            "reason", "Native NBO 7 executable is unavailable."
        )

    try:
        nao_identity = _nao_algorithm_identity() if probe.get("available") else None
        nao_reason = (
            "PySCF NAO producer is not activated."
            if probe.get("available")
            else probe.get("reason", "PySCF is unavailable.")
        )
    except (ImportError, OSError, ValueError, IndexError, TypeError) as exc:
        nao_identity, nao_reason = None, str(exc)
    definitions["wiberg_nao"] = {
        "available": bool(probe.get("available"))
        and bool(nao_identity)
        and _NAO_PROFILE_ACTIVATED,
        "required_engines": ["pyscf"],
        "supported_engines": ["pyscf"],
        "profile": _NAO_PROFILE,
        "activation_status": "locally_validated"
        if _NAO_PROFILE_ACTIVATED
        else "experimental",
        "scope": (
            "General-element all-electron closed-shell PySCF NAO-basis Wiberg "
            "pair descriptors supporting arbitrary periodic-table elements and "
            "user-specified restricted HF/DFT methods and basis sets."
        ),
        "required_options": ["engine"],
        "optional_options": [
            "method",
            "charge",
            "multiplicity",
            "cores",
            "memory_mb",
        ],
        "default_method": dict(_NAO_DEFAULT_METHOD),
        "supported_methods": sorted(_NAO_SUPPORTED_METHODS),
        "reference": "real_closed_shell_restricted",
        "basis_convention": "all_electron_spherical",
        "platform": f"{platform.system().lower()}-cpu",
        "input_elements": "all_periodic_table_all_electron",
        "local_validation_cases": list(_NAO_CONTROLLED_EVIDENCE["actual_cases"]),
        "local_validation_evidence": _NAO_CONTROLLED_EVIDENCE,
        "algorithm": nao_identity,
        "runtime_receipts": receipts,
    }
    if not definitions["wiberg_nao"]["available"]:
        definitions["wiberg_nao"]["reason"] = nao_reason
    return {
        "scientific_api_version": "cochem.module-provider/1",
        "execute_handoff": "cochem_torq.module_provider:execute_handoff",
        "operations": list(_OPERATIONS),
        "scientific_apis": definitions,
    }


def read_xyz(path: Path) -> tuple[list[str], list[list[float]], str]:
    """Read one complete Cartesian XYZ geometry, with coordinates in angstrom."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("The input artifact must be a regular XYZ file.")
    if path.stat().st_size > _MAX_INPUT_BYTES:
        raise ValueError("The XYZ artifact exceeds the 2 MB input limit.")
    raw = path.read_bytes()
    lines = raw.decode("utf-8-sig").splitlines()
    if len(lines) < 3 or not re.fullmatch(r"[1-9][0-9]*", lines[0].strip()):
        raise ValueError(
            "XYZ must start with a positive atom count and a comment line."
        )
    count = int(lines[0].strip())
    if count < 2 or count > _MAX_ATOMS:
        raise ValueError(f"Rigid-rotor analysis requires 2 to {_MAX_ATOMS} atoms.")
    if len(lines) < count + 2 or any(line.strip() for line in lines[count + 2 :]):
        raise ValueError("XYZ must contain exactly the declared number of atom rows.")
    symbols: list[str] = []
    coordinates: list[list[float]] = []
    for number, line in enumerate(lines[2 : count + 2], start=3):
        fields = line.split()
        if len(fields) != 4 or not _SYMBOL.fullmatch(fields[0]):
            raise ValueError(
                f"Invalid XYZ atom row {number}; expected symbol and three coordinates."
            )
        try:
            position = [float(value) for value in fields[1:]]
        except ValueError as exc:
            raise ValueError(
                f"Invalid numeric coordinate on XYZ row {number}."
            ) from exc
        if not all(math.isfinite(value) for value in position):
            raise ValueError(f"Nonfinite coordinate on XYZ row {number}.")
        label = fields[0]
        suffix = re.fullmatch(r"([A-Z][a-z]?)-?([1-9][0-9]{0,2})", label)
        symbols.append(
            suffix.group(2) + suffix.group(1)
            if suffix
            else {"D": "2H", "T": "3H"}.get(label, label)
        )
        coordinates.append(position)
    for index, position in enumerate(coordinates):
        for previous in coordinates[:index]:
            if math.dist(position, previous) < 1e-8:
                raise ValueError(
                    "Two atoms occupy the same position; rigid-rotor input is invalid."
                )
    return symbols, coordinates, hashlib.sha256(raw).hexdigest()


def _publish_json(path: Path, payload: dict[str, Any]) -> None:
    """Publish a complete strict JSON file atomically, without overwriting one."""
    encoded = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".torq-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        # An atomic link fails if the destination already exists, including a
        # dangling symlink. It cannot replace an artifact created concurrently.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def analyze(artifact: Path) -> dict[str, Any]:
    symbols, positions, digest = read_xyz(artifact)
    import mendeleev
    import numpy as np
    from scipy.constants import atomic_mass, h

    import Libraries.cochem_torq_alignment as provider

    geometry = np.asarray(positions, dtype=np.float64)
    native = provider.EckartAligner.align_molecule(
        symbols,
        geometry,
        metadata={"input_sha256": digest, "coordinate_unit": "angstrom"},
    )
    result = native.to_dict()
    masses = np.asarray(result["exact_masses"], dtype=np.float64)
    aligned = np.asarray(result["aligned_geometry"], dtype=np.float64)
    rotation = np.asarray(result["rotation_matrix"], dtype=np.float64)
    center = np.asarray(result["com_vector"], dtype=np.float64)
    moments = np.asarray(result["principal_moments"], dtype=np.float64)
    if (
        result["symbols"] != symbols
        or masses.shape != (len(symbols),)
        or not np.isfinite(masses).all()
        or not (masses > 0).all()
        or aligned.shape != geometry.shape
        or not np.isfinite(aligned).all()
        or rotation.shape != (3, 3)
        or not np.isfinite(rotation).all()
        or center.shape != (3,)
        or not np.isfinite(center).all()
        or moments.shape != (3,)
        or not np.isfinite(moments).all()
        or (moments < -1e-9).any()
    ):
        raise ValueError(
            "TORQ returned invalid geometry, masses, or principal-axis data."
        )
    isotope_masses: dict[str, float] = {}
    for symbol, observed_mass in zip(symbols, masses, strict=True):
        if symbol not in isotope_masses:
            normalized = {"D": "2H", "T": "3H"}.get(symbol, symbol)
            match = re.fullmatch(r"([0-9]*)([A-Z][a-z]?)", normalized)
            assert match is not None
            mass_number, element_symbol = match.groups()
            isotopes = getattr(mendeleev, "element")(element_symbol).isotopes
            if mass_number:
                candidates = [
                    item for item in isotopes if item.mass_number == int(mass_number)
                ]
            else:
                candidates = [
                    item
                    for item in isotopes
                    if item.abundance is not None and item.abundance > 0
                ]
                candidates.sort(
                    key=lambda item: (item.abundance, item.mass_number), reverse=True
                )
            if not candidates or candidates[0].mass is None:
                raise ValueError(
                    f"An explicitly documented isotope is required for {symbol}."
                )
            isotope_masses[symbol] = float(candidates[0].mass)
        if not math.isclose(
            float(observed_mass), isotope_masses[symbol], rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError(f"TORQ mass differs from requested isotope {symbol}.")
    if not np.allclose(rotation @ rotation.T, np.eye(3), rtol=0, atol=1e-10):
        raise ValueError("TORQ returned a nonorthogonal frame transformation.")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, rel_tol=0, abs_tol=1e-10):
        raise ValueError("TORQ returned a frame that is not right-handed.")
    roundtrip_error = float(np.max(np.abs(aligned @ rotation + center - geometry)))
    if roundtrip_error > 1e-9:
        raise ValueError(
            "TORQ's frame transformation did not preserve the input geometry."
        )
    if not np.allclose(
        center, np.average(geometry, axis=0, weights=masses), rtol=0, atol=1e-9
    ):
        raise ValueError("TORQ returned an inconsistent center of mass.")
    distance_error = 0.0
    for index in range(len(symbols)):
        before = np.linalg.norm(geometry[index + 1 :] - geometry[index], axis=1)
        after = np.linalg.norm(aligned[index + 1 :] - aligned[index], axis=1)
        if before.size:
            distance_error = max(distance_error, float(np.max(np.abs(before - after))))
    if distance_error > 1e-9:
        raise ValueError("TORQ's alignment changed interatomic distances.")
    centered = geometry - center
    inertia = sum(
        mass * (np.dot(position, position) * np.eye(3) - np.outer(position, position))
        for mass, position in zip(masses, centered, strict=True)
    )
    independent_moments = np.linalg.eigvalsh(inertia)
    if not np.allclose(moments, independent_moments, rtol=1e-10, atol=1e-9):
        raise ValueError(
            "TORQ's principal moments do not reproduce the supplied mass geometry."
        )
    if float(np.max(moments)) < 1e-10:
        raise ValueError(
            "All principal inertias are below the provider's numerical rotor threshold."
        )

    # A linear molecule has no finite rotational constant about its zero-inertia
    # axis. Preserve that absence explicitly; all other nonfinite data fail.
    undefined: list[dict[str, Any]] = []
    for field, unit in (
        ("rotational_constants_mhz", "MHz"),
        ("rotational_constants_ghz", "GHz"),
        ("rotational_constants_cm1", "cm^-1"),
    ):
        values = list(result[field])
        if len(values) != 3:
            raise ValueError("TORQ returned an invalid rotational-constant vector.")
        for axis, value in enumerate(values):
            if not math.isfinite(value):
                if value != math.inf or not -1e-9 <= moments[axis] < 1e-10:
                    raise ValueError(
                        "TORQ returned an unexpected nonfinite rotational constant."
                    )
                values[axis] = None
                undefined.append(
                    {
                        "field": field,
                        "axis": "ABC"[axis],
                        "unit": unit,
                        "reason": (
                            "No finite rigid-rotor constant for an axis with principal "
                            "inertia below TORQ's 1e-10 Da angstrom^2 threshold."
                        ),
                    }
                )
            elif value <= 0:
                raise ValueError("TORQ returned a nonpositive rotational constant.")
        result[field] = values
    a_value, b_value, c_value = result["rotational_constants_mhz"]
    if a_value is not None and c_value is not None and a_value == c_value:
        result["ray_kappa"] = None
        undefined.append(
            {
                "field": "ray_kappa",
                "reason": (
                    "Ray's asymmetry parameter has a zero denominator for a "
                    "spherical top (A = B = C)."
                ),
            }
        )
    elif a_value is not None and b_value is not None and c_value is not None:
        expected_kappa = (b_value - c_value) / (a_value - c_value) - (
            a_value - b_value
        ) / (a_value - c_value)
        if (
            not isinstance(result.get("ray_kappa"), (int, float))
            or not math.isfinite(result["ray_kappa"])
            or not math.isclose(
                result["ray_kappa"], expected_kappa, rel_tol=1e-10, abs_tol=1e-10
            )
        ):
            raise ValueError(
                "TORQ's Ray parameter does not reproduce the finite "
                "observed rotational constants."
            )
    prefactor_mhz = h / (8 * np.pi**2 * atomic_mass * 1e-20) / 1e6
    for axis, value in enumerate(result["rotational_constants_mhz"]):
        if value is not None and not math.isclose(
            value,
            float(prefactor_mhz / independent_moments[axis]),
            rel_tol=1e-10,
            abs_tol=1e-7,
        ):
            raise ValueError(
                "TORQ's rotational constants fail the independent inertia/unit check."
            )
    payload = {
        "schema_version": "cochem.module-operation/1",
        "module_id": "torq",
        "operation": "geometry_analysis",
        "status": "succeeded",
        "input_sha256": digest,
        "input_coordinate_unit": "angstrom",
        "scope": (
            "Rigid-geometry center of mass, principal inertia, "
            "principal-axis (Eckart) alignment and rotor properties of "
            "the supplied geometry. No optimization, electronic energy,"
            " vibrational correction, or measured spectral constants."
        ),
        "provider_file": str(Path(__file__).resolve()),
        "provider_sha256": _source_digest(),
        "provider_distribution": "CoChem-TORQ",
        "provider_version": importlib.metadata.version("CoChem-TORQ"),
        "mass_database": {
            "name": "mendeleev",
            "version": importlib.metadata.version("mendeleev"),
            "selection": (
                "Explicit isotopes when supplied; otherwise most abundant isotope."
            ),
        },
        "validation": {
            "coordinate_roundtrip_max_error_angstrom": roundtrip_error,
            "pair_distance_max_error_angstrom": distance_error,
            "independent_inertia_and_mhz_check": "passed",
        },
        "undefined_observables": undefined,
        "result": result,
    }
    json.dumps(payload, allow_nan=False)
    return payload


def _load_handoff(
    path: Path,
) -> tuple[dict[str, Any], Path, list[str], list[list[float]]]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 2_000_000:
        raise ValueError("The handoff must be a bounded retained JSON file.")

    def unique(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON handoff fields are prohibited.")
            result[key] = value
        return result

    record = json.loads(path.read_bytes(), object_pairs_hook=unique)
    json.dumps(record, allow_nan=False)
    allowed_fields = {
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
        not isinstance(record, dict)
        or set(record) - allowed_fields
        or record.get("schema_version") != "cochem.module-handoff/1"
        or record.get("module_id") != "torq"
        or record.get("operation") not in _OPERATIONS
        or record.get("scientific_execution_performed") is not False
    ):
        raise ValueError("Unsupported TORQ research handoff.")
    if not isinstance(record.get("options"), dict):
        raise ValueError("Scientific options must be an explicit JSON object.")
    reference = record.get("artifact", {})
    if not isinstance(reference, dict):
        raise ValueError("A geometry artifact reference is required.")
    filename = reference.get("filename", "")
    if (
        not isinstance(filename, str)
        or filename in {"", ".", ".."}
        or Path(filename).name != filename
        or reference.get("kind") != "geometry_xyz"
    ):
        raise ValueError("TORQ scans require an immutable XYZ handoff.")
    artifact = path.parent / filename
    if (
        artifact.is_symlink()
        or not artifact.is_file()
        or artifact.stat().st_size != reference.get("size_bytes")
    ):
        raise ValueError("The handoff geometry is absent or has changed.")
    symbols, positions, digest = read_xyz(artifact)
    if digest != reference.get("sha256"):
        raise ValueError("The handoff geometry checksum has changed.")
    metadata = reference.get("metadata", {})
    if (
        not isinstance(metadata, dict)
        or metadata.get("coordinates_unit", "angstrom") != "angstrom"
        or metadata.get("atom_count", len(symbols)) != len(symbols)
    ):
        raise ValueError(
            "The geometry metadata differs from its supplied angstrom XYZ."
        )
    return record, artifact, symbols, positions


def _scan_options(options: dict[str, Any], atom_count: int) -> dict[str, Any]:
    allowed = {
        "method",
        "charge",
        "multiplicity",
        "fragments",
        "distances_angstrom",
        "cores",
        "memory_mb",
    }
    if (
        not isinstance(options, dict)
        or set(options) - allowed
        or not 2 <= atom_count <= 50
    ):
        raise ValueError("Unsupported or oversized scan request.")
    method = options.get("method")
    if (
        not isinstance(method, dict)
        or set(method) != {"name", "basis"}
        or method["name"] not in {"hf", "pbe-d4", "b3lyp-d4", "mp2"}
        or not isinstance(method["basis"], str)
        or not method["basis"].strip()
        or re.fullmatch(r"[A-Za-z0-9+*(),_. -]{1,100}", method["basis"]) is None
    ):
        raise ValueError("Choose an explicit supported scan method and basis.")
    if (
        type(options.get("charge")) is not int
        or abs(options["charge"]) > 4
        or type(options.get("multiplicity")) is not int
        or options["multiplicity"] != 1
    ):
        raise ValueError(
            "The bounded scan requires an integer charge and "
            "closed-shell singlet state."
        )
    fragments = options.get("fragments")
    if (
        not isinstance(fragments, list)
        or len(fragments) != 2
        or any(not isinstance(part, list) or not part for part in fragments)
    ):
        raise ValueError("Two complete, nonempty fragments are required.")
    indices = [index for part in fragments for index in part]
    if any(type(index) is not int for index in indices) or sorted(indices) != list(
        range(atom_count)
    ):
        raise ValueError("Each atom must appear in exactly one fragment.")
    distances = options.get("distances_angstrom")
    if (
        not isinstance(distances, list)
        or not 1 <= len(distances) <= 20
        or any(
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
            or not 0.2 <= value <= 100
            for value in distances
        )
        or len(set(distances)) != len(distances)
    ):
        raise ValueError(
            "Choose 1–20 distinct finite mass-COM separations between "
            "0.2 and 100 angstrom."
        )
    cores, memory = options.get("cores", 1), options.get("memory_mb", 1500)
    if (
        type(cores) is not int
        or not 1 <= cores <= 2
        or type(memory) is not int
        or not 256 <= memory <= 2048
    ):
        raise ValueError(
            "The bounded scan permits 1–2 cores and 256–2048 MiB total memory."
        )
    return {**options, "cores": cores, "memory_mb": memory}


def _scan_nuclear_masses(symbols: list[str]) -> tuple[list[str], list[float]]:
    """Resolve measured nuclear masses with the BASE handoff convention.

    Natural atomic weights describe isotope mixtures. A scan of one molecular
    geometry must retain its explicit isotopes or use the measured principal
    isotope, matching the mass-COM coordinate shown before submission.
    """
    import mendeleev

    element = getattr(mendeleev, "element")

    elements, masses = [], []
    for symbol in symbols:
        normalized = {"D": "2H", "T": "3H"}.get(symbol, symbol)
        match = re.fullmatch(r"([0-9]*)([A-Z][a-z]?)", normalized)
        if match is None:
            raise ValueError("Unsupported nuclear identity.")
        number, label = match.groups()
        nucleus = element(label)
        measured = [
            isotope
            for isotope in nucleus.isotopes
            if isotope.mass is not None
            and math.isfinite(float(isotope.mass))
            and isotope.mass > 0
        ]
        if number:
            selected = next(
                (isotope for isotope in measured if isotope.mass_number == int(number)),
                None,
            )
        else:
            natural = [
                isotope
                for isotope in measured
                if isotope.abundance is not None and isotope.abundance > 0
            ]
            selected = (
                max(
                    natural,
                    key=lambda isotope: (isotope.abundance, -isotope.mass_number),
                )
                if natural
                else None
            )
        if selected is None:
            raise ValueError(
                "The scan requires a measured explicit isotope or a "
                "measured principal natural isotope."
            )
        elements.append(label)
        masses.append(float(selected.mass))
    return elements, masses


def research_scan(handoff_path: Path, directory: Path) -> dict[str, Any]:
    """Run authentic provider evaluations at rigid translated fragment geometries."""
    record, artifact, symbols, positions = _load_handoff(handoff_path)
    options = _scan_options(record["options"], len(symbols))
    import numpy as np
    from scipy.constants import physical_constants

    from cochem_torq.engines import pyscf_backend as provider

    if not scientific_metadata()["scientific_apis"]["research_scan"]["available"]:
        raise ValueError("The installed TORQ native scan producer is not available.")
    elements, nuclear_masses = _scan_nuclear_masses(symbols)
    masses = np.asarray(nuclear_masses)
    geometry = np.asarray(positions)
    a, b = options["fragments"]
    center_a = np.average(geometry[a], axis=0, weights=masses[a])
    center_b = np.average(geometry[b], axis=0, weights=masses[b])
    vector = center_b - center_a
    initial = float(np.linalg.norm(vector))
    if initial < 1e-8:
        raise ValueError(
            "Fragment mass centers coincide; a separation direction is required."
        )
    direction = vector / initial
    conversion = 1e-10 / physical_constants["Bohr radius"][0]
    planned = []
    for distance in options["distances_angstrom"]:
        shifted = geometry.copy()
        shifted[b] += (distance - initial) * direction
        pair = np.linalg.norm(shifted[:, None] - shifted[None, :], axis=2)
        np.fill_diagonal(pair, np.inf)
        if np.min(pair) < 0.1:
            raise ValueError("The scan places nuclei less than 0.1 angstrom apart.")
        planned.append((float(distance), shifted))
    points: list[dict[str, Any]] = []
    provider_path = Path(__file__).resolve()

    def snapshot() -> dict[str, Any]:
        published_points = list(points) + [
            {
                "coordinate_angstrom": distance,
                "geometry_angstrom": shifted.tolist(),
                "status": "not_run",
                "energy_hartree": None,
                "reason": "This planned point has no completed native evaluation.",
            }
            for distance, shifted in planned[len(points) :]
        ]
        status = (
            "completed"
            if all(point["status"] == "computed" for point in published_points)
            else "partial"
        )
        return {
            "schema_version": "cochem.module-operation/1",
            "module_id": "torq",
            "operation": "research_scan",
            "input_sha256": record["artifact"]["sha256"],
            "status": status,
            "provider_file": str(provider_path),
            "provider_sha256": hashlib.sha256(provider_path.read_bytes()).hexdigest(),
            "provider_distribution": "CoChem-TORQ",
            "provider_version": importlib.metadata.version("CoChem-TORQ"),
            "scope": (
                "Actual native electronic energies along a rigid "
                "two-fragment mass-COM separation coordinate; "
                "intramolecular geometries and orientation remain fixed. No"
                " relaxation, counterpoise, barrier/minimum verification or"
                " universal chemical-accuracy claim."
            ),
            "result": {
                "schema_version": "cochem.torq-rigid-scan/1",
                "points": published_points,
                "elements": elements,
                "nuclides": symbols,
                "method": options["method"],
                "fragments": options["fragments"],
                "electronic_state": {"charge": options["charge"], "multiplicity": 1},
                "initial_mass_com_separation_angstrom": initial,
                "masses_u": masses.tolist(),
                "mass_source": (
                    "dynamic_mendeleev_explicit_isotope_else_principal_isotope_mass"
                ),
                "mass_convention": "explicit_isotope_mass_else_principal_isotope_mass",
                "resources": {
                    "cores": options["cores"],
                    "memory_mb": options["memory_mb"],
                },
            },
        }

    for index, (distance, shifted) in enumerate(planned):
        point_directory = directory / f"scan-point-{index:03d}"
        native = None
        request = {
            "molecule": {
                "symbols": symbols,
                "geometry_bohr": (shifted * conversion).tolist(),
                "charge": options["charge"],
                "multiplicity": 1,
            },
            "method": options["method"],
            "properties": ["energy"],
            "settings": {
                "threads": options["cores"],
                "memory_mb": options["memory_mb"],
                "check_stability": True,
            },
        }
        try:
            native = provider.PySCFBackend().evaluate(request, point_directory)
            energy = native.get("energy_hartree")
            if (
                native.get("status") != "complete"
                or isinstance(energy, bool)
                or not isinstance(energy, (float, int))
                or not math.isfinite(energy)
            ):
                raise ValueError(
                    f"Native TORQ calculation did not complete: {native.get('errors')}"
                )
            result_path = point_directory / "result.json"
            points.append(
                {
                    "coordinate_angstrom": distance,
                    "geometry_angstrom": shifted.tolist(),
                    "status": "computed",
                    "energy_hartree": energy,
                    "result_path": str(result_path.relative_to(directory)),
                    "result_sha256": hashlib.sha256(
                        result_path.read_bytes()
                    ).hexdigest(),
                    "native_manifest_path": str(
                        Path(native["manifest_path"]).relative_to(directory)
                    ),
                    "native_manifest_sha256": native["manifest_sha256"],
                    "engine_version": native["engine_version"],
                }
            )
        except (ValueError, RuntimeError) as exc:
            failed = {
                "coordinate_angstrom": distance,
                "geometry_angstrom": shifted.tolist(),
                "status": "failed",
                "energy_hartree": None,
                "reason": str(exc),
            }
            result_path = point_directory / "result.json"
            if result_path.is_file():
                failed.update(
                    {
                        "result_path": str(result_path.relative_to(directory)),
                        "result_sha256": hashlib.sha256(
                            result_path.read_bytes()
                        ).hexdigest(),
                    }
                )
            if native is not None:
                failed.update(
                    {
                        "native_manifest_path": str(
                            Path(native["manifest_path"]).relative_to(directory)
                        ),
                        "native_manifest_sha256": native["manifest_sha256"],
                        "engine_version": native["engine_version"],
                    }
                )
            points.append(failed)
        _publish_json(directory / f"scan-progress-{index:03d}.json", snapshot())
    return snapshot()


def _restricted_scf_density(
    symbols: list[str],
    positions: list[list[float]],
    checked: dict[str, Any],
    directory: Path,
) -> tuple[Any, ...]:
    """Return authenticated native restricted orbitals and their original AO metric."""
    import numpy as np
    from pyscf import lib
    from scipy.constants import physical_constants

    from cochem_torq.engines import pyscf_backend as provider
    from cochem_torq.engines.diagnostics import verify_native_artifacts

    conversion = 1e-10 / physical_constants["Bohr radius"][0]
    native = provider.PySCFBackend().evaluate(
        {
            "molecule": {
                "symbols": symbols,
                "geometry_bohr": (np.asarray(positions) * conversion).tolist(),
                "charge": checked["charge"],
                "multiplicity": 1,
            },
            "method": checked["method"],
            "properties": ["energy"],
            "settings": {
                "threads": checked["cores"],
                "memory_mb": checked["memory_mb"],
                "check_stability": True,
            },
        },
        directory / "native-electronic",
    )
    if (
        native.get("status") != "complete"
        or native.get("scf", {}).get("converged") is not True
        or native.get("stability", {}).get("status") != "stable"
    ):
        raise ValueError(
            "Wiberg analysis requires a real converged and stable "
            "restricted SCF calculation."
        )
    native_directory = verify_native_artifacts(native)
    checkpoint = native_directory / "wavefunction.chk"
    molecule = lib.chkfile.load_mol(str(checkpoint))
    scf = lib.chkfile.load(str(checkpoint), "scf")
    coefficients, occupations = np.asarray(scf["mo_coeff"]), np.asarray(scf["mo_occ"])
    if (
        coefficients.shape != (molecule.nao_nr(), len(occupations))
        or np.iscomplexobj(coefficients)
        or np.iscomplexobj(occupations)
        or not np.isfinite(coefficients).all()
        or not np.isfinite(occupations).all()
        or not np.isin(occupations, [0.0, 2.0]).all()
        or abs(float(np.sum(occupations)) - molecule.nelectron) > 1e-8
        or molecule.spin != 0
        or not np.allclose(
            molecule.atom_coords(unit="Bohr"),
            native["geometry_bohr"],
            atol=1e-10,
            rtol=0,
        )
        or abs(float(scf["e_tot"]) - native["scf"]["energy_hartree"]) > 1e-8
    ):
        raise ValueError(
            "The native checkpoint differs from its retained restricted"
            " density identity, energy or geometry."
        )
    overlap = molecule.intor_symmetric("int1e_ovlp")
    eigenvalues, _ = np.linalg.eigh(overlap)
    if not np.isfinite(eigenvalues).all() or np.min(eigenvalues) <= 1e-10:
        raise ValueError(
            "Restricted density analysis requires a positive, numerically "
            "independent AO overlap basis."
        )
    if not np.allclose(
        coefficients.T @ overlap @ coefficients,
        np.eye(coefficients.shape[1]),
        rtol=0,
        atol=1e-7,
    ):
        raise ValueError(
            "The native restricted orbitals fail AO overlap normalization."
        )
    return native, checkpoint, molecule, coefficients, occupations, overlap


def wiberg_lowdin(handoff_path: Path, directory: Path) -> dict[str, Any]:
    """Compute named Löwdin-AO Wiberg indices from real density/overlap matrices."""
    record, _, symbols, positions = _load_handoff(handoff_path)
    options = record["options"]
    if not isinstance(options, dict) or set(options) - {
        "method",
        "charge",
        "multiplicity",
        "cores",
        "memory_mb",
    }:
        raise ValueError("Unsupported Wiberg analysis option.")
    # Reuse the exact checked model/state/resource contract without permitting
    # scan geometry fields in this independent operation.
    checked = _scan_options(
        {
            **options,
            "fragments": [[0], list(range(1, len(symbols)))],
            "distances_angstrom": [1.0],
        },
        len(symbols),
    )
    if checked["method"]["name"] == "mp2":
        raise ValueError(
            "A correlated MP2 density is not available in the retained "
            "restricted SCF checkpoint."
        )
    import numpy as np

    native, checkpoint, molecule, coefficients, occupations, overlap = (
        _restricted_scf_density(symbols, positions, checked, directory)
    )
    eigenvalues, eigenvectors = np.linalg.eigh(overlap)
    square_root = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
    density_ao = (coefficients * occupations) @ coefficients.T
    density_lowdin = square_root @ density_ao @ square_root
    if (
        not np.isfinite(density_lowdin).all()
        or not np.allclose(density_lowdin, density_lowdin.T, atol=1e-10, rtol=0)
        or abs(float(np.trace(density_lowdin)) - molecule.nelectron) > 1e-7
    ):
        raise ValueError(
            "The orthogonalized real density failed electron-count or "
            "symmetry validation."
        )
    slices = molecule.aoslice_by_atom()
    bonds, populations, charges = [], [], []
    for i, (_, _, begin, end) in enumerate(slices):
        population = float(np.trace(density_lowdin[begin:end, begin:end]))
        populations.append(population)
        charges.append(float(molecule.atom_charge(i)) - population)
        for j in range(i + 1, len(slices)):
            other_begin, other_end = slices[j][2:]
            value = float(np.sum(density_lowdin[begin:end, other_begin:other_end] ** 2))
            bonds.append({"atom_i": i, "atom_j": j, "value": value})
    if abs(math.fsum(charges) - checked["charge"]) > 1e-7:
        raise ValueError(
            "The Löwdin atomic charges do not conserve the requested total charge."
        )
    matrices = directory / "lowdin-density-overlap.npz"
    with matrices.open("xb") as stream:
        np.savez(
            stream,
            density_ao=density_ao,
            overlap_ao=overlap,
            density_lowdin=density_lowdin,
        )
    analysis = {
        "schema_version": "cochem.wiberg-lowdin/1",
        "analysis_kind": "wiberg_lowdin",
        "atoms": symbols,
        "coordinates_angstrom": positions,
        "bonds": bonds,
        "charges_e": charges,
        "populations_e": populations,
        "method": checked["method"],
        "electronic_state": {"charge": checked["charge"], "multiplicity": 1},
        "definition": (
            "W_AB = sum_(mu in A,nu in B) (P_Lowdin[mu,nu])^2; P_Lowdin"
            " = S^(1/2) P_AO S^(1/2), spin-summed restricted density."
        ),
        "scope": (
            "Löwdin orthogonalized AO Wiberg indices and charges. These"
            " are not NAO-basis Wiberg indices or Natural Bond Orbital "
            "analysis."
        ),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "native_manifest_sha256": native["manifest_sha256"],
        "matrix_sha256": hashlib.sha256(matrices.read_bytes()).hexdigest(),
        "validation": {
            "electron_count": int(molecule.nelectron),
            "density_trace": float(np.trace(density_lowdin)),
            "charge_sum": math.fsum(charges),
            "minimum_overlap_eigenvalue": float(np.min(eigenvalues)),
        },
    }
    analysis_path = directory / "bond-analysis.json"
    _publish_json(analysis_path, analysis)
    provider_path = Path(__file__).resolve()
    return {
        "schema_version": "cochem.module-operation/1",
        "module_id": "torq",
        "operation": "wiberg_lowdin",
        "status": "completed",
        "input_sha256": record["artifact"]["sha256"],
        "provider_file": str(provider_path),
        "provider_sha256": hashlib.sha256(provider_path.read_bytes()).hexdigest(),
        "scope": analysis["scope"],
        "result": {
            **analysis,
            "analysis_path": analysis_path.name,
            "analysis_sha256": hashlib.sha256(analysis_path.read_bytes()).hexdigest(),
        },
    }


def _nao_atom_columns(molecule: Any) -> list[dict[str, Any]]:
    """Validate original atom/shell slots preserved by pinned PySCF NAO construction."""
    import numpy as np

    locations = np.asarray(molecule.ao_loc_nr())
    slices = np.asarray(molecule.aoslice_by_atom(locations))
    if (
        locations.shape != (molecule.nbas + 1,)
        or slices.shape != (molecule.natm, 4)
        or locations[0] != 0
        or locations[-1] != molecule.nao_nr()
        or not np.issubdtype(locations.dtype, np.integer)
        or not np.issubdtype(slices.dtype, np.integer)
        or np.any(np.diff(locations) <= 0)
    ):
        raise ValueError("The native spherical atom/shell partition is incomplete.")
    mapping, covered = [], []
    previous_shell = previous_column = 0
    for atom, (shell_begin, shell_end, begin, end) in enumerate(slices):
        shell_begin, shell_end, begin, end = map(
            int, (shell_begin, shell_end, begin, end)
        )
        if (
            shell_begin != previous_shell
            or begin != previous_column
            or not shell_begin < shell_end <= molecule.nbas
            or not begin < end <= molecule.nao_nr()
            or locations[shell_begin] != begin
            or locations[shell_end] != end
            or any(
                molecule.bas_atom(shell) != atom
                for shell in range(shell_begin, shell_end)
            )
        ):
            raise ValueError("The native NAO atom-column ownership is inconsistent.")
        columns = list(range(begin, end))
        covered.extend(columns)
        mapping.append(
            {
                "atom_index": atom,
                "element": molecule.atom_pure_symbol(atom),
                "columns": columns,
                "shell_indices": list(range(shell_begin, shell_end)),
            }
        )
        previous_shell, previous_column = shell_end, end
    if (
        previous_shell != molecule.nbas
        or previous_column != molecule.nao_nr()
        or covered != list(range(molecule.nao_nr()))
    ):
        raise ValueError(
            "Every native NAO column must belong to exactly one original atom."
        )
    return mapping


def _nao_transform(
    molecule: Any,
    coefficients: Any,
    occupations: Any,
    orbital_energies: Any,
    overlap: Any,
    *,
    basis: str | None = None,
) -> tuple[Any, ...]:
    """Transform an authenticated real closed-shell SCF density into the NAO basis.

    Supports arbitrary periodic-table all-electron spherical basis sets without
    element-symbol restrictions.
    """
    import numpy as np
    from pyscf import scf
    from pyscf.lo import nao

    mol_symbols = [molecule.atom_pure_symbol(atom) for atom in range(molecule.natm)]
    before = _nao_algorithm_identity(symbols=mol_symbols, basis=basis)
    count = molecule.nao_nr()
    if (
        molecule.cart is not False
        or bool(molecule.ecp)
        or bool(molecule._ecp)
        or any(molecule.atom_nelec_core(atom) != 0 for atom in range(molecule.natm))
        or any(molecule.atom_charge(atom) < 1 for atom in range(molecule.natm))
        or molecule.spin != 0
        or molecule.nelectron <= 0
        or molecule.nelectron % 2
    ):
        raise ValueError(
            "NAO requires an actual all-electron spherical closed-shell "
            "checkpoint without ghosts or ECPs."
        )
    arrays = (coefficients, occupations, orbital_energies, overlap)
    if (
        any(np.iscomplexobj(value) or not np.isfinite(value).all() for value in arrays)
        or coefficients.shape != (count, count)
        or occupations.shape != (count,)
        or orbital_energies.shape != (count,)
        or overlap.shape != (count, count)
        or not np.isin(occupations, [0.0, 2.0]).all()
        or abs(float(np.sum(occupations)) - molecule.nelectron) > 1e-8
        or not np.allclose(
            coefficients.T @ overlap @ coefficients, np.eye(count), rtol=0, atol=1e-7
        )
        or not np.allclose(overlap, overlap.T, rtol=0, atol=1e-10)
        or float(np.min(np.linalg.eigvalsh(overlap))) <= 1e-10
    ):
        raise ValueError(
            "NAO requires complete finite normalized real native RHF "
            "orbitals and AO metric."
        )
    mapping = _nao_atom_columns(molecule)
    basis_before = json.dumps(molecule._basis, sort_keys=True, allow_nan=False)
    geometry_before = np.asarray(molecule.atom_coords(unit="Bohr")).copy()
    mean_field = scf.RHF(molecule)
    mean_field.mo_coeff, mean_field.mo_occ = coefficients, occupations
    mean_field.mo_energy = orbital_energies
    transform = np.asarray(nao.nao(molecule, mean_field, s=overlap, restore=True))
    if (
        transform.shape != (count, count)
        or np.iscomplexobj(transform)
        or not np.isfinite(transform).all()
    ):
        raise ValueError(
            "The genuine PySCF NAO producer returned an incomplete "
            "or non-real transform."
        )
    metric_residual = transform.T @ overlap @ transform - np.eye(count)
    if not np.allclose(metric_residual, 0, rtol=0, atol=1e-7):
        raise ValueError(
            "The genuine NAO transform failed the existing "
            "AO metric normalization guard."
        )
    density_ao = (coefficients * occupations) @ coefficients.T
    density_nao = transform.T @ overlap @ density_ao @ overlap @ transform
    occupied = transform.T @ overlap @ coefficients[:, occupations == 2.0]
    projector_density = 2 * occupied @ occupied.T
    trace = float(np.trace(density_nao))
    if (
        not np.isfinite(density_nao).all()
        or not np.allclose(density_nao, density_nao.T, rtol=0, atol=1e-10)
        or abs(trace - molecule.nelectron) > 1e-7
        or abs(float(np.trace(density_ao @ overlap)) - molecule.nelectron) > 1e-7
    ):
        raise ValueError(
            "The real NAO density failed symmetry or electron-count validation."
        )
    bonds, populations = [], []
    for atom, entry in enumerate(mapping):
        columns = entry["columns"]
        populations.append(float(np.trace(density_nao[np.ix_(columns, columns)])))
        for other in range(atom + 1, len(mapping)):
            block = density_nao[np.ix_(columns, mapping[other]["columns"])]
            value = float(np.sum(block**2))
            if not math.isfinite(value) or value < 0:
                raise ValueError("A NAO pair descriptor is nonfinite or negative.")
            bonds.append({"atom_i": atom, "atom_j": other, "value": value})
    charge_sum = math.fsum(
        float(molecule.atom_charge(atom)) - value
        for atom, value in enumerate(populations)
    )
    if abs(charge_sum - molecule.charge) > 1e-7:
        raise ValueError(
            "The internal NAO atom partition does not conserve total charge."
        )
    after_overlap = molecule.intor_symmetric("int1e_ovlp")
    if (
        _nao_algorithm_identity(symbols=mol_symbols, basis=basis) != before
        or _nao_atom_columns(molecule) != mapping
        or json.dumps(molecule._basis, sort_keys=True, allow_nan=False) != basis_before
        or not np.array_equal(molecule.atom_coords(unit="Bohr"), geometry_before)
        or not np.array_equal(after_overlap, overlap)
    ):
        raise ValueError(
            "The actual NAO algorithm, configuration, basis or geometry "
            "changed during analysis."
        )
    eigenvalues = np.linalg.eigvalsh(density_nao)
    validation = {
        "electron_count": int(molecule.nelectron),
        "density_trace": trace,
        "charge_sum_internal": charge_sum,
        "metric_max_abs_residual": float(np.max(np.abs(metric_residual))),
        "metric_norm_residual": float(np.linalg.norm(metric_residual)),
        "upstream_orthogonality_warning_threshold": 1e-9,
        "final_metric_exceeds_upstream_warning_threshold": (
            float(np.linalg.norm(metric_residual)) > 1e-9
        ),
        "density_symmetry_max_abs_residual": float(
            np.max(np.abs(density_nao - density_nao.T))
        ),
        "idempotency_max_abs_residual_diagnostic": float(
            np.max(np.abs(density_nao @ density_nao - 2 * density_nao))
        ),
        "backtransform_max_abs_residual_diagnostic": float(
            np.max(np.abs(transform @ density_nao @ transform.T - density_ao))
        ),
        "occupied_projector_max_abs_residual_diagnostic": float(
            np.max(np.abs(density_nao - projector_density))
        ),
        "density_eigenvalue_min_diagnostic": float(np.min(eigenvalues)),
        "density_eigenvalue_max_diagnostic": float(np.max(eigenvalues)),
        "minimum_overlap_eigenvalue": float(np.min(np.linalg.eigvalsh(overlap))),
        "diagnostic_cutoffs": None,
    }
    return transform, density_ao, density_nao, mapping, bonds, validation, before


def _nao_checkpoint_binding(
    native: dict[str, Any],
    checkpoint: Path,
    molecule: Any,
    checked: dict[str, Any],
    symbols: list[str],
) -> tuple[Any, str]:
    """Bind checkpoint energy/basis to native and requested identities."""
    import numpy as np
    from pyscf import gto, lib

    data = lib.chkfile.load(str(checkpoint), "scf")
    energy = data.get("e_tot")
    native_energy = native.get("scf", {}).get("energy_hartree")
    numeric = (int, float, np.integer, np.floating)
    if (
        isinstance(energy, (bool, np.bool_))
        or not isinstance(energy, numeric)
        or not math.isfinite(float(energy))
        or isinstance(native_energy, (bool, np.bool_))
        or not isinstance(native_energy, numeric)
        or not math.isfinite(float(native_energy))
        or abs(float(energy) - float(native_energy)) > 1e-8
    ):
        raise ValueError(
            "The NAO checkpoint requires a finite real energy matching "
            "the actual native result."
        )
    basis_file = checkpoint.parent / "basis-definition.json"
    basis_digest = hashlib.sha256(basis_file.read_bytes()).hexdigest()
    retained_basis = json.loads(basis_file.read_bytes())
    requested_source = checked["method"]["basis"]
    pure_symbols = [molecule.atom_pure_symbol(atom) for atom in range(molecule.natm)]
    if re.sub(r"[^a-z0-9]", "", requested_source.lower()) == "sto3g":
        identity = _nao_algorithm_identity(symbols=pure_symbols, basis=requested_source)
        requested_source = identity["source_files"]["gto/basis/sto-3g.dat"]["path"]
    requested_basis = {
        symbol: gto.basis.load(requested_source, symbol)
        for symbol in dict.fromkeys(pure_symbols)
    }

    def canonical(value: Any) -> str:
        return json.dumps(value, sort_keys=True, allow_nan=False)

    if (
        basis_digest != native.get("basis_definition_sha256")
        or canonical(molecule._basis) != canonical(retained_basis)
        or canonical(retained_basis) != canonical(requested_basis)
    ):
        raise ValueError(
            "The actual NAO checkpoint basis differs from its native "
            "or requested basis definition."
        )
    return np.asarray(data.get("mo_energy")), basis_digest


def wiberg_nao(handoff_path: Path, directory: Path) -> dict[str, Any]:
    """Produce general-element NAO-basis Wiberg pair descriptors from genuine PySCF."""
    from cochem_torq.engines.pyscf_backend import _environment

    record, _, symbols, positions = _load_handoff(handoff_path)
    checked = _nao_options(record["options"], symbols)
    source_before = _source_digest()
    algorithm_before = _nao_algorithm_identity(
        symbols=symbols, basis=checked["method"]["basis"]
    )
    phase = directory / "nao-runtime"
    phase.mkdir()
    with _environment({"threads": checked["cores"]}, phase):
        return _wiberg_nao_checked(
            record,
            symbols,
            positions,
            checked,
            directory,
            source_before,
            algorithm_before,
        )


def _wiberg_nao_checked(
    record: dict[str, Any],
    symbols: list[str],
    positions: list[list[float]],
    checked: dict[str, Any],
    directory: Path,
    source_before: str,
    algorithm_before: dict[str, Any],
) -> dict[str, Any]:
    """Execute the admitted general-element NAO operation in its resource context."""
    import numpy as np
    from pyscf import lib
    from scipy.constants import physical_constants

    from cochem_torq.engines.diagnostics import verify_native_artifacts

    _, expected_native_method = _normalize_nao_method(checked["method"])
    pure_symbols = [_pure_element_symbol(sym) for sym in symbols]
    native, checkpoint, molecule, coefficients, occupations, overlap = (
        _restricted_scf_density(symbols, positions, checked, directory)
    )
    if (
        native.get("engine") != "PySCF"
        or native.get("engine_version") != algorithm_before["engine_version"]
        or native.get("units", {}).get("geometry") != "bohr"
        or native.get("method") != expected_native_method
        or molecule.charge != checked["charge"]
        or [molecule.atom_pure_symbol(atom) for atom in range(molecule.natm)]
        != pure_symbols
        or not np.allclose(
            molecule.atom_coords(unit="Bohr"),
            np.asarray(positions) * (1e-10 / physical_constants["Bohr radius"][0]),
            rtol=0,
            atol=1e-10,
        )
    ):
        raise ValueError(
            "The actual native NAO tuple differs from the explicitly requested profile."
        )
    checkpoint_before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    orbital_energies, basis_digest = _nao_checkpoint_binding(
        native, checkpoint, molecule, checked, symbols
    )
    transform, density_ao, density_nao, mapping, bonds, validation, algorithm = (
        _nao_transform(
            molecule,
            coefficients,
            occupations,
            orbital_energies,
            overlap,
            basis=checked["method"]["basis"],
        )
    )
    validation["requested_cpu_cores"] = checked["cores"]
    validation["pyscf_threads_during_nao"] = int(lib.num_threads())
    if validation["pyscf_threads_during_nao"] != checked["cores"]:
        raise ValueError("The actual NAO phase differs from its CPU allocation.")
    verify_native_artifacts(native)
    after_energies, after_basis = _nao_checkpoint_binding(
        native, checkpoint, molecule, checked, symbols
    )
    if (
        algorithm != algorithm_before
        or _source_digest() != source_before
        or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != checkpoint_before
        or basis_digest != after_basis
        or not np.array_equal(orbital_energies, after_energies)
    ):
        raise ValueError(
            "The complete owning NAO producer source changed during execution."
        )
    matrices = directory / "nao-density-transform.npz"
    with matrices.open("xb") as stream:
        np.savez(
            stream,
            density_ao=density_ao,
            overlap_ao=overlap,
            density_nao=density_nao,
            coefficients_nao=transform,
            coefficients_mo=coefficients,
            occupations=occupations,
            orbital_energies_hartree=orbital_energies,
        )
    analysis = {
        "schema_version": "cochem.wiberg-nao/1",
        "analysis_kind": "wiberg_nao",
        "profile": _NAO_PROFILE,
        "engine": "pyscf",
        "engine_version": algorithm["engine_version"],
        "algorithm": algorithm,
        "atoms": symbols,
        "coordinates_angstrom": positions,
        "atom_column_indices": mapping,
        "bonds": bonds,
        "method": checked["method"],
        "electronic_state": {
            "charge": checked["charge"],
            "multiplicity": checked["multiplicity"],
        },
        "definition": (
            "P_NAO=C_NAO.T S D_AO S C_NAO; W_AB=sum(P_NAO[i,j]^2), real "
            "spin-summed restricted SCF density and assigned atom-column blocks."
        ),
        "scope": (
            "General-element all-electron PySCF natural-AO Wiberg pair descriptors "
            "at the requested closed-shell HF/DFT method and basis."
        ),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "basis_definition_sha256": basis_digest,
        "native_manifest_sha256": native["manifest_sha256"],
        "matrix_sha256": hashlib.sha256(matrices.read_bytes()).hexdigest(),
        "validation": validation,
    }
    target = directory / "bond-analysis.json"
    _publish_json(target, analysis)
    return {
        "schema_version": "cochem.module-operation/1",
        "module_id": "torq",
        "operation": "wiberg_nao",
        "status": "completed",
        "input_sha256": record["artifact"]["sha256"],
        "provider_file": str(Path(__file__).resolve()),
        "provider_sha256": source_before,
        "scope": analysis["scope"],
        "result": {
            **analysis,
            "analysis_path": target.name,
            "analysis_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        },
    }


def run_wiberg_nao(
    symbols: Sequence[str],
    coordinates_angstrom: Sequence[Sequence[float]],
    *,
    method: Mapping[str, str] | None = None,
    charge: int = 0,
    multiplicity: int = 1,
    cores: int = 1,
    memory_mb: int = 1024,
    output_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Direct entrypoint for general-element PySCF NAO Wiberg bond indices."""
    sym_list = [str(s) for s in symbols]
    coord_list = [[float(v) for v in row] for row in coordinates_angstrom]
    if len(sym_list) != len(coord_list):
        raise ValueError("symbols and coordinates_angstrom must have equal length.")
    xyz_lines = [str(len(sym_list)), "CoChem-TORQ general-element wiberg_nao input"]
    for sym, pos in zip(sym_list, coord_list, strict=True):
        xyz_lines.append(f"{sym} {pos[0]:.12f} {pos[1]:.12f} {pos[2]:.12f}")
    xyz_text = "\n".join(xyz_lines) + "\n"
    opts = {
        "engine": "pyscf",
        "method": dict(_NAO_DEFAULT_METHOD if method is None else method),
        "charge": charge,
        "multiplicity": multiplicity,
        "cores": cores,
        "memory_mb": memory_mb,
    }

    def _run(root: Path) -> dict[str, Any]:
        in_dir = root / "input"
        out_dir = root / "output"
        in_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        xyz_file = in_dir / "molecule.xyz"
        xyz_file.write_text(xyz_text, encoding="utf-8")
        handoff_record = {
            "schema_version": "cochem.module-handoff/1",
            "module_id": "torq",
            "operation": "wiberg_nao",
            "scientific_execution_performed": False,
            "options": opts,
            "artifact": {
                "kind": "geometry_xyz",
                "filename": xyz_file.name,
                "size_bytes": xyz_file.stat().st_size,
                "sha256": _sha256_file(xyz_file),
            },
        }
        handoff_file = in_dir / "handoff.json"
        handoff_file.write_text(
            json.dumps(handoff_record, allow_nan=False), encoding="utf-8"
        )
        return execute_handoff(handoff_path=handoff_file, output_directory=out_dir)

    if output_directory is not None:
        return _run(Path(output_directory).expanduser().resolve())
    with tempfile.TemporaryDirectory(prefix="torq_wiberg_nao_") as tmp_dir:
        return _run(Path(tmp_dir))


def nbo_analysis(handoff_path: Path, directory: Path) -> dict[str, Any]:
    """Execute authentic NBO 7 analysis for a BASE handoff artifact."""
    record, artifact, symbols, positions = _load_handoff(handoff_path)
    options = record["options"]
    allowed = {
        "engine",
        "method",
        "charge",
        "multiplicity",
        "cores",
        "memory_mb",
        "nbo_keywords",
        "archive_47_path",
        "timeout_seconds",
    }
    if not isinstance(options, dict) or set(options) - allowed:
        raise ValueError("Unsupported NBO analysis option.")

    engine_choice = options.get("engine")
    if engine_choice is not None and engine_choice not in {"orca", "gennbo", "nbo7"}:
        raise ValueError(
            f"Unsupported NBO engine {engine_choice!r}; expected 'orca', 'gennbo', or 'nbo7'."
        )
    archive_47 = options.get("archive_47_path")
    method_opt = options.get("method", dict(_NBO_DEFAULT_METHOD))
    charge = options.get("charge", 0)
    multiplicity = options.get("multiplicity", 1)
    cores = options.get("cores", 1)
    memory_mb = options.get("memory_mb", 1024)
    nbo_keywords = options.get("nbo_keywords", _NBO_DEFAULT_KEYWORDS)
    timeout_seconds = float(options.get("timeout_seconds", 300.0))

    run_dir = directory / "nbo7-native"
    run_dir.mkdir(parents=True, exist_ok=True)
    result = run_nbo7_analysis(
        symbols=None if archive_47 is not None else symbols,
        coordinates_angstrom=None if archive_47 is not None else positions,
        archive_47_path=archive_47,
        method=method_opt,
        charge=charge,
        multiplicity=multiplicity,
        cores=cores,
        memory_mb=memory_mb,
        nbo_keywords=nbo_keywords,
        work_dir=run_dir,
        timeout_seconds=timeout_seconds,
    )
    target = directory / "nbo-analysis.json"
    _publish_json(target, result)
    provider_path = Path(__file__).resolve()
    return {
        "schema_version": "cochem.module-operation/1",
        "module_id": "torq",
        "operation": "nbo_analysis",
        "status": "completed",
        "input_sha256": record["artifact"]["sha256"],
        "provider_file": str(provider_path),
        "provider_sha256": _source_digest(),
        "scope": (
            "Authentic NBO 7 Natural Population Analysis charges, NAO Wiberg "
            "bond indices, and E(2) donor-acceptor perturbation energies."
        ),
        "result": {
            **result,
            "analysis_path": target.name,
            "analysis_sha256": _sha256_file(target),
        },
    }


def execute_handoff(
    *,
    handoff_path: str | Path,
    output_directory: str | Path,
    registry: str | Path | None = None,
) -> dict[str, Any]:
    """Execute the exact immutable BASE request in an externally bounded worker.

    BASE owns process lifetime/cancellation and total resource authority. This
    provider enforces molecular, point, thread and memory bounds; it never reads
    credentials or treats the optional BASE registry as electronic-engine proof.
    """
    path = Path(handoff_path)
    record, artifact, _, _ = _load_handoff(path)
    handoff_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    source_digest = _source_digest()
    directory = Path(output_directory)
    if directory.is_symlink() or directory.resolve() == path.parent.resolve():
        raise ValueError("Scientific outputs require a separate nonsymlink workspace.")
    directory.mkdir(parents=True, exist_ok=True)
    operation = record["operation"]
    metadata = scientific_metadata()
    definition = metadata["scientific_apis"][operation]
    if not definition["available"]:
        payload = {
            "schema_version": "cochem.module-operation/1",
            "module_id": "torq",
            "operation": operation,
            "input_sha256": record["artifact"]["sha256"],
            "status": "unavailable",
            "scope": definition["scope"],
            "reason": definition["reason"],
            "result": None,
        }
    elif operation == "geometry_analysis":
        if record.get("options") != {}:
            raise ValueError(
                "Supplied-geometry analysis accepts no scientific options."
            )
        payload = analyze(artifact)
    elif operation == "research_scan":
        payload = research_scan(path, directory)
    elif operation == "wiberg_lowdin":
        payload = wiberg_lowdin(path, directory)
    elif operation == "nbo_analysis":
        payload = nbo_analysis(path, directory)
    elif operation == "wiberg_nao":
        payload = wiberg_nao(path, directory)
    else:
        raise ValueError("No qualified scientific dispatch exists for this operation.")
    if (
        _load_handoff(path)[0] != record
        or hashlib.sha256(path.read_bytes()).hexdigest() != handoff_digest
        or _source_digest() != source_digest
    ):
        raise ValueError(
            "Scientific input or provider source changed during execution."
        )
    payload.update(
        {
            "provider_file": str(Path(__file__).resolve()),
            "provider_sha256": source_digest,
            "provider_distribution": "CoChem-TORQ",
            "provider_version": _provider_version(),
            "handoff_sha256": handoff_digest,
        }
    )
    json.dumps(payload, allow_nan=False)
    return payload
