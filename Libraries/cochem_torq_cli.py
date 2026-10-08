#!/usr/bin/env python3
"""
CoChem-TORQ: Command-Line Interface (CLI) & Headless Pipeline Entrypoint
=======================================================================
Mandated by Readme.md Section 7.3, pyproject.toml, and Method Matrix v4
(§4.4, §8A, §8B, §9A, §9B, §10, §13).

Provides a production-grade, authentic, zero-stub command-line interface for direct terminal
execution, scripted workflow chaining, and automated headless HPC batch execution
(SLURM / NVIDIA MPS).

Authoritative Standards & Directives:
- Readme.md Section 7.3: CLI Entrypoint (cochem-torq --config ... --type ... --hess-diag ...)
- Method Matrix v4 §4.4: Dynamic Grid Cascades (defgrid1 -> defgrid3) & TolMaxG (1e-5 Eh/a0)
- Method Matrix v4 §8B.3: Initial Hessian Preconditioning (InHess XTB2 / Lindh; Calc_Hess true forbidden)
- Method Matrix v4 §8B.4: Zero-Cost Isotopologue Re-analysis
- Method Matrix v4 §9A.1-§9A.2: Frozen-Monomer Spatial Protections for Weak Complexes
- Method Matrix v4 §9B.1-§9B.2: Conformer Search (ORCA GOAT + CREST Union Referee)
- Mendeleev Library Mandate: Dynamic atomic and isotopic masses via mendeleev
- Anti-Spoofing Protocols v2: Physical execution, air-gap enforcement, zero mocks / stubs.

Usage Examples:
    # Direct execution per Readme.md Section 7.3:
    cochem-torq --config cochem_system_config.json --type TS --hess-diag Lindh

    # Full geometry optimization with explicit parameters:
    cochem-torq --input molecule.xyz --type OPT --tier t3 --method wB97X-D4 --basis def2-TZVP

    # Hardware schema and air-gap integrity audit:
    cochem-torq audit --json

    # Dynamic Mendeleev atomic and isotopic mass query:
    cochem-torq mass 13C

    # Ephemeral scratch and IPC memory purge:
    cochem-torq clean
"""

from __future__ import annotations

import argparse
import atexit
import json
import logging
import os
import platform
import re
import signal
import sys
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Reconfigure stream encodings for safe cross-platform terminal output (prevent Windows cp1252 crash)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(errors="replace")
    except Exception:
        pass

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Optional Mendeleev integration
try:
    import mendeleev
    from mendeleev import element as mendeleev_element
except ImportError:
    mendeleev = None  # type: ignore[assignment]
    mendeleev_element = None  # type: ignore[assignment]

# Core CoChem-TORQ imports
from Libraries.cochem_path_manager import CoChemPathManager
from Libraries.cochem_torq_init import (
    AirGapViolationError,
    cleanup_ipc_scratch,
    get_artifact_directory,
    verify_airgap,
)
from Libraries.cochem_torq_pipeline import (
    TorqPipeline,
    normalize_and_validate_payload,
)
from Libraries.cochem_torq_schema import (
    TorqSchemaValidationError,
    format_5_whys_error,
    validate_registry_state,
)
from Libraries.cochem_torq_vault import (
    get_atomic_mass,
    get_atomic_number,
    parse_external_mol,
    parse_external_xyz,
)
from Libraries.torq_config import TorqRunParams

# Setup module-level logger
logger = logging.getLogger("CoChem-TORQ.CLI")


# =============================================================================
# 1. Terminal ANSI Styling & Cross-Platform Encoding
# =============================================================================


def _can_encode_unicode() -> bool:
    """Checks whether the current stdout encoding supports unicode symbols."""
    try:
        encoding = sys.stdout.encoding or "ascii"
        "✅".encode(encoding)
        return True
    except Exception:
        return False


class TermColor:
    """Terminal ANSI escape styling with automated TTY and charset detection."""

    _USE_COLOR: bool = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
    _UNICODE: bool = _can_encode_unicode()

    RESET = "\033[0m" if _USE_COLOR else ""
    BOLD = "\033[1m" if _USE_COLOR else ""
    DIM = "\033[2m" if _USE_COLOR else ""
    RED = "\033[31m" if _USE_COLOR else ""
    GREEN = "\033[32m" if _USE_COLOR else ""
    YELLOW = "\033[33m" if _USE_COLOR else ""
    BLUE = "\033[34m" if _USE_COLOR else ""
    MAGENTA = "\033[35m" if _USE_COLOR else ""
    CYAN = "\033[36m" if _USE_COLOR else ""
    WHITE = "\033[37m" if _USE_COLOR else ""

    @classmethod
    def ok(cls, text: str) -> str:
        symbol = "✅ " if cls._UNICODE else "[OK] "
        return f"{cls.GREEN}{symbol}{text}{cls.RESET}"

    @classmethod
    def fail(cls, text: str) -> str:
        symbol = "❌ " if cls._UNICODE else "[FAIL] "
        return f"{cls.RED}{symbol}{text}{cls.RESET}"

    @classmethod
    def warn(cls, text: str) -> str:
        symbol = "⚠️  " if cls._UNICODE else "[WARN] "
        return f"{cls.YELLOW}{symbol}{text}{cls.RESET}"

    @classmethod
    def info(cls, text: str) -> str:
        symbol = "ℹ️  " if cls._UNICODE else "[INFO] "
        return f"{cls.CYAN}{symbol}{text}{cls.RESET}"

    @classmethod
    def title(cls, text: str) -> str:
        return f"{cls.BOLD}{cls.MAGENTA}{text}{cls.RESET}"


# =============================================================================
# 2. Resource Cleanup & Signal Handlers
# =============================================================================


def _handle_shutdown_signal(signum: int, frame: Any) -> None:
    """Graceful signal trap ensuring IPC buffers and scratch files are reclaimed."""
    sig_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
    sys.stderr.write(f"\n[INTERRUPT] Received signal {sig_name}. Reclaiming IPC resources...\n")
    cleanup_ipc_scratch()
    sys.exit(128 + signum)


try:
    signal.signal(signal.SIGINT, _handle_shutdown_signal)
    signal.signal(signal.SIGTERM, _handle_shutdown_signal)
except (ValueError, AttributeError):
    pass

atexit.register(cleanup_ipc_scratch)


# =============================================================================
# 3. Geometry Loading & Normalization Helpers
# =============================================================================


def load_input_geometry(
    input_source: str | Path | None = None,
    config_dict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolves and loads molecular geometry payload from files, inline strings, or configuration.

    Supports:
    - Cartesian XYZ files (.xyz)
    - MDL Molfile (.mol)
    - JSON geometry payload dictionaries
    - Config file embedded geometry payloads

    :param input_source: Path to geometry file, inline XYZ string, or None.
    :param config_dict: Parsed configuration dictionary possibly containing geometry.
    :return: Normalized geometry dictionary containing 'symbols', 'coordinates', 'charge', 'multiplicity'.
    :raises ValueError: If no valid geometry could be found or parsed.
    """
    # Case 1: Explicit input source specified
    if input_source is not None:
        clean_input = input_source
        if isinstance(clean_input, str) and "\\n" in clean_input:
            clean_input = clean_input.replace("\\n", "\n")

        source_path: Path | None = None
        is_file = False
        try:
            source_path = Path(clean_input)
            is_file = source_path.is_file() and source_path.exists()
        except Exception:
            is_file = False

        if is_file and source_path is not None:
            suffix = source_path.suffix.lower()
            if suffix == ".xyz":
                parsed = parse_external_xyz(source_path)
                return {
                    "symbols": parsed["symbols"],
                    "coordinates": parsed["coordinates"].tolist()
                    if isinstance(parsed["coordinates"], np.ndarray)
                    else parsed["coordinates"],
                    "charge": parsed.get("charge", 0),
                    "multiplicity": parsed.get("multiplicity", 1),
                }
            elif suffix in (".mol", ".sdf", ".mdl"):
                parsed = parse_external_mol(source_path)
                return {
                    "symbols": parsed["symbols"],
                    "coordinates": parsed["coordinates"].tolist()
                    if isinstance(parsed["coordinates"], np.ndarray)
                    else parsed["coordinates"],
                    "charge": parsed.get("charge", 0),
                    "multiplicity": parsed.get("multiplicity", 1),
                }
            elif suffix in (".json", ".jsn"):
                content = json.loads(source_path.read_text(encoding="utf-8"))
                if isinstance(content, dict):
                    syms, coords = normalize_and_validate_payload(content)
                    return {
                        "symbols": syms,
                        "coordinates": coords.tolist(),
                        "charge": int(content.get("charge", 0)),
                        "multiplicity": int(content.get("multiplicity", content.get("spin", 1))),
                    }
            else:
                # Attempt general text parse (e.g. XYZ format with non-standard extension)
                raw_text = source_path.read_text(encoding="utf-8")
                parsed = parse_external_xyz(raw_text)
                return {
                    "symbols": parsed["symbols"],
                    "coordinates": parsed["coordinates"].tolist()
                    if isinstance(parsed["coordinates"], np.ndarray)
                    else parsed["coordinates"],
                    "charge": parsed.get("charge", 0),
                    "multiplicity": parsed.get("multiplicity", 1),
                }
        elif isinstance(clean_input, str) and ("\n" in clean_input or " " in clean_input.strip()):
            # Inline raw XYZ string
            parsed = parse_external_xyz(clean_input)
            return {
                "symbols": parsed["symbols"],
                "coordinates": parsed["coordinates"].tolist()
                if isinstance(parsed["coordinates"], np.ndarray)
                else parsed["coordinates"],
                "charge": parsed.get("charge", 0),
                "multiplicity": parsed.get("multiplicity", 1),
            }
        else:
            raise FileNotFoundError(f"Input geometry file not found: {input_source}")

    # Case 2: Embedded geometry inside configuration dictionary
    if config_dict is not None:
        for geom_key in ("geometry", "molecule", "structure", "payload", "coordinates"):
            if geom_key in config_dict and isinstance(config_dict[geom_key], dict):
                sub_dict = config_dict[geom_key]
                try:
                    syms, coords = normalize_and_validate_payload(sub_dict)
                    return {
                        "symbols": syms,
                        "coordinates": coords.tolist(),
                        "charge": int(sub_dict.get("charge", config_dict.get("charge", 0))),
                        "multiplicity": int(
                            sub_dict.get("multiplicity", config_dict.get("multiplicity", 1))
                        ),
                    }
                except ValueError:
                    pass

        # Check top-level symbols/coords in config
        if any(k in config_dict for k in ("symbols", "atoms", "elements")) and any(
            k in config_dict for k in ("coordinates", "coords", "positions")
        ):
            syms, coords = normalize_and_validate_payload(config_dict)
            return {
                "symbols": syms,
                "coordinates": coords.tolist(),
                "charge": int(config_dict.get("charge", 0)),
                "multiplicity": int(config_dict.get("multiplicity", config_dict.get("spin", 1))),
            }

    # Case 3: Search Dynamic Artifact landing zone for recent upload
    art_dir = get_artifact_directory(create_if_missing=False)
    uploads_dir = art_dir / "Input_Files" / "uploads"
    if uploads_dir.exists() and uploads_dir.is_dir():
        xyz_files = sorted(uploads_dir.glob("*.xyz"), key=lambda f: f.stat().st_mtime, reverse=True)
        if xyz_files:
            latest_xyz = xyz_files[0]
            logger.info(f"Auto-selected latest uploaded geometry: {latest_xyz}")
            parsed = parse_external_xyz(latest_xyz)
            return {
                "symbols": parsed["symbols"],
                "coordinates": parsed["coordinates"].tolist()
                if isinstance(parsed["coordinates"], np.ndarray)
                else parsed["coordinates"],
                "charge": parsed.get("charge", 0),
                "multiplicity": parsed.get("multiplicity", 1),
            }

    raise ValueError(
        "[MISSING_GEOMETRY] No molecular geometry provided. Specify --input <file.xyz> "
        "or define 'symbols' and 'coordinates' in the --config JSON file."
    )


# =============================================================================
# 4. CLI Actions Implementation
# =============================================================================


def action_run(args: argparse.Namespace) -> int:
    """Executes the CoChem-TORQ pipeline conforming to Method Matrix v4 and Readme.md Section 7.3."""
    t0 = time.perf_counter()
    artifacts_dir = (
        Path(args.output_dir).resolve() if args.output_dir else get_artifact_directory()
    )

    # 1. Enforce Air-Gap Boundary
    try:
        verify_airgap(cwd=Path.cwd(), artifacts_dir=artifacts_dir)
    except AirGapViolationError as airgap_err:
        logger.error(f"Air-gap validation failed: {airgap_err}")
        if args.json:
            print(
                json.dumps(
                    {
                        "status": "FAILED",
                        "error_type": "AirGapViolationError",
                        "message": str(airgap_err),
                    },
                    indent=2,
                )
            )
        else:
            print(TermColor.fail(f"Air-Gap Violation: {airgap_err}"))
        return 1

    # 2. Resolve and Validate System Config
    config_dict: dict[str, Any] = {}
    config_path: Path | None = None
    if args.config:
        config_path = Path(args.config).resolve()
        if not config_path.exists():
            # Check relative to Registry subdirectory in artifacts
            reg_cand = artifacts_dir / "Registry" / args.config
            if reg_cand.exists():
                config_path = reg_cand
            else:
                logger.error(f"Configuration file not found: {args.config}")
                if args.json:
                    print(
                        json.dumps(
                            {
                                "status": "FAILED",
                                "error_type": "FileNotFoundError",
                                "message": f"Config not found at {args.config}",
                            },
                            indent=2,
                        )
                    )
                else:
                    print(TermColor.fail(f"Configuration file not found: {args.config}"))
                return 1

        try:
            with open(config_path, encoding="utf-8") as fp:
                config_dict = json.load(fp)
        except Exception as json_err:
            logger.error(f"Failed to read JSON config {config_path}: {json_err}")
            if args.json:
                print(
                    json.dumps(
                        {
                            "status": "FAILED",
                            "error_type": "JSONDecodeError",
                            "message": str(json_err),
                        },
                        indent=2,
                    )
                )
            else:
                print(TermColor.fail(f"Invalid JSON in config {config_path}: {json_err}"))
            return 1

        if not isinstance(config_dict, dict):
            print(json.dumps({
                "status": "FAILED", "error_type": "InvalidConfiguration",
                "message": "Configuration must be a JSON object.",
            }) if args.json else TermColor.fail("Configuration must be a JSON object."))
            return 1
        if (
            config_dict.get("schema_version") == "cochem.torq.legacy-cascade-planning/1"
            or config_dict.get("status") == "planning_only"
            or config_dict.get("dispatch_authorized") is False
        ):
            message = (
                "This document is a non-executable plan or explicitly prohibits dispatch. "
                "Select a supported, separately reviewed runtime recipe."
            )
            print(json.dumps({
                "status": "FAILED", "error_type": "DispatchNotAuthorized",
                "message": message,
            }) if args.json else TermColor.fail(message))
            return 1

        # Validate hardware schema if this is cochem_system_config.json
        if "mpi_threads" in config_dict or "maxcore_mb" in config_dict:
            try:
                validate_registry_state(config_dict)
            except TorqSchemaValidationError as schema_err:
                logger.error(f"Hardware schema validation failed: {schema_err}")
                if args.json:
                    print(
                        json.dumps(
                            {
                                "status": "FAILED",
                                "error_type": "TorqSchemaValidationError",
                                "message": str(schema_err),
                                "trace": schema_err.five_whys_trace,
                            },
                            indent=2,
                        )
                    )
                else:
                    print(TermColor.fail(f"Hardware Schema Error: {schema_err}"))
                    if schema_err.five_whys_trace:
                        print(schema_err.five_whys_trace)
                return 1

    # 3. Resolve Execution Parameters (CLI flags take precedence over config file)
    calc_type = (args.type or config_dict.get("type", "OPT")).upper()
    hess_diag = args.hess_diag or config_dict.get("hess_diag", "Lindh")
    tier = args.tier or config_dict.get("tier", "t1")
    method = args.method or config_dict.get("method", "B3LYP")
    basis_set = args.basis_set or config_dict.get("basis_set", "def2-SVP")
    wall_time_tier = args.wall_time_tier or config_dict.get("wall_time_tier", "normal")
    engine = args.engine or config_dict.get("engine", "orca")
    dispersion = args.dispersion or config_dict.get("dispersion", None)
    bsse = args.bsse or config_dict.get("bsse_correction", None)
    anharmonicity = args.anharmonicity or config_dict.get("anharmonicity", None)
    charge = args.charge if args.charge is not None else config_dict.get("charge", 0)
    multiplicity = (
        args.multiplicity
        if args.multiplicity is not None
        else config_dict.get("multiplicity", 1)
    )

    # 4. Construct Quantum Engine Keywords
    keywords = list(args.keywords) if args.keywords else list(config_dict.get("keywords", []))

    # Configure optimization keywords based on calculation type
    if calc_type == "TS":
        if not any("opt" in kw.lower() for kw in keywords):
            keywords.append("OptTS")
        if not any("freq" in kw.lower() for kw in keywords):
            keywords.append("Freq")
    elif calc_type in ("OPT", "MIN"):
        if not any("opt" in kw.lower() for kw in keywords):
            keywords.append("Opt")
    elif calc_type == "SCAN":
        if not any("opt" in kw.lower() for kw in keywords):
            keywords.append("Opt")

    # Configure Hessian preconditioning keyword per Method Matrix §8B.3
    # Enforces InHess XTB2 / Lindh (Calc_Hess true strictly forbidden)
    if hess_diag:
        clean_hess = hess_diag.strip()
        inhess_kw = f"InHess {clean_hess}" if not clean_hess.lower().startswith("inhess") else clean_hess
        if not any("inhess" in kw.lower() for kw in keywords):
            keywords.append(inhess_kw)

    # 5. Build and Validate TorqRunParams
    try:
        run_params = TorqRunParams(
            tier=tier,
            wall_time_tier=wall_time_tier,
            engine=engine,
            method=method,
            basis_set=basis_set,
            keywords=keywords,
            anharmonicity=anharmonicity,
            dispersion=dispersion,
            bsse_correction=bsse,
            cabs_mappings=config_dict.get("cabs_mappings", None),
        )
    except Exception as param_err:
        logger.error(f"Parameter validation failed: {param_err}")
        if args.json:
            print(
                json.dumps(
                    {
                        "status": "FAILED",
                        "error_type": "ValidationError",
                        "message": str(param_err),
                    },
                    indent=2,
                )
            )
        else:
            print(TermColor.fail(f"Configuration Parameter Error: {param_err}"))
        return 1

    # 6. Load Molecular Geometry Payload
    try:
        geometry_payload = load_input_geometry(
            input_source=args.input, config_dict=config_dict
        )
        geometry_payload["charge"] = charge
        geometry_payload["multiplicity"] = multiplicity
    except Exception as geom_err:
        logger.error(f"Geometry intake failed: {geom_err}")
        if args.json:
            print(
                json.dumps(
                    {
                        "status": "FAILED",
                        "error_type": type(geom_err).__name__,
                        "message": str(geom_err),
                    },
                    indent=2,
                )
            )
        else:
            print(TermColor.fail(f"Geometry Intake Error: {geom_err}"))
        return 1

    # 7. Print Execution Header in Interactive Terminal Mode
    if not args.json and not args.quiet:
        print(TermColor.title("=" * 78))
        print(TermColor.title(" CoChem-TORQ: Torsional Optimization & Rotational Quantification "))
        print(TermColor.title(" Method Matrix v4 Compliant Execution Pipeline "))
        print(TermColor.title("=" * 78))
        print(f"Target Type:          {TermColor.BOLD}{calc_type}{TermColor.RESET}")
        print(f"Hessian Precondition: {TermColor.BOLD}{hess_diag}{TermColor.RESET}")
        print(f"Method / Basis:       {TermColor.BOLD}{method} / {basis_set}{TermColor.RESET}")
        print(f"Method Tier:          {tier} (Wall-time: {wall_time_tier})")
        print(f"Engine Backend:       {engine}")
        print(f"Keywords:             {', '.join(keywords)}")
        print(f"Atoms Count:          {len(geometry_payload.get('symbols', []))}")
        print(f"Artifact Store:       {artifacts_dir}")
        if args.dry_run:
            print(TermColor.info("DRY-RUN MODE ACTIVE: Preflight validation only."))
        print("-" * 78)

    # 8. Dry-Run Fast Exit Mode
    if args.dry_run:
        elapsed = time.perf_counter() - t0
        syms, coords = normalize_and_validate_payload(geometry_payload)
        dry_run_summary = {
            "status": "PASSED",
            "mode": "DRY_RUN",
            "calculation_type": calc_type,
            "hessian_preconditioner": hess_diag,
            "run_params": run_params.model_dump(),
            "atoms_count": len(syms),
            "symbols": syms,
            "airgap_verified": True,
            "execution_time_sec": round(elapsed, 4),
        }
        if args.json:
            print(json.dumps(dry_run_summary, indent=2))
        else:
            print(TermColor.ok(f"Dry-run preflight validation completed successfully in {elapsed:.3f}s."))
            print("=" * 78)
        return 0

    # 9. Execute Full End-to-End Pipeline
    try:
        pipeline = TorqPipeline(config=run_params)
        result = pipeline.run(geometry_payload=geometry_payload)
        elapsed = time.perf_counter() - t0
        result["execution_time_sec"] = round(elapsed, 3)

        if args.json:
            # Serialize cleanly to stdout
            serializable_result = {
                "status": result.get("status", "failed"),
                "calculation_type": calc_type,
                "hessian_preconditioner": hess_diag,
                "tier": tier,
                "method": method,
                "basis_set": basis_set,
                "state_history": result.get("state_history", []),
                "normalized_geometry": result.get("normalized_geometry", {}),
                "mlff_results": result.get("mlff_results", {}),
                "conformer_results": result.get("conformer_results", {}),
                "spectral_results": result.get("spectral_results", {}),
                "export_metadata": result.get("export_metadata", {}),
                "execution_time_sec": round(elapsed, 3),
            }
            print(json.dumps(serializable_result, indent=2))
        else:
            spec = result.get("spectral_results", {})
            rot = spec.get("rotational_constants_mhz", {})
            print(f"CoChem-TORQ status: {result.get('status', 'failed')} ({elapsed:.2f}s)")
            print("\n" + TermColor.BOLD + "Spectroscopic Observables Summary:" + TermColor.RESET)
            print(f"  Point Group:       {spec.get('point_group') or 'unavailable'}")
            print(f"  Symmetry Number σ: {spec.get('sigma') if spec.get('sigma') is not None else 'unavailable'}")
            if rot:
                print(f"  Constants label:   {spec.get('rotational_constants_label', 'unavailable')}")
                for axis in ("A", "B", "C"):
                    value = rot.get(axis)
                    print(f"  {axis}: {value:.4f} MHz" if value is not None else f"  {axis}: unavailable/undefined")
            print(f"\nFinal State:         {TermColor.BOLD}{pipeline.state}{TermColor.RESET}")
            print(f"Deliverables Vault:  {artifacts_dir / 'Processed' / 'Deliverables'}")
            print("=" * 78)

        from Libraries.cochem_torq_pipeline import pipeline_exit_code
        return pipeline_exit_code(result.get("status", "failed"))

    except Exception as run_err:
        elapsed = time.perf_counter() - t0
        logger.error(f"Pipeline execution halted: {run_err}")
        context = {
            "field": "execution_pipeline",
            "value": f"type={calc_type}, hess_diag={hess_diag}, method={method}",
            "rule": "Method Matrix v4 Execution Constraints",
            "origin": "TorqPipeline.run()",
            "remediation": "Check quantum engine binaries, scratch space, and convergence criteria.",
        }
        trace = format_5_whys_error(run_err, context)

        if args.json:
            print(
                json.dumps(
                    {
                        "status": "FAILED",
                        "error_type": type(run_err).__name__,
                        "message": str(run_err),
                        "five_whys_trace": trace,
                        "execution_time_sec": round(elapsed, 3),
                    },
                    indent=2,
                )
            )
        else:
            print(TermColor.fail(f"Pipeline Execution Failed: {run_err}"))
            print(trace)
            print("=" * 78)

        return 2 if isinstance(run_err, ValueError) else 3 if isinstance(run_err, (NotImplementedError, FileNotFoundError)) else 5


def action_audit(args: argparse.Namespace) -> int:
    """Executes non-mutating environment, hardware schema, air-gap, and engine audit."""
    t0 = time.perf_counter()
    artifacts_dir = (
        Path(args.output_dir).resolve() if args.output_dir else get_artifact_directory(create_if_missing=False)
    )

    if not args.json:
        print(TermColor.title("=" * 78))
        print(TermColor.title(" CoChem-TORQ: Environment & Hardware Architecture Audit "))
        print(TermColor.title("=" * 78))

    audit_results: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "detected_tier": CoChemPathManager.detect_tier().value,
        },
        "airgap_audit": {},
        "schema_audit": {},
        "mendeleev_audit": {},
        "overall_status": "PASSED",
    }

    all_passed = True

    # 1. Air-gap verification
    try:
        verify_airgap(cwd=Path.cwd(), artifacts_dir=artifacts_dir)
        audit_results["airgap_audit"] = {
            "status": "PASSED",
            "cwd": str(Path.cwd()),
            "artifacts_dir": str(artifacts_dir),
            "airgap_disjoint": True,
        }
        if not args.json:
            print(TermColor.ok("Filesystem Air-Gap: Verified (Domain A and Domain B are strictly disjoint)."))
    except Exception as e:
        all_passed = False
        audit_results["airgap_audit"] = {"status": "FAILED", "error": str(e)}
        if not args.json:
            print(TermColor.fail(f"Filesystem Air-Gap: Failed ({e})"))

    # 2. Hardware Schema Audit
    config_file = artifacts_dir / "Registry" / "cochem_system_config.json"
    if config_file.exists():
        try:
            schema = validate_registry_state(config_file)
            audit_results["schema_audit"] = {
                "status": "PASSED",
                "mpi_threads": schema.mpi_threads,
                "maxcore_mb": schema.maxcore_mb,
                "gpu_vram_gb": schema.gpu_vram_gb,
                "cuda_enabled": schema.cuda_enabled,
            }
            if not args.json:
                print(
                    TermColor.ok(
                        f"Hardware Schema: Validated (Threads={schema.mpi_threads}, "
                        f"MaxCore={schema.maxcore_mb}MB, VRAM={schema.gpu_vram_gb}GB)."
                    )
                )
        except Exception as e:
            all_passed = False
            audit_results["schema_audit"] = {"status": "FAILED", "error": str(e)}
            if not args.json:
                print(TermColor.fail(f"Hardware Schema: Failed ({e})"))
    else:
        audit_results["schema_audit"] = {
            "status": "SKIPPED",
            "message": f"cochem_system_config.json not found at {config_file}",
        }
        if not args.json:
            print(TermColor.info(f"Hardware Schema: Skipped (No registry file at {config_file})."))

    # 3. Mendeleev Mandate Audit
    if mendeleev is not None:
        try:
            mass_13c = get_atomic_mass("13C")
            z_c = get_atomic_number("C")
            audit_results["mendeleev_audit"] = {
                "status": "PASSED",
                "mendeleev_installed": True,
                "sample_13C_mass": mass_13c,
                "sample_C_z": z_c,
            }
            if not args.json:
                print(
                    TermColor.ok(
                        f"Mendeleev Authority: Active (Dynamic CIAAW 13C mass={mass_13c:.8f} u, Z={z_c})."
                    )
                )
        except Exception as e:
            all_passed = False
            audit_results["mendeleev_audit"] = {"status": "FAILED", "error": str(e)}
            if not args.json:
                print(TermColor.fail(f"Mendeleev Authority: Failed ({e})"))
    else:
        all_passed = False
        audit_results["mendeleev_audit"] = {
            "status": "FAILED",
            "error": "mendeleev library not installed",
        }
        if not args.json:
            print(TermColor.fail("Mendeleev Authority: Failed (mendeleev library not installed)."))

    elapsed = time.perf_counter() - t0
    audit_results["execution_time_sec"] = round(elapsed, 4)
    audit_results["overall_status"] = "PASSED" if all_passed else "FAILED"

    if args.json:
        print(json.dumps(audit_results, indent=2))
    else:
        print("-" * 78)
        if all_passed:
            print(TermColor.ok(f"Ecosystem Audit PASSED ({elapsed:.3f}s). All physical invariants satisfied."))
        else:
            print(TermColor.fail(f"Ecosystem Audit FAILED ({elapsed:.3f}s). Correct issues above."))
        print("=" * 78)

    return 0 if all_passed else 1


def action_clean(args: argparse.Namespace) -> int:
    """Purges ephemeral scratch buffers, temporary memory maps, and reaps zombies."""
    t0 = time.perf_counter()
    reclaimed = cleanup_ipc_scratch()
    elapsed = time.perf_counter() - t0

    payload = {
        "status": "SUCCESS",
        "reclaimed_resources_count": reclaimed,
        "execution_time_sec": round(elapsed, 4),
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        total_count = sum(reclaimed.values()) if isinstance(reclaimed, dict) else reclaimed
        print(TermColor.title("=" * 60))
        print(TermColor.title(" CoChem-TORQ: Workspace Scratch & IPC Cleaner "))
        print(TermColor.title("=" * 60))
        print(TermColor.ok(f"Cleaned {total_count} ephemeral scratch resources and memory maps."))
        if isinstance(reclaimed, dict):
            for k, v in reclaimed.items():
                if v > 0:
                    print(f"  - {k.replace('_', ' ').title()}: {v}")
        print(f"Execution Time: {elapsed:.4f}s")
        print("=" * 60)

    return 0


def action_mass(args: argparse.Namespace) -> int:
    """Queries dynamic elemental and isotopic masses via mendeleev adhering to the Mendeleev Mandate."""
    symbol = args.symbol.strip()
    if not symbol:
        logger.error("Element/isotope symbol required.")
        return 1

    if mendeleev is None or mendeleev_element is None:
        logger.error("mendeleev library is required by the Mendeleev Library Mandate but not installed.")
        return 1

    # Extract mass number if given (e.g. 13C -> mass_num=13, elem='C')
    match = re.match(r"^(\d+)?([A-Za-z]+)$", symbol)
    if not match:
        logger.error(f"Unrecognized elemental/isotopic symbol: {symbol}")
        return 1

    iso_str, elem_str = match.groups()
    elem_str = elem_str.capitalize()

    # Handle deuterium and tritium
    if elem_str in ("D", "2h"):
        elem_str = "H"
        iso_str = "2"
    elif elem_str in ("T", "3h"):
        elem_str = "H"
        iso_str = "3"

    try:
        elem = mendeleev_element(elem_str)
        standard_mass = float(elem.mass)

        payload: dict[str, Any] = {
            "element": elem.name,
            "symbol": elem.symbol,
            "atomic_number": elem.atomic_number,
            "standard_atomic_weight": standard_mass,
            "isotopes": [],
        }

        matched_iso_mass: float | None = None
        for iso in elem.isotopes:
            iso_info = {
                "mass_number": iso.mass_number,
                "mass": float(iso.mass) if iso.mass else None,
                "abundance": float(iso.abundance) if iso.abundance is not None else None,
                "is_radioactive": bool(iso.is_radioactive),
            }
            payload["isotopes"].append(iso_info)
            if iso_str and int(iso_str) == iso.mass_number:
                matched_iso_mass = float(iso.mass) if iso.mass else None

        if iso_str:
            payload["requested_isotope"] = {
                "mass_number": int(iso_str),
                "mass": matched_iso_mass,
            }

        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(TermColor.title("=" * 60))
            print(TermColor.title(" CoChem Mendeleev Dynamic Atomic Mass Query "))
            print(TermColor.title("=" * 60))
            print(f"Element:         {elem.name} ({elem.symbol}, Z={elem.atomic_number})")
            print(f"Standard Weight: {standard_mass:.8f} u")
            if iso_str:
                iso_display = f"{matched_iso_mass:.8f} u" if matched_iso_mass else "Not Available"
                print(f"Isotope ^{iso_str}{elem.symbol}:   {iso_display}")
            print("-" * 60)
            print("Stable / Common Isotopes:")
            for iso in elem.isotopes:
                if iso.abundance and iso.abundance > 0.01:
                    print(
                        f"  ^{iso.mass_number}{elem.symbol}: {iso.mass:12.8f} u "
                        f"(Abundance: {iso.abundance:6.2f}%)"
                    )
            print("=" * 60)

        return 0

    except Exception as exc:
        logger.error(f"Mendeleev query failed for '{symbol}': {exc}")
        return 1


def action_status(args: argparse.Namespace) -> int:
    """Inspects dynamic artifacts tier, registry configuration, and HDF5 landscape database."""
    artifacts_dir = (
        Path(args.output_dir).resolve() if args.output_dir else get_artifact_directory(create_if_missing=False)
    )
    registry_file = artifacts_dir / "Registry" / "cochem_system_config.json"
    h5_db = artifacts_dir / "Databases" / "landscape.h5"

    reg_exists = registry_file.exists()
    h5_exists = h5_db.exists()

    status_data: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts_directory": str(artifacts_dir),
        "registry_file": str(registry_file),
        "registry_exists": reg_exists,
        "database_file": str(h5_db),
        "database_exists": h5_exists,
    }

    if reg_exists:
        try:
            with open(registry_file, encoding="utf-8") as fp:
                status_data["registry_content"] = json.load(fp)
        except Exception as e:
            status_data["registry_content"] = {"error": str(e)}

    if h5_exists:
        status_data["database_size_bytes"] = h5_db.stat().st_size

    if args.json:
        print(json.dumps(status_data, indent=2))
    else:
        print(TermColor.title("=" * 65))
        print(TermColor.title(" CoChem-TORQ: Ecosystem & Artifact Tier Status "))
        print(TermColor.title("=" * 65))
        print(f"Artifact Store:  {artifacts_dir}")
        print(f"System Registry: {TermColor.ok('Found') if reg_exists else TermColor.warn('Missing')} ({registry_file})")
        print(f"HDF5 Database:   {TermColor.ok('Found') if h5_exists else TermColor.info('Not Created')} ({h5_db})")
        print("=" * 65)

    return 0


# =============================================================================
# 5. CLI Parser Builder
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    """Builds and returns the master argument parser for the CoChem-TORQ CLI."""
    parser = argparse.ArgumentParser(
        prog="cochem-torq",
        description=(
            "CoChem-TORQ: Torsional Optimization & Rotational Quantification Engine\n"
            "Command-Line Interface supporting direct terminal and headless automated pipeline execution\n"
            "per Readme.md Section 7.3 and Method Matrix v4."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Method Matrix v4 & Physical Invariants:
  - Transition State & Optimization: --type TS / --type OPT / --type SCAN
  - Hessian Preconditioning: --hess-diag Lindh / XTB2 (Calc_Hess true forbidden per §8B.3)
  - Diffuse Basis Functions: Diffuse-in-base mandated (additive diffuse prohibited)
  - Weak Complex Optimization: Frozen-monomer spatial protections and TolMaxG 1e-5 Eh/a0
  - Dispersion Correction: D3/D4 mandatory for DFT on non-covalent complexes

Usage Examples:
  cochem-torq --config cochem_system_config.json --type TS --hess-diag Lindh
  cochem-torq --input molecule.xyz --type OPT --tier t3 --method wB97X-D4 --basis def2-TZVP
  cochem-torq audit --json
  cochem-torq mass 13C
  cochem-torq clean
""",
    )

    # Top-level global flags
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debug telemetry")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress non-essential output")
    parser.add_argument("--version", action="version", version="CoChem-TORQ 0.1.0 (Method Matrix v4)")

    # Readme.md Section 7.3 direct execution arguments
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Path to system configuration JSON file (e.g. cochem_system_config.json)",
    )
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        default="OPT",
        choices=["TS", "OPT", "MIN", "SCAN", "CONFORMER", "GOAT", "CREST", "PIPELINE", "DVR", "SPCAT"],
        help="Calculation / job target type (default: OPT; use TS for transition states)",
    )
    parser.add_argument(
        "-H",
        "--hess-diag",
        "--hess",
        type=str,
        default="Lindh",
        help="Initial Hessian preconditioning / diagonalization scheme (e.g. Lindh, XTB2, Model)",
    )
    parser.add_argument(
        "-i",
        "--input",
        "-g",
        "--geometry",
        type=str,
        default=None,
        help="Path to input molecular geometry (.xyz, .mol, .json) or inline XYZ string",
    )
    parser.add_argument(
        "--tier",
        type=str,
        default="t1",
        help="Method tier conforming to Method Matrix (e.g. t1, t2, t3, t4, T3-3h, T3-12h)",
    )
    parser.add_argument(
        "--wall-time-tier",
        type=str,
        default="normal",
        choices=["fast", "normal", "extended"],
        help="Wall-time constraint tier (default: normal)",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="orca",
        choices=["orca", "cfour", "mace", "xtb", "pyscf"],
        help="Quantum chemistry engine backend (default: orca)",
    )
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        default="B3LYP",
        help="Electronic structure method / functional (e.g. B3LYP, wB97X-D4, wB97M-V, r2SCAN-3c, DLPNO-CCSD(T))",
    )
    parser.add_argument(
        "-b",
        "--basis-set",
        "--basis",
        type=str,
        default="def2-SVP",
        help="Basis set (e.g. def2-SVP, def2-TZVP, def2-QZVPP, jun-cc-pVTZ)",
    )
    parser.add_argument(
        "-d",
        "--dispersion",
        type=str,
        default=None,
        help="Empirical dispersion correction (e.g. D3BJ, D4, VV10; mandatory for DFT weak complexes)",
    )
    parser.add_argument(
        "--bsse",
        type=str,
        default=None,
        help="BSSE correction method (e.g. counterpoise, cp, none; restricted to triple-zeta)",
    )
    parser.add_argument(
        "--anharmonicity",
        type=str,
        default=None,
        help="Anharmonicity treatment (e.g. VPT2, none)",
    )
    parser.add_argument(
        "--charge",
        type=int,
        default=None,
        help="Total molecular net charge (default: 0)",
    )
    parser.add_argument(
        "--multiplicity",
        "--spin",
        type=int,
        default=None,
        help="Spin multiplicity 2S+1 (default: 1)",
    )
    parser.add_argument(
        "-k",
        "--keywords",
        type=str,
        nargs="+",
        default=None,
        help="Additional quantum engine execution keywords",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Output deliverables directory (enforces air-gap boundaries)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute preflight validation and schema checks without running heavy electronic structure calculations",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in machine-readable JSON format to stdout",
    )

    # Subparsers for dedicated subcommands
    subparsers = parser.add_subparsers(
        dest="subcommand", title="Subcommands", description="Available operational actions"
    )

    # Subcommand: run
    p_run = subparsers.add_parser("run", help="Execute TORQ calculation pipeline")
    p_run.add_argument("-c", "--config", type=str, default=None, help="Path to config JSON")
    p_run.add_argument("-t", "--type", type=str, default="OPT", help="Calculation type")
    p_run.add_argument("-H", "--hess-diag", "--hess", type=str, default="Lindh", help="Hessian preconditioner")
    p_run.add_argument("-i", "--input", "-g", "--geometry", type=str, default=None, help="Input geometry")
    p_run.add_argument("--tier", type=str, default="t1", help="Method tier")
    p_run.add_argument("--wall-time-tier", type=str, default="normal", help="Wall-time tier")
    p_run.add_argument("--engine", type=str, default="orca", help="Quantum engine")
    p_run.add_argument("-m", "--method", type=str, default="B3LYP", help="Method")
    p_run.add_argument("-b", "--basis-set", "--basis", type=str, default="def2-SVP", help="Basis set")
    p_run.add_argument("-d", "--dispersion", type=str, default=None, help="Dispersion")
    p_run.add_argument("--bsse", type=str, default=None, help="BSSE correction")
    p_run.add_argument("--anharmonicity", type=str, default=None, help="Anharmonicity")
    p_run.add_argument("--charge", type=int, default=None, help="Charge")
    p_run.add_argument("--multiplicity", "--spin", type=int, default=None, help="Multiplicity")
    p_run.add_argument("-k", "--keywords", type=str, nargs="+", default=None, help="Keywords")
    p_run.add_argument("-o", "--output-dir", type=str, default=None, help="Output directory")
    p_run.add_argument("--dry-run", action="store_true", default=argparse.SUPPRESS, help="Dry run mode")
    p_run.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help="JSON output format")

    # Subcommand: audit / validate
    p_audit = subparsers.add_parser("audit", aliases=["validate"], help="Run hardware, schema, and air-gap audit")
    p_audit.add_argument("-o", "--output-dir", type=str, default=None, help="Artifact directory to test")
    p_audit.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help="Output audit results in JSON format")

    # Subcommand: clean
    p_clean = subparsers.add_parser("clean", help="Purge ephemeral scratch buffers and memory maps")
    p_clean.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help="Output cleanup results in JSON format")

    # Subcommand: mass / element
    p_mass = subparsers.add_parser("mass", aliases=["element"], help="Query dynamic atomic and isotopic masses")
    p_mass.add_argument("symbol", type=str, help="Elemental or isotopic symbol (e.g. C, 13C, 18O, D)")
    p_mass.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help="Output mass data in JSON format")

    # Subcommand: status / info
    p_status = subparsers.add_parser("status", aliases=["info"], help="Query artifact store and database status")
    p_status.add_argument("-o", "--output-dir", type=str, default=None, help="Artifact directory to inspect")
    p_status.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help="Output status in JSON format")

    return parser


# =============================================================================
# 6. Main Entrypoint
# =============================================================================


def main(argv: Sequence[str] | None = None) -> int:
    """Master entrypoint function for the CoChem-TORQ CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Configure logging
    if getattr(args, "verbose", False):
        logging.getLogger().setLevel(logging.DEBUG)
    elif getattr(args, "quiet", False):
        logging.getLogger().setLevel(logging.WARNING)

    subcommand = args.subcommand

    # If explicit subcommand provided, dispatch to handler
    if subcommand in ("run", None):
        # Direct CLI execution per Section 7.3 or explicit 'run' subcommand
        return action_run(args)
    elif subcommand in ("audit", "validate"):
        return action_audit(args)
    elif subcommand == "clean":
        return action_clean(args)
    elif subcommand in ("mass", "element"):
        return action_mass(args)
    elif subcommand in ("status", "info"):
        return action_status(args)
    else:
        logger.error(f"Unrecognized subcommand: {subcommand}")
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
