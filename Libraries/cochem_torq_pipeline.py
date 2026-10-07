"""
CoChem-TORQ: End-to-End Orchestration Pipeline
Refine supplied geometries and preserve explicitly typed spectroscopy results.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
from mendeleev import element as mendeleev_element
from mendeleev import isotope as mendeleev_isotope

from Libraries.torq_config import TorqRunParams

ARTIFACTS_DIR = os.environ.get("COCHEM_ARTIFACTS_DIR", str(Path.home() / "cochem_artifacts"))

logger = logging.getLogger("TorqPipeline")


def normalize_and_validate_payload(
    geometry_payload: dict[str, Any]
) -> tuple[list[str], np.ndarray]:
    """
    Normalizes geometric dictionary aliases and executes strict dimension and Mendeleev validation.

    Supported aliases:
    - Symbols: 'symbols', 'atoms', 'elements'
    - Coordinates: 'coordinates', 'coords', 'geometry', 'positions'

    :param geometry_payload: Input structural dictionary.
    :return: Tuple of (validated_symbols, (N, 3) float64 coordinate array).
    :raises ValueError: If payload is missing, malformed, dimensionally inconsistent, or contains invalid elements.
    """
    if not isinstance(geometry_payload, dict) or not geometry_payload:
        raise ValueError("[MISSING DATA] Pipeline requires a non-empty geometry payload dict.")

    for field in ("units", "coordinate_units", "coordinates_units"):
        if field in geometry_payload and geometry_payload[field] not in ("angstrom", "angstroms", "Angstrom", "Å"):
            raise ValueError(f"[INVALID_PAYLOAD] {field} must explicitly be angstrom; unsupported units cannot be relabeled.")

    # 1. Resolve atomic symbols alias
    raw_symbols = None
    for key in ("symbols", "atoms", "elements"):
        if key in geometry_payload:
            raw_symbols = geometry_payload[key]
            break

    if (
        raw_symbols is None
        or not isinstance(raw_symbols, (list, tuple, np.ndarray))
        or len(raw_symbols) == 0
    ):
        raise ValueError(
            "[INVALID_PAYLOAD] Geometry payload must contain a non-empty 'symbols' or 'atoms' sequence."
        )

    # 2. Resolve coordinates alias
    raw_coords = None
    for key in ("coordinates", "coords", "geometry", "positions"):
        if key in geometry_payload:
            raw_coords = geometry_payload[key]
            break

    if raw_coords is None:
        raise ValueError(
            "[INVALID_PAYLOAD] Geometry payload must contain 'coordinates' or 'coords'."
        )

    # 3. Strict dimension validation: (N, 3) float64
    try:
        coords = np.asarray(raw_coords, dtype=np.float64)
    except Exception as e:
        raise ValueError(f"[INVALID_PAYLOAD] Failed to parse coordinates as float64 array: {e}")

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(
            f"[INVALID_PAYLOAD] Coordinates must have shape (N, 3), got {coords.shape}."
        )

    n_atoms = coords.shape[0]
    if n_atoms == 0:
        raise ValueError("[INVALID_PAYLOAD] Coordinate array cannot be empty.")

    if len(raw_symbols) != n_atoms:
        raise ValueError(
            f"[INVALID_PAYLOAD] Length of symbols ({len(raw_symbols)}) does not match coordinates count ({n_atoms})."
        )

    if not np.all(np.isfinite(coords)):
        raise ValueError("[INVALID_PAYLOAD] Coordinates must all be finite.")

    # 4. Atomic and isotopic symbol validation via Mendeleev library
    validated_symbols: list[str] = []
    for s in raw_symbols:
        if not isinstance(s, str) or not s.strip():
            raise ValueError(f"[INVALID_SYMBOL] Invalid atomic symbol: '{s}'")
        sym_clean = s.strip()

        # Handle Deuterium and Tritium aliases
        if sym_clean in ("D", "2H", "T", "3H"):
            validated_symbols.append(sym_clean)
            continue

        # Check for isotopic notation e.g. '13C', '18O', '15N'
        match = re.match(r"^(\d+)([A-Za-z]+)$", sym_clean)
        if match:
            mass_num = int(match.group(1))
            elem_sym = match.group(2).capitalize()
            try:
                iso = mendeleev_isotope(elem_sym, mass_num)
                if iso is None or not hasattr(iso, "mass"):
                    raise ValueError(f"[INVALID_SYMBOL] '{sym_clean}' is not a recognized chemical isotope.")
                validated_symbols.append(sym_clean)
                continue
            except Exception as err:
                raise ValueError(f"[INVALID_SYMBOL] Mendeleev isotope lookup failed for '{sym_clean}': {err}")

        sym_cap = sym_clean.capitalize()
        try:
            elem = mendeleev_element(sym_cap)
            if elem is None or not hasattr(elem, "atomic_number"):
                raise ValueError(f"[INVALID_SYMBOL] '{sym_clean}' is not a recognized chemical element.")
            validated_symbols.append(sym_cap)
        except Exception as err:
            raise ValueError(f"[INVALID_SYMBOL] Mendeleev element lookup failed for '{sym_clean}': {err}")

    return validated_symbols, coords


def resolve_concurrency_device() -> str:
    """
    Non-blocking lock-free device acquisition across CUDA / MPS / CPU.
    Automatically reserves 1 CPU P-core for background I/O operations (TORQ_RESERVED_PCORES=1).
    Honors CUDA_VISIBLE_DEVICES or falls back safely to CPU if unavailable.
    """
    os.environ.setdefault("TORQ_RESERVED_PCORES", "1")

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if cuda_visible is not None and (cuda_visible.strip() == "" or cuda_visible.strip() == "-1"):
        return "cpu"

    try:
        import torch
        if torch.cuda.is_available():
            dev_idx = torch.cuda.current_device()
            return f"cuda:{dev_idx}"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass

    return "cpu"


def verify_nvidia_mps_health() -> bool:
    """
    Queries NVIDIA MPS daemon health under Method Matrix v4 §8A.4:
    - Verifies active daemon process via psutil.
    - Checks named pipe sockets in Ring 2 scratch ($COCHEM_SCRATCH/nvidia_mps).
    """
    import psutil
    mps_running = False
    try:
        for proc in psutil.process_iter(["name"]):
            pname = (proc.info.get("name") or "").lower()
            if any(term in pname for term in ["nvidia-cuda-mps", "mps-control", "mps-server"]):
                mps_running = True
                break
    except Exception:
        pass

    try:
        scratch = Path(os.environ.get("COCHEM_SCRATCH", tempfile.gettempdir()))
        mps_dir = scratch / "nvidia_mps"
        if mps_dir.exists():
            mps_running = True
    except Exception:
        pass

    return mps_running


def deduplicate_conformer_union(
    conformers: list[dict[str, Any]],
    delta_b_rel_threshold: float = 0.005,
    rmsd_threshold: float = 0.15,
) -> list[dict[str, Any]]:
    """
    Deduplicates a union pool of conformers (e.g. from GOAT and CREST).
    Groups conformers by rotational constant similarity (Delta B / B <= delta_b_rel_threshold)
    and pairwise heavy-atom/all-atom RMSD (<= rmsd_threshold).
    Pool is sorted by energy (ascending, Hartree) and filtered against accepted conformers.
    """
    if not conformers:
        return []

    from Libraries.cochem_torq_goat import deduplicate_conformers

    return deduplicate_conformers(
        conformers=conformers,
        delta_rot_rel_threshold=delta_b_rel_threshold,
        rmsd_threshold=rmsd_threshold,
    )


class TorqPipeline:
    """
    Refines a supplied geometry with explicit engine dispatch, then exports a
    typed spectroscopy report with independent availability for every stage.
    """

    def __init__(self, config: TorqRunParams) -> None:
        """
        Initialize the TORQ execution pipeline with configuration parameters.

        :param config: TorqRunParams validating method tier, convergence, and basis sets.
        """
        self.config: TorqRunParams = config
        self.state: str = "S_0"
        self.state_history: list[str] = [self.state]
        logger.info(
            f"Initialized TorqPipeline for tier '{self.config.tier}' "
            f"({self.config.method}/{self.config.basis_set}) in state '{self.state}'."
        )

    def _transition_state(self, new_state: str) -> None:
        """Transitions state machine and records audit trail."""
        self.state = new_state
        self.state_history.append(new_state)
        logger.info(f"Transitioned pipeline state to: {self.state}")

    def run(self, geometry_payload: dict[str, Any]) -> dict[str, Any]:
        """Refine a supplied geometry and report each spectroscopy stage honestly.

        ``requested_product`` defaults to ``catalog``. A supported earlier product
        can be requested explicitly; unavailable later stages remain in the report.
        Discovery and ML filtering require independent validated workflows and are
        not silently applied to a supplied structure.
        """
        try:
            symbols, coordinates = normalize_and_validate_payload(geometry_payload)
            requested_product = geometry_payload.get("requested_product", "catalog")
            if requested_product not in ("equilibrium_constants", "harmonic_analysis", "ground_state_constants", "catalog"):
                raise ValueError(f"Unsupported requested spectroscopy product: {requested_product}")
            profile = geometry_payload.get("workflow_profile", "refine_geometry")
            if profile != "refine_geometry":
                raise NotImplementedError(f"Workflow profile {profile!r} has no validated pipeline adapter.")
            engine = self.config.engine.strip().upper()
            tier = self.config.tier.upper().replace("-", "")
            if engine != "ORCA" or "CFOUR" in self.config.method.upper() or tier.startswith(("T3C", "T4C", "CFOUR")):
                raise NotImplementedError(
                    f"Requested engine/tier {self.config.engine}/{self.config.tier} is not implemented by this pipeline; "
                    "the ORCA adapter cannot execute a different engine's job."
                )
            if self.config.bsse_correction:
                raise NotImplementedError(
                    "This pipeline does not implement a CP-corrected optimization or discrete fragment CP calculations; "
                    "the requested BSSE correction cannot be silently omitted."
                )
            charge = geometry_payload.get("charge", 0)
            multiplicity = geometry_payload.get("multiplicity", geometry_payload.get("spin", 1))
            if type(charge) is not int or type(multiplicity) is not int or multiplicity < 1:
                raise ValueError("Charge and multiplicity must be integers; multiplicity must be positive.")
            from Libraries.cochem_torq_topology import TorqTopology, get_atomic_number
            electrons = sum(get_atomic_number(symbol) for symbol in symbols) - charge
            if electrons < 0 or multiplicity - 1 > electrons or (electrons - multiplicity + 1) % 2:
                raise ValueError("Charge, atom identities and multiplicity imply an inconsistent electron count.")

            from Libraries.cochem_torq_engine import (
                ExecutionContext, detect_complex_and_monomers,
                route_cascade_rules, opi_persistent_threading,
            )
            from Libraries.cochem_torq_spectroscopy import build_spectroscopy_report

            self._transition_state("S_TOPOLOGY")
            is_complex, components = detect_complex_and_monomers(symbols, coordinates)
            topology = TorqTopology(symbols=symbols, coordinates=coordinates, is_complex=is_complex)
            input_coordinates = coordinates.copy()
            context = ExecutionContext()
            resources = geometry_payload.get("resources", {})
            if resources:
                for name in ("num_cores", "max_memory_mb"):
                    value = resources.get(name)
                    if value is not None:
                        if type(value) is not int or value <= 0:
                            raise ValueError(f"{name} must be a positive integer.")
                        setattr(context, name, min(value, getattr(context, name)))
                if resources.get("scratch_dir"):
                    context.custom_scratch_dir = Path(resources["scratch_dir"])

            self._transition_state("S_QUANTUM_OPT")
            keyword_list = list(self.config.keywords)
            if self.config.dispersion:
                dispersion = self.config.dispersion.strip()
                if dispersion.upper() not in self.config.method.upper() and dispersion.upper() not in [k.upper() for k in keyword_list]:
                    keyword_list.append(dispersion)
            extra_parts: list[str] = []
            initial_hessian = "Lindh"
            for keyword in keyword_list:
                clean = keyword.strip()
                if "inhess" in clean.lower():
                    match = re.fullmatch(r"InHess\s+([A-Za-z0-9_]+)", clean, re.IGNORECASE)
                    if match is None:
                        raise ValueError(f"Unsupported Hessian keyword syntax: {clean!r}")
                    initial_hessian = match.group(1)
                else:
                    extra_parts.append(clean if clean.startswith(("!", "%")) else f"! {clean}")
            if self.config.anharmonicity:
                # Engine output for this request can be retained, but the typed
                # VPT2 stage remains unavailable until a validated parser exists.
                option = self.config.anharmonicity.strip()
                extra_parts.append(option if option.startswith(("!", "%")) else f"! {option}")
            dispatch = route_cascade_rules(
                point_coords=coordinates, context=context, symbols=symbols,
                charge=charge, multiplicity=multiplicity, method=self.config.method,
                basis_set=self.config.basis_set, is_complex=is_complex,
                initial_hessian=initial_hessian,
                counterpoise=False,
                extra_options="\n".join(extra_parts),
            )
            if self.config.cabs_mappings:
                raise NotImplementedError("CABS mappings are not compiled by the current ORCA pipeline adapter.")
            engine_results = list(opi_persistent_threading(input_payload=dispatch, context=context, n_steps=1))
            if len(engine_results) != 1:
                raise RuntimeError("A single requested engine calculation did not return exactly one result.")
            engine_result = engine_results[0]
            self._transition_state("S_SPECTROSCOPY_VALIDATION")
            report = build_spectroscopy_report(
                engine_result, symbols, engine="ORCA", method=dispatch.method, basis_set=dispatch.basis_set,
            )
            report_data = report.model_dump(mode="json")
            status = "success" if report.product_available(requested_product) else "partial"
            geometry_stage = report.equilibrium_geometry
            optimized = geometry_stage.status == "available"
            final_coordinates = np.asarray(geometry_stage.value.coordinates_angstrom) if optimized else input_coordinates
            constants = report.equilibrium_constants.value
            rotational_constants = ({"A": constants.A_mhz, "B": constants.B_mhz, "C": constants.C_mhz}
                                    if constants is not None else None)
            self._transition_state("S_EXPORT")
            export_metadata = {
                "tier": self.config.tier, "requested_product": requested_product,
                "method": dispatch.method, "basis_set": dispatch.basis_set,
                "engine": "ORCA", "charge": charge, "multiplicity": multiplicity,
                "symbols": symbols, "coordinates": final_coordinates.tolist(),
                "geometry_role": "optimized" if optimized else "input",
                "rotational_constants_label": "Be" if constants else None,
                "rotational_constants": rotational_constants, "status": status,
                "spectroscopy": report_data, "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            export_dir = context.get_scratch_dir() / "exports"
            export_dir.mkdir(parents=True, exist_ok=True)
            (export_dir / "pipeline_export_manifest.json").write_text(
                json.dumps(export_metadata, indent=2, allow_nan=False), encoding="utf-8")
            self._transition_state("S_COMPLETE" if status == "success" else "S_PARTIAL")
            return {
                "status": status, "requested_product": requested_product,
                "processed_payload": geometry_payload, "state_history": self.state_history,
                "normalized_geometry": {
                    "symbols": symbols, "coordinates": final_coordinates.tolist(),
                    "input_coordinates": input_coordinates.tolist(),
                    "geometry_role": "optimized" if optimized else "input",
                    "charge": charge, "multiplicity": multiplicity, "is_complex": is_complex,
                    "topology_nodes": topology.graph.number_of_nodes(),
                    "topology_edges": topology.graph.number_of_edges(),
                },
                "mlff_results": {"status": "not_requested"},
                "conformer_results": {"status": "not_requested", "profile": "refine_geometry"},
                "engine_results": engine_results,
                "spectral_results": {
                    "status": status, "rotational_constants_label": "Be" if constants else None,
                    "rotational_constants_mhz": rotational_constants, "stages": report_data,
                },
                "export_metadata": export_metadata,
            }
        except Exception as exc:
            self._transition_state("S_FAILED")
            logger.error("Pipeline execution failed: %s", exc)
            raise

    def run_active_learning_pes_sampling(
        self,
        grid_geometries: list[dict[str, Any]],
        anchor_evaluator: Optional[Any] = None,
        threshold_sigma_mev: float = 10.0,
    ) -> list[dict[str, Any]]:
        """
        Execute active learning committee sampling and Delta-ML surface correction across a PES grid (§13.2, Suggestion #158).
        Only configurations where committee epistemic uncertainty sigma > threshold_sigma_mev (10.0 meV)
        trigger expensive high-level anchor evaluations ([M]); other points are interpolated via Delta-ML ([E]).
        """
        from Libraries.cochem_torq_active_learning import ActiveLearningSampler
        sampler = ActiveLearningSampler(threshold_sigma_mev=threshold_sigma_mev)
        return sampler.sample_pes_grid(grid_geometries, anchor_evaluator=anchor_evaluator)


try:
    from cochem_base.schemas import TorqPipelineCliArgs
except ImportError:
    from pydantic import BaseModel, Field, ConfigDict
    class TorqPipelineCliArgs(BaseModel):
        model_config = ConfigDict(frozen=True, extra="allow")
        input_geometry: Path
        output_directory: Path
        theory_level: str = "B3LYP-D4/def2-TZVP"
        cpus_per_task: int = Field(default=1, ge=1)
        memory_mb: int = Field(default=4096, ge=1024)
        scratch_dir: Path
        device: str = "cpu"
        task_id: Optional[int] = None
        mode: str = "full"


def parse_cli_args(args_list: Optional[list[str]] = None) -> TorqPipelineCliArgs:
    """Parses command-line arguments and dynamically maps SLURM HPC environment parameters."""
    import argparse
    parser = argparse.ArgumentParser(description="CoChem-TORQ HPC Batch Pipeline Driver")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input molecular geometry")
    parser.add_argument("--output", "--output-dir", "-o", dest="output", type=str, required=True, help="Destination directory for artifacts")
    parser.add_argument("--theory", "-t", type=str, default="B3LYP-D4/def2-TZVP", help="Electronic structure method")
    parser.add_argument("--scratch", "--scratch-dir", "-s", dest="scratch", type=str, default=None, help="Ephemeral scratch directory")
    parser.add_argument("--device", "-d", type=str, default="cpu", help="Compute device (cpu/cuda/gpu)")
    parser.add_argument("--task-id", type=int, default=None, help="Task ID or SLURM array index")
    parser.add_argument("--mode", "-m", type=str, default="full", help="Pipeline execution mode")
    parsed = parser.parse_args(args_list)

    # Dynamically read SLURM environment variables when available (Method Matrix §8A.6 & Suggestion #70)
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    mem_mb = int(os.environ.get("SLURM_MEM_PER_NODE", 4096))
    task_id = parsed.task_id if parsed.task_id is not None else (
        int(os.environ["SLURM_ARRAY_TASK_ID"]) if "SLURM_ARRAY_TASK_ID" in os.environ else None
    )

    scratch_path = Path(parsed.scratch) if parsed.scratch else (
        Path(os.environ.get("COCHEM_SCRATCH", os.environ.get("TMPDIR", Path.home() / "cochem_scratch")))
    )

    return TorqPipelineCliArgs(
        input_geometry=Path(parsed.input).resolve(),
        output_directory=Path(parsed.output).resolve(),
        theory_level=parsed.theory,
        cpus_per_task=cpus,
        memory_mb=mem_mb,
        scratch_dir=scratch_path.resolve(),
        device=parsed.device,
        task_id=task_id,
        mode=parsed.mode,
    )


def execute_cli_pipeline(cli_args: TorqPipelineCliArgs) -> dict[str, Any]:
    """Executes the complete TorqPipeline using validated CLI arguments and enforces Tripartite Air-Gaps."""
    if cli_args.device.lower() != "cpu":
        raise NotImplementedError("The CLI ORCA adapter has no validated GPU/device execution contract.")
    if not cli_args.input_geometry.exists():
        raise FileNotFoundError(f"Input geometry file not found: {cli_args.input_geometry}")

    # Enforce Tripartite Storage Ring Air-Gaps
    cochem_root = os.environ.get("COCHEM_ROOT")
    if cochem_root:
        root_path = Path(cochem_root).resolve()
        if root_path in cli_args.scratch_dir.parents or root_path == cli_args.scratch_dir:
            raise RuntimeError(f"[AIR-GAP VIOLATION] Scratch cannot write into Ring 1 root: {cli_args.scratch_dir}")
        if root_path in cli_args.output_directory.parents or root_path == cli_args.output_directory:
            raise RuntimeError(f"[AIR-GAP VIOLATION] Output cannot write into Ring 1 root: {cli_args.output_directory}")

    cli_args.scratch_dir.mkdir(parents=True, exist_ok=True)
    cli_args.output_directory.mkdir(parents=True, exist_ok=True)

    # XYZ line 2 is a comment and may be blank. Do not filter it out.
    lines = cli_args.input_geometry.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise ValueError(f"Malformed XYZ file: {cli_args.input_geometry}")
    try:
        n_atoms = int(lines[0])
    except ValueError as exc:
        raise ValueError("First line of XYZ must be an integer atom count.") from exc
    if n_atoms <= 0 or len(lines) < n_atoms + 2 or any(line.strip() for line in lines[n_atoms + 2:]):
        raise ValueError("XYZ atom count must exactly match one nonempty geometry.")
    symbols, coords = [], []
    for line in lines[2:n_atoms + 2]:
        parts = line.split()
        if len(parts) != 4:
            raise ValueError("Each XYZ atom row must contain a symbol and three coordinates.")
        symbols.append(parts[0])
        coords.append([float(value) for value in parts[1:]])
    symbols, coordinate_array = normalize_and_validate_payload({"symbols": symbols, "coordinates": coords})
    if "/" not in cli_args.theory_level:
        raise ValueError("Specify the complete method/basis pair; a basis is never supplied silently.")
    method, basis = (part.strip() for part in cli_args.theory_level.split("/", 1))
    if not method or not basis:
        raise ValueError("Method and basis must both be specified.")
    products = {"full": "catalog", "equilibrium": "equilibrium_constants", "harmonic": "harmonic_analysis"}
    if cli_args.mode not in products:
        raise ValueError(f"Unsupported CLI mode: {cli_args.mode}; choose full, equilibrium or harmonic.")
    run_params = TorqRunParams(
        tier="T1", wall_time_tier="T1-30min", engine="ORCA", method=method, basis_set=basis,
        keywords=["Opt", "TightOpt", "InHess Lindh"] + (["Freq"] if cli_args.mode != "equilibrium" else []),
    )
    pipeline = TorqPipeline(config=run_params)
    payload = {
        "symbols": symbols, "coordinates": coordinate_array.tolist(),
        "requested_product": products[cli_args.mode],
        "resources": {"num_cores": cli_args.cpus_per_task, "max_memory_mb": cli_args.memory_mb,
                      "scratch_dir": str(cli_args.scratch_dir)},
    }
    try:
        results = pipeline.run(payload)
    except Exception as exc:
        logger.error("Quantum pipeline failed: %s", exc)
        results = {
            "status": "blocked" if isinstance(exc, (NotImplementedError, FileNotFoundError)) else "failed",
            "theory_level": cli_args.theory_level, "symbols": symbols,
            "initial_coordinates": coords, "error": str(exc), "error_type": type(exc).__name__,
            "state_history": pipeline.state_history,
        }
    # A unique artifact name prevents a failed rerun from appearing to produce
    # an earlier run's optimized geometry in the same output directory.
    normalized = results.get("normalized_geometry", {})
    optimized = normalized.get("geometry_role") == "optimized"
    role = "optimized" if optimized else "input"
    structure_path = cli_args.output_directory / f"{role}_structure_{uuid.uuid4().hex}.xyz"
    structure_coordinates = normalized["coordinates"] if optimized else coords
    comment = (f"Optimization converged: {cli_args.theory_level}; spectroscopy status={results['status']}"
               if optimized else f"Input geometry; no converged optimization exported; status={results['status']}")
    with structure_path.open("w", encoding="utf-8") as handle:
        handle.write(f"{len(symbols)}\n{comment}\n")
        for symbol, position in zip(symbols, structure_coordinates):
            handle.write(f"{symbol:<3} {position[0]:14.8f} {position[1]:14.8f} {position[2]:14.8f}\n")
    results["structure_artifact"] = {"path": str(structure_path), "geometry_role": role}

    def serialize(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, bytes):
            return {"sha256": hashlib.sha256(value).hexdigest(), "size_bytes": len(value)}
        if hasattr(value, "model_dump"):
            return value.model_dump()
        raise TypeError(f"Unsupported result value type: {type(value).__name__}")

    results_path = cli_args.output_directory / "pipeline_results.json"
    results_path.write_text(json.dumps(results, indent=2, default=serialize, allow_nan=False), encoding="utf-8")
    logger.info("Pipeline status=%s; results written to %s", results["status"], results_path)
    return results


def pipeline_exit_code(status: str) -> int:
    """SRS §14.1: complete=0, invalid=2, blocked=3, partial=4, failed=5, cancelled=130."""
    return {"success": 0, "invalid": 2, "blocked": 3, "partial": 4,
            "rejected": 4, "failed": 5, "cancelled": 130}.get(status.lower(), 5)


if __name__ == "__main__":
    try:
        cli_args = parse_cli_args()
        results = execute_cli_pipeline(cli_args)
        sys.exit(pipeline_exit_code(results["status"]))
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(2 if isinstance(e, ValueError) else 3 if isinstance(e, (NotImplementedError, FileNotFoundError)) else 5)
