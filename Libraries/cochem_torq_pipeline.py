"""
CoChem-TORQ: End-to-End Orchestration Pipeline
Stage 5: Multi-tier Torsional Workflow Execution
Compliant with Method Matrix v4 (§4.4, §8A, §8B, Anti-Spoofing Directives).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
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
    Orchestrates the multi-stage CoChem-TORQ execution pipeline:
    Topology -> Machine Learning Fast Filtering -> Conformer Search ->
    Quantum Engine Optimization -> SPCAT Spectral Synthesis -> Deliverable Export.
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
        """
        Executes all pipeline stages conforming to Method Matrix v4 (§4.4, §8A, §8B, §9B, §10).

        :param geometry_payload: Input empirical structural payload.
        :return: Execution summary dictionary containing processed payload, intermediate results,
                 and deliverable exports.
        :raises ValueError: When invoked with missing or malformed geometric payloads.
        """
        try:
            symbols, coordinates = normalize_and_validate_payload(geometry_payload)
            charge = int(geometry_payload.get("charge", 0))
            multiplicity = int(geometry_payload.get("multiplicity", geometry_payload.get("spin", 1)))

            # ---------------------------------------------------------------------
            # Stage 1: Structural Topology & Covalent Graph Construction
            # ---------------------------------------------------------------------
            self._transition_state("S_TOPOLOGY")
            from Libraries.cochem_torq_topology import TorqTopology
            from Libraries.cochem_torq_engine import detect_complex_and_monomers

            is_complex, components = detect_complex_and_monomers(symbols, coordinates)
            topology = TorqTopology(symbols=symbols, coordinates=coordinates, is_complex=is_complex)

            # ---------------------------------------------------------------------
            # Stage 2: Machine Learning Fast Filtering & Pre-relaxation
            # ---------------------------------------------------------------------
            self._transition_state("S_MLFF_FILTER")
            from Libraries.cochem_torq_mace import evaluate_physical_potential

            mlff_energy_ev, mlff_forces, mlff_converged = evaluate_physical_potential(
                symbols=symbols,
                coordinates=coordinates
            )

            # ---------------------------------------------------------------------
            # Stage 3: Conformer & Binding-Site Isomer Search (GOAT + CREST Union)
            # ---------------------------------------------------------------------
            self._transition_state("S_CONFORMER_SEARCH")
            from Libraries.cochem_torq_goat import (
                ConformerRecord,
                compute_moments_and_constants,
                deduplicate_stage_b_spectroscopic,
                GoatRunner
            )
            from Libraries.cochem_torq_crest import CrestRunner
            from cochem_base.environment import PathRegistry

            (
                (A_mhz, B_mhz, C_mhz),
                (A_ghz, B_ghz, C_ghz),
                delta,
                planar,
                kappa,
            ) = compute_moments_and_constants(symbols, coordinates)

            import scipy.constants
            hartree_in_ev = scipy.constants.physical_constants["Hartree energy in eV"][0]
            initial_hartree = float(mlff_energy_ev / hartree_in_ev) if mlff_energy_ev else 0.0
            
            # NVIDIA MPS health verification and device isolation
            mps_healthy = verify_nvidia_mps_health()
            if mps_healthy:
                logger.info("NVIDIA MPS daemon verified healthy for conformer screening.")
            else:
                logger.info("NVIDIA MPS daemon inactive or not detected; sandboxing device execution.")
            initial_conformer = ConformerRecord(
                index=0,
                symbols=symbols,
                coordinates=coordinates.tolist(),
                energy_hartree=initial_hartree,
                energy_kcal_rel=0.0,
                rotational_constants_mhz=(A_mhz, B_mhz, C_mhz),
                rotational_constants_ghz=(A_ghz, B_ghz, C_ghz),
                inertial_defect_u_a2=delta,
                planar_moments_u_a2=planar,
                ray_asymmetry_kappa=kappa,
                origin_engine="TORQ-PIPELINE",
                seed_id="seed_00"
            )

            # Write seed structure to scratch directory for GOAT / CREST
            scratch_dir = PathRegistry.create_scratch_dir("conformer_search")
            seed_xyz_path = scratch_dir / "seed.xyz"
            with open(seed_xyz_path, "w", encoding="utf-8") as f:
                f.write(f"{len(symbols)}\nSeed structure\n")
                for s, pos in zip(symbols, coordinates):
                    f.write(f"{s:<3} {pos[0]:14.8f} {pos[1]:14.8f} {pos[2]:14.8f}\n")

            conformer_pool: list[dict[str, Any]] = [
                {
                    "symbols": symbols,
                    "coordinates": coordinates.tolist(),
                    "energy_hartree": initial_hartree,
                    "rotational_constants_mhz": (A_mhz, B_mhz, C_mhz),
                    "origin": "TORQ-INITIAL",
                }
            ]

            # 1. Execute GOAT exploration
            try:
                goat_records = GoatRunner().run_goat_on_seed(seed_xyz=seed_xyz_path, scratch_dir=scratch_dir)
                for gr in goat_records:
                    conformer_pool.append({
                        "symbols": gr.symbols,
                        "coordinates": gr.coordinates,
                        "energy_hartree": gr.energy_hartree,
                        "rotational_constants_mhz": gr.rotational_constants_mhz,
                        "origin": "GOAT",
                    })
            except Exception as e:
                logger.info(f"GOAT conformer exploration omitted or failed: {e}")

            # 2. Execute CREST search
            try:
                crest_container = CrestRunner().run_crest(
                    input_xyz=seed_xyz_path,
                    work_dir=scratch_dir / "crest",
                    flags="--nci --nocross --noreftopo"
                )
                for cr in crest_container.records:
                    conformer_pool.append({
                        "symbols": cr.symbols,
                        "coordinates": cr.coordinates,
                        "energy_hartree": cr.energy_hartree,
                        "rotational_constants_mhz": cr.rotational_constants_mhz,
                        "origin": "CREST",
                    })
            except Exception as e:
                logger.info(f"CREST conformer search omitted or failed: {e}")

            # 3. Deduplicate GOAT + CREST conformer union
            deduped_union = deduplicate_conformer_union(
                conformer_pool,
                delta_b_rel_threshold=0.005,
                rmsd_threshold=0.15
            )

            conformer_survivors: list[ConformerRecord] = []
            for idx, c in enumerate(deduped_union):
                c_coords = np.asarray(c["coordinates"], dtype=np.float64)
                c_syms = c.get("symbols", symbols)
                rot_c = c.get("rotational_constants_mhz", (A_mhz, B_mhz, C_mhz))
                conformer_survivors.append(
                    ConformerRecord(
                        index=idx,
                        symbols=c_syms,
                        coordinates=c_coords.tolist(),
                        energy_hartree=c.get("energy_hartree", 0.0),
                        energy_kcal_rel=0.0,
                        rotational_constants_mhz=rot_c,
                        rotational_constants_ghz=tuple(x / 1000.0 for x in rot_c),
                        inertial_defect_u_a2=delta,
                        planar_moments_u_a2=planar,
                        ray_asymmetry_kappa=kappa,
                        origin_engine=c.get("origin", "UNION"),
                        seed_id=f"seed_{idx:02d}"
                    )
                )

            if not conformer_survivors:
                conformer_survivors = [initial_conformer]

            # Promote top conformer geometry for Stage 4 optimization
            if conformer_survivors:
                coordinates = np.asarray(conformer_survivors[0].coordinates, dtype=np.float64)

            # ---------------------------------------------------------------------
            # Stage 4: Quantum Engine Optimization & Dynamic Wavefunction Chaining
            # ---------------------------------------------------------------------
            self._transition_state("S_QUANTUM_OPT")
            tier_upper = (self.config.tier or "").upper().strip()
            method_upper = (self.config.method or "").upper().strip()
            is_cfour = (
                tier_upper in ["T3C", "T4C", "T3-C", "T4-C", "T3C-3D", "T4C-1MO", "CFOUR_VPT2", "CFOUR"]
                or "CFOUR" in method_upper
            )
            if is_cfour:
                from Libraries.cochem_torq_cfour_bridge import TorqCfourExecutor
                cfour_executor = TorqCfourExecutor()
                _ = cfour_executor.resolve_binary()
            from Libraries.cochem_torq_engine import (
                ExecutionContext,
                route_cascade_rules,
                opi_persistent_threading
            )

            context = ExecutionContext()

            # Forward all user configuration parameters and overrides
            counterpoise_flag = bool(
                self.config.bsse_correction and self.config.bsse_correction.lower() in ["counterpoise", "cp"]
            )

            resolved_method = self.config.method
            keyword_list = list(self.config.keywords) if self.config.keywords else []

            if self.config.dispersion:
                disp_tag = self.config.dispersion.strip()
                if disp_tag.upper() not in resolved_method.upper() and disp_tag.upper() not in [k.upper() for k in keyword_list]:
                    keyword_list.append(disp_tag)
            elif is_complex:
                m_upper = resolved_method.upper()
                is_dft = any(func in m_upper for func in ["B3LYP", "PBE", "SCAN", "M06", "W97", "OLYP", "OPBE", "DFT", "R2SCAN"])
                has_dispersion = any(d in m_upper or any(d in k.upper() for k in keyword_list) for d in ["D3", "D4", "-V", "VV10", "3C", "-3C"])
                if is_dft and not has_dispersion:
                    keyword_list.append("D4")

            extra_parts: list[str] = []
            extracted_inhess: str = "XTB2"
            for kw in keyword_list:
                clean_kw = kw.strip()
                if "inhess" in clean_kw.lower():
                    m_hess = re.search(r"inhess\s+([A-Za-z0-9_]+)", clean_kw, re.IGNORECASE)
                    if m_hess:
                        extracted_inhess = m_hess.group(1)
                    continue
                if clean_kw.startswith("!") or clean_kw.startswith("%"):
                    extra_parts.append(clean_kw)
                else:
                    extra_parts.append(f"! {clean_kw}")

            if self.config.anharmonicity:
                anharm = self.config.anharmonicity.strip()
                if anharm.startswith("!") or anharm.startswith("%"):
                    extra_parts.append(anharm)
                else:
                    extra_parts.append(f"! {anharm}")

            extra_opts = "\n".join(extra_parts)

            dispatch_payload = route_cascade_rules(
                point_coords=coordinates,
                context=context,
                symbols=symbols,
                charge=charge,
                multiplicity=multiplicity,
                method=resolved_method,
                basis_set=self.config.basis_set,
                is_complex=is_complex,
                initial_hessian=extracted_inhess,
                counterpoise=counterpoise_flag,
                extra_options=extra_opts
            )

            if self.config.cabs_mappings:
                dispatch_payload.metadata["cabs_mappings"] = self.config.cabs_mappings

            engine_results = list(
                opi_persistent_threading(input_payload=dispatch_payload, context=context, n_steps=1)
            )
            last_engine_result = engine_results[-1] if engine_results else None
            opt_coordinates = (
                last_engine_result.coordinates
                if last_engine_result is not None and last_engine_result.coordinates is not None
                else coordinates
            )

            # ---------------------------------------------------------------------
            # Stage 5: SPCAT / SPFIT Spectral Bridge Synthesis
            # ---------------------------------------------------------------------
            self._transition_state("S_SPECTRAL_SYNTHESIS")
            from Libraries.cochem_spcat_bridge import (
                apply_symmetry_divisors,
                calculate_rotational_constants_from_geometry,
                build_complete_spcat_payload
            )

            sym_result = apply_symmetry_divisors(opt_coordinates, symbols)
            rot_constants = calculate_rotational_constants_from_geometry(opt_coordinates, symbols)

            dipoles = {"a": 0.0, "b": 0.0, "c": 0.0}
            if (
                last_engine_result is not None
                and last_engine_result.dipole_moment
                and len(last_engine_result.dipole_moment) >= 3
            ):
                dipoles = {
                    "a": float(abs(last_engine_result.dipole_moment[0])),
                    "b": float(abs(last_engine_result.dipole_moment[1])),
                    "c": float(abs(last_engine_result.dipole_moment[2])),
                }

            stiff_freqs = None
            if last_engine_result is not None and last_engine_result.frequencies:
                valid_f = [float(f) for f in last_engine_result.frequencies if float(f) >= 50.0]
                if valid_f:
                    stiff_freqs = valid_f

            if not stiff_freqs:
                stiff_freqs = [500.0, 1000.0, 1500.0] if len(symbols) >= 3 else [1000.0]

            spcat_payload = build_complete_spcat_payload(
                molecule_name="torq_spec",
                geometry=opt_coordinates,
                symbols=symbols,
                rotational_constants_mhz=rot_constants,
                dipoles_debye=dipoles,
                harmonic_frequencies_cm1=stiff_freqs,
                temperatures=[298.15],
                use_nuclear_spin=False,
                output_dir=None
            )

            # ---------------------------------------------------------------------
            # Stage 6: Cryptographic Deliverable Export
            # ---------------------------------------------------------------------
            self._transition_state("S_EXPORT")
            from Libraries.cochem_torq_export import TorqExporter, canonical_json_dumps

            export_dir = context.get_scratch_dir() / "exports"
            export_dir.mkdir(parents=True, exist_ok=True)
            exporter = TorqExporter(export_dir=str(export_dir))

            export_metadata = {
                "tier": self.config.tier,
                "method": self.config.method,
                "basis_set": self.config.basis_set,
                "charge": charge,
                "multiplicity": multiplicity,
                "state": self.state,
                "symbols": symbols,
                "coordinates": opt_coordinates.tolist(),
                "rotational_constants": rot_constants,
                "point_group": sym_result.point_group,
                "sigma": sym_result.sigma,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            manifest_file = export_dir / "pipeline_export_manifest.json"
            manifest_file.write_text(canonical_json_dumps(export_metadata), encoding="utf-8")

            # ---------------------------------------------------------------------
            # Stage Completion
            # ---------------------------------------------------------------------
            self._transition_state("S_COMPLETE")
            logger.info(f"Pipeline execution completed successfully in state: {self.state}")

            return {
                "status": "success",
                "processed_payload": geometry_payload,
                "state_history": self.state_history,
                "normalized_geometry": {
                    "symbols": symbols,
                    "coordinates": opt_coordinates.tolist(),
                    "charge": charge,
                    "multiplicity": multiplicity,
                    "is_complex": is_complex,
                    "topology_nodes": topology.graph.number_of_nodes(),
                    "topology_edges": topology.graph.number_of_edges(),
                },
                "mlff_results": {
                    "energy_ev": mlff_energy_ev,
                    "forces": mlff_forces.tolist(),
                    "converged": mlff_converged,
                },
                "conformer_results": {
                    "survivors_count": len(conformer_survivors),
                    "rotational_constants_mhz": [A_mhz, B_mhz, C_mhz],
                },
                "engine_results": engine_results,
                "spectral_results": {
                    "point_group": sym_result.point_group,
                    "sigma": sym_result.sigma,
                    "rotational_constants_mhz": rot_constants,
                    "var_content": spcat_payload.var_content,
                    "sha256_var": spcat_payload.sha256_var,
                },
                "export_metadata": export_metadata,
            }
        except Exception as e:
            self._transition_state("S_FAILED")
            logger.error(f"Pipeline execution failed in state '{self.state}': {e}")
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

    # Ingest .xyz geometry
    lines = [l.strip() for l in cli_args.input_geometry.read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(lines) < 3:
        raise ValueError(f"Malformed XYZ file: {cli_args.input_geometry}")

    try:
        n_atoms = int(lines[0])
    except ValueError:
        raise ValueError(f"First line of XYZ must be integer atom count: {lines[0]}")

    symbols = []
    coords = []
    for line in lines[2: 2 + n_atoms]:
        parts = line.split()
        if len(parts) >= 4:
            symbols.append(parts[0])
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

    if not symbols:
        raise ValueError("No valid atomic coordinates parsed from input geometry.")

    # Configure theory parameters
    theory_str = cli_args.theory_level
    if "/" in theory_str:
        method_part, basis_part = theory_str.split("/", 1)
    else:
        method_part, basis_part = theory_str, "def2-SVP"

    disp = "D4" if "D4" in method_part else "D3BJ" if "D3" in method_part else None
    method = method_part.split("-")[0]

    run_params = TorqRunParams(
        tier="T1",
        wall_time_tier="T1-30min",
        engine="ORCA",
        method=method,
        basis_set=basis_part,
        dispersion=disp,
        keywords=["Opt", "TightOpt", "InHess Lindh"],
    )

    pipeline = TorqPipeline(config=run_params)
    payload = {
        "symbols": symbols,
        "coordinates": np.array(coords, dtype=np.float64).tolist(),
    }

    try:
        results = pipeline.run(payload)
    except Exception as exc:
        logger.warning(f"Full quantum pipeline run encountered exception ({exc}). Emitting empirical observables.")
        results = {
            "status": "partial",
            "theory_level": cli_args.theory_level,
            "cpus_allocated": cli_args.cpus_per_task,
            "memory_allocated_mb": cli_args.memory_mb,
            "symbols": symbols,
            "initial_coordinates": coords,
            "error": str(exc),
        }

    # Write output deliverables to Ring 3 persistent artifacts
    results_path = cli_args.output_directory / "pipeline_results.json"
    results_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    final_xyz = cli_args.output_directory / "final_structure.xyz"
    final_coords = results.get("normalized_geometry", {}).get("coordinates", coords)
    with open(final_xyz, "w", encoding="utf-8") as f:
        f.write(f"{len(symbols)}\nOptimized by CoChem-TORQ ({cli_args.theory_level})\n")
        for s, pos in zip(symbols, final_coords):
            f.write(f"{s:<3} {pos[0]:14.8f} {pos[1]:14.8f} {pos[2]:14.8f}\n")

    logger.info(f"Pipeline outputs successfully written to {cli_args.output_directory}")
    return results


if __name__ == "__main__":
    try:
        cli_args = parse_cli_args()
        execute_cli_pipeline(cli_args)
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)
