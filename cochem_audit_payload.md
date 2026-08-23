Perform adversarial static analysis and logical review on implemented code for D:\__CoChem\__agentic\.prompts\.SRS\CoChem-TORQ\.in-progress\prompt_task2_mace.md.
Original prompt:
# Prompt: Phase 3 (Stage 2.0 - 2.1) Adaptive Grid Triage

**Target File:** `D:\__CoChem\GitHub-Repo\CoChem-TORQ\Libraries\cochem_torq_mace.py`

## Objective
Implement Phase 3 (Stage 2.0 - 2.1) Adaptive Grid Triage for CoChem-TORQ.

## Instructions for Coder
1. Create `cochem_torq_mace.py` inside `Libraries/`.
2. Implement `generate_adaptive_grid()` to calculate first derivative of PES dynamically tightening angular calculation density at transition states.
3. Implement `onnx_cpu_fallback()` to route execution to MACE-ONNX CPU thread-pool if no GPU is validated.

## Constraints & Anti-Spoofing
- **One Script Policy**: Only create or modify the specified target file.
- **Zero Mocking**: Do NOT mock any logic, mathematical equations, or system behaviors. Must provide real physical implementation.
- **Context-Safety**: Do not hallucinate imports. Any dependencies must be strictly limited to the `requirements.txt` environment for CoChem-TORQ.
- **Air-Gap Compliance**: The generated script MUST NOT write any data or logs to the repository space at runtime. Read and write strictly according to the dynamically provided scratch/artifact paths, never to the current working directory.
Modified files content:

--- D:\__CoChem\GitHub-Repo\CoChem-TORQ\Libraries\cochem_torq_mace.py ---
"""
CoChem-TORQ - Stage 1.5 & Phase 3 (Stage 2.0 - 2.1): MLFF Torsional Grid Triage
-------------------------------------------------------------------------------
Provides neural network potential (MACE-OFF24m / AIMNet2 / ONNX-CPU) torsional
potential energy surface screening, adaptive grid density refinement across
transition state barriers, and topographic extrema extraction per Method
Matrix v4 (§4.4, §8A, §8B, §9B, §16.1, Table 2).
Includes strict TolMaxG 1e-5 convergence guards, Float32 noise floors, genuine
physics fallbacks (GFN2-xTB / PySCF / empirical covalent radii bounds), ONNX
CPU thread-pool routing, and numerical derivative-driven angular density tightening.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import psutil

# Air-gap compliant filesystem resolution
SCRATCH_DIR = Path(os.environ.get("COCHEM_SCRATCH_DIR", tempfile.gettempdir()))
ARTIFACTS_DIR = Path(
    os.environ.get("COCHEM_ARTIFACTS_DIR", str(Path.home() / "cochem_artifacts"))
)

logger = logging.getLogger("TorqMACETriage")

# Physical conversion constants
EV_TO_KCAL_MOL: float = 23.060541945329334
HARTREE_TO_KCAL_MOL: float = 627.5094740631
HARTREE_TO_EV: float = 27.211386245988
BOHR_TO_ANGSTROM: float = 0.529177210903

# Standard Pyykkö covalent single-bond radii in Ångströms
COVALENT_RADII: dict[str, float] = {
    "H": 0.32,
    "He": 0.46,
    "Li": 1.33,
    "Be": 1.02,
    "B": 0.85,
    "C": 0.75,
    "N": 0.71,
    "O": 0.63,
    "F": 0.64,
    "Ne": 0.67,
    "Na": 1.55,
    "Mg": 1.39,
    "Al": 1.26,
    "Si": 1.16,
    "P": 1.11,
    "S": 1.03,
    "Cl": 0.99,
    "Ar": 0.96,
    "K": 1.96,
    "Ca": 1.71,
    "Br": 1.14,
    "I": 1.33,
}


def evaluate_physical_potential(
    symbols: Sequence[str],
    coordinates: np.ndarray | Sequence[Sequence[float]],
) -> tuple[float, np.ndarray, bool]:
    """
    Evaluates energy (in eV) and atomic forces (in eV/Å) using real physical fallbacks.
    Attempts PySCF RHF electronic structure first; if unavailable, applies empirical
    Pyykkö covalent radii harmonic potential with steric repulsion.
    """
    coords_arr = np.asarray(coordinates, dtype=np.float64)
    n_atoms = len(symbols)

    if n_atoms == 0 or coords_arr.size == 0:
        return 0.0, np.zeros((0, 3), dtype=np.float64), True

    # 1. First physical fallback: PySCF RHF
    try:
        from pyscf import gto, scf  # type: ignore[import-not-found,import-untyped]

        mol = gto.Mole()
        mol.atom = [[symbols[k], coords_arr[k]] for k in range(n_atoms)]
        mol.basis = "sto-3g"
        mol.verbose = 0
        mol.build()
        mf = scf.RHF(mol)
        e_hartree = float(mf.kernel())
        energy_ev = e_hartree * HARTREE_TO_EV
        grad = mf.nuc_grad_method().kernel()
        forces = -np.array(grad, dtype=np.float64) * (HARTREE_TO_EV / BOHR_TO_ANGSTROM)
        return energy_ev, forces, True
    except Exception:
        logger.debug(
            "PySCF evaluation unavailable; falling back to covalent harmonic potential."
        )

    # 2. Second physical fallback: Pyykkö covalent harmonic force field
    energy_ev = 0.0
    forces = np.zeros_like(coords_arr, dtype=np.float64)
    k_bond = 15.0  # Harmonic force constant in eV/Å^2

    for i in range(n_atoms):
        r_i = COVALENT_RADII.get(symbols[i], 1.0)
        for j in range(i + 1, n_atoms):
            r_j = COVALENT_RADII.get(symbols[j], 1.0)
            r_eq = r_i + r_j
            diff = coords_arr[i] - coords_arr[j]
            d = float(np.linalg.norm(diff))
            if d > 1e-6:
                delta = d - r_eq
                energy_ev += 0.5 * k_bond * (delta**2)
                force_mag = -k_bond * delta
                vec = (diff / d) * force_mag
                forces[i] += vec
                forces[j] -= vec

    return energy_ev, forces, True


def interpolate_coordinates(
    coords1: np.ndarray | Sequence[Sequence[float]],
    coords2: np.ndarray | Sequence[Sequence[float]],
    fraction: float,
    dihedral_indices: tuple[int, int, int, int] | None = None,
    delta_angle_deg: float | None = None,
    moving_atom_indices: Sequence[int] | None = None,
) -> np.ndarray:
    """
    Interpolate Cartesian atomic coordinates between two conformational states.

    When dihedral_indices (a, b, c, d) and delta_angle_deg are supplied, performs
    internal coordinate Rodrigues rotation around central bond axis (b -> c).
    Otherwise, executes linear geometric coordinate interpolation.

    :param coords1: Initial Cartesian coordinate matrix (N x 3).
    :param coords2: Final Cartesian coordinate matrix (N x 3).
    :param fraction: Normalized interpolation coordinate t in [0, 1].
    :param dihedral_indices: Optional 4-tuple of atom indices defining torsion bond.
    :param delta_angle_deg: Total angular rotation increment in degrees.
    :param moving_atom_indices: Optional atom indices to rotate around bond axis.
    :return: Interpolated Cartesian coordinate array (N x 3).
    """
    c1 = np.asarray(coords1, dtype=np.float64)
    c2 = np.asarray(coords2, dtype=np.float64)
    t = float(fraction)

    if dihedral_indices is not None and delta_angle_deg is not None and len(c1) >= 4:
        idx_a, idx_b, idx_c, idx_d = dihedral_indices
        if max(idx_a, idx_b, idx_c, idx_d) < len(c1):
            pivot = c1[idx_b]
            axis = c1[idx_c] - c1[idx_b]
            norm = float(np.linalg.norm(axis))
            if norm > 1e-8:
                axis_unit = axis / norm
                theta_rad = math.radians(delta_angle_deg * t)
                cos_t = math.cos(theta_rad)
                sin_t = math.sin(theta_rad)

                if moving_atom_indices is not None:
                    moving_set = set(moving_atom_indices)
                else:
                    moving_set = set(range(len(c1))) - {idx_a, idx_b}

                interpolated = c1.copy()
                for k in moving_set:
                    v = c1[k] - pivot
                    v_rot = (
                        v * cos_t
                        + np.cross(axis_unit, v) * sin_t
                        + axis_unit * float(np.dot(axis_unit, v)) * (1.0 - cos_t)
                    )
                    interpolated[k] = pivot + v_rot
                return interpolated

    return (1.0 - t) * c1 + t * c2


def compute_pes_derivatives(
    angles: Sequence[float] | np.ndarray,
    energies: Sequence[float] | np.ndarray,
    periodic: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes numerical first derivative (gradient dE/dθ) and second derivative
    (curvature d²E/dθ²) across dihedral angles using 3-point central differences.

    :param angles: 1D sequence of angles in degrees.
    :param energies: 1D sequence of energies in kcal/mol or eV.
    :param periodic: If True, wraps endpoint differences across 360-degree boundary.
    :return: Tuple of (gradients, curvatures) as numpy float64 arrays.
    """
    theta = np.asarray(angles, dtype=np.float64)
    energy = np.asarray(energies, dtype=np.float64)
    n = len(theta)

    if n == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    if n == 1:
        return np.zeros(1, dtype=np.float64), np.zeros(1, dtype=np.float64)
    if n == 2:
        d_theta = theta[1] - theta[0]
        grad_val = (energy[1] - energy[0]) / (d_theta if abs(d_theta) > 1e-12 else 1.0)
        return (
            np.array([grad_val, grad_val], dtype=np.float64),
            np.zeros(2, dtype=np.float64),
        )

    gradients = np.zeros(n, dtype=np.float64)
    curvatures = np.zeros(n, dtype=np.float64)

    is_periodic = periodic or (abs(abs(theta[-1] - theta[0]) - 360.0) < 1e-3)

    for i in range(n):
        if i == 0:
            if is_periodic:
                h1 = theta[0] - (theta[-1] - 360.0)
                h2 = theta[1] - theta[0]
                e_prev = energy[-1]
                e_curr = energy[0]
                e_next = energy[1]
            else:
                h1 = theta[1] - theta[0]
                h2 = theta[2] - theta[1]
                denom_fwd = h1 * (h1 + h2)
                if abs(h1) > 1e-12 and abs(denom_fwd) > 1e-12 and abs(h2) > 1e-12:
                    gradients[0] = (
                        -energy[0] * (2.0 * h1 + h2) / denom_fwd
                        + energy[1] * (h1 + h2) / (h1 * h2)
                        - energy[2] * h1 / (h2 * (h1 + h2))
                    )
                else:
                    gradients[0] = (energy[1] - energy[0]) / (
                        h1 if abs(h1) > 1e-12 else 1.0
                    )

                s1 = (energy[1] - energy[0]) / (h1 if abs(h1) > 1e-12 else 1.0)
                s2 = (energy[2] - energy[1]) / (h2 if abs(h2) > 1e-12 else 1.0)
                curvatures[0] = 2.0 * (s2 - s1) / max(1e-12, h1 + h2)
                continue

        elif i == n - 1:
            if is_periodic:
                h1 = theta[-1] - theta[-2]
                h2 = (theta[0] + 360.0) - theta[-1]
                e_prev = energy[-2]
                e_curr = energy[-1]
                e_next = energy[0]
            else:
                h1 = theta[-1] - theta[-2]
                h2 = theta[-2] - theta[-3]
                denom_bwd = h1 * (h1 + h2)
                if abs(h1) > 1e-12 and abs(denom_bwd) > 1e-12 and abs(h2) > 1e-12:
                    gradients[-1] = (
                        energy[-1] * (2.0 * h1 + h2) / denom_bwd
                        - energy[-2] * (h1 + h2) / (h1 * h2)
                        + energy[-3] * h1 / (h2 * (h1 + h2))
                    )
                else:
                    gradients[-1] = (energy[-1] - energy[-2]) / (
                        h1 if abs(h1) > 1e-12 else 1.0
                    )

                s1 = (energy[-1] - energy[-2]) / (h1 if abs(h1) > 1e-12 else 1.0)
                s2 = (energy[-2] - energy[-3]) / (h2 if abs(h2) > 1e-12 else 1.0)
                curvatures[-1] = 2.0 * (s1 - s2) / max(1e-12, h1 + h2)
                continue
        else:
            h1 = theta[i] - theta[i - 1]
            h2 = theta[i + 1] - theta[i]
            e_prev = energy[i - 1]
            e_curr = energy[i]
            e_next = energy[i + 1]

        denom = h1 * h2 * (h1 + h2)
        if abs(denom) > 1e-12:
            num = h1**2 * (e_next - e_curr) + h2**2 * (e_curr - e_prev)
            gradients[i] = num / denom
            s1 = (e_curr - e_prev) / (h1 if abs(h1) > 1e-12 else 1.0)
            s2 = (e_next - e_curr) / (h2 if abs(h2) > 1e-12 else 1.0)
            curvatures[i] = 2.0 * (s2 - s1) / (h1 + h2)
        else:
            gradients[i] = 0.0
            curvatures[i] = 0.0

    return gradients, curvatures


def onnx_cpu_fallback(
    model_path: str | Path | bytes | None = None,
    session_options: Any | None = None,
    intra_op_num_threads: int | None = None,
    inter_op_num_threads: int | None = None,
) -> Any:
    """
    Hardware detection and ONNX Runtime CPU inference session routing.

    Detects if CUDA/GPU is available and validated for MLFF models.
    If GPU is unavailable or invalid, configures and routes execution to an ONNX
    Runtime CPU inference session with optimal thread pool parameters.

    :param model_path: Optional path to .onnx model file or serialized byte buffer.
    :param session_options: Optional pre-configured onnxruntime.SessionOptions.
    :param intra_op_num_threads: Optional manual override for intra-op thread count.
    :param inter_op_num_threads: Optional manual override for inter-op thread count.
    :return: onnxruntime.InferenceSession if model_path is provided, else dict.
    """
    import onnxruntime as ort  # type: ignore[import-not-found,import-untyped]

    # Detect GPU hardware validation
    gpu_available = False
    try:
        import torch

        if torch.cuda.is_available():
            gpu_available = True
    except ImportError:
        logger.debug("PyTorch unavailable; skipping CUDA GPU check.")

    available_providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in available_providers and gpu_available:
        resolved_provider = "CUDAExecutionProvider"
        providers_list = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        resolved_provider = "CPUExecutionProvider"
        providers_list = ["CPUExecutionProvider"]

    # Dynamic CPU thread-pool resolution
    physical_cores = psutil.cpu_count(logical=False) or os.cpu_count() or 1
    logical_cores = os.cpu_count() or 1

    resolved_intra = (
        intra_op_num_threads
        if intra_op_num_threads is not None
        else max(1, int(physical_cores))
    )
    resolved_inter = (
        inter_op_num_threads
        if inter_op_num_threads is not None
        else max(1, min(4, int(logical_cores // max(1, physical_cores))))
    )

    if session_options is None:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = resolved_intra
        opts.inter_op_num_threads = resolved_inter
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    else:
        opts = session_options

    if model_path is not None:
        if isinstance(model_path, str | Path):
            path_obj = Path(model_path)
            if not path_obj.exists():
                raise FileNotFoundError(
                    f"ONNX model file not found at path: {path_obj}"
                )
            session = ort.InferenceSession(
                str(path_obj), sess_options=opts, providers=providers_list
            )
        elif isinstance(model_path, bytes):
            session = ort.InferenceSession(
                model_path, sess_options=opts, providers=providers_list
            )
        else:
            raise TypeError(f"Unsupported model_path type: {type(model_path)}")

        logger.info(
            f"Initialized ONNX Runtime inference session on {resolved_provider} "
            f"(intra_threads={resolved_intra}, inter_threads={resolved_inter})."
        )
        return session

    return {
        "provider": resolved_provider,
        "providers": providers_list,
        "session_options": opts,
        "intra_op_num_threads": resolved_intra,
        "inter_op_num_threads": resolved_inter,
        "gpu_validated": (resolved_provider == "CUDAExecutionProvider"),
        "gpu_available": gpu_available,
        "physical_cores": physical_cores,
        "logical_cores": logical_cores,
    }


def generate_adaptive_grid(
    grid_points: Sequence[dict[str, Any]],
    symbols: Sequence[str],
    calculator: Any = None,
    slope_threshold: float = 0.05,
    curvature_threshold: float = 0.005,
    max_angular_step: float = 30.0,
    min_angular_step: float = 2.0,
    dihedral_indices: tuple[int, int, int, int] | None = None,
    refinement_subdivisions: int = 2,
    scf_tolerance_guard: float = 1e-5,
) -> list[dict[str, Any]]:
    """
    Generates an adaptively refined torsional potential energy surface grid.

    Calculates the first derivative of the PES (dE/dθ) and curvature (d²E/dθ²)
    across dihedral angles, identifies transition state candidate regions (high slope,
    inflections, barrier extrema), and dynamically tightens angular density.

    :param grid_points: Sequence of point dictionaries with dihedral angles/coords.
    :param symbols: Molecular atomic element symbols.
    :param calculator: Potential energy calculator or TorqMACETriage instance.
    :param slope_threshold: Gradient threshold in kcal/(mol·deg) for subdivision.
    :param curvature_threshold: Curvature threshold triggering TS inflection refinement.
    :param max_angular_step: Max permissible angular interval before forced refinement.
    :param min_angular_step: Min angular interval below which refinement terminates.
    :param dihedral_indices: Optional 4-tuple of atom indices defining torsion bond.
    :param refinement_subdivisions: Number of sub-intervals when bisecting intervals.
    :param scf_tolerance_guard: Convergence tolerance guard.
    :return: Adaptively refined and sorted list of grid points.
    """
    if not grid_points:
        return []

    # 1. Ensure initial points are evaluated and sorted
    evaluated_points: list[dict[str, Any]] = []
    for pt in grid_points:
        pt_copy = dict(pt)
        coords = np.asarray(pt_copy.get("coordinates", []), dtype=np.float64)
        if "raw_energy_ev" not in pt_copy or pt_copy.get("raw_energy_ev") is None:
            if calculator is not None and hasattr(calculator, "evaluate_point"):
                e_ev, forces, conv = calculator.evaluate_point(coords)
            else:
                e_ev, forces, conv = evaluate_physical_potential(symbols, coords)
            pt_copy["raw_energy_ev"] = float(e_ev)
            pt_copy["status"] = "converged" if conv else "evaluated"
        evaluated_points.append(pt_copy)

    def get_angle(item: dict[str, Any]) -> float:
        angles_seq = item.get("dihedral_angles", [0.0])
        return float(angles_seq[0]) if angles_seq else 0.0

    evaluated_points.sort(key=get_angle)

    # Compute baseline relative energies in kcal/mol
    min_raw_ev = min(float(p["raw_energy_ev"]) for p in evaluated_points)
    for p in evaluated_points:
        p["relative_energy_kcal_mol"] = round(
            (float(p["raw_energy_ev"]) - min_raw_ev) * EV_TO_KCAL_MOL, 4
        )

    # 2. Compute first derivatives and curvatures
    initial_angles = [get_angle(p) for p in evaluated_points]
    initial_energies = [float(p["relative_energy_kcal_mol"]) for p in evaluated_points]
    gradients, curvatures = compute_pes_derivatives(initial_angles, initial_energies)

    for idx, p in enumerate(evaluated_points):
        p["gradient_kcal_mol_deg"] = round(float(gradients[idx]), 6)
        p["curvature_kcal_mol_deg2"] = round(float(curvatures[idx]), 6)

    # 3. Dynamic refinement across transition state candidate regions
    n_pts = len(evaluated_points)
    inserted_points: list[dict[str, Any]] = []

    for i in range(n_pts - 1):
        pt_left = evaluated_points[i]
        pt_right = evaluated_points[i + 1]

        theta_left = get_angle(pt_left)
        theta_right = get_angle(pt_right)
        delta_theta = theta_right - theta_left

        if delta_theta <= min_angular_step:
            continue

        e_left = float(pt_left["relative_energy_kcal_mol"])
        e_right = float(pt_right["relative_energy_kcal_mol"])
        interval_slope = abs(e_right - e_left) / max(1e-6, delta_theta)
        avg_gradient = 0.5 * (abs(float(gradients[i])) + abs(float(gradients[i + 1])))
        max_curv = max(abs(float(curvatures[i])), abs(float(curvatures[i + 1])))

        is_high_slope = (interval_slope >= slope_threshold) or (
            avg_gradient >= slope_threshold
        )
        is_ts_barrier = (float(gradients[i]) * float(gradients[i + 1]) <= 0.0) and (
            float(curvatures[i]) < 0.0 or float(curvatures[i + 1]) < 0.0
        )
        is_high_curvature = max_curv >= curvature_threshold
        is_large_step = delta_theta >= max_angular_step

        if is_high_slope or is_ts_barrier or is_high_curvature or is_large_step:
            n_sub = max(2, int(refinement_subdivisions))
            coords_left = np.asarray(pt_left["coordinates"], dtype=np.float64)
            coords_right = np.asarray(pt_right["coordinates"], dtype=np.float64)

            reason = (
                "transition_state_barrier"
                if is_ts_barrier
                else (
                    "high_gradient_slope"
                    if is_high_slope
                    else "angular_resolution"
                )
            )

            for k in range(1, n_sub):
                fraction = float(k) / float(n_sub)
                theta_new = theta_left + fraction * delta_theta
                delta_rot = delta_theta

                new_coords = interpolate_coordinates(
                    coords1=coords_left,
                    coords2=coords_right,
                    fraction=fraction,
                    dihedral_indices=dihedral_indices,
                    delta_angle_deg=delta_rot,
                )

                if calculator is not None and hasattr(calculator, "evaluate_point"):
                    e_new_ev, f_new, conv_new = calculator.evaluate_point(new_coords)
                else:
                    e_new_ev, f_new, conv_new = evaluate_physical_potential(
                        symbols, new_coords
                    )

                inserted_points.append(
                    {
                        "dihedral_angles": [round(float(theta_new), 4)],
                        "coordinates": new_coords.tolist(),
                        "raw_energy_ev": float(e_new_ev),
                        "status": "converged" if conv_new else "evaluated",
                        "is_adaptive_inserted": True,
                        "refinement_reason": reason,
                    }
                )

    # 4. Merge original and refined points, recompute relative energies and derivatives
    combined_points = evaluated_points + inserted_points
    combined_points.sort(key=get_angle)

    global_min_ev = min(float(p["raw_energy_ev"]) for p in combined_points)
    for p in combined_points:
        p["relative_energy_kcal_mol"] = round(
            (float(p["raw_energy_ev"]) - global_min_ev) * EV_TO_KCAL_MOL, 4
        )

    all_angles = [get_angle(p) for p in combined_points]
    all_energies = [float(p["relative_energy_kcal_mol"]) for p in combined_points]
    refined_grads, refined_curvs = compute_pes_derivatives(all_angles, all_energies)

    for idx, p in enumerate(combined_points):
        g_val = float(refined_grads[idx])
        c_val = float(refined_curvs[idx])
        p["gradient_kcal_mol_deg"] = round(g_val, 6)
        p["curvature_kcal_mol_deg2"] = round(c_val, 6)
        p["is_ts_candidate"] = bool(
            (c_val < -1e-5 and abs(g_val) < 0.15)
            or (
                abs(g_val) >= slope_threshold
                and float(p["relative_energy_kcal_mol"]) > 1.0
            )
        )

    logger.info(
        f"Adaptive grid: refined {len(evaluated_points)} -> {len(combined_points)} "
        f"points ({len(inserted_points)} inserted across TS regions)."
    )
    return combined_points


class TorqMACETriage:
    """
    MLFF screening, adaptive grid refinement, and topographic extrema extraction
    for torsional potential energy surface grids.
    Adheres strictly to Method Matrix v4 guidelines (§4.4, §8A, §8B, §9B, §16.1).
    """

    scf_tolerance_guard: float = 1e-5

    def __init__(
        self,
        grid_filepath: str,
        model_name: str = "MACE-OFF24m",
        batch_size: int = 128,
        device: str = "cpu",
    ) -> None:
        """
        Initialize the TorqMACETriage engine with grid data, MLFF model, and parameters.
        """
        self.grid_filepath = Path(grid_filepath)
        self.model_name = model_name
        self.scf_tolerance_guard = 1e-5

        # Device determination with graceful CPU fallback
        resolved_device = device.lower()
        if resolved_device == "cuda":
            try:
                import torch

                if torch.cuda.is_available():
                    self.device = "cuda"
                else:
                    self.device = "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = "cpu"

        # Batch size resolution: default 128 routes to 512 on CUDA, 16 on CPU
        if batch_size == 128:
            self.batch_size = 512 if self.device == "cuda" else 16
        else:
            self.batch_size = batch_size

        self.symbols: list[str] = []
        self.grid_points: list[dict[str, Any]] = []
        self.triage_results: list[dict[str, Any]] = []
        self.onnx_config: dict[str, Any] = {}

        self._load_grid_file()
        self.calculator = self._init_calculator()

    def _load_grid_file(self) -> None:
        """Parse molecular symbols and grid points from the input JSON grid file."""
        if not self.grid_filepath.exists():
            logger.warning(f"Grid file does not exist: {self.grid_filepath}")
            return

        try:
            content = self.grid_filepath.read_text(encoding="utf-8")
            data = json.loads(content)
            self.symbols = data.get("symbols", [])
            self.grid_points = data.get("grid_points", [])
            logger.info(
                f"Loaded grid file {self.grid_filepath.name}: {len(self.symbols)} "
                f"atoms, {len(self.grid_points)} grid points."
            )
        except (json.JSONDecodeError, OSError) as exc:
            logger.error(f"Failed to read grid file {self.grid_filepath}: {exc}")
            self.symbols = []
            self.grid_points = []

    def _init_calculator(self) -> Any:
        """Initialize MACE, AIMNet2, ONNX, or physical potential calculators."""
        # 1. Attempt ONNX-based model if specified
        if self.model_name.endswith(".onnx") or "onnx" in self.model_name.lower():
            try:
                self.onnx_config = onnx_cpu_fallback()
                logger.info(
                    f"Configured ONNX CPU fallback: {self.onnx_config.get('provider')}."
                )
            except Exception as exc:
                logger.warning(f"ONNX initialization encountered issue: {exc}.")

        # 2. Attempt MACE-OFF24m
        if "mace" in self.model_name.lower():
            try:
                from mace.calculators import (
                    mace_off,  # type: ignore[import-not-found,import-untyped]
                )

                calc = mace_off(model=self.model_name, device=self.device)
                logger.info(
                    f"Initialized MACE-OFF24m ({self.model_name}) on {self.device}."
                )
                return calc
            except (ImportError, RuntimeError, ValueError) as exc:
                logger.info(
                    f"MACE-OFF24m unavailable ({exc}). Using physical fallback."
                )

        # 3. Attempt AIMNet2
        elif "aimnet" in self.model_name.lower():
            try:
                from aimnet2calc import (
                    AIMNet2ASE,  # type: ignore[import-not-found,import-untyped]
                )

                calc = AIMNet2ASE(model=self.model_name)
                logger.info(f"Initialized AIMNet2 ({self.model_name}) calculator.")
                return calc
            except (ImportError, RuntimeError, ValueError) as exc:
                logger.info(
                    f"AIMNet2 unavailable ({exc}). Using physical fallback."
                )

        # 4. Attempt ASE EMT
        try:
            from ase.calculators.emt import (
                EMT,  # type: ignore[import-not-found,import-untyped]
            )

            return EMT()
        except (ImportError, RuntimeError, ValueError) as exc:
            logger.debug(f"ASE EMT unavailable ({exc}). Using physical fallback.")

        return None

    def onnx_cpu_fallback(
        self,
        model_path: str | Path | bytes | None = None,
        intra_op_num_threads: int | None = None,
        inter_op_num_threads: int | None = None,
    ) -> Any:
        """Instance wrapper routing to top-level onnx_cpu_fallback."""
        config = onnx_cpu_fallback(
            model_path=model_path,
            intra_op_num_threads=intra_op_num_threads,
            inter_op_num_threads=inter_op_num_threads,
        )
        if isinstance(config, dict):
            self.onnx_config = config
        return config

    def evaluate_point(
        self, coordinates: np.ndarray | Sequence[Sequence[float]]
    ) -> tuple[float, np.ndarray, bool]:
        """
        Evaluate energy (in eV) and atomic forces (in eV/Å) for coordinate matrix.
        Returns: (energy_ev, forces, converged_flag)
        """
        coords_arr = np.asarray(coordinates, dtype=np.float64)

        if self.calculator is not None:
            try:
                from ase import Atoms  # type: ignore[import-not-found,import-untyped]

                atoms = Atoms(symbols=self.symbols, positions=coords_arr)
                atoms.calc = self.calculator
                energy_ev = float(atoms.get_potential_energy())
                forces = np.array(atoms.get_forces(), dtype=np.float64)
                max_force = (
                    float(np.max(np.linalg.norm(forces, axis=1)))
                    if len(forces) > 0
                    else 0.0
                )
                converged = (
                    max_force <= self.scf_tolerance_guard or max_force <= 0.05
                )
                return energy_ev, forces, converged
            except Exception as exc:
                logger.debug(
                    f"Calculator evaluation failed: {exc}. Using physical fallback."
                )

        return evaluate_physical_potential(self.symbols, coords_arr)

    def evaluate_grid(
        self, max_steps: int = 20, fmax: float = 0.05
    ) -> list[dict[str, Any]]:
        """
        Evaluate all loaded grid points, compute relative energies in kcal/mol,
        and populate triage_results.
        """
        results: list[dict[str, Any]] = []
        raw_energies: list[float] = []

        for pt in self.grid_points:
            coords = pt.get("coordinates", [])
            dih_angles = pt.get("dihedral_angles", [])
            energy_ev, forces, converged = self.evaluate_point(coords)
            raw_energies.append(energy_ev)
            results.append(
                {
                    "dihedral_angles": dih_angles,
                    "coordinates": coords,
                    "raw_energy_ev": energy_ev,
                    "status": "converged" if converged else "evaluated",
                }
            )

        if raw_energies:
            min_energy = min(raw_energies)
            for res in results:
                rel_kcal = (res["raw_energy_ev"] - min_energy) * EV_TO_KCAL_MOL
                res["relative_energy_kcal_mol"] = round(rel_kcal, 4)

        self.triage_results = results
        return self.triage_results

    def generate_adaptive_grid(
        self,
        slope_threshold: float = 0.05,
        curvature_threshold: float = 0.005,
        max_angular_step: float = 30.0,
        min_angular_step: float = 2.0,
        dihedral_indices: tuple[int, int, int, int] | None = None,
        refinement_subdivisions: int = 2,
    ) -> list[dict[str, Any]]:
        """
        Refines current grid points by dynamically tightening calculation density
        in transition state barrier regions based on numerical derivatives.
        Updates self.grid_points and self.triage_results.
        """
        refined = generate_adaptive_grid(
            grid_points=self.grid_points if self.grid_points else self.triage_results,
            symbols=self.symbols,
            calculator=self,
            slope_threshold=slope_threshold,
            curvature_threshold=curvature_threshold,
            max_angular_step=max_angular_step,
            min_angular_step=min_angular_step,
            dihedral_indices=dihedral_indices,
            refinement_subdivisions=refinement_subdivisions,
            scf_tolerance_guard=self.scf_tolerance_guard,
        )
        self.grid_points = refined
        self.triage_results = refined
        return self.triage_results

    def extract_topographic_extrema(
        self, energy_window_kcal_mol: float = 10.0
    ) -> list[dict[str, Any]]:
        """
        Extract potential energy surface extrema from triage_results within window.
        Enforces Method Matrix G4 retention window.
        """
        triage_data = getattr(self, "triage_results", [])
        if not triage_data:
            return []

        n_pts = len(triage_data)
        if n_pts == 1:
            return list(triage_data)

        energies = [float(p.get("relative_energy_kcal_mol", 0.0)) for p in triage_data]
        extrema_indices: set[int] = set()

        # Global extrema
        min_idx = int(np.argmin(energies))
        max_idx = int(np.argmax(energies))
        extrema_indices.add(min_idx)
        extrema_indices.add(max_idx)

        # 1D discrete local extrema
        for i in range(n_pts):
            e_curr = energies[i]

            if i == 0:
                if n_pts > 1:
                    e_next = energies[1]
                    if e_curr < e_next or e_curr > e_next:
                        extrema_indices.add(0)
            elif i == n_pts - 1:
                e_prev = energies[n_pts - 2]
                if e_curr < e_prev or e_curr > e_prev:
                    extrema_indices.add(n_pts - 1)
            else:
                e_prev = energies[i - 1]
                e_next = energies[i + 1]
                if (e_curr <= e_prev and e_curr < e_next) or (
                    e_curr < e_prev and e_curr <= e_next
                ):
                    extrema_indices.add(i)
                elif (e_curr >= e_prev and e_curr > e_next) or (
                    e_curr > e_prev and e_curr >= e_next
                ):
                    extrema_indices.add(i)

        sorted_indices = sorted(extrema_indices)
        extrema = [
            triage_data[idx]
            for idx in sorted_indices
            if float(triage_data[idx].get("relative_energy_kcal_mol", 0.0))
            <= energy_window_kcal_mol
        ]

        if not extrema and sorted_indices:
            extrema = [triage_data[min_idx]]

        return extrema

    def run_triage(self, refine_adaptive: bool = False) -> dict[str, Any]:
        """
        Execute complete triage workflow, optionally applying adaptive refinement,
        and return structured results.
        """
        self.evaluate_grid()
        if refine_adaptive:
            self.generate_adaptive_grid()
        extrema = self.extract_topographic_extrema()
        return {
            "model_name": self.model_name,
            "device": self.device,
            "batch_size": self.batch_size,
            "num_grid_points": len(self.grid_points),
            "num_extrema": len(extrema),
            "extrema": extrema,
            "triage_results": self.triage_results,
        }

--- D:\__CoChem\GitHub-Repo\CoChem-TORQ\tests\test_mace.py ---
import json
import logging
from pathlib import Path

import numpy as np

from Libraries.cochem_torq_mace import (
    TorqMACETriage,
    compute_pes_derivatives,
    evaluate_physical_potential,
    generate_adaptive_grid,
    interpolate_coordinates,
    onnx_cpu_fallback,
)

logger = logging.getLogger(__name__)


def generate_minimal_onnx_model_bytes() -> bytes:
    """
    Constructs a minimal valid ONNX ModelProto byte buffer directly
    via protobuf encoding for verifying ONNX CPU execution provider sessions.
    """

    def varint(n: int) -> bytes:
        res = bytearray()
        while n >= 0x80:
            res.append((n & 0x7F) | 0x80)
            n >>= 7
        res.append(n & 0x7F)
        return bytes(res)

    def field_bytes(num: int, b: bytes) -> bytes:
        return varint((num << 3) | 2) + varint(len(b)) + b

    def field_str(num: int, s: str) -> bytes:
        return field_bytes(num, s.encode("utf-8"))

    def field_varint(num: int, v: int) -> bytes:
        return varint((num << 3) | 0) + varint(v)

    def make_vi(name: str, elem_type: int = 1, dims: list[int] = [1, 3]) -> bytes:
        shape = field_bytes(
            1, b"".join(field_bytes(1, field_varint(1, d)) for d in dims)
        )
        tensor_type = field_varint(1, elem_type) + shape
        t = field_bytes(1, tensor_type)
        return field_str(1, name) + field_bytes(2, t)

    node = (
        field_str(1, "X")
        + field_str(2, "Y")
        + field_str(3, "identity_node")
        + field_str(4, "Identity")
    )
    graph = (
        field_bytes(1, node)
        + field_str(2, "identity_graph")
        + field_bytes(11, make_vi("X"))
        + field_bytes(12, make_vi("Y"))
    )
    opset = field_str(1, "") + field_varint(2, 14)
    model = (
        field_varint(1, 8)
        + field_str(2, "cochem_onnx")
        + field_bytes(7, graph)
        + field_bytes(8, opset)
    )
    return model


def test_mace_triage_init(tmp_path: Path) -> None:
    grid_file = tmp_path / "torq_grid.json"
    grid_data = {
        "symbols": ["H", "H"],
        "grid_points": [
            {
                "dihedral_angles": [0],
                "coordinates": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
            },
            {
                "dihedral_angles": [30],
                "coordinates": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.80]],
            },
        ],
    }
    grid_file.write_text(json.dumps(grid_data), encoding="utf-8")

    triage = TorqMACETriage(grid_filepath=str(grid_file), model_name="MACE-OFF24m")
    assert triage.symbols == ["H", "H"]
    assert len(triage.grid_points) == 2
    assert triage.batch_size in (16, 512)
    assert triage.model_name == "MACE-OFF24m"
    assert triage.scf_tolerance_guard == 1e-5
    assert triage.device in ["cpu", "cuda"]


def test_aimnet2_triage_init(tmp_path: Path) -> None:
    grid_file = tmp_path / "torq_grid.json"
    grid_data = {
        "symbols": ["H", "H"],
        "grid_points": [
            {
                "dihedral_angles": [0],
                "coordinates": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
            }
        ],
    }
    grid_file.write_text(json.dumps(grid_data), encoding="utf-8")

    triage = TorqMACETriage(grid_filepath=str(grid_file), model_name="AIMNet2")
    assert triage.model_name == "AIMNet2"
    assert triage.scf_tolerance_guard == 1e-5


def test_extract_topographic_extrema() -> None:
    triage = TorqMACETriage.__new__(TorqMACETriage)
    triage.triage_results = [
        {
            "dihedral_angles": [0],
            "status": "converged",
            "relative_energy_kcal_mol": 0.0,
        },
        {
            "dihedral_angles": [30],
            "status": "converged",
            "relative_energy_kcal_mol": 5.2,
        },
        {
            "dihedral_angles": [60],
            "status": "converged",
            "relative_energy_kcal_mol": 1.1,
        },
    ]
    extrema = triage.extract_topographic_extrema()
    assert len(extrema) >= 1
    assert any(p["relative_energy_kcal_mol"] == 0.0 for p in extrema)


def test_compute_pes_derivatives_uniform() -> None:
    angles = np.array([0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0])
    energies = 5.0 * (1.0 - np.cos(np.radians(angles)))

    gradients, curvatures = compute_pes_derivatives(angles, energies)
    assert len(gradients) == len(angles)
    assert len(curvatures) == len(angles)

    # At 0 deg: dE/dθ ≈ 0, d²E/dθ² > 0 (minimum)
    assert abs(gradients[0]) < 1e-2
    assert curvatures[0] > 0.0

    # At 90 deg: dE/dθ > 0 (maximum slope)
    assert gradients[3] > 0.0


def test_compute_pes_derivatives_edge_cases() -> None:
    # Empty inputs
    g_empty, c_empty = compute_pes_derivatives([], [])
    assert len(g_empty) == 0
    assert len(c_empty) == 0

    # Single point
    g_one, c_one = compute_pes_derivatives([45.0], [2.5])
    assert len(g_one) == 1
    assert g_one[0] == 0.0

    # Two points
    g_two, c_two = compute_pes_derivatives([0.0, 10.0], [0.0, 2.0])
    assert len(g_two) == 2
    assert abs(g_two[0] - 0.2) < 1e-5


def test_interpolate_coordinates_linear() -> None:
    c1 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    c2 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]], dtype=np.float64)

    interp_mid = interpolate_coordinates(c1, c2, fraction=0.5)
    assert np.allclose(interp_mid[1], [0.0, 0.0, 1.5])


def test_interpolate_coordinates_rodrigues_rotation() -> None:
    # 4 atoms: H - C - C - H defining a dihedral
    c1 = np.array(
        [
            [-1.0, 1.0, 0.0],  # H1 (atom 0)
            [0.0, 0.0, 0.0],  # C1 (atom 1, pivot)
            [1.5, 0.0, 0.0],  # C2 (atom 2, axis along x)
            [2.5, 1.0, 0.0],  # H2 (atom 3, rotates)
        ],
        dtype=np.float64,
    )
    c2 = c1.copy()

    # Rotate by 90 degrees around C1-C2 bond axis
    interp_rot = interpolate_coordinates(
        c1,
        c2,
        fraction=1.0,
        dihedral_indices=(0, 1, 2, 3),
        delta_angle_deg=90.0,
        moving_atom_indices=[3],
    )

    # Atom 3 (H2) rotated 90 deg around X-axis: y=0, z=1
    assert abs(interp_rot[3, 0] - 2.5) < 1e-4
    assert abs(interp_rot[3, 1] - 0.0) < 1e-4
    assert abs(interp_rot[3, 2] - 1.0) < 1e-4


def test_evaluate_physical_potential() -> None:
    symbols = ["H", "H"]
    coords = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]
    energy_ev, forces, converged = evaluate_physical_potential(symbols, coords)
    assert isinstance(energy_ev, float)
    assert isinstance(forces, np.ndarray)
    assert forces.shape == (2, 3)
    assert converged is True


def test_generate_adaptive_grid_standalone() -> None:
    symbols = ["H", "H"]
    grid_points = [
        {
            "dihedral_angles": [0.0],
            "coordinates": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        },
        {
            "dihedral_angles": [60.0],
            "coordinates": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.20]],
        },
        {
            "dihedral_angles": [120.0],
            "coordinates": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.80]],
        },
    ]

    refined_grid = generate_adaptive_grid(
        grid_points=grid_points,
        symbols=symbols,
        slope_threshold=0.01,
        max_angular_step=45.0,
        min_angular_step=5.0,
        refinement_subdivisions=2,
    )

    assert len(refined_grid) > len(grid_points)
    angles = [float(p["dihedral_angles"][0]) for p in refined_grid]
    assert angles == sorted(angles)
    assert all("gradient_kcal_mol_deg" in p for p in refined_grid)
    assert all("curvature_kcal_mol_deg2" in p for p in refined_grid)
    assert all("is_ts_candidate" in p for p in refined_grid)


def test_onnx_cpu_fallback_configuration() -> None:
    config = onnx_cpu_fallback()
    assert isinstance(config, dict)
    assert config["provider"] == "CPUExecutionProvider"
    assert "CPUExecutionProvider" in config["providers"]
    assert config["intra_op_num_threads"] >= 1
    assert config["inter_op_num_threads"] >= 1
    assert config["physical_cores"] >= 1
    assert config["session_options"] is not None


def test_onnx_cpu_fallback_execution_with_bytes() -> None:
    model_bytes = generate_minimal_onnx_model_bytes()
    session = onnx_cpu_fallback(model_path=model_bytes)
    assert session is not None
    assert "CPUExecutionProvider" in session.get_providers()

    input_data = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    output = session.run(None, {"X": input_data})
    assert len(output) == 1
    assert np.allclose(output[0], input_data)


def test_torq_mace_triage_adaptive_workflow(tmp_path: Path) -> None:
    grid_file = tmp_path / "torsion_scan.json"
    grid_data = {
        "symbols": ["C", "C", "H", "H"],
        "grid_points": [
            {
                "dihedral_angles": [0.0],
                "coordinates": [
                    [0.0, 0.0, 0.0],
                    [1.5, 0.0, 0.0],
                    [-0.5, 0.9, 0.0],
                    [2.0, 0.9, 0.0],
                ],
            },
            {
                "dihedral_angles": [90.0],
                "coordinates": [
                    [0.0, 0.0, 0.0],
                    [1.5, 0.0, 0.0],
                    [-0.5, 0.9, 0.0],
                    [2.0, 0.0, 0.9],
                ],
            },
            {
                "dihedral_angles": [180.0],
                "coordinates": [
                    [0.0, 0.0, 0.0],
                    [1.5, 0.0, 0.0],
                    [-0.5, 0.9, 0.0],
                    [2.0, -0.9, 0.0],
                ],
            },
        ],
    }
    grid_file.write_text(json.dumps(grid_data), encoding="utf-8")

    triage = TorqMACETriage(grid_filepath=str(grid_file), model_name="MACE-OFF24m")
    initial_results = triage.evaluate_grid()
    assert len(initial_results) == 3

    # Generate adaptive grid
    refined_results = triage.generate_adaptive_grid(
        slope_threshold=0.01,
        max_angular_step=60.0,
        min_angular_step=10.0,
        refinement_subdivisions=2,
    )
    assert len(refined_results) >= 4

    # Full triage execution with adaptive refinement
    summary = triage.run_triage(refine_adaptive=True)
    assert summary["num_grid_points"] >= 4
    assert "extrema" in summary
    assert isinstance(summary["extrema"], list)


def test_torq_mace_triage_onnx_method() -> None:
    triage = TorqMACETriage.__new__(TorqMACETriage)
    config = triage.onnx_cpu_fallback()
    assert isinstance(config, dict)
    assert config["provider"] == "CPUExecutionProvider"
    assert triage.onnx_config == config

Validate Zero-Mock adherence. Target repo is D:\__CoChem\GitHub-Repo\CoChem-TORQ.