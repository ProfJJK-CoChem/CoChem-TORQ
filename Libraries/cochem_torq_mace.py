"""
CoChem-TORQ - Stage 1.5 & Phase 3 (Stage 2.0 - 2.1): MLFF Torsional Grid Triage
-------------------------------------------------------------------------------
Provides neural network potential (MACE-OFF24m / AIMNet2 / ONNX-CPU) torsional
potential energy surface screening, adaptive grid density refinement across
transition state barriers, and topographic extrema extraction per Method
Matrix v4 (§4.4, §8A, §8B, §9B, §16.1, Table 2).
Includes strict TolMaxG 1e-5 convergence guards, Float32 noise floors, genuine
explicit calculator selection and unavailable-result errors, ONNX
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

# Float32 precision noise floor (~4e-6 Eh) per Method Matrix §8A.5 & §9B.4
FLOAT32_NOISE_FLOOR_EH: float = 4.0e-6
FLOAT32_NOISE_FLOOR_EV: float = FLOAT32_NOISE_FLOOR_EH * HARTREE_TO_EV
FLOAT32_NOISE_FLOOR_KCAL_MOL: float = FLOAT32_NOISE_FLOOR_EV * EV_TO_KCAL_MOL

# Standard Pyykkö covalent single-bond radii in Ångströms (fallback table)
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


def get_covalent_radius(symbol: str) -> float:
    """
    Dynamically retrieve Pyykkö covalent single-bond radius in Ångströms via Mendeleev.
    Falls back to standard Pyykkö table if Mendeleev is unavailable.
    """
    try:
        import mendeleev

        elem = mendeleev.element(symbol)
        rad_pm = getattr(elem, "covalent_radius_pyykko", None) or getattr(
            elem, "covalent_radius", None
        )
        if rad_pm is not None:
            return float(rad_pm) / 100.0
    except Exception:
        pass
    if symbol in COVALENT_RADII:
        return COVALENT_RADII[symbol]
    raise ValueError(f"Covalent radius unavailable for {symbol!r}.")


def partition_molecular_graph(
    symbols: Sequence[str],
    coordinates: np.ndarray | Sequence[Sequence[float]],
) -> tuple[list[list[int]], np.ndarray]:
    """Partitions system into bonded molecular fragments via Pyykkö covalent radii.

    Returns:
        fragments: list of lists of atom indices belonging to each connected monomer.
        adj_matrix: boolean adjacency matrix (N, N).
    """
    coords_arr = np.asarray(coordinates, dtype=np.float64)
    n = len(symbols)
    if n == 0:
        return [], np.full((0, 0), False, dtype=bool)

    cov_radii = [get_covalent_radius(s) for s in symbols]
    adj = np.full((n, n), False, dtype=bool)

    for i in range(n):
        for j in range(i + 1, n):
            dist = float(np.linalg.norm(coords_arr[i] - coords_arr[j]))
            cutoff = 1.25 * (cov_radii[i] + cov_radii[j])
            if dist <= cutoff:
                adj[i, j] = True
                adj[j, i] = True

    visited = set()
    fragments: list[list[int]] = []
    for i in range(n):
        if i not in visited:
            comp = []
            queue = [i]
            visited.add(i)
            while queue:
                curr = queue.pop(0)
                comp.append(curr)
                for neighbor in range(n):
                    if adj[curr, neighbor] and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            fragments.append(comp)

    return fragments, adj


def evaluate_physical_potential(
    symbols: Sequence[str],
    coordinates: np.ndarray | Sequence[Sequence[float]],
    use_pyscf: bool = False,
) -> tuple[float, np.ndarray, bool]:
    """Evaluate explicitly requested RHF/STO-3G; never substitute an invented potential.

    This limited, closed-shell reference calculation is not MACE or high-level
    electronic structure. Calling without an explicit engine selection fails.
    """
    if not use_pyscf:
        raise RuntimeError(
            "No potential calculator selected; an empirical replacement is not a scientific result."
        )
    coords_arr = np.asarray(coordinates, dtype=np.float64)
    if (
        not symbols
        or coords_arr.shape != (len(symbols), 3)
        or not np.isfinite(coords_arr).all()
    ):
        raise ValueError("A nonempty finite molecular geometry is required.")
    try:
        from pyscf import gto, scf

        mol = gto.M(
            atom=[[s, r] for s, r in zip(symbols, coords_arr)],
            basis="sto-3g",
            verbose=0,
        )
        mf = scf.RHF(mol)
        energy_ev = float(mf.kernel()) * HARTREE_TO_EV
        if not mf.converged:
            raise RuntimeError("RHF/STO-3G did not converge.")
        forces = (
            -np.asarray(mf.nuc_grad_method().kernel())
            * HARTREE_TO_EV
            / BOHR_TO_ANGSTROM
        )
        if not np.isfinite(energy_ev) or not np.isfinite(forces).all():
            raise RuntimeError("RHF/STO-3G returned non-finite results.")
        return energy_ev, forces, True
    except Exception as exc:
        raise RuntimeError(
            f"Requested RHF/STO-3G calculation unavailable or failed: {exc}"
        ) from exc


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
    if (
        theta.ndim != 1
        or energy.shape != theta.shape
        or not np.isfinite(theta).all()
        or not np.isfinite(energy).all()
    ):
        raise ValueError(
            "PES derivatives require aligned finite one-dimensional angles and energies."
        )
    if len(theta) < 3:
        raise ValueError(
            "At least three distinct samples are required to estimate PES curvature."
        )
    if np.any(np.diff(theta) <= 1e-12):
        raise ValueError("PES angles must be strictly increasing and distinct.")
    if not periodic:
        gradients = np.gradient(energy, theta, edge_order=2)
        curvatures = np.gradient(gradients, theta, edge_order=2)
        return gradients, curvatures
    span = theta[-1] - theta[0]
    if span >= 360.0:
        if not np.isclose(span, 360.0) or not np.isclose(
            energy[-1], energy[0], atol=1e-10, rtol=1e-10
        ):
            raise ValueError(
                "Periodic PES endpoints must span at most 360 degrees and duplicate energies must agree."
            )
        gradients, curvatures = compute_pes_derivatives(
            theta[:-1], energy[:-1], periodic=True
        )
        return np.append(gradients, gradients[0]), np.append(curvatures, curvatures[0])
    x = np.concatenate(([theta[-1] - 360.0], theta, [theta[0] + 360.0]))
    e = np.concatenate(([energy[-1]], energy, [energy[0]]))
    h1, h2 = theta - x[:-2], x[2:] - theta
    gradients = (h1**2 * (e[2:] - energy) + h2**2 * (energy - e[:-2])) / (
        h1 * h2 * (h1 + h2)
    )
    curvatures = 2 * ((e[2:] - energy) / h2 - (energy - e[:-2]) / h1) / (h1 + h2)
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
        angles_seq = item.get("dihedral_angles")
        if not angles_seq or not np.isfinite(angles_seq[0]):
            raise ValueError("A finite dihedral angle is required for PES refinement.")
        return float(angles_seq[0])

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
                else ("high_gradient_slope" if is_high_slope else "angular_resolution")
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
        """Initialize only the requested calculator; unavailable engines fail explicitly."""
        model = self.model_name.lower()
        try:
            if "mace" in model:
                from mace.calculators import mace_off

                return mace_off(
                    model=self.model_name, device=self.device, default_dtype="float64"
                )
            if "aimnet" in model:
                from aimnet2calc import AIMNet2ASE

                return AIMNet2ASE(model=self.model_name)
            if model == "emt":
                from ase.calculators.emt import EMT

                return EMT()
            if model.endswith(".onnx") or "onnx" in model:
                raise RuntimeError(
                    "ONNX model-to-ASE energy/force adapter is not implemented."
                )
            raise ValueError(f"Unsupported calculator {self.model_name!r}.")
        except Exception as exc:
            raise RuntimeError(
                f"Requested calculator {self.model_name!r} is unavailable: {exc}"
            ) from exc

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

        if (
            coords_arr.shape != (len(self.symbols), 3)
            or not self.symbols
            or not np.isfinite(coords_arr).all()
        ):
            raise ValueError("A nonempty finite molecular geometry is required.")
        if self.calculator is not None:
            try:
                from ase import Atoms  # type: ignore[import-not-found,import-untyped]

                atoms = Atoms(symbols=self.symbols, positions=coords_arr)
                atoms.calc = self.calculator
                energy_ev = float(atoms.get_potential_energy())
                forces = np.array(atoms.get_forces(), dtype=np.float64)
                if (
                    forces.shape != coords_arr.shape
                    or not np.isfinite(energy_ev)
                    or not np.isfinite(forces).all()
                ):
                    raise ValueError("Calculator returned invalid energy or forces.")
                max_force_ev_ang = (
                    float(np.max(np.linalg.norm(forces, axis=1)))
                    if len(forces) > 0
                    else 0.0
                )
                max_force_eh_bohr = max_force_ev_ang * (
                    BOHR_TO_ANGSTROM / HARTREE_TO_EV
                )
                converged = max_force_eh_bohr <= self.scf_tolerance_guard
                return energy_ev, forces, converged
            except Exception as exc:
                raise RuntimeError(
                    f"Requested calculator {self.model_name!r} failed: {exc}"
                ) from exc

        raise RuntimeError(
            "No calculator is configured; energy and forces are unavailable."
        )

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
        Annotate suggested extrema without removing candidates (advisory ML only).
        """
        triage_data = getattr(self, "triage_results", [])
        if not triage_data:
            return []

        n_pts = len(triage_data)
        if n_pts == 1:
            return list(triage_data)

        energies = [float(p["relative_energy_kcal_mol"]) for p in triage_data]
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
            if float(triage_data[idx]["relative_energy_kcal_mol"])
            <= energy_window_kcal_mol
        ]

        if not extrema and sorted_indices:
            extrema = [triage_data[min_idx]]

        # Advisory ML policy: annotate priorities while retaining every input.
        priority_ids = {id(point) for point in extrema}
        return [
            dict(
                point,
                is_extremum=id(point) in priority_ids,
                advisory_only=True,
                eligible_for_pruning=False,
            )
            for point in triage_data
        ]

    def audit_rank_inversion(
        self,
        reference_energies_kcal_mol: Sequence[float],
        spearman_threshold: float = 0.9,
        retention_window_kcal_mol: float = 10.0,
    ) -> dict[str, Any]:
        """
        Executes Method Matrix Guard G4 rank-inversion audit against high-level
        reference energies. Calculates Spearman rank correlation coefficient
        (rho >= 0.90) and verifies that no structure outside the retained set
        falls within the high-level retention window.

        :param reference_energies_kcal_mol: Sequence of high-level reference energies
            in kcal/mol.
        :param spearman_threshold: Minimum acceptable Spearman rank correlation.
        :param retention_window_kcal_mol: Energy retention window in kcal/mol.
        :return: Audit report dictionary with correlation, rank displacement,
            and culling eligibility.
        """
        triage_data = getattr(self, "triage_results", [])
        if not triage_data or len(reference_energies_kcal_mol) != len(triage_data):
            raise ValueError(
                f"Mismatch: {len(triage_data)} MLFF points vs "
                f"{len(reference_energies_kcal_mol)} reference energies."
            )

        mlff_energies = np.array(
            [float(p["relative_energy_kcal_mol"]) for p in triage_data],
            dtype=np.float64,
        )
        ref_energies = np.asarray(reference_energies_kcal_mol, dtype=np.float64)
        ref_rel = ref_energies - np.min(ref_energies)

        if len(mlff_energies) < 3:
            raise ValueError(
                "Rank diagnostics require at least three observed paired energies."
            )
        if not np.isfinite(mlff_energies).all() or not np.isfinite(ref_energies).all():
            raise ValueError("Rank diagnostics require finite observed energies.")
        if np.ptp(mlff_energies) == 0 or np.ptp(ref_energies) == 0:
            raise ValueError(
                "Spearman correlation is undefined for a constant energy series."
            )
        if not np.isfinite(spearman_threshold) or not -1 <= spearman_threshold <= 1:
            raise ValueError("Spearman threshold must be finite and within [-1,1].")
        if not np.isfinite(retention_window_kcal_mol) or retention_window_kcal_mol <= 0:
            raise ValueError("A finite positive retention window is required.")
        try:
            from scipy.stats import rankdata, spearmanr
        except ImportError as exc:
            raise RuntimeError(
                "SciPy rank statistics are unavailable; no statistical result was computed."
            ) from exc
        result = spearmanr(mlff_energies, ref_rel)
        rho, pval = float(result.statistic), float(result.pvalue)
        if (
            not np.isfinite(rho)
            or not np.isfinite(pval)
            or not -1 <= rho <= 1
            or not 0 <= pval <= 1
        ):
            raise ValueError(
                "SciPy returned an undefined Spearman statistic or p-value."
            )
        # Average ranks treat ties consistently with the actual Spearman statistic.
        rank_displacements = np.abs(rankdata(mlff_energies) - rankdata(ref_rel))
        max_rank_disp = float(np.max(rank_displacements))

        # Verify G4 retention window
        retained_mask_mlff = mlff_energies <= retention_window_kcal_mol
        retained_mask_ref = ref_rel <= retention_window_kcal_mol
        omitted_in_window = bool(np.any((~retained_mask_mlff) & retained_mask_ref))

        diagnostic_passed = bool(
            (rho >= spearman_threshold) and (not omitted_in_window)
        )
        cull_eligible = (
            False  # Approved advisory policy: diagnostics never authorize pruning.
        )

        report = {
            "spearman_rho": round(rho, 6),
            "p_value": float(pval),
            "spearman_threshold": spearman_threshold,
            "max_rank_displacement": max_rank_disp,
            "retention_window_kcal_mol": retention_window_kcal_mol,
            "num_structures": len(triage_data),
            "num_retained_mlff": int(np.sum(retained_mask_mlff)),
            "num_retained_ref": int(np.sum(retained_mask_ref)),
            "omitted_in_window": omitted_in_window,
            "cull_eligible": cull_eligible,
            "g4_status": "DIAGNOSTIC_PASSED_ADVISORY_ONLY"
            if diagnostic_passed
            else "REJECTED_RANK_INVERSION",
            "diagnostic_passed": diagnostic_passed,
            "advisory_only": True,
            "eligible_for_pruning": False,
            "statistical_method": "scipy.stats.spearmanr, asymptotic p-value",
            "qualification": "Rank diagnostics alone do not establish search completeness or transferable accuracy.",
        }
        logger.info(
            f"Method Matrix G4 rank audit: rho={rho:.4f} "
            f"(thresh={spearman_threshold}), "
            f"max_disp={max_rank_disp}, culling_eligible={cull_eligible}."
        )
        return report

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
