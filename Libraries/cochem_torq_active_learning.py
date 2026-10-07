"""Active Learning Orchestrator for dynamic QM selection, Kabsch RMSD, and air-gapped manifests.

Method Matrix v4 Provenance Tags: [M] Mandated, [D] Derived, [E] Empirical.
Strict Zero-Mock Mandate v3: Completely authentic physics, dynamic Mendeleev masses, and air-gapped serialization.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import shutil
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np
from filelock import FileLock

from Libraries.cochem_torq_inference_errors import (
    ActiveLearningSelectionError,
)
from Libraries.cochem_torq_inference_schemas import ActiveLearningOrchestratorConfig
from Libraries.cochem_torq_masses import get_monoisotopic_mass


class ActiveLearningState(str, Enum):
    """Lifecycle state machine for active learning candidates. [M]"""

    UNLABELED = "UNLABELED"
    CANDIDATE_SELECTED = "CANDIDATE_SELECTED"
    MANIFEST_EMITTED = "MANIFEST_EMITTED"
    QM_COMPLETED = "QM_COMPLETED"
    INGESTED = "INGESTED"


@dataclass
class CandidateGeometry:
    """Authentic physical molecular geometry with committee uncertainty telemetry. [M]"""

    candidate_id: str
    coordinates: np.ndarray             # (N, 3) in Angstroms
    atomic_numbers: list[int]           # (N,) atomic numbers Z
    energy_variance: float              # sigma_E^2 in eV^2
    max_force_std: float                # alpha_F^std in eV/Angstrom
    rotational_constants: tuple[float, float, float] # (A, B, C) in cm^-1
    state: ActiveLearningState = ActiveLearningState.UNLABELED
    assigned_tier: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def compute_qbc_energy_variance(energies: Sequence[float]) -> float:
    """Compute Query-by-Committee (QBC) unbiased energy sample variance. [D]
    
    sigma_E^2(X) = 1/(M-1) sum_m (E_m - E_mean)^2
    """
    m = len(energies)
    if m < 2:
        raise ActiveLearningSelectionError(
            f"QBC energy variance requires at least 2 committee predictions, got M={m}.",
            diagnostics={"num_models": m},
        )
    arr = np.asarray(energies, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise ActiveLearningSelectionError("Committee energies must be finite; missing predictions are not zero uncertainty.")
    mean_e = np.mean(arr)
    var_e = np.sum((arr - mean_e) ** 2) / float(m - 1)
    return float(max(0.0, var_e))


def compute_max_force_epistemic_std(forces: np.ndarray) -> float:
    """Compute maximum atomic force epistemic standard deviation alpha_F^std in eV/Angstrom. [D]
    
    forces shape: (M, N, 3)
    alpha_F^std = max_i sqrt( 1/(M-1) sum_m ||F_{i,m} - F_{i,mean}||_2^2 )
    """
    forces = np.asarray(forces, dtype=np.float64)
    if forces.ndim != 3 or forces.shape[1] == 0 or forces.shape[2] != 3 or not np.isfinite(forces).all():
        raise ActiveLearningSelectionError("Committee forces must be finite with shape (M, N, 3), N > 0.")
    m, n, _ = forces.shape
    if m < 2:
        raise ActiveLearningSelectionError(
            f"Force epistemic variance requires at least 2 committee predictions, got M={m}.",
            diagnostics={"num_models": m},
        )
    mean_f = np.mean(forces, axis=0, keepdims=True) # (1, N, 3)
    diff_f = forces - mean_f                        # (M, N, 3)
    sq_norm = np.sum(diff_f ** 2, axis=-1)          # (M, N)
    var_f = np.sum(sq_norm, axis=0) / float(m - 1)  # (N,)
    stds = np.sqrt(np.maximum(0.0, var_f))          # (N,)
    return float(np.max(stds))


def center_geometry_mass_weighted(
    coordinates: np.ndarray,
    atomic_numbers: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Mendeleev mass-weighted center of mass and translate to origin. [M]/[D]
    
    Returns (centered_coords, com).
    """
    coords = np.asarray(coordinates, dtype=np.float64)
    masses = np.array([get_monoisotopic_mass(int(z)) for z in atomic_numbers], dtype=np.float64)
    total_mass = np.sum(masses)
    if total_mass <= 1e-12:
        raise ActiveLearningSelectionError("Total molecular mass must be strictly positive.")

    com = np.sum(coords * masses[:, np.newaxis], axis=0) / total_mass
    centered = coords - com[np.newaxis, :]
    return centered, com


def compute_rotational_constants(
    coordinates: np.ndarray,
    atomic_numbers: Sequence[int],
) -> tuple[float, float, float]:
    """Compute principal rotational constants (A, B, C) in cm^-1 via Mendeleev inertia tensor. [D]
    
    I_{alpha, beta} = sum_i m_i (||x'_i||^2 delta_{alpha, beta} - x'_{i, alpha} x'_{i, beta})
    A = h / (8 pi^2 c I_A), B = h / (8 pi^2 c I_B), C = h / (8 pi^2 c I_C)
    """
    centered, _ = center_geometry_mass_weighted(coordinates, atomic_numbers)
    masses = np.array([get_monoisotopic_mass(int(z)) for z in atomic_numbers], dtype=np.float64)

    r2 = np.sum(centered ** 2, axis=1) # (N,)
    inertia = np.array(
        [
            [
                np.sum(masses * ((1.0 if alpha == beta else 0.0) * r2 - centered[:, alpha] * centered[:, beta]))
                for beta in range(3)
            ]
            for alpha in range(3)
        ],
        dtype=np.float64,
    )

    # Diagonalize inertia tensor
    eigvals = np.linalg.eigvalsh(inertia)
    eigvals = np.sort(np.maximum(1e-12, eigvals))
    ia, ib, ic = eigvals[0], eigvals[1], eigvals[2]

    # Conversion factor from u * Angstrom^2 to cm^-1:
    # h / (8 * pi^2 * c * 1.66053906660e-47 kg m^2) approx 16.8576292
    h_c_factor = 16.857629204031
    const_a = float(h_c_factor / ia)
    const_b = float(h_c_factor / ib)
    const_c = float(h_c_factor / ic)
    return (const_a, const_b, const_c)


def kabsch_rmsd(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
) -> float:
    """Compute pairwise minimum root-mean-square deviation via Kabsch SVD in SO(3). [D]
    
    D_geom = sqrt( 1/N sum_i ||x_{a,i} - x_{b,i} R||_2^2 )
    """
    p = np.asarray(coords_a, dtype=np.float64)
    q = np.asarray(coords_b, dtype=np.float64)

    # Center both coordinates at centroid
    p_c = p - np.mean(p, axis=0, keepdims=True)
    q_c = q - np.mean(q, axis=0, keepdims=True)

    # Covariance matrix H = p_c^T @ q_c
    h = np.dot(p_c.T, q_c)
    u, s, vt = np.linalg.svd(h)
    v = vt.T

    # Reflection determinant correction
    d = np.linalg.det(np.dot(v, u.T))
    s_mat = np.diag([1.0, 1.0, np.sign(d)])

    # Optimal rotation matrix R in SO(3)
    r = np.dot(np.dot(v, s_mat), u.T)

    # Rotated q coordinates
    q_rot = np.dot(q_c, r)
    diff = p_c - q_rot
    rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
    return float(rmsd)


def check_stage_b_rotational_redundancy(
    rot_a: tuple[float, float, float],
    rot_b: tuple[float, float, float],
    threshold: float = 0.001,
) -> bool:
    """Stage B relative rotational constant invariance check max(|Delta B| / B) < threshold. [M]"""
    a_a, b_a, c_a = rot_a
    a_b, b_b, c_b = rot_b

    rel_a = abs(a_a - a_b) / max(1e-12, a_a)
    rel_b = abs(b_a - b_b) / max(1e-12, b_a)
    rel_c = abs(c_a - c_b) / max(1e-12, c_a)

    max_rel = max(rel_a, rel_b, rel_c)
    return max_rel < threshold


def route_qm_tier(
    max_force_std: float,
    hardware_topology: Any | None = None,
    compute_budget_hours: float | None = None,
    available_engines: Sequence[str] | None = None,
    interactive_gate: Callable[..., bool] | None = None,
    **kwargs: Any,
) -> str:
    """Route candidate dynamically based on epistemic force uncertainty severity and hardware availability [M].

    - Moderate (0.05 < alpha_F <= 0.20 eV/A): Tier T3-10s GFN2-xTB
    - High (0.20 < alpha_F <= 0.80 eV/A): Tier T3O-1h ORCA 6.0 omegaB97X-V/jun-cc-pVTZ
    - Extreme (alpha_F > 0.80 eV/A): Tier T3O-12h Canonical junChS composite scheme

    Hardware Triage Gate:
    Inspects available local computational engines (e.g. ORCA, CFOUR) and the allocated compute budget
    before assigning high-force-uncertainty candidates to high-cost composite tiers.
    If a required engine is absent or projected wall-clock time exceeds budget, triggers an interactive
    decision gate or gracefully degrades to the highest supported tier (e.g., 'B3LYP-D4/def2-TZVP' or 'T3O-1h').
    """
    if max_force_std <= 0.20:
        nominal_tier = "T3-10s"
    elif max_force_std <= 0.80:
        nominal_tier = "T3O-1h"
    else:
        nominal_tier = "T3O-12h"

    if nominal_tier == "T3-10s":
        return nominal_tier

    # Normalize engine availability
    engines: list[str] = []
    if available_engines is not None:
        engines = [e.lower() for e in available_engines]
    elif hardware_topology is not None and hasattr(hardware_topology, "available_engines") and hardware_topology.available_engines:
        engines = [e.lower() for e in hardware_topology.available_engines]
    else:
        import shutil
        if shutil.which("orca") is not None:
            engines.append("orca")
        if shutil.which("xcfour") is not None or shutil.which("cfour") is not None:
            engines.append("cfour")
        if shutil.which("xtb") is not None:
            engines.append("xtb")

    # Ingest compute budget from hardware topology if not explicitly provided
    if compute_budget_hours is None and hardware_topology is not None and hasattr(hardware_topology, "compute_budget_hours"):
        compute_budget_hours = float(hardware_topology.compute_budget_hours)

    needs_cfour = (nominal_tier == "T3O-12h")
    needs_orca = (nominal_tier in ("T3O-1h", "T3O-12h"))
    has_cfour = "cfour" in engines
    has_orca = "orca" in engines

    projected_hours = 12.0 if nominal_tier == "T3O-12h" else 1.0
    budget_exceeded = (compute_budget_hours is not None and compute_budget_hours < projected_hours)

    # Core capacity gate from hardware topology: T3O-12h requires >= 4 P-cores for parallel CC
    if hardware_topology is not None and nominal_tier == "T3O-12h":
        p_cores = getattr(hardware_topology, "p_cores", None)
        if p_cores is not None and p_cores < 4:
            budget_exceeded = True

    missing_engines: list[str] = []
    if needs_cfour and not has_cfour:
        missing_engines.append("CFOUR")
    if needs_orca and not has_orca:
        missing_engines.append("ORCA")

    if missing_engines or budget_exceeded:
        if interactive_gate is not None:
            decision = interactive_gate(
                nominal_tier=nominal_tier,
                missing_engines=missing_engines,
                budget_exceeded=budget_exceeded,
                compute_budget_hours=compute_budget_hours,
            )
            if decision:
                return nominal_tier

        # Graceful degradation cascade [M]
        if has_orca:
            if budget_exceeded and compute_budget_hours is not None and compute_budget_hours < 1.0:
                return "T3-10s" if "xtb" in engines else "B3LYP-D4/def2-TZVP"
            return "B3LYP-D4/def2-TZVP" if nominal_tier == "T3O-12h" else "T3O-1h"
        elif "xtb" in engines:
            return "T3-10s"
        else:
            return "B3LYP-D4/def2-TZVP"

    return nominal_tier


class ActiveLearningOrchestrator:
    """Active Learning engine coordinating candidate acquisition, deduplication, and air-gapped manifests. [M]"""

    def __init__(self, config: ActiveLearningOrchestratorConfig) -> None:
        self.config = config
        self.pool: list[CandidateGeometry] = []
        self.selected_batch: list[CandidateGeometry] = []

    def evaluate_pool(
        self,
        geometries: Sequence[np.ndarray],
        atomic_numbers: Sequence[Sequence[int]],
        committee_energies: Sequence[Sequence[float]],
        committee_forces: Sequence[np.ndarray],
        candidate_ids: Sequence[str] | None = None,
    ) -> list[CandidateGeometry]:
        """Evaluate unlabeled pool and instantiate CandidateGeometry representations. [M]"""
        p = len(geometries)
        self.pool.clear()

        for idx in range(p):
            cid = candidate_ids[idx] if candidate_ids is not None else f"cand_{idx:04d}"
            coords = np.asarray(geometries[idx], dtype=np.float64)
            z = list(atomic_numbers[idx])
            e_m = list(committee_energies[idx])
            f_m = np.asarray(committee_forces[idx], dtype=np.float64)

            var_e = compute_qbc_energy_variance(e_m)
            max_f_std = compute_max_force_epistemic_std(f_m)
            rot_consts = compute_rotational_constants(coords, z)

            candidate = CandidateGeometry(
                candidate_id=cid,
                coordinates=coords,
                atomic_numbers=z,
                energy_variance=var_e,
                max_force_std=max_f_std,
                rotational_constants=rot_consts,
                state=ActiveLearningState.UNLABELED,
            )
            self.pool.append(candidate)

        return self.pool

    def select_active_batch(self) -> list[CandidateGeometry]:
        """Apply acquisition triggering and two-stage deduplication to assemble active batch. [M]"""
        triggered: list[CandidateGeometry] = []

        # 1. Acquisition Filter
        th_f = self.config.force_uncertainty_threshold_ev_per_angstrom
        th_e = self.config.energy_uncertainty_threshold_ev2_per_atom

        for cand in self.pool:
            n_atoms = len(cand.atomic_numbers)
            per_atom_e_var = cand.energy_variance / float(max(1, n_atoms))

            if cand.max_force_std > th_f or per_atom_e_var > th_e:
                cand.state = ActiveLearningState.CANDIDATE_SELECTED
                cand.assigned_tier = route_qm_tier(cand.max_force_std)
                triggered.append(cand)

        # 2. Sort descending by uncertainty score alpha_F^std
        triggered.sort(key=lambda c: c.max_force_std, reverse=True)

        # 3. Two-Stage Deduplication
        rmsd_th = self.config.stage_a_rmsd_threshold_angstrom
        rot_th = self.config.stage_b_rotational_threshold
        capacity = self.config.batch_capacity_k

        self.selected_batch.clear()

        for cand in triggered:
            is_redundant = False
            for selected in self.selected_batch:
                # Require identical chemical composition
                if cand.atomic_numbers != selected.atomic_numbers:
                    continue

                # Stage A: Kabsch SVD RMSD
                d_geom = kabsch_rmsd(cand.coordinates, selected.coordinates)
                if d_geom < rmsd_th:
                    is_redundant = True
                    break

                # Stage B: Rotational Constant Invariance
                if check_stage_b_rotational_redundancy(
                    cand.rotational_constants,
                    selected.rotational_constants,
                    threshold=rot_th,
                ):
                    is_redundant = True
                    break

            if not is_redundant:
                self.selected_batch.append(cand)
                if len(self.selected_batch) >= capacity:
                    break

        return self.selected_batch

    def emit_air_gapped_manifest(
        self,
        manifest_filename: str = "active_learning_manifest.json",
    ) -> tuple[Path, str]:
        """Atomically serialize selected structures into an air-gapped job manifest with SHA-256. [M]"""
        staging_dir = Path(self.config.staging_manifest_dir)
        staging_dir.mkdir(parents=True, exist_ok=True)

        manifest_data = {
            "version": "1.0.0",
            "batch_capacity": self.config.batch_capacity_k,
            "selected_count": len(self.selected_batch),
            "candidates": [
                {
                    "candidate_id": c.candidate_id,
                    "atomic_numbers": c.atomic_numbers,
                    "coordinates": c.coordinates.tolist(),
                    "energy_variance": c.energy_variance,
                    "max_force_std": c.max_force_std,
                    "assigned_tier": c.assigned_tier,
                    "rotational_constants": list(c.rotational_constants),
                    "state": ActiveLearningState.MANIFEST_EMITTED.value,
                }
                for c in self.selected_batch
            ],
        }

        manifest_path = staging_dir / manifest_filename
        tmp_path = staging_dir / f"{manifest_filename}.tmp"

        payload_bytes = json.dumps(manifest_data, indent=2).encode("utf-8")
        sha256_hash = hashlib.sha256(payload_bytes).hexdigest()

        # Atomic serialization: write .tmp -> fsync -> replace
        with open(tmp_path, "wb") as f:
            f.write(payload_bytes)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_path, manifest_path)

        # Accompanying SHA-256 digest file
        digest_path = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
        digest_path.write_text(f"{sha256_hash}  {manifest_path.name}\n", encoding="utf-8")

        # Update candidate state transitions
        for c in self.selected_batch:
            c.state = ActiveLearningState.MANIFEST_EMITTED

        return manifest_path, sha256_hash


# =============================================================================
# ActiveLearner Candidate Gating Controller (Suggestion #79)
# =============================================================================
class ActiveLearner:
    """
    Active Learning candidate gating and epistemic uncertainty controller.
    Mandated by Method Matrix §8A.5 (Integrity Guard G5) and Suggestion #79.
    """

    @classmethod
    def evaluate_candidate_gating(
        cls,
        sigma_e_mev: float,
        threshold_sigma_mev: float = 10.0,
    ) -> dict[str, Any]:
        """
        Evaluate candidate gating based on epistemic uncertainty sigma_E (meV).
        If sigma_E <= threshold_sigma_mev (10.0 meV):
            Action: 'SURROGATE_PREDICT' (query_anchor: False)
        If sigma_E > threshold_sigma_mev (10.0 meV):
            Action: 'QUERY_ANCHOR' (query_anchor: True)
        """
        if not np.isfinite(sigma_e_mev) or sigma_e_mev < 0 or not np.isfinite(threshold_sigma_mev) or threshold_sigma_mev < 0:
            raise ActiveLearningSelectionError("Uncertainty and threshold must be finite and nonnegative.")
        query_anchor = bool(sigma_e_mev > threshold_sigma_mev)
        action = "QUERY_ANCHOR" if query_anchor else "SURROGATE_PREDICT"
        return {
            "action": action,
            "query_anchor": query_anchor,
            "sigma_e_mev": float(sigma_e_mev),
            "threshold_sigma_mev": float(threshold_sigma_mev),
            "advisory_only": True,
            "eligible_for_pruning": False,
        }


# =============================================================================
# Thread-Safe HDF5 Snapshot Persistence Manager (Suggestion #103)
# =============================================================================
class ActiveLearningHDF5Manager:
    """Thread-safe and multi-process HDF5 snapshot persistence manager for Active Learning candidate pool [M].

    Enforces:
    - IPC process synchronization via filelock.FileLock.
    - In-process thread safety via threading.RLock.
    - Companion JSON lease metadata (f"{h5_path}.lease.json").
    - Atomic publication of complete snapshots, safe for dynamic group creation.
    - Fletcher32 data integrity checksums and shuffle filters on chunked datasets.
    """

    def __init__(self, hdf5_path: str | Path, timeout: float = 60.0) -> None:
        self.hdf5_path = Path(hdf5_path).resolve()
        self.timeout = float(timeout)
        self._rlock = threading.RLock()
        self.lock_file = Path(f"{self.hdf5_path}.lock")
        self.lease_file = Path(f"{self.hdf5_path}.lease.json")
        self._file_lock = FileLock(str(self.lock_file), timeout=self.timeout)

        self.hdf5_path.parent.mkdir(parents=True, exist_ok=True)

    @contextlib.contextmanager
    def _acquire_lease(self):
        """Context manager acquiring in-process RLock, cross-process FileLock, and writing companion lease."""
        with self._rlock:
            with self._file_lock:
                current_pid = os.getpid()
                now_ts = time.time()
                payload = {
                    "pid": current_pid,
                    "thread_id": threading.get_ident(),
                    "hostname": platform.node() if hasattr(platform, "node") else "localhost",
                    "timestamp": now_ts,
                    "h5_file": str(self.hdf5_path),
                    "mode": "SNAPSHOT_PUBLICATION",
                }
                staging = self.lease_file.with_name(
                    f"{self.lease_file.name}.tmp.{current_pid}_{threading.get_ident()}_{time.monotonic_ns()}"
                )
                try:
                    staging.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                    os.replace(staging, self.lease_file)
                    yield
                finally:
                    staging.unlink(missing_ok=True)
                    self.lease_file.unlink(missing_ok=True)

    def append_candidate(self, record: dict[str, Any]) -> None:
        """Append without replacement, publishing a complete immutable HDF5 snapshot.

        Dynamic group creation is not performed in SWMR mode. Existing readers
        retain their old inode; subsequent readers see the complete new snapshot.
        """
        cid = str(record.get("candidate_id", f"cand_{uuid.uuid4().hex}"))
        if not cid or "/" in cid or cid in {".", ".."}:
            raise ValueError("Candidate ID must be a nonempty single HDF5 group name.")
        coords = np.asarray(record["coordinates"], dtype=np.float64)
        raw_z = np.asarray(record["atomic_numbers"])
        if (raw_z.ndim != 1 or not len(raw_z) or coords.shape != (len(raw_z), 3)
                or not np.isfinite(coords).all() or not np.isfinite(raw_z).all()
                or np.any(raw_z != np.floor(raw_z)) or np.any((raw_z < 1) | (raw_z > 118))):
            raise ValueError("Candidate requires finite geometry and valid aligned atomic numbers.")
        z = raw_z.astype(np.int32)
        metadata = {k: v for k, v in record.items() if k not in ("coordinates", "atomic_numbers")}
        metadata["candidate_id"] = cid
        metadata_json = json.dumps(metadata, allow_nan=False)
        with self._acquire_lease():
            fd, staging_name = tempfile.mkstemp(prefix=self.hdf5_path.name + ".", suffix=".tmp", dir=self.hdf5_path.parent)
            os.close(fd)
            staging = Path(staging_name)
            try:
                if self.hdf5_path.exists():
                    shutil.copy2(self.hdf5_path, staging)
                    mode = "a"
                else:
                    mode = "w"
                with h5py.File(staging, mode, libver="latest") as handle:
                    group = handle.require_group("candidates")
                    if cid in group:
                        raise ValueError(f"Candidate {cid!r} already exists; overwriting scientific records is forbidden.")
                    candidate = group.create_group(cid)
                    candidate.create_dataset("coordinates", data=coords, chunks=coords.shape, fletcher32=True, shuffle=True)
                    candidate.create_dataset("atomic_numbers", data=z, chunks=z.shape, fletcher32=True, shuffle=True)
                    candidate.attrs["candidate_id"] = cid
                    candidate.attrs["metadata_json"] = metadata_json
                    handle.flush()
                with staging.open("rb") as committed:
                    os.fsync(committed.fileno())
                os.replace(staging, self.hdf5_path)
                directory_fd = os.open(self.hdf5_path.parent, os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            finally:
                staging.unlink(missing_ok=True)

    def read_candidates(self) -> list[dict[str, Any]]:
        """Read one complete snapshot, preserving JSON null and structured metadata."""
        with self._rlock:
            if not self.hdf5_path.exists():
                return []
            records = []
            with h5py.File(self.hdf5_path, "r", libver="latest") as handle:
                if "candidates" not in handle:
                    return []
                for key in sorted(handle["candidates"].keys()):
                    candidate = handle["candidates"][key]
                    if "metadata_json" in candidate.attrs:
                        record = json.loads(candidate.attrs["metadata_json"])
                    else:
                        # Older records retain their original stored attributes;
                        # an empty string cannot be reinterpreted as known null.
                        record = dict(candidate.attrs)
                        record.setdefault("candidate_id", key)
                    record["coordinates"] = np.asarray(candidate["coordinates"])
                    record["atomic_numbers"] = [int(z) for z in candidate["atomic_numbers"]]
                    records.append(record)
            return records


# =============================================================================
# Direct DAG ActiveLearningSampler & Delta-ML Controller (Suggestion #158)
# =============================================================================
class ActiveLearningSampler:
    """Active Learning & Delta-ML PES sampling integration controller (§13.2, Suggestion #158).

    Wires committee epistemic uncertainty gating (Guard G5: sigma > 10 meV) and Delta-ML surface correction.
    """

    def __init__(
        self,
        threshold_sigma_mev: float = 10.0,
        delta_ml_model: Optional[Any] = None,
    ) -> None:
        if not np.isfinite(threshold_sigma_mev) or threshold_sigma_mev < 0:
            raise ActiveLearningSelectionError("Uncertainty threshold must be finite and nonnegative.")
        self.threshold_sigma_mev = threshold_sigma_mev
        self.delta_ml_model = delta_ml_model

    def evaluate_configuration(
        self,
        candidate_geometry: Any,
        scout_energy_ha: float,
        committee_sigma_mev: float,
        anchor_evaluator: Optional[Callable[[Any], float]] = None,
    ) -> dict[str, Any]:
        """Evaluate a single configuration through the Active Learning & Delta-ML gate.

        If committee_sigma_mev > threshold_sigma_mev (10.0 meV):
            Dispatches high-level anchor evaluation (tagged [M]).
        Else:
            Interpolates high-level correction via Delta-ML surrogate (tagged [E]).
        """
        if not np.isfinite(scout_energy_ha):
            raise ActiveLearningSelectionError("A finite scout energy is required.")
        if not np.isfinite(committee_sigma_mev) or committee_sigma_mev < 0:
            raise ActiveLearningSelectionError("Committee uncertainty must be finite and nonnegative.")
        gate_tripped = bool(committee_sigma_mev > self.threshold_sigma_mev)

        if gate_tripped:
            if anchor_evaluator is not None:
                anchor_energy = float(anchor_evaluator(candidate_geometry))
            else:
                raise ActiveLearningSelectionError(
                    "High-level anchor required, but no anchor evaluator is configured; scout result remains unchanged."
                )
            provenance = "[M]"
            final_energy = anchor_energy
            action = "QUERY_ANCHOR"
        else:
            if self.delta_ml_model is None or not callable(getattr(self.delta_ml_model, "predict", None)):
                raise ActiveLearningSelectionError(
                    "Delta-ML prediction unavailable: a trained predictor is required; no zero correction is substituted."
                )
            delta_e = float(self.delta_ml_model.predict(candidate_geometry))
            final_energy = scout_energy_ha + delta_e
            provenance = "[E]"
            action = "SURROGATE_PREDICT"

        if not np.isfinite(final_energy):
            raise ActiveLearningSelectionError("Evaluator returned a non-finite energy.")
        return {
            "action": action,
            "query_anchor": gate_tripped,
            "advisory_only": True,
            "eligible_for_pruning": False,
            "sigma_mev": float(committee_sigma_mev),
            "threshold_mev": float(self.threshold_sigma_mev),
            "energy_hartree": float(final_energy),
            "provenance": provenance,
        }

    def sample_pes_grid(
        self,
        candidates: Sequence[dict[str, Any]],
        anchor_evaluator: Optional[Callable[[Any], float]] = None,
    ) -> list[dict[str, Any]]:
        """Run active learning committee sampling across a candidate PES grid."""
        results = []
        for cand in candidates:
            scout_energy = cand.get("scout_energy", cand.get("energy_hartree"))
            sigma = cand.get("sigma_mev", cand.get("uncertainty_mev"))
            if scout_energy is None or sigma is None or cand.get("coordinates") is None:
                raise ActiveLearningSelectionError("Each candidate requires coordinates, scout energy, and measured committee uncertainty.")
            res = self.evaluate_configuration(
                candidate_geometry=cand.get("coordinates"),
                scout_energy_ha=float(scout_energy),
                committee_sigma_mev=float(sigma),
                anchor_evaluator=anchor_evaluator,
            )
            cand_result = dict(cand)
            cand_result.update(res)
            results.append(cand_result)
        return results


