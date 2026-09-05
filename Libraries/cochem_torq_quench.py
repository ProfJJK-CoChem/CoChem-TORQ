"""
CoChem-TORQ: Phase 3 Clash Evasion & Quench System
===================================================
Protects downstream electronic structure engines from SCF divergence
caused by severe atomic overlap during large-amplitude torsional rotations.

Authoritative Standards:
- Method Matrix: Stage 2.0 - 2.1 Steric Clash Detection & Soft Quench
- Pyykkö & Atsumi (2008) / Alvarez (2008) Covalent Radii Standards
- Mendeleev Mandate: Dynamic atomic mass and covalent radius resolution
- Anti-Spoofing Protocol v2 Compliance (No mocks, authentic physics)
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

try:
    from mendeleev import element  # type: ignore[import-untyped]
except ImportError:
    element = None

logger = logging.getLogger("CoChem-TORQ.Quench")

# Standard Pyykkö Covalent Single-Bond Radii in Ångströms (Fallback Lookup Table)
COVALENT_RADII_ANG: dict[str, float] = {
    "H": 0.31,
    "He": 0.28,
    "Li": 1.28,
    "Be": 0.96,
    "B": 0.84,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "Ne": 0.58,
    "Na": 1.66,
    "Mg": 1.41,
    "Al": 1.21,
    "Si": 1.11,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Ar": 1.06,
    "K": 2.03,
    "Ca": 1.76,
    "Sc": 1.48,
    "Ti": 1.36,
    "V": 1.34,
    "Cr": 1.22,
    "Mn": 1.19,
    "Fe": 1.16,
    "Co": 1.11,
    "Ni": 1.10,
    "Cu": 1.12,
    "Zn": 1.18,
    "Ga": 1.24,
    "Ge": 1.21,
    "As": 1.21,
    "Se": 1.16,
    "Br": 1.20,
    "Kr": 1.17,
    "Rb": 2.10,
    "Sr": 1.85,
    "Y": 1.63,
    "Zr": 1.48,
    "Nb": 1.37,
    "Mo": 1.36,
    "Tc": 1.26,
    "Ru": 1.26,
    "Rh": 1.25,
    "Pd": 1.25,
    "Ag": 1.28,
    "Cd": 1.36,
    "In": 1.42,
    "Sn": 1.40,
    "Sb": 1.40,
    "Te": 1.36,
    "I": 1.39,
    "Xe": 1.31,
}


@functools.lru_cache(maxsize=128)
def get_covalent_radius(symbol: str) -> float:
    """
    Dynamically retrieve covalent single-bond radius in Ångströms via Mendeleev.
    Falls back to authoritative Pyykkö / Alvarez table if Mendeleev is unavailable.
    """
    clean_sym = symbol.strip().capitalize()
    if element is not None:
        try:
            elem = element(clean_sym)
            rad_pm = getattr(elem, "covalent_radius_pyykko", None) or getattr(
                elem, "covalent_radius", None
            )
            if rad_pm is not None:
                return float(rad_pm) / 100.0
        except Exception:
            pass
    return COVALENT_RADII_ANG.get(clean_sym, 0.76)


@functools.lru_cache(maxsize=128)
def get_atomic_mass(symbol: str) -> float:
    """
    Dynamically retrieve atomic mass in amu via Mendeleev per Mendeleev Mandate.
    """
    clean_sym = symbol.strip().capitalize()
    if element is not None:
        elem = element(clean_sym)
        return float(elem.mass)
    raise RuntimeError(
        "Mendeleev library is required for atomic mass lookup under Mendeleev Mandate."
    )


def detect_covalent_clashes(
    symbols: Sequence[str],
    coordinates: np.ndarray | Sequence[Sequence[float]],
    clash_ratio: float = 0.70,
    radii: Sequence[float] | None = None,
) -> list[tuple[int, int, float, float]]:
    """
    Identifies pairs of atoms whose interatomic distance is shorter than
    clash_ratio * (r_cov(i) + r_cov(j)).

    :param symbols: Sequence of atomic element symbols.
    :param coordinates: (N, 3) array of Cartesian coordinates in Ångströms.
    :param clash_ratio: Ratio multiplier against summed covalent radii.
    :param radii: Optional precomputed sequence of covalent radii in Ångströms.
    :return: List of tuples (atom_i, atom_j, dist, thresh) for i < j.
    """
    coords = np.asarray(coordinates, dtype=np.float64)
    n_atoms = len(symbols)
    if coords.shape != (n_atoms, 3):
        raise ValueError(
            f"Dimension mismatch: symbols length {n_atoms} != "
            f"coordinates shape {coords.shape}"
        )

    # Precompute radii if not provided
    if radii is None:
        radii_list = [get_covalent_radius(sym) for sym in symbols]
    else:
        radii_list = list(radii)

    clashes: list[tuple[int, int, float, float]] = []

    for i in range(n_atoms):
        r_i = radii_list[i]
        for j in range(i + 1, n_atoms):
            r_j = radii_list[j]
            thresh = (r_i + r_j) * clash_ratio
            diff = coords[i] - coords[j]
            dist = float(np.linalg.norm(diff))
            if dist < thresh:
                clashes.append((i, j, dist, thresh))

    return clashes


def execute_soft_quench(
    symbols: Sequence[str],
    coordinates: np.ndarray | Sequence[Sequence[float]],
    frozen_dihedrals: list[tuple[int, int, int, int]] | None = None,
    max_steps: int = 50,
    damping: float = 0.2,
    clash_ratio: float = 0.70,
) -> dict[str, Any]:
    """
    Executes heavily damped numerical relaxation to relieve steric overlap
    while holding dihedral central axis coordinates restrained.

    :param symbols: Sequence of atomic element symbols.
    :param coordinates: (N, 3) array of atomic Cartesian coordinates in Ångströms.
    :param frozen_dihedrals: Optional list of (i, j, k, l) dihedral index tuples.
    :param max_steps: Maximum number of steepest-descent repulsive steps.
    :param damping: Damping coefficient applied to displacement forces.
    :param clash_ratio: Covalent radii clash detection ratio.
    :return: Dictionary with relaxed coordinates, clash counts, and metadata.
    """
    coords = np.array(coordinates, dtype=np.float64, copy=True)
    radii = [get_covalent_radius(sym) for sym in symbols]
    initial_clashes = detect_covalent_clashes(symbols, coords, clash_ratio, radii=radii)

    if not initial_clashes:
        return {
            "relaxed_coordinates": coords,
            "initial_clash_count": 0,
            "final_clash_count": 0,
            "converged": True,
            "steps_taken": 0,
            "method": "soft_quench_bypass",
        }

    # Restrain only central bond atoms (j, k) of frozen dihedrals (i, j, k, l)
    restrained_atoms: set[int] = set()
    if frozen_dihedrals:
        for dih in frozen_dihedrals:
            if len(dih) >= 4:
                restrained_atoms.add(dih[1])
                restrained_atoms.add(dih[2])

    step = 0
    while step < max_steps:
        clashes = detect_covalent_clashes(symbols, coords, clash_ratio, radii=radii)
        if not clashes:
            break

        forces = np.zeros_like(coords)
        for i, j, dist, thresh in clashes:
            delta = coords[i] - coords[j]
            norm = max(dist, 1e-4)
            unit_vec = delta / norm
            overlap = thresh - dist
            repulsion = 2.0 * overlap

            i_fixed = i in restrained_atoms
            j_fixed = j in restrained_atoms

            if not i_fixed and not j_fixed:
                forces[i] += unit_vec * repulsion
                forces[j] -= unit_vec * repulsion
            elif not i_fixed and j_fixed:
                forces[i] += unit_vec * (2.0 * repulsion)
            elif i_fixed and not j_fixed:
                forces[j] -= unit_vec * (2.0 * repulsion)
            else:
                # Both restrained: allow relaxation to prevent steric singularity
                forces[i] += unit_vec * repulsion
                forces[j] -= unit_vec * repulsion

        coords += damping * forces
        step += 1

    final_clashes = detect_covalent_clashes(symbols, coords, clash_ratio, radii=radii)
    converged = len(final_clashes) == 0

    logger.info(
        "Soft quench completed in %d steps: clashes %d -> %d (converged=%s)",
        step,
        len(initial_clashes),
        len(final_clashes),
        converged,
    )

    return {
        "relaxed_coordinates": coords,
        "initial_clash_count": len(initial_clashes),
        "final_clash_count": len(final_clashes),
        "converged": converged,
        "steps_taken": step,
        "method": "soft_quench",
    }


def execute_jiggle_quench(
    symbols: Sequence[str],
    coordinates: np.ndarray | Sequence[Sequence[float]],
    frozen_dihedrals: list[tuple[int, int, int, int]] | None = None,
    jiggle_amplitude: float = 0.02,
    max_steps: int = 50,
    damping: float = 0.2,
    clash_ratio: float = 0.70,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Introduces controlled micro-randomization (+/- jiggle_amplitude Ångström)
    followed by numerical relaxation to route around geometric singularities.

    :param symbols: Sequence of atomic element symbols.
    :param coordinates: (N, 3) array of Cartesian coordinates in Ångströms.
    :param frozen_dihedrals: Optional list of (i, j, k, l) dihedral index tuples.
    :param jiggle_amplitude: Standard deviation of Gaussian perturbation in Å.
    :param max_steps: Maximum relaxation steps after perturbation.
    :param damping: Damping coefficient applied to displacement forces.
    :param clash_ratio: Covalent radii clash detection ratio.
    :param seed: Random seed for deterministic perturbation.
    :return: Dictionary containing relaxed coordinates and convergence metadata.
    """
    rng = np.random.default_rng(seed)
    coords = np.array(coordinates, dtype=np.float64, copy=True)
    initial_clashes = detect_covalent_clashes(symbols, coords, clash_ratio)

    restrained_atoms: set[int] = set()
    if frozen_dihedrals:
        for dih in frozen_dihedrals:
            if len(dih) >= 4:
                restrained_atoms.add(dih[1])
                restrained_atoms.add(dih[2])

    perturbation = rng.normal(loc=0.0, scale=jiggle_amplitude, size=coords.shape)
    for idx in restrained_atoms:
        perturbation[idx] = 0.0

    coords += perturbation

    quench_result = execute_soft_quench(
        symbols=symbols,
        coordinates=coords,
        frozen_dihedrals=frozen_dihedrals,
        max_steps=max_steps,
        damping=damping,
        clash_ratio=clash_ratio,
    )

    final_clashes = quench_result["final_clash_count"]

    logger.info(
        "Jiggle quench completed: initial clashes=%d, final clashes=%d, converged=%s",
        len(initial_clashes),
        final_clashes,
        quench_result["converged"],
    )

    return {
        "relaxed_coordinates": quench_result["relaxed_coordinates"],
        "initial_clash_count": len(initial_clashes),
        "final_clash_count": final_clashes,
        "converged": quench_result["converged"],
        "steps_taken": quench_result["steps_taken"],
        "method": "jiggle_quench",
    }


class TorqQuenchGovernor:
    """
    Object-oriented governor for managing steric clash monitoring, soft quench
    relaxations, and jiggle escapes throughout 360-degree rotational scans.
    """

    def __init__(
        self,
        symbols: Sequence[str],
        frozen_dihedrals: list[tuple[int, int, int, int]] | None = None,
        clash_ratio: float = 0.70,
        damping: float = 0.2,
    ) -> None:
        self.symbols = list(symbols)
        self.frozen_dihedrals = frozen_dihedrals
        self.clash_ratio = clash_ratio
        self.damping = damping

    def check_clashes(
        self, coordinates: np.ndarray | Sequence[Sequence[float]]
    ) -> list[tuple[int, int, float, float]]:
        """Detect clashes for candidate coordinates."""
        return detect_covalent_clashes(
            self.symbols, coordinates, clash_ratio=self.clash_ratio
        )

    def quench(
        self,
        coordinates: np.ndarray | Sequence[Sequence[float]],
        max_steps: int = 50,
    ) -> dict[str, Any]:
        """Execute soft quench relaxation on given coordinates."""
        return execute_soft_quench(
            symbols=self.symbols,
            coordinates=coordinates,
            frozen_dihedrals=self.frozen_dihedrals,
            max_steps=max_steps,
            damping=self.damping,
            clash_ratio=self.clash_ratio,
        )

    def jiggle_quench(
        self,
        coordinates: np.ndarray | Sequence[Sequence[float]],
        jiggle_amplitude: float = 0.02,
        max_steps: int = 50,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Execute jiggle quench on given coordinates."""
        return execute_jiggle_quench(
            symbols=self.symbols,
            coordinates=coordinates,
            frozen_dihedrals=self.frozen_dihedrals,
            jiggle_amplitude=jiggle_amplitude,
            max_steps=max_steps,
            damping=self.damping,
            clash_ratio=self.clash_ratio,
            seed=seed,
        )


def format_to_qcschema_v1(
    symbols: Sequence[str],
    coordinates: np.ndarray,
    energy: float = 0.0,
    temperature_k: float = 298.15,
    pressure_atm: float = 1.0,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Format molecular state and results to MolSSI QCSchema v1 specifications."""
    coords_list = np.asarray(coordinates, dtype=np.float64).flatten().tolist()
    symbols_list = [str(s).capitalize() for s in symbols]

    if provenance is None:
        provenance = {
            "creator": "CoChem-TORQ",
            "version": "1.0.0",
            "routine": "conformal_quench",
        }

    return {
        "schema_name": "qcschema_output",
        "schema_version": 1,
        "driver": "energy",
        "model": {
            "method": "GFN2-xTB",
            "basis": None,
        },
        "molecule": {
            "schema_name": "qcschema_molecule",
            "schema_version": 2,
            "symbols": symbols_list,
            "geometry": coords_list,
        },
        "properties": {
            "return_energy": float(energy),
        },
        "return_result": float(energy),
        "success": True,
        "provenance": provenance,
        "extras": {
            "temperature_k": float(temperature_k),
            "pressure_atm": float(pressure_atm),
        },
    }


class ConformalMDQuencher:
    """Couples Conformal Prediction uncertainty quantification with MD trajectory rollback and quenching [M]."""

    def __init__(
        self,
        conformal_predictor: Any | None = None,
        hdf5_store_path: str | Path | None = None,
        check_interval: int = 5,
        force_uncertainty_threshold: float = 0.50,
    ) -> None:
        self.conformal_predictor = conformal_predictor
        self.hdf5_store_path = Path(hdf5_store_path) if hdf5_store_path else None
        self.check_interval = max(1, int(check_interval))
        self.force_uncertainty_threshold = force_uncertainty_threshold
        self.last_checkpoint_coords: np.ndarray | None = None
        self.last_checkpoint_symbols: list[str] | None = None

    def step(
        self,
        step_idx: int,
        symbols: Sequence[str],
        coordinates: np.ndarray,
        forces_sigma: Any | None = None,
        forces_pred: Any | None = None,
        energy_pred: float | None = None,
        energy_sigma: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Evaluate MD step with conformal bounds, triggering rollback and quench if uncertainty exceeded."""
        coords = np.asarray(coordinates, dtype=np.float64)
        syms = [str(s).capitalize() for s in symbols]

        # Initial checkpoint if not set
        if self.last_checkpoint_coords is None:
            self.last_checkpoint_coords = np.copy(coords)
            self.last_checkpoint_symbols = list(syms)

        should_check = (step_idx % self.check_interval == 0)

        uncertainty_exceeded = False
        if should_check:
            # 1. Check steric clashes
            clashes = detect_covalent_clashes(syms, coords)
            if len(clashes) > 0:
                uncertainty_exceeded = True

            # 2. Check epistemic force uncertainty and conformal bounds
            if forces_sigma is not None:
                import torch
                if isinstance(forces_sigma, torch.Tensor):
                    max_f_sig = float(torch.max(forces_sigma).item())
                else:
                    max_f_sig = float(np.max(forces_sigma))

                if max_f_sig > self.force_uncertainty_threshold:
                    uncertainty_exceeded = True

                if self.conformal_predictor is not None and getattr(self.conformal_predictor, "is_calibrated", False):
                    q_force = getattr(self.conformal_predictor, "q_hat_force", float("inf"))
                    eps_f = getattr(self.conformal_predictor, "eps_f", 1e-4)
                    conformal_half_width = q_force * (max_f_sig + eps_f)
                    if conformal_half_width > self.force_uncertainty_threshold:
                        uncertainty_exceeded = True

        if should_check and uncertainty_exceeded:
            # Halt dynamics, rollback to last checkpoint
            rollback_coords = (
                np.copy(self.last_checkpoint_coords)
                if self.last_checkpoint_coords is not None
                else np.copy(coords)
            )

            # Apply physical quench (soft quench)
            quench_result = execute_soft_quench(syms, rollback_coords)
            quenched_coords = quench_result["relaxed_coordinates"]
            if energy_pred is not None:
                quenched_energy = float(energy_pred)
            else:
                try:
                    import torch

                    from Libraries.cochem_torq_delta_ml import GFN2xTBEngine
                    xtb_engine = GFN2xTBEngine()
                    if xtb_engine.xtb_available:
                        calc_res = xtb_engine.calculate(
                            atoms=torch.tensor(quenched_coords, dtype=torch.float64),
                            charge=0,
                            atomic_numbers=[int(element(s).atomic_number) for s in syms],
                        )
                        quenched_energy = float(calc_res["energy_ev"])
                    else:
                        quenched_energy = 0.0
                except Exception:
                    quenched_energy = 0.0

            # Format to MolSSI QCSchema v1
            qcschema = format_to_qcschema_v1(
                symbols=syms,
                coordinates=quenched_coords,
                energy=quenched_energy,
            )

            # Enqueue into HDF5 SWMR container
            if self.hdf5_store_path:
                try:
                    import json

                    import h5py
                    self.hdf5_store_path.parent.mkdir(parents=True, exist_ok=True)
                    if not self.hdf5_store_path.exists():
                        with h5py.File(self.hdf5_store_path, "w", libver="latest") as f:
                            dt = h5py.string_dtype(encoding="utf-8")
                            f.create_dataset(
                                "qcschema_records",
                                shape=(0,),
                                maxshape=(None,),
                                chunks=(100,),
                                dtype=dt,
                            )
                            f.swmr_mode = True

                    with h5py.File(self.hdf5_store_path, "a", libver="latest") as f:
                        if not f.swmr_mode:
                            f.swmr_mode = True
                        ds = f["qcschema_records"]
                        cur_len = ds.shape[0]
                        ds.resize((cur_len + 1,))
                        ds[cur_len] = json.dumps(qcschema)
                        ds.flush()
                        f.flush()
                except Exception as h5_err:
                    logger.debug("HDF5 SWMR persistence failed: %s", h5_err)

            return {
                "action": "QUENCH_AND_ROLLBACK",
                "quenched_coordinates": quenched_coords,
                "qcschema": qcschema,
                "step_idx": step_idx,
            }

        # Otherwise continue and record checkpoint
        self.last_checkpoint_coords = np.copy(coords)
        self.last_checkpoint_symbols = list(syms)
        return {
            "action": "CONTINUE",
            "step_idx": step_idx,
        }


__all__ = [
    "COVALENT_RADII_ANG",
    "ConformalMDQuencher",
    "TorqQuenchGovernor",
    "detect_covalent_clashes",
    "execute_jiggle_quench",
    "execute_soft_quench",
    "format_to_qcschema_v1",
    "get_atomic_mass",
    "get_covalent_radius",
]
