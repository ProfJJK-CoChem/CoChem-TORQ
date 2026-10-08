"""Actual checkpoint diagnostics and bounded stationary-point classification.

Heuristic flags require scientific review. They do not select a new electronic
state, active space, method, tunneling model or reaction pathway.
"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from ..spectroscopy.harmonic import analyze_hessian
from .pyscf_backend import BackendCalculationError


def verify_native_artifacts(native: dict[str, Any]) -> Path:
    """Check real retained bytes before reading a scientific checkpoint."""
    manifest_path = Path(native["manifest_path"]).resolve()
    if sha256(manifest_path.read_bytes()).hexdigest() != native["manifest_sha256"]:
        raise BackendCalculationError("Native artifact manifest digest mismatch.")
    manifest = json.loads(manifest_path.read_text())
    directory = manifest_path.parent
    for entry in manifest["artifacts"]:
        candidate = directory / entry["path"]
        if candidate.is_symlink() or not candidate.resolve().is_relative_to(directory):
            raise BackendCalculationError(
                "Native artifact path escapes its retained directory."
            )
        if (
            not candidate.is_file()
            or candidate.stat().st_size != entry["size_bytes"]
            or sha256(candidate.read_bytes()).hexdigest() != entry["sha256"]
        ):
            raise BackendCalculationError(
                "Native artifact bytes differ from their retained manifest."
            )
    stored = json.loads((directory / "result.json").read_text())
    for key in (
        "status",
        "method",
        "settings",
        "geometry_bohr",
        "energy_hartree",
        "scf",
        "stability",
        "gradient_hartree_bohr",
        "hessian_hartree_bohr2",
    ):
        if native.get(key) != stored.get(key):
            raise BackendCalculationError(
                "Supplied native values differ from their authenticated retained "
                "result."
            )
    return directory


def electronic_diagnostics(native: dict[str, Any]) -> dict[str, Any]:
    """Read actual occupied/virtual orbitals and a declared restricted state.

    Integer restricted occupations and singlet S2 do not exclude static
    correlation. No coupled-cluster amplitudes, T1/D1 or correlated natural
    occupations are invented from an HF/DFT checkpoint.
    """
    from pyscf import lib

    if native.get("status") != "complete" or not native.get("scf", {}).get("converged"):
        raise BackendCalculationError(
            "Electronic diagnostics require a converged real native calculation."
        )
    directory = verify_native_artifacts(native)
    checkpoint = directory / "wavefunction.chk"
    if not checkpoint.is_file():
        checkpoint = directory / "final" / "wavefunction.chk"
    if not checkpoint.is_file():
        raise BackendCalculationError(
            "No retained final real SCF checkpoint is available."
        )
    mol, data = (
        lib.chkfile.load_mol(str(checkpoint)),
        lib.chkfile.load(str(checkpoint), "scf"),
    )
    if not np.allclose(
        mol.atom_coords(unit="Bohr"), native["geometry_bohr"], atol=1e-10, rtol=0
    ):
        raise BackendCalculationError(
            "Checkpoint geometry differs from the claimed native result."
        )
    occupations = np.asarray(data["mo_occ"], dtype=float)
    energies = np.asarray(data["mo_energy"], dtype=float)
    if (
        occupations.ndim != 1
        or energies.shape != occupations.shape
        or not np.isfinite(occupations).all()
        or not np.isfinite(energies).all()
        or not np.all(np.isin(occupations, [0.0, 2.0]))
        or abs(float(np.sum(occupations)) - mol.nelectron) > 1e-8
        or mol.spin != 0
    ):
        raise BackendCalculationError(
            "Checkpoint does not support the claimed restricted singlet identity."
        )
    if abs(float(data["e_tot"]) - native["scf"]["energy_hartree"]) > 1e-8:
        raise BackendCalculationError(
            "SCF checkpoint energy differs from the actual native result."
        )
    occupied, virtual = energies[occupations == 2], energies[occupations == 0]
    gap = (
        float(np.min(virtual) - np.max(occupied))
        if len(occupied) and len(virtual)
        else None
    )
    review = []
    if native.get("stability", {}).get("status") != "stable":
        review.append("restricted_wavefunction_stability_not_accepted")
    if gap is not None and gap <= 0:
        review.append("nonpositive_occupied_virtual_gap")
    return {
        "schema_version": "cochem.torq.electronic-diagnostics/1",
        "status": "available",
        "reference": "restricted closed-shell real orbitals",
        "electron_count": int(mol.nelectron),
        "occupations": occupations.tolist(),
        "orbital_energies_hartree": energies.tolist(),
        "homo_lumo_gap_hartree": gap,
        "spin_squared": 0.0,
        "spin_squared_definition": (
            "exact restricted closed-shell determinant identity, not a "
            "correlated multireference diagnostic"
        ),
        "scf_stability": native.get("stability"),
        "unavailable": {
            "T1": (
                "A separately requested actual correlated-amplitude calculation "
                "with a verified diagnostic convention is required."
            ),
            "D1": (
                "A separately requested actual correlated-amplitude calculation "
                "with a verified diagnostic convention is required."
            ),
            "correlated_natural_occupations": (
                "The retained SCF checkpoint contains no correlated one-particle "
                "density matrix."
            ),
        },
        "review_conditions": review,
        "automatic_model_or_state_changes": False,
        "source_manifest_sha256": native["manifest_sha256"],
        "checkpoint_sha256": sha256(checkpoint.read_bytes()).hexdigest(),
        "accuracy_flags": [
            "no_single_reference_sufficiency_proof",
            "no_unique_scf_root_proof",
        ],
    }


def stationary_point_diagnostics(
    native: dict[str, Any],
    masses_u: list[float],
    *,
    gradient_max_tolerance_hartree_bohr: float = 1.5e-5,
    harmonic_symmetry_relative_tolerance: float = 1e-8,
    external_residual_relative_tolerance: float = 1e-4,
    zero_frequency_tolerance_cm1: float = 0.1,
) -> dict[str, Any]:
    """Classify numerical stationarity without certifying reaction identity."""
    thresholds = (
        gradient_max_tolerance_hartree_bohr,
        harmonic_symmetry_relative_tolerance,
        external_residual_relative_tolerance,
    )
    if (
        any(not np.isfinite(value) or value <= 0 for value in thresholds)
        or not np.isfinite(zero_frequency_tolerance_cm1)
        or zero_frequency_tolerance_cm1 < 0
    ):
        raise ValueError(
            "Stationarity/invariance tolerances must be finite and positive."
        )
    if (
        native.get("status") != "complete"
        or native.get("stability", {}).get("status") != "stable"
    ):
        raise BackendCalculationError(
            "Stationary classification requires a complete stable native calculation."
        )
    verify_native_artifacts(native)
    geometry = np.asarray(native["geometry_bohr"], dtype=float)
    gradient = np.asarray(native.get("gradient_hartree_bohr"), dtype=float)
    if gradient.shape != geometry.shape or not np.isfinite(gradient).all():
        raise BackendCalculationError(
            "No finite actual gradient is available for stationary classification."
        )
    harmonic = analyze_hessian(
        geometry,
        masses_u,
        native["hessian_hartree_bohr2"],
        symmetry_tolerance=harmonic_symmetry_relative_tolerance,
        zero_frequency_tolerance_cm1=zero_frequency_tolerance_cm1,
    )
    maximum_gradient = float(np.max(np.abs(gradient)))
    negatives = np.flatnonzero(harmonic.frequencies_cm1 < -zero_frequency_tolerance_cm1)
    zeros = np.flatnonzero(
        np.abs(harmonic.frequencies_cm1) <= zero_frequency_tolerance_cm1
    )
    if maximum_gradient > gradient_max_tolerance_hartree_bohr:
        classification = "not_stationary"
    elif harmonic.external_residual_relative > external_residual_relative_tolerance:
        classification = "external_invariance_failed"
    elif len(zeros):
        classification = "unresolved_internal_zero_modes"
    elif len(negatives) == 0:
        classification = "minimum"
    elif len(negatives) == 1:
        classification = "first_order_saddle_candidate"
    else:
        classification = "higher_order_saddle"
    return {
        "schema_version": "cochem.torq.stationary-diagnostics/1",
        "classification": classification,
        "maximum_gradient_hartree_bohr": maximum_gradient,
        "gradient_max_tolerance_hartree_bohr": gradient_max_tolerance_hartree_bohr,
        "imaginary_mode_count": int(len(negatives)),
        "unresolved_internal_mode_count": int(len(zeros)),
        "signed_frequencies_cm1": harmonic.frequencies_cm1.tolist(),
        "imaginary_cartesian_modes": harmonic.cartesian_modes[:, negatives].T.tolist(),
        "external_rank": harmonic.external_rank,
        "external_residual_relative": harmonic.external_residual_relative,
        "external_residual_relative_tolerance": external_residual_relative_tolerance,
        "harmonic_source_digest": harmonic.source_digest,
        "source_manifest_sha256": native["manifest_sha256"],
        "ts_verified": False,
        "missing_ts_verification": [
            "chemically relevant mode acceptance",
            "mapped electronic/structural endpoint identity",
            ("both-direction IRC or explicitly accepted connectivity alternative"),
        ],
        "automatic_model_or_state_changes": False,
    }
