"""Explicit recipes; the historical method matrix never acts as an enable switch."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .domain import digest

_BASE = {
    "engine": "PySCF",
    "engine_version": "2.14.0",
    "reference": "restricted",
    "optimizer": "geomeTRIC",
    "optimizer_version": "1.1.1",
    "frozen_core": False,
    "density_fitting": False,
    "dispersion": None,
    "relativity": "none",
    "solvent": None,
    "constraints": "none",
    "derivatives": {"gradient": "analytic", "hessian": "analytic"},
    "numerical": {
        "scf_energy_tolerance": 1e-11,
        "scf_gradient_tolerance": 1e-7,
        "check_stability": True,
    },
    "max_atoms": 12,
    "elements": ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne"],
    "multiplicity": 1,
    "harmonic_external_residual_relative_tolerance": 1e-4,
    "geometry_method_equals_property_method": True,
    "products": [
        "geometry",
        "harmonic",
        "equilibrium_constants",
        "rigid_rotor_catalog",
    ],
    "accuracy": "uncalibrated; no laboratory/astronomical identification claim",
}
_PROFILES = {
    "hf-sto-3g-education": {
        **_BASE,
        "id": "hf-sto-3g-education",
        "version": 1,
        "label": "RHF / STO-3G teaching calculation",
        "method": "hf",
        "basis": "sto-3g",
        "availability": "experimental",
        "runnable": True,
        "reason": "Real restricted HF teaching baseline. Numerical qualification is scoped to tested molecules; experimental spectroscopy accuracy is uncalibrated.",
        "matrix_refs": [],
        "method_family": None,
        "citation": "Roothaan, Rev. Mod. Phys. 23, 69 (1951), doi:10.1103/RevModPhys.23.69",
    },
    "hf-cc-pvdz-research": {
        **_BASE,
        "id": "hf-cc-pvdz-research",
        "version": 1,
        "label": "RHF / cc-pVDZ validation experiment",
        "method": "hf",
        "basis": "cc-pvdz",
        "availability": "experimental",
        "runnable": True,
        "reason": "Explicit research validation experiment; Hartree–Fock omits correlation and dispersion. It is not a matrix high-accuracy substitute.",
        "matrix_refs": [],
        "method_family": None,
        "citation": "Dunning, J. Chem. Phys. 90, 1007 (1989), doi:10.1063/1.456153",
    },
    "revdsd-pbep86-d4-experimental": {
        "id": "revdsd-pbep86-d4-experimental",
        "version": 1,
        "label": "Exact revDSD-PBEP86-D4 research milestone",
        "engine": "TORQ/PySCF research",
        "method": "revdsd-pbep86-d4",
        "basis": "jun-cc-pVTZ",
        "availability": "documented",
        "runnable": False,
        "products": [],
        "reason": "Activation requires primary-source D4 recipe verification, independent energy comparison, orbital-response derivatives and separately validated higher derivatives.",
        "matrix_refs": [],
        "method_family": "F05",
    },
}

for identifier, name, family in (
    ("pbe-d4-def2-svp-validation", "pbe-d4", "F03"),
    ("b3lyp-d4-def2-svp-validation", "b3lyp-d4", "F04"),
    ("mp2-cc-pvdz-validation", "mp2", "F06"),
):
    correlated = name == "mp2"
    _PROFILES[identifier] = {
        **deepcopy(_BASE),
        "id": identifier,
        "version": 1,
        "label": f"{name.upper()} / {'cc-pVDZ' if correlated else 'def2-SVP'} local validation",
        "method": name,
        "basis": "cc-pvdz" if correlated else "def2-svp",
        "dispersion": None if correlated else "d4",
        "dispersion_version": None if correlated else "3.7.0",
        "availability": "experimental",
        "runnable": True,
        "reason": "Genuine CPU validation experiment with tested finite-difference derivatives; requires explicit local execution. Experimental spectroscopy accuracy remains uncalibrated.",
        "products": ["geometry", "harmonic", "equilibrium_constants"],
        "matrix_refs": [],
        "method_family": family,
        "derivatives": {
            "gradient": "analytic",
            "hessian": "centered_difference_of_analytic_gradient",
        },
        "harmonic_symmetry_relative_tolerance": 1e-5,
        "numerical": {
            **_BASE["numerical"],
            "hessian_step_bohr": 0.002 if not correlated else 0.005,
        },
        "qualification_evidence": "benchmarks/generated/pyscf_cpu_qualification.json; exact tested tuples only",
    }

_PROFILES["hf-sto-3g-anharmonic-validation"] = {
    **deepcopy(_PROFILES["hf-sto-3g-education"]),
    "id": "hf-sto-3g-anharmonic-validation",
    "label": "RHF / STO-3G local anharmonic validation experiment",
    "max_atoms": 3,
    "reason": "Actual finite-displacement vibrational force field, bounded to three modes. Rotational VPT2/B0 and identification qualification remain blocked.",
    "products": [
        "geometry",
        "harmonic",
        "equilibrium_constants",
        "anharmonic_force_field",
    ],
    "numerical": {**_BASE["numerical"], "scf_energy_tolerance": 1e-12},
    "anharmonic": {
        "max_modes": 3,
        "steps_dimensionless": [0.08, 0.04],
        "max_evaluations": 200,
        "absolute_tolerance_hartree": 1e-6,
        "relative_tolerance": 0.03,
        "scientific_scope": "experimental finite-displacement vibrational field; independent molecular accuracy unqualified",
    },
}


def get_profile(identifier: str) -> dict[str, Any]:
    if identifier not in _PROFILES:
        raise ValueError(
            f"Unknown recipe {identifier!r}; historical row aliases require explicit mapping."
        )
    result = deepcopy(_PROFILES[identifier])
    result["recipe_sha256"] = digest(result)
    return result


def list_method_profiles() -> list[dict[str, Any]]:
    return [get_profile(identifier) for identifier in _PROFILES]


def matrix_index() -> list[dict[str, str]]:
    """Preserve all 140 table/track/budget identities without inventing engine capability."""
    budgets = ["10s", "1min", "30min", "1h", "3h", "12h", "1d", "3d", "1w", "1mo"]
    tracks = [
        "T1",
        "T2",
        "T3O",
        "T3C",
        "T4O",
        "T4C",
        "T5",
        "T6O",
        "T6C",
        "T7",
        "T8O",
        "T8C",
        "T9",
        "T10",
    ]
    return [
        {
            "legacy_row_id": f"{track}-{budget}",
            "availability": "documented",
            "dispatch": "requires_exact_qualified_recipe",
            "budget_label": budget,
            "owner": "TOPOS"
            if track == "T1"
            else "sibling_handoff"
            if track in {"T8O", "T8C", "T9", "T10"}
            else "TORQ",
        }
        for track in tracks
        for budget in budgets
    ]
