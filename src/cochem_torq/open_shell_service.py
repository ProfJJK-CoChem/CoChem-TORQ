"""Typed local observations of explicitly requested unrestricted electronic states.

These are unprojected determinant observations, not exact-spin electronic terms,
benchmark-qualified energies, equilibrium minima or spectroscopy predictions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field, StrictFloat, StrictInt, model_validator
from typing_extensions import Self

from .domain import CalculationRequest, Molecule, digest, scientific_cache_key
from .scientific_values import PhysicalValue


class UnrestrictedSCF(PhysicalValue):
    converged: Literal[True]
    energy_hartree: StrictFloat
    cycles: StrictInt = Field(ge=0)
    electron_count: StrictInt = Field(gt=0)
    alpha_electrons: StrictInt = Field(ge=1)
    beta_electrons: StrictInt = Field(ge=0)
    requested_multiplicity: StrictInt = Field(ge=2, le=7)
    requested_two_ms: StrictInt = Field(ge=1, le=6, alias="requested_two_Ms")
    native_molecular_spin: StrictInt = Field(ge=1, le=6)
    electronic_energy_excluding_dispersion_hartree: StrictFloat | None = None

    @model_validator(mode="after")
    def preserve_sector(self) -> Self:
        if (
            self.alpha_electrons + self.beta_electrons != self.electron_count
            or self.alpha_electrons - self.beta_electrons != self.requested_two_ms
            or self.requested_two_ms != self.requested_multiplicity - 1
            or self.native_molecular_spin != self.requested_two_ms
        ):
            raise ValueError(
                "Native electron counts do not preserve the requested M_S sector."
            )
        return self


class UnrestrictedSpin(PhysicalValue):
    definition: Literal["Unprojected unrestricted determinant expectation <S^2>"]
    spin_squared: StrictFloat = Field(ge=0)
    requested_spin_squared: StrictFloat = Field(ge=0)
    spin_contamination: StrictFloat
    effective_multiplicity_from_spin_expectation: StrictFloat = Field(ge=1)
    requested_multiplicity: StrictInt = Field(ge=2, le=7)
    effective_multiplicity_is_state_label: Literal[False]
    spin_projection_applied: Literal[False]
    declared_contamination_tolerance: StrictFloat = Field(gt=0)

    @model_validator(mode="after")
    def spin_expectation(self) -> Self:
        spin = (self.requested_multiplicity - 1) / 2
        if (
            abs(self.requested_spin_squared - spin * (spin + 1)) > 1e-12
            or abs(
                self.spin_squared
                - self.requested_spin_squared
                - self.spin_contamination
            )
            > 1e-12
            or abs(
                self.effective_multiplicity_from_spin_expectation
                - np.sqrt(1 + 4 * self.spin_squared)
            )
            > 1e-10
        ):
            raise ValueError(
                "Spin diagnostics disagree with the actual expectation "
                "and requested sector."
            )
        return self


class UnrestrictedStability(PhysicalValue):
    status: Literal["stable", "unstable", "unavailable", "not_requested"]
    internal_stable: bool | None
    external_stable: bool | None
    internal_scope: Literal["unrestricted real alpha/beta orbital variations"]
    external_scope: Literal["native returned UHF/UKS -> GHF/GKS status only"]
    real_to_complex_status: Literal["native_log_only_not_separately_extracted"]
    requested_state_preserved: Literal[True]
    candidate_orbitals_applied: Literal[False]
    electronic_branch_continuity: Literal["not_qualified"]
    solver_source_sha256: str = Field(pattern="^[0-9a-f]{64}$")
    controls: dict[str, Any]
    orbital_sha256_before: str = Field(pattern="^[0-9a-f]{64}$")
    orbital_sha256_after: str = Field(pattern="^[0-9a-f]{64}$")
    checkpoint_sha256_before: str = Field(pattern="^[0-9a-f]{64}$")
    checkpoint_sha256_after: str = Field(pattern="^[0-9a-f]{64}$")
    reference_unchanged: Literal[True]
    diagnostics: dict[str, Any]

    @model_validator(mode="after")
    def native_reference_unchanged(self) -> Self:
        if (
            self.orbital_sha256_before != self.orbital_sha256_after
            or self.checkpoint_sha256_before != self.checkpoint_sha256_after
        ):
            raise ValueError("Stability observations changed the reference.")
        expected = (
            "unstable"
            if False in (self.internal_stable, self.external_stable)
            else "stable"
            if self.internal_stable is True and self.external_stable is True
            else "unavailable"
        )
        if self.status != "not_requested" and self.status != expected:
            raise ValueError("Stability status contradicts native returned flags.")
        if self.status == "not_requested" and (
            self.internal_stable is not None or self.external_stable is not None
        ):
            raise ValueError("Unrequested stability cannot supply diagnostic flags.")
        return self


class OpenShellElectronicEnergy(PhysicalValue):
    energy_hartree: StrictFloat
    engine: Literal["PySCF"]
    engine_version: str = Field(min_length=1)
    method: dict[str, Any]
    molecule: Molecule
    scf: UnrestrictedSCF
    spin: UnrestrictedSpin
    stability: UnrestrictedStability
    native_manifest_sha256: str = Field(pattern="^[0-9a-f]{64}$")
    adapter_source_sha256: str = Field(pattern="^[0-9a-f]{64}$")
    engine_installation_sha256: str = Field(pattern="^[0-9a-f]{64}$")
    requested_state_preserved: Literal[True]
    lowest_electronic_state_certified: Literal[False]
    energy_definition: Literal[
        "Unprojected energy of the declared unrestricted model at the retained "
        "geometry, including D4 when requested."
    ]
    spin_squared_unit: Literal["<S^2>/hbar^2, dimensionless"]

    @model_validator(mode="after")
    def exact_observation(self) -> Self:
        from .domain import ATOMIC_NUMBERS

        if (
            self.method.get("reference") != "unrestricted"
            or self.method.get("name") not in {"hf", "pbe", "b3lyp"}
            or self.scf.requested_multiplicity != self.molecule.multiplicity
            or self.spin.requested_multiplicity != self.molecule.multiplicity
            or self.scf.electron_count
            != sum(ATOMIC_NUMBERS[s] for s in self.molecule.symbols)
            - self.molecule.charge
            or self.energy_hartree != self.scf.energy_hartree
        ):
            raise ValueError(
                "Electronic observation differs from the requested unrestricted tuple."
            )
        return self


class OpenShellOptimizedGeometry(PhysicalValue):
    molecule: Molecule
    optimization: dict[str, Any]
    gradient_hartree_bohr: list[list[StrictFloat]]
    native_manifest_sha256: str = Field(pattern="^[0-9a-f]{64}$")
    stationary_character: Literal["unclassified_until_validated_unrestricted_hessian"]
    requested_state_preserved: Literal[True]
    spin_projection_applied: Literal[False]
    minimum_established: Literal[False]
    electronic_branch_continuity: Literal["not_qualified"]

    @model_validator(mode="after")
    def native_stationarity(self) -> Self:
        optimization = self.optimization
        gradient = np.asarray(self.gradient_hartree_bohr)
        if (
            self.molecule.multiplicity < 2
            or gradient.shape != (len(self.molecule.symbols), 3)
            or optimization.get("converged") is not True
            or optimization.get("final_gradient_verified") is not True
            or optimization.get("final_reference_stability_passed") is not True
            or optimization.get("requested_state_preserved") is not True
            or optimization.get("spin_projection_applied") is not False
            or optimization.get("stationary_point") != self.stationary_character
        ):
            raise ValueError(
                "Unrestricted geometry did not pass its declared "
                "native stationarity gates."
            )
        parameters = optimization["parameters"]
        if (
            np.sqrt(np.mean(gradient**2)) > parameters["convergence_grms"]
            or np.max(np.abs(gradient)) > parameters["convergence_gmax"]
        ):
            raise ValueError("Retained final gradient contradicts native stationarity.")
        return self


def open_shell_plan_for_request(
    request: CalculationRequest, profile: dict[str, Any]
) -> dict[str, Any]:
    from .registry import profile_capabilities

    plan = {
        "schema_version": "cochem.torq.plan/1",
        "capability_purpose": "controlled_validation",
        "serialization_profile": request.serialization_profile,
        "request": request.model_dump(mode="json"),
        "request_sha256": digest(request.model_dump(mode="json")),
        "scientific_cache_key": scientific_cache_key(request, profile),
        "scientific_recipe": profile,
        "requested_products": request.products,
        "declared_capability_records": [
            record.model_dump(mode="json")
            for record in profile_capabilities(profile["id"])
        ],
        "tasks": [
            {
                "id": "unrestricted_initial",
                "depends_on": [],
                "operation": "unprojected_energy_gradient_spin_stability",
                "engine": profile["engine"],
                "recipe_sha256": profile["recipe_sha256"],
            },
            {
                "id": "unrestricted_optimize",
                "depends_on": ["unrestricted_initial"],
                "operation": "explicit_unrestricted_gradient_optimization",
                "engine": profile["engine"],
                "recipe_sha256": profile["recipe_sha256"],
                "conditional_on": (
                    "initial convergence and requested reference stability"
                ),
                "optimizer_maxsteps": 80,
            },
        ],
        "resources": request.resources.model_dump(mode="json"),
        "recovery": "same_exact_unrestricted_tuple_and_Ms_sector_only",
        "constraints": "none; fully relaxed unprojected isolated-molecule optimization",
        "execution_environment": "isolated_linux_cpu_local_validation_worker",
        "spin_projection_applied": False,
        "minimum_established": False,
        "spectroscopy_requested": False,
    }
    return {**plan, "plan_sha256": digest(plan)}


def _energy(native: dict[str, Any]) -> dict[str, Any]:
    return OpenShellElectronicEnergy.model_validate(
        {
            **{
                key: native[key]
                for key in (
                    "energy_hartree",
                    "engine",
                    "engine_version",
                    "method",
                    "molecule",
                    "scf",
                    "spin",
                    "stability",
                    "adapter_source_sha256",
                    "engine_installation_sha256",
                    "requested_state_preserved",
                    "lowest_electronic_state_certified",
                )
            },
            "native_manifest_sha256": native["manifest_sha256"],
            "energy_definition": (
                "Unprojected energy of the declared unrestricted model at the retained "
                "geometry, including D4 when requested."
            ),
            "spin_squared_unit": "<S^2>/hbar^2, dimensionless",
        }
    ).model_dump(mode="json", by_alias=True)


def execute_open_shell_worker(
    request: CalculationRequest, directory: Path
) -> dict[str, Any]:
    """Retain valid initial observations if a later native optimization fails."""
    from .application import _checkpoint, _empty_result, _isotope_provenance, _stage
    from .engines.open_shell import OpenShellPySCFBackend
    from .registry import get_profile

    profile = get_profile(request.recipe)
    if profile.get("reference") != "unrestricted" or request.products != ["geometry"]:
        raise ValueError(
            "This worker accepts only an exact unrestricted "
            "geometry validation profile."
        )
    result = _empty_result(
        request, profile, "Unrestricted spectroscopy is outside this geometry request."
    )
    for name in result["stages"]:
        result["stages"][name] = _stage(
            "blocked",
            name,
            reason="Outside the explicit unrestricted geometry product.",
            absence_kind="not_requested",
        )
    result["resolved_isotopes"] = _isotope_provenance(request)
    backend = OpenShellPySCFBackend()
    probe = backend.probe()
    if (
        not probe["available"]
        or probe.get("version") != profile["engine_version"]
        or probe.get("geometric_version") != profile["optimizer_version"]
        or profile.get("dispersion") == "d4"
        and probe.get("dftd4_version") != profile["dispersion_version"]
    ):
        result["errors"].append(
            {
                "code": "OPEN_SHELL_EXACT_ENGINE_UNAVAILABLE",
                "message": (
                    "The exact declared PySCF/geomeTRIC/D4 installation is unavailable."
                ),
                "probe": probe,
            }
        )
        _checkpoint(directory, result)
        return result
    backend_request = {
        "molecule": request.molecule.model_dump(mode="json"),
        "method": {
            "name": profile["method"],
            "basis": profile["basis"],
            "reference": "unrestricted",
            "dispersion": profile.get("dispersion"),
        },
        "properties": ["energy", "gradient"],
        "settings": {
            **profile["numerical"],
            "threads": request.resources.cores,
            "memory_mb": request.resources.memory_mb,
        },
    }
    initial = backend.evaluate(backend_request, directory / "unrestricted-initial")
    result["initial_native_result"] = initial
    result["native_result"] = initial
    stages = result["stages"]
    if (
        initial.get("energy_hartree") is None
        or not initial.get("scf", {}).get("converged")
        or initial.get("spin") is None
        or initial.get("stability") is None
    ):
        result["errors"].extend(initial.get("errors", []))
        stages["electronic_structure"] = _stage(
            "failed",
            "open_shell_electronic_energy",
            reason=(
                "Native unrestricted SCF, preserved electron sector and spin "
                "diagnostics were not established."
            ),
        )
        stages["equilibrium_geometry"] = _stage(
            "blocked",
            "open_shell_optimized_geometry",
            reason="Initial unrestricted characterization failed.",
        )
        _checkpoint(directory, result)
        return result
    stages["electronic_structure"] = _stage(
        "available",
        "open_shell_electronic_energy",
        value=_energy(initial),
        flags=initial["quality_flags"],
    )
    stages["initial_electronic_structure"] = dict(stages["electronic_structure"])
    result["status"] = "partial"
    _checkpoint(directory, result)
    if initial["status"] != "complete" or initial["stability"]["status"] != "stable":
        result["errors"].extend(initial.get("errors", []))
        stages["equilibrium_geometry"] = _stage(
            "unavailable",
            "open_shell_optimized_geometry",
            reason=(
                "The requested unrestricted reference stability gates did not "
                "pass; actual initial observations remain available."
            ),
            parents=["electronic_structure"],
            flags=initial["quality_flags"],
        )
        _checkpoint(directory, result)
        return result
    native = backend.optimize(backend_request, directory / "unrestricted-optimization")
    result["native_result"] = native
    result["errors"].extend(native.get("errors", []))
    if (
        native.get("energy_hartree") is not None
        and native.get("scf", {}).get("converged")
        and native.get("spin") is not None
        and native.get("stability") is not None
    ):
        stages["electronic_structure"] = _stage(
            "available",
            "open_shell_electronic_energy",
            value=_energy(native),
            parents=["initial_electronic_structure"],
            flags=native["quality_flags"],
        )
    optimization = native.get("optimization", {})
    if native["status"] != "complete" or not all(
        optimization.get(key) is True
        for key in (
            "converged",
            "final_gradient_verified",
            "final_reference_stability_passed",
        )
    ):
        stages["equilibrium_geometry"] = _stage(
            "failed",
            "open_shell_optimized_geometry",
            reason=(
                "Native unrestricted convergence, independent final gradient "
                "and requested stability were not all established."
            ),
            parents=["electronic_structure"],
        )
        _checkpoint(directory, result)
        return result
    geometry = OpenShellOptimizedGeometry.model_validate(
        {
            "molecule": native["molecule"],
            "optimization": optimization,
            "gradient_hartree_bohr": native["gradient_hartree_bohr"],
            "native_manifest_sha256": native["manifest_sha256"],
            "stationary_character": "unclassified_until_validated_unrestricted_hessian",
            "requested_state_preserved": True,
            "spin_projection_applied": False,
            "minimum_established": False,
            "electronic_branch_continuity": "not_qualified",
        }
    )
    stages["equilibrium_geometry"] = _stage(
        "available",
        "open_shell_optimized_geometry",
        value=geometry.model_dump(mode="json"),
        parents=["electronic_structure"],
        flags=[
            "minimum_not_characterized",
            "geometry_accuracy_uncalibrated",
            "electronic_branch_continuity_unqualified",
        ],
    )
    result["status"] = "complete" if not result["errors"] else "partial"
    _checkpoint(directory, result)
    return result
