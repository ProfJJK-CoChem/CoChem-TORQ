"""Explicit recipes; the historical method matrix never acts as an enable switch."""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .capabilities import CapabilityRecord, CapabilityTuple

from .domain import digest

_BASE: dict[str, Any] = {
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
_PROFILES: dict[str, dict[str, Any]] = {
    "hf-sto-3g-education": {
        **_BASE,
        "id": "hf-sto-3g-education",
        "version": 1,
        "label": "RHF / STO-3G teaching calculation",
        "method": "hf",
        "basis": "sto-3g",
        "availability": "experimental",
        "runnable": True,
        "reason": (
            "Real restricted HF teaching baseline. Numerical "
            "qualification is scoped to tested molecules; "
            "experimental spectroscopy accuracy is uncalibrated."
        ),
        "matrix_refs": [],
        "method_family": None,
        "citation": (
            "Roothaan, Rev. Mod. Phys. 23, 69 (1951), doi:10.1103/RevModPhys.23.69"
        ),
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
        "reason": (
            "Explicit research validation experiment; Hartree–Fock "
            "omits correlation and dispersion. It is not a matrix "
            "high-accuracy substitute."
        ),
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
        "reason": (
            "Activation requires primary-source D4 recipe "
            "verification, independent energy comparison, "
            "orbital-response derivatives and separately validated "
            "higher derivatives."
        ),
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
        "label": (
            f"{name.upper()} / "
            f"{'cc-pVDZ' if correlated else 'def2-SVP'} local validation"
        ),
        "method": name,
        "basis": "cc-pvdz" if correlated else "def2-svp",
        "dispersion": None if correlated else "d4",
        "dispersion_version": None if correlated else "3.7.0",
        "availability": "experimental",
        "runnable": True,
        "reason": (
            "Genuine CPU validation experiment with tested "
            "finite-difference derivatives; requires explicit local "
            "execution. Experimental spectroscopy accuracy remains "
            "uncalibrated."
        ),
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
        "qualification_evidence": (
            "benchmarks/generated/pyscf_cpu_qualification.json; "
            "exact tested tuples only"
        ),
    }

_PROFILES["hf-sto-3g-anharmonic-validation"] = {
    **deepcopy(_PROFILES["hf-sto-3g-education"]),
    "id": "hf-sto-3g-anharmonic-validation",
    "label": "RHF / STO-3G local anharmonic validation experiment",
    "max_atoms": 3,
    "reason": (
        "Actual finite-displacement vibrational force field, "
        "bounded to three modes, with separately gated nonresonant Watson "
        "vibration–rotation model predictions. Independent accuracy, resonant "
        "GVPT2 and identification qualification remain blocked."
    ),
    "products": [
        "geometry",
        "harmonic",
        "equilibrium_constants",
        "anharmonic_force_field",
        "vpt2",
        "ground_state_constants",
    ],
    "numerical": {**_BASE["numerical"], "scf_energy_tolerance": 1e-12},
    "anharmonic": {
        "max_modes": 3,
        "steps_dimensionless": [0.08, 0.04],
        "max_evaluations": 200,
        "absolute_tolerance_hartree": 1e-6,
        "relative_tolerance": 0.03,
        "scientific_scope": (
            "experimental finite-displacement vibrational field; "
            "independent molecular accuracy unqualified"
        ),
    },
    "rovibrational": {
        "maximum_modes": 3,
        "maximum_intermediate_states": 5000,
        "minimum_denominator_cm1": 10.0,
        "maximum_anharmonic_coupling_ratio": 0.1,
        "maximum_coriolis_coupling_ratio": 0.1,
        "maximum_relative_rotational_correction": 0.1,
        "maximum_relative_vibrational_correction": 0.1,
        "maximum_J": 1,
        "coupling_floor_cm1": 1e-6,
    },
}


_PROFILES["hf-sto-3g-pes-validation"] = {
    **deepcopy(_PROFILES["hf-sto-3g-education"]),
    "id": "hf-sto-3g-pes-validation",
    "version": 1,
    "label": "RHF / STO-3G bounded fixed H2 scan validation",
    "max_atoms": 2,
    "elements": ["H"],
    "optimizer": None,
    "optimizer_version": None,
    "constraints": "explicit fixed bond-coordinate sampling; no relaxation",
    "products": ["pes_scan"],
    "reason": (
        "Genuine approved restricted H2 fixed-point energy/gradient sampling. "
        "No constrained optimizer, state-following, search completeness or "
        "independent molecular accuracy qualification. Local validation only."
    ),
}

_PROFILES["hf-sto-3g-internal-pes-validation"] = {
    **deepcopy(_PROFILES["hf-sto-3g-education"]),
    "id": "hf-sto-3g-internal-pes-validation",
    "version": 1,
    "label": "RHF / STO-3G declared internal-coordinate scan validation",
    "optimizer": None,
    "optimizer_version": None,
    "constraints": (
        "fixed minimum-displacement Cartesian embeddings "
        "of declared bond/angle/torsion coordinates"
    ),
    "products": ["pes_scan"],
    "reason": (
        "Genuine approved restricted molecular fixed-point energy/gradient sampling "
        "on an explicit multidimensional or periodic finite design. Coordinate "
        "construction is mathematical and does not optimize electronic energy. "
        "No branch/search completeness or independent molecular accuracy claim. "
        "Local validation only."
    ),
}


_PROFILES["hf-sto-3g-constrained-pes-validation"] = {
    **deepcopy(_PROFILES["hf-sto-3g-education"]),
    "id": "hf-sto-3g-constrained-pes-validation",
    "version": 1,
    "label": "RHF / STO-3G constrained energy optimization validation",
    "optimizer": "SciPy SLSQP",
    "optimizer_version": "1.18.1",
    "constraints": "explicit internal-coordinate targets and movable atom rows",
    "products": ["constrained_geometry"],
    "reason": (
        "Genuine bounded energy optimization with independently checked constraints "
        "and tangent gradients. Constrained stationarity is a separate product from "
        "unconstrained equilibrium geometry; accuracy remains uncalibrated. "
        "Local validation only."
    ),
}


def get_profile(identifier: str) -> dict[str, Any]:
    if identifier not in _PROFILES:
        raise ValueError(
            f"Unknown recipe {identifier!r}; "
            "historical row aliases require explicit mapping."
        )
    result = deepcopy(_PROFILES[identifier])
    result["recipe_sha256"] = digest(result)
    return result


def list_method_profiles() -> list[dict[str, Any]]:
    return [get_profile(identifier) for identifier in _PROFILES]


def profile_capabilities(identifier: str) -> tuple[CapabilityRecord, ...]:
    """Immutable exact experimental tuples for implemented native CPU routes.

    These declarations authorize controlled validation, never automatic production
    or an unrun chemical/identification accuracy profile. Runtime evidence may
    qualify a separately recorded exact native request via capabilities.py.
    """
    from .capabilities import CapabilityRecord, CapabilityTuple

    profile = get_profile(identifier)
    if not profile.get("runnable"):
        return ()
    if profile.get("reference") == "unrestricted":
        return tuple(
            CapabilityRecord(
                tuple_definition=CapabilityTuple(
                    method=profile["method"],
                    basis=profile["basis"],
                    electronic_reference="unrestricted_open_shell",
                    property=property_name,
                    derivative=derivative,
                    engine=profile["engine"],
                    engine_version=profile["engine_version"],
                    optimizer_version=profile["optimizer_version"]
                    if property_name == "optimization"
                    else None,
                    hardware="linux-x86_64-cpu",
                    recipe_sha256=profile["recipe_sha256"],
                ),
                availability="experimental",
                reason="Explicit local unprojected unrestricted validation only; "
                "spin, native reference stability and model accuracy remain separate.",
            )
            for property_name, derivative in (
                ("energy", "none"),
                ("gradient", "analytic"),
                ("optimization", "analytic_gradient_optimization"),
            )
        )
    records = []
    for property_name, derivative in (
        ("energy", "none"),
        ("gradient", "analytic"),
        ("hessian", profile["derivatives"]["hessian"]),
        ("dipole", "density_expectation"),
        ("optimization", "analytic_gradient_optimization"),
    ):
        if identifier in {
            "hf-sto-3g-pes-validation",
            "hf-sto-3g-internal-pes-validation",
        } and property_name not in {
            "energy",
            "gradient",
        }:
            continue
        if (
            identifier == "hf-sto-3g-constrained-pes-validation"
            and property_name == "dipole"
        ):
            continue
        unsupported = property_name == "dipole" and profile["method"] == "mp2"
        records.append(
            CapabilityRecord(
                tuple_definition=CapabilityTuple(
                    method=profile["method"],
                    basis=profile["basis"],
                    electronic_reference="restricted_closed_shell",
                    property=property_name,
                    derivative=derivative,
                    engine=profile["engine"],
                    engine_version=profile["engine_version"],
                    optimizer_version=profile["optimizer_version"]
                    if property_name == "optimization"
                    else None,
                    hardware="linux-x86_64-cpu",
                    recipe_sha256=profile["recipe_sha256"],
                ),
                availability="unsupported" if unsupported else "experimental",
                reason="Relaxed-response MP2 dipoles are not implemented."
                if unsupported
                else "Explicit pinned native route for controlled "
                "numerical validation; chemical accuracy remains unqualified.",
            )
        )
    return tuple(records)


def resolve_exact_capability(
    definition: CapabilityTuple | dict[str, Any],
) -> CapabilityRecord:
    """An unlisted tuple is unknown; a similar recipe cannot establish support."""
    from .capabilities import CapabilityRecord, CapabilityTuple

    definition = CapabilityTuple.model_validate(definition)
    for identifier in _PROFILES:
        for record in profile_capabilities(identifier):
            if record.tuple_definition == definition:
                return record
    return CapabilityRecord(
        tuple_definition=definition,
        availability="unknown",
        reason="This exact method/basis/reference/property/version/hardware tuple "
        "has no registered implementation or qualification; "
        "no alternative is substituted.",
    )


def route_profile_capabilities(
    identifier: str, products: Iterable[str], *, execution: str
) -> dict[str, Any]:
    """Resolve exact native dependencies before the worker can dispatch."""
    from .capabilities import authorize_capability, observed_cpu_hardware

    if execution not in {"github_actions", "local_validation"}:
        raise ValueError("Unknown capability execution environment.")
    profile = get_profile(identifier)
    hardware = (
        "linux-x86_64-cpu" if execution == "github_actions" else observed_cpu_hardware()
    )
    selected_products = set(products)
    if identifier in {"hf-sto-3g-pes-validation", "hf-sto-3g-internal-pes-validation"}:
        required = {"energy", "gradient"}
    else:
        required = {"energy", "gradient", "optimization"}
    if (
        profile.get("reference") != "unrestricted"
        and profile.get("method") != "mp2"
        and identifier
        not in {
            "hf-sto-3g-pes-validation",
            "hf-sto-3g-internal-pes-validation",
            "hf-sto-3g-constrained-pes-validation",
        }
    ):
        required.add("dipole")  # The actual worker always calculates this property.
    if (
        profile.get("reference") != "unrestricted"
        and selected_products - {"geometry"}
        and identifier
        not in {
            "hf-sto-3g-pes-validation",
            "hf-sto-3g-internal-pes-validation",
            "hf-sto-3g-constrained-pes-validation",
        }
    ):
        required.add("hessian")
    if identifier == "hf-sto-3g-constrained-pes-validation":
        # Curvature is an explicit request option; retain its exact analytic
        # dependency in the profile authorization before any native launch.
        required.add("hessian")
    receipts, reasons = [], []
    records = {
        record.tuple_definition.property: record
        for record in profile_capabilities(identifier)
    }
    for property_name in sorted(required):
        record = records.get(property_name)
        if record is None:
            reasons.append(
                f"No exact registered native capability for {property_name}."
            )
            continue
        definition = record.tuple_definition.model_dump(mode="json")
        definition["hardware"] = hardware
        exact = resolve_exact_capability(definition)
        try:
            receipt = authorize_capability(exact, purpose="controlled_validation")
            receipts.append(
                {**receipt, "tuple": exact.tuple_definition.model_dump(mode="json")}
            )
        except ValueError as error:
            reasons.append(f"{property_name}: {error}")
    return {
        "schema_version": "cochem.torq.capability-routing/1",
        "purpose": "controlled_validation",
        "hardware": hardware,
        "hardware_evidence": "declared_canonical_actions_cpu_target"
        if execution == "github_actions"
        else "observed_local_os_and_cpu_architecture",
        "executable": not reasons,
        "blocking_reasons": reasons,
        "resolved": receipts,
        "chemical_accuracy_established": False,
    }


def matrix_index() -> list[dict[str, str]]:
    """Preserve all 140 table/track/budget identities.

    Indexing a method never invents engine capability.
    """
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


# These explicit profiles are separate from restricted classroom spectroscopy.
# Source-bound local validation does not establish matrix or experimental accuracy.
_OPEN_SHELL_BASE: dict[str, Any] = {
    **_BASE,
    "reference": "unrestricted",
    "multiplicities": [2, 3, 4, 5, 6, 7],
    "charges": [-2, -1, 0, 1, 2],
    "products": ["geometry"],
    "derivatives": {"gradient": "analytic", "hessian": "unsupported"},
    "numerical": {
        "scf_energy_tolerance": 1e-11,
        "scf_gradient_tolerance": 1e-7,
        "scf_max_cycle": 150,
        "dft_grid_level": 4,
        "check_stability": True,
        "stability_tolerance": 1e-4,
        "stability_nroots": 3,
        "spin_contamination_tolerance": 0.1,
    },
    "availability": "experimental",
    "runnable": True,
    "local_validation_only": True,
    "ui_visible": False,
    "minimum_established": False,
    "lowest_electronic_state_certified": False,
    "spin_projection_applied": False,
    "matrix_refs": [],
    "method_family": None,
    "reason": (
        "Genuine bounded unrestricted local validation of the requested electron "
        "count and M_S sector. Native spin contamination and reference stability "
        "are retained. No exact-spin term, lowest-state, Hessian, minimum, "
        "spectroscopy or experimental-accuracy qualification."
    ),
}
for _open_identifier, _open_method, _open_dispersion, _open_label in (
    (
        "uhf-sto-3g-open-shell-validation",
        "hf",
        None,
        "UHF / STO-3G local state validation",
    ),
    (
        "uks-pbe-d4-sto-3g-open-shell-validation",
        "pbe",
        "d4",
        "UKS PBE-D4 / STO-3G local state validation",
    ),
    (
        "uks-b3lyp-d4-sto-3g-open-shell-validation",
        "b3lyp",
        "d4",
        "UKS B3LYP-D4 / STO-3G local state validation",
    ),
):
    _open_profile = {
        **_OPEN_SHELL_BASE,
        "id": _open_identifier,
        "version": 1,
        "label": _open_label,
        "method": _open_method,
        "basis": "sto-3g",
        "dispersion": _open_dispersion,
        "citation": (
            "Native PySCF 2.14.0 unrestricted implementation; method and D4 "
            "references retained in authentic engine artifacts."
        ),
    }
    _open_profile.pop("multiplicity", None)
    if _open_dispersion == "d4":
        _open_profile["dispersion_version"] = "3.7.0"
    _PROFILES[_open_identifier] = _open_profile
del _open_identifier, _open_method, _open_dispersion, _open_label, _open_profile
