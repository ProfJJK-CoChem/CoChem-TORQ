"""Explicit interaction, counterpoise, relaxation and RRHO energy identities.

These named models do not establish transferable chemical accuracy. A supplied
geometry is never certified relaxed; only genuine optimization, stationarity,
SCF-stability and projected-Hessian checks can produce a De/D0 result.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field, StrictFloat, StrictInt, model_validator
from typing_extensions import Self

from .domain import ATOMIC_NUMBERS, Contract, Molecule
from .engines.counterpoise import GhostPySCFBackend
from .engines.pyscf_backend import (
    BackendCalculationError,
    BackendInputError,
    PySCFBackend,
    _json,
    _normalize,
    _seal,
)
from .spectroscopy.harmonic import (
    HarmonicResult,
    analyze_hessian,
    equilibrium_rotor,
)
from .units import BOHR_METRE, HARTREE_JOULE, atomic_mass, h, k, pi

_SOURCE_SHA256 = sha256(Path(__file__).read_bytes()).hexdigest()


class Fragment(Contract):
    fragment_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    atom_ids: list[str] = Field(min_length=1)
    charge: StrictInt
    multiplicity: StrictInt = Field(ge=1)

    @model_validator(mode="after")
    def distinct_atoms(self) -> Self:
        if len(set(self.atom_ids)) != len(self.atom_ids):
            raise ValueError("A fragment cannot repeat atom IDs.")
        if self.fragment_id == "complex":
            raise ValueError("'complex' is reserved for the entire system.")
        return self


class RRHOSettings(Contract):
    temperature_kelvin: StrictFloat = Field(gt=0, le=2000)
    standard_pressure_pa: StrictFloat = Field(gt=0)
    rotational_symmetry_numbers: dict[str, StrictInt]

    @model_validator(mode="after")
    def positive_symmetry(self) -> Self:
        if any(value < 1 for value in self.rotational_symmetry_numbers.values()):
            raise ValueError(
                "Rotational symmetry numbers must be explicit positive integers."
            )
        return self


class InteractionRequest(Contract):
    schema_version: Literal["cochem.torq.interaction-request/1"] = (
        "cochem.torq.interaction-request/1"
    )
    molecule: Molecule
    method: dict[str, Any]
    fragments: list[Fragment] = Field(min_length=2, max_length=8)
    geometry_protocol: Literal[
        "supplied_frozen_monomers", "fully_relaxed", "counterpoise_single_point"
    ]
    optimization: dict[str, Any] = Field(default_factory=dict)
    rrho: RRHOSettings | None = None
    harmonic_external_residual_relative_tolerance: StrictFloat = Field(
        default=1e-4, gt=0, le=1e-3
    )
    harmonic_symmetry_relative_tolerance: StrictFloat = Field(
        default=1e-8, gt=0, le=1e-4
    )

    @model_validator(mode="after")
    def fragment_partition(self) -> Self:
        if not self.molecule.atom_ids:
            raise ValueError(
                "Explicit stable molecule atom IDs are required for fragment "
                "calculations."
            )
        if len({f.fragment_id for f in self.fragments}) != len(self.fragments):
            raise ValueError("Fragment IDs must be unique.")
        assigned = [atom for fragment in self.fragments for atom in fragment.atom_ids]
        if len(assigned) != len(set(assigned)) or set(assigned) != set(
            self.molecule.atom_ids
        ):
            raise ValueError(
                "Fragments must be disjoint and exactly partition all molecule "
                "atom IDs."
            )
        if sum(f.charge for f in self.fragments) != self.molecule.charge:
            raise ValueError("Fragment charges must sum to the complex charge.")
        if self.molecule.multiplicity != 1 or any(
            f.multiplicity != 1 for f in self.fragments
        ):
            raise ValueError(
                "This restricted CPU workflow requires explicit singlet complex "
                "and fragment states."
            )
        lookup = dict(zip(self.molecule.atom_ids, self.molecule.symbols))
        for fragment in self.fragments:
            electrons = (
                sum(ATOMIC_NUMBERS[lookup[atom]] for atom in fragment.atom_ids)
                - fragment.charge
            )
            if electrons <= 0 or electrons % 2:
                raise ValueError(
                    "Each explicit fragment charge/spin must define a "
                    "positive even electron count."
                )
        if self.geometry_protocol != "fully_relaxed" and self.optimization:
            raise ValueError(
                "Single-point protocols do not perform or claim an optimization."
            )
        if self.rrho is not None:
            if self.geometry_protocol != "fully_relaxed":
                raise ValueError(
                    "RRHO requires real optimized and Hessian-characterized endpoints."
                )
            expected = {"complex", *(f.fragment_id for f in self.fragments)}
            if set(self.rrho.rotational_symmetry_numbers) != expected:
                raise ValueError(
                    "Explicit RRHO symmetry numbers are required for "
                    "the complex and every fragment."
                )
        data = _normalize(
            {
                "molecule": self.molecule,
                "method": self.method,
                "properties": ["energy", "gradient"],
            }
        )
        if data["settings"]["check_stability"] is not True:
            raise ValueError(
                "Accepted interaction calculations require actual SCF stability checks."
            )
        if self.geometry_protocol == "fully_relaxed" and data["method"]["name"] != "hf":
            # Numerical Hessians need an independently qualified invariance
            # tolerance before this endpoint/thermochemistry profile is enabled.
            raise ValueError(
                "Fully relaxed De/D0/RRHO is currently qualified only for "
                "restricted HF; DFT/MP2 CP single points remain separately "
                "supported."
            )
        return self


class EnergyQuantity(Contract):
    status: Literal["available", "unavailable"]
    value_hartree: StrictFloat | None = None
    definition: str
    parents: list[str]
    reason: str | None = None

    @model_validator(mode="after")
    def truthful_value(self) -> Self:
        if self.status == "available" and (
            self.value_hartree is None or self.reason is not None
        ):
            raise ValueError(
                "An available energy requires a finite value and no failure reason."
            )
        if self.status == "unavailable" and (
            self.value_hartree is not None or not self.reason
        ):
            raise ValueError(
                "An unavailable energy requires a reason and no substitute value."
            )
        return self


class InteractionResult(Contract):
    schema_version: Literal["cochem.torq.interaction-result/1"] = (
        "cochem.torq.interaction-result/1"
    )
    status: Literal["available", "failed"]
    geometry_protocol: str
    method: dict[str, Any]
    settings: dict[str, Any]
    molecule: Molecule
    fragments: list[Fragment]
    components: dict[str, dict[str, Any]]
    quantities: dict[str, EnergyQuantity]
    gradients_hartree_bohr: dict[str, list[list[StrictFloat]]]
    stationary_evidence: dict[str, dict[str, Any]]
    thermochemistry: dict[str, dict[str, Any]]
    basis_identity_evidence: dict[str, Any]
    errors: list[str]
    quality_flags: list[str]
    adapter_source_sha256: str
    manifest_path: str | None = None
    manifest_sha256: str | None = None
    artifacts: dict[str, Any] | None = None

    @model_validator(mode="after")
    def consistent_identities(self) -> Self:
        if self.status == "failed":
            if not self.errors:
                raise ValueError(
                    "A failed interaction result requires an actual error."
                )
            return self
        if self.errors or "interaction_uncorrected" not in self.quantities:
            raise ValueError(
                "An available interaction requires accepted components "
                "and its defined energy."
            )

        def available_value(name: str) -> float:
            quantity = self.quantities[name]
            if quantity.status != "available" or quantity.value_hartree is None:
                raise ValueError("An available identity requires actual energy values.")
            return quantity.value_hartree

        identities = [
            (
                "interaction_counterpoise",
                "interaction_uncorrected",
                "bsse_correction",
                1,
            ),
            ("binding_electronic", "interaction_uncorrected", "deformation", 1),
            ("D0_harmonic", "De", "zpe_binding_correction", -1),
        ]
        for target, first, second, sign in identities:
            if all(
                key in self.quantities and self.quantities[key].status == "available"
                for key in [target, first, second]
            ):
                value = available_value(first) + sign * available_value(second)
                if abs(available_value(target) - value) > 1e-9:
                    raise ValueError(
                        "Available interaction energy identities are inconsistent."
                    )
        if all(
            key in self.quantities and self.quantities[key].status == "available"
            for key in ["De", "binding_electronic"]
        ):
            if (
                abs(available_value("De") + available_value("binding_electronic"))
                > 1e-9
            ):
                raise ValueError("De and electronic binding must have opposite signs.")
        return self


def _fragment_molecule(molecule: Molecule, fragment: Fragment) -> Molecule:
    atom_ids = molecule.atom_ids
    if not atom_ids:
        raise BackendInputError(
            "Fragment calculations require explicit molecule atom IDs."
        )
    indices = [
        index for index, atom_id in enumerate(atom_ids) if atom_id in fragment.atom_ids
    ]
    return Molecule(
        symbols=[molecule.symbols[i] for i in indices],
        geometry_bohr=[molecule.geometry_bohr[i] for i in indices],
        atom_ids=[atom_ids[i] for i in indices],
        isotopes=[molecule.isotopes[i] for i in indices]
        if molecule.isotopes is not None
        else None,
        charge=fragment.charge,
        multiplicity=fragment.multiplicity,
    )


def _accepted_energy(component: dict[str, Any], label: str) -> float:
    if (
        component.get("status") != "complete"
        or component.get("scf", {}).get("converged") is not True
        or component.get("stability", {}).get("status") != "stable"
    ):
        raise BackendCalculationError(
            f"{label} failed real convergence or SCF-stability acceptance: "
            f"{component.get('errors')}"
        )
    energy = component.get("energy_hartree")
    if energy is None or not np.isfinite(energy):
        raise BackendCalculationError(f"{label} has no finite accepted energy.")
    return float(energy)


def _basis_signature(component: dict[str, Any]) -> dict[str, Any]:
    from pyscf import lib

    from .engines.diagnostics import verify_native_artifacts

    directory = verify_native_artifacts(component)
    checkpoint = directory / "wavefunction.chk"
    if not checkpoint.is_file():
        checkpoint = directory / "final" / "wavefunction.chk"
    molecule = lib.chkfile.load_mol(str(checkpoint))
    return {
        "cartesian_gaussians": bool(molecule.cart),
        "basis_function_count": int(molecule.nao_nr()),
        "basis_center_geometry_bohr": molecule.atom_coords(unit="Bohr").tolist(),
        "shells": [
            {
                "center_index": int(molecule.bas_atom(shell)),
                "angular_momentum": int(molecule.bas_angular(shell)),
                "exponents": molecule.bas_exp(shell).tolist(),
                "contraction_coefficients": molecule.bas_ctr_coeff(shell).tolist(),
            }
            for shell in range(molecule.nbas)
        ],
    }


def electronic_checkpoint_digest(component: dict[str, Any]) -> str:
    directory = Path(component["manifest_path"]).parent
    checkpoint = directory / "wavefunction.chk"
    if not checkpoint.is_file():
        checkpoint = directory / "final" / "wavefunction.chk"
    return sha256(checkpoint.read_bytes()).hexdigest()


def _isotope_records(molecule: Molecule) -> list[dict[str, Any]]:
    from Libraries.cochem_isotopes import isotope_record

    numbers = molecule.isotopes or [None] * len(molecule.symbols)
    return [
        isotope_record(f"{number}{symbol}" if number is not None else symbol)
        for number, symbol in zip(numbers, molecule.symbols)
    ]


def _minimum(
    component: dict[str, Any], molecule: Molecule, request: InteractionRequest
) -> tuple[HarmonicResult, dict[str, Any]]:
    _accepted_energy(component, "optimized endpoint")
    optimization = component.get("optimization", {})
    if (
        optimization.get("converged") is not True
        or optimization.get("final_gradient_verified") is not True
    ):
        raise BackendCalculationError(
            "An optimized endpoint must pass both actual optimizer and "
            "independent final-gradient gates."
        )
    records = _isotope_records(molecule)
    harmonic = analyze_hessian(
        component["geometry_bohr"],
        [record["mass_u"] for record in records],
        component["hessian_hartree_bohr2"],
        symmetry_tolerance=request.harmonic_symmetry_relative_tolerance,
    )
    if harmonic.stationary_character != "positive_definite_vibrational_hessian":
        raise BackendCalculationError(
            "Endpoint is not a resolved minimum of its actual projected Hessian."
        )
    if (
        harmonic.external_residual_relative
        > request.harmonic_external_residual_relative_tolerance
    ):
        raise BackendCalculationError(
            "Endpoint Hessian fails the explicit external-invariance tolerance."
        )
    return harmonic, {
        "optimizer_converged": True,
        "independent_final_gradient_verified": True,
        "classification": harmonic.stationary_character,
        "harmonic_source_digest": harmonic.source_digest,
        "harmonic_frequencies_cm1": harmonic.frequencies_cm1.tolist(),
        "harmonic_zpe_hartree": harmonic.harmonic_zpe_hartree,
        "isotopes": records,
        "external_residual_relative": harmonic.external_residual_relative,
        "external_residual_relative_tolerance": (
            request.harmonic_external_residual_relative_tolerance
        ),
        "component_manifest_sha256": component["manifest_sha256"],
    }


def rrho_thermochemistry(
    harmonic: HarmonicResult,
    *,
    temperature_kelvin: float,
    standard_pressure_pa: float,
    rotational_symmetry_number: int,
) -> dict[str, Any]:
    """Ideal-gas, classical-rigid-rotor/harmonic-oscillator corrections.

    The pressure is a one-species standard state. Vibrational ZPE is included
    exactly once; low modes are retained and flagged, never replaced or clipped.
    Nuclear-spin statistical weights and electronic excited states are omitted.
    """
    if (
        not np.isfinite(temperature_kelvin)
        or temperature_kelvin <= 0
        or not np.isfinite(standard_pressure_pa)
        or standard_pressure_pa <= 0
        or type(rotational_symmetry_number) is not int
        or rotational_symmetry_number < 1
        or harmonic.stationary_character != "positive_definite_vibrational_hessian"
        or harmonic.harmonic_zpe_hartree is None
    ):
        raise ValueError(
            "RRHO requires positive explicit state parameters and a real "
            "minimum Hessian."
        )
    rotor = equilibrium_rotor(harmonic.coordinates_bohr, harmonic.isotope_masses_u)
    temperature = float(temperature_kelvin)
    kt_hartree = k * temperature / HARTREE_JOULE
    mass_kg = float(np.sum(harmonic.isotope_masses_u)) * atomic_mass
    log_qtrans = 1.5 * np.log(2 * pi * mass_kg * k * temperature / h**2) + np.log(
        k * temperature / standard_pressure_pa
    )
    if rotor.rotor_type == "atom":
        if rotational_symmetry_number != 1:
            raise ValueError("A monatomic species has rotational symmetry number one.")
        log_qrot, rotational_heat = 0.0, 0.0
    else:
        moments_kg_m2 = rotor.principal_moments_u_bohr2 * atomic_mass * BOHR_METRE**2
        positive = moments_kg_m2[moments_kg_m2 > 0]
        theta = h**2 / (8 * pi * pi * positive * k)
        if rotor.rotor_type == "linear":
            log_qrot = float(
                np.log(temperature / (rotational_symmetry_number * theta[-1]))
            )
            rotational_heat = kt_hartree
        else:
            log_qrot = float(
                0.5 * np.log(pi)
                + 1.5 * np.log(temperature)
                - np.log(rotational_symmetry_number)
                - 0.5 * np.sum(np.log(theta))
            )
            rotational_heat = 1.5 * kt_hartree
    quanta = harmonic.angular_frequencies_au
    x = quanta / kt_hartree
    if np.any(x <= 0):
        raise ValueError(
            "Unresolved/negative vibrations cannot define this minimum RRHO model."
        )
    # exp(-x) is stable for arbitrarily high positive vibrational quanta.
    boltzmann = np.exp(-x)
    thermal_vibration = float(np.sum(quanta * boltzmann / (-np.expm1(-x))))
    vibrational_free = float(
        harmonic.harmonic_zpe_hartree + kt_hartree * np.sum(np.log(-np.expm1(-x)))
    )
    enthalpy = (
        2.5 * kt_hartree
        + rotational_heat
        + harmonic.harmonic_zpe_hartree
        + thermal_vibration
    )
    gibbs = -kt_hartree * (log_qtrans + log_qrot) + vibrational_free
    entropy_hartree_kelvin = float((enthalpy - gibbs) / temperature)
    return {
        "model": "ideal-gas classical rigid-rotor harmonic-oscillator ground-singlet",
        "temperature_kelvin": temperature,
        "standard_pressure_pa": float(standard_pressure_pa),
        "rotational_symmetry_number": rotational_symmetry_number,
        "rotor_type": rotor.rotor_type,
        "harmonic_source_digest": harmonic.source_digest,
        "zero_point_energy_hartree": harmonic.harmonic_zpe_hartree,
        "enthalpy_correction_hartree": float(enthalpy),
        "gibbs_correction_hartree": float(gibbs),
        "entropy_hartree_per_kelvin": entropy_hartree_kelvin,
        "log_translational_partition_function": float(log_qtrans),
        "log_rotational_partition_function": float(log_qrot),
        "zpe_included_exactly_once": True,
        "quality_flags": [
            "harmonic_approximation",
            "classical_rotation",
            "ideal_gas_standard_state",
            "electronic_excited_states_omitted",
            "nuclear_spin_statistics_omitted",
        ]
        + (
            ["low_frequency_rrho_entropy_uncalibrated"]
            if np.any(harmonic.frequencies_cm1 < 100)
            else []
        ),
    }


def _minimum_zpe(harmonic: HarmonicResult) -> float:
    if harmonic.harmonic_zpe_hartree is None:
        raise BackendCalculationError(
            "A minimum requires actual harmonic zero-point energy."
        )
    return harmonic.harmonic_zpe_hartree


def evaluate_interaction(
    request: InteractionRequest | dict[str, Any], workspace: str | Path
) -> InteractionResult:
    """Execute every real component for the explicitly selected energy model."""
    if not isinstance(request, InteractionRequest):
        request = InteractionRequest.model_validate(request)
    normalized = _normalize(
        {
            "molecule": request.molecule,
            "method": request.method,
            "properties": ["energy", "gradient"],
        }
    )
    directory = Path(workspace).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise BackendInputError(
            "Interaction workspace must be empty to prevent stale component reuse."
        )
    _json(directory / "request.json", request.model_dump(mode="json"))
    backend = PySCFBackend()
    result = {
        "status": "failed",
        "geometry_protocol": request.geometry_protocol,
        "method": normalized["method"],
        "settings": normalized["settings"],
        "molecule": request.molecule.model_dump(mode="json"),
        "fragments": [
            fragment.model_dump(mode="json") for fragment in request.fragments
        ],
        "components": {},
        "quantities": {},
        "gradients_hartree_bohr": {},
        "stationary_evidence": {},
        "thermochemistry": {},
        "basis_identity_evidence": {},
        "errors": [],
        "quality_flags": [
            "accuracy_uncalibrated",
            "no_automatic_electronic_state_or_method_change",
            "fragment_states_defined_by_explicit_charge_and_spin",
            "stable_restricted_occupations_do_not_prove_unique_scf_roots",
        ],
        "adapter_source_sha256": _SOURCE_SHA256,
    }

    def quantity(name: str, value: float, definition: str, parents: list[str]) -> None:
        result["quantities"][name] = EnergyQuantity(
            status="available",
            value_hartree=float(value),
            definition=definition,
            parents=parents,
        ).model_dump(mode="json")

    def unavailable(name: str, definition: str, reason: str) -> None:
        result["quantities"][name] = EnergyQuantity(
            status="unavailable", definition=definition, reason=reason, parents=[]
        ).model_dump(mode="json")

    try:
        molecule = request.molecule
        atom_ids = molecule.atom_ids
        if not atom_ids:
            raise BackendInputError(
                "Interaction calculations require explicit atom IDs."
            )
        core_request = {
            "molecule": molecule,
            "method": request.method,
            "properties": ["energy", "gradient"],
        }
        relaxed = request.geometry_protocol == "fully_relaxed"
        harmonic_results = {}
        if relaxed:
            core_request.update(
                properties=["energy", "gradient", "hessian"],
                optimization=request.optimization,
            )
            complex_result = backend.optimize(
                core_request, directory / "complex-relaxed"
            )
        else:
            complex_result = backend.evaluate(
                core_request, directory / "complex-supplied"
            )
        result["components"]["complex"] = complex_result
        complex_energy = _accepted_energy(complex_result, "complex")
        molecule = molecule.model_copy(
            update={"geometry_bohr": complex_result["geometry_bohr"]}
        )
        result["molecule"] = molecule.model_dump(mode="json")
        if relaxed:
            harmonic, evidence = _minimum(complex_result, molecule, request)
            harmonic_results["complex"] = harmonic
            result["stationary_evidence"]["complex"] = evidence
        native_energies, native_ids, relaxed_energies, ghost_energies = [], [], [], []
        raw_gradient = np.asarray(complex_result["gradient_hartree_bohr"]).copy()
        cp_gradient = raw_gradient.copy()
        for fragment in request.fragments:
            monomer = _fragment_molecule(molecule, fragment)
            component_id = f"{fragment.fragment_id}_native_at_complex"
            component = backend.evaluate(
                {
                    "molecule": monomer,
                    "method": request.method,
                    "properties": ["energy", "gradient"],
                },
                directory / component_id,
            )
            result["components"][component_id] = component
            native_energies.append(_accepted_energy(component, component_id))
            native_ids.append(component_id)
            indices = [
                i for i, atom_id in enumerate(atom_ids) if atom_id in fragment.atom_ids
            ]
            raw_gradient[indices] -= np.asarray(component["gradient_hartree_bohr"])
            if relaxed:
                reference_id = f"{fragment.fragment_id}_relaxed"
                reference = backend.optimize(
                    {
                        "molecule": monomer,
                        "method": request.method,
                        "properties": ["energy", "gradient", "hessian"],
                        "optimization": request.optimization,
                    },
                    directory / reference_id,
                )
                result["components"][reference_id] = reference
                relaxed_energies.append(_accepted_energy(reference, reference_id))
                harmonic, evidence = _minimum(reference, monomer, request)
                harmonic_results[fragment.fragment_id] = harmonic
                result["stationary_evidence"][fragment.fragment_id] = evidence
            if request.geometry_protocol == "counterpoise_single_point":
                ghost_id = f"{fragment.fragment_id}_full_complex_basis"
                ghost = GhostPySCFBackend().evaluate(
                    {
                        "molecule": molecule,
                        "method": request.method,
                        "properties": ["energy", "gradient"],
                    },
                    directory / ghost_id,
                    active_atom_ids=fragment.atom_ids,
                    fragment_charge=fragment.charge,
                    fragment_multiplicity=fragment.multiplicity,
                )
                result["components"][ghost_id] = ghost
                ghost_energies.append(_accepted_energy(ghost, ghost_id))
                complex_basis = _basis_signature(complex_result)
                ghost_basis = _basis_signature(ghost)
                if complex_basis != ghost_basis:
                    raise BackendCalculationError(
                        "Complex and ghost monomer Gaussian basis/center "
                        "conventions differ."
                    )
                result["basis_identity_evidence"][ghost_id] = {
                    "same_ordered_basis_centers": True,
                    "same_angular_exponents_and_contractions": True,
                    "cartesian_gaussians": complex_basis["cartesian_gaussians"],
                    "basis_function_count": complex_basis["basis_function_count"],
                    "complex_checkpoint_sha256": electronic_checkpoint_digest(
                        complex_result
                    ),
                    "ghost_checkpoint_sha256": electronic_checkpoint_digest(ghost),
                }
                _json(
                    directory / "complex-gaussian-basis-signature.json", complex_basis
                )
                cp_gradient -= np.asarray(ghost["gradient_hartree_bohr"])
        raw_interaction = complex_energy - sum(native_energies)
        quantity(
            "interaction_uncorrected",
            raw_interaction,
            (
                "E_complex(full basis) - sum E_fragment(own basis; geometry frozen"
                " at complex)"
            ),
            ["complex", *native_ids],
        )
        result["gradients_hartree_bohr"]["interaction_uncorrected"] = (
            raw_gradient.tolist()
        )
        np.save(
            directory / "interaction-uncorrected-gradient.npy",
            raw_gradient,
            allow_pickle=False,
        )
        if ghost_energies:
            parents = [
                "complex",
                *(f"{f.fragment_id}_full_complex_basis" for f in request.fragments),
            ]
            cp_interaction = complex_energy - sum(ghost_energies)
            quantity(
                "interaction_counterpoise",
                cp_interaction,
                (
                    "E_complex(full basis) - sum E_fragment(full complex ghost basis; "
                    "same geometry)"
                ),
                parents,
            )
            quantity(
                "bsse_correction",
                sum(native_energies) - sum(ghost_energies),
                "sum[E_fragment(own basis) - E_fragment(full complex ghost basis)]",
                [*native_ids, *parents[1:]],
            )
            result["gradients_hartree_bohr"]["interaction_counterpoise"] = (
                cp_gradient.tolist()
            )
            np.save(
                directory / "interaction-counterpoise-gradient.npy",
                cp_gradient,
                allow_pickle=False,
            )
            result["quality_flags"].extend(
                [
                    "counterpoise_single_point_not_cp_optimized",
                    "counterpoise_surface_uncertainty_uncalibrated",
                ]
            )
        if relaxed:
            deformation = sum(native_energies) - sum(relaxed_energies)
            binding = complex_energy - sum(relaxed_energies)
            relaxed_ids = [f"{f.fragment_id}_relaxed" for f in request.fragments]
            quantity(
                "deformation",
                deformation,
                (
                    "sum[E_fragment(at complex geometry) - E_fragment(actual relaxed "
                    "minimum)]"
                ),
                [*native_ids, *relaxed_ids],
            )
            quantity(
                "binding_electronic",
                binding,
                (
                    "E_complex(actual uncorrected relaxed minimum) - sum "
                    "E_fragment(actual relaxed minimum)"
                ),
                ["complex", *relaxed_ids],
            )
            quantity(
                "De",
                -binding,
                (
                    "sum E_fragment(relaxed minimum) - E_complex(relaxed minimum); "
                    "positive for a bound complex"
                ),
                ["complex", *relaxed_ids],
            )
            zpe_difference = _minimum_zpe(harmonic_results["complex"]) - sum(
                _minimum_zpe(harmonic_results[f.fragment_id]) for f in request.fragments
            )
            quantity(
                "zpe_binding_correction",
                zpe_difference,
                (
                    "ZPE_complex(actual harmonic Hessian) - sum ZPE_fragment(actual "
                    "harmonic Hessian)"
                ),
                ["complex", *relaxed_ids],
            )
            quantity(
                "D0_harmonic",
                -binding - zpe_difference,
                (
                    "De + sum ZPE_fragment - ZPE_complex; harmonic "
                    "isotopologue-specific approximation"
                ),
                ["complex", *relaxed_ids],
            )
            result["quality_flags"].extend(
                [
                    "uncorrected_relaxed_surface",
                    "D0_harmonic_approximation_not_anharmonic_D0",
                    "local_minima_not_global_minimum_proof",
                    "optimized_fragment_connectivity_requires_separate_validation",
                ]
            )
            if request.rrho is not None:
                for key, harmonic in harmonic_results.items():
                    result["thermochemistry"][key] = rrho_thermochemistry(
                        harmonic,
                        temperature_kelvin=request.rrho.temperature_kelvin,
                        standard_pressure_pa=request.rrho.standard_pressure_pa,
                        rotational_symmetry_number=request.rrho.rotational_symmetry_numbers[
                            key
                        ],
                    )
                corrections = result["thermochemistry"]
                quantity(
                    "binding_gibbs_rrho",
                    binding
                    + corrections["complex"]["gibbs_correction_hartree"]
                    - sum(
                        corrections[f.fragment_id]["gibbs_correction_hartree"]
                        for f in request.fragments
                    ),
                    (
                        "G_complex - sum G_fragment; ideal gas RRHO at "
                        "explicitly recorded pressure/temperature"
                    ),
                    ["complex", *relaxed_ids],
                )
                quantity(
                    "binding_enthalpy_rrho",
                    binding
                    + corrections["complex"]["enthalpy_correction_hartree"]
                    - sum(
                        corrections[f.fragment_id]["enthalpy_correction_hartree"]
                        for f in request.fragments
                    ),
                    "H_complex - sum H_fragment; ideal gas RRHO with ZPE included once",
                    ["complex", *relaxed_ids],
                )
        else:
            unavailable(
                "De",
                "relaxed-minimum electronic dissociation energy",
                (
                    "Supplied geometries and CP single points do not establish relaxed"
                    " stationary endpoints."
                ),
            )
            unavailable(
                "D0_harmonic",
                "De + sum ZPE_fragment - ZPE_complex",
                (
                    "No matched real relaxed endpoint Hessians/ZPE were requested or "
                    "calculated."
                ),
            )
            unavailable(
                "binding_gibbs_rrho",
                "G_complex - sum G_fragment",
                (
                    "No matched real relaxed endpoint Hessians and explicit RRHO state"
                    " parameters were requested."
                ),
            )
        result["status"] = "available"
    except (
        BackendCalculationError,
        BackendInputError,
        RuntimeError,
        ValueError,
        np.linalg.LinAlgError,
    ) as exc:
        result["errors"].append(str(exc))
    typed = InteractionResult.model_validate(result)
    sealed = _seal(directory, typed.model_dump(mode="json"))
    return InteractionResult.model_validate(sealed)
