"""Bounded genuine anharmonic validation; no production rovibrational claim.

The runner executes actual energy calculations through the same PySCF adapter
and retains every displaced result. Vibrational-only and semirigid Watson results
remain separately named, with explicit applicability gates and no identification
accuracy claim.
"""

from __future__ import annotations

import math
import os
import time
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from hashlib import sha256
from itertools import combinations_with_replacement, product
from pathlib import Path
from typing import Any

import numpy as np

from .domain import CalculationRequest, canonical_json, digest
from .engines.pyscf_backend import PySCFBackend
from .registry import get_profile
from .spectroscopy import (
    EnergyEvaluation,
    HarmonicResult,
    analyze_resonances,
    build_force_field,
    vibrational_vpt2,
)
from .spectroscopy.forcefield import _STENCILS
from .spectroscopy.harmonic import equilibrium_rotor
from .spectroscopy.results import (
    ForceFieldData,
    ResonanceAnalysisData,
    VibrationalVPT2Data,
    make_scientific_context,
)


def _plain(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _checkpoint(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".checkpoint")
    with temporary.open("wb") as stream:
        stream.write(canonical_json(_plain(value)) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def planned_energy_evaluations(mode_count: int, steps: tuple[float, float]) -> int:
    """Count distinct points of the actual full cubic/quartic central stencil."""
    displacements = {tuple([0.0] * mode_count)}
    for step in steps:
        for order in (3, 4):
            for indices in combinations_with_replacement(range(mode_count), order):
                multiplicities = sorted(Counter(indices).items())
                for choice in product(
                    *[_STENCILS[power][0] for _, power in multiplicities]
                ):
                    q = [0.0] * mode_count
                    for (mode, _), node in zip(multiplicities, choice):
                        q[mode] = node * step
                    displacements.add(tuple(q))
    return len(displacements)


def execute_anharmonic_validation(
    request: CalculationRequest | dict[str, Any],
    harmonic_result: HarmonicResult,
    backend_request: dict[str, Any],
    workspace: str | Path,
) -> dict[str, Any]:
    """Execute a ≤3-mode RHF/STO-3G experimental validation protocol.

    Mode count, request scope, actual engine/recipe and bounded task/resource plan
    are checked before creating files or running a displaced calculation. Defaults
    are explicitly recorded protocol parameters, never missing physical values.
    The enclosing application owns process/wall-time isolation. Direct callers
    also receive a checked cumulative wall-time budget between actual engine calls.
    """
    model = (
        request
        if isinstance(request, CalculationRequest)
        else CalculationRequest.model_validate(request)
    )
    if model.recipe != "hf-sto-3g-anharmonic-validation":
        raise ValueError(
            "Select the explicit local HF/STO-3G anharmonic-validation recipe."
        )
    profile = get_profile(model.recipe)
    if not set(model.products) & {
        "anharmonic_force_field",
        "vpt2",
        "ground_state_constants",
    }:
        raise ValueError(
            "Anharmonic work requires an explicit anharmonic/VPT2 validation product."
        )
    if not isinstance(harmonic_result, HarmonicResult):
        raise ValueError("A genuine projected HarmonicResult is required.")
    mode_count = len(harmonic_result.frequencies_cm1)
    if not 1 <= mode_count <= 3:
        raise ValueError(
            "This bounded anharmonic validation profile permits one to three modes."
        )
    if harmonic_result.stationary_character != "positive_definite_vibrational_hessian":
        raise ValueError("A complete positive internal Hessian is required.")
    if (
        not math.isfinite(harmonic_result.external_residual_relative)
        or harmonic_result.external_residual_relative
        > profile["harmonic_external_residual_relative_tolerance"]
    ):
        raise ValueError(
            "The reference Hessian fails the recipe's external-motion residual gate."
        )
    transform = harmonic_result.dimensionless_to_cartesian
    if (
        transform is None
        or transform.shape != (harmonic_result.coordinates_bohr.size, mode_count)
        or not np.isfinite(transform).all()
        or np.any(harmonic_result.angular_frequencies_au <= 0)
        or not np.isfinite(harmonic_result.angular_frequencies_au).all()
        or not np.isfinite(harmonic_result.frequencies_cm1).all()
        or len(harmonic_result.angular_frequencies_au) != mode_count
    ):
        raise ValueError(
            "The complete normal-coordinate transform/frequency shape is invalid."
        )
    method = backend_request.get("method", {})
    if (
        method.get("name") != "hf"
        or str(method.get("basis", "")).lower() != "sto-3g"
        or method.get("reference", "restricted") != "restricted"
        or method.get("dispersion") is not None
        or method.get("frozen_core", False) is not False
    ):
        raise ValueError(
            "This runner executes restricted all-electron HF/STO-3G without dispersion."
        )
    molecule = backend_request.get("molecule", {})
    if (
        molecule.get("symbols") != model.molecule.symbols
        or molecule.get("charge") != model.molecule.charge
        or molecule.get("multiplicity") != model.molecule.multiplicity
        or model.molecule.multiplicity != 1
    ):
        raise ValueError(
            "Backend symbols/order and state must match the original request."
        )
    if harmonic_result.coordinates_bohr.shape != (len(model.molecule.symbols), 3):
        raise ValueError("Harmonic atom count must match the requested molecule.")
    from Libraries.cochem_isotopes import isotope_record

    isotope_numbers = model.molecule.isotopes or [None] * len(model.molecule.symbols)
    isotope_records = [
        isotope_record(f"{number}{symbol}" if number is not None else symbol)
        for number, symbol in zip(isotope_numbers, model.molecule.symbols)
    ]
    masses = [record["mass_u"] for record in isotope_records]
    if not np.array_equal(harmonic_result.isotope_masses_u, np.asarray(masses)):
        raise ValueError(
            "Harmonic masses must exactly match the resolved requested isotopologue."
        )

    # This explicit protocol is intentionally narrow. Additional recipes/settings
    # need their own named validation protocol; no method switching occurs here.
    configuration = profile["anharmonic"]
    if configuration["max_modes"] != 3:
        raise ValueError("The recipe and bounded runner mode scopes disagree.")
    protocol: dict[str, Any] = {
        "steps_dimensionless": configuration["steps_dimensionless"],
        "max_energy_evaluations": configuration["max_evaluations"],
        "absolute_derivative_tolerance_hartree": configuration[
            "absolute_tolerance_hartree"
        ],
        "relative_derivative_tolerance": configuration["relative_tolerance"],
        "resonance_detuning_threshold_cm1": 10.0,
        "coupling_to_detuning_threshold": 0.1,
        "reference_gradient_max_hartree_bohr": 1.5e-5,
        "reference_gradient_rms_hartree_bohr": 1e-5,
        "normal_mode_convention": harmonic_result.convention,
        "displacement_coordinate_convention": (
            "q=sqrt(omega)*Q; dimensionless; Cartesian=M^(-1/2)LQ"
        ),
        "vibrational_hamiltonian": (
            "rectilinear H0+V3+V4; no rotational/Coriolis/curvilinear kinetic terms"
        ),
        "rovibrational_protocol": configuration_for_rovibrational(profile),
    }
    planned = planned_energy_evaluations(
        mode_count, tuple(protocol["steps_dimensionless"])
    )
    if planned > protocol["max_energy_evaluations"]:
        raise ValueError(
            "The complete requested derivative stencil exceeds the bounded task budget."
        )
    settings = backend_request.get("settings", {})
    if (
        settings.get("threads", 1) > model.resources.cores
        or settings.get("memory_mb", 1500) > model.resources.memory_mb
    ):
        raise ValueError(
            "The derivative evaluator exceeds the approved CPU/memory budget."
        )
    probe = PySCFBackend.probe()
    if not probe.get("available"):
        raise RuntimeError(
            "A genuine PySCF installation is required for anharmonic validation."
        )
    if probe.get("version") != "2.14.0":
        raise ValueError("This explicit research protocol requires PySCF 2.14.0.")
    directory = Path(workspace).absolute()
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(
            "Use an empty new anharmonic workspace; raw observations are immutable."
        )
    directory.mkdir(parents=True, exist_ok=True)
    protocol["planned_energy_evaluations"] = planned
    protocol["engine"] = probe
    protocol["method"] = deepcopy(method)
    protocol["settings"] = deepcopy(settings)
    protocol["request_sha256"] = digest(model.model_dump(mode="json"))
    protocol["recipe_sha256"] = profile["recipe_sha256"]
    protocol["harmonic_source_digest"] = harmonic_result.source_digest
    protocol["isotope_provenance"] = isotope_records
    protocol["runner_source_sha256"] = sha256(Path(__file__).read_bytes()).hexdigest()
    protocol_digest = digest(protocol)
    stages: dict[str, dict[str, Any]] = {
        name: {
            "status": "blocked",
            "value": None,
            "reason": "A required preceding numerical stage has not completed.",
        }
        for name in (
            "anharmonic_force_field",
            "resonance_analysis",
            "vibrational_vpt2",
            "rovibrational_precursors",
            "vibration_rotation_corrections",
            "semirigid_vpt2",
            "centrifugal_distortion",
            "ground_state_constants",
        )
    }
    result: dict[str, Any] = {
        "schema_version": "cochem.torq.anharmonic-validation/1",
        "status": "experimental_unqualified",
        "outcome": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol,
        "protocol_sha256": protocol_digest,
        "stages": stages,
        "displaced_calculations": [],
        "errors": [],
        "rotation_vibration_available": False,
        "independent_scientific_qualification": False,
        "identification_ready": False,
    }
    _checkpoint(directory / "protocol.json", protocol)
    _checkpoint(directory / "harmonic-input.json", _plain(harmonic_result))
    _checkpoint(directory / "request.json", model.model_dump(mode="json"))
    _checkpoint(directory / "result.json", result)
    started = time.monotonic()
    backend = PySCFBackend()

    def evaluate(coordinates: np.ndarray) -> EnergyEvaluation:
        if time.monotonic() - started >= model.resources.wall_seconds:
            raise RuntimeError(
                "Anharmonic validation exhausted its approved wall-time budget."
            )
        calculation = deepcopy(backend_request)
        calculation["molecule"]["geometry_bohr"] = coordinates.tolist()
        identifier = len(result["displaced_calculations"])
        # Independently check stationarity at the force-field reference;
        # a positive projected Hessian alone does not prove a stationary point.
        calculation["properties"] = (
            ["energy", "gradient"] if identifier == 0 else ["energy"]
        )
        raw_directory = directory / f"energy-{identifier:03d}"
        raw = backend.evaluate(calculation, raw_directory)
        manifest_sha256 = sha256(
            (raw_directory / "manifest.json").read_bytes()
        ).hexdigest()
        if raw.get("manifest_sha256") != manifest_sha256:
            raise RuntimeError(
                "Displaced manifest digest differs from actual artifact bytes."
            )
        observation = {
            "id": identifier,
            "workspace": raw_directory.name,
            "geometry_sha256": digest(coordinates.tolist()),
            "manifest_sha256": manifest_sha256,
            "status": raw.get("status"),
            "energy_hartree": raw.get("energy_hartree"),
            "errors": raw.get("errors", []),
        }
        result["displaced_calculations"].append(observation)
        _checkpoint(directory / "result.json", result)
        if (
            raw.get("status") != "complete"
            or raw.get("scf", {}).get("converged") is not True
            or raw.get("stability", {}).get("status") != "stable"
        ):
            raise RuntimeError(
                f"Actual displaced calculation {identifier} failed "
                "convergence/stability; raw artifacts retained."
            )
        if (
            raw.get("method", {}).get("name") != "hf"
            or raw.get("method", {}).get("basis", "").lower() != "sto-3g"
            or not np.array_equal(np.asarray(raw.get("geometry_bohr")), coordinates)
        ):
            raise RuntimeError(
                "Displaced geometry/method differs from the approved calculation."
            )
        if identifier == 0:
            gradient = np.asarray(raw.get("gradient_hartree_bohr"), dtype=float)
            if gradient.shape != coordinates.shape or not np.isfinite(gradient).all():
                raise RuntimeError(
                    "Reference stationarity requires an actual finite gradient."
                )
            gmax = float(np.max(np.abs(gradient)))
            grms = float(np.sqrt(np.mean(gradient * gradient)))
            observation["gradient_max_hartree_bohr"] = gmax
            observation["gradient_rms_hartree_bohr"] = grms
            _checkpoint(directory / "result.json", result)
            if (
                gmax > protocol["reference_gradient_max_hartree_bohr"]
                or grms > protocol["reference_gradient_rms_hartree_bohr"]
            ):
                raise RuntimeError(
                    "Reference geometry fails independent stationarity criteria."
                )
        energy = raw.get("energy_hartree")
        if (
            isinstance(energy, bool)
            or not isinstance(energy, (float, int))
            or not math.isfinite(energy)
        ):
            raise RuntimeError(
                "Displaced calculation did not supply finite authentic hartree energy."
            )
        return EnergyEvaluation(float(energy), manifest_sha256)

    try:
        field = build_force_field(
            harmonic_result,
            evaluate,
            evaluator_identity=(
                f"PySCF 2.14.0 RHF/STO-3G; "
                f"protocol SHA256 {protocol_digest}; experimental"
            ),
            steps=tuple(protocol["steps_dimensionless"]),
            absolute_tolerance_hartree=protocol[
                "absolute_derivative_tolerance_hartree"
            ],
            relative_tolerance=protocol["relative_derivative_tolerance"],
            max_evaluations=protocol["max_energy_evaluations"],
        )
        context_arguments = {
            "molecule": model.molecule.model_dump(mode="json"),
            "geometry_bohr": harmonic_result.coordinates_bohr,
            "isotope_provenance": isotope_records,
            "recipe_sha256": profile["recipe_sha256"],
            "protocol_sha256": protocol_digest,
            "evidence_class": "engine_calculation",
            "harmonic": harmonic_result,
            "principal_axes_columns": equilibrium_rotor(
                harmonic_result.coordinates_bohr, masses
            ).principal_axes_columns,
        }
        field_value = _plain(field)
        field_value["scientific_context"] = make_scientific_context(
            **context_arguments,
            parent_artifact_sha256=[
                sha256((directory / name).read_bytes()).hexdigest()
                for name in ("protocol.json", "harmonic-input.json")
            ]
            + [
                record["manifest_sha256"] for record in result["displaced_calculations"]
            ],
        )
        field_value = ForceFieldData.model_validate(field_value).model_dump(mode="json")
        _checkpoint(directory / "force-field.json", field_value)
        field_artifact_sha256 = sha256(
            (directory / "force-field.json").read_bytes()
        ).hexdigest()
        stages["anharmonic_force_field"] = {
            "status": "available",
            "value": field_value,
            "reason": None,
            "qualification": "experimental_unqualified",
            "quality_flags": []
            if field.derivative_converged
            else ["derivative_convergence_not_established"],
        }
        result["outcome"] = "partial"
        _checkpoint(directory / "result.json", result)
        if not field.derivative_converged:
            result["errors"].append(
                {
                    "code": "DERIVATIVE_CONVERGENCE_FAILED",
                    "message": (
                        "Coarse/fine tensors retained; failed comparison "
                        "blocks dependent perturbation stages."
                    ),
                }
            )
        else:
            resonances = analyze_resonances(
                field,
                detuning_threshold_cm1=protocol["resonance_detuning_threshold_cm1"],
                coupling_to_detuning_threshold=protocol[
                    "coupling_to_detuning_threshold"
                ],
            )
            stages["resonance_analysis"] = {
                "status": "available",
                "value": {
                    "resonances": _plain(resonances),
                    "force_field_sha256": field.source_digest,
                    "coriolis_resonances": "not_implemented",
                    "protocol_sha256": protocol_digest,
                    "scientific_context": make_scientific_context(
                        **context_arguments,
                        parent_artifact_sha256=[field_artifact_sha256],
                    ),
                },
                "reason": None,
                "qualification": "experimental_unqualified",
            }
            resonance_value = ResonanceAnalysisData.model_validate(
                stages["resonance_analysis"]["value"]
            ).model_dump(mode="json")
            stages["resonance_analysis"]["value"] = resonance_value
            _checkpoint(directory / "resonance-analysis.json", resonance_value)
            _checkpoint(directory / "result.json", result)
            try:
                vpt = vibrational_vpt2(
                    field,
                    detuning_threshold_cm1=protocol["resonance_detuning_threshold_cm1"],
                    coupling_to_detuning_threshold=protocol[
                        "coupling_to_detuning_threshold"
                    ],
                )
                vpt_value = _plain(vpt)
                vpt_value["scientific_context"] = make_scientific_context(
                    **context_arguments,
                    parent_artifact_sha256=[
                        field_artifact_sha256,
                        sha256(
                            (directory / "resonance-analysis.json").read_bytes()
                        ).hexdigest(),
                    ],
                )
                vpt_value = VibrationalVPT2Data.model_validate(vpt_value).model_dump(
                    mode="json"
                )
                _checkpoint(directory / "vibrational-vpt2.json", vpt_value)
                stages["vibrational_vpt2"] = {
                    "status": "available",
                    "value": vpt_value,
                    "reason": None,
                    "qualification": "experimental_unqualified",
                }
                result["outcome"] = "complete"
            except ValueError as error:
                stages["vibrational_vpt2"] = {
                    "status": "blocked",
                    "value": None,
                    "reason": str(error),
                    "qualification": "experimental_unqualified",
                }
            _execute_watson_stages(
                model,
                harmonic_result,
                field_value,
                context_arguments,
                result,
                directory,
            )
    except (ValueError, RuntimeError, OSError, np.linalg.LinAlgError) as error:
        if stages["anharmonic_force_field"]["status"] != "available":
            stages["anharmonic_force_field"] = {
                "status": "failed",
                "value": None,
                "reason": str(error),
            }
            result["outcome"] = "failed"
        result["errors"].append(
            {"code": "ANHARMONIC_VALIDATION_FAILED", "message": str(error)}
        )
    result["elapsed_wall_seconds"] = time.monotonic() - started
    _checkpoint(directory / "result.json", result)
    records = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "research-manifest.json":
            records.append(
                {
                    "path": path.relative_to(directory).as_posix(),
                    "sha256": sha256(path.read_bytes()).hexdigest(),
                    "size_bytes": path.stat().st_size,
                }
            )
    manifest = {
        "schema_version": "cochem.torq.anharmonic-artifacts/1",
        "protocol_sha256": protocol_digest,
        "files": records,
    }
    _checkpoint(directory / "research-manifest.json", manifest)
    result["artifact_manifest_sha256"] = sha256(
        (directory / "research-manifest.json").read_bytes()
    ).hexdigest()
    result["artifact_manifest_path"] = str(directory / "research-manifest.json")
    return result


def configuration_for_rovibrational(profile: dict[str, Any]) -> dict[str, Any]:
    """Hash explicit model limits in the reviewed protocol before engine work."""
    from .spectroscopy.rovibrational_perturbation import WatsonCorrectionProtocol

    return WatsonCorrectionProtocol.model_validate(profile["rovibrational"]).model_dump(
        mode="json"
    )


def _execute_watson_stages(
    request: CalculationRequest,
    harmonic: HarmonicResult,
    field_value: dict[str, Any],
    context_arguments: dict[str, Any],
    result: dict[str, Any],
    directory: Path,
) -> None:
    """Derive bounded canonical model products from authenticated native parents."""
    from .spectroscopy.advanced_products import (
        ModelGroundStateConstants,
        UnreducedHarmonicDistortion,
    )
    from .spectroscopy.harmonic import equilibrium_rotor
    from .spectroscopy.results import ScientificContext
    from .spectroscopy.rovibrational import build_rovibrational_precursors
    from .spectroscopy.rovibrational_perturbation import (
        WatsonCorrectionProtocol,
        calculate_watson_vibration_rotation,
    )
    from .spectroscopy.rovibrational_solver import StationaryReference

    stages = result["stages"]
    names = (
        "rovibrational_precursors",
        "vibration_rotation_corrections",
        "semirigid_vpt2",
        "centrifugal_distortion",
        "ground_state_constants",
    )
    try:
        field = ForceFieldData.model_validate(field_value)
        rotor = equilibrium_rotor(harmonic.coordinates_bohr, harmonic.isotope_masses_u)
        context = ScientificContext.model_validate(field.scientific_context)
        precursors = build_rovibrational_precursors(harmonic, rotor, context)
        _checkpoint(
            directory / "rovibrational-precursors.json",
            precursors.model_dump(mode="json"),
        )
        stages["rovibrational_precursors"] = {
            "status": "available",
            "value": precursors.model_dump(mode="json"),
            "reason": None,
            "qualification": "experimental_unqualified",
        }
        zero = next(
            row
            for row in result["displaced_calculations"]
            if row["id"] == 0
            and row["geometry_sha256"] == digest(harmonic.coordinates_bohr.tolist())
        )
        native_directory = directory / zero["workspace"]
        from .scan import _verify_native

        native, actual_manifest = _verify_native(native_directory)
        if actual_manifest != zero["manifest_sha256"]:
            raise ValueError("The retained force-field reference manifest changed.")
        if not np.array_equal(native["geometry_bohr"], harmonic.coordinates_bohr):
            raise ValueError(
                "The actual stationarity reference differs from the force field."
            )
        stationary = StationaryReference.model_validate_json(
            canonical_json(
                {
                    "source_artifact_sha256": actual_manifest,
                    "geometry_sha256": digest(harmonic.coordinates_bohr.tolist()),
                    "evidence_class": "engine_calculation",
                    "gradient_hartree_bohr": native["gradient_hartree_bohr"],
                    "maximum_allowed_gradient_hartree_bohr": result["protocol"][
                        "reference_gradient_max_hartree_bohr"
                    ],
                }
            )
        )
        correction = calculate_watson_vibration_rotation(
            precursors,
            field,
            stationary_reference=stationary,
            protocol=WatsonCorrectionProtocol.model_validate_json(
                canonical_json(result["protocol"]["rovibrational_protocol"])
            ),
        )
        _checkpoint(
            directory / "watson-corrections.json", correction.model_dump(mode="json")
        )
        correction_sha256 = sha256(
            (directory / "watson-corrections.json").read_bytes()
        ).hexdigest()
        derived_context = make_scientific_context(
            **context_arguments,
            parent_artifact_sha256=[correction_sha256],
        )
        stages["vibration_rotation_corrections"] = {
            "status": "available",
            "value": correction.model_dump(mode="json"),
            "reason": None,
            "qualification": "experimental_unqualified",
            "quality_flags": [
                "gated_nonresonant_model",
                "identification_accuracy_not_established",
            ],
        }
        distortion = UnreducedHarmonicDistortion.model_validate(
            {
                "scientific_context": derived_context,
                "distortion": correction.harmonic_distortion,
                "watson_result_artifact_sha256": correction_sha256,
            }
        )
        _checkpoint(
            directory / "unreduced-harmonic-distortion.json",
            distortion.model_dump(mode="json"),
        )
        stages["centrifugal_distortion"] = {
            "status": "available",
            "value": distortion.model_dump(mode="json"),
            "reason": None,
            "qualification": "experimental_unqualified",
            "quality_flags": [
                "unreduced_harmonic_model",
                "A_and_S_reductions_unavailable",
            ],
        }
        if correction.semirigid_vpt2 is not None:
            stages["semirigid_vpt2"] = {
                "status": "available",
                "value": correction.semirigid_vpt2.model_dump(mode="json"),
                "reason": None,
                "qualification": "experimental_unqualified",
                "quality_flags": [
                    "nonresonant_semirigid_model",
                    "full_resonant_GVPT2_unavailable",
                ],
            }
        else:
            stages["semirigid_vpt2"] = {
                "status": "blocked",
                "value": None,
                "reason": "; ".join(correction.semirigid_vpt2_blocking_reasons),
            }
        if correction.ground_state_constants_mhz is not None:
            constants = ModelGroundStateConstants.model_validate(
                {
                    "scientific_context": derived_context,
                    "constants_mhz": correction.ground_state_constants_mhz,
                    "watson_result_artifact_sha256": correction_sha256,
                    "watson_result": correction,
                }
            )
            _checkpoint(
                directory / "model-ground-state-constants.json",
                constants.model_dump(mode="json"),
            )
            stages["ground_state_constants"] = {
                "status": "available",
                "value": constants.model_dump(mode="json"),
                "reason": None,
                "qualification": "experimental_unqualified",
                "quality_flags": [
                    "nonresonant_model_B0",
                    "identification_accuracy_not_established",
                ],
            }
            result["rotation_vibration_available"] = True
        else:
            stages["ground_state_constants"] = {
                "status": "blocked",
                "value": None,
                "reason": "; ".join(correction.states[0].blocking_reasons),
            }
    except (
        ValueError,
        RuntimeError,
        OSError,
        StopIteration,
        np.linalg.LinAlgError,
    ) as exc:
        for name in names:
            if stages[name]["status"] != "available":
                stages[name] = {"status": "blocked", "value": None, "reason": str(exc)}
    if (
        "ground_state_constants" in request.products
        and stages["ground_state_constants"]["status"] != "available"
    ):
        if result["outcome"] == "complete":
            result["outcome"] = "partial"
        result["errors"].append(
            {
                "code": "MODEL_GROUND_STATE_CONSTANTS_UNAVAILABLE",
                "message": stages["ground_state_constants"]["reason"],
            }
        )
