"""Declaration/leakage and exact mathematical checks, not molecular evidence.

Deliberately invalid metadata are rejection cases only. No constructed datum is
presented as a measured molecular reference or a qualifying engine result.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from cochem_torq.scientific_contracts import (
    BenchmarkSplit,
    ErrorBudget,
    ErrorComponent,
    FamilyAssignment,
    ObservableTarget,
    ProductDeclaration,
    ScientificGoal,
    SpectralRange,
    propagate_linear_uncertainty,
)

ROOT = Path(__file__).resolve().parents[1]


def de_novo_goal(**changes):
    values = {
        "products": (
            ProductDeclaration(
                product_class="A", observables=("equilibrium_rotational_constants",)
            ),
        ),
        "chemical_domain": "restricted closed-shell first-row molecules",
        "allowed_approximations": ("Born-Oppenheimer equilibrium geometry",),
        "target_observables": ("equilibrium_rotational_constants",),
        "reference_policy": "de_novo_no_calibration",
        "unknown_achievable_accuracy_reason": "Independent selected-domain benchmark has not been executed.",
    }
    values.update(changes)
    return ScientificGoal(**values)


def test_de_novo_declaration_roundtrips_immutable_and_unqualified():
    goal = de_novo_goal()
    assert goal.accuracy_status == "unqualified"
    assert goal.budget_binding == "request.resources"
    assert ScientificGoal.model_validate_json(goal.model_dump_json()) == goal
    assert ScientificGoal.model_validate(goal.model_dump(mode="json")) == goal
    with pytest.raises(ValidationError):
        goal.accuracy_status = "qualified"
    with pytest.raises(ValidationError):
        de_novo_goal(chemical_domain="  ")


@pytest.mark.parametrize("product_class", ["B", "C"])
def test_experimental_products_without_actual_anchors_are_rejected(product_class):
    definition = {
        "product_class": product_class,
        "observables": ("equilibrium_rotational_constants",),
    }
    if product_class == "C":
        definition.update(
            difference_origin="experimentally_anchored",
            difference_definition="isotopologue minus parent",
        )
    with pytest.raises(ValidationError, match="anchor"):
        ProductDeclaration(**definition)
    definition["anchor_ids"] = ("deliberately-absent-reference-rejection-case",)
    with pytest.raises(ValidationError, match="actual curated anchor"):
        de_novo_goal(
            products=(ProductDeclaration(**definition),),
            reference_policy="experimental_inference",
        )


def test_theoretical_difference_has_explicit_origin_and_no_hidden_anchors():
    product = ProductDeclaration(
        product_class="C",
        observables=("equilibrium_rotational_constants",),
        difference_origin="fully_theoretical",
        difference_definition="isotopologue minus parent at matched equilibrium level",
    )
    assert (
        de_novo_goal(products=(product,)).products[0].difference_origin
        == "fully_theoretical"
    )
    with pytest.raises(ValidationError, match="cannot use"):
        ProductDeclaration(**{**product.model_dump(), "anchor_ids": ("undeclared",)})


def test_actual_unaccepted_benchmark_proposal_stays_proposed_without_references():
    path = ROOT / "benchmarks/rotational-identification/preregistration.json"
    proposal = json.loads(path.read_text())
    assert proposal["acceptance_record"] is None
    assert (
        proposal["actual_evidence_runner_contract"]["execution_status"]
        == "not_executed"
    )
    assignments = []
    for partition, field in (
        ("development", "development_families"),
        ("calibration", "calibration_families"),
        ("validation", "held_out_families"),
    ):
        for family in proposal["split"][field]:
            parents = tuple(
                candidate["parent_id"]
                for candidate in proposal["primary_candidates"]
                if candidate["chemical_family"] == family
            )
            assert parents
            assignments.append(
                FamilyAssignment(
                    family_id=family, parent_ids=parents, partition=partition
                )
            )
    split = BenchmarkSplit(
        assignments=tuple(assignments),
        preregistration_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        policy_status="proposed",
    )
    goal = de_novo_goal(
        products=(
            ProductDeclaration(
                product_class="BENCHMARK",
                observables=("equilibrium_rotational_constants",),
            ),
        ),
        reference_policy="independent_benchmark",
        benchmark_split=split,
    )
    assert goal.anchors == ()
    assert goal.accuracy_status == "unqualified"
    assert goal.benchmark_split.frozen_before_predictions is None
    assert ScientificGoal.model_validate_json(goal.model_dump_json()) == goal


def assignment(family, parents, partition, sources=()):
    return FamilyAssignment(
        family_id=family,
        parent_ids=parents,
        partition=partition,
        shared_reference_source_ids=sources,
    )


@pytest.mark.parametrize(
    "assignments",
    [
        (
            assignment("same-family", ("one",), "calibration"),
            assignment("same-family", ("two",), "validation"),
        ),
        (
            assignment("family-one", ("same-parent",), "calibration"),
            assignment("family-two", ("same-parent",), "validation"),
        ),
        (
            assignment(
                "family-one",
                ("one",),
                "calibration",
                (
                    hashlib.sha256(
                        b"deliberately-shared-mathematical-metadata-source"
                    ).hexdigest(),
                ),
            ),
            assignment(
                "family-two",
                ("two",),
                "validation",
                (
                    hashlib.sha256(
                        b"deliberately-shared-mathematical-metadata-source"
                    ).hexdigest(),
                ),
            ),
        ),
    ],
)
def test_family_parent_and_source_leakage_are_rejected(assignments):
    with pytest.raises(ValidationError, match="partition|independent"):
        BenchmarkSplit(
            assignments=assignments,
            preregistration_sha256=hashlib.sha256(
                b"deliberately-invalid-contract-metadata"
            ).hexdigest(),
            policy_status="proposed",
        )


def test_inspected_validation_data_cannot_remain_blind():
    values = {
        "assignments": (
            assignment("test-family", ("inspected-parent",), "validation"),
        ),
        "preregistration_sha256": hashlib.sha256(
            b"deliberately-invalid-contract-metadata"
        ).hexdigest(),
        "policy_status": "proposed",
        "inspected_validation_parent_ids": ("inspected-parent",),
    }
    with pytest.raises(ValidationError, match="reclassified"):
        BenchmarkSplit(**values)
    values["assignments"] = (
        assignment("test-family", ("inspected-parent",), "development"),
    )
    assert BenchmarkSplit(**values).assignments[0].partition == "development"


def test_unknown_accuracy_targets_and_spectral_range_are_distinct():
    with pytest.raises(ValidationError, match="unknown achievable"):
        de_novo_goal(unknown_achievable_accuracy_reason=None)
    target = ObservableTarget(
        observable="equilibrium_rotational_constants",
        unit="MHz",
        target_kind="maximum_absolute_error",
        maximum=1.0,
        rationale="Requested tolerance only; independent qualification is still required.",
    )
    goal = de_novo_goal(
        unknown_achievable_accuracy_reason=None,
        error_targets=(target,),
        spectral_range=SpectralRange(minimum=1.0, maximum=10.0, unit="GHz"),
    )
    assert goal.accuracy_status == "unqualified"
    with pytest.raises(ValidationError, match="relative-error"):
        ObservableTarget(**{**target.model_dump(), "target_kind": "relative_error"})
    with pytest.raises(ValidationError, match="increase"):
        SpectralRange(minimum=10.0, maximum=1.0, unit="GHz")
    with pytest.raises(ValidationError, match="duplicate targets"):
        de_novo_goal(error_targets=(target, target))


def unknown_components():
    return tuple(
        ErrorComponent(
            source=source,
            kind="unknown",
            unit="MHz",
            reason="This component has not been measured or calibrated.",
        )
        for source in (
            "numerical_convergence",
            "basis_physical_model",
            "vibrational_nuclear_motion",
            "surrogate",
            "experimental_fit",
        )
    )


def test_unknown_error_components_stay_null_and_cannot_be_combined():
    budget = ErrorBudget(
        observable="equilibrium_rotational_constants",
        unit="MHz",
        components=unknown_components(),
    )
    assert all(component.value is None for component in budget.components)
    assert budget.qualification_status == "uncalibrated"
    with pytest.raises(ValidationError, match="null"):
        ErrorComponent(**{**budget.components[0].model_dump(), "value": 0.0})
    with pytest.raises(ValidationError, match="cannot be summed"):
        ErrorBudget(
            **{
                **budget.model_dump(),
                "independence_assumption": "Deliberately invalid rejection case.",
            }
        )
    with pytest.raises(ValidationError, match="distinct error sources"):
        ErrorBudget(
            observable=budget.observable,
            unit=budget.unit,
            components=(budget.components[0],) * 5,
        )


def test_exact_mathematical_covariance_propagation_includes_correlation():
    # Exact constructed random-variable covariance, not spectroscopy observations.
    result = propagate_linear_uncertainty(
        [2.0, -1.0],
        [[4.0, 1.0], [1.0, 9.0]],
        linearization_domain="Exact linear function of two mathematical random variables.",
    )
    assert result["variance"] == 21.0
    assert result["standard_uncertainty"] == np.sqrt(21.0)
    assert result["coverage_status"] == "uncalibrated"
    independent = propagate_linear_uncertainty(
        [2.0, -1.0],
        [[4.0, 0.0], [0.0, 9.0]],
        linearization_domain="The same exact function under a separately declared diagonal covariance.",
    )
    assert independent["variance"] == 25.0


@pytest.mark.parametrize(
    "covariance",
    [
        [[1.0, 2.0], [2.0, 1.0]],
        [[1.0, 0.0], [1.0, 1.0]],
        [[float("nan"), 0.0], [0.0, 1.0]],
        [[1.0]],
    ],
)
def test_invalid_covariance_cannot_produce_an_uncertainty(covariance):
    with pytest.raises(ValueError, match="Covariance"):
        propagate_linear_uncertainty(
            [1.0, 1.0],
            covariance,
            linearization_domain="Deliberately invalid mathematical rejection case.",
        )


def test_uncertainty_propagation_requires_declared_linearization_domain():
    with pytest.raises(ValueError, match="linearization domain"):
        propagate_linear_uncertainty([1.0], [[1.0]], linearization_domain="")


def test_finite_input_overflow_cannot_be_reported_as_uncertainty():
    with pytest.raises(ValueError, match="finite numeric range"):
        propagate_linear_uncertainty(
            [1e100],
            [[1e200]],
            linearization_domain="Deliberate finite-range mathematical rejection case.",
        )


def test_reference_policy_cannot_imply_an_undeclared_experiment_or_benchmark():
    with pytest.raises(ValidationError, match="identified experimental anchors"):
        de_novo_goal(reference_policy="experimental_inference")
    with pytest.raises(ValidationError, match="BENCHMARK product"):
        de_novo_goal(reference_policy="independent_benchmark")


@pytest.mark.parametrize(
    "covariance",
    [
        [[-1e-16, 0.0], [0.0, 1.0]],
        [[0.0, 1.0], [1.0, 1e16]],
        [[1e-30, 2e-15], [2e-15, 1.0]],
    ],
)
def test_unrelated_large_variance_cannot_hide_invalid_small_components(covariance):
    with pytest.raises(ValueError, match="Covariance"):
        propagate_linear_uncertainty(
            [1.0, 1.0],
            covariance,
            linearization_domain="Deliberately malformed mathematical covariance across disparate scales.",
        )


def test_covariance_diagonal_must_match_even_small_declared_uncertainties():
    mathematical_case = b"Exact constructed five-dimensional mathematical covariance; no molecular reference or measurement."
    components = tuple(
        ErrorComponent(
            source=component.source,
            kind="standard_uncertainty",
            value=1e-8,
            unit="MHz",
            evidence_sha256=hashlib.sha256(mathematical_case).hexdigest(),
        )
        for component in unknown_components()
    )
    values = {
        "observable": "constructed_mathematical_quantity",
        "unit": "MHz",
        "components": components,
        "covariance_order": tuple(component.source for component in components),
        "covariance": tuple(
            tuple(1e-16 if row == column else 0.0 for column in range(5))
            for row in range(5)
        ),
    }
    assert ErrorBudget(**values).qualification_status == "uncalibrated"
    values["covariance"] = tuple(
        tuple(1e-14 if row == column else 0.0 for column in range(5))
        for row in range(5)
    )
    with pytest.raises(ValidationError, match="diagonal must match"):
        ErrorBudget(**values)


def malformed_anchor_metadata():
    # Rejection input only: the cited repository is software, not an experiment.
    # The numbers and bytes here deliberately cannot establish physical evidence.
    digest = hashlib.sha256(
        b"Deliberately invalid nonexperimental anchor metadata."
    ).hexdigest()
    return {
        "anchor_id": "rejection-case-only",
        "observable": "equilibrium_rotational_constants",
        "parent_id": "held-out-parent",
        "isotopologue_id": "rejection-case-only",
        "state_id": "rejection-case-only",
        "frame_or_hamiltonian_convention": "rejection-case-only",
        "value": {"value": 0.0, "unit": "MHz"},
        "uncertainty": {"value": 1.0, "unit": "MHz"},
        "uncertainty_kind": "standard_uncertainty",
        "data_usage": "validation",
        "source": {
            "citation": "Deliberately invalid nonexperimental metadata rejection case.",
            "persistent_identifier": "https://github.com/ProfJJK-CoChem/CoChem-TORQ",
            "source_artifact_sha256": digest,
            "datum_locator": "rejection case only",
            "curator": "deliberately-invalid-metadata",
            "curated_at": datetime.now(timezone.utc),
            "curation_record_sha256": digest,
            "reference_kind": "experimental_molecular_record",
        },
    }


def test_validation_observation_cannot_be_used_as_prediction_anchor():
    product = ProductDeclaration(
        product_class="B",
        observables=("equilibrium_rotational_constants",),
        anchor_ids=("rejection-case-only",),
    )
    with pytest.raises(ValidationError, match="cannot anchor a prediction"):
        de_novo_goal(
            products=(product,),
            reference_policy="experimental_inference",
            anchors=(malformed_anchor_metadata(),),
        )


def test_anchor_source_must_bind_the_declared_family_source_digest():
    declared_source = hashlib.sha256(
        b"A different deliberately-invalid source declaration."
    ).hexdigest()
    split = BenchmarkSplit(
        assignments=(
            assignment(
                "held-out-family",
                ("held-out-parent",),
                "validation",
                (declared_source,),
            ),
        ),
        preregistration_sha256=declared_source,
        policy_status="proposed",
    )
    product = ProductDeclaration(
        product_class="BENCHMARK",
        observables=("equilibrium_rotational_constants",),
        anchor_ids=("rejection-case-only",),
    )
    with pytest.raises(ValidationError, match="bind the source digest"):
        de_novo_goal(
            products=(product,),
            reference_policy="independent_benchmark",
            anchors=(malformed_anchor_metadata(),),
            benchmark_split=split,
        )
