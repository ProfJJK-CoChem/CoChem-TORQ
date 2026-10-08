"""Scientific declarations and leakage controls; declarations do not qualify results.

These contracts distinguish a requested scientific product from an achieved
accuracy claim. Reference digests identify bytes; they do not authenticate an
experiment, replace curation or establish independent predictive validation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, model_validator

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Name = Annotated[str, Field(min_length=1, max_length=256, pattern=r"\S")]
Unit = Literal[
    "MHz",
    "GHz",
    "cm^-1",
    "hartree",
    "bohr",
    "angstrom",
    "debye",
    "kJ/mol",
    "dimensionless",
    "ppm",
]
Partition = Literal["development", "calibration", "validation"]
ERROR_SOURCES = frozenset(
    {
        "numerical_convergence",
        "basis_physical_model",
        "vibrational_nuclear_motion",
        "surrogate",
        "experimental_fit",
    }
)


class ScientificContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Quantity(ScientificContract):
    value: StrictFloat
    unit: Unit


class ObservableTarget(ScientificContract):
    observable: Name
    unit: Unit
    target_kind: Literal["maximum_absolute_error", "search_window", "relative_error"]
    maximum: StrictFloat = Field(gt=0)
    rationale: Name

    @model_validator(mode="after")
    def target_units(self):
        if self.target_kind == "relative_error" and self.unit not in {
            "ppm",
            "dimensionless",
        }:
            raise ValueError(
                "A relative-error target requires ppm or dimensionless units."
            )
        return self


class SpectralRange(ScientificContract):
    minimum: StrictFloat = Field(ge=0)
    maximum: StrictFloat = Field(gt=0)
    unit: Literal["MHz", "GHz", "cm^-1"]

    @model_validator(mode="after")
    def increasing(self):
        if self.maximum <= self.minimum:
            raise ValueError("The declared spectral range must increase.")
        return self


class ReferenceSource(ScientificContract):
    """Curated source declaration; byte identity is independently checked."""

    citation: Name
    persistent_identifier: Name
    source_artifact_sha256: Sha256
    datum_locator: Name
    curator: Name
    curated_at: datetime
    curation_record_sha256: Sha256
    reference_kind: Literal["experimental_molecular_record"]

    @model_validator(mode="after")
    def curation(self):
        if self.curated_at.utcoffset() is None:
            raise ValueError("Reference curation requires an aware timestamp.")
        text = (
            self.citation,
            self.persistent_identifier,
            self.datum_locator,
            self.curator,
        )
        if any(not item.strip() for item in text):
            raise ValueError("Reference provenance cannot contain blank fields.")
        if not (
            self.persistent_identifier.startswith("doi:")
            or self.persistent_identifier.startswith("https://")
        ):
            raise ValueError(
                "Use an explicit DOI or HTTPS persistent source identifier."
            )
        if (
            self.source_artifact_sha256 == "0" * 64
            or self.curation_record_sha256 == "0" * 64
        ):
            raise ValueError(
                "A zero digest does not identify an actual reference artifact."
            )
        return self


class ExperimentalAnchor(ScientificContract):
    anchor_id: Name
    observable: Name
    parent_id: Name
    isotopologue_id: Name
    state_id: Name
    frame_or_hamiltonian_convention: Name
    value: Quantity
    uncertainty: Quantity
    uncertainty_kind: Literal["standard_uncertainty", "expanded_uncertainty"]
    coverage_factor: StrictFloat | None = Field(default=None, gt=0)
    source: ReferenceSource
    data_usage: Literal["inference", "calibration", "validation"]

    @model_validator(mode="after")
    def explicit_uncertainty(self):
        if self.value.unit != self.uncertainty.unit or self.uncertainty.value <= 0:
            raise ValueError(
                "Experimental molecular uncertainty must be positive in "
                "the observable unit."
            )
        if (self.uncertainty_kind == "expanded_uncertainty") != (
            self.coverage_factor is not None
        ):
            raise ValueError(
                "Expanded uncertainty requires its coverage factor; "
                "standard uncertainty does not."
            )
        return self


class ProductDeclaration(ScientificContract):
    product_class: Literal["A", "B", "C", "BENCHMARK"]
    observables: tuple[Name, ...] = Field(min_length=1)
    anchor_ids: tuple[Name, ...] = ()
    difference_origin: (
        Literal["fully_theoretical", "experimentally_anchored"] | None
    ) = None
    difference_definition: Name | None = None

    @model_validator(mode="after")
    def product_semantics(self):
        if len(set(self.observables)) != len(self.observables) or len(
            set(self.anchor_ids)
        ) != len(self.anchor_ids):
            raise ValueError(
                "Product observable and anchor identities must be distinct."
            )
        if self.product_class == "B" and not self.anchor_ids:
            raise ValueError(
                "Product B requires explicitly identified experimental anchors."
            )
        if self.product_class == "C":
            if self.difference_origin is None or self.difference_definition is None:
                raise ValueError(
                    "Product C requires its difference definition and "
                    "theoretical/anchored origin."
                )
            if (self.difference_origin == "experimentally_anchored") != bool(
                self.anchor_ids
            ):
                raise ValueError(
                    "Anchored differences require anchors; theoretical "
                    "differences cannot use them."
                )
        elif (
            self.difference_origin is not None or self.difference_definition is not None
        ):
            raise ValueError("Difference semantics belong to product C only.")
        if self.product_class == "A" and self.anchor_ids:
            raise ValueError(
                "Absolute de novo product A cannot be experimentally anchored."
            )
        return self


class FamilyAssignment(ScientificContract):
    """All related parents/isotopologues/conformers stay in one family partition."""

    family_id: Name
    parent_ids: tuple[Name, ...] = Field(min_length=1)
    partition: Partition
    # Source identity is the actual retained source-artifact digest, not an alias.
    shared_reference_source_ids: tuple[Sha256, ...] = ()

    @model_validator(mode="after")
    def unique_members(self):
        if len(set(self.parent_ids)) != len(self.parent_ids) or len(
            set(self.shared_reference_source_ids)
        ) != len(self.shared_reference_source_ids):
            raise ValueError("Family parent and reference source IDs must be distinct.")
        if "0" * 64 in self.shared_reference_source_ids:
            raise ValueError(
                "A zero digest cannot identify a benchmark reference source."
            )
        return self


class BenchmarkSplit(ScientificContract):
    schema_version: Literal["cochem.torq.benchmark-split/1"] = (
        "cochem.torq.benchmark-split/1"
    )
    sampling_unit: Literal["chemical_family"] = "chemical_family"
    assignments: tuple[FamilyAssignment, ...] = Field(min_length=1)
    preregistration_sha256: Sha256
    frozen_before_predictions: datetime | None = None
    policy_status: Literal["proposed", "frozen"]
    inspected_validation_parent_ids: tuple[Name, ...] = ()

    @model_validator(mode="after")
    def leakage_controls(self):
        families, parents, source_partitions = set(), {}, {}
        for assignment in self.assignments:
            if assignment.family_id in families:
                raise ValueError(
                    "A chemical family cannot occupy multiple benchmark partitions."
                )
            families.add(assignment.family_id)
            for parent in assignment.parent_ids:
                if parent in parents:
                    raise ValueError(
                        "A parent and all its related states must occupy one "
                        "family partition."
                    )
                parents[parent] = assignment.partition
            for source in assignment.shared_reference_source_ids:
                existing = source_partitions.setdefault(source, assignment.partition)
                if existing != assignment.partition:
                    raise ValueError(
                        "A shared reference source cannot count as independent "
                        "across partitions."
                    )
        if (self.policy_status == "frozen") != (
            self.frozen_before_predictions is not None
        ):
            raise ValueError(
                "A frozen split requires a timestamp; proposed splits "
                "cannot claim freezing."
            )
        if (
            self.frozen_before_predictions is not None
            and self.frozen_before_predictions.utcoffset() is None
        ):
            raise ValueError("Split freezing requires an aware timestamp.")
        if any(
            parent not in parents for parent in self.inspected_validation_parent_ids
        ):
            raise ValueError("Inspected parents must exist in the declared split.")
        if any(
            parents[parent] == "validation"
            for parent in self.inspected_validation_parent_ids
        ):
            raise ValueError(
                "Inspected held-out parents must be reclassified before "
                "subsequent validation."
            )
        return self


class ErrorComponent(ScientificContract):
    source: Literal[
        "numerical_convergence",
        "basis_physical_model",
        "vibrational_nuclear_motion",
        "surrogate",
        "experimental_fit",
    ]
    kind: Literal["unknown", "sensitivity", "standard_uncertainty"]
    value: StrictFloat | None = Field(default=None, ge=0)
    unit: Unit
    reason: Name | None = None
    evidence_sha256: Sha256 | None = None

    @model_validator(mode="after")
    def unknown_is_not_zero(self):
        if self.kind == "unknown":
            if self.value is not None or self.reason is None:
                raise ValueError("Unknown error components remain null with a reason.")
        elif (
            self.value is None
            or self.evidence_sha256 is None
            or self.reason is not None
        ):
            raise ValueError(
                "A numerical sensitivity/uncertainty requires its value "
                "and actual evidence identity."
            )
        if self.evidence_sha256 == "0" * 64:
            raise ValueError(
                "A zero digest does not identify actual uncertainty evidence."
            )
        return self


class ErrorBudget(ScientificContract):
    observable: Name
    unit: Unit
    components: tuple[ErrorComponent, ...] = Field(min_length=5, max_length=5)
    covariance: tuple[tuple[StrictFloat, ...], ...] | None = None
    covariance_order: tuple[Name, ...] = ()
    independence_assumption: Name | None = None
    calibration_id: Name | None = None
    qualification_status: Literal["uncalibrated"] = "uncalibrated"

    @model_validator(mode="after")
    def separate_sources(self):
        sources = tuple(component.source for component in self.components)
        if set(sources) != ERROR_SOURCES or any(
            component.unit != self.unit for component in self.components
        ):
            raise ValueError(
                "All five distinct error sources require the observable unit."
            )
        if self.covariance is not None:
            if (
                self.covariance_order != sources
                or self.independence_assumption is not None
            ):
                raise ValueError(
                    "Covariance requires exact component ordering, without "
                    "an independence shortcut."
                )
            if any(
                component.kind != "standard_uncertainty"
                for component in self.components
            ):
                raise ValueError(
                    "Sensitivity/unknown values are not covariance uncertainties."
                )
            matrix = checked_covariance(self.covariance, size=len(sources))
            values = np.array([component.value for component in self.components])
            with np.errstate(over="raise", invalid="raise"):
                try:
                    variances = values**2
                except FloatingPointError as exc:
                    raise ValueError(
                        "Declared uncertainty squares exceed finite numeric range."
                    ) from exc
            if np.any((values > 0) & (variances == 0)) or not np.allclose(
                np.diag(matrix), variances, rtol=1e-10, atol=0
            ):
                raise ValueError(
                    "Covariance diagonal must match declared standard "
                    "uncertainties squared."
                )
        elif self.covariance_order:
            raise ValueError("Covariance ordering without covariance is invalid.")
        if self.independence_assumption is not None and any(
            component.kind != "standard_uncertainty" for component in self.components
        ):
            raise ValueError(
                "Unknown/sensitivity terms cannot be summed as "
                "independent uncertainties."
            )
        return self


class ScientificGoal(ScientificContract):
    schema_version: Literal["cochem.torq.scientific-goal/1"] = (
        "cochem.torq.scientific-goal/1"
    )
    products: tuple[ProductDeclaration, ...] = Field(min_length=1, max_length=4)
    chemical_domain: Name
    allowed_approximations: tuple[Name, ...]
    target_observables: tuple[Name, ...] = Field(min_length=1)
    error_targets: tuple[ObservableTarget, ...] = ()
    unknown_achievable_accuracy_reason: Name | None = None
    spectral_range: SpectralRange | None = None
    budget_binding: Literal["request.resources"] = "request.resources"
    reference_policy: Literal[
        "de_novo_no_calibration", "experimental_inference", "independent_benchmark"
    ]
    anchors: tuple[ExperimentalAnchor, ...] = ()
    benchmark_split: BenchmarkSplit | None = None
    error_budgets: tuple[ErrorBudget, ...] = ()
    accuracy_status: Literal["unqualified"] = "unqualified"

    @model_validator(mode="after")
    def declared_scope(self):
        product_classes = [product.product_class for product in self.products]
        anchors = {anchor.anchor_id: anchor for anchor in self.anchors}
        if len(set(product_classes)) != len(product_classes) or len(anchors) != len(
            self.anchors
        ):
            raise ValueError(
                "Product classes and experimental anchor IDs must be unique."
            )
        if len(set(self.target_observables)) != len(self.target_observables) or len(
            set(self.allowed_approximations)
        ) != len(self.allowed_approximations):
            raise ValueError(
                "Observable and approximation declarations must be distinct."
            )
        declared = set(self.target_observables)
        if (
            set().union(*(set(product.observables) for product in self.products))
            != declared
        ):
            raise ValueError(
                "Goal observables and declared product observables must agree."
            )
        referenced_anchors = set()
        for product in self.products:
            if set(product.observables) - declared:
                raise ValueError("Every product observable must belong to the goal.")
            if set(product.anchor_ids) - anchors.keys():
                raise ValueError(
                    "An anchored product requires actual curated anchor declarations."
                )
            if product.product_class in {"B", "C"}:
                if any(
                    anchors[key].data_usage == "validation"
                    for key in product.anchor_ids
                ):
                    raise ValueError(
                        "Independent validation observations cannot anchor a "
                        "prediction."
                    )
            referenced_anchors.update(product.anchor_ids)
        if anchors.keys() - referenced_anchors:
            raise ValueError(
                "Unreferenced experimental anchors cannot silently affect a campaign."
            )
        if referenced_anchors and self.reference_policy == "de_novo_no_calibration":
            raise ValueError(
                "Experimental anchors contradict a no-calibration reference policy."
            )
        if self.reference_policy == "experimental_inference" and not referenced_anchors:
            raise ValueError(
                "Experimental inference requires identified experimental anchors."
            )
        if (
            self.reference_policy == "independent_benchmark"
            and "BENCHMARK" not in product_classes
        ):
            raise ValueError(
                "Independent benchmark policy requires the BENCHMARK "
                "product declaration."
            )
        if "BENCHMARK" in product_classes:
            if (
                self.reference_policy != "independent_benchmark"
                or self.benchmark_split is None
            ):
                raise ValueError(
                    "BENCHMARK requires an explicit independent reference "
                    "and family split policy."
                )
        elif self.benchmark_split is not None:
            raise ValueError(
                "A benchmark split belongs to a declared BENCHMARK product."
            )
        if not self.error_targets and self.unknown_achievable_accuracy_reason is None:
            raise ValueError(
                "Declare observable targets or explain unknown achievable accuracy."
            )
        if any(target.observable not in declared for target in self.error_targets):
            raise ValueError("Error targets must identify requested observables.")
        if len(
            {(target.observable, target.target_kind) for target in self.error_targets}
        ) != len(self.error_targets):
            raise ValueError(
                "An observable cannot have duplicate targets of the same kind."
            )
        if any(budget.observable not in declared for budget in self.error_budgets):
            raise ValueError("Error budgets must identify requested observables.")
        if len({budget.observable for budget in self.error_budgets}) != len(
            self.error_budgets
        ):
            raise ValueError("An observable cannot have competing error budgets.")
        if self.benchmark_split is not None:
            partitions = {
                parent: assignment.partition
                for assignment in self.benchmark_split.assignments
                for parent in assignment.parent_ids
            }
            source_partitions = {
                source: assignment.partition
                for assignment in self.benchmark_split.assignments
                for source in assignment.shared_reference_source_ids
            }
            parent_sources = {
                parent: assignment.shared_reference_source_ids
                for assignment in self.benchmark_split.assignments
                for parent in assignment.parent_ids
            }
            for anchor in self.anchors:
                if anchor.parent_id not in partitions:
                    raise ValueError(
                        "Every benchmark anchor parent must belong to the family split."
                    )
                usage = anchor.data_usage
                expected = "development" if usage == "inference" else usage
                if partitions[anchor.parent_id] != expected:
                    raise ValueError(
                        "Anchor use conflicts with the nonoverlapping family partition."
                    )
                if (
                    anchor.source.source_artifact_sha256
                    not in parent_sources[anchor.parent_id]
                ):
                    raise ValueError(
                        "Benchmark anchors must bind the source digest declared "
                        "in their family partition."
                    )
                source_partition = source_partitions.setdefault(
                    anchor.source.source_artifact_sha256, expected
                )
                if source_partition != expected:
                    raise ValueError(
                        "The same experimental source cannot provide independent"
                        " calibration and validation evidence."
                    )
        return self


def checked_covariance(values, *, size: int) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.shape != (size, size) or not np.isfinite(matrix).all():
        raise ValueError(
            "Covariance must be a finite square matrix matching parameter count."
        )
    tolerance = 64 * np.finfo(float).eps
    diagonal = np.diag(matrix)
    if np.any(diagonal < 0) or not np.allclose(
        matrix, matrix.T, rtol=tolerance, atol=0
    ):
        raise ValueError("Covariance must be symmetric and positive semidefinite.")
    zero_variance = diagonal == 0
    if np.any(matrix[zero_variance] != 0) or np.any(matrix[:, zero_variance] != 0):
        raise ValueError("Covariance cannot correlate a zero-variance parameter.")
    positive = diagonal > 0
    if np.any(positive):
        deviations = np.sqrt(diagonal[positive])
        with np.errstate(over="raise", invalid="raise"):
            try:
                correlation = (
                    matrix[np.ix_(positive, positive)]
                    / deviations[:, None]
                    / deviations[None, :]
                )
            except FloatingPointError as exc:
                raise ValueError(
                    "Covariance normalization exceeds finite numeric range."
                ) from exc
        if np.any(np.abs(correlation) > 1 + tolerance) or np.linalg.eigvalsh(
            correlation
        ).min(initial=0) < -tolerance * max(1, size):
            raise ValueError(
                "Covariance must be positive semidefinite in every parameter scale."
            )
    return matrix


def propagate_linear_uncertainty(
    jacobian, covariance, *, linearization_domain: str
) -> dict:
    """Apply J Sigma J^T; this mathematical propagation does not calibrate coverage."""
    if not linearization_domain.strip():
        raise ValueError("Uncertainty propagation requires its linearization domain.")
    vector = np.asarray(jacobian, dtype=float)
    if vector.ndim != 1 or not len(vector) or not np.isfinite(vector).all():
        raise ValueError("The observable Jacobian must be a nonempty finite vector.")
    matrix = checked_covariance(covariance, size=len(vector))
    with np.errstate(over="raise", invalid="raise"):
        try:
            variance = float(vector @ matrix @ vector)
            magnitude = float(np.abs(vector) @ np.abs(matrix) @ np.abs(vector))
        except FloatingPointError as exc:
            raise ValueError(
                "Covariance propagation exceeds finite numeric range."
            ) from exc
    tolerance = 64 * np.finfo(float).eps * magnitude
    if variance < -tolerance:
        raise ValueError("Propagated variance is negative beyond numerical roundoff.")
    roundoff_clipped = variance < 0
    variance = max(variance, 0.0)
    return {
        "variance": variance,
        "standard_uncertainty": float(np.sqrt(variance)),
        "linearization_domain": linearization_domain,
        "coverage_status": "uncalibrated",
        "propagation": "first_order_J_Sigma_JT",
        "numerical_roundoff_clipped": roundoff_clipped,
    }
