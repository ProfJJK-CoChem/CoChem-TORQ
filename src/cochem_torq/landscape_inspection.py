"""Read-only authentic candidate and sampled-landscape evidence.

Display filtering cannot select candidates, compare different models, interpolate
missing nodes, infer stationary points, or create a thermodynamic ensemble.
"""

from __future__ import annotations

import html
import json
import sqlite3
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

import h5py
import numpy as np
from pydantic import Field, StrictBool, StrictFloat, model_validator
from typing_extensions import Self

from .artifacts import file_digest, verify_shard
from .candidate_ledger import CandidateLedger, CandidateRevisionConflictError
from .capabilities import _actual_file, _native_bundle
from .domain import (
    CalculationRequest,
    Contract,
    Molecule,
    StageResult,
    digest,
    read_json,
)
from .scan import ScanPointResult, ScanSurface, _verify_native
from .scientific_values import ElectronicEnergy
from .service import ApprovedPlan

EnergyStatus = Literal[
    "available", "failed", "not_computed", "unavailable", "changed_or_invalid"
]
Selection = Literal["retained", "excluded", "quarantined", "not_applicable"]


class LandscapeObservation(Contract):
    observation_id: str
    source_kind: Literal["candidate", "scan_node"]
    origin: str
    selection_state: Selection
    energy_status: EnergyStatus
    energy_hartree: StrictFloat | None
    energy_unit: Literal["hartree"] = "hartree"
    reason: str | None
    energy_recipe: str | None
    requested_recipe: str | None
    geometry_status: str
    coordinate_values: tuple[StrictFloat, ...] = ()
    comparability_sha256: str | None = None
    relative_energy_hartree: StrictFloat | None = None
    relative_reference_id: str | None = None
    uncertainty: dict[str, Any] = Field(
        default_factory=lambda: {
            "status": "uncalibrated",
            "reason": "No calibrated model error is established.",
        }
    )
    repeat_energy_spread_hartree: StrictFloat | None = None
    references: dict[str, Any] = Field(default_factory=dict)
    history: tuple[dict[str, Any], ...] = ()
    quality_flags: tuple[str, ...] = ()
    stationary_status: Literal["not_established_by_this_view"] = (
        "not_established_by_this_view"
    )

    @model_validator(mode="after")
    def truthful_energy(self) -> Self:
        if self.energy_status == "available":
            if (
                self.energy_hartree is None
                or self.reason is not None
                or self.comparability_sha256 is None
            ):
                raise ValueError("Available energies need authenticated evidence.")
        elif (
            self.energy_hartree is not None
            or self.relative_energy_hartree is not None
            or not self.reason
        ):
            raise ValueError("Missing energies remain null with a reason.")
        if (self.relative_energy_hartree is None) != (
            self.relative_reference_id is None
        ):
            raise ValueError("Relative energy requires its actual reference.")
        return self


class EnergyComparisonGroup(Contract):
    comparability_sha256: str
    definition: dict[str, Any]
    observation_ids: tuple[str, ...]
    reference_observation_id: str
    reference_energy_hartree: StrictFloat
    zero_definition: Literal["lowest_authenticated_observed_energy_in_exact_group"] = (
        "lowest_authenticated_observed_energy_in_exact_group"
    )
    cross_group_ranking_authorized: Literal[False] = False
    thermodynamic_population_established: Literal[False] = False


class LandscapeInspection(Contract):
    schema_version: Literal["cochem.torq.landscape-inspection/1"] = (
        "cochem.torq.landscape-inspection/1"
    )
    source_kind: Literal["candidate_ledger", "approved_scan"]
    source_reference: dict[str, Any]
    observations: tuple[LandscapeObservation, ...]
    comparison_groups: tuple[EnergyComparisonGroup, ...]
    coordinate_definitions: tuple[dict[str, Any], ...] = ()
    coverage: dict[str, Any]
    selection_snapshot: dict[str, Any] | None = None
    independent_accuracy_established: Literal[False] = False
    search_completeness_established: Literal[False] = False
    stationary_points_established: Literal[False] = False
    automatic_pruning_enabled: Literal[False] = False
    populations: None = None
    conformational_entropy: None = None

    @model_validator(mode="after")
    def bind_relative_observations(self) -> Self:
        records = {row.observation_id: row for row in self.observations}
        groups = {group.comparability_sha256: group for group in self.comparison_groups}
        if len(records) != len(self.observations) or len(groups) != len(
            self.comparison_groups
        ):
            raise ValueError("Inspection observation/group identities must be unique.")
        for group in self.comparison_groups:
            members = [
                row
                for row in self.observations
                if row.comparability_sha256 == group.comparability_sha256
                and row.energy_status == "available"
            ]
            if (
                digest(group.definition) != group.comparability_sha256
                or tuple(row.observation_id for row in members) != group.observation_ids
                or group.reference_observation_id not in group.observation_ids
            ):
                raise ValueError(
                    "Relative group differs from actual member definitions."
                )
            reference = records[group.reference_observation_id]
            energies = [
                row.energy_hartree for row in members if row.energy_hartree is not None
            ]
            if (
                not energies
                or reference.energy_hartree != min(energies)
                or group.reference_energy_hartree != reference.energy_hartree
            ):
                raise ValueError("Group zero must use its actual lowest observation.")
            for row in members:
                if row.energy_hartree is None or reference.energy_hartree is None:
                    raise ValueError("A relative comparison lost its measured energy.")
                if (
                    row.relative_reference_id != reference.observation_id
                    or row.relative_energy_hartree
                    != row.energy_hartree - reference.energy_hartree
                ):
                    raise ValueError(
                        "Relative values must bind the actual energy zero."
                    )
        if any(
            row.energy_status == "available" and row.comparability_sha256 not in groups
            for row in self.observations
        ):
            raise ValueError("An available energy has no exact comparison group.")
        return self


class LandscapeFilter(Contract):
    selection_states: tuple[Selection, ...] = (
        "retained",
        "excluded",
        "quarantined",
        "not_applicable",
    )
    available_energies_only: StrictBool = False
    maximum_relative_energy_hartree: StrictFloat | None = Field(default=None, ge=0)
    comparison_group_sha256: str | None = Field(default=None, pattern="^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def distinct_states(self) -> Self:
        if len(set(self.selection_states)) != len(self.selection_states):
            raise ValueError("Display selection states must be distinct.")
        return self


def _group_definition(
    molecule: dict[str, Any],
    *,
    recipe_sha256: str,
    method: dict[str, Any],
    numerical: dict[str, Any],
    native_identity: dict[str, Any],
    geometry_treatment: str,
    constraint_identity: str | None,
) -> dict[str, Any]:
    state = Molecule.model_validate(molecule).model_dump(mode="json")
    state.pop("geometry_bohr")
    return {
        "quantity": "total_electronic_energy",
        "unit": "hartree",
        "terms": "electronic_and_nuclear_energy_with_exact_declared_dispersion",
        "recipe_sha256": recipe_sha256,
        "method": method,
        "numerical": numerical,
        "native_implementation_and_basis": {
            key: native_identity[key]
            for key in (
                "engine",
                "engine_version",
                "engine_installation_sha256",
                "basis_definition_sha256",
                "adapter_source_sha256",
            )
        },
        "ordered_nuclear_state": state,
        "geometry_treatment": geometry_treatment,
        "constraint_identity": constraint_identity,
        "geometry_binding": "each energy at its own authenticated retained geometry",
        "electronic_root_assignment": "not_established_by_this_inspection",
        "excluded_terms": [
            "zero_point",
            "thermal",
            "entropy",
            "standard_state_correction",
        ],
    }


def _group_rows(
    rows: list[LandscapeObservation], definitions: dict[str, dict[str, Any]]
) -> tuple[tuple[LandscapeObservation, ...], tuple[EnergyComparisonGroup, ...]]:
    groups = []
    replacements: dict[str, LandscapeObservation] = {}
    for identity in sorted(definitions):
        members = [
            row
            for row in rows
            if row.comparability_sha256 == identity and row.energy_status == "available"
        ]
        if not members:
            continue

        def ordering(row: LandscapeObservation) -> tuple[float, str]:
            if row.energy_hartree is None:
                raise ValueError("An available observation lost its physical energy.")
            return row.energy_hartree, row.observation_id

        reference = min(members, key=ordering)
        assert reference.energy_hartree is not None
        groups.append(
            EnergyComparisonGroup(
                comparability_sha256=identity,
                definition=definitions[identity],
                observation_ids=tuple(row.observation_id for row in members),
                reference_observation_id=reference.observation_id,
                reference_energy_hartree=reference.energy_hartree,
            )
        )
        for row in members:
            assert row.energy_hartree is not None
            replacements[row.observation_id] = LandscapeObservation.model_validate(
                {
                    **row.model_dump(mode="json"),
                    "relative_energy_hartree": (
                        row.energy_hartree - reference.energy_hartree
                    ),
                    "relative_reference_id": reference.observation_id,
                }
            )
    return (
        tuple(replacements.get(row.observation_id, row) for row in rows),
        tuple(groups),
    )


def _point_native(
    point: ScanPointResult, native: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    record, identity = _verify_native(native)
    if identity != point.native_manifest_sha256:
        raise ValueError("Native manifest differs from the retained point.")
    expected = point.molecule.model_dump(mode="json")
    if any(
        record.get("molecule", {}).get(key) != value for key, value in expected.items()
    ):
        raise ValueError("Native geometry/state differs from the retained point.")
    if point.status != "available" or record.get("status") != "complete":
        raise ValueError("The retained point has no accepted physical energy.")
    electronic = ElectronicEnergy.model_validate(
        {
            "energy_hartree": record["energy_hartree"],
            "engine": record["engine"],
            "engine_version": record["engine_version"],
            "method": record["method"],
            "native_manifest_sha256": identity,
            "scf": record["scf"],
            "stability": record["stability"],
        }
    )
    if electronic.energy_hartree != point.energy_hartree or record.get(
        "gradient_hartree_bohr"
    ) != [list(row) for row in point.gradient_hartree_bohr or ()]:
        raise ValueError("Native energy/gradient differs from the retained point.")
    checkpoint = _actual_file(native / "wavefunction.chk")
    if (
        point.density is None
        or file_digest(checkpoint) != point.density.native_checkpoint_sha256
    ):
        raise ValueError("Point density lacks its actual saved orbital checkpoint.")
    with h5py.File(checkpoint, "r") as stream:
        coefficients = np.asarray(stream["scf/mo_coeff"])
        occupations = np.asarray(stream["scf/mo_occ"])
    if (
        coefficients.ndim != 2
        or occupations.shape != (coefficients.shape[1],)
        or np.iscomplexobj(coefficients)
        or not np.isfinite(coefficients).all()
        or not np.isfinite(occupations).all()
        or not np.all(np.isin(occupations, [0.0, 2.0]))
        or tuple(occupations.tolist()) != point.density.mo_occupations
        or int(occupations.sum()) != point.density.electron_count
        or not np.allclose(
            (coefficients * occupations) @ coefficients.T,
            point.density.matrix,
            atol=1e-12,
            rtol=0,
        )
    ):
        raise ValueError("Saved orbitals disagree with the retained density.")
    request = read_json(_actual_file(native / "request.json"))
    if (
        request["method"] != record["method"]
        or request["settings"] != record["settings"]
    ):
        raise ValueError("Native request and observed recipe differ.")
    if record.get("basis_definition_sha256") != file_digest(
        native / "basis-definition.json"
    ):
        raise ValueError("Actual basis definitions differ from the result.")
    if record.get("engine_installation_sha256") != read_json(
        native / "engine-installation.json"
    ).get("digest"):
        raise ValueError("Retained engine installation definition differs.")
    return record, {
        "native_manifest_path": str(native / "manifest.json"),
        "native_manifest_sha256": identity,
        "checkpoint_sha256": file_digest(checkpoint),
        "geometry_sha256": digest(expected),
    }


def _candidate_observation(
    record: dict[str, Any], history: list[dict[str, Any]]
) -> tuple[LandscapeObservation, dict[str, Any] | None]:
    fields: dict[str, Any] = {
        "observation_id": record["candidate_id"],
        "source_kind": "candidate",
        "origin": record["origin"],
        "selection_state": record["selection_state"],
        "energy_status": "not_computed",
        "energy_hartree": None,
        "reason": "Supplied candidate input has no calculated energy.",
        "energy_recipe": None,
        "requested_recipe": record["recipe"]["id"],
        "geometry_status": record["geometry_status"],
        "history": tuple(history),
        "uncertainty": {
            "status": record["quality"].get("uncertainty_status", "uncalibrated"),
            "reason": "No calibrated energy uncertainty is available in this record.",
        },
        "references": {
            "content_sha256": record["content_sha256"],
            "ledger_revision": record["revision"],
            "result_reference": record["result_reference"],
        },
    }
    definition = None
    try:
        origin = (
            record["request"].get("source_provenance", {}).get("adaptive_candidate")
        )
        reference = record["result_reference"]
        if reference is not None:
            root = Path(reference["directory"])
            manifest = verify_shard(root)
            if (
                file_digest(root / "manifest.json") != reference["manifest_sha256"]
                or manifest["files"] != reference["inventory"]
            ):
                raise ValueError("Result bytes differ from the candidate reference.")
            result = read_json(root / "result.json")
            stage = StageResult.model_validate(result["stages"]["electronic_structure"])
            fields["uncertainty"] = stage.uncertainty
            fields["references"]["result_manifest_sha256"] = reference[
                "manifest_sha256"
            ]
            if stage.status != "available":
                fields.update(
                    energy_status="failed"
                    if stage.status == "failed"
                    else "unavailable",
                    reason=stage.reason,
                )
                return LandscapeObservation.model_validate(fields), None
            assert stage.value is not None
            electronic = ElectronicEnergy.model_validate(stage.value)
            _, native = _native_bundle(root / "engine")
            if (
                file_digest(root / "engine" / "manifest.json")
                != electronic.native_manifest_sha256
                or native["energy_hartree"] != electronic.energy_hartree
                or native["method"] != electronic.method
                or any(
                    native["molecule"].get(key) != value
                    for key, value in record["molecule"].items()
                )
            ):
                raise ValueError("Energy is not bound to its own native geometry.")
            definition = _group_definition(
                record["molecule"],
                recipe_sha256=result["recipe"]["recipe_sha256"],
                method=native["method"],
                numerical=native["settings"],
                native_identity=native,
                geometry_treatment=record["geometry_status"],
                constraint_identity=None,
            )
            fields.update(
                energy_status="available",
                energy_hartree=electronic.energy_hartree,
                reason=None,
                energy_recipe=result["recipe"]["id"],
                comparability_sha256=digest(definition),
            )
            fields["references"].update(
                native_manifest_sha256=electronic.native_manifest_sha256,
                geometry_sha256=digest(record["molecule"]),
            )
        elif origin is not None:
            point_path = _actual_file(Path(origin["point_result_path"]))
            if file_digest(point_path) != origin["point_result_sha256"]:
                raise ValueError("The candidate's source scan point bytes changed.")
            point = ScanPointResult.model_validate(read_json(point_path))
            native_path = Path(origin["native_manifest_path"])
            if (
                point.point_id != origin["point_id"]
                or point.native_manifest_sha256 != origin["native_manifest_sha256"]
                or point.molecule.model_dump(mode="json") != record["molecule"]
            ):
                raise ValueError("Candidate differs from its source scan observation.")
            native, references = _point_native(point, native_path.parent)
            definition = _group_definition(
                point.molecule.model_dump(mode="json"),
                recipe_sha256=point.recipe_sha256,
                method=native["method"],
                numerical=native["settings"],
                native_identity=native,
                geometry_treatment="fixed_nonstationary_sample",
                constraint_identity=point.scan_sha256,
            )
            fields.update(
                origin="sampled_candidate",
                energy_status="available",
                energy_hartree=point.energy_hartree,
                reason=None,
                energy_recipe=point.recipe,
                geometry_status="fixed_nonstationary_sample",
                comparability_sha256=digest(definition),
                coordinate_values=point.coordinate_values,
                uncertainty={
                    "status": "uncalibrated",
                    "reason": "No calibrated scan energy model-error budget.",
                },
                quality_flags=(
                    "sampled_candidate_requires_separate_stationarity_verification",
                ),
            )
            fields["references"].update(references)
            fields["references"]["point_result_sha256"] = origin["point_result_sha256"]
    except (OSError, KeyError, TypeError, ValueError) as exc:
        fields.update(
            energy_status="changed_or_invalid",
            reason=str(exc),
            energy_hartree=None,
            comparability_sha256=None,
        )
        definition = None
    return LandscapeObservation.model_validate(fields), definition


def inspect_candidate_ledger(path: str | Path) -> LandscapeInspection:
    """Observe a real ledger and raw references without changing selections."""
    source = Path(path).absolute()
    rows: list[LandscapeObservation] = []
    definitions: dict[str, dict[str, Any]] = {}
    snapshot = None
    if source.exists() or source.is_symlink():
        _actual_file(source)
        # A read-only viewer must not initialize or migrate an unrelated file.
        uri = "file:" + quote(source.as_posix(), safe="/") + "?mode=ro"
        connection = sqlite3.connect(uri, uri=True)
        try:
            if connection.execute("PRAGMA user_version").fetchone()[0] != 1:
                raise ValueError("Inspection requires an existing candidate ledger.")
        finally:
            connection.close()
        with CandidateLedger(source) as ledger:
            before = ledger.selection_snapshot()
            for record in ledger.candidates(include_excluded=True):
                row, definition = _candidate_observation(
                    record, ledger.history(record["candidate_id"])
                )
                rows.append(row)
                if definition is not None:
                    definitions[digest(definition)] = definition
            snapshot = ledger.selection_snapshot()
            if before != snapshot:
                raise CandidateRevisionConflictError(
                    "Selection changed during inspection; refresh the view."
                )
    observations, groups = _group_rows(rows, definitions)
    return LandscapeInspection(
        source_kind="candidate_ledger",
        source_reference={
            "path": str(source),
            "status": "observed" if snapshot is not None else "absent",
        },
        observations=observations,
        comparison_groups=groups,
        selection_snapshot=snapshot,
        coverage={
            "available_energy_candidates": sum(
                row.energy_status == "available" for row in rows
            ),
            "recorded_candidates": len(rows),
            "candidate_search_coverage": None,
            "reason": "A selection ledger cannot establish conformer/search coverage.",
        },
    )


def inspect_scan_landscape(
    directory: str | Path, *, expected_result_sha256: str
) -> LandscapeInspection:
    """Authenticate retained scan evidence, preserving absent approved nodes."""
    root = Path(directory).absolute()
    result_path = _actual_file(root / "result.json")
    if file_digest(result_path) != expected_result_sha256:
        raise ValueError("Scan bytes differ from the independently supplied reference.")
    surface = ScanSurface.model_validate(read_json(result_path))
    request = CalculationRequest.model_validate(
        read_json(_actual_file(root / "request.json"))
    )
    approved = ApprovedPlan.model_validate(
        read_json(_actual_file(root / "approved-plan.json"))
    )
    if (
        approved.plan["request"] != request.model_dump(mode="json")
        or surface.request_sha256 != approved.plan["request_sha256"]
        or surface.recipe != request.recipe
        or surface.recipe_sha256 != approved.plan["recipe_sha256"]
        or surface.scan_sha256 != approved.plan["scan_sha256"]
        or surface.scan_sha256 != digest(surface.scan.model_dump(mode="json"))
        or surface.source_identity_sha256 != digest(approved.source_identity)
        or len(surface.nodes) != len(surface.scan.grid)
    ):
        raise ValueError("Retained scan/approval/request identities differ.")
    files = {}
    for point_path in sorted((root / "points").glob("*/point-result.json")):
        point = ScanPointResult.model_validate(read_json(_actual_file(point_path)))
        if point.point_id in files:
            raise ValueError("Duplicate retained point identities.")
        files[point.point_id] = (point, point_path)
    if set(files) != {point.point_id for point in surface.points}:
        raise ValueError("Point files differ from the retained surface inventory.")
    authenticated: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for point in surface.points:
        saved, point_path = files[point.point_id]
        if (
            saved != point
            or point.scan_sha256 != surface.scan_sha256
            or point.recipe_sha256 != surface.recipe_sha256
            or point.request_sha256 != surface.request_sha256
            or point.source_identity_sha256 != surface.source_identity_sha256
            or point.sample_index >= len(surface.scan.grid)
            or point.coordinate_values != surface.scan.grid[point.sample_index]
        ):
            raise ValueError("Point differs from its retained surface/grid identity.")
        if point.status == "available":
            if point.native_manifest_path is None:
                raise ValueError("An available point needs authentic native evidence.")
            native_path = root / point.native_manifest_path
            if (
                not native_path.absolute().is_relative_to(root)
                or ".." in native_path.parts
                or native_path.parent.parent != point_path.parent
            ):
                raise ValueError("Native evidence must remain in its point bundle.")
            authenticated[point.point_id] = _point_native(point, native_path.parent)
    rows = []
    definitions = {}
    for index, node in enumerate(surface.nodes):
        points = [point for point in surface.points if point.sample_index == index]
        available = [point for point in points if point.status == "available"]
        representative = available[0] if available else None
        expected_status = (
            "available" if available else "failed" if points else "not_evaluated"
        )
        if (
            node.sample_index != index
            or node.coordinate_values != surface.scan.grid[index]
            or node.status != expected_status
            or node.observation_ids != tuple(point.point_id for point in points)
            or node.failed_observation_ids
            != tuple(point.point_id for point in points if point.status != "available")
            or node.representative_point_id
            != (representative.point_id if representative else None)
            or node.energy_hartree
            != (representative.energy_hartree if representative else None)
        ):
            raise ValueError(
                "Node values differ from the actual representative policy."
            )
        fields: dict[str, Any] = {
            "observation_id": f"sample-{index}",
            "source_kind": "scan_node",
            "origin": "approved_finite_grid",
            "selection_state": "not_applicable",
            "energy_status": (
                "available"
                if representative
                else "failed"
                if points
                else "not_computed"
            ),
            "energy_hartree": node.energy_hartree,
            "reason": (
                None
                if representative
                else "; ".join(
                    point.reason or "Actual scan point unavailable." for point in points
                )
                if points
                else "Approved grid node has not been physically evaluated."
            ),
            "energy_recipe": surface.recipe if representative else None,
            "requested_recipe": surface.recipe,
            "geometry_status": "fixed_nonstationary_sample",
            "coordinate_values": node.coordinate_values,
            "references": {
                "sample_index": index,
                "observation_ids": node.observation_ids,
                "failed_observation_ids": node.failed_observation_ids,
                "retained_point_references": [
                    {
                        "point_id": point.point_id,
                        "point_result_path": str(files[point.point_id][1]),
                        "point_result_sha256": file_digest(files[point.point_id][1]),
                        "native_manifest_path": point.native_manifest_path,
                        "native_manifest_sha256": point.native_manifest_sha256,
                        "status": point.status,
                        "purpose": point.purpose,
                        "parent_point_ids": point.parent_point_ids,
                        "reason": point.reason,
                    }
                    for point in points
                ],
            },
            "quality_flags": ("finite_sampling_no_stationarity_or_completeness_proof",),
        }
        if representative is not None:
            native, references = authenticated[representative.point_id]
            definition = _group_definition(
                representative.molecule.model_dump(mode="json"),
                recipe_sha256=surface.recipe_sha256,
                method=native["method"],
                numerical=native["settings"],
                native_identity=native,
                geometry_treatment="fixed_nonstationary_sample",
                constraint_identity=surface.scan_sha256,
            )
            identity = digest(definition)
            definitions[identity] = definition
            fields["comparability_sha256"] = identity
            fields["references"].update(references)
            if len(available) >= 2:
                energies = [
                    point.energy_hartree
                    for point in available
                    if point.energy_hartree is not None
                ]
                fields["repeat_energy_spread_hartree"] = max(energies) - min(energies)
            if any(
                item.inconsistent and item.sample_index == index
                for item in surface.comparisons
            ):
                fields["quality_flags"] += (
                    "retained_independent_rechecks_inconsistent",
                )
        rows.append(LandscapeObservation.model_validate(fields))
    observations, groups = _group_rows(rows, definitions)
    available_count = sum(node.status == "available" for node in surface.nodes)
    return LandscapeInspection(
        source_kind="approved_scan",
        source_reference={
            "directory": str(root),
            "result_sha256": expected_result_sha256,
            "approved_plan_sha256": approved.approval.plan_sha256,
            "scan_sha256": surface.scan_sha256,
            "request_sha256": surface.request_sha256,
            "historical_inspection_only": True,
        },
        observations=observations,
        comparison_groups=groups,
        coordinate_definitions=tuple(
            item.model_dump(mode="json") for item in surface.scan.coordinates
        ),
        coverage={
            "approved_grid_nodes": len(surface.nodes),
            "attempted_grid_nodes": sum(
                bool(node.observation_ids) for node in surface.nodes
            ),
            "available_grid_nodes": available_count,
            "failed_grid_nodes": sum(node.status == "failed" for node in surface.nodes),
            "uncomputed_grid_nodes": sum(
                node.status == "not_evaluated" for node in surface.nodes
            ),
            "observed_grid_fraction": available_count / len(surface.nodes),
            "physical_call_count": len(surface.points),
            "continuous_domain_coverage": None,
            "reason": "Authenticated finite-grid fraction; no basin completeness.",
        },
    )


def filter_landscape(
    inspection: LandscapeInspection, filters: LandscapeFilter | None = None
) -> tuple[LandscapeObservation, ...]:
    """Reversible display-only filters preserve original zeros and history."""
    selected = filters or LandscapeFilter()
    if (
        selected.comparison_group_sha256 is not None
        and selected.comparison_group_sha256
        not in {group.comparability_sha256 for group in inspection.comparison_groups}
    ):
        raise ValueError("Choose an actual observed comparison group.")
    return tuple(
        row
        for row in inspection.observations
        if row.selection_state in selected.selection_states
        and (not selected.available_energies_only or row.energy_status == "available")
        and (
            selected.comparison_group_sha256 is None
            or row.comparability_sha256 == selected.comparison_group_sha256
        )
        and (
            selected.maximum_relative_energy_hartree is None
            or row.relative_energy_hartree is None
            or row.relative_energy_hartree <= selected.maximum_relative_energy_hartree
        )
    )


def _point_plot(
    inspection: LandscapeInspection, rows: tuple[LandscapeObservation, ...]
) -> str:
    """Plot actual markers separately by group, without interpolation."""
    plots = []
    for group in inspection.comparison_groups:
        members = [
            row
            for row in rows
            if row.comparability_sha256 == group.comparability_sha256
            and row.relative_energy_hartree is not None
        ]
        if not members:
            continue
        coordinate_axis = bool(inspection.coordinate_definitions)
        x = [
            row.coordinate_values[0] if coordinate_axis else float(index)
            for index, row in enumerate(members)
        ]
        y = [
            float(row.relative_energy_hartree)
            for row in members
            if row.relative_energy_hartree is not None
        ]
        low, high, maximum = min(x), max(x), max(y)
        points = []
        for row, xx, yy in zip(members, x, y):
            px = 35 + 430 * ((xx - low) / (high - low) if high > low else 0.5)
            py = 165 - 130 * (yy / maximum if maximum > 0 else 0)
            title = html.escape(
                f"{row.observation_id}: coordinates={row.coordinate_values}; "
                f"absolute E={row.energy_hartree} hartree; relative E={yy} "
                f"hartree; reference={row.relative_reference_id}"
            )
            points.append(
                f'<circle cx="{px:.6f}" cy="{py:.6f}" r="5" fill="#2463aa">'
                f"<title>{title}</title></circle>"
            )
        xlabel = (
            f"{inspection.coordinate_definitions[0]['coordinate_id']} "
            f"({inspection.coordinate_definitions[0]['unit']})"
            if coordinate_axis
            else "Observation index within this group"
        )
        plots.append(
            "<figure><figcaption>Exact energy group "
            + html.escape(group.comparability_sha256[:12])
            + "; zero = "
            + html.escape(group.reference_observation_id)
            + " across all recorded states. Separate groups cannot be ranked."
            "</figcaption>"
            '<svg viewBox="0 0 500 225" role="img" '
            'aria-label="Actual energy markers; no interpolated surface">'
            '<path d="M35 30 V165 H470" fill="none" stroke="#555"/>'
            + "".join(points)
            + '<text x="35" y="190" font-size="12">'
            + html.escape(xlabel)
            + '</text><text x="35" y="215" font-size="12">'
            "Relative total electronic energy (hartree); actual markers only"
            "</text></svg></figure>"
        )
    return "".join(plots) or (
        "<p>No authenticated energy markers are available in this display.</p>"
    )


def render_landscape_html(
    inspection: LandscapeInspection, filters: LandscapeFilter | None = None
) -> str:
    """Deterministic escaped table and actual-marker notebook projection."""
    rows = filter_landscape(inspection, filters)
    topology = ", ".join(
        f"{coordinate['coordinate_id']}: {coordinate['kind']} "
        f"[{coordinate['unit']}], "
        + (
            f"periodic, period={coordinate['domain']['period']}"
            if coordinate["domain"]["periodic"]
            else "nonperiodic"
        )
        for coordinate in inspection.coordinate_definitions
    )
    table = []
    for row in rows:
        values = [
            row.observation_id,
            row.selection_state,
            row.energy_status,
            row.coordinate_values,
            row.energy_recipe,
            row.requested_recipe,
            row.energy_hartree,
            row.relative_energy_hartree,
            row.relative_reference_id,
            row.geometry_status,
            row.uncertainty,
            row.repeat_energy_spread_hartree,
            row.reason,
            row.quality_flags,
        ]
        cells = "".join(
            "<td>"
            + html.escape("unavailable" if value is None else str(value))
            + "</td>"
            for value in values
        )
        detail = html.escape(
            json.dumps(
                {"references": row.references, "history": row.history},
                sort_keys=True,
                allow_nan=False,
            )
        )
        table.append(
            "<tr>" + cells + "<td><details><summary>"
            "Raw references and retained history</summary><pre>"
            + detail
            + "</pre></details></td></tr>"
        )
    headings = (
        "Observation",
        "Selection",
        "Energy status",
        "Coordinates",
        "Observed energy recipe",
        "Requested target recipe",
        "Absolute E (hartree)",
        "Relative E (hartree)",
        "Actual zero reference",
        "Geometry status",
        "Uncertainty",
        "Observed repeat spread (hartree)",
        "Absence reason",
        "Quality flags",
        "Evidence and history",
    )
    return (
        "<p>Read-only observed ensemble/landscape: "
        + str(len(rows))
        + " of "
        + str(len(inspection.observations))
        + " records visible. Filters "
        "preserve all records, zero references and selection history. "
        "No populations, entropy, automatic pruning, stationarity or search "
        "completeness are inferred.</p><p>Coverage: "
        + html.escape(json.dumps(inspection.coverage, sort_keys=True, allow_nan=False))
        + "</p><p>Coordinate topology: "
        + html.escape(topology or "No declared scan coordinates.")
        + " Only the first coordinate is projected; all coordinates remain "
        "in the table.</p>"
        + _point_plot(inspection, rows)
        + "<table><thead><tr>"
        + "".join("<th>" + html.escape(label) + "</th>" for label in headings)
        + "</tr></thead><tbody>"
        + "".join(table)
        + "</tbody></table>"
    )
