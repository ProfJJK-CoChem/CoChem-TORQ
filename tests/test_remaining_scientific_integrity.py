"""Real serialization/database checks and explicitly abstract statistical math.

The residual arrays below test mathematics only. They are not molecular data,
quantum-engine outputs, or evidence of calibrated molecular coverage.
"""

import datetime
import hashlib
import json

import pyarrow.parquet as pq
import pytest
import torch

from Libraries.cochem_torq_adaptive_error import EscalationManifest, export_manifest_pyarrow
from Libraries.cochem_torq_conformal import CalibrationSample, ConformalPredictor
from Libraries.cochem_torq_goat import ConformerRecord, deduplicate_conformers, write_xyz_string
from Libraries.cochem_torq_inference_schemas import ConformalPredictorConfig
from cochem.topos.clash import GeometricClashDetector
from cochem.topos.exceptions import StericClashError


def test_empty_manifest_exports_no_invented_physical_rows(tmp_path):
    manifest = EscalationManifest(
        manifest_id="empty-selection", total_anchors_evaluated=0,
        escalated_anchors_count=0,
        provenance_hash=hashlib.sha256(b"").hexdigest(),
        created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
    destination = export_manifest_pyarrow(manifest, tmp_path / "empty-selection.parquet")
    table = pq.read_table(destination)
    assert table.num_rows == 0
    assert "energy_low" in table.column_names
    assert table.schema.metadata[b"energy_units"] == b"kcal/mol"
    assert json.loads(table.schema.metadata[b"torq_manifest"]) == manifest.model_dump()


def abstract_calibration_data(sample_count, vector_count=1):
    """Abstract increasing nonconformity scores; no physical interpretation."""
    return [
        CalibrationSample(
            energy_true=float(index + 1), energy_pred=0.0, energy_sigma=1.0,
            forces_true=torch.full((vector_count, 3), float(index + 1), dtype=torch.float64),
            forces_pred=torch.zeros((vector_count, 3), dtype=torch.float64),
            forces_sigma=torch.ones((vector_count, 3), dtype=torch.float64),
        )
        for index in range(sample_count)
    ]


def test_force_only_interval_has_no_invented_energy_bounds():
    predictor = ConformalPredictor(ConformalPredictorConfig(alpha=0.1))
    predictor.calibrate(abstract_calibration_data(20))
    interval = predictor.predict_forces(torch.zeros((1, 3)), torch.ones((1, 3)))
    assert interval.energy_lower is None
    assert interval.energy_upper is None
    assert interval.force_lower.shape == (1, 3)
    assert torch.all(interval.force_lower <= interval.force_upper)


def test_quantile_beyond_available_independent_energy_scores_is_infinite():
    predictor = ConformalPredictor()
    predictor.calibrate(abstract_calibration_data(sample_count=1, vector_count=20))
    # Twenty force vectors cannot manufacture twenty independent energy scores.
    assert predictor.q_hat_energy == float("inf")


def test_unknown_conformer_values_remain_absent_and_cannot_justify_deletion():
    # An explicit input geometry only; no optimization or quantum result claimed.
    first = ConformerRecord(index=0, symbols=["O", "H", "H"],
                            coordinates=[[0.0, 0.0, 0.0], [0.0, 0.75, 0.58], [0.0, -0.75, 0.58]])
    second = first.model_copy(update={"index": 1})
    assert first.energy_kcal_rel is None
    assert first.rotational_constants_mhz is None
    assert first.inertial_defect_u_a2 is None
    assert "energy=unavailable" in write_xyz_string([first])
    assert "rel_kcal=0" not in write_xyz_string([first])
    preserved = deduplicate_conformers([first, second], ethr_kcal=0.1)
    assert len(preserved) == 2
    assert preserved[0] is first
    assert preserved[1] is second


def test_missing_real_database_vdw_radius_is_not_scaled_from_covalent_radius():
    detector = GeometricClashDetector()
    # The installed Mendeleev database has covalent Rf but no vdW Rf radius.
    with pytest.raises(StericClashError, match="No database Van der Waals radius"):
        detector.get_vdw_radius("Rf")
    assert detector.get_vdw_radius("C") > 0
