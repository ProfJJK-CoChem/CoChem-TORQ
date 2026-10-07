"""Actual model metadata resolution; no execution or model outputs are simulated."""
import pytest

from Libraries.cochem_torq_matrix_loader import (
    MLFF_CATALOG, UnsupportedElementError, parse_execution_cascade, resolve_mlff_model,
)


def test_model_is_never_selected_implicitly():
    with pytest.raises(ValueError, match="no ML model is selected implicitly"):
        resolve_mlff_model(["C", "H"])


def test_unsupported_requested_model_does_not_substitute_another_model():
    with pytest.raises(UnsupportedElementError, match="no alternative"):
        resolve_mlff_model(["Fe"], requested_model="MACE-OFF23")
    with pytest.raises(ValueError, match="Unknown or unsupported"):
        resolve_mlff_model(["C"], requested_model="absent-model")


def test_old_empirical_and_all_element_fallbacks_are_not_models():
    assert "EMPIRICAL_COVALENT" not in MLFF_CATALOG
    assert "PySCF_RHF" not in MLFF_CATALOG
    with pytest.raises(ValueError, match="substitution is disabled"):
        resolve_mlff_model(["C"], fallback_chain=["MACE-OFF23", "AIMNet2"])


def test_explicit_model_is_only_advisory_candidate_metadata():
    model, evidence = resolve_mlff_model(["C", "H"], requested_model="AIMNet2")
    assert model.name == "AIMNet2"
    assert "advisory, unvalidated" in evidence[0]


def test_cascade_does_not_add_unrequested_ml_screening():
    cascade = parse_execution_cascade(["O", "H", "H"], tier="T1-30min")
    assert cascade.selected_mlff is None
    assert all("ML" not in stage["stage_name"] for stage in cascade.stages)


def test_requested_cascade_advisory_cannot_prune_or_certify_convergence():
    cascade = parse_execution_cascade(["O", "H", "H"], tier="T1-30min", requested_mlff="AIMNet2")
    advisory = cascade.stages[0]
    assert advisory["stage_name"] == "ML_Advisory"
    assert advisory["advisory_only"] is True
    assert advisory["capability_status"] == "unvalidated"
    assert "convergence_tol_max_g" not in advisory


def test_unknown_polar_model_domain_does_not_claim_periodic_table_coverage():
    assert not MLFF_CATALOG["MACE-POLAR-1"].supported_z
    with pytest.raises(UnsupportedElementError, match="manifest"):
        resolve_mlff_model(["H"], requested_model="MACE-POLAR-1")
