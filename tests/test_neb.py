import logging
logger = logging.getLogger(__name__)
import pytest
from Libraries.cochem_torq_neb import generate_neb_input, run_ts_optimization

def test_generate_neb_input_lindh() -> None:
    inp = generate_neb_input(
        job_name="test_neb",
        initial_coords=[["O", 0.0, 0.0, 0.0]],
        final_coords=[["O", 1.0, 1.0, 1.0]],
        n_images=8,
        inhess="Lindh"
    )
    assert "InHess Lindh" in inp
    assert "TolE 1e-7" in inp
    assert "TolRMSG 3e-6" in inp
    assert "TolMaxG 1e-5" in inp
    assert "TolRMSD 5e-5" in inp
    assert "TolMaxD 1e-4" in inp
    assert "InHess XTB2" not in inp

def test_generate_neb_input_xtb_matrix_import() -> None:
    inp = generate_neb_input(
        job_name="test_neb_xtb",
        initial_coords=[["O", 0.0, 0.0, 0.0]],
        final_coords=[["O", 1.0, 1.0, 1.0]],
        n_images=10,
        xtb_hessian_file="xtb_initial.hess"
    )
    assert 'InHess Name "xtb_initial.hess"' in inp
    assert "Nimages 10" in inp
    assert "InHess XTB2" not in inp
