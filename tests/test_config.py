import pytest
from pydantic import ValidationError
from Libraries.torq_config import TorqRunParams

def test_valid_torq_run_params():
    params = TorqRunParams(
        tier="T3-1h",
        wall_time_tier="T3-1h",
        engine="MPQC",
        method="CCSD(T)-F12/CBS",
        basis_set="cc-pVTZ-F12",
        keywords=[
            "! CCSD(T)-F12",
            "cc-pVTZ-F12",
            "def2/J",
            "def2/C",
            "ExtremeSCF",
            "defgrid1",
            "Opt"
        ],
        anharmonicity="! VPT2",
        dispersion="D4",
        bsse_correction=None,
        cabs_mappings={
            "OptRI": "cc-pVTZ-F12-OptRI",
            "JKFIT": "cc-pVTZ-F12-JKFIT",
            "MP2FIT": "cc-pVTZ-F12-MP2FIT"
        }
    )
    assert params.tier == "T3-1h"

def test_prohibit_calc_hess_true():
    with pytest.raises(ValidationError, match="Calc_Hess true is strictly prohibited"):
        TorqRunParams(
            tier="T3-1h",
            wall_time_tier="T3-1h",
            engine="MPQC",
            method="CCSD(T)",
            basis_set="cc-pVTZ",
            keywords=["! Opt", "Calc_Hess true"]
        )

def test_valid_bsse_counterpoise():
    params = TorqRunParams(
        tier="T3-1h",
        wall_time_tier="T3-1h",
        engine="MPQC",
        method="CCSD(T)",
        basis_set="cc-pVTZ",
        keywords=["! Opt"],
        bsse_correction="Counterpoise"
    )
    assert params.bsse_correction == "Counterpoise"

def test_invalid_bsse_counterpoise_augmented():
    with pytest.raises(ValidationError, match="restricted to non-augmented"):
        TorqRunParams(
            tier="T3-1h",
            wall_time_tier="T3-1h",
            engine="MPQC",
            method="CCSD(T)",
            basis_set="aug-cc-pVTZ",
            keywords=["! Opt"],
            bsse_correction="CP"
        )

def test_invalid_bsse_counterpoise_not_triple_zeta():
    with pytest.raises(ValidationError, match="does not appear to be triple zeta"):
        TorqRunParams(
            tier="T3-1h",
            wall_time_tier="T3-1h",
            engine="MPQC",
            method="CCSD(T)",
            basis_set="cc-pVDZ",
            keywords=["! Opt"],
            bsse_correction="Counterpoise"
        )
