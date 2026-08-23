import hashlib
import json
import logging
import math
from pathlib import Path
import numpy as np
import pytest

from Libraries.cochem_spcat_bridge import (
    BOLTZMANN_CONSTANT_JK,
    C_ROT,
    CONSTANTS,
    HC_OVER_KB,
    KB_OVER_H,
    PLANCK_CONSTANT_JS,
    SPEED_OF_LIGHT_CMS,
    LAMTriggerError,
    FortranOverflowError,
    TorqSpcatBridge,
    apply_symmetry_divisors,
    calculate_rotational_partition_function,
    calculate_vibrational_partition_function,
    compute_coupled_partition_functions,
    fortran_double_precision_formatter,
    fortran_overflow_guard,
    generate_spcat_int,
    generate_spcat_var,
    low_frequency_lam_trap,
    low_frequency_trap,
    vibrational_partition_coupling,
)

logger = logging.getLogger(__name__)

# Real experimental / ab initio Cartesian geometry for Water (H2O in Angstroms)
H2O_GEOMETRY = np.array([
    [0.000000,  0.000000,  0.117300],
    [0.000000,  0.757200, -0.469200],
    [0.000000, -0.757200, -0.469200],
], dtype=np.float64)
H2O_SYMBOLS = ["O", "H", "H"]


def test_torq_spcat_bridge_init(tmp_path: Path) -> None:
    tensor_file = tmp_path / "tensor.h5"
    tensor_file.touch()
    
    mpqc_file = tmp_path / "mpqc.out"
    mpqc_file.touch()
    
    bridge = TorqSpcatBridge(str(tensor_file), str(mpqc_file), temperature_k=298.15)
    assert bridge.temperature_k == 298.15
    assert bridge.mpqc_file == Path(mpqc_file)


def test_torq_spcat_bridge_extract_orca(tmp_path: Path) -> None:
    tensor_file = tmp_path / "tensor.h5"
    tensor_file.touch()
    mpqc_file = tmp_path / "mpqc.out"
    mpqc_file.write_text("FINAL SINGLE POINT ENERGY -76.123\n", encoding="utf-8")
    
    bridge = TorqSpcatBridge(str(tensor_file), str(mpqc_file), temperature_k=298.15)
    q_rot, q_vib, q_total = bridge.calculate_partition_functions()
    assert q_rot > 0.0
    assert q_vib >= 1.0


def test_exact_codata_2022_constants() -> None:
    """Validate immutable CODATA 2022 physical constants."""
    assert CONSTANTS.H == 6.62607015e-34
    assert CONSTANTS.K_B == 1.380649e-23
    assert CONSTANTS.C_CM_S == 29979245800.0
    assert abs(CONSTANTS.C_ROT - 505379.008435) < 1e-4
    assert abs(CONSTANTS.HC_OVER_KB - 1.4387768775) < 1e-6


def test_low_frequency_lam_trap_enforcement() -> None:
    """Verify LAM trap raises LAMTriggerError for modes < 50 cm^-1 and passes stiff modes."""
    with pytest.raises(LAMTriggerError) as exc_info:
        low_frequency_lam_trap([3100.0, 1500.0, 105.0, 24.5])
    assert 24.5 in exc_info.value.details["flagged_frequencies"]
    assert "DVR" in exc_info.value.message

    # Alias check
    assert low_frequency_trap is low_frequency_lam_trap
    stiff = low_frequency_trap([1500.0, 3600.0])
    assert stiff == [1500.0, 3600.0]


def test_apply_symmetry_divisors_water() -> None:
    """Verify symmetry and spin weights for water (C2v, sigma=2, '3 1')."""
    res = apply_symmetry_divisors(H2O_GEOMETRY, H2O_SYMBOLS)
    assert res.point_group == "C2v"
    assert res.sigma == 2
    assert res.spin_weight_ratio_str == "3 1"
    assert res.effective_divisor == 2.0


def test_vibrational_partition_coupling_with_lam_drop() -> None:
    """Verify vibrational partition coupling drops LAM frequency across temperature gradient."""
    temps = [2.0, 10.0, 50.0, 298.15]
    all_freqs = [3100.0, 1500.0, 105.0, 24.5]
    lam_mode = 24.5
    q_rot_dvr = {2.0: 1.05, 10.0: 4.8, 50.0: 35.2, 298.15: 185.0}

    q_coupled = vibrational_partition_coupling(
        q_rot_dvr=q_rot_dvr,
        q_vib_orca=all_freqs,
        temp_array=temps,
        lam_frequency=lam_mode,
    )

    q_vib_without_lam = calculate_vibrational_partition_function(all_freqs, 298.15, exclude_frequencies=[lam_mode])
    expected = q_rot_dvr[298.15] * q_vib_without_lam
    assert math.isclose(q_coupled[298.15], expected, rel_tol=1e-6)


def test_fortran_overflow_guard_and_formatter() -> None:
    """Verify Fortran Double Precision overflow protection and 'D' format."""
    with pytest.raises(FortranOverflowError):
        fortran_overflow_guard({"DJ": 1.5e310})

    formatted = fortran_double_precision_formatter(20000, 1.567e-5, uncertainty=1e-7, label="DJ")
    assert "20000" in formatted
    assert "D-05" in formatted
    assert "/ DJ" in formatted


def test_generate_spcat_var_and_int(tmp_path: Path) -> None:
    """Verify generation of .var and .int files."""
    var_file = tmp_path / "test.var"
    var_content = generate_spcat_var("H2O", {"A": 825360.0, "B": 435360.0, "C": 278130.0}, filepath=var_file)
    assert var_file.exists()
    assert "H2O Ground State" in var_content

    int_file = tmp_path / "test_{T}K.int"
    int_dict = generate_spcat_int("H2O", {"mu_a": 0.0, "mu_b": 1.85, "mu_c": 0.0}, temperatures=[298.15], filepath_template=int_file)
    assert 298.15 in int_dict
    assert (tmp_path / "test_298.1K.int").exists()

