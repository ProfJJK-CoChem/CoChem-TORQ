import json
import logging
import math
from pathlib import Path

import numpy as np
import pytest

from Libraries.cochem_spcat_bridge import (
    CONSTANTS,
    FortranOverflowError,
    LAMTriggerError,
    SPCATBridgeError,
    ThreeTierRoutingResult,
    TorqSpcatBridge,
    apply_symmetry_divisors,
    build_complete_spcat_payload,
    calculate_rotational_partition_function,
    calculate_vibrational_partition_function,
    compute_coupled_partition_functions,
    compute_sha256,
    format_fortran_double,
    fortran_double_precision_formatter,
    fortran_overflow_guard,
    generate_spcat_int,
    generate_spcat_provenance_manifest,
    generate_spcat_var,
    low_frequency_lam_trap,
    low_frequency_trap,
    route_3tier_abinitio_payload,
    validate_airgap_boundary,
    vibrational_partition_coupling,
)

logger = logging.getLogger(__name__)

# Real experimental / ab initio Cartesian geometry for Water (H2O in Angstroms)
H2O_GEOMETRY = np.array(
    [
        [0.000000, 0.000000, 0.117300],
        [0.000000, 0.757200, -0.469200],
        [0.000000, -0.757200, -0.469200],
    ],
    dtype=np.float64,
)
H2O_SYMBOLS = ["O", "H", "H"]

# Real geometry for Ammonia (NH3 in Angstroms)
NH3_GEOMETRY = np.array(
    [
        [0.000000, 0.000000, 0.116489],
        [0.000000, 0.939731, -0.271808],
        [0.813831, -0.469865, -0.271808],
        [-0.813831, -0.469865, -0.271808],
    ],
    dtype=np.float64,
)
NH3_SYMBOLS = ["N", "H", "H", "H"]

# Real geometry for Ethylene (C2H4 in Angstroms)
C2H4_GEOMETRY = np.array(
    [
        [0.000000, 0.000000, 0.669500],
        [0.000000, 0.000000, -0.669500],
        [0.000000, 0.928900, 1.232100],
        [0.000000, -0.928900, 1.232100],
        [0.000000, 0.928900, -1.232100],
        [0.000000, -0.928900, -1.232100],
    ],
    dtype=np.float64,
)
C2H4_SYMBOLS = ["C", "C", "H", "H", "H", "H"]


def test_torq_spcat_bridge_init(tmp_path: Path) -> None:
    tensor_file = tmp_path / "tensor.json"
    tensor_file.write_text(
        json.dumps(
            {
                "point_id": "001",
                "coordinates": H2O_GEOMETRY.tolist(),
                "symbols": H2O_SYMBOLS,
                "tensors": {
                    "rotational_constants_MHz": {
                        "A": 825360.0,
                        "B": 435360.0,
                        "C": 278130.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    mpqc_file = tmp_path / "mpqc.out"
    mpqc_file.write_text("FINAL SINGLE POINT ENERGY -76.123\n", encoding="utf-8")

    bridge = TorqSpcatBridge(str(tensor_file), str(mpqc_file), temperature_k=298.15)
    assert bridge.temperature_k == 298.15
    assert bridge.mpqc_file == Path(mpqc_file)
    assert bridge.rot_A_MHz == 825360.0
    assert bridge.rot_B_MHz == 435360.0
    assert bridge.rot_C_MHz == 278130.0
    assert bridge.sigma == 2


def test_torq_spcat_bridge_extract_orca(
    tmp_path: Path
) -> None:
    tensor_file = tmp_path / "tensor.json"
    tensor_file.write_text(
        json.dumps(
            {
                "point_id": "002",
                "coordinates": H2O_GEOMETRY.tolist(),
                "symbols": H2O_SYMBOLS,
                "tensors": {
                    "rotational_constants_MHz": {
                        "A": 825360.0,
                        "B": 435360.0,
                        "C": 278130.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    mpqc_file = tmp_path / "mpqc.out"
    mpqc_file.write_text(
        "Total Dipole Moment : 0.000 0.000 1.854\n"
        "VIBRATIONAL FREQUENCIES\n"
        "-----------------------\n"
        " 1: 1595.00 cm**-1\n"
        " 2: 3657.00 cm**-1\n"
        " 3: 3756.00 cm**-1\n",
        encoding="utf-8",
    )

    bridge = TorqSpcatBridge(str(tensor_file), str(mpqc_file), temperature_k=298.15)
    bridge.parse_mpqc_observables()
    assert len(bridge.frequencies_cm1) == 3
    assert bridge.dipole_moments["c"] == 1.854

    q_rot, q_vib, q_total = bridge.calculate_partition_functions()
    assert q_rot > 0.0
    assert q_vib >= 1.0
    assert math.isclose(q_total, q_rot * q_vib, rel_tol=1e-9)

    spcat_dir = tmp_path / "scratch"
    
    import os
    original_env = os.environ.get("COCHEM_ARTIFACT_DIR")
    os.environ["COCHEM_ARTIFACT_DIR"] = str(spcat_dir)
    try:
        bridge.export_spcat_catalog()
    finally:
        if original_env is not None:
            os.environ["COCHEM_ARTIFACT_DIR"] = original_env
        else:
            del os.environ["COCHEM_ARTIFACT_DIR"]
    assert (spcat_dir / "spcat" / "spcat_002.var").exists()
    assert (spcat_dir / "spcat" / "spcat_002.int").exists()


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


def test_apply_symmetry_divisors_ammonia_and_ethylene() -> None:
    """Verify symmetry and spin weights for NH3 and C2H4."""
    res_nh3 = apply_symmetry_divisors(NH3_GEOMETRY, NH3_SYMBOLS)
    assert res_nh3.point_group == "C3v"
    assert res_nh3.sigma == 3
    assert res_nh3.spin_weight_ratio_str == "2 1"

    res_c2h4 = apply_symmetry_divisors(C2H4_GEOMETRY, C2H4_SYMBOLS)
    assert res_c2h4.point_group == "D2h"
    assert res_c2h4.sigma == 4

    # Nuclear spin flag sets effective divisor to 1.0 to avoid double counting
    res_spin = apply_symmetry_divisors(H2O_GEOMETRY, H2O_SYMBOLS, use_nuclear_spin=True)
    assert res_spin.effective_divisor == 1.0
    assert "EXACT_NUCLEAR_SPIN_APPLIED" in res_spin.guardrail_status


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

    q_vib_without_lam = calculate_vibrational_partition_function(
        all_freqs, 298.15, exclude_frequencies=[lam_mode]
    )
    expected = q_rot_dvr[298.15] * q_vib_without_lam
    assert math.isclose(q_coupled[298.15], expected, rel_tol=1e-6)


def test_fortran_overflow_guard_and_formatter() -> None:
    """Verify Fortran Double Precision overflow protection and 'D' format."""
    with pytest.raises(FortranOverflowError):
        fortran_overflow_guard({"DJ": 1.5e310})

    with pytest.raises(FortranOverflowError):
        fortran_overflow_guard({"tensor_matrix": np.array([1.0, 2.0, np.inf])})

    with pytest.raises(FortranOverflowError):
        fortran_overflow_guard({"tensor_matrix": np.array([1.0, 2.0, np.nan])})

    formatted = fortran_double_precision_formatter(
        20000, 1.567e-5, uncertainty=1e-7, label="DJ"
    )
    assert "20000" in formatted
    assert "D-05" in formatted
    assert "/ DJ" in formatted

    single_fmt = format_fortran_double(0.00012345, compact=True)
    assert "D-04" in single_fmt


def test_generate_spcat_var_and_int(tmp_path: Path) -> None:
    """Verify generation of .var and .int files."""
    var_file = tmp_path / "scratch" / "test.var"
    var_content = generate_spcat_var(
        "H2O", {"A": 825360.0, "B": 435360.0, "C": 278130.0}, filepath=var_file
    )
    assert var_file.exists()
    assert "H2O Ground State" in var_content

    int_file = tmp_path / "scratch" / "test_{T}K.int"
    int_dict = generate_spcat_int(
        "H2O",
        {"mu_a": 0.0, "mu_b": 1.85, "mu_c": 0.0},
        temperatures=[298.15],
        filepath_template=int_file,
    )
    assert 298.15 in int_dict
    assert (tmp_path / "scratch" / "test_298.1K.int").exists()


def test_3tier_routing_protocol() -> None:
    """Verify the 3-Tier Routing Protocol (MPQC primary, ORCA secondary, CFOUR legacy)."""
    mpqc_data = {
        "energy_hartree": -76.4321,
        "frequencies": [1595.0, 3657.0, 3756.0],
        "dipoles": {"mu_a": 0.0, "mu_b": 0.0, "mu_c": 1.85},
    }
    orca_data = {
        "electronic_energy": -76.4300,
        "harmonic_frequencies": [1590.0, 3650.0, 3750.0],
        "anharmonic_x_matrix": np.array([
            [-42.6, -15.9, -165.8],
            [-15.9, -42.9, -166.1],
            [-165.8, -166.1, -47.8]
        ]),
        "dipole_moments": {"mu_a": 0.0, "mu_b": 0.0, "mu_c": 1.84},
    }
    cfour_data = {
        "eccsd_t": -76.4310,
        "frequencies": [1592.0, 3652.0, 3752.0],
        "dipoles": {"mu_a": 0.0, "mu_b": 0.0, "mu_c": 1.845},
    }

    # Standard routing selects Tier 1 (MPQC)
    res_tier1 = route_3tier_abinitio_payload(
        mpqc_data=mpqc_data, orca_data=orca_data, cfour_data=cfour_data
    )
    assert res_tier1.selected_tier == 1
    assert res_tier1.primary_engine == "MPQC"
    assert res_tier1.is_mpqc_primary is True
    assert res_tier1.electronic_energy_hartree == -76.4321

    # When analytic VPT2 is requested, Tier 2 (ORCA) is selected
    res_tier2 = route_3tier_abinitio_payload(
        mpqc_data=mpqc_data,
        orca_data=orca_data,
        cfour_data=cfour_data,
        require_analytic_vpt2=True,
    )
    assert res_tier2.selected_tier == 2
    assert res_tier2.primary_engine == "ORCA"
    assert res_tier2.is_analytic_vpt2_active is True

    # Fallback to Tier 3 when only CFOUR is available
    res_tier3 = route_3tier_abinitio_payload(cfour_data=cfour_data)
    assert res_tier3.selected_tier == 3
    assert res_tier3.primary_engine == "CFOUR"

    with pytest.raises(ValueError, match="No ab initio data provided"):
        route_3tier_abinitio_payload()


def test_build_complete_spcat_payload_and_manifest(tmp_path: Path) -> None:
    """Verify build_complete_spcat_payload creates verified files and cryptographic manifest."""
    out_dir = tmp_path / "scratch" / "spcat_out"
    payload = build_complete_spcat_payload(
        molecule_name="Water",
        geometry=H2O_GEOMETRY,
        symbols=H2O_SYMBOLS,
        rotational_constants_mhz={"A": 825360.0, "B": 435360.0, "C": 278130.0},
        dipoles_debye={"mu_a": 0.0, "mu_b": 0.0, "mu_c": 1.854},
        harmonic_frequencies_cm1=[1595.0, 3657.0, 3756.0],
        temperatures=[2.0, 10.0, 298.15],
        output_dir=out_dir,
    )

    assert payload.molecule_name == "Water"
    assert len(payload.sha256_var) == 64
    assert len(payload.sha256_int) == 3
    assert Path(payload.var_filepath).exists()
    assert Path(payload.provenance_filepath).exists()

