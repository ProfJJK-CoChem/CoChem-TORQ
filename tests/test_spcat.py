import json
import logging
import math
from pathlib import Path

import numpy as np
import pytest

from Libraries.cochem_spcat_bridge import (
    CONSTANTS,
    AirGapViolationError,
    FortranOverflowError,
    LAMTriggerError,
    TorqSpcatBridge,
    apply_symmetry_divisors,
    build_complete_spcat_payload,
    calculate_rotational_constants_from_geometry,
    calculate_vibrational_partition_function,
    format_fortran_double,
    fortran_double_precision_formatter,
    fortran_overflow_guard,
    generate_spcat_int,
    generate_spcat_var,
    get_atomic_mass,
    low_frequency_lam_trap,
    low_frequency_trap,
    route_3tier_abinitio_payload,
    validate_airgap_boundary,
    vibrational_partition_coupling,
)

logger = logging.getLogger(__name__)

# Declared water-shaped mathematical geometry (angstrom); no optimized-state claim
H2O_GEOMETRY = np.array(
    [
        [0.000000, 0.000000, 0.117300],
        [0.000000, 0.757200, -0.469200],
        [0.000000, -0.757200, -0.469200],
    ],
    dtype=np.float64,
)
H2O_SYMBOLS = ["O", "H", "H"]

# Declared mathematical geometry for Ammonia (NH3 in Angstroms)
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

# Declared mathematical geometry for Ethylene (C2H4 in Angstroms)
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
    """Construct an explicit mathematical geometry record with no engine properties."""
    constants = calculate_rotational_constants_from_geometry(H2O_GEOMETRY, H2O_SYMBOLS)
    path = tmp_path / "model-geometry.json"
    path.write_text(
        json.dumps(
            {
                "point_id": "geometry-only",
                "is_linear": False,
                "coordinates": H2O_GEOMETRY.tolist(),
                "symbols": H2O_SYMBOLS,
                "symmetry": {
                    "sigma": 2,
                    "source": "Exact twofold rotation of the declared mathematical geometry",
                },
                "tensors": {
                    "rotational_constants_MHz": {
                        axis: constants[axis] for axis in ("A", "B", "C")
                    }
                },
            }
        )
    )
    bridge = TorqSpcatBridge(
        path, tmp_path / "unavailable-native-output", temperature_k=298.15
    )
    assert bridge.rot_A_MHz == constants["A"]
    assert bridge.rot_B_MHz == constants["B"]
    assert bridge.rot_C_MHz == constants["C"]
    assert bridge.frequencies_cm1 is None and bridge.dipole_moments is None


def test_torq_spcat_bridge_extract_orca(tmp_path: Path) -> None:
    """A mixed vendor parser must reject unsupported property extraction."""
    constants = calculate_rotational_constants_from_geometry(H2O_GEOMETRY, H2O_SYMBOLS)
    path = tmp_path / "geometry.json"
    path.write_text(
        json.dumps(
            {
                "point_id": "geometry-only",
                "is_linear": False,
                "symmetry": {
                    "sigma": 2,
                    "source": "Declared mathematical geometry C2 rotation",
                },
                "tensors": {
                    "rotational_constants_MHz": {
                        axis: constants[axis] for axis in ("A", "B", "C")
                    }
                },
            }
        )
    )
    bridge = TorqSpcatBridge(path, tmp_path / "unavailable-native-output")
    with pytest.raises(NotImplementedError, match="unqualified"):
        bridge.parse_mpqc_observables()
    with pytest.raises(ValueError, match="unavailable"):
        bridge.calculate_partition_functions()
    with pytest.raises(NotImplementedError, match="unqualified"):
        bridge.export_spcat_catalog()
    assert not list(tmp_path.glob("*.var")) and not list(tmp_path.glob("*.int"))


def test_exact_codata_2022_constants() -> None:
    """Validate immutable CODATA 2022 physical constants."""
    assert CONSTANTS.H == 6.62607015e-34
    assert CONSTANTS.K_B == 1.380649e-23
    assert CONSTANTS.C_CM_S == 29979245800.0
    assert abs(CONSTANTS.C_ROT - 505379.008435) < 1e-4
    assert abs(CONSTANTS.HC_OVER_KB - 1.4387768775) < 1e-6
    assert abs(CONSTANTS.AMU_KG - 1.66053906892e-27) < 1e-35


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
    """Retain actual archived ORCA energy without inventing other engine/VPT2 data."""
    from Libraries.cochem_torq_engine import _read_orca_engrad

    energy, _, _, _ = _read_orca_engrad(Path(__file__).parents[1] / "test.engrad", 10)
    data = {"energy_hartree": energy, "method": "archived; method identity unverified"}
    result = route_3tier_abinitio_payload(orca_data=data)
    assert result.primary_engine == "ORCA" and result.selected_tier == 2
    assert result.electronic_energy_hartree == energy
    assert result.harmonic_frequencies is None
    assert result.dipole_moments_debye is None
    assert result.vpt2_x_matrix is None
    assert result.is_analytic_vpt2_active is False
    with pytest.raises(ValueError, match="No supplied record"):
        route_3tier_abinitio_payload(orca_data=data, require_analytic_vpt2=True)
    with pytest.raises(ValueError, match="No supplied record"):
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


def test_mendeleev_dynamic_mass_retrieval() -> None:
    """Verify dynamic IUPAC atomic mass and isotopic mass retrieval via Mendeleev library."""
    m_c = get_atomic_mass("C")
    assert abs(m_c - 12.011) < 0.01

    m_h = get_atomic_mass("H")
    assert abs(m_h - 1.008) < 0.01

    m_o = get_atomic_mass("O")
    assert abs(m_o - 15.999) < 0.01

    # Isotopes
    m_13c = get_atomic_mass("13C")
    assert abs(m_13c - 13.00335) < 0.001

    m_18o = get_atomic_mass("18O")
    assert abs(m_18o - 17.99916) < 0.001

    m_d = get_atomic_mass("D")
    assert abs(m_d - 2.01410) < 0.001

    m_2h = get_atomic_mass("2H")
    assert abs(m_2h - 2.01410) < 0.001

    with pytest.raises(ValueError):
        get_atomic_mass("???")


def test_calculate_rotational_constants_from_geometry_water() -> None:
    """Verify calculation of rotational constants from Cartesian coordinates and Mendeleev masses."""
    rot_constants = calculate_rotational_constants_from_geometry(
        H2O_GEOMETRY, H2O_SYMBOLS
    )
    assert "A" in rot_constants
    assert "B" in rot_constants
    assert "C" in rot_constants
    assert "Ia" in rot_constants
    assert "Ib" in rot_constants
    assert "Ic" in rot_constants

    # For water: A ~ 820,601 MHz, B ~ 437,225 MHz, C ~ 285,244 MHz
    assert abs(rot_constants["A"] - 820601.0) < 500.0
    assert abs(rot_constants["B"] - 437225.0) < 500.0
    assert abs(rot_constants["C"] - 285244.0) < 500.0
    assert rot_constants["Ia"] < rot_constants["Ib"] < rot_constants["Ic"]


def test_multi_rotor_lam_dropping() -> None:
    """Verify multiple LAM modes are simultaneously dropped from Q_vib."""
    temps = [2.0, 10.0, 50.0, 298.15]
    all_freqs = [3100.0, 1500.0, 105.0, 38.0, 24.5]
    lam_modes = [24.5, 38.0]
    q_rot_dvr = {2.0: 1.05, 10.0: 4.8, 50.0: 35.2, 298.15: 185.0}

    q_coupled = vibrational_partition_coupling(
        q_rot_dvr=q_rot_dvr,
        q_vib_orca=all_freqs,
        temp_array=temps,
        lam_frequency=lam_modes,
    )

    q_vib_stiff = calculate_vibrational_partition_function(
        all_freqs, 298.15, exclude_frequencies=lam_modes
    )
    expected = q_rot_dvr[298.15] * q_vib_stiff
    assert math.isclose(q_coupled[298.15], expected, rel_tol=1e-6)


def test_build_complete_spcat_payload_traps_lam() -> None:
    """Verify build_complete_spcat_payload raises LAMTriggerError when modes < 50 cm^-1 are unhandled."""
    with pytest.raises(LAMTriggerError) as exc_info:
        build_complete_spcat_payload(
            molecule_name="FluxionalMolecule",
            geometry=H2O_GEOMETRY,
            symbols=H2O_SYMBOLS,
            rotational_constants_mhz={"A": 825360.0, "B": 435360.0, "C": 278130.0},
            dipoles_debye={"mu_a": 0.0, "mu_b": 0.0, "mu_c": 1.854},
            harmonic_frequencies_cm1=[3600.0, 1500.0, 32.0],
            lam_frequency=None,
        )
    assert 32.0 in exc_info.value.details["flagged_frequencies"]


def test_build_complete_spcat_payload_nuclear_spin(tmp_path: Path) -> None:
    """Verify build_complete_spcat_payload respects use_nuclear_spin flag and double counting guardrail."""
    out_dir = tmp_path / "scratch" / "spcat_spin"
    payload = build_complete_spcat_payload(
        molecule_name="WaterSpin",
        geometry=H2O_GEOMETRY,
        symbols=H2O_SYMBOLS,
        rotational_constants_mhz={"A": 825360.0, "B": 435360.0, "C": 278130.0},
        dipoles_debye={"mu_a": 0.0, "mu_b": 0.0, "mu_c": 1.854},
        harmonic_frequencies_cm1=[1595.0, 3657.0, 3756.0],
        temperatures=[298.15],
        use_nuclear_spin=True,
        output_dir=out_dir,
    )
    assert payload.provenance_manifest["symmetry"]["effective_divisor"] == 1.0


def test_airgap_violation_enforcement() -> None:
    """Verify that writing into Ring 1 static repository tiers raises AirGapViolationError."""
    repo_root = Path(__file__).resolve().parent.parent
    protected_target = repo_root / "Libraries" / "malicious_write.tmp"
    with pytest.raises(AirGapViolationError):
        validate_airgap_boundary(protected_target)

    # Repository root itself must also be protected
    with pytest.raises(AirGapViolationError):
        validate_airgap_boundary(repo_root)


def test_fortran_fixed_header_and_precision_clamp() -> None:
    """Verify strict 4I5 control line header alignment and 22-column precision clamping."""
    var_content = generate_spcat_var(
        "H2O", {"A": 825360.0, "B": 435360.0, "C": 278130.0}
    )
    lines = var_content.splitlines()
    control_line = lines[1]
    # Control line should have exactly 4 leading fields of width 5 (4I5)
    # NPAR = 3 -> '    3', NLINE = 100 -> '  100', NOPT = 0 -> '    0', NWARN = 0 -> '    0'
    assert control_line[:20] == "    3  100    0    0"

    # Extreme exponent formatted double should never exceed 22 columns
    clamped_str = format_fortran_double(-1.0e-105, width=22, precision=15)
    assert len(clamped_str) == 22
    assert clamped_str.strip() == "-1.00000000000000D-105"
