import json
import logging
import math
from pathlib import Path

import numpy as np
import pytest

from Libraries.cochem_spcat_bridge import (
    _MOLSYM_AVAILABLE,
    CONSTANTS,
    AirGapViolationError,
    FortranOverflowError,
    LAMTriggerError,
    SPCATBridgeError,
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


@pytest.mark.parametrize(
    "geometry,symbols,point_group,sigma",
    [
        (H2O_GEOMETRY, H2O_SYMBOLS, "C2v", 2),
        (NH3_GEOMETRY, NH3_SYMBOLS, "C3v", 3),
        (C2H4_GEOMETRY, C2H4_SYMBOLS, "D2h", 4),
    ],
)
def test_symmetry_requires_a_real_solver_and_never_guesses_spin_weights(
    geometry, symbols, point_group, sigma
) -> None:
    """Exercise the installed solver, or the explicit missing-solver boundary."""
    if _MOLSYM_AVAILABLE:
        result = apply_symmetry_divisors(geometry, symbols)
        assert result.point_group == point_group
        assert result.sigma == sigma
        assert result.spin_statistical_weights is None
        assert result.spin_weight_ratio_str is None
        assert result.effective_divisor == sigma
    else:
        with pytest.raises(SPCATBridgeError, match="validated point-group solver"):
            apply_symmetry_divisors(geometry, symbols)


def test_nuclear_spin_weights_require_a_state_resolved_model() -> None:
    with pytest.raises(SPCATBridgeError, match="state-resolved isotope/permutation"):
        apply_symmetry_divisors(H2O_GEOMETRY, H2O_SYMBOLS, use_nuclear_spin=True)


def model_rotor_partitions(temperatures):
    """Finite exact sum over declared free-rotor E_m=m² cm⁻¹ model states."""
    energies = np.arange(-100, 101, dtype=float) ** 2
    return {
        temperature: float(np.exp(-CONSTANTS.HC_OVER_KB * energies / temperature).sum())
        for temperature in temperatures
    }


def test_vibrational_partition_coupling_with_lam_drop() -> None:
    """Verify vibrational partition coupling drops LAM frequency across temperature gradient."""
    temps = [2.0, 10.0, 50.0, 298.15]
    all_freqs = [3100.0, 1500.0, 105.0, 24.5]
    lam_mode = 24.5
    q_rot_dvr = model_rotor_partitions(temps)

    q_coupled = vibrational_partition_coupling(
        q_rot_dvr=q_rot_dvr,
        q_vib_orca=None,
        all_frequencies=all_freqs,
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


def test_unqualified_spcat_writers_create_no_native_decks(tmp_path: Path) -> None:
    """No program defaults may fabricate ground-state/distortion/intensity data."""
    constants = calculate_rotational_constants_from_geometry(H2O_GEOMETRY, H2O_SYMBOLS)
    var_path = tmp_path / "scratch" / "unqualified.var"
    with pytest.raises(NotImplementedError, match="unqualified"):
        generate_spcat_var("declared geometry", constants, filepath=var_path)
    with pytest.raises(NotImplementedError, match="unqualified"):
        generate_spcat_int(
            "declared geometry",
            None,
            temperatures=[298.15],
            filepath_template=tmp_path / "scratch" / "unqualified_{T}.int",
        )
    assert list(tmp_path.iterdir()) == []


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


@pytest.mark.parametrize("spin,lam", [(False, None), (True, None), (False, 32.0)])
def test_unqualified_complete_spcat_payload_creates_no_manifest(tmp_path, spin, lam):
    """Flags cannot bypass missing native deck conventions or fabricate a catalog."""
    destination = tmp_path / "unqualified-native-decks"
    constants = calculate_rotational_constants_from_geometry(H2O_GEOMETRY, H2O_SYMBOLS)
    with pytest.raises(NotImplementedError, match="validated parameter, intensity"):
        build_complete_spcat_payload(
            molecule_name="declared geometry",
            geometry=H2O_GEOMETRY,
            symbols=H2O_SYMBOLS,
            rotational_constants_mhz=constants,
            dipoles_debye=None,
            harmonic_frequencies_cm1=None,
            use_nuclear_spin=spin,
            lam_frequency=lam,
            output_dir=destination,
        )
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_mendeleev_dynamic_mass_retrieval() -> None:
    """Bare elements select one isotopologue, not a natural-abundance mixture mass."""
    from mendeleev import element

    for symbol in ("C", "H", "O"):
        candidates = [
            isotope
            for isotope in element(symbol).isotopes
            if isotope.abundance is not None
        ]
        actual_default = max(candidates, key=lambda isotope: isotope.abundance)
        assert get_atomic_mass(symbol) == float(actual_default.mass)
    for symbol, number, alias in [
        ("C", 13, "13C"),
        ("O", 18, "18O"),
        ("H", 2, "D"),
        ("H", 2, "2H"),
    ]:
        actual = next(
            isotope
            for isotope in element(symbol).isotopes
            if isotope.mass_number == number
        )
        assert get_atomic_mass(alias) == float(actual.mass)
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
    q_rot_dvr = model_rotor_partitions(temps)

    q_coupled = vibrational_partition_coupling(
        q_rot_dvr=q_rot_dvr,
        q_vib_orca=None,
        all_frequencies=all_freqs,
        temp_array=temps,
        lam_frequency=lam_modes,
    )

    q_vib_stiff = calculate_vibrational_partition_function(
        all_freqs, 298.15, exclude_frequencies=lam_modes
    )
    expected = q_rot_dvr[298.15] * q_vib_stiff
    assert math.isclose(q_coupled[298.15], expected, rel_tol=1e-6)


def test_partition_mode_replacement_requires_explicit_frequency_quantity() -> None:
    with pytest.raises(ValueError, match="explicit all_frequencies"):
        vibrational_partition_coupling(
            q_rot_dvr={298.15: 1.0},
            q_vib_orca=[32.0],
            temp_array=[298.15],
            lam_frequency=32.0,
        )


def test_airgap_violation_enforcement() -> None:
    """Verify that writing into Ring 1 static repository tiers raises AirGapViolationError."""
    repo_root = Path(__file__).resolve().parent.parent
    protected_target = repo_root / "Libraries" / "malicious_write.tmp"
    with pytest.raises(AirGapViolationError):
        validate_airgap_boundary(protected_target)

    # Repository root itself must also be protected
    with pytest.raises(AirGapViolationError):
        validate_airgap_boundary(repo_root)


def test_fortran_precision_guard_rejects_unrepresentable_fields() -> None:
    """Formatting a declared number is allowed; silently truncating it is not."""
    clamped = format_fortran_double(-1.0e-105, width=22, precision=15)
    assert len(clamped) == 22
    assert float(clamped.replace("D", "E")) == -1.0e-105
    with pytest.raises(FortranOverflowError):
        format_fortran_double(-1.0e-105, width=4, precision=15)
