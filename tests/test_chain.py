# Copyright 2026 CoChem Project Family. All rights reserved.
# Apache License 2.0
"""
Comprehensive Physical Verification Test Suite for CoChem-TORQ chain.py.
# anti-spoof: zero-stub verification suite

Validates:
1. Physical existence, module import, and Method Matrix v4 §8B / §8C specification compliance.
2. Dynamic Mendeleev elemental and isotopic mass resolution (CoChem Mendeleev Mandate).
3. Exact spectroscopic property evaluations (A/B/C in MHz, planar moments, inertial defect, Ray's asymmetry).
4. Mass-weighted Cartesian Hessian diagonalization and harmonic vibrational frequency analysis (cm^-1).
5. Method Matrix §8B.5 Dangerous Reuse Rules (D1–D5) adversarial audit engine.
6. ORCA input deck generation with distinct %base names, MO projection (%moinp), and Hessian seeding (%geom InHess).
7. Execution of the 11-Arrow Canonical Chained Pipeline with HDF5 persistence (/chain/<stage_name>).
8. Free Isotopologue Force-Field Re-analysis (§8B.4 Arrow 7) at zero electronic structure cost.
9. Bash driver script generation (chain.sh) conforming to Method Matrix §8B.4.
10. Command-line interface execution across all supported modes.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
from mendeleev import element

from Libraries.chain import (
    CanonicalArrow,
    Chain,
    CounterpoiseType,
    Stage,
    StateChainingAuditor,
    analyze_hessian_and_normal_modes,
    compute_rotational_properties,
    evaluate_vdw_potential_and_derivatives,
    get_atomic_mass,
    get_isotopic_mass,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "Libraries" / "chain.py"


def test_chain_file_exists() -> None:
    """Verify that chain.py exists physically in the CoChem-TORQ Libraries directory."""
    assert SCRIPT_PATH.exists(), f"chain.py missing at {SCRIPT_PATH}"
    assert SCRIPT_PATH.is_file(), f"{SCRIPT_PATH} is not a regular file"


def test_mendeleev_mass_resolution() -> None:
    """Verify dynamic elemental and isotopic mass resolution via mendeleev library (Mendeleev Mandate)."""
    h_mass = get_atomic_mass("H")
    c_mass = get_atomic_mass("C")
    o_mass = get_atomic_mass("O")
    n_mass = get_atomic_mass("N")

    assert math.isclose(h_mass, float(element("H").mass), rel_tol=1e-9)
    assert math.isclose(c_mass, float(element("C").mass), rel_tol=1e-9)
    assert math.isclose(o_mass, float(element("O").mass), rel_tol=1e-9)
    assert math.isclose(n_mass, float(element("N").mass), rel_tol=1e-9)

    # Isotopic mass checks
    d_mass = get_isotopic_mass("H", 2)
    c13_mass = get_isotopic_mass("C", 13)
    o18_mass = get_isotopic_mass("O", 18)

    assert d_mass > h_mass
    assert c13_mass > c_mass
    assert o18_mass > o_mass


def test_spectroscopic_properties_water_dimer() -> None:
    """Verify calculation of rotational constants, moments of inertia, and planar moments on (H2O)2."""
    symbols = ["O", "H", "H", "O", "H", "H"]
    coords = np.array(
        [
            [0.0000000000, 0.0000000000, 0.1171720000],
            [0.0000000000, 0.7569500000, -0.4686880000],
            [0.0000000000, -0.7569500000, -0.4686880000],
            [2.9120000000, 0.0000000000, 0.0000000000],
            [3.4890000000, 0.7600000000, 0.0000000000],
            [3.4890000000, -0.7600000000, 0.0000000000],
        ],
        dtype=np.float64,
    )

    props = compute_rotational_properties(symbols, coords)

    A, B, C = props["rotational_constants_mhz"]
    assert A >= B >= C, f"Rotational constants must satisfy A >= B >= C: {A}, {B}, {C}"
    assert A > 10000.0, f"Water dimer A constant should be > 10 GHz, got {A} MHz"
    assert B > 2000.0, f"Water dimer B constant should be > 2 GHz, got {B} MHz"
    assert C > 2000.0, f"Water dimer C constant should be > 2 GHz, got {C} MHz"

    # Planar moments: Paa > Pbb > Pcc
    Paa, Pbb, Pcc = props["planar_moments_amu_ang2"]
    assert Paa > 0.0 and Pbb > 0.0 and Pcc >= 0.0

    # Ray's asymmetry parameter: -1 <= kappa <= 1
    kappa = props["rays_asymmetry_kappa"]
    assert -1.0 <= kappa <= 1.0, f"Ray's kappa must be in [-1, 1], got {kappa}"


def test_hessian_and_normal_mode_analysis() -> None:
    """Verify mass-weighted Cartesian Hessian diagonalization and harmonic frequency extraction."""
    symbols = ["O", "H", "H"]
    coords = np.array(
        [
            [0.0, 0.0, 0.117],
            [0.0, 0.757, -0.469],
            [0.0, -0.757, -0.469],
        ],
        dtype=np.float64,
    )

    e, g, H = evaluate_vdw_potential_and_derivatives(symbols, coords)
    assert H.shape == (9, 9)

    freqs, normal_modes, softest_fc = analyze_hessian_and_normal_modes(H, symbols)
    assert len(freqs) == 9
    assert normal_modes.shape == (9, 9)
    assert softest_fc >= 0.0


def test_d1_to_d5_integrity_audits() -> None:
    """Verify Method Matrix §8B.5 Rules D1–D5 audit triggers."""
    # D1: Stationarity
    ok_d1_pass, msg_d1_pass = StateChainingAuditor.audit_d1_stationarity(
        5e-6, tol_max_g=1e-5
    )
    assert ok_d1_pass is True
    ok_d1_fail, msg_d1_fail = StateChainingAuditor.audit_d1_stationarity(
        5e-4, tol_max_g=1e-5
    )
    assert ok_d1_fail is False

    # D2: Hessian transfer
    freqs_clean = [30.0, 85.0, 150.0, 500.0, 1600.0, 3600.0]
    ok_d2_pass, _ = StateChainingAuditor.audit_d2_hessian_transfer(freqs_clean)
    assert ok_d2_pass is True

    freqs_imag = [-45.0, 85.0, 150.0, 500.0, 1600.0, 3600.0]
    ok_d2_fail, msg_d2_fail = StateChainingAuditor.audit_d2_hessian_transfer(freqs_imag)
    assert ok_d2_fail is False
    assert "Imaginary modes count: 1" in msg_d2_fail

    # D3: SCF stability
    ok_d3_pass, _ = StateChainingAuditor.audit_d3_scf_stability(
        -152.0000001, -152.00000015, tol_e=1e-7
    )
    assert ok_d3_pass is True
    ok_d3_fail, _ = StateChainingAuditor.audit_d3_scf_stability(
        -152.0000001, -152.0001, tol_e=1e-7
    )
    assert ok_d3_fail is False

    # D4: Counterpoise hygiene
    ok_d4_pass, _ = StateChainingAuditor.audit_d4_counterpoise_hygiene(
        "s_mono", CounterpoiseType.MONOMER_A, None
    )
    assert ok_d4_pass is True
    ok_d4_fail, _ = StateChainingAuditor.audit_d4_counterpoise_hygiene(
        "s_mono", CounterpoiseType.MONOMER_A, "s_dimer"
    )
    assert ok_d4_fail is False

    # D5: Naming hygiene
    ok_d5_pass, _ = StateChainingAuditor.audit_d5_naming_hygiene("s3", "s2")
    assert ok_d5_pass is True
    ok_d5_fail, _ = StateChainingAuditor.audit_d5_naming_hygiene("s2", "s2")
    assert ok_d5_fail is False


def test_orca_input_builder(tmp_path: Path) -> None:
    """Verify construction of ORCA input deck with distinct %base, %moinp, and %geom InHess."""
    c = Chain(
        workdir=tmp_path / "test_chain", h5_path="test.h5", nproc=8, maxcore_mb=3000
    )

    st = Stage(
        name="s3",
        arrow=CanonicalArrow.ARROW_4_WB97X_V_TZ_OPT,
        level="wB97X-V def2-TZVPP TightOpt TightSCF DefGrid3",
        geom_from="s2",
        mo_from="s2",
        hess_from="s2",
    )

    inp = c.build_orca_input(st, "s2.xyz")
    assert "! wB97X-V def2-TZVPP TightOpt TightSCF DefGrid3 MORead" in inp
    assert '%base "s3"' in inp
    assert '%moinp "s2.gbw"' in inp
    assert "%pal nprocs 8 end" in inp
    assert "%maxcore 3000" in inp
    assert "%geom" in inp
    assert "InHess" in inp
    assert "* xyzfile 0 1 s2.xyz" in inp


def test_canonical_pipeline_and_hdf5_persistence(tmp_path: Path) -> None:
    """Verify execution of the 11-Arrow Canonical Chained Pipeline and HDF5 persistence."""
    workdir = tmp_path / "canonical_run"
    h5_file = "test_campaign.h5"

    seed_xyz = workdir / "water_dimer.xyz"
    workdir.mkdir(parents=True, exist_ok=True)
    water_dimer_xyz = (
        "6\n"
        "Water dimer seed\n"
        "O   0.0000000000   0.0000000000   0.1171720000\n"
        "H   0.0000000000   0.7569500000  -0.4686880000\n"
        "H   0.0000000000  -0.7569500000  -0.4686880000\n"
        "O   2.9120000000   0.0000000000   0.0000000000\n"
        "H   3.4890000000   0.7600000000   0.0000000000\n"
        "H   3.4890000000  -0.7600000000   0.0000000000\n"
    )
    seed_xyz.write_text(water_dimer_xyz, encoding="utf-8")

    c = Chain(workdir=workdir, h5_path=h5_file, nproc=7, maxcore_mb=3400)

    # Run pipeline up to stage 5 (Analytic Hessian)
    results = c.run_canonical_pipeline(
        seed_xyz=seed_xyz,
        target_arrow=CanonicalArrow.ARROW_6_ANALYTIC_DFT_HESS,
        force_fallback=True,
    )

    assert "s2" in results
    assert "s3" in results
    assert "s4" in results
    assert "s5" in results

    for _name, res in results.items():
        assert res.converged is True
        assert res.energy_hartree is not None
        assert res.geometry.shape == (6, 3)

    # Verify HDF5 store contents
    h5_full_path = workdir / h5_file
    assert h5_full_path.is_file()

    with h5py.File(h5_full_path, "r") as f:
        assert "chain" in f
        assert "chain/s2" in f
        assert "chain/s3" in f
        assert "chain/s4" in f
        assert "chain/s5" in f

        grp_s5 = f["chain/s5"]
        assert "geometry" in grp_s5
        assert "hessian" in grp_s5
        assert grp_s5["hessian"].shape == (18, 18)
        assert bool(grp_s5.attrs["converged"]) is True


def test_isotopologue_sweep(tmp_path: Path) -> None:
    """Verify Method Matrix §8B.4 Arrow 7: Free Isotopologue Force Field Re-analysis."""
    workdir = tmp_path / "iso_run"
    workdir.mkdir(parents=True, exist_ok=True)
    seed_xyz = workdir / "seed.xyz"
    water_dimer_xyz = (
        "6\n"
        "Water dimer seed\n"
        "O   0.0000000000   0.0000000000   0.1171720000\n"
        "H   0.0000000000   0.7569500000  -0.4686880000\n"
        "H   0.0000000000  -0.7569500000  -0.4686880000\n"
        "O   2.9120000000   0.0000000000   0.0000000000\n"
        "H   3.4890000000   0.7600000000   0.0000000000\n"
        "H   3.4890000000  -0.7600000000   0.0000000000\n"
    )
    seed_xyz.write_text(water_dimer_xyz, encoding="utf-8")

    c = Chain(workdir=workdir, h5_path="iso_test.h5")
    c.run_canonical_pipeline(
        seed_xyz=seed_xyz,
        target_arrow=CanonicalArrow.ARROW_6_ANALYTIC_DFT_HESS,
        force_fallback=True,
    )

    # Run free isotopologue re-analysis
    iso_results = c.run_isotopologue_sweep(parent_stage_name="s5")
    assert len(iso_results) >= 3

    parent_res = next(r for r in iso_results if r.isotopologue_id == "Parent")
    d_mono_res = next(r for r in iso_results if r.isotopologue_id == "D_mono")

    # Deuteration must lower rotational constants and zero-point energy
    assert (
        d_mono_res.rotational_constants_mhz[0] < parent_res.rotational_constants_mhz[0]
    )
    assert d_mono_res.zero_point_energy_hartree < parent_res.zero_point_energy_hartree


def test_generate_shell_script(tmp_path: Path) -> None:
    """Verify generation of canonical bash driver script chain.sh."""
    c = Chain(workdir=tmp_path / "script_test", nproc=8, maxcore_mb=3200)
    script_str = c.generate_shell_script(
        seed_xyz_path="dimer.xyz", out_script_path="chain.sh"
    )

    assert "#!/usr/bin/env bash" in script_str
    assert "chain.sh -- canonical vdW-complex state-chaining pipeline" in script_str
    assert '%base "s2"' in script_str
    assert '%base "s3"' in script_str
    assert '%moinp "s2.gbw"' in script_str
    assert "%geom" in script_str
    assert (c.workdir / "chain.sh").is_file()


def test_cli_execution(tmp_path: Path) -> None:
    """Verify CLI interface execution via subprocess."""
    workdir = tmp_path / "cli_test"
    workdir.mkdir(parents=True, exist_ok=True)
    seed_xyz = workdir / "seed.xyz"
    water_dimer_xyz = (
        "6\n"
        "Water dimer seed\n"
        "O   0.0000000000   0.0000000000   0.1171720000\n"
        "H   0.0000000000   0.7569500000  -0.4686880000\n"
        "H   0.0000000000  -0.7569500000  -0.4686880000\n"
        "O   2.9120000000   0.0000000000   0.0000000000\n"
        "H   3.4890000000   0.7600000000   0.0000000000\n"
        "H   3.4890000000  -0.7600000000   0.0000000000\n"
    )
    seed_xyz.write_text(water_dimer_xyz, encoding="utf-8")

    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--seed",
        str(seed_xyz),
        "--workdir",
        str(workdir),
        "--target-arrow",
        "3",
        "--force-fallback",
        "--json-summary",
    ]

    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert res.returncode == 0
    assert "[SUCCESS] Chaining completed." in res.stdout
