"""Zero-mock tests for CHEM 490 Wk 7/9 VdW torsional scan + ORCA 6.1.1 VPT2 bridge."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from cochem_torq import module_provider
from cochem_torq.chem490_vpt2_bridge import (
    COLLABORATOR_INSTITUTION,
    HANDOFF_SCHEMA,
    SCAN_SCHEMA,
    OrcaVpt2RotationalRecord,
    RelaxedTorsionalScanReceipt,
    SpycFitGeomHandoffPacket,
    export_spycfit_geom_handoff,
    parse_orca_vpt2_rotational_constants,
    run_relaxed_1d_torsional_scan,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = REPO_ROOT / "benchmarks" / "chem490_vdw_torsion_vpt2"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "chem490_vpt2"


def _resolve_fixture(filename: str) -> Path:
    candidate = FIXTURES_DIR / filename
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Authentic ORCA fixture missing at {candidate}")


def test_module_provider_reexports_bridge_without_breaking_operations() -> None:
    assert (
        module_provider.parse_orca_vpt2_rotational_constants
        is parse_orca_vpt2_rotational_constants
    )
    assert module_provider.export_spycfit_geom_handoff is export_spycfit_geom_handoff
    assert (
        module_provider.run_relaxed_1d_torsional_scan is run_relaxed_1d_torsional_scan
    )


def test_parse_orca_equilibrium_only_never_equates_be_with_b0(
    tmp_path: Path,
) -> None:
    orca_out = _resolve_fixture("h2o_ccsd_pvtz.out")
    record = parse_orca_vpt2_rotational_constants(orca_out)

    assert record.orca_version == "6.1.1"
    assert record.terminated_normally is True
    assert record.has_vpt2_block is False
    assert record.b0_status == "PENDING"
    assert record.b0_cm1 is None
    assert record.b0_mhz is None
    assert record.delta_b_vib_cm1 is None
    assert record.delta_b_vib_mhz is None
    assert record.source_sha256 == hashlib.sha256(orca_out.read_bytes()).hexdigest()

    # Exact H2O CCSD/cc-pVTZ equilibrium rotational constants in MHz
    assert math.isclose(record.be_mhz.a, 810993.954021, rel_tol=1e-9)
    assert math.isclose(record.be_mhz.b, 441326.841003, rel_tol=1e-9)
    assert math.isclose(record.be_mhz.c, 285800.093093, rel_tol=1e-9)
    assert record.be_mhz.a >= record.be_mhz.b >= record.be_mhz.c > 0.0

    # Last dipole block must be correlated MDCI(SD), not HF reference
    assert record.dipole is not None
    assert record.dipole.method_label == "MDCI(SD)"
    assert math.isclose(record.dipole.mu_b_debye, -1.908676, abs_tol=1e-6)
    assert record.zpve.harmonic_hartree is not None
    assert math.isclose(record.zpve.harmonic_hartree, 0.02171666, abs_tol=1e-8)

    # Export SpycFit/GEOM handoff packet and verify strict status locks
    packet, json_path = export_spycfit_geom_handoff(
        record, tmp_path, molecule_id="H2O_CCSD_cc-pVTZ"
    )
    assert json_path.is_file()
    assert packet.schema_version == HANDOFF_SCHEMA
    assert packet.collaborator_institution == COLLABORATOR_INSTITUTION
    assert packet.b0_status == "PENDING"
    assert packet.re_se_status == "PENDING"
    assert packet.pickett_parameter_basis == "Be_equilibrium_only_not_B0"
    var_path = tmp_path / packet.pickett_var_filename
    int_path = tmp_path / packet.pickett_int_filename
    assert hashlib.sha256(var_path.read_bytes()).hexdigest() == (
        packet.pickett_var_sha256
    )
    assert hashlib.sha256(int_path.read_bytes()).hexdigest() == (
        packet.pickett_int_sha256
    )
    assert "002     -1.908676" in int_path.read_text(encoding="utf-8")


def test_parse_orca_vpt2_ar_so2_and_export_handoff(tmp_path: Path) -> None:
    vpt2_out = _resolve_fixture("S3b_b3lyp_vpt2.out")
    record = parse_orca_vpt2_rotational_constants(vpt2_out)

    assert record.orca_version == "6.1.1"
    assert record.terminated_normally is True
    assert record.has_vpt2_block is True
    assert record.b0_status == "AVAILABLE"
    assert record.b0_cm1 is not None
    assert record.b0_mhz is not None
    assert record.delta_b_vib_cm1 is not None
    assert record.delta_b_vib_mhz is not None

    # Exact identity: B_e + Delta B_vib == B_0 in both cm^-1 and MHz
    assert math.isclose(
        record.be_cm1.a + record.delta_b_vib_cm1.delta_a, record.b0_cm1.a, abs_tol=1e-12
    )
    assert math.isclose(
        record.be_cm1.b + record.delta_b_vib_cm1.delta_b, record.b0_cm1.b, abs_tol=1e-12
    )
    assert math.isclose(
        record.be_cm1.c + record.delta_b_vib_cm1.delta_c, record.b0_cm1.c, abs_tol=1e-12
    )
    assert math.isclose(
        record.be_mhz.a + record.delta_b_vib_mhz.delta_a, record.b0_mhz.a, abs_tol=1e-9
    )

    # Quartic CDCs & anharmonic ZPVE must be populated from real ORCA VPT2 output
    assert record.quartic_cdc is not None
    assert record.quartic_cdc.s_reduced_cm1 is not None
    assert set(record.quartic_cdc.s_reduced_cm1) == {"DJ", "DK", "DJK", "D1", "D2"}
    assert record.quartic_cdc.a_reduced_cm1 is not None
    assert {"DelJ", "DelK", "DelJK", "delJ", "delK"} <= set(
        record.quartic_cdc.a_reduced_cm1
    )
    assert record.zpve.harmonic_cm1 is not None
    assert record.zpve.anharmonic_correction_cm1 is not None
    assert record.zpve.total_anharmonic_cm1 is not None

    packet, _ = export_spycfit_geom_handoff(record, tmp_path, molecule_id="Ar-SO2_VPT2")
    assert packet.b0_status == "AVAILABLE"
    assert packet.re_se_status == "PENDING"
    assert packet.pickett_parameter_basis == "B0_vpt2"
    var_text = (tmp_path / packet.pickett_var_filename).read_text(encoding="utf-8")
    for code in ("10000", "20000", "30000", "200", "1100", "2000", "40100", "50000"):
        assert code in var_text


def test_parse_orca_vpt2_water_exact_values() -> None:
    water_vpt2 = _resolve_fixture("vpt2_water_b3lyp.out")
    record = parse_orca_vpt2_rotational_constants(water_vpt2)

    assert record.has_vpt2_block is True
    assert record.b0_status == "AVAILABLE"
    assert record.b0_cm1 is not None
    assert record.delta_b_vib_cm1 is not None
    assert math.isclose(record.be_cm1.a, 27.45356, abs_tol=1e-5)
    assert math.isclose(record.b0_cm1.a, 27.89286, abs_tol=1e-5)
    assert math.isclose(record.delta_b_vib_cm1.delta_a, 0.43930, abs_tol=1e-5)
    assert math.isclose(record.zpve.harmonic_cm1 or 0.0, 4673.190153, abs_tol=1e-4)
    assert math.isclose(record.zpve.total_anharmonic_cm1 or 0.0, 4602.410, abs_tol=1e-2)


def test_rejects_empty_or_non_rotational_orca_file(tmp_path: Path) -> None:
    empty_file = tmp_path / "empty.out"
    empty_file.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="nonempty"):
        parse_orca_vpt2_rotational_constants(empty_file)

    no_rot = tmp_path / "no_rot.out"
    no_rot.write_text("Program Version 6.1.1\nNo rotational data\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No ORCA rotational constant block"):
        parse_orca_vpt2_rotational_constants(no_rot)


def test_real_pyscf_relaxed_1d_torsional_scan_h2o2(tmp_path: Path) -> None:
    out_json = tmp_path / "h2o2_3pt_scan.json"
    receipt = run_relaxed_1d_torsional_scan(
        molecule_id="H2O2",
        dihedral_angles_deg=(0.0, 115.0, 180.0),
        method="rhf",
        basis="6-31g*",
        output_json=out_json,
    )

    assert receipt.schema_version == SCAN_SCHEMA
    assert receipt.collaborator_institution == COLLABORATOR_INSTITUTION
    assert len(receipt.points) == 3
    assert all(pt.scf_converged for pt in receipt.points)
    # Physical H2O2 internal rotation ordering: E(cis=0) > E(trans=180) > E(skew=115)
    assert receipt.cis_barrier_cm1 is not None and receipt.cis_barrier_cm1 > 2000.0
    assert receipt.trans_barrier_cm1 is not None and receipt.trans_barrier_cm1 > 100.0
    assert receipt.cis_barrier_cm1 > receipt.trans_barrier_cm1
    assert receipt.cis_barrier_kcal_mol is not None
    assert receipt.cis_barrier_kcal_mol > 5.0
    assert out_json.is_file()
    reloaded = RelaxedTorsionalScanReceipt.model_validate(
        json.loads(out_json.read_text(encoding="utf-8"))
    )
    assert reloaded.scan_sha256 == receipt.scan_sha256


def test_committed_benchmark_receipts_are_valid_and_provenance_checked() -> None:
    scan_path = BENCH_DIR / "h2o2_rhf_631gs_relaxed_torsion_scan.json"
    scan_receipt = RelaxedTorsionalScanReceipt.model_validate(
        json.loads(scan_path.read_text(encoding="utf-8"))
    )
    assert len(scan_receipt.points) == 13
    assert scan_receipt.collaborator_institution == COLLABORATOR_INSTITUTION
    assert scan_receipt.cis_barrier_cm1 is not None
    assert scan_receipt.trans_barrier_cm1 is not None
    assert scan_receipt.cis_barrier_cm1 > scan_receipt.trans_barrier_cm1 > 0.0

    for name, expected_b0_status in (
        ("h2o_ccsd_pvtz_handoff.json", "PENDING"),
        ("ar_so2_b3lyp_vpt2_handoff.json", "AVAILABLE"),
        ("h2o_b3lyp_vpt2_handoff.json", "AVAILABLE"),
    ):
        pkt_path = BENCH_DIR / name
        packet = SpycFitGeomHandoffPacket.model_validate(
            json.loads(pkt_path.read_text(encoding="utf-8"))
        )
        assert packet.collaborator_institution == COLLABORATOR_INSTITUTION
        assert packet.b0_status == expected_b0_status
        assert packet.re_se_status == "PENDING"
        OrcaVpt2RotationalRecord.model_validate(
            packet.rotational_record.model_dump(mode="json")
        )
        var_path = BENCH_DIR / packet.pickett_var_filename
        int_path = BENCH_DIR / packet.pickett_int_filename
        assert hashlib.sha256(var_path.read_bytes()).hexdigest() == (
            packet.pickett_var_sha256
        )
        assert hashlib.sha256(int_path.read_bytes()).hexdigest() == (
            packet.pickett_int_sha256
        )
