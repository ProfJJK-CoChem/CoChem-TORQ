"""Generate authentic CHEM 490 Wk 7/9 VdW torsional scan & ORCA VPT2 receipts."""

from __future__ import annotations

from pathlib import Path

from cochem_torq.chem490_vpt2_bridge import (
    export_spycfit_geom_handoff,
    parse_orca_vpt2_rotational_constants,
    run_relaxed_1d_torsional_scan,
)

BENCH_DIR = Path(__file__).resolve().parent


def main() -> None:
    BENCH_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Real PySCF + geomeTRIC 13-point relaxed 1D torsional PES scan on H2O2
    scan = run_relaxed_1d_torsional_scan(
        molecule_id="H2O2",
        method="rhf",
        basis="6-31g*",
        output_json=BENCH_DIR / "h2o2_rhf_631gs_relaxed_torsion_scan.json",
    )
    print(
        f"[SCAN] H2O2 RHF/6-31G* ({len(scan.points)} pts): "
        f"eq_dih={scan.equilibrium_dihedral_deg:.2f} deg, "
        f"cis={scan.cis_barrier_cm1:.1f} cm^-1 "
        f"({scan.cis_barrier_kcal_mol:.3f} kcal/mol), "
        f"trans={scan.trans_barrier_cm1:.1f} cm^-1 "
        f"({scan.trans_barrier_kcal_mol:.3f} kcal/mol), "
        f"sha256={scan.scan_sha256[:16]}..."
    )

    # 2. Real ORCA 6.1.1 equilibrium-only (H2O CCSD/cc-pVTZ)
    h2o_eq_path = Path(
        "/mnt/d/__CoChem/GitHub-Repo/CoChem-GEOM/benchmarks/"
        "h2o_t_geom_01/orca_ccsd_cc_pvtz/h2o_ccsd_pvtz.out"
    )
    h2o_eq = parse_orca_vpt2_rotational_constants(h2o_eq_path)
    pkt_eq, _ = export_spycfit_geom_handoff(
        h2o_eq,
        BENCH_DIR,
        molecule_id="H2O_CCSD_cc-pVTZ",
        file_stem="h2o_ccsd_pvtz",
    )
    print(
        f"[ORCA-EQ] {h2o_eq.source_filename}: ver={h2o_eq.orca_version}, "
        f"b0_status={h2o_eq.b0_status}, delta_b_vib={h2o_eq.delta_b_vib_mhz}, "
        f"Be(MHz)=({h2o_eq.be_mhz.a:.2f}, {h2o_eq.be_mhz.b:.2f}, "
        f"{h2o_eq.be_mhz.c:.2f}), re_se={pkt_eq.re_se_status}"
    )

    # 3. Real ORCA 6.1.1 VPT2 (Ar-SO2 VdW complex)
    arso2_vpt2_path = Path(
        "/mnt/d/_orca_jobs/ar-so2/canary_s3_revdsd_dual_track/S3b_b3lyp_vpt2.out"
    )
    arso2_dip_path = Path(
        "/mnt/d/_orca_jobs/ar-so2/canary_s3_revdsd_dual_track/S2a_revdsd_dipole.out"
    )
    arso2_vpt2 = parse_orca_vpt2_rotational_constants(arso2_vpt2_path)
    arso2_dip = parse_orca_vpt2_rotational_constants(arso2_dip_path)
    dip_override = (
        (
            arso2_dip.dipole.mu_a_debye,
            arso2_dip.dipole.mu_b_debye,
            arso2_dip.dipole.mu_c_debye,
        )
        if arso2_dip.dipole is not None
        else None
    )
    pkt_arso2, _ = export_spycfit_geom_handoff(
        arso2_vpt2,
        BENCH_DIR,
        molecule_id="Ar-SO2_B3LYP-D4_VPT2",
        file_stem="ar_so2_b3lyp_vpt2",
        dipole_override_debye=dip_override,
        dipole_source_sha256=(
            arso2_dip.source_sha256 if dip_override is not None else None
        ),
    )
    assert arso2_vpt2.b0_mhz is not None
    assert arso2_vpt2.delta_b_vib_mhz is not None
    print(
        f"[ORCA-VPT2-VDW] {arso2_vpt2.source_filename}: ver={arso2_vpt2.orca_version}, "
        f"b0_status={arso2_vpt2.b0_status}, "
        f"Be(MHz)=({arso2_vpt2.be_mhz.a:.2f}, {arso2_vpt2.be_mhz.b:.2f}, "
        f"{arso2_vpt2.be_mhz.c:.2f}), "
        f"B0(MHz)=({arso2_vpt2.b0_mhz.a:.2f}, {arso2_vpt2.b0_mhz.b:.2f}, "
        f"{arso2_vpt2.b0_mhz.c:.2f}), "
        f"dBvib(MHz)=({arso2_vpt2.delta_b_vib_mhz.delta_a:.2f}, "
        f"{arso2_vpt2.delta_b_vib_mhz.delta_b:.2f}, "
        f"{arso2_vpt2.delta_b_vib_mhz.delta_c:.2f}), "
        f"re_se={pkt_arso2.re_se_status}"
    )

    # 4. Real ORCA 6.1.1 VPT2 (H2O B3LYP-D4 VPT2)
    h2o_vpt2_path = Path(
        "/mnt/d/_orca_jobs/hcooh-hsh/HCOOH-H2S_mw_predictions/"
        "tests/fixtures/vpt2_water_b3lyp.out"
    )
    h2o_vpt2 = parse_orca_vpt2_rotational_constants(h2o_vpt2_path)
    pkt_h2o_vpt2, _ = export_spycfit_geom_handoff(
        h2o_vpt2,
        BENCH_DIR,
        molecule_id="H2O_B3LYP-D4_VPT2",
        file_stem="h2o_b3lyp_vpt2",
    )
    assert h2o_vpt2.b0_cm1 is not None
    print(
        f"[ORCA-VPT2-H2O] {h2o_vpt2.source_filename}: b0_status={h2o_vpt2.b0_status}, "
        f"Be(cm-1)=({h2o_vpt2.be_cm1.a:.5f}, {h2o_vpt2.be_cm1.b:.5f}, "
        f"{h2o_vpt2.be_cm1.c:.5f}), "
        f"B0(cm-1)=({h2o_vpt2.b0_cm1.a:.5f}, {h2o_vpt2.b0_cm1.b:.5f}, "
        f"{h2o_vpt2.b0_cm1.c:.5f}), "
        f"re_se={pkt_h2o_vpt2.re_se_status}"
    )


if __name__ == "__main__":
    main()
