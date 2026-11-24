# CoChem-TORQ User Guide

**CoChem-TORQ** (Torsional Optimization & Rotational Quantification Engine) is the rotational spectroscopy, constrained torsional scan, and vibrational-perturbation handoff subsystem of the [CoChem](https://github.com/ProfJJK-CoChem) ecosystem.

For installation, canonical execution environments, and scope boundaries, start with [README.md](../README.md).

## Documentation Map

| Guide | Purpose |
| --- | --- |
| [README.md](../README.md) | Repository front door, Codespaces/Actions quickstart, public recipe bounds, and license. |
| [Student Guide](Student_Guide.md) | Step-by-step classroom notebook controls, approval flow, missing-data statuses, and BASE export. |
| [CoChem User Manual](wiki/CoChem_User_Manual.md) | Extended operational manual covering torsional PES scans, rotational spectroscopy, and deliverables. |
| [Implementation SRS](wiki/CoChem-TORQ_Implementation_SRS.md) | Canonical requirements, scientific invariants, and typed interface contracts. |
| [Spectroscopy Implementation](development/spectroscopy_implementation.md) | Validated derivative orders, anharmonic force fields, and rotational response bounds. |
| [Ecosystem Contracts](development/ecosystem_contracts.md) | Versioned file handoff boundaries across CoChem-BASE, CoChem-TOPOS, CoChem-TORQ, and CoChem-SPYFY. |

## Core Workflow

1. **Discover & Validate Recipes**
   ```sh
   cochem-torq recipes
   cochem-torq validate --request examples/student/water-hf-teaching.json
   ```
2. **Plan, Approve & Submit (Canonical GitHub Actions Route)**
   ```sh
   cochem-torq plan --request examples/student/water-hf-teaching.json --output ~/CoChem_Artifacts/water-plan.json
   cochem-torq approve-plan --plan ~/CoChem_Artifacts/water-plan.json --actor COURSE_IDENTITY --output ~/CoChem_Artifacts/water-approved.json
   cochem-torq submit --approved-plan ~/CoChem_Artifacts/water-approved.json --idempotency-key WATER_ATTEMPT_ID
   cochem-torq status RUN_ID
   cochem-torq download RUN_ID --destination ~/CoChem_Artifacts/water-download
   ```
3. **Interactive Notebook Interface (GitHub Codespaces)**
   Open [UI/Start_TORQ.ipynb](../UI/Start_TORQ.ipynb) or launch via `cochem-torq interface` to import an XYZ structure or reviewed `CoChem-TOPOS` ensemble, inspect the 3D principal-axis frame, and manage approvals.
4. **Local Research Validation & Publication Export**
   ```sh
   cochem-torq validate --request /path/to/research-request.json --execution local_validation
   cochem-torq execute --request /path/to/research-request.json --output-dir ~/CoChem_Artifacts/new-research-run
   cochem-torq export RESULT_SHARD --format publication --destination NEW_BUNDLE
   ```

## CoChem-SPYFY & CoChem-GEOM Spectroscopy Handoff

`CoChem-TORQ` maintains strict scientific separation between equilibrium constants ($B_e$) and vibrationally corrected ground-state constants ($B_0 = B_e + \Delta B_{\text{vib}}$). When no VPT2 block is present in an ORCA output, `b0_status` remains `"PENDING"` and `delta_b_vib_mhz` remains `None` (never `0.0`).

- **VPT2 / Pickett Seed Export (`src/cochem_torq/chem490_vpt2_bridge.py`):**
  - `parse_orca_vpt2_rotational_constants(out_path)` parses authenticated ORCA 6.1.1 rotational constants, VPT2 shifts, quartic centrifugal distortion constants, ZPVE, and rotational-axis dipole components with SHA-256 provenance.
  - `export_spyfy_geom_handoff(record, output_dir, molecule_id=...)` emits a versioned JSON receipt (`cochem.torq.vpt2-spectroscopy-handoff/1`) together with Pickett `.var` and `.int` seed files for **CoChem-SPYFY** spectral assignment and **CoChem-GEOM** semi-experimental $r_e^{\text{SE}}$ analysis (`re_se_status = "PENDING"` until isotope-specific $\Delta B_{\text{vib}}$ and experimental $B_0$ are supplied).
- **Cryptographic Payload Bundling (`Libraries/cochem_torq_export.py`):**
  - `lock_provenance_payload(target_directory)` generates an RFC 8785 canonical `spyfy_manifest.json` (`format: "CoChem-SPYFY-Manifest"`) with streaming SHA-256 digests.
  - `bundle_spyfy_payload(...)` packages deliverables into a deterministic `.tar.zst` archive (`CoChem_<Project>_SPYFY_Payload.tar.zst`) with normalized POSIX metadata, verifiable via `verify_payload_integrity(...)`.
