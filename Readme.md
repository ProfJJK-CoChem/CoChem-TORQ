# CoChem-TORQ

**Torsional Optimization & Rotational Quantification Engine** is the rotational
spectroscopy component of [CoChem](https://github.com/ProfJJK-CoChem).
CoChem-BASE, TOPOS and TORQ exchange explicit molecular identities, electronic
states and method provenance through verified files.

The current student route runs genuine bounded CPU calculations: optimized
geometry, harmonic characterization, equilibrium rotational constants and a
labeled rigid-rotor teaching catalog. **GitHub Actions is the canonical
calculation environment; GitHub Codespaces is the canonical interface.**
The implemented teaching route does not establish identification accuracy or
complete the full research SRS. Advanced products retain explicit qualification
gates and scientifically valid earlier results when a later stage is unavailable.

**Author/PI:** Dr. Joshua John Klaassen
**ORCiD:** [0009-0007-1506-4401](https://orcid.org/0009-0007-1506-4401)

## Start in Codespaces

Your course repository must first publish the reviewed
[calculation workflow](.github/workflows/calculation.yml) on its default branch,
enable Actions, and provide your account with Actions write access for dispatch
and read access for results. Local changes or a pull request alone do not install
the remote workflow. The interface checks workflow availability before dispatch.

1. Open the course repository in GitHub Codespaces. Its lifecycle setup installs
   TORQ's interface and the reviewed BASE/TOPOS revisions in isolated environments.
2. Open its private forwarded port 8888. The authenticated JupyterLab server starts
   automatically. Its local token is stored privately in
   `~/CoChem_Artifacts/interface/token`; never publish that file or its log.
3. Open [UI/Start_TORQ.ipynb](UI/Start_TORQ.ipynb) and run its launch cell. Import
   an XYZ geometry, a complete request, or a supported BASE/TOPOS handoff. Specify
   charge, multiplicity, method, products and resources explicitly.
4. Preview the actual three-dimensional mass frame, check the request, and submit
   it to Actions. Save the run link, refresh status, download the result, and read
   the scientific stage statuses separately from the Actions job status.

Opening the notebook does not start an electronic calculation. The
[student guide](docs/Student_Guide.md) explains each control, missing-data status
and BASE export. [Canonical environments](docs/development/canonical_environments.md)
documents setup, authentication, image identities and deployment evidence.

Recover or check the existing interface from the repository root:

```sh
python scripts/student_setup.py start --port 8888
python scripts/student_setup.py check --port 8888
```

`cochem-torq interface` also launches the notebook through actual JupyterLab when
an inherited `JUPYTER_TOKEN` is configured. Codespaces manages this authentication
through the lifecycle helper. `python scripts/student_setup.py stop` stops only
the interface process owned by that helper.

## Student calculation scope

| Public recipe | Scientific scope |
|---|---|
| `hf-sto-3g-education` | Restricted Hartree–Fock/STO-3G teaching workflow. |
| `hf-cc-pvdz-research` | Restricted Hartree–Fock/cc-pVDZ validation experiment. Correlation and dispersion remain absent. |

Both profiles remain experimental and uncalibrated against identification targets.
The public worker permits 1–12 atoms from H through Ne, a closed-shell singlet,
absolute charge at most 4, 1–2 CPU cores, 256–2048 MB of requested memory and at
most 600 seconds. The normalized request transport is limited to 16 KiB.
All supported isotope masses must exist in the actual mass database.
The [workflow input validator](ci_tools/actions_request.py) and the application
enforce these bounds before execution.

A real engine image runs PySCF/geomeTRIC and performs an actual native derivative
qualification experiment before a public calculation. An enabled software
profile and passed numerical checks do not establish universal molecular accuracy.
Results contain request and recipe digests, actual source/engine identities,
units, isotope/state metadata, convergence evidence, uncertainty availability and
native artifacts. Downloaded inventories and scientific identities are verified.
Actions artifacts expire after 30 days; export important work before expiry.

Start with [the water input request](examples/student/water-hf-teaching.json).
Its coordinates are starting inputs, not experimental reference data.
The same application handles the notebook and CLI:

```sh
cochem-torq recipes
cochem-torq validate --request examples/student/water-hf-teaching.json
cochem-torq submit --request examples/student/water-hf-teaching.json
cochem-torq status RUN_ID
cochem-torq download RUN_ID --destination ~/CoChem_Artifacts/water-download
```

Set `COCHEM_TORQ_GITHUB_REPOSITORY=OWNER/REPOSITORY` for a course calculation
repository when it differs from the configured repository. Authentication uses
the existing official GitHub CLI identity; never embed credentials in requests.
An access error stays visible and does not start a local replacement calculation.

## Scientific architecture and research gates

The typed spectroscopy route is:

```text
electronic structure → optimized r_e → equilibrium constants → harmonic analysis
→ anharmonic force field → resonance analysis → VPT2
→ vibration-corrected constants → catalog
```

Each stage records provenance, units, quality checks and uncertainty availability.
Missing frequencies, energies, corrections, dipoles and uncertainties remain
missing. A harmonic Hessian alone does not establish VPT2 or ground-state B₀.
Teaching rigid-rotor lines use equilibrium constants Bₑ and do not certify
laboratory or astronomical identification.

The independent research path includes genuine small-system finite-displacement
force fields and perturbation/resonance machinery. Its validated derivative
orders, molecule bounds and blocked rotational responses are documented in
[the spectroscopy implementation](docs/development/spectroscopy_implementation.md).
Explicit local validation is available through `cochem-torq execute`; it is
separate from the public classroom service:

```sh
cochem-torq validate --request /path/to/research-request.json --execution local_validation
cochem-torq execute --request /path/to/research-request.json --output-dir ~/CoChem_Artifacts/new-research-run
```

`cochem-torq recipes` lists the exact locally supported DFT-D4, MP2 and bounded
HF anharmonic validation profiles and their limitations. Installing an engine
never enables an undocumented method or grants identification eligibility.
Exact open-source revDSD-PBEP86-D4 remains blocked pending independent recipe,
energy-component and orbital-response/higher-derivative qualification; see
[its implementation milestones](docs/development/revdsd_implementation.md).

The [implementation SRS](docs/wiki/CoChem-TORQ_Implementation_SRS.md) and
[method-matrix contract](docs/wiki/Method_Matrix_Implementation_Contract.md) are
the implementation requirements. All 140 historical matrix row identities are
preserved; a documented row does not activate a scientific profile. The
[complete release scope audit](docs/development/student_release_scope.md),
[scientific acceptance gaps](docs/development/scientific_acceptance_gaps.md),
[bounded benchmark plan](benchmarks/rotational-identification/preregistration.json)
and [unverified-claims register](docs/wiki/review/Unverified_Claims_Register.md)
state what further evidence is required for publication and identification.

## BASE and TOPOS interoperability

BASE and TOPOS have overlapping Python namespaces and are installed in separate
virtual environments at the reviewed revisions in
[ecosystem-modules.json](ci_tools/ecosystem-modules.json). TORQ consumes their
actual versioned file handoffs rather than importing their environments together.
An old geometry/energy-only TOPOS HDF5 file needs explicit state, units, atom
mapping and source method metadata; unknown source convergence stays unknown.
After verified TORQ results are downloaded, the interface can export an actual
electronic-energy artifact to BASE. Frequencies or spectra are distinct quantities.
See [the ecosystem contract](docs/development/ecosystem_contracts.md).

Runtime inputs, results, logs and scratch belong outside the source checkout,
normally in `~/CoChem_Artifacts`. Immutable result shards carry an inventory and
preserve failed/partial evidence; a failed calculation is never replaced with
invented successful data. Historical licensed/GPU/HPC adapters need their own
genuine site provisioning and scientific qualification.

## Development and license

Use the pinned Codespaces/Actions definitions for canonical environments.
[integrity.yml](.github/workflows/integrity.yml) separates software/mathematical
checks from mandatory genuine engine checks. Live Codespaces-to-Actions execution
and independently held-out scientific benchmarks are additional release evidence;
local test success does not establish either. The
[fabrication-policy audit](docs/development/fabrication_policy_audit.md) records
the inspected scientific paths, repaired defects and remaining evidence limits.

TORQ source is distributed under [Apache License 2.0](LICENSE). Third-party engine
and model licenses apply independently; no licensed engine is supplied by the
public classroom image.
