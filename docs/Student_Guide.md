# Student guide

Open the course repository in GitHub Codespaces and wait for its setup to finish.
Open the private forwarded port 8888, open `UI/Start_TORQ.ipynb` in JupyterLab,
and run its launch cell. Codespaces provides the interface. GitHub Actions
provides the canonical calculation environment; your instructor must approve the
repository's runner and available methods.

The calculation workflow must be merged onto the course repository's default
branch and enabled in Actions before submission. A local checkout or pull request
does not deploy that workflow. Your account needs Actions write access to submit
and read access to retrieve results. See the
[canonical environment instructions](development/canonical_environments.md)
for setup, the private Jupyter token and lifecycle checks. Opening the interface
does not run electronic calculations.

## Your first calculation

1. Paste one XYZ structure with coordinates in Å, or paste a complete TORQ request
   JSON. Set the molecular charge and spin multiplicity explicitly. The supplied
   geometry is an input structure until a converged optimization verifies it.
2. Choose a recipe and products. The HF/STO-3G teaching recipe illustrates the
   workflow with a small basis. It does not establish spectroscopic accuracy.
   Method availability and limitations appear with the selected recipe.
3. Select cores, memory and wall time within the course runner's limits. The
   public classroom worker permits 1–12 atoms from H through Ne, a closed-shell
   singlet, absolute charge at most 4, up to 2 cores, 2048 MB and 600 seconds. Select
   **Import and preview**. Rotate and tilt the mass-centered three-dimensional
   structure to check the actual atoms and geometry. Tabulated isotope masses
   determine this display frame; coincident principal moments make its orientation
   nonunique. The display does not infer bonds or certify an optimized structure.
4. Select **Check request** and read its report. Correct an invalid state or
   resource request. A product can be unavailable even when its earlier stages
   are supported. Choose a supported request or retain its earlier valid results.
5. Select **Submit to Actions**. Save the returned run ID and link. Submission
   uses the GitHub CLI authentication configured in your Codespace. Never paste
   a token into a notebook, request or source file.
6. Select **Refresh status** to read the actual run state. Select **Download
   results** after completion and inspect the result manifest and provenance.
   You can close Codespaces while Actions runs and resume using its saved run ID.

The notebook uses the same application validation, submission and result handling
as the CLI. Opening or previewing a molecule does not start an electronic-structure
calculation. Local scientific execution is an explicitly selected CLI activity;
the student interface always submits to Actions.

Changing the molecule, state, recipe, products or resources invalidates the
prepared request. Import and check it again before submitting the revised
calculation. A complete JSON request supplies its own method and resource fields;
the separate form controls are hidden in that input mode.

## BASE, TOPOS and TORQ

BASE provides the shared module handoff contract. TOPOS can provide candidate
structures and named method provenance. TORQ requires atom identities, coordinates
and units, charge, multiplicity and the actual method recipe. An older TOPOS HDF5
geometry/energy file does not supply all of that information: import it with
explicit state and provenance using the supported application importer. Do not
label its convergence as known when the source did not record it. Paste the
resulting request JSON into the interface; the same validation applies to all
input sources. Alternatively, choose **BASE handoff manifest** or **TOPOS HDF5**
in the interface, enter the local file path in the input box, and supply the
explicit import metadata in **Handoff**. The importer verifies the actual file
and preserves its complete source provenance.

After downloading a verified result, choose its **BASE source** and select
**Export energy to BASE**. This produces a checked electronic-energy artifact
for BASE's consumer. Export requires actual converged/stable electronic evidence
and verified final geometry; it carries the full method and uncertainty record.
This scalar electronic energy is distinct from frequencies, rotational constants
and spectral intensities. A result that lacks the required evidence reports why
export is unavailable.

BASE import metadata requires `molecule_id`, `charge`, `multiplicity` and
`atom_ids` in the source geometry row order. `isotope_mass_numbers` and
`repository_revision` are optional; omitting an isotope records it as unspecified.
The manifest comes from BASE's `cochem.module-handoff/1` exporter and must pass
its file-integrity checks.

TOPOS additionally requires `geometry_id`, `tier`, `geometry_unit`,
`method_provenance`, `energy_kind` and `reference_id`. The source method record
requires the actual `recipe_id`, `engine`, `engine_version`, `method`, `basis`,
`parameters` and their `recipe_sha256` digest. Obtain those fields from the
calculation's own record. Do not copy the newly requested TORQ recipe into the
source method record. Keep `source_convergence` as `unknown` unless the source
provides its actual convergence evidence. A claimed convergence status needs
the corresponding `convergence_evidence` file. Missing required information
stops import with an explicit error.

Start with [the water teaching request](../examples/student/water-hf-teaching.json)
to learn the input format. Its coordinates are a starting structure, not reference
experimental or calculated results.

## Reading scientific results

TORQ distinguishes electronic structure, optimized equilibrium geometry,
equilibrium rotational constants, harmonic analysis, anharmonic force fields,
resonance analysis, VPT2, vibration-corrected constants and spectral catalogs.
Every stage needs its own evidence and provenance. If a later stage fails,
scientifically valid earlier stages remain available.

Equilibrium constants from a validated equilibrium structure are **Bₑ**. Obtaining
**B₀** requires qualified vibrational corrections. A harmonic Hessian alone cannot
provide general VPT2 or B₀. A rigid-rotor teaching catalog omits effects that can
matter for laboratory or astronomical identification; only a qualified
identification product can support that claim.

An Actions `completed` status describes workflow execution. Read the scientific
result's stage status separately. `partial`, `unavailable` and `failed` carry
different meanings; none silently supplies missing frequencies, corrections,
dipoles or uncertainties. Requesting exact open-source revDSD currently returns
its qualification limitation until the independent recipe/derivative milestones
pass. Unsupported data are never substituted with another method under the same
label.

The [release scope audit](development/student_release_scope.md) maps the complete
SRS to implemented contracts and outstanding scientific/deployment evidence.
Optional local research profiles have separately bounded support for DFT-D4, MP2
and small HF anharmonic force fields. Inspect `cochem-torq recipes`, then use
`cochem-torq validate --request /path/to/request.json --execution local_validation`
and `cochem-torq execute --request /path/to/request.json --output-dir /path/to/new-output`
only when your instructor selects that environment. Local results do not qualify
the public Actions deployment or establish B₀/identification accuracy.

## Troubleshooting

- **No imported request:** import the geometry before checking or submitting.
- **Invalid XYZ:** supply an integer atom count, a comment line and exactly that
  many symbol/x/y/z rows. All coordinates must be finite.
- **Unknown isotope:** use a supported element symbol or an explicit mass number
  such as `13C`. Elements without a natural-abundance default need an isotope.
- **State or resource rejection:** review charge, multiplicity and your approved
  runner's limits in the validation report.
- **Unavailable method/product:** read the reported qualification gap. Select a
  qualified earlier stage, or ask your instructor about the appropriate backend.
- **GitHub authentication/access error:** check `gh auth status` and repository
  access in the terminal. Your account needs permission to dispatch the course
  workflow and read its artifacts.
- **No run ID:** save the submission report. Existing run IDs can be entered
  directly in a later session; submitting again creates a new job.
- **Missing/expired artifacts:** open the real Actions run link and inspect its
  logs and retention status. A failed calculation needs its actual diagnostic,
  not a synthetic replacement result.

Include the request ID, run URL, recipe, requested product, result stage status
and error message when asking an instructor for help. Do not include credentials.
