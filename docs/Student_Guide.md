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

An installed modern TORQ module also supplies `cochem-torq-ui --workspace
/path/to/private-interface --port 8888`. It copies the packaged notebook into a
separate workspace, preserves existing files and starts authenticated JupyterLab.
Use the executable from TORQ's own module environment. BASE, TOPOS and the modern
TORQ distribution contain overlapping shared package paths, so the modern TORQ
student module must stay in its separate environment. The mandatory TOPOS kit's
compatibility importer remains distinct from this student calculation interface.

The current `57c81eb` candidate passed all 2486 local repository tests with zero
failures/errors/skips, plus selected installed-wheel, real candidate-refinement
CLI and authenticated notebook checks. The [recorded results](development/release_candidate_check_results.json)
qualify that local scope. Hosted jobs were blocked before execution
by billing, the default-branch calculation endpoint returns 404, and Codespaces
machine access returns 403; no live Codespace was provisioned. The canonical
student journey remains unverified. See the
[readiness report](development/release_candidate_readiness.md) before starting.

## Your first calculation

1. Paste one XYZ structure with coordinates in Å, or paste a complete TORQ request
   JSON. Set the molecular charge and spin multiplicity explicitly. The supplied
   coordinates are an input geometry. A converged optimization produces
   model-dependent stationary coordinates; minimum classification also requires
   the recorded harmonic and electronic-state checks.
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
4. Select **Review plan** and read its report, tasks, scientific warnings and
   calculation budget. Correct an invalid state or
   resource request. A product can be unavailable even when its earlier stages
   are supported. Choose a supported request or retain its earlier valid results.
5. Enter your name or course identity in **Approved by**, then select
   **Approve this plan**. Approval applies to this exact request and its declared
   calculation budget and expires after one hour. It does not certify scientific
   accuracy or grant GitHub permissions. **Submit to Actions** becomes available
   after explicit approval. Save the returned run ID and link. Submission
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
prepared request and approval. Import, review and approve it again before submitting the revised
calculation. A complete JSON request supplies its own method and resource fields;
the separate form controls are hidden in that input mode.

To cancel a run created through this interface, enter its run ID and a reason,
then select **Cancel owned run**. TORQ verifies the actual local submission
receipt against the hosted run before requesting cancellation. Cancellation is a
workflow state; the downloaded scientific stage records describe retained results.

After downloading an incomplete verified result, choose it in **BASE source**
and select **Prepare new attempt**. This checks the shard and source compatibility,
creates a new request UUID, and retains the earlier attempt's provenance. It does
not claim reuse of an engine checkpoint. Review and explicitly approve the new
attempt; unsupported recipes remain blocked. Submission retries for an unchanged
approved request retain its request UUID and idempotency key.

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

## Continue from an external reviewed TOPOS ensemble

Choose **Reviewed TOPOS ensemble JSON** in the notebook. Enter the absolute
Python executable path in your separately installed modern TOPOS environment.
Upload the ensemble JSON, or enter its existing local path. Select **Inspect
reviewed TOPOS ensemble**, choose the exact member, then select your new TORQ
recipe, products and budget and **Import and preview**. Changing the file,
producer interpreter or member clears the prepared request and approval.

The actual installed producer verifies the complete source record, review chain,
ensemble and member identities in a separate process. TORQ retains the original
JSON bytes, producer observation, fresh request and content manifest under the
private `topos-imports` directory next to your downloads. The new request preserves
coordinates, atom IDs, isotope declarations, charge and multiplicity. Its source
provenance separately identifies the original requested method, observed engine
attempt, matrix row and source digests. Keep the complete import directory for
reproducibility; its compact request does not embed the entire original ensemble.

The equivalent CLI entry is:

```bash
cochem-torq import-topos-reviewed \
  --handoff /path/to/reviewed-ensemble.json \
  --producer-python /path/to/topos-environment/bin/python \
  --member EXACT_MEMBER_ID --recipe hf-sto-3g-education \
  --product geometry --cores 1 --memory-mb 2048 --wall-seconds 600 \
  --output-dir /path/to/new-import
cochem-torq plan --request /path/to/new-import/request.json --output /path/to/plan.json
```

Add `--expected-handoff-sha256` when selecting an already recorded handoff identity.
The importer rejects stale identities, changed source records and existing output
directories. Import starts no calculation and returns no consumption receipt.
Review and approve the new plan through the ordinary student pathway.

This is a geometry entry point. Original topology, fragment definitions,
environment, constraints, energies and derivative data remain observations in
the retained source. They do not become target calculation settings or verified
TORQ results. The new recipe's capability and chemistry checks still apply.
Native wavefunction checkpoint continuation uses the separately qualified API;
arbitrary external Hessian, force-field or spectroscopy files require their own
typed evidence adapter and are not accepted as computed stages by this importer.

## Reading scientific results

The notebook's candidate table retains exclusions and their reasons. Use
**Record input candidate**, **Exclude candidate** and **Restore candidate** to
make reversible selection decisions. Candidate history distinguishes supplied
inputs from verified calculated geometries; missing energy or model uncertainty
stays unavailable. A selection change clears approval and requires a new plan
review. An excluded or quarantined current candidate cannot be submitted.

Headless candidate inspection and selection are available with
`cochem-torq candidates --help`. These records do not prove that a conformer
search is complete, and exclusions do not automatically prune calculations.

For a reproducibility bundle, use
`cochem-torq export RESULT_SHARD --format publication --destination NEW_BUNDLE`.
It preserves the authentic result, native files, source snapshot and limitations.
Its eligibility remains exploratory, partial or failed evidence. Packaging does
not establish peer-review readiness or identification accuracy.

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
  An unavailable explicitly requested isotope stays unavailable; its mass is
  never replaced by an ordinary atomic weight or its integer mass number.
- **State or resource rejection:** review charge, multiplicity and your approved
  runner's limits in the validation report.
- **Unavailable method/product:** read the reported qualification gap. Select a
  qualified earlier stage, or ask your instructor about the appropriate backend.
- **GitHub authentication/access error:** check `gh auth status` and repository
  access in the terminal. Your account needs permission to dispatch the course
  workflow and read its artifacts.
  In Codespaces, the native GitHub CLI uses your stored account login by default
  rather than the injected Codespaces session. `COCHEM_PRIVATE_GH_AUTH` accepts
  `auto`, `stored-cli` or `environment`; an explicit environment selection uses
  injected account authentication. Actions always retains its owning project's
  job authentication. The software never reads stored credential values.
  BASE's separate installed TORQ interface submits to the student's private
  project `calculation.yml`. The approved installed scientific code digest must
  match the catalog-pinned native worker. A plan created directly from a TORQ
  Git checkout retains that checkout's commit authority and cannot be silently
  reassigned to a different BASE controller commit. Prepare and approve through
  the verified separately installed interface for the BASE project pathway.
- **No run ID or uncertain submission:** save the complete submission report and
  retain its request UUID and idempotency key. Inspect the saved receipt and the
  UUID-bearing Actions run before retrying. An unchanged retry returns its
  existing receipt without dispatching another job. Create a new request and
  approval only when you intend a separate calculation.
- **Missing/expired artifacts:** open the real Actions run link and inspect its
  logs and retention status. A failed calculation needs its actual diagnostic,
  not a synthetic replacement result.

Include the request ID, run URL, recipe, requested product, result stage status
and error message when asking an instructor for help. Do not include credentials.
