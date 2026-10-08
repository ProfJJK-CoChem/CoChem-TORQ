# Source-bound benchmark and calibration workflow

`cochem_torq.benchmarking` implements reference-byte verification, an immutable
accepted design, prediction sealing, numerical scoring, and conservative grouped
split-conformal diagnostics. It does not supply molecular observations, establish
reference authorship, choose an accuracy target, or qualify a scientific recipe.
The [existing proposal](../../benchmarks/rotational-identification/preregistration.json)
remains proposed, unaccepted, and without curated reference values.

The implemented sequence is:

1. Curate actual primary reference artifacts and their exact observable identities.
2. Retain independently trusted reference-manifest and curator-record digests.
3. Accept explicit targets, recipes, code identity, partitions and protocol.
4. Freeze that accepted design without opening held-out reference values.
5. Execute new requests that bind the exact freeze digest.
6. Export and verify each calculation's genuine publication bundle.
7. Seal all selected predictions, including explicit failures/unavailable stages.
8. Open the matching curated references and score the sealed evidence.

Every artifact destination is immutable. Repeated publication cannot overwrite a
previous freeze, prediction seal or score. Private files use mode `0600`; new
directories use `0700`. Paths through symbolic links are rejected. Publication
uses a private fsynced temporary file and a no-overwrite atomic hard link.

## Command-line operations

The student CLI exposes `benchmark-references`, `benchmark-freeze`,
`benchmark-seal`, and `benchmark-score`. Run each command with `--help` for its
mandatory file paths and independently retained digests. These operations verify
supplied evidence; they do not generate curator attestations, accept scientific
targets, or supply absent measurements.

`benchmark-freeze --design accepted-design.json --acceptance-record acceptance.json
--output freeze.json` opens the externally accepted metadata and original
acceptance bytes without loading reference values. `benchmark-seal` takes that
freeze, its trusted SHA256, and a versioned bundle-location JSON map. Its
`schema_version` is `cochem.torq.prediction-bundle-locations/1`; each entry in
`locations` contains `reference_id`, `path`, and `manifest_sha256`. Relative paths
are resolved relative to the location-map file, and duplicate reference IDs are
rejected. Every selected non-anchor requires genuine complete or explicitly failed
publication evidence.

`benchmark-references` and `benchmark-score` require the original reference
manifest, original curator attestation, both independently retained digests, and
a source-location map. That map has `schema_version`
`cochem.torq.reference-source-locations/1` and a `locations` array containing exact
`source_sha256` and `path` fields. Every inventoried raw source is required, and
additional or duplicated source digests are rejected. Input paths through symlinks
are rejected. Verification checks the supplied attestation and bytes; it cannot
establish the source's authorship by itself.

Scoring additionally requires the prediction seal and its independently retained
digest. The CLI checks the seal's bytes and freeze identity before opening
reference sources. The scoring service then rechecks native publication evidence,
accepted design, actual reference bytes, and chronology. Freeze, prediction seal,
and score outputs are new immutable artifacts. A successful command is an artifact
operation, not independent accuracy qualification.

## Reference curation

`ReferenceManifest` binds every `ReferenceDatum` to an inventoried raw source
artifact. Each source requires a citation, DOI or HTTPS locator, exact version,
retrieval time, source-byte digest, reuse permission, and measured/fitted/
semi-experimental origin. Each datum records its location within that source,
exact units, parent/family identity, canonical atom order and explicit isotope
numbers, charge/multiplicity, conformer, electronic/vibrational/tunneling state,
Hamiltonian convention, observable and component.

An unavailable measurement uncertainty remains `None`. A positive standard
uncertainty must be explicitly declared in the observable's unit. Covariance is
currently either unavailable or inapplicable; this service does not infer a
covariance matrix or perform a covariance-aware difference comparison. Catalogue
predictions are rejected as independent observed references. An equilibrium
rotational reference requires semi-experimental origin and independently reviewed
correction provenance; a fitted ground-state constant cannot become an equilibrium
constant by changing its label.

`import_curated_references` takes the actual manifest bytes, all inventoried raw
source bytes, the actual curator attestation bytes, and independently retained
digests. It checks byte inventory, datum/source binding, aware timestamps, curator
binding, and source retrieval before review. Duplicate JSON keys are rejected.
`VerifiedReferences` retains the original immutable bytes; scoring verifies them
again against the frozen design, including when a caller constructed that Python
object directly. A future curator review is rejected, and the frozen acceptance
must follow completion of that reference review.

A curator must independently inspect extraction, attribution, molecule/state
identity, origin, independence and permission. An internally consistent checksum
does not prove those facts. The software checks the supplied external attestation;
it does not generate an experimental review or validate a publisher signature.

## Accepted design and leakage controls

`BenchmarkDesign` requires the exact proposal/reference/curation/recipe/code
digests, reviewed source commit, protocol, selected reference metadata, accepted
numerical targets with rationale, nominal coverage and an exchangeability review.
There are no default accepted error limits. Every enabled observable needs one
positive `maximum_absolute_error` in the same unit as its selected references.

All states, transitions, conformers and isotopologues of a parent remain in one
family and partition. Families cannot cross development, calibration, held-out
or anchoring partitions. A shared raw source artifact cannot cross partitions.
Calibration and held-out partitions are both required. Anchoring references are
excluded from prediction scoring.

`freeze_benchmark` requires an actual acceptance record whose bytes match its
declared digest and whose `design_sha256` binds the complete design excluding
only `acceptance_record_sha256`. This exclusion avoids a circular digest. The
record also binds `accepted_by`. Its purpose is an external acceptance record,
not authorization created by the application. The freeze stores the original
acceptance bytes and checks them on readback.

After a target, recipe, model or selection changes, create a new versioned design.
Any held-out subject inspected during tuning must become development data for
the revised campaign. The service prevents its own scoring call from preceding
prediction sealing; it cannot prove that a person never inspected reference data
elsewhere. That obligation belongs in the external preregistration review.

## Genuine prediction import

`import_publication_prediction` verifies an actual TORQ publication bundle against
its independently retained manifest digest, including its shard, native artifact
inventory, stage payloads and packaged analysis sources. It checks atom order,
isotope numbers, charge and multiplicity against the original request. Chemical
labels must already be sealed in `request.source_provenance.benchmark_identity`;
the importer cannot establish a conformer's scientific identity from a label.

The current scalar extractors support total electronic energy and each defined
equilibrium rotational constant. Available observables require retained native
engine evidence. Stationary/nonminimum constants cannot be relabeled `Be`.
An undefined linear-rotor axis or unavailable `B0` remains an explicit absence.
Available ground-state constants, measured-transition assignments and relative
intensity products require their own validated extractors before import.

`seal_predictions` accepts one actual verified bundle for every selected
non-anchor reference. The original request must contain
`source_provenance.benchmark_freeze_sha256` matching this freeze. Recipe and
scientific code digests must match. A recorded non-null source commit must also
match; an installed distribution whose engine receipt records no Git checkout
uses its exact scientific code digest with the externally reviewed design commit.
Experimentally anchored predictions cannot receive absolute independent
calibration or held-out scoring.

The seal retains bundle locators and manifest digests. Scoring re-verifies every
actual bundle and compares the complete extracted prediction against the seal.
Changing a value and recomputing only the seal digest cannot replace native
evidence. Scalar import verifies the full publication inventory both before and
after reading its request, result and summary, rejecting persistent byte changes
during extraction. This is a readback guard, not a filesystem snapshot or a
cryptographic signature of an author. Keep the original publication bundles
accessible and unchanged during
scoring and retain them with any exported research record. A score JSON file is
not a self-contained deposit of every native/reference artifact.

## Numerical diagnostics and finite-sample calibration

Residuals use `prediction - reference`; relative residuals use that signed
residual divided by the reference, in ppm. A standardized residual is divided
only by the supplied *reference measurement* standard uncertainty. It is not a
combined model/experiment z-score and cannot establish theoretical uncertainty.
Missing observations stay in the expected-case denominator and failure fraction.

The service reports parent-weighted MAE and RMSE, maximum absolute residual,
parent/datum counts and missing-case fractions separately by partition and
observable. A parent with many transitions receives the same parent weight as a
parent with few transitions. No observed errors means unavailable metrics,
not zero error.

For simultaneous group diagnostics, each residual is divided by its frozen
observable target scale. All related members of a family form one sampling unit.
Families sharing source artifacts are joined transitively into a conservative
sampling group. The group score is the maximum normalized absolute residual.
Incomplete groups cannot provide a complete calibration or held-out coverage
score; missing cases also prevent an observed-target pass.

For `n` independent exchangeable calibration groups and nominal coverage `c`,
split conformal uses the **exact order statistic**

`k = ceil((n + 1) * c)`.

The threshold is the `k`-th sorted group score. It does not use an interpolated
quantile. If `k > n`, finite coverage is unavailable and the threshold is `None`;
the program does not substitute the maximum score or a fabricated bound. At 95%
nominal coverage, at least **19 independent exchangeable calibration groups** are
needed even to produce a finite mathematical threshold. That is a mathematical
minimum, not a sufficient sample size for a defensible transfer/accuracy claim.

The current proposal's three calibration parents are all halomethanes, so they
provide at most **one chemical-family calibration unit**, before any additional
source-dependence grouping. Its three held-out parents occupy two families.
Counting individual lines, isotopologues or related parents as independent units
would misstate the evidence. Selecting different chemical families for calibration
and testing also does not establish exchangeability; a transfer claim needs an
external domain/measurement/model assessment or a different preregistered sampling
design.

Outputs separate mathematical thresholds, descriptive observed group coverage,
insufficient sample size, missing observations and external exchangeability
assessments. They do not invent an empirical confidence interval, infer latent
error-free reference values, certify general coverage, or activate a recipe.
`experimental_accuracy_established` and `identification_ready` remain false.
Observed held-out targets passing is a bounded numerical comparison that still
requires independent scientific acceptance.

## Validation and remaining research acceptance

The module's tests exercise exact mathematical order statistics, parent weighting,
family/source leakage rejection, immutable real files, strict identity/unit/
uncertainty checks and genuine HF/STO-3G energy/optimized-geometry/Hessian bundle
import. Mathematical metadata in those tests are explicitly contract exercises;
they are not molecular measurements, curated references or scientific acceptance.
Additional parser checks reject the actual unaccepted proposal as a curated
reference dataset, duplicate manifest members and malformed directly constructed
reference objects. A genuine bundle corruption check and a static guard-position
check verify failure handling; no timed concurrent-mutation experiment is claimed.

No experimental benchmark is recorded as completed by this implementation. An
attempt to retrieve the authoritative NIST WebBook hydrogen reference was denied
by the environment's network proxy. Reference curation therefore remains pending.
To qualify a campaign, complete external reference review and target acceptance,
freeze a scientifically defensible design, execute the exact qualified recipes,
seal authentic predictions, run scoring, and independently review all failures,
domain assumptions and calibration limitations. Keep reference/source/native
artifacts and all amendments with the publication evidence.
