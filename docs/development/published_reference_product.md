# Genuine published-value comparison product

**Latest access/source update:** [the new source-acquisition note](reference_acquisition_20261009.md)
and [agent handoff](AGENT_HANDOFF.md) record genuine successful NIST WebBook,
CCCBDB and QCArchive information requests, with an independent HTTP 200/exit-zero
recheck on 9 October 2026 UTC. Their earlier denied requests remain historical.
Nine unqualified source candidates are retained; zero accepted benchmark records
or identification qualifications were added. The live API information response
is server metadata, not a molecular calculation record.

Student deployment is **on hold**. The base product can retrieve bounded public
scientific sources, parse original QM9 records, execute a genuine small-molecule
calculation, verify its native publication bundle and compare individual results
with attributed published values. This does not complete the research SRS or
qualify an identification method. The [researcher context](agent_and_researcher_context.md)
separates remaining agent implementation from actual access, authority and
experimental prerequisites.

The [machine-readable acquisition inventory](published_reference_acquisition.json)
records source classes, actual hashes, retained counts, observed denied hosts and
qualification limits. An HTTP success, a hash match and a scientific acceptance
are separately recorded facts.

## Retrieved source material

Ramakrishnan, Dral, Rupp and von Lilienfeld, *Quantum chemistry structures and
properties of 134 kilo molecules*, *Scientific Data* **1**, 140022 (2014),
[DOI 10.1038/sdata.2014.22](https://doi.org/10.1038/sdata.2014.22), supplies the
peer-reviewed source for the included QM9 records. The publisher's current
[dataset metadata](https://api.figshare.com/v2/articles/1057646) declares CC0 and
[dataset DOI 10.6084/m9.figshare.978904_D12](https://doi.org/10.6084/m9.figshare.978904_D12).
The original README is distributed separately as
[10.6084/m9.figshare.978904_D7](https://doi.org/10.6084/m9.figshare.978904_D7).

The complete 86,144,227-byte compressed archive was streamed once. Its observed
SHA-256 is `3a63848ac80691bdb8d41834b575afad345b9300d7a2db0c38adb7f6eaa8360c`;
its MD5 matches publisher metadata. Only the first 20 original XYZ members were
retained. [The source index](../../benchmarks/published-values/qm9-samples/source-index.json)
binds each retained member and the original README/API metadata to actual bytes.
The water packet contains those original water bytes, six extracted values,
source locators, a crawler's review and independently retained digest pins.
Its review explicitly declares `human_reviewed=false` and
`independent_curation_completed=false`.

| Water quantity | Published QM9 value | Original unit |
|---|---:|---|
| Equilibrium A | 799.58812 | GHz |
| Equilibrium B | 437.90386 | GHz |
| Equilibrium C | 282.94545 | GHz |
| Harmonic frequency rank 0 | 1671.4222 | cm⁻¹ |
| Harmonic frequency rank 1 | 3803.6305 | cm⁻¹ |
| Harmonic frequency rank 2 | 3907.698 | cm⁻¹ |

These are **theoretical** values at B3LYP/6-31G(2df,p). The acquired original
XYZ/README does not fully specify isotope masses, explicit state/multiplicity,
Gaussian version, B3LYP variant or integration grid. Unreported fields stay
unavailable. Matching sorted harmonic ranks does not establish physical mode
assignment. QM9 `U0` includes zero-point energy; the parser exposes `U0 - zpve`
only as an explicitly derived rounded-data proxy, never as an original published
total electronic energy.

The [experimental B₀ packet](../../benchmarks/published-values/experimental-b0-water-formaldehyde/)
contains the original supporting information to Piccardo, Penocchio, Puzzarini,
Biczysko and Barone, [10.1021/jp511432m](https://doi.org/10.1021/jp511432m)
(2015), its exact PDF text extraction, publisher metadata and six original
experimental ground-state constants reproduced in Table 1, in MHz:

| Parent species | A₀ / MHz | B₀ / MHz | C₀ / MHz |
|---|---:|---:|---:|
| H₂O | 835839.10(13) | 435347.353(27) | 278139.826(57) |
| H₂CO | 281970.5578(61) | 38833.98715(31) | 34004.24349(31) |

The parenthesized uncertainty notation is retained exactly. Its confidence
convention and covariance were not established, so it is not asserted to be a
standard uncertainty. The cited original measurements are Matsushima et al.,
*J. Mol. Struct.* **352**, 371–378 (1995), and Brünken et al.,
*Phys. Chem. Chem. Phys.* **5**, 1515–1518 (2003); their full article bytes were
not retrieved. Isotope and Hamiltonian/reduction details remain unresolved.
This is a source-bound peer-reviewed secondary reproduction, with CC BY-NC 4.0
attribution/licence retained. It does not accept the independent benchmark.
The strict importer actually replayed the original PDF extraction and verified
all six numeric tokens. Use its `derived-texts.json` as well as its raw-source
map when importing. The baseline HF water example has no B₀ prediction;
equilibrium Bₑ cannot substitute for these experimental B₀ observations.

The [primary 2-chloroethanol packet](../../benchmarks/published-values/chloroethanol-primary-ground-state/)
retains original author-published ground-state `.par`, `.res` and `.int` files
for the chlorine-35 and chlorine-37 isotopologues, from Bunn et al.,
*A High-Resolution Rotational Analysis of 2-Chloroethanol (HOCH2CH2Cl) and Its
First Vibrationally Excited State*,
[10.1021/acsearthspacechem.6c00221](https://doi.org/10.1021/acsearthspacechem.6c00221)
(2026). The original publisher scientific-data archive was genuinely retrieved
and matched its publisher MD5 and observed SHA-256
`2de98bdd28827c83fc28a47c74cdb420dfb4224af8face85796cfcee45de247e`.
The six complete fitting files were replayed byte for byte against the original
ZIP; that ZIP remains outside the bounded packet. The source is scientific
fitting/calculation output, with CC BY-NC 4.0 attribution retained.

| Ground-state chlorine isotope | A0 / MHz | B0 / MHz | C0 / MHz |
|---|---:|---:|---:|
| 35Cl | 12747.96193(20) | 3505.789127(62) | 2981.257712(49) |
| 37Cl | 12705.91773(17) | 3423.536116(58) | 2919.494355(49) |

These are primary experimentally fitted ground-state constants, in the
publisher's explicitly stated A reduction and `I^r` representation of the
single-state distorted rotor Hamiltonian. Chlorine isotopologues and the
ground/excited vibrational-state distinction are explicit. The chosen `.res`
section is “PARAMETERS IN FIT WITH STANDARD ERRORS ON THOSE THAT ARE FITTED,”
with “Ndegf=Nlines-Nconst” statistics. The earlier Nlines table has different
rounding/errors and is retained separately; `.par` control-field numbers are
not reported measurement uncertainties. Printed fit errors and rounded A/B/C
correlations stay source facts. No universal standard uncertainty, confidence
level or full-precision covariance is inferred. Light-atom isotope masses,
electronic-state and conformer/tunneling identity remain unconfirmed.

A separate agent checked all original archive members, the six selected
literals and unchanged strict importer. That importer actually loaded all six
typed descriptive records; it did not calculate a TORQ B0 prediction or pass an
accuracy benchmark. The publisher reports fitting challenges and a larger-than-expected fit RMS
(`sigma_fit`) up to 0.060 MHz. These limitations remain in the packet. Comparison requires a
genuine computed B0 in matching Hamiltonian conventions; equilibrium Be cannot
substitute. The preserved archive's CCSD(T)/cc-pCVTZ VPT2 outputs also retain
resonance and force-field-consistency warnings for future agent parser/method
work; their presence does not qualify TORQ's VPT2.

Direct NIST WebBook, CCCBDB, HITRAN documentation, live QCArchive, Crossref,
arXiv, Nature, Europe PMC, JPL, RSC, Semantic Scholar, CDMS and ScienceDirect
attempts encountered actual proxy CONNECT 403 denials. No denied origin response or missing constant was substituted. An
independently hosted NIST-authored MATS repository yielded 11 original CO line
candidates, but not eligible equilibrium/ground-state constants: primary
measurement citations, state, uncertainty and third-party reuse details remain
unresolved. These candidates do not enter the accepted benchmark.

**QCSchema is an interchange schema. QCArchive is a calculation repository.**
The official MolSSI tutorial mirror provided a historical NH₃ B2PLYP record and
its printed SCF/OS/SS/total components. Genuine density-fitted PySCF and TORQ's
existing conventional evaluator were run against those components. See
[the component package](../../benchmarks/published-values/qcarchive-nh3/).
That exercise does not establish exact revDSD, derivatives or experimental
spectroscopic accuracy. [The revDSD source ledger](revdsd_component_recipe_ledger.md)
keeps original D3(BJ) sample coefficients separate from official D4 dispersion
parameters and retains unresolved D4 electronic/GKS definitions.

The [primary method/source follow-up](release_candidate_evidence/final-audited-published-reference-20261008/primary-method-followup/revdsd_primary_and_upstream_followup.md) retains genuine 2025 publisher supporting information, DOI **10.1021/acs.jpca.5c01035.s001**, and a pinned official PySCF-forge implementation review. The later D4 tuple is distinct from the original-2019 target; it does not authorize substitution. Upstream GKS-response/gradient source was reviewed, not executed or independently qualified. Exact target identity, compatibility, moving-grid response, frozen-core conventions and higher derivatives remain agent work. The CC BY-NC 4.0 excerpt keeps its original attribution; unlicensed author-repository files are not redistributed, and no recipe or acceptance gate is activated.

## Run the included water example

Use a Python 3.12 checkout and an isolated environment. The repository's
hash-locked core and PySCF requirements are the tested installation route:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install --require-hashes -r ci_tools/requirements-core-py312.txt
.venv/bin/python -m pip install --require-hashes -r ci_tools/requirements-pyscf-py312.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m pip check
```

From the checkout, run the genuine calculation and all six comparisons with a
new output directory on a filesystem that meets the scientific disk reserve:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 COCHEM_DISABLE_SANDBOX_CHECK=1 \
  .venv/bin/python -m scripts.run_qm9_water_comparison \
  --output-dir /tmp/cochem_exec_qm9_water_first_run
```

The documented sandbox override bypasses the historical launch-directory-name
check only. Scientific resource, data integrity and method qualification checks
remain active. Use a different, previously nonexistent output directory for
each run; the program does not overwrite previous evidence. The original request
allows one core, 1024 MB and 120 seconds. It requests optimized geometry,
harmonic analysis and equilibrium rotational constants. The source starting
geometry is retained and converted explicitly from ångström to bohr.

When using an installed wheel outside the checkout, pass `--example-dir` with
the path to the actual checked-out `benchmarks/published-values/qm9-water`
packet. Source data are retained in the repository; the driver does not create a
replacement packet when those files are absent.

The driver verifies the bundled request/reference/review pins before launch,
executes the genuine HF/STO-3G educational recipe, exports and verifies native
publication evidence, then writes six descriptive comparison records and a
summary. It retains actual failures. It creates no curator, acceptance record,
missing frequency, uncertainty, engine version or benchmark pass.

Inspect `summary.json` for `status`, `native_status`,
`publication_manifest_sha256`, `reference_identity_quality_flags` and the six
`comparisons` entries. Each entry records its reference ID, observable/unit,
source value, prediction value, residuals and method/basis/geometry context.
`comparisons/01.json` through `06.json` retain the detailed source-bound records;
`publication/` retains the verified native bundle and `run.log` the actual
execution log. `complete_descriptive_comparison` means those operations
completed; scientific qualification remains false.

HF/STO-3G and QM9 B3LYP/6-31G(2df,p) differ. The receipts report method, basis
and geometry differences, unreported source isotope/state information and the
limited harmonic rank correspondence. A small residual for one quantity cannot
qualify the different protocols. Larger residuals are visible rather than
filtered from the output.

## Fetch and compare additional genuine sources

`python -m cochem_torq reference-fetch` uses HTTPS with the inherited proxy and
normal CA verification. It requires a citation and a declared reuse status,
retains the actual response bytes, retrieval status, SHA-256, size and media type,
and supports an independently retained expected digest. It rejects embedded
credentials, credential-bearing query parameters, non-HTTPS URLs, linked paths,
existing destinations and oversized responses. A failed request remains a failed
request. A digest mismatch quarantines the actual response as unverified bytes.

For example, the original publisher README was genuinely fetched and verified
using:

```sh
python -m cochem_torq reference-fetch \
  --url https://ndownloader.figshare.com/files/3195392 \
  --citation 'QM9 README, 10.6084/m9.figshare.978904_D7; paper 10.1038/sdata.2014.22' \
  --reuse-permission 'CC0 as declared by publisher article 1057641' \
  --expected-sha256 0ee83fc21faa9527ebe89fad0ce8f2bd2da1cbd35a3c7edc199f4ad21d31d97a \
  --output-dir /tmp/qm9_readme_new_retrieval
```

The declaration is retained provenance, not an automated legal assessment or
publisher-signature verification. Inspect the original publisher's terms and
do not relabel third-party material with an assumed license.

`python -m cochem_torq reference-compare --help` lists the exact inputs for one
datum: pinned original manifest/review, all raw source locations, reference ID,
an authentic publication bundle with its trusted manifest digest, and a new
output path. The comparator rechecks original bytes, quoted scalar literals and
numeric positions. QM9-specific checks bind rotational A/B/C columns and harmonic
ranks. A derived PDF extraction requires its original PDF digest, exact tool
version/arguments and an actual repeat of the specified extraction; it is not
accepted merely because text happens to have a matching declared checksum.

To compare one water datum again using the actual completed example bundle,
read its retained manifest digest from the original run summary. Run the
following from the checkout after the water example succeeds, with a new output
file. The digest binds that actual native bundle; it does not create scientific
acceptance.

```sh
set -e
torq_publication_manifest_sha256=$(
  .venv/bin/python - <<'PYTHON'
import json
import re
from pathlib import Path

summary = json.loads(
    Path('/tmp/cochem_exec_qm9_water_first_run/summary.json').read_text()
)
assert summary['status'] == 'complete_descriptive_comparison'
value = summary['publication_manifest_sha256']
assert isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value)
print(value)
PYTHON
)
.venv/bin/python -m cochem_torq reference-compare \
  --reference-manifest benchmarks/published-values/qm9-water/references.json \
  --review benchmarks/published-values/qm9-water/automated-review.json \
  --sources benchmarks/published-values/qm9-water/source-locations.json \
  --reference-id qm9-water-Be-A \
  --expected-reference-manifest-sha256 65c5280b4e3b5e0a644f94ee304cadcf71328de2daeba930587a88631a8eabaa \
  --expected-review-sha256 135b9a58c04a3779d8cb01da6a39c249835f4f88d9d72cf896196b51eaf1df80 \
  --publication-bundle /tmp/cochem_exec_qm9_water_first_run/publication \
  --expected-publication-manifest-sha256 "$torq_publication_manifest_sha256" \
  --output /tmp/qm9_water_A_new_comparison.json
```

A missing summary, incomplete native run or malformed digest fails the command
sequence before comparison. Do not substitute a digest from a different run.
Use the original completed run's paths if you chose another output directory.
The expected reference/review digests are the source packet's retained pins.

The bundled experimental B0 packet additionally requires the real `pdftotext`
executable and recorded Poppler version **25.03.0**. Check `pdftotext -v` before
importing that packet. The strict importer repeats
`pdftotext -layout -enc UTF-8` and requires the recorded tool version and exact
extracted bytes. The Python dependency locks do not install this system
executable. An unavailable or different extractor remains an explicit
prerequisite failure; the QM9 water example does not require it.

Descriptive residuals use `prediction - reference`; relative residuals use
`(prediction - reference) / abs(reference)`. A zero reference gives an explicitly
unavailable relative residual. An unavailable standard uncertainty remains
`null`. The output does not set an accepted accuracy limit or calibrate a
prediction from its discrepancy.

## Experimental qualification remains a separate task

The [preregistered benchmark workflow](benchmarking.md) requires original
observations with exact molecule/isotopologue/state/Hamiltonian conventions,
traceable corrections for semi-experimental equilibrium constants, an externally
accepted bounded design, grouped calibration/held-out partitions, sealed genuine
predictions and independent scientific review. Automated descriptive extraction
cannot impersonate an external curator or reviewer. Research targets and
uncertainty must be justified for the actual molecular domain and observable.
Source acquisition, extraction, agent curation, independently derived solver
comparisons and preparing a concrete benchmark design remain agent work.
A required named person's review or retained target acceptance must come from
that authority; the word “independent” does not by itself designate a task as
human-only. The context identifies those conditional authority boundaries.

The current full SRS audit remains blocked. Remaining revDSD response and higher
derivatives, resonant GVPT2, high-accuracy identification corrections and domain
calibration are agent-executable research/implementation work, with their
scientific evidence gates retained. They are not assigned to the user as manual
literature searches. Direct source acquisition can resume after the actual
network policy allows the observed denied hosts. Student deployment stays on
hold until the user resumes it and the product satisfies its intended scope.

The [bounded benchmark proposal](bounded_preregistration_proposal.md) now supplies concrete candidate domains, source criteria, target definitions, resource planning and tradeoffs. It recommends a prospective split amendment because inspected QM9 HCN/acetonitrile data cannot remain blind nitrile holdouts. It is a proposal, not an accepted or executed benchmark; the available family counts cannot support a finite 95% calibrated interval.

## Completed local software validation

The latest complete software regression ran in canonical GitHub Actions for exact commit `d43b8aa3eefc4768f2d52857cd3945677ed3a794`, source hash `164926d7cb0778a7138398cd85ba8a78906da237f70946ea02cfc15b048e7902` and application hash `680f1415ee5e10867709c8d6c67f26f169c2c098d16b81ef0a6ef2331b65e1be`. [The original-artifact verification](release_candidate_evidence/w07-published-reference-20261008/hosted-verified/summary.json) records **3,277 passed, zero failures/errors/skips**; original pytest and audit exits are zero, and the full runner's source hash is unchanged before/after. [Run 37856692326](https://github.com/ProfJJK-CoChem/CoChem-TORQ/actions/runs/37856692326) completed all eight jobs successfully. The authenticated original ZIP bytes were checked against fresh GitHub API size/digests; all archive members and each included JUnit outcome were read. This is current source CI evidence, separate from the held live student journey. A later source edit does not inherit these executions.

The separate installed-wheel Python 3.10 and 3.12 core lanes passed **1,079** and **1,079** cases respectively, each with zero failures/errors/skips. Both actual hosted wheels contain **262 byte-verified Python payloads** matching this frozen checkout, including the installed UI resource package and original notebook. Their immutable wheel hashes are retained in the artifact proof. Hosted marker-deselection counts are unknown because the original archived reports do not record them; a local earlier count is not transferred. Full/core/native selections overlap and must not be summed.

The [bounded W07 native checks](release_candidate_evidence/w07-published-reference-20261008/w07-native/summary.json) actually passed **28 cases** on the same clean commit/source/application with original waited pytest exit zero. **36 complete native evaluation inventories/3,197,961 bytes** were read back and hashed. This implements a closed-shell RHF/STO-3G nonperiodic relaxed valence-coordinate scan through the actual API and CLI, preserving physical coordinates, independent MINAO SCF starts, declared constraints, budget stops and genuine native lineage. The positive integration exercises **one unique target through three independent scheduled attempts**; it does not establish an entire curve or generalized surface. AO-density comparisons stay unavailable when independent relaxed geometries differ, with their actual coordinate difference retained. Genuine exhausted-budget attempts keep incomplete results unavailable. Periodic/multistate scans, a coupled large-amplitude Hamiltonian, electronic-branch completeness, surrogate calibration and global basin recall remain unqualified. W07 remains an unfinished broader agent workstream.

The full runner retained **64 complete native bundles/8,063,591 bytes**, with retention truncated=`true`. All retained inventory byte bindings were verified, but the bounded curator does not retain every scientific calculation from the suite and does not establish independent method/domain accuracy. Public proof contains bounded metadata and parameter-hashed JUnit; original private logs, arrays, checkpoints and raw diagnostics remain separate.

The [current existing-instance checks](release_candidate_evidence/w07-published-reference-20261008/current-onboarding/summary.json) separately record **50 passing software-interface checks**, **38 passing host-allocation checks**, idempotent canonical SDK setup/verification exits zero, and **163 compatible installed distributions**. These counts overlap other lanes and are not summed. The actual [147-file AST policy/static observation](release_candidate_evidence/w07-published-reference-20261008/current-static/policy-observation.json) records zero explicit runtime-replacement findings and two pre-existing environment-configuration calls; its narrow scope does not certify universal scientific integrity. This is existing-instance evidence, without a fresh-task restoration or full install-script replay claim.

The [initial archive request history](release_candidate_evidence/w07-published-reference-20261008/hosted-initial-download-history/summary.json) preserves genuine exit-one/zero-byte downloads and the earlier observer authentication failure. [Fresh managed requests and independent readback](release_candidate_evidence/w07-published-reference-20261008/hosted-download-followup/summary.json) retrieved the same original archives successfully and supersede the earlier inferred activation prerequisite. Reported allowlist absence with unknown enforcement did not establish the failure cause. Artifact retrieval for this run is complete; those earlier request failures are not failed pytest executions.

### Earlier source-specific checks and genuine failure history

The [preserved source4d evidence](release_candidate_evidence/final-audited-published-reference-20261008/summary.json) binds commit `bb6c23ed9b048091838fef0aba1c40fecc04cc20`, source `4d310c51dab2a0a2b4cfaf39a25a8333351b23bd35664d00e60c29c365278ca5` and application `60f6e63ddb9232db144248b2f0ef900d673add4406aad5a19fba4876e7a47b62` to **3,248 local passes** and an authenticated earlier hosted **3,248-pass** full regression. Its installed local Python 3.12 core and hosted Python 3.10/3.12 core lanes each passed **1,078** cases; the local core has 63 explicit marker exclusions, while its hosted exclusion count remains unknown. Original failures and metadata-only observations remain unchanged alongside distinct verified-artifact receipts. These outcomes are historical to the newer scientific source.

The [first genuine W07 attempt](release_candidate_evidence/w07-published-reference-20261008/w07-failed-history/summary.json) retained original exit `1` and **24 passes/4 setup errors**, with `0` failures and `0` skips, on commit `a65f3edcf5f5292d1074c2083829cd2546213e91`. Its completed underlying native evaluations remain valid evaluation evidence; they do not turn the failed enclosing checks into successful scans. The actual process-authority timestamp serialization defect was corrected and tested in a new source/attempt; the original receipt was not overwritten.

The earlier clean `c0df7a0` water/QCArchive examples retain their actual original source and application provenance in [the descriptive native proof](release_candidate_evidence/final-audited-published-reference-20261008/native-examples/summary.json). All six water comparisons, 101 publication files and 14 QCArchive arrays were genuinely read back; no new calculation or application-byte equivalence is claimed for the newer W07 application. HF/STO-3G versus QM9 B3LYP/6-31G(2df,p) remains descriptive, and B2PLYP components do not qualify revDSD derivatives or spectroscopy.

Canonical SDK setup and independent verification both actually exited zero for BASE `83462724849f1ef0be8c70ffcad6265d6af99388` and TOPOS `6a01b0f2adb7cff02edda6e339facf3d6f93904d`; their isolated module/source/wheel/dependency identities are preserved in [the SDK receipt](release_candidate_evidence/final-audited-published-reference-20261008/sdk/verified-installation.json). The original failed setup/retirement history remains separate. This verifies the observed instance, not fresh-task restoration or a whole saved-script replay.

The original source `6a2bd288` **3,192-pass** run and `95b05627` **1,072-pass** installed core/eight-pass provenance retry remain earlier source-specific history. The provenance lane's genuine first four-pass/four-error attempt required actual pip **25.2** before its successful retry; a runtime running genuine installation checks needs the pip module even when ordinary installation uses uv. The source `6036867` hosted core failures and the source4d wheel's original 1,069-pass/nine-failure stale-audit attempt also remain failure history. Counts from overlapping suites are never aggregated.

Passing software and bounded native calculations do not supply full-SRS scientific acceptance. All **73 requirements and 41 gates remain incompletely qualified**, all **41 gates remain blocked**, and all 140 method-matrix identities remain documentary until exact profile evidence permits activation. The context retains **17 agent implementation/research workstreams** and **six conditional researcher/access categories**. Student deployment remains **on hold**, and no live student pilot, licensed-engine staging, deployment or billing was performed.
