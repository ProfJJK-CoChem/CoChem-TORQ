# Genuine published-value comparison product

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

The [original source-bound local evidence](release_candidate_evidence/published-reference-20261008/summary.json) records **3,192 passed, zero failed/errors/skipped**, with source hash `6a2bd28829ca376b879e8f4b726c0516a3ca62fec52ef76d08205cbc6a25b484` unchanged during the complete 1,720-second regression. Lint, formatting, typing and the scientific test-policy inventory also passed for that snapshot. Its installed-wheel core lanes each passed 1,072 tests on Python 3.10 and 3.12; that earlier wheel contained 256 matching packaged Python files. The original complete run with two ignore-test failures is retained alongside the corrected 75-case ignore check and complete rerun. These are historical source-specific results, not proof for subsequent merges, and overlapping lanes must not be summed.

Clean-commit documented water and QCArchive examples both completed with actual zero exits on commit `00aa3ffcc8f3c218752791861f8f4363f3af383a`. All six water comparisons, 97 publication files, 16 QCArchive native files and 14 array hashes were checked against genuine retained outputs. The separate CLI comparison and 5,198-byte source retrieval also completed. The application/source hashes remained unchanged and the checkout remained clean during those native examples.

A subsequent immutable wheel was built at merged commit `7bcc0431dff9ddef35bd43f86080ccc3e588be4a`, source hash `95b05627133131fe9ae879b01859bf46d71e652d6351c7ca03b237722ddcaf82`. Its SHA-256 is `ee25d6c6372024c5375ef715cf91384520198d265870de3c10bd17323792fcf3`; all **259 packaged Python files** matched that frozen source. The genuinely installed Python 3.12 wheel passed the canonical 44-file core selection: **1,072 passed, 63 explicitly marker-deselected, zero failures/errors/skips**. Eight genuine wheel-install/source-identity tests also passed, and the installed `cochem-torq-ui` entry point and packaged notebook were checked. Python 3.10 was **not** run for this merged wheel. These completed checks belong to this specific merge; later source changes require their own evidence.

The provenance-test lane initially reported four setup errors because the uv-created main environment had no `pip` module. Those original outcomes are retained. Installing **pip 25.2** in that environment satisfied the genuine wheel-build/install prerequisite; the complete eight-case rerun passed. Scientific dependencies were not changed by that prerequisite fix. A main environment used for this lane must supply `python -m pip`, even when uv performs its ordinary dependency installation.

The temporary Python 3.10 and Python 3.12 core validation environments were retired after their completed evidence and immutable wheels were retained; future replay requires recreating those environments. The main Python 3.12 scientific environment remains an editable install of the checkout. Its installed CLI and dependency compatibility checks passed. This is current-instance setup evidence; the full saved setup script and fresh-task restoration were not independently replayed.

The final integrated full regression remains pending at this snapshot. The revised ecosystem manifest pins BASE `83462724849f1ef0be8c70ffcad6265d6af99388`; setup correctly rejected overwriting the earlier verified SDK receipt and requires a new isolated artifact root. Provisioning that root and running the integrated regression remain agent environment/testing work. New hosted-source validation is also pending at this snapshot. Software success does not establish an accepted independent accuracy benchmark or full-SRS readiness; all **41 full-scope gates remain blocked**. Student deployment remains **on hold**.
