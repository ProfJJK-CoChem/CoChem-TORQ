# Canonical student environments

GitHub **Codespaces hosts the interface**. GitHub **Actions performs calculations**.
TORQ shares typed, integrity-checked artifacts with BASE and TOPOS. An interface
installation does not establish a quantum-engine qualification or spectroscopic
identification accuracy.

## Codespaces

Open this repository with its `.devcontainer/devcontainer.json`. The image pins
Python 3.12.14/Debian Bookworm by SHA-256 and runs as the unprivileged `vscode` user.
`postCreateCommand` invokes the actual `scripts/student_setup.py setup` lifecycle:

1. Install the hash-locked interface/core dependencies in `/opt/torq-venv` and
   install the current TORQ checkout.
2. Fetch the reviewed immutable BASE commit, verify its detached checkout,
   and invoke BASE's genuine `scripts/manage_modules.py` installer.
3. Build and install BASE and TOPOS in **different virtual environments**. Their
   overlapping `cochem`, `core`, `scripts` and `Libraries` namespaces are not
   imported into TORQ's environment.
4. Retain BASE's wheel/source/environment-integrity receipts and a student
   environment report outside the checkout in `~/CoChem_Artifacts`.

The reviewed source and adapter requirements are in
[`ci_tools/ecosystem-modules.json`](../../ci_tools/ecosystem-modules.json). The
TOPOS adapter explicitly requires `filelock`: a genuine isolated serializer
execution exposed that missing upstream dependency. These requirements are not
an assertion that TOPOS's conformer-search chemistry is qualified.

BASE and TOPOS are private repositories. The Codespaces configuration requests
contents-read access to these two repositories; GitHub asks the authorized
student to approve that additional repository access when creating a Codespace.
The bootstrap uses the authorized Codespaces credential only for Git source
retrieval. BASE's reviewed installer receives the `COCHEM_SOURCE_READ_TOKEN`
contract and excludes that credential from builds and dependency installation.
No source token is written to Git configuration, command arguments, or receipts.

Each container start launches the actual authenticated JupyterLab interface on
port 8888 and checks the actual student notebook through Jupyter's authenticated
contents API. The forwarded Codespaces port remains private. Jupyter's local
token is stored with mode `0600` in `~/CoChem_Artifacts/interface/token`; use it
at the login prompt. Never publish that file or the Jupyter log as a result
artifact. No electronic calculation starts merely by opening the notebook.

The interface image includes the official GitHub CLI from signed Debian
packages. Codespaces' injected `GITHUB_TOKEN`, or an explicitly configured
`GH_TOKEN`, authorizes the user's Actions requests; these two variables are
preserved only in the interface runtime. They are excluded from build/install
hooks and never passed into the calculation container. `gh auth status` can
diagnose the current identity. Dispatch needs Actions write access to the selected
repository; viewing/downloading results needs Actions read access. A failed
authorization remains a visible error and never triggers a local calculation.

Useful lifecycle commands:

```sh
python scripts/student_setup.py setup
python scripts/student_setup.py start --port 8888
python scripts/student_setup.py check --port 8888
python scripts/student_setup.py stop
```

Repeated setup verifies existing installations. When a reviewed module
specification changes, an old receipt is rejected with an explicit instruction
to preserve the environment and select a new `--artifact-dir`. No verified
environment is silently modified to accommodate changed dependencies.

`COCHEM_MODULE_ROOT` and `COCHEM_MODULE_MANIFEST` identify the separate module
installations and reviewed manifest. Scientific handoffs are file-based and
versioned; TORQ does not mix sibling namespaces in its interpreter. BASE's own
upstream distribution manifest still pins an older TORQ revision. A separate
BASE manifest update to the published TORQ release commit is necessary before
that upstream installer distributes this release.

The mandatory Actions ecosystem jobs also need source access to both private
siblings. TORQ's built-in `GITHUB_TOKEN` is limited to TORQ and cannot supply
this access. The preferred configuration is a GitHub App installed on only
`CoChem-BASE` and `CoChem-TOPOS`, with repository contents-read permission:

1. Set the TORQ repository variable `COCHEM_MODULES_APP_ID` to the App ID.
2. Store its private key in the Actions secret
   `COCHEM_MODULES_APP_PRIVATE_KEY` through GitHub's administrative interface.
3. The pinned official `actions/create-github-app-token` action mints a temporary
   installation token restricted to these two repositories and contents-read
   access; its post-job cleanup revokes the token.

An already provisioned fine-grained contents-read token for exactly those two
repositories may instead be stored in `COCHEM_SOURCE_READ_TOKEN`. That token is
passed only to the sibling installer step. Neither route passes source-read
credentials into calculations or numerical tests. Missing authorization fails
the required jobs explicitly; it does not skip ecosystem qualification. The
managed session cannot create these repository secrets because its GitHub
integration lacks Secrets administration permission.

## Actions calculation service

The [`calculation.yml`](../../.github/workflows/calculation.yml) workflow is the
canonical public student calculation endpoint. The interface submits:

| Input | Meaning |
|---|---|
| `request_b64` | Strict base64 encoding of at most 16 KiB of explicit UTF-8 `cochem.torq.request/1` JSON. |
| `request_sha256` | SHA-256 of exactly those JSON bytes. |
| `request_id` | UUID matching the request's embedded UUID. |
| `expected_source_sha` | Immutable source commit reviewed at submission; a branch movement before dispatch fails closed. |
| `engine` | `pyscf`, the explicitly supported public engine profile. |

The dispatcher resolves the actual commit and checks the source identity before
submission. The workflow checks out `github.sha` and validates all transport
identities before building an engine. Duplicate JSON fields, nonfinite numbers,
malformed geometry, and exceeded public classroom limits are rejected. Inputs
enter environment variables and are never expanded into shell program text.

The public limits are 12 atoms from H through Ne, a closed-shell singlet,
1–2 CPU cores, 256–2048 MB of requested electronic-structure memory, and at most
600 seconds of chemistry execution. The container additionally enforces two CPU
cores, 3 GiB total memory, 256 processes, no network access, a read-only image and
a bounded temporary filesystem. A subprocess wall-time limit and an outer
container limit are both applied. Each workflow attempt has separately named
containers and cleans them up if interrupted.

The currently exposed public recipes are `hf-sto-3g-education` and
`hf-cc-pvdz-research`. They are accurately named Hartree–Fock teaching/validation
experiments. They do not stand in for any historical method-matrix row, exact
revDSD, or an identification-grade spectroscopy recipe. Products that lack a
validated backend retain an explicit blocked status and preserve valid earlier
stages.

Only `contents: read` is granted to the calculation workflow; no repository-write
permission or engine-download credentials are exposed. Actions dependencies are
pinned to reviewed commits. A request's run title includes its UUID. The result
artifact name includes UUID, hosted run ID and attempt number and uses the
immutable upload-artifact v4 backend without overwriting. Results and transport,
image, engine and numerical-qualification evidence are retained for 30 days;
students must export important work before expiry.

## Isolated calculation image

[`docker/engines/pyscf/Dockerfile`](../../docker/engines/pyscf/Dockerfile) uses the
same immutable Python base and a separate `/opt/engine-env`. Its complete Python
runtime is hash-locked in
[`requirements-pyscf-py312.txt`](../../ci_tools/requirements-pyscf-py312.txt),
including actual PySCF 2.14.0, geomeTRIC 1.1.1 and DFTD4 3.7.0. It runs as the
unprivileged `student` user. The interface lock contains no electronic engine.

Before every public calculation, `ci_tools/qualify_pyscf_image.py` performs a
genuine H2/RHF/STO-3G SCF and verifies native gradients and Hessians against
independently displaced SCFs/gradients. The evidence includes actual installed
versions, exact image identity, source commit/status, raw numerical arrays and
their hash, convergence and error thresholds. This qualifies the named software
and numerical derivative experiment only. It does not establish accuracy for
other molecules, other methods, anharmonic spectroscopy or identification.

ORCA, CFOUR, MPQC and accelerator/site engines require distinct environments,
legitimate site provisioning and their own actual qualification. No licensed
engine image is downloaded or declared available by the public workflow. Exact
revDSD research has independent activation milestones and cannot become enabled
by selecting a historical row or installing PySCF/DFTD4.

Build and qualify locally:

```sh
docker build -f docker/engines/pyscf/Dockerfile -t torq-pyscf:review .
mkdir -p /tmp/torq-engine-evidence
image_id="$(docker image inspect torq-pyscf:review --format '{{.Id}}')"
source_commit="$(git rev-parse HEAD)"
docker run --rm --network none --cpus 2 --memory 3g \
  --mount type=bind,src=/tmp/torq-engine-evidence,dst=/evidence \
  --entrypoint python torq-pyscf:review \
  /opt/torq-source/ci_tools/qualify_pyscf_image.py \
  --output /evidence/qualification --image-identity "$image_id" \
  --source-commit "$source_commit" --source-worktree-status modified
```

Use `clean` only for a verified clean checkout. The mounted output directory
must be writable by container UID 1000. Managed cloud builds additionally need
their public session CA as a BuildKit secret; preserve the configured proxy and
TLS verification:

```sh
docker build --secret id=proxy_ca,src=/etc/ssl/certs/ca-certificates.crt \
  -f docker/engines/pyscf/Dockerfile -t torq-pyscf:review .
```

## Release checks and current evidence

`integrity.yml` retains the earlier software-integrity jobs and adds mandatory
real PySCF energy/derivative/optimization checks, genuine pinned sibling-producer
interoperability, actual notebook/widget checks and authenticated Jupyter
start/check/stop. A missing sibling checkout fails the mandatory integration gate.
An unavailable engine cannot be replaced by a simulated calculation.

The self-contained `real-student-engine` job qualifies native scientific methods
without requiring private sibling access. The separately required
`real-ecosystem-consumers` job runs all 33 ecosystem tests, including two genuine
TORQ-calculation → BASE-consumer checks, and fails when source authorization is
unavailable. The interface lane explicitly deselects
those two `real_engine` tests and executes the 31 producer/file checks without
installing PySCF into the interface. Both lanes reject skipped selected tests
using their actual pytest JUnit reports. Artifact publication and bounded
anharmonic research checks are also mandatory in the calculation lane; research
passes never activate an unqualified public method.

The independent `student-calculation-image` job builds the exact calculation
Dockerfile, normalizes and hashes the committed water teaching input, qualifies
actual native derivatives, and executes every baseline product offline under
the unprivileged image user. It verifies the sealed result and typed stages
inside the same image and retains its request, native records, source/image
identity and verification evidence. Its water request has two cores, 2048 MiB
application memory and a 600-second deadline; Docker enforces two cores and
3 GiB total memory. It requires no private sibling credential.

Local validation has built the calculation and interface Docker images, executed
real H2 native derivative qualification offline in the calculation container,
installed both pinned sibling distributions through BASE's actual isolated
installer, and started/read/stopped the authenticated Jupyter notebook. The
software request-transport suite passed 16 tests. Numerical H2 maximum errors
were approximately `2.58e-9` hartree/bohr for the gradient and `8.35e-9`
hartree/bohr² for the Hessian against the specified finite differences.

The actual offline water worker completed geometry optimization, projected
harmonic characterization, equilibrium constants and the requested rigid-rotor
teaching catalog; all 34 files in its sealed result inventory were verified
against the typed stage contracts and their recorded byte hashes. Optimization
and harmonic characterization retain separate native engine directories. The
actual vibrational Hessian was positive definite after removing six external
modes, and its external-invariance residual was `1.21e-7` relative. This
establishes a local model minimum; it does not establish experimental accuracy.
The authenticated notebook lifecycle also passed inside the final non-root
interface image, whose genuine GitHub CLI version is recorded. Actual isolated
BASE/TOPOS producer/consumer interoperability passed 33 checks with zero skips.
See the machine-readable
[canonical environment evidence](canonical-environment-check-results.json) and
[ecosystem evidence](ecosystem-check-results.json). Image evidence identifies
the tested source snapshot; later application changes require another image
build and its own execution checks.

Hosted Actions integrity run `37696948074` was performed for TORQ pull request
3. It exposed real import-boundary defects in clean dependency profiles and
missing authorization for the private sibling repositories. That failed run
does not qualify a hosted scientific or ecosystem profile. The corrected jobs
retain every required gate and explicitly require administrator-provisioned
sibling source access. A live Codespace has not been started. Local
image/lifecycle evidence and hosted deployment evidence remain distinct. The
repository's student release report records the broader final application and
scientific validation scope and subsequent hosted observations.
