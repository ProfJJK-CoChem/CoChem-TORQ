# Development profiles and verified packaging

The current student deployment is defined by
[Canonical student environments](canonical_environments.md): Codespaces hosts
the interface; Actions performs calculations; BASE and TOPOS install in separate
verified environments. The historical repair evidence below describes the
earlier core devcontainer and is not a qualification of the newer engine images.

## Supported installation paths

The wheel includes `Libraries`, `cochem`, `cochem_torq`, `orchestrator`, `scripts`,
`UI` and `core`, including PWA JavaScript/JSON and the unexecuted startup notebook.
The `cochem-torq` console script invokes the shared student application CLI. Importing `cochem_torq`
does not initialize scratch directories or assert provenance; applications call
its exported `bootstrap_environment` function explicitly when appropriate.

Core installation declares its direct scientific, isotope-database, Arrow,
HDF5/filter, compression, hashing and locking dependencies. Pydantic v2 is
required. Optional profiles are independent:

| Extra | Purpose | Limitations |
|---|---|---|
| `qcschema` | Reference QCSchema validation through QCElemental | Schema conformity does not establish scientific accuracy or genuine engine execution. |
| `dvr` | JAX numerical routines | Install and qualify accelerator support separately. |
| `ml` | PyTorch, ASE, scikit-learn, safetensors | Does not supply trained models, checkpoints, calibrated uncertainty or external quantum engines. |
| `molecular` | RDKit and ASE | Does not certify generated conformers as optimized electronic-structure results. |
| `notebooks` | JupyterLab, widgets, plotting | Individual workflow readiness still depends on implemented features and their configured engines. |
| `ipc` | ZeroMQ, xxHash | Optional IPC integrations. |
| `dev` | pytest, Ruff, mypy, build | Tests of external engines require those actual executables and authentic evidence. |

For ordinary development:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev,qcschema]'
python -m pip check
cochem-torq --help
```

For the pinned Python 3.12 core/QCSchema runtime:

```sh
python -m pip install --require-hashes -r ci_tools/requirements-core-py312.txt
python -m pip install --no-deps .
```

`ci_tools/requirements-core-py312.txt` pins transitive runtime versions and
upstream artifact SHA-256 hashes. It does not pin the OS image, compiler or
isolated Python build requirements. Notebook, test and ML extras are currently
version-bounded rather than fully locked. Python 3.10 dependency resolution was
checked; the CI matrix is configured to execute the minimum-version profile. Local execution
reported below used Python 3.12.14.

To select CPU PyTorch explicitly before installing the ML profile:

```sh
python -m pip install 'torch>=2.4' --index-url https://download.pytorch.org/whl/cpu
python -m pip install '.[dev,dvr,ml,qcschema]'
```

Engine binaries, licenses, model weights and HPC/MPI services remain separately
provisioned requirements. Installing an extra does not imply that those resources
exist or that an adapter is scientifically validated.

## Devcontainer

The Dockerfile uses Python 3.12 on Debian Bookworm, creates the `vscode` user,
and gives that user an owned `/opt/torq-venv`. The post-create command installs
the hashed core requirements and development/QCSchema/notebook extras, checks
dependency consistency and invokes CLI help. It no longer references an absent
`requirements.txt` or a nonexistent user.

The managed cloud validation built this image with its public CA supplied as
a BuildKit secret:

```sh
docker build --secret id=proxy_ca,src=/etc/ssl/certs/ca-certificates.crt \
  -f .devcontainer/Dockerfile -t cochem-torq-dev:integrity .
```

The CA mount is optional for ordinary hosts. Preserve the managed environment's
proxy and registry configuration. HTTPS processes inside a running container
need the session CA mounted read-only with the appropriate tool trust variable
(for pip, `PIP_CERT`). No TLS or package-signature verification was disabled.

Actual validation ran the built image as `vscode`, checked virtual-environment
writability, copied the checkout into the container's temporary directory and
executed the post-create installation. Core hash verification, editable install,
`pip check`, CLI help and JupyterLab/ipywidgets imports passed. The additionally
declared anywidget/traitlets dependencies were installed and imported in a fresh
container. This validates the development image and installation path; it does
not constitute an application deployment or a live ORCA/CFOUR/SPCAT calculation.

## CI and local evidence

`.github/workflows/integrity.yml` has a Python 3.10/3.12 core job and a Python 3.12
CPU numerical/ML job. The core job builds the wheel, installs declared dependencies,
and checks installed imports and CLI execution from `/tmp`. The CPU job selects
the official CPU PyTorch index and runs numerical, surrogate and conformal
integrity checks. Three PyTorch storage/trajectory checks carry the `cpu_ml`
marker: the core job deselects them and the CPU job executes them explicitly.
No required engine failure is converted into a passing mock calculation.

Local evidence recorded during this repair:

- A wheel was built and installed in a fresh `/tmp/torq-wheel-check` environment.
  Ten package/module imports were checked from `/tmp` using Python `-I`, with
  assertions that files came from the installed wheel. CLI help and dependency
  consistency passed.
- The initial clean core regression selection passed 100 tests. Its three CPU
  storage/trajectory tests then passed with actual PyTorch; none was silently
  skipped. The additional element, SPCAT, asynchronous runner, method-selection
  and WKB core selections passed another 52 tests in the same clean environment.
- Export integrity and updated legacy export tests passed: 26 and 20 respectively.
  They exercise actual JSON/XML/Arrow/HDF5/Zstandard I/O, invalid inputs and an
  explicitly labelled analytic hydrogenic eigenenergy. These are not simulated
  quantum-program outputs or an external-engine accuracy benchmark.
- The real Docker image build and non-root post-create install passed.
- The CI workflow YAML was parsed locally. Hosted GitHub Actions have not been
  executed as part of this uncommitted working-tree change.

SPCAT's legacy automatic parameter/intensity writers are deliberately unavailable:
they previously invented uncertainties/partition values and malformed decks.
`PickettSPCATRunner.run_spcat(base_name, working_dir)` accepts externally prepared
`.var`/`.int` files, executes the configured binary in a fresh directory, preserves
input/binary/catalog hashes and rejects unsupported or malformed output. Its
current parser covers integer `J,Ka,Kc` format 303 only. Live execution was not
validated locally because SPCAT is not installed; no alternate solver or invented
quantum assignment substitutes for it.
