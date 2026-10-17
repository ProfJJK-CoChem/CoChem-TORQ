# Pinned native Pickett consumer

`cochem_torq.spectroscopy.pickett_backend` runs genuine externally provisioned
SPCAT and SPFIT executables. It writes typed, bounded native decks and retains
the original input bytes, output files, process logs, analysis, and a complete
immutable inventory. It never substitutes another solver for Pickett.

## Source, provisioning, and permission boundary

The pinned source is Pickett's own distribution at JPL,
<https://spec.jpl.nasa.gov/ftp/pub/calpgm/>, which the CoChem ecosystem uses for
SPFIT/SPCAT (the PI's 2026-10-17 instruction, recorded in CoChem-SpycFit as
decision D-SPYC-17). JPL publishes no version tags, so the pin is the bytes:
`SUPPORTED_SOURCE_INVENTORY` lists the SHA-256 and size of the 21 source files
that `make spcat spfit` compiles, the `README`, the `whats.new` change log and
the `calpgm.pdf` and `spinv.pdf` manuals. Its canonical inventory SHA-256 is
`de701d2b8a0fe232987518739cd9544a7e34a8c0b63a759c421083cfe8f55919`. The
`spinv.pdf` hash is
`4dd04df5c2fa82e87e4ff0a6942a12e5d7d710be01b09e66f37c6952e2d488e9`, the same
bytes as in the previous pin.

`ci_tools/provision_pickett_validation.py` downloads exactly these files over
HTTPS. A changed size or hash, a redirect or an HTTP 4xx response stops
provisioning; an HTTP 5xx response or connection failure is retried up to four
times and then stops it. The prebuilt executables, objects and `splib.a` that
the same JPL directory serves are never downloaded. Any change at JPL therefore
blocks provisioning until a reviewed pin update. On 2026-10-17 all 25 files
were retrieved from JPL on the first attempt; on the same day JPL answered every
`.var` request (for example the HCO example's `hco.var`) with HTTP 500. No
`.var` file is part of this pin.

Inspection found Caltech copyright notices, “all rights reserved,” and a NASA
sponsorship acknowledgement in the source headers. The JPL catalog site states:
“These programs are copyrighted, but made freely available as a service to the
spectroscopic community.” No license grants source or binary redistribution.
Availability and file hashes do not establish the author's authentication,
redistribution permission, or permission for every deployment.

No vendor source, binary, archive, or container layer is added to TORQ. The
local build stays outside the checkout. Automatic student-image provisioning
remains blocked until the laboratory establishes appropriate distribution/use
rights.

Until 2026-10-17 TORQ pinned the laserkelvin/Pickett GitHub mirror at
`f5af2ff0f1acb1eb11ecffe7ef3b55d457801f7b`. Of the 24 pinned JPL files it also
carries (it has no `README`), it differs only in `Makefile` (build flags) and
`calcat.c`, where a 2020 change prints the partition function at 1,001
temperatures from 1000 K to 1 K instead of JPL's eight. TORQ writes its own
partition function into the `.int` deck and does not read that printed table.
The TORQ Pickett tests gave identical outcomes on both builds.

An observed build on 2026-10-17, `gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0`
running `make -j1 spcat spfit` with the JPL `Makefile` (`-O2`, no debug
information), produced these binaries; a different compiler or flags give
different hashes, which the provisioning manifest records:

| Program | SHA-256 |
| --- | --- |
| SPCAT | `8d3eb357347e9678d699cee1ee7d9df12c20b91d220f788aea7ad50be3ca533c` |
| SPFIT | `784ce379142ce700dbf0345ca1cdec6f82be344f687433ee585e5cd279012071` |

The provisioning manifest binds the upstream URL, all inventoried source hashes,
actual executable hashes, compiler and build-log provenance, and the explicit
`unverified_not_bundled` distribution status. The adapter verifies these bytes
before and after actual execution. A different build needs an independently
reviewed external manifest; an ELF header alone does not qualify an engine.

Native tests require both `COCHEM_TORQ_PICKETT_PROVISIONING_JSON` and
`COCHEM_TORQ_PICKETT_PROVISIONING_SHA256`, which the provisioning receipt
prints as `provisioning_manifest_path` and `provisioning_manifest_sha256`:

```sh
python ci_tools/provision_pickett_validation.py --root /absolute/new/external/dir
export COCHEM_TORQ_PICKETT_PROVISIONING_JSON=<provisioning_manifest_path>
export COCHEM_TORQ_PICKETT_PROVISIONING_SHA256=<provisioning_manifest_sha256>
PYTHONPATH=src COCHEM_DISABLE_SANDBOX_CHECK=1 python -m pytest tests/test_pickett_backend.py -q
```

These are environment-local artifact paths, not assets checked into Git.
Unprovisioned native tests fail; they are not silently skipped or counted as
scientific qualification. Native cases carry `real_engine` and `research`
markers so the release orchestrator must select and provision them explicitly.
Ordinary educational CI does not establish a Pickett release pass.

## Supported numerical and provenance contract

The current contract is one spin-free asymmetric-top vibronic state, integer
`J,Ka,Kc`, explicitly chosen Watson A or S reduction, and Ir axes. It supports
rigid constants and complete five-parameter quartic sets, with all parameter
values in MHz and principal-axis dipoles in Debye. Other representations,
hyperfine states, spin weights, tunneling, multiple vibrational states, and
reduction conversions need independent implementation and validation.

The native reduction option selects parameter labels; it does not transform
Hamiltonian operators. Explicit parameter identifiers/signs are:

| TORQ parameter | Pickett identifier | Multiplier applied to supplied value |
| --- | ---: | ---: |
| A, B, C | 10000, 20000, 30000 | +1 |
| Delta_J, Delta_JK, Delta_K | 200, 1100, 2000 | −1 |
| delta_J, delta_K | 40100, 41000 | −1 |
| D_J, D_JK, D_K | 200, 1100, 2000 | −1 |
| d1, d2 | 40100, 50000 | +1 |

Identifier `50000` is the fourth-power off-diagonal operator. `60000` would
introduce a sixth-power operator and is not the S-reduction quartic d2 term.
Genuine native A/S tests compare with independently constructed finite-J
angular-momentum matrices, including the operator signs and anticommutators.

Catalog parsing accepts the pinned 79-column QNFMT=303 record and harmless
trailing blanks emitted by this engine. Extra nonblank fields, duplicate
transitions, overflow, unsupported assignments and nonfinite values fail.
The actual two-column codec includes `a0=-10`, `a9=-19`, `z9=-269`, `A0=100`
and `Z9=359`; the inspected manual has a typo at the negative extreme, so the
implementation follows inspected native source bytes and codec checks.

Catalog frequencies and uncertainty columns are MHz. The `.int` frequency
cap is GHz. LGINT is `log10(nm² MHz)`, an integrated absorption quantity, not
relative line strength. The partition function, its origin, temperature,
maximum J and frequency band are explicit; no unseen vibrational or nuclear
spin partition model is invented.

Native covariance uses a packed lower factor **L** with **C = LᵀL**, normalized
by parameter-error columns. This differs from NumPy's usual lower Cholesky
factor convention. The writer uses the appropriate reversed-axis factorization
and operator signs; the reader reverses native signs and reconstructs the
actual full covariance. An independent Hellmann–Feynman derivative comparison
checks propagated native uncertainties at catalog serialization precision.
An absent covariance leaves scientific uncertainty unavailable even when the
native catalog error field happens to print zero. Native local linearization
does not include Hamiltonian inadequacy, electronic-method error or independent
experimental calibration.

Each original deck line is ASCII and at most 79 characters. The upstream build
warns about an 82-byte buffer in SPFIT's backup routine; unrestricted supplied
decks are therefore not an API of this adapter. Covariance input is packed into
at most seven ten-column fields per line. No vendor source is modified.

## Molecular evidence and fitting

`parameter_origin` and `constant_observable` distinguish an explicitly
mathematical model from molecular ab-initio Be/B0 and genuine experimental fit
parameters. Mathematical cases are not fabricated molecular observations.
Ab-initio rigid export requires an independently pinned, fully verified TORQ
publication bundle, the exact available typed constants, and the actual
electronic-density dipole projected using the retained principal axes.
Neutral systems are supported; charged dipole-origin conventions remain
unsupported. Be cannot become B0 by changing a label. An unknown ab-initio
covariance cannot become a numerical error estimate.

The genuine HF/STO-3G water test exercises optimization, Hessian, native dipole,
typed Be, publication, and native SPCAT. It establishes that consumer path for
that bounded calculation, not laboratory accuracy or a vibrational correction.

Current SPFIT targets come unchanged from verified genuine SPCAT calculations.
They are explicitly native prediction regressions, not measured transitions.
Every fit line matches the exact source catalog state and frequency; the
complete source catalogs and their independently retained manifests are copied
into the immutable fit bundle. Reverification recomputes all input decks and
output analysis and verifies those retained catalogs. Re-sealing a changed
request or summary cannot change the actual source values.

Fit regression tolerances represent native serialization tolerances. Wide
native fit-control scales are execution controls, not experimental parameter
uncertainties. Rank-deficient native covariance remains a diagnostic. The
adapter does not silently discard rejected lines or claim physical coverage
from a full-rank numerical fit. Importing curated independent experimental
measurements awaits its source/assignment/uncertainty contract and currently
fails explicitly.

## Observed checks and remaining qualification

The actual rigid model A/B/C = 10000/5000/3000 MHz, three unit mathematical
dipoles, T=10 K and J=0…5 produces 229 native lines below 100 GHz, matching the
independent finite-J line count. The observed maximum frequency difference is
about `4.97e-5 MHz`, and the maximum log10 integrated-intensity difference is
about `5.00e-5`, consistent with native four-decimal serialization. The
comparison uses the inspected native legacy coefficient `4.16231e-5` and keeps
it distinct from contemporary SI constants. It matches energy separations
within J; it is not an independent proof of Ka/Kc assignment or spin statistics.

Genuine SPFIT regression recovers the same three constants from perturbed
initial values, retains full covariance and actual residual/rejection
diagnostics, and successfully feeds a second genuine SPCAT calculation.
A delta-J=0-only fit intentionally demonstrates actual underidentification.
The test suite also retains real timeout failure artifacts and rejects actual
corruption, re-sealed false summaries/requests and invented regression targets.

All successful receipts remain `completed_native_unqualified`,
`identification_ready=false`, and `experimental_accuracy_established=false`.
Immutable hashes establish byte consistency relative to independently retained
pins, not remote execution attestation or an author signature. Completion of
the consumer does not supply licensed redistribution, canonical hosted native
execution, experimental-reference validation, hyperfine/tunneling treatment,
anharmonic distortion tensors, nuclear spin restrictions, or publication-grade
identification qualification. Those remain separate release gates.
