# Pinned native Pickett consumer

`cochem_torq.spectroscopy.pickett_backend` runs genuine externally provisioned
SPCAT and SPFIT executables. It writes typed, bounded native decks and retains
the original input bytes, output files, process logs, analysis, and a complete
immutable inventory. It never substitutes another solver for Pickett.

## Source, provisioning, and permission boundary

The inspected source is the publicly accessible
[laserkelvin/Pickett mirror](https://github.com/laserkelvin/Pickett), pinned to
`f5af2ff0f1acb1eb11ecffe7ef3b55d457801f7b`. Its README traces the base to CDMS
Pickett sources, with subsequent modifications. Inspection found Caltech
copyright notices, “all rights reserved,” and a NASA sponsorship acknowledgement.
The repository has no license granting source or binary redistribution.
Availability and file hashes do not establish the author's authentication,
redistribution permission, or permission for every deployment.

No vendor source, binary, archive, or container layer is added to TORQ. The
observed local build stays outside the checkout. Official JPL/CDMS retrieval was
denied by this session's network proxy; that denial is not evidence of upstream
unavailability or a license grant. Automatic student-image provisioning remains
blocked until the laboratory establishes appropriate distribution/use rights.

The source inventory contains 63 unchanged non-build Git blobs, independently
compared with the pinned commit. Its canonical inventory SHA-256 is
`9ce0f0b761a4094eee4aaf2d32017d80b47c1fca4083861d68c80a3accd46053`.
The tracked prebuilt `splib.a` is deliberately excluded because `make` rebuilds
it; the external provisioning record retains both archive hashes separately.
The inspected `spinv.pdf` hash is
`4dd04df5c2fa82e87e4ff0a6942a12e5d7d710be01b09e66f37c6952e2d488e9`.

The actual local compiler was `gcc (Debian 14.2.0-19) 14.2.0`, running
`make spcat spfit`. Actual local binary hashes are:

| Program | SHA-256 |
| --- | --- |
| SPCAT | `6fa5c0a61a9b52a55bad093bd70ab87420acc0ddbff85302d0c1b34346e962f8` |
| SPFIT | `4b2e27d044de0ef6e313925c979c3c9bdb1deebb3c9d63bf306b8be98eb5b372` |

The provisioning manifest binds source paths, all inventoried source hashes,
actual executable hashes, compiler and build-log provenance, and the explicit
`unverified_not_bundled` distribution status. The adapter verifies these bytes
before and after actual execution. A different build needs an independently
reviewed external manifest; an ELF header alone does not qualify an engine.

Native tests require both `COCHEM_TORQ_PICKETT_PROVISIONING_JSON` and
`COCHEM_TORQ_PICKETT_PROVISIONING_SHA256`. For this observed environment they are:

```sh
export COCHEM_TORQ_PICKETT_PROVISIONING_JSON=/workspace/torq-review/pickett-f5af2ff-provisioning.json
export COCHEM_TORQ_PICKETT_PROVISIONING_SHA256=e5ec8f43cf1d71de894c3329e671665abf4fa8cba1a55af11a410fa855eb8843
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
