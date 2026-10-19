# Canonical BASE scientific provider: bounded migration evidence

This increment moves the TORQ-owned scientific assembly previously implemented
in BASE's adapter into the installed `cochem_torq.module_provider` namespace.
It extends `cochem_torq.ecosystem:module_provider`, preserving its existing
discovery fields and adding `cochem.module-provider/1` metadata. The callable is
`cochem_torq.module_provider:execute_handoff`; it receives the existing
`cochem.module-handoff/1` request and returns `cochem.module-operation/1`.
The source baseline was `e0019e7b0150e56a425942a06094b55cb60e0b73`.

| Operation | Exact scope and availability |
| --- | --- |
| `geometry_analysis` | Supplied-geometry isotope masses, center of mass, proper principal axes, inertia and rigid-rotor constants. Linear undefined axes remain null with reasons. No optimization, stationarity or B0 claim. |
| `research_scan` | Bounded rigid two-fragment mass-COM separation scan using genuine restricted native PySCF evaluations; no relaxation, counterpoise or completeness/accuracy claim. Requires the observed PySCF runtime. |
| `wiberg_lowdin` | Wiberg indices and charges in the Löwdin orthogonalized AO basis from the authenticated stable, converged, real restricted SCF checkpoint. Requires PySCF; an MP2 correlated density is unavailable. |
| `nbo_analysis`, `wiberg_nao` | Explicitly unavailable: no qualified canonical executable producer of these named analyses. Neither an engine installation nor Löwdin indices can enable them. |

BASE retains installation, isolated process execution, resource authority,
cancellation, immutable request transport and presentation. Consumers must verify
the installed distribution, wheel RECORD, provider origin and unchanged handoff;
they must not fall back to BASE scientific copies. The provider accepts a
registry argument for the versioned boundary but does not read credentials or
infer engine authorization from it. Whole-product and engine-profile
qualification remain governed by the [normative SRS](../wiki/CoChem-TORQ_Implementation_SRS.md)
and [complete requirement audit](full_srs_requirement_audit.json).

## Retained validation

An installed-wheel run from outside the checkout with Python isolated mode
passed **220 tests, zero failures, errors or skips** in 89.17 seconds. This includes
all **28 provider tests** and existing alignment, engine-integrity,
topology-mass and unit tests. An independent review reran the 28 provider tests
successfully and verified all 278 installed wheel RECORD rows. These overlapping
counts must not be added. Lint, formatting and whitespace checks passed.

The genuine positive calculations comprise HF/STO-3G H2 at 0.74 and 0.9 angstrom,
one rigid HCl/water mixed-fragment point, and restricted H2 Löwdin analysis.
H2 at 0.74 angstrom agrees with the declared -1.1167593074 hartree reference
within 1e-8 hartree; the H2 Wiberg index is 1 within 1e-10. The mixed-fragment
test retains 4 angstrom mass-COM separation and rigid intrafragment distances
within 1e-12 angstrom. Explicit isotope aliases and measured masses are preserved;
unspecified Tc and the unmeasured 999C isotope reject. Native checkpoint energy
identity remains within 1e-8 hartree, overlap eigenvalues must exceed 1e-10,
and density/charge conservation remains within 1e-7. Additional real-orbital and
AO-normalization checks do not relax those gates.

A deliberately invalid basis produces an authentic failed native attempt,
null energy, retained native files and a partial scan; it is not a successful
electronic calculation. Its PySCF missing-basis warning is expected. A separate
core-only wheel installation without PySCF successfully performed geometry
analysis while marking both engine-dependent operations unavailable.

The wheel contains 263 Python payloads. Its SHA256 is
`b8ead95bc1cd1d085c0a3844440b5bfa64e15bf88f381b969e13e8088ac5a529`.
The tested handler source SHA256 is
`ea5080defc0dd60089cc9c30449b5ba57305d4fdf8284bd36ad1b9dfc98910f4`;
the ecosystem source SHA256 is
`3c062600f5a08a616a84899ba81d0572cff27aadc533221f0f7bba5a883f8812`.
Documentation added after these runs does not change that scientific source.

Session-local evidence is retained under
`/workspace/.onboarding/artifacts/torq-provider-native-3/`; its sibling JUnit
file SHA256 is `5abb89524f7f168f4679a1102e0e3e89cbb254e8cd0833aebe59b3ad87f31bbd`.
The sibling `torq-provider-independent-review.xml` SHA256 is
`a1869504fab575e1fb126394471167c048c1b8914d97a7250db751999378883f`.
The following hashes identify actual retained bytes relative to that directory.
These local files are not a published archival deposit.

| Retained native file | SHA256 |
| --- | --- |
| `test_actual_rigid_h2_scan_reta0/out/scan-point-000/result.json` | `2dd91d09d2fb9cf6612b106862575e2dae7238dda68677a93f040fbbaa762090` |
| `test_actual_rigid_h2_scan_reta0/out/scan-point-001/result.json` | `4f0cf37dd85c1a6c794ac4c21420172703106c02bca3c73bd39743feb9d08fac` |
| `test_actual_rigid_h2_scan_reta0/out/scan-point-000/manifest.json` | `08b2316a12cf4e7f1a8cc08d9fd1494a43c79bcb78fcdcfcd0ecdce7ee801565` |
| `test_mixed_fragment_mass_com_n0/out/scan-point-000/result.json` | `ca72908de95baeef6aa258e75afd75298b7ab91e150aa52b53cc671e00324bab` |
| `test_mixed_fragment_mass_com_n0/out/scan-point-000/manifest.json` | `180efc274dc0a1a31202d4d6cfd2e8514ae20eb32ff2dde1582fc061ad90bd36` |
| `test_actual_h2_lowdin_wiberg_d0/out/native-electronic/wavefunction.chk` | `545b9253c61fe1624f70ef49e583e0c88652b52001e6b207b5b2c2a084a1d773` |
| `test_actual_h2_lowdin_wiberg_d0/out/lowdin-density-overlap.npz` | `5567abfc8cb5d56900cdcc10da3af36827b43c3fdc2ec972198b77683b50d2bd` |
| `test_actual_h2_lowdin_wiberg_d0/out/bond-analysis.json` | `8133d88e682fee771bcac482794e0d69b45736dc392e543e27f0bf4d7af16010` |
| `test_actual_failed_native_scan0/out/scan-point-000/result.json` | `3060594cbf0634085783f73e62c5b506f923acc911735c56128e5a7a3c967c2d` |

CI now includes the provider tests in its existing core and genuine-engine
lists. Those configuration changes are not themselves hosted execution evidence.
These results do not qualify all accepted method/basis tuples, scientific
accuracy, licensed ORCA/CFOUR engines, full SRS conformance, cancellation/restart,
or the student's real Codespaces/Actions journey. BASE integration and student
deployment retain their separate source-bound evidence and remaining gates.
