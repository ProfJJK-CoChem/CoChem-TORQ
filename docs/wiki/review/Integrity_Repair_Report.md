# Scientific-data integrity repair report

Prepared 2026-10-07. Source baseline: `d7a4739a5f7d6f22ed659b32eeb4706bef16225e`. This report describes the follow-up code changes; the original [architecture audit](Architecture_Implementation_Audit.md) remains a record of pre-repair findings. The [implementation SRS](../CoChem-TORQ_Implementation_SRS.md) is normative. Passing software checks is distinct from method validation, predictive accuracy and a deployment release.

## Changes to the twelve original blockers

| Audit item | Implemented correction | Remaining qualification |
|---|---|---|
| ARCH-01 parser | Missing energy/gradient is unavailable; strict finite/shape/unit checks; normal exit, SCF convergence and optimization convergence are separate; malformed/truncated output rejects. | Live calculations for each advertised engine/release/property tuple. Tracked engine artifacts test parsing but do not establish new calculation accuracy. |
| ARCH-02 missing observables | Removed fabricated frequencies and zero-filled missing dipoles. Typed stage availability controls downstream products. | Qualified harmonic, anharmonic and property backends. |
| ARCH-03 final geometry | Actual parsed final geometry is distinguished from input; only a converged optimization receives that role. Input units cannot be silently ignored. | Independent live non-equilibrium optimization/gradient checks. |
| ARCH-04 CLI | Distinct result statuses and exit codes; submitted coordinates are not labeled optimized after failure; artifacts are attempt-specific. | Full successful engine workflow and UI journey on each advertised platform. |
| ARCH-05 dispatch | Supported engine dispatch is explicit; unsupported engines/products/counterpoise requests reject before expensive work. | Additional adapters qualify independently; the generic CFOUR gradient executor is disabled because it did not implement its contract. |
| ARCH-06 workflow | Refinement of supplied geometry is an explicit profile; no mandatory ML/GOAT/CREST discovery or claim that one optimization is a complete cascade. | Planned PES/discovery profiles and actual TOPOS handoff tests. |
| ARCH-07 queue | Opaque lease token and monotonic generation fence heartbeat, failure and completion; original worker credentials propagate into process supervision. | Actual BASE service integration and multi-host operational qualification. |
| ARCH-08 TORQ storage | Exact molecular shape/ordered identity required; no atom truncation/padding. Unknown uncertainty has an explicit availability mask. | Validated migration for existing ambiguous files. |
| ARCH-09 HDF5 | Native locks retained; single writer ownership; committed-prefix publication/readers. Training, trajectory and asynchronous telemetry paths receive corresponding protections. | Network filesystems, power loss and every possible crash point are not certified by local tests. See [migration/recovery guidance](../../development/persistence_integrity.md). |
| ARCH-10 packaging | Wheel includes runtime packages and explicit dependency profiles; optional interfaces do not invent a sibling package; standalone errors/imports are local contracts. | All optional GUI/site/engine integrations need their own evidence. |
| ARCH-11 development environment | Correct provisioned user/venv and real package install; core dependency lock and CI lanes; actual local image/install checks. | Hosted CI must run after publication. Local container success does not establish Slurm/GPU or all OS support. |
| ARCH-12 interchange | Genuine QCSchema validation for representable complete electronic results; separately named TORQ bundles; explicit units/state/method/provenance; no missing energy zero. | Actual downstream SpycFit/SPCAT consumer qualification. |

## Additional fabrication and provenance corrections

- Requested GFN2, MACE and other models no longer silently run unrelated EMT/Lennard-Jones or ad hoc potentials. Separately selected empirical calculators retain their own identity and limits.
- Anchor evaluations and delta learning require actual evaluators/trained models. ML is advisory, with no candidate deletion justified by uncalibrated uncertainty alone.
- Missing GOAT/CREST results cannot be replaced by relabeled input seeds. Energies compared across engines require matching method and unit provenance.
- Quenching preserves an unevaluated geometry proposal until an actual evaluation at those exact coordinates is supplied. A previous geometry's energy cannot be reused.
- Wavefunction arrays are not renamed into valid GBW/Hessian files. Missing wavefunctions and incompatible representations are explicit errors.
- Requested isotope masses cannot be replaced by natural-average masses, another isotope, a mass number, atomic number or carbon defaults. Unqualified radioactive-element choices require explicit isotopes.
- ZPE scaling is unscaled by default. Nonunit scaling needs exact method/basis/domain/source/target provenance; a ZPE factor does not supply vibrational fundamentals.
- Equilibrium `Be`, corrected `B0`, linear-rotor undefined constants, optional tensors and unavailable uncertainties retain their meanings. Arrow stores nulls rather than zero or negative physical sentinels.
- Removed geometry-derived invented NMR shielding and inertia-derived Raman polarizability. Those unqualified adapters report unavailable.
- Nonfinite Hamiltonians are rejected; missing anharmonic matrices/spectra are not zero-filled; low positive frequencies are not raised to an arbitrary floor. Reduced nuclear-motion models retain explicit assumptions.
- Unknown symmetry and nuclear-spin statistics are not guessed. Unqualified mixed-engine dipole parsing and malformed native Pickett writers are disabled; actual verified input decks can use the strict SPCAT runner.
- Kraitchman singular inversions reject; negative squared coordinates remain explicitly unavailable instead of becoming zero coordinates.
- Methods text no longer invents engine versions, grids, harmonic calculations, counterpoise/frozen-monomer treatment, Hamiltonian conventions, spin weights or catalog execution.
- Missing CFOUR quantities remain absent; process exit is distinct from scientific convergence. Frequencies are not used as Hessians, and Cartesian dipoles are not relabeled principal-axis dipoles.
- Hash algorithms retain their real identity. A digest is not described as a digital signature or proof that a calculation occurred.

## Evidence policy and limits

The new integrity suites import the real production modules and exercise actual numerical algorithms, files, subprocesses, SQLite transactions and HDF5/Arrow/schema operations. Exact point-mass, matrix, potential and hydrogenic examples are labeled mathematical checks. They are not simulated quantum-program runs or claimed experimental evidence. Authentic tracked engine artifacts are used for parsing; deliberately malformed/truncated inputs test rejection. No mocked engines, fabricated wavefunction fixtures or AST-extracted stand-in production code are used in the new suites.

The [executable test manifest](integrity-check-results.json) records **274 passed, zero failed/errors/skipped** in the final combined run, its exact command, per-suite counts and source/artifact hashes. The [JUnit record](integrity-tests.xml) retains individual cases. Ruff F checks, compilation, dependency consistency and whitespace checks also passed. See the [development-environment evidence](../../development/environment_validation.md) for clean-wheel and actual devcontainer verification. Historical suites include assumptions contradicted by the corrected requirements and are not represented as a fully passing suite. The focused regression suites establish the repaired behaviors only; they cannot certify every repository algorithm or prove that no further defect exists.

Exact open-source revDSD energy/response derivatives, qualified VPT2, validated LAM/PI treatments, vibration-corrected constants and high-accuracy identification remain scientific implementation/qualification work. Their complete contracts and validation sequence are specified in the [revDSD/spectroscopy protocol](../RevDSD_Spectroscopy_Protocol.md). No fresh ORCA/CFOUR/CREST/SPCAT campaign, trained-MACE benchmark, real experimental fit or astronomical identification was performed here. Unsupported paths stop explicitly rather than pretending those stages exist.

## Compatibility and upgrade notes

Stop old workers before upgrading queue lease schema. Never let an old worker acquire another attempt's token. Back up existing data; unmarked/ambiguous HDF5 files require a separately validated migration, and a killed writer can require offline recovery or copying a verified committed prefix into a new file. Existing clients must handle null/unavailable observables, partial statuses and fenced queue APIs. Native unqualified spectroscopy writers now raise explicit errors; this is a deliberate removal of unsupported output, not a certified replacement solver.

The code and documentation remain reviewable working-tree changes. This report does not record a production deployment or a scientific release.

Installed-wheel and container outcomes are also retained in [packaging-check-results.json](packaging-check-results.json). The working environment is `/workspace/.venvs/cochem-torq`; its repeatable full-validation installer and pinned package list are saved under `/workspace/.environment/cochem-torq/`. These environment files restore software checks, not external quantum-engine capability.
