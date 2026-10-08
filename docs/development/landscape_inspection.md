# Authentic candidate and sampled-landscape inspection

The student notebook displays the actual candidate ledger and retained,
approved scan results through cochem_torq.landscape_inspection. Inspection is a
read-only view of evidence. Recording, excluding, restoring, or manually retaining
a candidate remains an explicit, revision-checked ledger action.

## Public contracts and entry points

LandscapeInspection contains typed LandscapeObservation and
EnergyComparisonGroup records, source references, declared coordinates, and
coverage. The schema is cochem.torq.landscape-inspection/1. The underlying
contracts reject nonfinite numerical values and inconsistent relative-energy
references.

    from cochem_torq.landscape_inspection import (
        LandscapeFilter,
        inspect_candidate_ledger,
        inspect_scan_landscape,
        render_landscape_html,
    )

    ensemble = inspect_candidate_ledger("/path/to/candidates.sqlite3")

    # Use an independently retained digest from the original scan receipt.
    # A new digest of an untrusted file does not authenticate its origin.
    sampled = inspect_scan_landscape(
        "/path/to/retained/scan",
        expected_result_sha256=original_receipt_result_sha256,
    )

    display = render_landscape_html(
        ensemble,
        LandscapeFilter(available_energies_only=True),
    )

An absent ledger returns an explicit absent source and no observations; viewing
does not create a database. An unrelated SQLite file is rejected without
initializing or migrating it. An existing candidate ledger must pass its
selection-history integrity checks, and its snapshot must remain unchanged
during inspection.

StudentSession.inspect_ensemble() and
StudentSession.inspect_scan_landscape(...) return JSON-compatible contracts.
The widget application exposes selection-state, authenticated-energy, and
relative-energy display filters, retained scan inspection, raw references,
history, and an explicit "Retain reviewed candidate" action. Inspecting a
retained scan starts no engine process and imports no PySCF module.

Ordinary spectroscopy recipe selection excludes profiles with pes_scan or
constrained_geometry products. Those products require their dedicated
approved scan or constrained-optimization request contracts.

## Authentication and missing results

Candidate results must retain their sealed shard manifest, exact inventory,
electronic-structure stage, and original native artifact. Their observed energy
is bound to the actual native method, geometry, state, and manifest. Changed,
missing, or invalid raw artifacts remove the rankable energy and preserve the
reason and references.

For a sampled candidate, inspection follows its actual source point result and
native manifest. Its observed source recipe remains separate from its requested
refinement recipe. A sampled geometry does not acquire stationary-point status
by being retained.

Scan inspection validates the retained request, historical approved plan,
surface, complete point-file inventory, point identities, grid coordinates,
node representative policy, and native inventory. Available points must also
match the saved wavefunction density read back directly from the actual HDF5
orbital checkpoint. Engine-installation and resolved-basis records must agree
with the native observation. Historical inspection does not renew an expired
approval or authorize fresh calculations.

Every approved grid node remains visible, including genuinely failed and
uncomputed nodes. Missing energy remains null with an explicit status and
reason. No zero, guessed spectrum, interpolation, or synthetic observation
replaces it. Failed point references and physical call counts remain available.

## Energy comparability and the reference zero

Absolute and relative total electronic energies are explicitly in hartree.
Comparison groups bind:

- Registered recipe hash, actual method, and numerical settings.
- Observed engine/version, retained installation identity, adapter source, and
  actual resolved basis definition.
- Ordered atom identities, elements, isotopes, charge, and multiplicity.
- Geometry treatment and the exact scan/constraint identity where applicable.
- Electronic and nuclear energy and exactly declared dispersion; zero-point,
  thermal, entropy, and standard-state terms are excluded.

Each energy belongs to its own authenticated geometry. Different geometries
within the same bound model may be compared as sampled electronic energies.
Different groups receive separate plots and separate zeros; the interface does
not order their energies as a shared ranking.

The reference is the lowest authenticated observed energy across all recorded
selection states in that exact group. Ties use the actual observation identity
for deterministic ordering. The contract validates every relative difference
against that reference. Hiding or excluding a record does not change this
display reference; the reference identity is shown even when it is hidden by a
filter. This zero is a finite observed minimum, not a proven global minimum or
thermodynamic free-energy reference.

## Uncertainty, coverage, and visualization

Supplied inputs preserve their actual not-computed uncertainty status.
Calculated stages preserve their recorded uncertainty. Scan energies remain
uncalibrated without a separately established model-error budget. A retained
repeat-energy spread is reported only when at least two available observations
exist at the same node; it is not a model-error estimate.

Coverage reports available, failed, attempted, and uncomputed approved grid
nodes, available-grid fraction, and actual physical call count. Candidate
search coverage, continuous-domain coverage, and basin completeness remain
unknown. Electronic-root assignment is explicitly not established by this view.

The HTML view uses actual observed energy markers, with separate comparison
groups and escaped user-supplied text. The first declared coordinate is
projected; all coordinate values remain in the table. Periodic domains and
periods are labeled from the retained contract. Missing nodes appear in the
table rather than as plotted zero-energy points. No connecting interpolation
or surface is presented as an actual observation.

Display filters preserve the full inspection, reference zeros, candidate
revisions, raw references, and ledger history. They do not calculate populations
or conformational entropy, establish stationarity or independent accuracy, or
enable automatic pruning.

## Validation scope

tests/test_landscape_inspection.py uses real ledger operations, original
rejected shards, genuinely unexecuted approved plans, supplied mathematical
coordinate arrays, and actual H2 PySCF calculations. Genuine integrations cover
partial scan coverage, failed SCF, retained density readback, different recipe
groups, sampled candidate provenance, and changed native/checkpoint evidence.
The subprocess inspection check reads a retained real scan and verifies that
inspection imports no quantum engine.

These checks establish the inspection and evidence-handling behavior. They do
not qualify a spectroscopy method, benchmark conformer-search completeness, or
calibrate chemical prediction errors. Real-engine release evidence and
independent scientific validation remain separate requirements.
