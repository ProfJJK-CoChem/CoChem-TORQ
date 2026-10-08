# Reviewed TOPOS ensemble import

The mandatory CoChem ecosystem can import the versioned TOPOS 0.1.0 producer
contract without claiming that a TORQ calculation has run:

```bash
python -m Libraries.cochem_torq_topos_handoff handoff.json imported-ensemble
```

TOPOS must already have exported a reviewed, eligible ensemble. The consumer
validates its schema, review chain, exact member geometry/state/protocol hashes,
and source record. It stores the accepted producer handoff and a TORQ import
manifest in a new directory, flushes the transaction, and only then returns a
`topos-torq-consumption/0.1.0` receipt in `receipt.json`. TOPOS can accept that
receipt through its `accept_torq_receipt` API or `receive-torq` CLI command.

The import state is `imported-awaiting-calculation`. `computation_performed` is
false. An acknowledgment means that TORQ imported those exact reviewed members;
it does not establish a transition state, reaction rate, or completed dynamics.
A subsequent scientific consumer must check its own physical capabilities and
obtain the verified TOPOS research bundle if it needs original raw artifacts.
No private/local raw-file paths are automatically followed by this importer.

Identical import requests are idempotent. Changed ensembles need a new output
directory. Corrupt, incomplete, incompatible, modified or symlinked handoffs are
rejected. Durable imports have explicit file membership and content digests.
Checksums establish content consistency, not sender authentication.

The retained water fixture came from actual BASE-approved xTB 6.7.1 optimization
through a noneditable TOPOS 0.1.0 wheel (`E=-5.070544054679 Eh`). Contract tests
consume this retained fixture; they do not rerun xTB or a TORQ solver. The separate
installed-wheel acceptance performed the actual TOPOS producer → installed TORQ
consumer → TOPOS receipt-validation round trip with that genuine calculation.
The [fixture provenance](../tests/fixtures/topos_native_xtb_reviewed_water.PROVENANCE.md)
records the source revisions, original acceptance and raw artifact hashes, and
the preserved historical fixture's incompatibility with the current record profile.

Run the focused contract tests from an ordinary temporary source snapshot whose
path satisfies TORQ's sandbox guard. From the TORQ checkout, with the mandatory
CoChem packages installed in the active Linux environment:

```bash
cochem_test_snapshot=$(mktemp -d "${TMPDIR:-/tmp}/cochem_exec_torq_handoff.XXXXXX")
git archive HEAD | tar -x -C "$cochem_test_snapshot"
cd "$cochem_test_snapshot"
python -m pytest tests/test_topos_handoff.py
```

This copies committed source without creating a Git worktree or changing the
checkout. Keep the snapshot for inspection after a failure. All producer and
consumer integrity assertions remain enabled; no sandbox opt-out is needed.
