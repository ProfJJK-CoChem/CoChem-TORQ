# Publication reproducibility exports

`export_publication_bundle(shard, destination)` implements the reproducibility
packaging portion of UQ004. It verifies a real sealed TORQ shard, every retained
native engine file and engine manifest, and the matching numerical-analysis
source. It publishes an immutable directory by a locked atomic rename, after
bounded artifact-store quota admission. An existing destination is never replaced.
The caller should use a dedicated artifact-store parent and retain the returned
publication manifest SHA-256 independently.

The corresponding CLI is `cochem-torq export ACTUAL_SHARD --format publication
--destination FRESH_DESTINATION`. `--json` returns the normal versioned CLI
response containing the same receipt and original request ID.

```python
from cochem_torq.publication import export_publication_bundle, verify_publication_bundle

receipt = export_publication_bundle("actual-run", "publication-artifacts/run-001")
verify_publication_bundle(
    receipt["directory"], expected_manifest_sha256=receipt["manifest_sha256"]
)
```

The bundle contains unchanged `shard/` bytes, `publication.json`, a complete
`publication-manifest.json` inventory, `README.md` replay instructions and
`analysis/source/` with a hash-bound source inventory. These analysis sources are
the actual installed calculation/analysis modules, including their identity-bound
isotope and campaign helpers. They are a source snapshot, not an invented
standalone engine installation or a new calculation. The currently installed
scientific code SHA must equal the result's sealed SHA before source packaging;
a changed or missing source edition fails with instructions to obtain the
matching legitimate installation. Portable verification subsequently checks the
retained source and shard bytes without replacing them with a viewer's local code.

The summary preserves the initial and available equilibrium geometries, charge,
multiplicity, source atom IDs and isotope selection, actual resolved isotope
records, exact recipe, numerical settings, observed native engine/library/build
identity, native basis definitions, convergence/stability records, errors,
unavailable stages and quality flags. The original complete records remain in the
shard. Citations are copied verbatim only from the recorded recipe or native
engine references; absent references remain absent. Export does not independently
verify those bibliographic claims.

A complete calculation is labeled **exploratory**, an incomplete calculation
**partial**, and an actual failed calculation **failed evidence only**. None is
labeled publication validated. Calibration evidence and calibrated uncertainty
remain explicit null values because these version-1 profiles have no independent
qualification gate establishing them. Declared scientific goals and error targets
are preserved as declarations, not converted into measured accuracy. DOI,
identification readiness and experimentally established accuracy are not granted
by packaging. Inventory digests establish consistency, not source signatures.

Replay instructions require a new compatible request and request ID, an explicit
plan review and approval, matching legitimate source and recorded engine
dependencies, and an already activated canonical GitHub Actions workflow. The
GitHub client uses `COCHEM_TORQ_GITHUB_REPOSITORY` and
`COCHEM_TORQ_CALCULATION_REF`; the approved plan path, a new `--idempotency-key`
and a fresh `--destination` are required for `cochem-torq run`. The
exporter does not submit calculations, deploy a site, modify original artifacts,
publish a DOI or silently activate a default-branch workflow. A replay and an
independent benchmark/peer-review campaign remain separate steps.

`tests/test_publication_integrity.py` exercises actual application rejections,
filesystem corruption, quota admission and concurrent OS processes. Its three
`real_engine` tests generate a genuine PySCF 2.14.0 HF/STO-3G H₂ optimization and
Hessian, preserve native checkpoints and source identity, and reject damaged
native bytes. Those release checks must run in the actual calculation environment;
software-only passes do not qualify an unrun engine or establish spectroscopy
accuracy.
