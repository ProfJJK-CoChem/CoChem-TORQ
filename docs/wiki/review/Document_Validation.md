# Specification validation and evidence report

Reviewed source commit: `d7a4739a5f7d6f22ed659b32eeb4706bef16225e` from `ProfJJK-CoChem/CoChem-TORQ` main. The clean local checkout was fast-forwarded to this commit before review. All changes made by this review are under `docs/wiki`; application logic, dependencies and tests were not edited.

## Coverage

The [source inventory](SRS_Point_Inventory.json) contains 601 uniquely identified records: 223 distinct numbered sections (with repeated source occurrences retained), 25 explicit contracts, 345 improvement entries and eight architecture pillars. The Comprehensive source has 165 checkbox entries: 145 substantive entries and 20 placeholders. The Iterated source has 180 entries. Placeholders are explicitly identified; they are not counted as implemented ideas.

The [method audit](Method_Matrix_Review.md) includes all twenty revision claims, all sections, R1–R9 and all 140 distinct method-table row IDs. The [canonical SRS](../CoChem-TORQ_Implementation_SRS.md) defines 70 uniquely identified requirements, 40 acceptance-test definitions and work packages WP00–WP12. Twelve material decisions include benefits, costs, risks, recommendations and interim behavior.

## Reproducible document checks

From the repository root:

```bash
python docs/wiki/review/validate_spec.py
git diff --check
```

The validator checks source hashes/line spans, complete checkbox occurrence coverage, unique inventory IDs, byte-for-byte preservation of nine historical sources below their review banners, all 140 method rows, R1–R9, unique requirement IDs, defined acceptance IDs, requirement-to-acceptance mappings, D01–D12 fields, relative file links, fenced JSON examples and code-fence balance. Direct arithmetic checks cover the corrected conformal-interval example and campaign-cost example. Its current machine-readable result is [document-check-results.json](document-check-results.json).

These checks establish document coverage and internal structure. They cannot establish the correctness of every scientific assertion or replace a peer review. Independent scientific and architecture passes reviewed the revised contracts and led to additional corrections for constrained-Lagrangian Hessians, exactly linear versus quasi-linear molecules, monatomic fragments, degenerate modes, charged-dipole origins, validated surrogate eligibility, fencing and atomic budgets.

## Implementation evidence and limits

The implementation audit includes two focused software checks: unchanged parser logic extracted for isolated execution on deliberately incomplete/example text, and the actual SQLite queue exercised with a temporary database and stale worker completion. Results are in [architecture-focused-checks.json](architecture-focused-checks.json). They demonstrate software defects; the parser inputs are not physical quantum results. Static inspection supplies the remaining code findings, with exact baseline locations and evidence levels.

No full application test suite, live quantum engine, GPU, Slurm, UI or SpycFit integration was run during this documentation task. Earlier environment-onboarding test results applied to an older commit and do not certify the newly retrieved program. Seventy requirements and forty test definitions are specification content, not passed application tests.

The [reference ledger](Scientific_References.md) records fresh primary-source checks and immutable upstream revisions. Some vendor/journal destinations were inaccessible through the configured network policy; in particular, full CFOUR manual validation remains outstanding. Unverified source claims are not promoted to verified capabilities. Upstream source inspection is not numerical validation of TORQ adapters.

## Completion assessment

The revised specification, corrected method contract, exhaustive source review and decision register are ready for implementation planning and scientific decision review. Core implementation can proceed under the defined contracts. Production deployment and publication-level result claims remain blocked by the documented code defects, unexecuted capability/benchmark gates and applicable open decisions. No software implementation or deployment is claimed by this documentation revision.


## Follow-up validation

The canonical specification now has 73 requirements and 41 acceptance definitions, including the full typed spectroscopy stages. The executable document check also covers all 24 three-option decision questions and 29 unique claims-register entries. Historical document bodies remain byte-preserved. `document-check-results.json` records the current check; these counts describe specification coverage, not scientific implementation or engine qualification.
