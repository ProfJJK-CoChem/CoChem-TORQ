# CoChem-TORQ specification and review

Start with the [implementation SRS](CoChem-TORQ_Implementation_SRS.md). It replaces the conflicting historical SRS drafts as the coding contract and defines scope, scientific invariants, typed interfaces, engine capabilities, workflow/state machines, storage, deployment and acceptance tests.

| Document | Purpose |
|---|---|
| [Implementation SRS](CoChem-TORQ_Implementation_SRS.md) | Canonical requirements and work packages. |
| [Method matrix implementation contract](Method_Matrix_Implementation_Contract.md) | Correct equations, method/row/budget identity and recipe activation gates. |
| [Decisions and risks](review/Decisions_and_Risks.md) | Twelve material choices, benefits, costs, risks and recommended interim behavior. |
| [SRS point review](review/SRS_Point_Review.md) | All 14 chapters, 25 contracts, 345 improvement entries and eight architecture pillars; duplicate section occurrences retained. |
| [Source inventory](review/SRS_Point_Inventory.json) | Machine-readable 601-record source/disposition inventory. |
| [Method matrix review](review/Method_Matrix_Review.md) | Every section, all 20 revision claims, R1–R9 and all 140 table rows. |
| [Implementation audit](review/Architecture_Implementation_Audit.md) | Existing-code evidence, release blockers and architecture gaps. |
| [Scientific references](review/Scientific_References.md) | Verified primary-source evidence, immutable revisions and unverified claims. |
| [Validation report](review/Document_Validation.md) | Structural/coverage checks and limits of this review. |
| [Coding handoff](Handoff_Coding_Manifest.md) | Required reading and implementation sequence. |

Reviewed repository baseline: `d7a4739a5f7d6f22ed659b32eeb4706bef16225e`. Source line references in the reviews are pinned to that commit. Low-risk improvements are implemented in the specification; material scientific choices remain gated. The current program is not certified deployable or publication-ready by these documents.

Historical sources remain available: [Comprehensive SRS](CoChem-TORQ_Comprehensive_SRS.md), [True SRS](CoChem-TORQ_True_SRS.md), [Iterated SRS](CoChem-TORQ_True_SRS_Iterated.md), [AutoArchitect](CoChem-TORQ_AutoArchitect_Phase1.md), [integration architecture](CoChem-TORQ_Integration_Architecture.md), [Method Matrix](Method_Matrix.md), [user manual](CoChem_User_Manual.md), [BASE blueprint](CoChem-BASE_Draco_Blueprint.md), and [NBO notes](NBO_Analysis.md). They preserve research history; contradictory commands, scope claims and placeholders are not implementation authority.

TOPOS owns global conformer discovery; TORQ refines supplied candidates and selected coordinates; BASE provides shared infrastructure; SpycFit consumes validated spectroscopic artifacts. Cross-repository contracts require actual consumer validation.

## Follow-up integrity and scientific decisions

- [Decision questions: 24 questions with three options each](review/Decision_Questions.md)
- [Exact open-source revDSD and full spectroscopy protocol](RevDSD_Spectroscopy_Protocol.md)
- [29 specific unverified, unsupported or conflicting claims](review/Unverified_Claims_Register.md)
- [Integrity repairs, verification and remaining scientific gates](review/Integrity_Repair_Report.md)
- [Cloud/development setup and real installation evidence](../development/environment_validation.md)
