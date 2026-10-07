# Bounded rotational-identification benchmark proposal

[preregistration.json](preregistration.json) implements the **D10 proposal**, not a completed preregistration or an executed benchmark. It has no experimental reference values and no fabricated benchmark outcomes. It cannot satisfy `V-BENCH` or qualify an identification claim in its current state. The SRS requires curator-approved references, accepted numeric targets and an immutable registration digest before held-out evaluation.

## Proposed scope and split

The proposal contains ten primary parents and at most fourteen isotopologue cases. No structure or numerical reference is assumed merely because its candidate name appears here.

| Partition | Chemical families | Proposed parents |
|---|---|---|
| Development | Hydrides, diatomic carbonyls, aldehydes | Water, CO, formaldehyde, acetaldehyde |
| Calibration | Halomethanes | Methyl fluoride, difluoromethane, fluoroform |
| Held out | Nitriles, ketenes | HCN, acetonitrile, ketene |
| Applicability challenges | Inversion, internal rotation, weak binding/tunneling | Ammonia, methanol, water dimer, HF dimer |

All isotopologues, conformers, vibrational states and transitions of a chemical family remain within its partition. The nitrile family, for example, includes both HCN and acetonitrile; their relationship must not be hidden by a linear/asymmetric-top category split. A changed family definition requires an amendment before evaluation. Development or calibration error cannot be reported as held-out performance.

The challenge cases test appropriate branching, documented limitations and truthful retained partial results. They cannot be used to certify VPT2 for large-amplitude motion. A successful equilibrium calculation is a valid earlier result even when the nuclear-motion or identification model is unavailable.

This deliberately small initial benchmark has only one calibration family and two held-out families. Three calibration parents and three held-out parents cannot establish general 95% uncertainty coverage or broad chemical transfer. Report the actual parent/family counts, observed errors and uncertainty of any estimated coverage. Twenty lines from one parent are not twenty independent molecular validation cases.

## Four distinct observables

- **Equilibrium constants:** `Ae, Be, Ce`, or the correct linear representation, need independently justified equilibrium references. A semi-experimental equilibrium structure incorporates an explicit correction model and experimental anchors; label it accordingly. Measured/fitted `B0` must never be used as an uncorrected `Be` reference.
- **Ground-state constants:** `A0, B0, C0` require exact isotope, vibrational/tunneling state and Hamiltonian identity, plus the measured/fitted source and uncertainty/covariance status.
- **Transitions in a selected band:** Curate actual measured line positions and quantum assignments. Freeze frequency band, J/Ka limits, assignment/blend exclusions and up to twenty selected lines per isotopologue before inspecting predictions. A catalog's calculated line is not an independent measured line.
- **Relative intensities:** Compare only compatible measurement temperature, instrument response, population treatment and normalization. Experimental absolute intensities require their own physical calibration; a catalog model value cannot supply that evidence. Missing conditions make this observable unavailable rather than zero.

Absolute de novo predictions, experimentally anchored inference and isotope/conformer differences have separate score records. An anchor used to fit or shift a prediction is excluded from independent scoring. Shared sources and covariance must remain visible.

## Reference curation and registration

CDMS and JPL appear only as candidate provider locators. They were not retrieved for this proposal, and their inclusion supplies neither data nor a verified primary citation. Every eligible reference requires a primary source locator, retained raw reference and checksum, reuse permission, isotope/state/conformer identity, units, uncertainty and Hamiltonian convention. Source parsing must distinguish measured and predicted entries. A checksum establishes retained-byte identity; it does not prove authenticity, measurement quality or permission.

Predictive target fields are intentionally `null` and marked **proposed, pending domain/reference/pilot justification**. A universal percentage target would be unsupported. After curating references and conducting a numerical/cost pilot only on development cases, record proposed numeric targets with the intended identification context, measurable uncertainty and a scientific rationale. Accept and freeze them before any held-out evaluation. Null targets forbid benchmark pass/fail and release qualification. The existing SRS `V-ROT` relative numerical-kernel tolerance of `1e-10` remains a numerical implementation criterion, not an experimental spectroscopy accuracy guarantee.

The machine-readable record contains a `proposal_sha256`, computed over canonical compact sorted JSON before that field is added. It also records the reviewed SRS file digest. These identify this proposal and specification snapshot. The accepted registration needs a separate frozen digest and acceptance record; neither is supplied by changing its status string. SHA-256 is a digest, not a digital signature.

## Actual evidence runner contract

A benchmark execution must refuse to qualify a result until the following exist:

1. An accepted, frozen registration with all enabled numeric targets and budgets.
2. Curator-approved real references and the fixed split/line-selection manifest.
3. Exact qualified engine, recipe, basis, state and derivative/property tuples for each enabled stage.
4. Authentic calculation requests, raw outputs, engine image/dependency manifests and numerical convergence/state diagnostics.
5. Predictions sealed before held-out scoring, complete failure/exclusion accounting, and scored observable-specific residuals.

The `actual_evidence_runner_contract` in the JSON defines the required evidence fields and outcomes. It is a contract for a genuine runner, not a claim that this benchmark has executed. A local parser test, a mathematical limit, a PySCF teaching run or a successful image build cannot qualify the full method. Preserve failed, blocked and budget-incomplete attempts in reports; never change a missing observation to a number.

GitHub Actions is the canonical calculation environment, using individually qualified worker images; Codespaces provides the interface. A larger method-development campaign requires a separately accepted measured budget and authorized worker profile. The proposed teaching limits are two cores, 4096 MB and 1800 seconds per job, at most 64 jobs. Exact revDSD and anharmonic budgets remain unset until complete-energy/displacement costs are measured. Exceeding the budget produces an incomplete scientific goal with reusable earlier artifacts.

## Present release blockers

Reference curation, accepted predictive targets, registration freeze, qualified complete spectroscopy profiles and held-out execution remain outstanding. See [student_release_scope.md](../../docs/development/student_release_scope.md) for requirement-to-code/evidence mapping. No scientific benchmark pass or student identification accuracy is asserted by this proposal.
