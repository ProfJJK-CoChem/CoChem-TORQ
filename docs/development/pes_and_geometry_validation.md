# Bounded geometry and PES validation

This guide describes the implemented experimental local APIs and CLI. It does
not qualify the complete SRS, an independently accurate molecular potential, or
the deployed student workflow. The geometry/PES requirements remain partial
supported scope; see the [requirement audit](full_srs_requirement_audit.json),
[release scope](student_release_scope.md) and
[remaining closure gates](release_candidate_readiness.md).

GitHub Actions remains the canonical calculation environment and Codespaces the
canonical interface. The named `hf-sto-3g-pes-validation` profile is currently
**local validation only**. It cannot be submitted through the public Actions
teaching profile, and a blocked hosted request does not authorize local fallback.
Use the genuine calculation environment and separately approve local work. Real
RDKit is required for supplied-graph checks; PySCF/geomeTRIC and their actual
declared dependencies are required for the applicable native calculations. See
[canonical environment setup](canonical_environments.md).

## Supplied graph and geometry identity

[geometry_identity.py](../../src/cochem_torq/geometry_identity.py) exposes
`indexed_geometry_from_rdkit(mol, molecule, graph_source=...)`,
`compare_indexed_geometries(source, target, policy)` and
`propose_nuclear_symmetry(geometry, policy)`. `IndexedGeometry` retains the
complete original `Molecule`, stable atom identifiers, canonical indexed graph,
graph assertion source and actual RDKit version. The supplied graph must contain
every explicit atom in coordinate-row order, with matching elements, isotope
mass numbers and total declared formal charge. Supported asserted tetrahedral
and E–Z stereo must survive the canonical round trip. XYZ coordinates do not
supply connectivity, formal charge or stereochemical assignments.

Every `GeometryComparison` retains both complete original input records and
their digests, the declared policy and evaluated atom maps. Alignment uses the
row-vector convention `target = source @ rotation + translation`, with a proper
SO(3) rotation and determinant +1. Its metric is unweighted atom RMSD in bohr;
it also records maximum atom residual and underdetermined orientations.
Reflection cannot make enantiomers equivalent.

The policy declares separate positive equivalence and distinctness thresholds,
an enumeration bound and either graph-verified permutations or authoritative
shared atom IDs. Unassigned stereo, truncated enumeration, multiple acceptable
maps and RMSD between the thresholds produce ambiguity. All candidates remain
retained, including geometrically equivalent inputs; the comparison never
automatically excludes an input or certifies scientific accuracy. Enhanced
relative, non-tetrahedral and atropisomer stereo need separate adapters.

The following creates **declared input geometries**, supplied H–H connectivity
and numerical policies. The chosen distances/tolerances are example inputs, not
observed equilibrium distances, calibrated chemical thresholds or engine output.
Use a fresh directory; it intentionally fails rather than overwrite earlier work.

```python
from pathlib import Path
from rdkit import Chem

from cochem_torq.domain import Molecule, canonical_json
from cochem_torq.geometry_identity import (
    ComparisonPolicy, SymmetryPolicy, indexed_geometry_from_rdkit,
)

workspace = Path("/tmp/cochem_exec_torq_pes_demo")
workspace.mkdir()

parser = Chem.SmilesParserParams()
parser.removeHs = False
graph = Chem.MolFromSmiles("[1H][1H]", parser)
if graph is None:
    raise ValueError("The declared example graph did not parse.")

source = indexed_geometry_from_rdkit(
    graph,
    Molecule(
        symbols=["H", "H"], isotopes=[1, 1], atom_ids=["H1", "H2"],
        geometry_bohr=[[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]],
        charge=0, multiplicity=1,
    ),
    graph_source="caller-declared connectivity; documentation input only",
)
target = indexed_geometry_from_rdkit(
    graph,
    Molecule(
        symbols=["H", "H"], isotopes=[1, 1], atom_ids=["H1", "H2"],
        geometry_bohr=[[2.0, -1.0, 3.0], [3.4, -1.0, 3.0]],
        charge=0, multiplicity=1,
    ),
    graph_source="translated documentation input; no optimization performed",
)
comparison_policy = ComparisonPolicy(
    equivalent_rmsd_bohr=1e-8, distinct_rmsd_bohr=1e-5,
)
symmetry_policy = SymmetryPolicy(tolerances_bohr=(1e-8, 1e-5))

for name, value in (
    ("source-geometry.json", source), ("target-geometry.json", target),
    ("comparison-policy.json", comparison_policy),
    ("symmetry-policy.json", symmetry_policy),
):
    with (workspace / name).open("xb") as stream:
        stream.write(canonical_json(value.model_dump(mode="json")) + b"\n")
```

```bash
TORQ_DEMO=/tmp/cochem_exec_torq_pes_demo
cochem-torq geometry-compare \
  --source "$TORQ_DEMO/source-geometry.json" \
  --target "$TORQ_DEMO/target-geometry.json" \
  --policy "$TORQ_DEMO/comparison-policy.json" \
  --output "$TORQ_DEMO/comparison.json"
cochem-torq symmetry-proposal \
  --geometry "$TORQ_DEMO/source-geometry.json" \
  --policy "$TORQ_DEMO/symmetry-policy.json" \
  --output "$TORQ_DEMO/symmetry.json"
```

Inspect the actual result rather than prescribe a favorable classification.
Identical H atoms permit multiple atom mappings, and a linear geometry has an
underdetermined rotational frame. Those facts remain visible.

## Nuclear-position symmetry boundaries

The bounded symmetry profile handles at most 24 nuclear rows. It enumerates
element-labelled nuclear-position permutations, fits proper and improper
orthogonal actions, and records residuals, set closure, enumeration completeness
and sensitivity across increasing tolerances. Isotope assertions and separately
resolved authentic isotope-mass actions are retained. Missing mass records remain
unavailable rather than using an invented mass.

Only complete, closed actions of geometry rank at least two can assign the
implemented finite proposals C1, Cs, Ci, C2, C2v, C2h or D2. Other finite groups
and linear/continuous groups remain unassigned. A nuclear-position action does
not establish a graph/stereo automorphism or dynamically feasible nuclear motion.
Engine computational subgroups, feasible permutation-inversion groups,
nuclear-spin weights and tunneling selection rules remain explicitly unavailable.
Constraint references record supplied metadata only; this API does not execute
constrained optimization or establish an unconstrained minimum.

## Approved fixed-coordinate H2 scans

[scan.py](../../src/cochem_torq/scan.py) defines `CoordinateDomain`,
`InternalCoordinate`, `ScanBudget`, `ScanPass` and `ScanPlan`.
`source_provenance.pes_scan` must contain the complete normalized plan. Declared
bond/angle/dihedral or periodic syntax does not enable unsupported execution.
The executable profile requires all of the following:

- Neutral singlet H2, explicit stable atom IDs, `hf-sto-3g-pes-validation`, and
  exactly the `pes_scan` product.
- One nonperiodic atom-0/atom-1 bond coordinate, fixed geometry, frozen unscanned
  Cartesian coordinates and no additional constraints.
- Independent `pyscf_minao` initial guesses; density continuation is unavailable.
- An explicit finite grid, ordered forward and reverse passes, independent
  challenge indices and finite actual-call/per-point/total resource ceilings.

The full-grid example below declares three distinct design nodes and nine
physical calls: three forward, three reverse and three challenge evaluations.
Its 20-second point ceiling and 180-second total ceiling are approved limits,
not measured timings. No point is asserted stationary or optimized. The example
values are chosen input coordinates and discrepancy tolerances, not observations
or an independent accuracy target.

```python
from pathlib import Path

from cochem_torq.domain import CalculationRequest, Molecule, Resources, canonical_json
from cochem_torq.scan import (
    CoordinateDomain, InternalCoordinate, ScanBudget, ScanPass, ScanPlan,
)

workspace = Path("/tmp/cochem_exec_torq_pes_demo")
scan = ScanPlan(
    coordinates=(InternalCoordinate(
        coordinate_id="H1-H2", kind="bond", atom_indices=(0, 1), unit="bohr",
        domain=CoordinateDomain(minimum=1.2, maximum=1.6, periodic=False),
    ),),
    grid=((1.2,), (1.4,), (1.6,)), sampling_strategy="full_grid",
    passes=(
        ScanPass(purpose="forward", sample_indices=(0, 1, 2)),
        ScanPass(purpose="reverse", sample_indices=(2, 1, 0)),
        ScanPass(purpose="challenge", sample_indices=(0, 1, 2)),
    ),
    coordinate_treatment="fixed", unscanned_coordinates="frozen_cartesian",
    additional_constraints=(), initial_guess_policy="independent_pyscf_minao",
    budget=ScanBudget(max_physical_calls=9, per_point_wall_seconds=20),
    energy_recheck_tolerance_hartree=1e-9, density_recheck_tolerance=1e-8,
)
request = CalculationRequest(
    molecule=Molecule(
        symbols=["H", "H"], isotopes=[1, 1], atom_ids=["H1", "H2"],
        geometry_bohr=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]],
        charge=0, multiplicity=1,
    ),
    recipe="hf-sto-3g-pes-validation", products=["pes_scan"],
    resources=Resources(cores=1, memory_mb=512, wall_seconds=180),
    source_provenance={"pes_scan": scan.model_dump(mode="json")},
)
with (workspace / "fixed-request.json").open("xb") as stream:
    stream.write(canonical_json(request.model_dump(mode="json")) + b"\n")
```

Planning does not start chemistry. Inspect exact source/recipe/request identity,
task counts, budgets, limitations and blocking reasons before the explicit
approval command. The actor string must identify the actual caller; it does not
authenticate a remote identity.

```bash
cochem-torq doctor --execution local_validation
cochem-torq plan --request "$TORQ_DEMO/fixed-request.json" \
  --execution local_validation --output "$TORQ_DEMO/fixed-plan.json"
# After reviewing the actual plan:
cochem-torq approve-plan --plan "$TORQ_DEMO/fixed-plan.json" \
  --actor "local-os:$(id -u)" --scratch-mb 64 \
  --output "$TORQ_DEMO/fixed-approved.json"
cochem-torq execute --request "$TORQ_DEMO/fixed-request.json" \
  --approved-plan "$TORQ_DEMO/fixed-approved.json" \
  --output-dir "$TORQ_DEMO/fixed-scan"
```

The equivalent API is
`execute_scan_request(request, destination, approved_plan=...)`.
`ApprovedScanExecutor(approved_plan, destination)` exposes bounded individual
`evaluate(sample_index, purpose=..., parent_point_ids=...)`, `run()` and
`finish()` methods. Approval/source/lease/resource checks apply to genuine calls;
outcomes retain failed SCF, timeout, resource, collision and admission errors.
Every physical restart counts against the call budget, including repeat nodes.

The workspace retains the original request and approval, coordinator events and
accounting, individual point results and native manifests/logs/checkpoints.
`result.json` is a `ScanSurface` with actual absolute electronic energies and
native AO density observations, unavailable masks, point/pass lineage and
energy/density recheck comparisons. Ordered independent restarts do not implement
continuation hysteresis or electronic-branch tracking. `complete` means the
declared calls succeeded without the declared recheck discrepancies; it does
not establish a complete surface, a minimum or chemical accuracy.

A scan surface workspace is not the ordinary sealed spectroscopy-worker shard
accepted by `cochem-torq verify`. Preserve its original native inventories and
point records. Candidate refinement authenticates those original bytes and
paths before using an observation. Native relocation needs a future authenticated
import contract; copied summaries or merely schema-valid JSON cannot replace it.

## Finite-design adaptive acquisition

[adaptive.py](../../src/cochem_torq/adaptive.py) defines `AdaptiveScanPlan` and
`run_adaptive_scan(executor, plan, ledger=..., actor=...)`. The approved request
uses `sampling_strategy="bounded_adaptive"`, an explicit finite H2 bond grid,
empty fixed-pass schedule and a declared maximum number of physical calls.
`source_provenance.adaptive_scan` contains the complete normalized adaptive plan
whose `scan_sha256` matches the exact scan definition.

The policy specifies distinct seed indices, disjoint frozen interior held-out
indices, maximum normalized finite-design distance, numerical interpolation
residual tolerance and an explicit `ScientificGoal`. Its sole error target is
`numerical_interpolation_energy` in hartree with the same tolerance; the reference
policy is `de_novo_no_calibration`. Those are declared within-design numerical
criteria, not chemical-accuracy claims. The
[actual adaptive test input](../../tests/test_adaptive_pes_integrity.py) shows
the complete typed goal/scan/policy construction.

The implemented acquisition uses deterministic maximin coordinate diversity and
genuine approved seed/anchor calls. Piecewise-linear interpolation is labeled a
numerical estimate, never an observed energy, trained surrogate or calibrated
uncertainty. Fresh endpoint restarts and the frozen interior challenges are
evaluated physically. No candidate is automatically pruned. The explicit stops
are `criteria_met`, `budget_exhausted`, `insufficient_coverage` and `failed`.
`criteria_met` concerns inspected finite-design spacing and numerical residuals
only; it supplies no between-node bound, global basin completeness or family-held-
out calibration.

The same `plan` → `approve-plan` → `execute` CLI sequence applies to a complete
adaptive request, with a new plan/approval/output directory. Generic execution
retains `adaptive-result.json`, `candidate-ledger.sqlite`,
`candidates/<actual-candidate-UUID>.json` and
`candidate-requests/<actual-candidate-UUID>.json`, alongside actual scan evidence.
Read the recorded stop reason and failures before considering an extremum.

## Separate candidate review and target verification

A sampled minimum/maximum or lowest observed point starts as an immutable,
quarantined **input-only** ledger candidate. Its original native point result,
source/recipe/state/atom/isotope/constraint identities and paths remain bound.
It is not yet a stationary point. Choose an actual candidate UUID from the saved
adaptive result and inspect the current ledger record/history:

```bash
# Set these to the actual retained adaptive workspace and recorded candidate UUID.
TORQ_ADAPTIVE=/absolute/path/to/your/adaptive-scan
TORQ_CANDIDATE_ID=actual-recorded-candidate-UUID
cochem-torq candidates inspect \
  --ledger "$TORQ_ADAPTIVE/candidate-ledger.sqlite" \
  --candidate-id "$TORQ_CANDIDATE_ID"
```

After explicitly reviewing release of the scan's fixed-coordinate constraints,
retain the candidate using the **current** inspected revision, actual caller
and a reason. Use the returned revision for refinement. A concurrent selection
change invalidates the old compare-and-swap revision; inspect and review again.
`restore` reverses an exclusion and does not replace this retention review.

```bash
TORQ_CURRENT_REVISION=2  # Replace with the current inspected revision.
cochem-torq candidates retain \
  --ledger "$TORQ_ADAPTIVE/candidate-ledger.sqlite" \
  --candidate-id "$TORQ_CANDIDATE_ID" --revision "$TORQ_CURRENT_REVISION" \
  --actor "local-os:$(id -u)" --reason "Reviewed fixed-coordinate release for target optimization and Hessian"
```

Export the candidate's exact fresh target request, create a separate plan and
review its costs/constraint release before separately approving it. The initial
supported target is the matching HF/STO-3G teaching recipe with `geometry` and
`harmonic` products. This contract does not permit silently changing electronic
state or choosing an unrelated higher-level method.

```bash
cochem-torq candidate-request \
  --candidate "$TORQ_ADAPTIVE/candidates/$TORQ_CANDIDATE_ID.json" \
  --output "$TORQ_DEMO/target-request.json"
cochem-torq plan --request "$TORQ_DEMO/target-request.json" \
  --execution local_validation --output "$TORQ_DEMO/target-plan.json"
# After reviewing this separate target plan:
cochem-torq approve-plan --plan "$TORQ_DEMO/target-plan.json" \
  --actor "local-os:$(id -u)" --output "$TORQ_DEMO/target-approved.json"
```

Set `TORQ_RETAINED_REVISION` to the revision returned by the successful retain
operation, then execute the explicitly approved refinement:

```bash
TORQ_RETAINED_REVISION=3  # Replace with the actual returned revision.
cochem-torq refine-candidate \
  --candidate "$TORQ_ADAPTIVE/candidates/$TORQ_CANDIDATE_ID.json" \
  --ledger "$TORQ_ADAPTIVE/candidate-ledger.sqlite" \
  --revision "$TORQ_RETAINED_REVISION" --actor "local-os:$(id -u)" \
  --approved-plan "$TORQ_DEMO/target-approved.json" \
  --output-dir "$TORQ_DEMO/target-shard" \
  --verification-output "$TORQ_DEMO/target-verification.json"
cochem-torq verify "$TORQ_DEMO/target-shard"
```

Before any target calculation, `validate_candidate_refinement(...)` rechecks
current retained ledger revision, immutable supplied-input origin, exact request,
original native and point-result hashes/paths, successful physical observation,
matching model and exact current approval. The CLI performs this check before
`execute_request(...)`. `verify_candidate_minimum(...)` then independently
verifies the complete actual shard and exact approved target, converged geometry,
actual gradient, positive projected vibrational Hessian and declared
external-invariance gate. Missing, failed or mismatched evidence cannot create a
verified-minimum result.

Success registers a **separate** verified-result ledger record and retains the
original grid candidate unchanged. Keep the verification JSON outside the
immutable shard; all output destinations must be fresh. This is a minimum of
the declared local electronic model, not experimental geometry, an accurate
spectrum, a verified transition state/IRC or global search completeness.

## Verification scope

The relevant suites are [geometry identity](../../tests/test_geometry_identity_integrity.py),
[fixed scan integrity](../../tests/test_pes_scan_integrity.py) and
[adaptive/candidate integrity](../../tests/test_adaptive_pes_integrity.py).
They distinguish mathematical/graph contracts from genuine native calculations.
Source presence or declared example inputs are not execution evidence. New final
aggregate counts and source-bound receipts are supplied separately by the release
report; prior tested baselines cannot qualify later changes.

General relaxed/constrained, periodic/coupled/multistate surfaces, density
continuation, electronic-branch tracking, trained-surrogate applicability,
global recall/calibration, feasible PI/spin/tunneling dynamics and independently
accurate molecular identification remain outstanding. The
[spectroscopy guide](spectroscopy_implementation.md) records the separately
implemented harmonic and geometric rovibrational stages and their missing
higher products.
