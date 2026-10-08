# Native hosted controller and scientific source identities

The BASE student-project workflow owns its Actions controller commit. Its
scientific TORQ checkout is independently pinned by the controller's
`scripts/module-distribution.json`. These commits normally differ. TOPOS or
TORQ calculations do not acquire scientific provenance from the BASE controller
commit alone.

TORQ submission reads actual GitHub commit/tree objects and exact catalog blob
bytes at the immutable controller commit. It retains the controller repository
and commit, scientific repository/commit/tree, and catalog and TORQ-spec hashes
in the owned submission receipt. A source-checkout approval must match the
scientific commit. An installed-wheel approval continues to bind the actual
implementation through its reviewed `code_sha256` rather than an unrelated
ancestor Git directory.

For BASE-hosted retrieval, the completed run must match the owned receipt's
controller, request UUID, workflow and manual dispatch. The receiver rederives
the immutable source binding from GitHub and verifies the original request and
approved-plan bytes, controller/source sidecar, request transport record,
scientific revision record, and the sealed scientific shard. Existing request,
recipe, implementation, molecular identity and artifact-inventory checks remain
required. The controller's source-byte inventory hash is retained as execution
evidence; the independently checked Git tree and reviewed implementation hash
provide its source binding. These checks establish integrity and ownership, not
scientific accuracy or engine qualification.

The native TORQ repository workflow keeps the existing same-commit protocol:
its workflow commit is its scientific commit. Unknown controller layouts,
incomplete Git inventories, incompatible source pins and missing owned receipts
for BASE-hosted results fail explicitly. Older owned BASE submission receipts
can be read by rederiving their binding from their original immutable controller
commit; original request and approval hashes remain mandatory.

The native transport validator receives `TORQ_SCIENTIFIC_SOURCE_SHA` from the
BASE workflow's separately resolved TORQ revision. It validates approval source
identity against that scientific revision while preserving the controller SHA
in the original request transport receipt. Direct TORQ execution omits this
variable and uses its owning workflow commit.

API requests and artifact downloads use the same existing private GitHub CLI
authentication selection. In Codespaces, automatic selection uses the student's
stored CLI login; explicit environment selection remains available. Actions
retains its owning job's authentication. Credential values are never read or
included in result diagnostics.

Focused tests translate real local Git objects into the source validator's
input format and check genuine HF shards and publication exports, including
separate source identities, altered original bytes and unsafe retained files.
They contain no simulated GitHub API, workflow run or artifact download. These
local checks do not establish live submission, authentication selection or
retrieval. A real student-project Actions journey remains unrun while student
deployment is on hold.
