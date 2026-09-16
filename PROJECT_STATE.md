# Project State

Last updated: 2026-09-16

## Current stage

Stage 6.4 sparse five-point linewidth/Q tracking has an approved design and a
twenty-task detailed TDD implementation plan whose final review is closed and
which is in task-by-task execution. Tasks 1–19 now provide the immutable sparse
record layer, canonical bound-source model, pure fit geometry, ordered fit
outcomes, exact fast-pair transitions, the first due-scan scheduler boundary,
and complete sparse transitions through point five, including scientific fit
application, live projection, independent epochs, resources, and update CPU.
Pure fit geometry remains separate from scheduler-owned query clocks, and
evaluator value types, the exact sparse runner shell/clean bind boundary,
runner-neutral exact-identity verified-calibration authority, and authenticated
calibration/start transitions. Initial resources, accepted resource
integration, and terminal behavior then form a forward-only runner chain.
Production truth lookup and public linewidth-dynamics additions remain absent:
deterministic linewidth drift and release-gated truth evaluation now exist only
as closed test support. The plan uses configured
`max_nfev`, stops fit CPU timing before result
construction, and keeps private authority tests in the existing calibration
test module. No Stage 6.5 claim has been added. Final plan review also closes
resource-builder phase semantics: pre-start calls raise the public state error,
public evaluator resources begin only after successful tracking start, and
`None` is exclusive to an unavailable terminal resource join. Timing isolation
uses a query-scoped test dynamics spy rather than a nonexistent instrument
counter, without pinning configurable quadrature call counts.

Task 10 accepts the fifth and only the fifth already-reserved sparse query,
passes the exact frozen five-query/five-observation tuples to the isolated
fitter once, and treats both success and scientific failure as completed,
charged scans. Completion appends the exact returned scan object, advances the
global and per-ID scan counters/parity, clears the partial reservation, resets
the fast-pair cadence, and preserves every fast-center value/source field.
Success alone refreshes active FWHM and its public/reference release epochs;
failure retains the prior FWHM source while all identities age to the fifth
endpoint. Live Q remains the finite signed/zero asynchronous fast-center/FWHM
projection, with nonrepresentable division translated transactionally as
`aggregate_estimate_construction_failed`. Calibration, fast, sparse,
interleaved tracking, and charged resources retain independent arrival-order
folds for both budget treatments. Accepted process CPU starts before fitting,
ends after identity/resource construction, and updates the sparse subtotal and
global total independently. Exact fifth-point validation, every construction
and clock boundary, ordinary-exception rollback, and identical process-control
propagation are regression-pinned. Evaluator/runner behavior and Stage 6.5
claims remain deferred.
Task 10's independent review found the production transition conformant and
identified three test-evidence gaps. The closed matrix now drives negative,
negative-zero, positive-zero, and positive live Q through both successful and
scientifically failed fifth-point completion; exercises all eight legal
scientific-failure diagnostic-presence shapes, including the five with fitted
widths; and proves that no failure refreshes the active width source. Ordinary
and identical process-control rollback now also cover metadata, private state,
identity construction calls two through eight, and resource construction calls
two and three. Targeted mutation REDs pin signed-zero preservation, failure
source aging, exact typed exception translation, and commit-last rollback.
The pushed Task 10 review head `b3c83ab` passed the complete native GitHub
Linux x86 matrix on Python 3.11 and Python 3.12 (run 35023707270).

Task 11 adds the sparse evaluator's exact closed phase, abort, preflight, and
start-code aliases plus its frozen/slotted acquisition, query-failure, abort,
scan-timing, resource, runner-state, and typed outcome records. Intrinsic
constructors canonicalize scalar and string values, defensively snapshot every
public sequence, enforce authenticated versus unavailable resource joins,
preserve the exact five-midpoint truth fold, close the complete phase/presence
and outcome-kind matrices, and require evaluator CPU totals to equal the
tracker snapshot's independent fast, sparse, and interleaved totals. The
package exports only this approved value surface: it defines no runner class,
resource builder, provenance authority, or state transition behavior.
Task 11's revised independent review withdrew broader replay and state-history
findings as Task 14–15 responsibilities and identified one intrinsic
resource-cardinality gap. Evaluator resources now require the incomplete fast
count to equal the accepted-fast tuple remainder modulo two, the incomplete
sparse count to equal the accepted-sparse tuple remainder modulo five, and the
unaccepted count to equal the unaccepted tuple length. Exact tuple-type, count-
range, and mutual-exclusion precedence is preserved; resource replay,
accepted-stream partitioning, and trace/timing joins remain deferred.
The pushed Task 11 review head `1595c9c` passed the complete native GitHub
Linux x86 matrix on Python 3.11 and Python 3.12 (run 35027274372).

Task 12 adds the exact seven-signature `SparseLinewidthEvaluatorRunner` shell.
Binding accepts only an exact clean virtual instrument, snapshots its immutable
configuration and zero resource boundary into a frozen `ready` state, mints one
registered token, and exposes state read-only; calibration/start and tracking
transitions remain explicit typed placeholders for Tasks 13 and 15–16. The
private verified-calibration transaction is now runner-neutral and accepts only
an unforgeable issuer authenticated against exact runner, instrument, token,
and configuration identities. Registration exact-allowlists the live two-point
and sparse runner classes rather than forward names. Public construction,
subclassing, copying, serialization, `object.__new__`, detached identities,
unregistered exact instances, and runner subclasses cannot obtain authority.
Token-indexed revocation remains correct even if a commit-then-raise path
mutates both the public binding record and issuer slots. The Stage 6.3 runner
now delegates through this issuer without changing any public signature,
calibration outcome, resource, rollback, or tracking trace. The sparse package
`__all__` intentionally remains the Task 11 value surface until Task 19 owns
the complete public export expansion.
The pushed Task 12 head `0a080fa` passed the complete native GitHub Linux x86
matrix on Python 3.11 and Python 3.12 (run 35028977742).

Task 13 delegates sparse verified-calibration acquisition through Task 12's
private exact-issuer core and translates only its ordered public preflight codes
to the sparse error surface. Successful and typed-failure outcomes retain the
exact sparse runner state, full/safe observations, resources, sequence, and
virtual-time boundary; commit-then-fail private success binding is revoked
before the typed failure state is published. Tracking start authenticates the
exact verified outcome, source, calibration, runner, instrument, token, clock,
metadata, treatment, and current resource boundary in declared code order
before touching the tracker. It permits same-run acquisition and an exact
other-runner conditional source, restores all three tracker slots on ordinary
or process-control reset failure, and commits an empty tracking audit state
whose calibration, estimate, resources, and independent CPU totals are the
exact reset products. Step/resource/terminal transitions remain Tasks 14–16.
Task 13's independent review found that conditional other-runner start joined
the outcome/source pair but did not authenticate the source runner's complete
live private identity graph. The fix now requires the exact registered issuer,
runner, instrument, instrument configuration, token, successful phase, retained
calibration outcome, retained verified outcome, empty tracker slot, and binding
registry entry to agree before metadata checks or reset. Ten adversarial REDs
cover mutated binding runner/instrument/configuration (including another exact
registered runner) and source-runner token/instrument/configuration/phase/
outcome disagreements. An eleventh RED requires the exact runner class to match
its corresponding exact runner-state class, rejecting a structurally valid
cross-class state transplant. All fail atomically with
`run_provenance_mismatch` while the valid conditional path remains accepted.
Stage 6.3 code remains unchanged.
The pushed Task 13 review head `bed4704` passed the complete native GitHub
Linux x86 matrix on Python 3.11 and Python 3.12 (run 35031508425).

Task 14 adds the first sparse evaluator full-resource record at the exact
successful tracking-start boundary. All three pre-start phases raise the public
state error before instrument, calibration, or resource inputs are read. A
started builder reauthenticates the exact target runner/instrument/token/tracker
graph and the issuing runner's retained verified source, then joins the full
and estimator-safe calibration observations, midpoint/timestamp recurrence,
source sampling rules, and full/safe resource boundaries. Calibration atoms
are replayed left-associatively from zero; fast, sparse, interleaved tracking,
and unaccepted ledgers begin at exact zero. Included same-run calibration is
charged exactly once, while conditional precalibration is reported but not
charged, and both treatments must equal their physical start boundary. The
builder never returns `None` in this task. Accepted atoms and terminal joins
remain Tasks 15–16. The resource builder remains a direct-module interface and
is intentionally absent from package `__all__` until Task 19 publishes the
complete Stage 6.4 surface. Scientific self-review found no truth path or
future-data access: only immutable acquired calibration observations and
resource metadata are replayed. Software self-review confirmed exact-type and
identity joins, phase-first rejection, nonmutating construction, deterministic
binary64 arrival order, and no Stage 6.3 production changes.
Task 14's independent review found three provenance/start-boundary gaps. The
resource builder now authenticates source phases with runner-specific closed
post-success sets: sparse sources alone admit `geometry_stopped`, while both
exact source-runner classes retain legitimate real progression from calibration
success into tracking. Unknown phases are rejected. A conditional target's own
token binding must retain empty success/source authority, leaving the external
issuer as the sole calibration owner. Finally, the initial runner and estimate
fast, sparse, and total CPU ledgers must each be exactly `0.0` as well as equal;
coordinated nonzero mutations no longer pass. These fixes do not add accepted
tracking atoms or terminal behavior and do not modify Stage 6.3 production.
The pushed Task 14 review head `3f3ac01` passed the complete native GitHub
Linux x86 matrix on Python 3.11 and Python 3.12 (run 35033455166).

Task 15 integrates accepted fast and sparse acquisitions through one causal
query-to-instrument-to-safe-observation-to-tracker transaction. Ordinary
pre-return instrument exceptions remain retryable, retain the exact pending
query, and charge no acquisition resource. Accepted atoms preserve full/safe
identity, expected and actual midpoint joins, exact physical resource
boundaries, the global trace, and the tracker update echo; runner CPU totals
are the exact estimator fast, sparse, and global arrival-order totals. Completed
fast pairs append the existing pair timing, while completed sparse scans append
one five-midpoint ordered-mean timing whose release is the fifth acquisition.
No evaluator truth snapshot or spectral value is requested: a query-scoped
dynamics spy proves every post-start signal evaluation remains inside the exact
instrument query. The evaluator resource builder independently authenticates
and replays accepted fast, sparse, interleaved, and charged full-observation
ledgers, including expected/realized photons, both calibration treatments,
partial blocks of every legal length, and scientifically failed completed
scans. Returned-observation aborts and clean terminal transitions remain solely
Task 16 behavior.

Task 16 completes the sparse evaluator terminal state machine. Pair and
five-point budget exhaustion, plus due-scan geometry failure, stop before any
instrument query; geometry outcomes retain the tracker's exact diagnostic.
External stop performs no acquisition and preserves pending queries and partial
fast/sparse blocks. `run_until_event` advances only through accepted outcomes
and returns the first retryable or terminal event. Every returned-but-unaccepted
observation is terminal: authenticated validation, construction, and unexpected
update exceptions retain one full unaccepted atom with canonical exception
strings, while an unavailable physical resource join retains no exception
strings and returns `resources=None`. Tracker state and CPU totals roll back
exactly; process-control `BaseException` values are re-raised identically after
cleanup. Full-resource construction distinguishes the accepted charged prefix
from the final charged ledger containing the authenticated unaccepted atom and
authenticates the prior-endpoint midpoint recurrence. The eight-phase operation
matrix rejects illegal calls before tracker or instrument side effects. No
production truth lookup was added.

Task 16's independent review exposed four audit gaps that the original green
suite did not exercise. Returned sequence/frequency echo corruption now keeps
its timing-derived midpoint and terminates as a charged validation abort;
unaccepted physical authentication no longer incorrectly requires a rejected
tracker echo. Ordinary failures while constructing pair/scan timing, runner
state, or the accepted outcome after a returned observation now roll back the
tracker update and become a typed unexpected abort, while `BaseException`
identity semantics remain unchanged. Retryable query-failure evidence is now
resource-auditable at zero and nonempty prefixes and survives an external stop,
including partial fast-pair and sparse-scan reservations. The authenticated
abort resource branch now reauthenticates phase, reason/exception class,
unaccepted cardinality, estimate identities, and the pending-query join before
publishing the accepted/final charged split. These corrections preserve all
three estimator CPU totals and add no truth access.

The final Task 16 re-review found that exception class names are not type
identities: a foreign exception may legitimately share either reserved sparse
error name, and public subclasses must retain the Stage 6.3-compatible
`isinstance` classification. Terminal aborts now receive a private exact-
identity causal binding to their runner, reason, canonical exception strings,
acquisition, and before/after estimates. Resource authentication validates that
binding rather than inferring causality from a lossy public name. Foreign
same-name exceptions therefore retain unexpected-error evidence, public sparse
error subclasses retain validation/construction classification, and a later
terminal resource or outcome construction failure re-raises without replacing
the already committed tracker exception evidence.

The causal binding is owned by one private runner slot rather than a process-
global registry. Its lifetime is therefore bounded by the runner, with no
integer-ID reuse or shared mutable registry across independent runs. GC
regressions create and discard multiple authenticated and unavailable aborted
runs after revoking their separate calibration authority and prove their
instrument/dynamics graphs are collectible. A live terminal runner continues
to authenticate repeated resource builds, including after injected terminal
resource or outcome construction failures.

Task 17 adds a deterministic, test-only linewidth-drift composition for the
closed scientific regressions. Its frozen/slotted configuration accepts either
one finite scalar slew or an exact per-resonance-ID mapping, snapshots reference
width and slew mappings into immutable canonical floats, and composes over an
arbitrary `SpectralDynamics` provider at validated non-negative virtual time.
It preserves the base snapshot's resonance tuple order, physical IDs, centers,
amplitudes, eta values, and baseline while replacing only FWHM with the explicit
`reference + slew * time` value. Missing/extra IDs and non-real, nonfinite,
nonpositive, or generated-unphysical widths fail explicitly. This remains under
`tests/evaluation`: no production dynamics API, stochastic state, callback,
truth access, result claim, or Stage 6.5 comparison was added.

Task 18 closes the generated scientific acceptance matrix with named,
download-free public-input recipes. Exact static data recover all four local
fit parameters; seeded Poisson data use fixed error bounds; composed center and
linewidth drift retains exact release timing; and center/width sources retain
independent epochs. Included calibration, tracking, photon, and CPU joins stay
auditable. Scientific fit failure consumes the complete five-point block while
retaining its prior width. Prospective invalid geometry is rejected before
reset commits, and due invalid geometry stops cleanly before sparse reservation.
Separate regressions preserve finite signed/zero asynchronous Q and state the
affine-baseline and within-scan-dynamics mismatch limitations without claiming
unbiased recovery. The test-only truth helper requires the exact completed
release index, makes no call on rejection, and calls test-held dynamics exactly
once at the recorded truth timestamp after release. A query-scoped spy proves
all production signal evaluations after instrument construction occur inside
instrument queries; no estimator or sparse evaluator production module invokes
hidden dynamics.
The Task 18 independent review found that runner/tracker CPU equality alone was
tautological and that the two mismatch cases admitted negligible one-ULP
differences or invariant-only failure records. Test support now returns every
accepted outcome and independently left-folds all 21 update CPU atoms into
fast, sparse, and global totals in exact arrival order. Affine-baseline and
nonlinear-width cases now require success, positive normalized residual, and
respective linewidth biases above 25 kHz and 3 kHz. Setting either dynamics
strength to zero makes its named test fail before restoring the declared
fixture. No production code changed.

Task 19 publishes the complete Stage 6.4 package surface without exporting the
private fitter, source-model helper, provenance authority, registration, or
token machinery. The sparse tracker configuration/records/errors/tracker and
the evaluator runner/states/outcomes/resources/builder are now importable from
their package namespaces. A download-free conditional-precalibration example
uses only public APIs, completes eight fast pairs plus one sparse scan, and
labels the terminal phase, asynchronous live and scan-local Q, independent
center/linewidth epochs, public/truth timing references, resource ledgers, and
pair/scan counts. Estimator documentation now fixes the offset/order policy,
four free versus frozen parameters, ordered gate and diagnostic-presence
contract, timing and resource semantics, partial/retry/stop/abort behavior,
model-mismatch limitations, and explicit Stage 6.5/6.6 nonclaims. A fresh
isolated wheel imported the tracker, runner, and builder and ran the example
from outside the checkout; no network dataset or generated benchmark result is
required.
Task 19's independent review found one export-regression gap and one
documentation omission, without finding an incorrect production export. The
package test now pins the complete estimator `__all__`, preserves every
pre-Stage-6.4 name in order, guards the real private
`fit_sparse_linewidth` symbol alongside authority/token/registration/model
helpers, and proves an appended fitter mutation fails. The estimator guide now
gives the complete compact diagnostic-presence table, including
`nonfinite_solution` with solver fields present and fit/RMSE/rank/condition
groups absent.

Task 20 software review node SW-20-I1 exposed a bind-transaction gap shared by
the Stage 6.3 two-point and Stage 6.4 sparse runners: a ready-state construction
fault after token minting but before registration left the exact provisional
capability in the process-global mint registry. Both bind transactions now
begin immediately after minting and unconditionally revoke the token and any
partial authority graph for every later `BaseException`. Symmetric named
regressions inject both an ordinary exception and a process-control exception,
prove identical-object propagation, and require all provenance registries to
remain unchanged with the captured token neither minted nor bound. This node
does not address the separately reviewed successful-run lifecycle, scientific,
documentation, or wheel-isolation findings.

Task 20 software review node SW-20-I2 closes the successful-registration
lifetime leak without adding a public close protocol. Each live runner now owns
its exact binding and issuer, while global token indexes use weak values and
the runner index uses weak keys with weak issuer references. Exact object-key
lookups retain token identity and avoid integer-ID reuse races; discarded
runners therefore release their instrument, dynamics, calibration, tracker,
and authority graph automatically. Conditional target runners retain only the
source binding required for their live authenticated dependency, so repeated
resource construction and tracking remain valid after caller source references
are dropped; the source graph is collected with the target. Dead tokens fail
closed, live lookup identity remains stable, cross-runner authentication and
transaction rollback remain unchanged, and the former abort test no longer
calls private cleanup. This node does not address scientific, documentation,
wheel-isolation, or Task 20 closeout findings. I2 re-review additionally found
that failure during the first binding weak-value insertion or issuer weak-key
insertion could leave the new authority object on the captured failed runner.
Both bind transactions now pass their exact fresh runner into rollback, so
cleanup clears both private authority slots even when no weak index was ever
published; ordinary and identical process-control faults cover both insertion
seams for both runner types.

Task 20 scientific review nodes SCI-20-001 and SCI-20-002 add no production
behavior. A cycle-safe recursive regression now audits a completed sparse
estimator graph through dataclass fields, instance dictionaries, inherited
slots, mapping keys and values, and built-in containers. Its forbidden families
cover full observations/resources, hidden spectral snapshots/dynamics,
truth/future objects, evaluator/instrument/noise objects, callbacks, and the
evaluator-only expected-photon capability; independent mutation witnesses cover
every type family and storage form and prove cycle safety. Public estimator
guidance now states that, for centered acquisition indices, each parity order
independently has exact binary64 `sum(t*x) == 0`. Reversal balances only
acquisition-order systematics across repeated scans and neither identifies nor
corrects within-scan parameter dynamics nor establishes robustness to them.
Scientific re-review node SCI-20-001-R1 closed a composed-storage blind spot:
mapping and built-in-container traversal is now additive with dataclass,
instance-dictionary, and inherited-slot traversal rather than returning early.
Dict/list subclasses with an attached future capability are detected, while
clean self-cyclic mapping/container subclasses terminate without a false hit.
Final scientific review node SCI-20-001-R2 now reports each forbidden path once
through deterministic first-discovery deduplication; identical field-name/type
findings collapse without masking distinct paths or object identities. Bounded
re-review node SCI-20-001-R3 removes the broad `evaluation` module-name token:
clean cyclic carriers now scan identically under pytest's top-level loader and
ordinary `tests.evaluation...` imports, while explicit evaluator runners,
structural sentinels, and attached futures remain forbidden.

Task 20 software review node SW-20-M1 replaces the editable-checkout sparse
example smoke with an installed-artifact boundary. The named package regression
builds one wheel in a validated temporary directory, requires all seven sparse
estimator/evaluator modules and rejects test/cache payloads, installs the exact
wheel with resolved dependencies into a fresh virtual environment, and probes
the exact public/private export contract under isolated mode. A copied example
runs from an unrelated directory with the checkout absent from `sys.path`, and
the package must resolve beneath that environment's `site-packages`. This node
changes no production or documentation API.

Task 9 accepts only sparse points one through four. Each accepted query becomes
the exact tail of a new immutable `SparsePartialScan`, clears the pending slot,
advances the global sequence/endpoint and every identity age, and applies one
arrival-order atom independently to sparse, interleaved, and charged resource
ledgers. Fast resources, pair/scan histories, completed counters, scan parity,
active FWHM, live Q, and all frozen source facts remain unchanged. The next
selection exposes the identical already-reserved query object, including point
five after the fourth partial transition; point five remains unaccepted and no
fit runs before Task 10. Sparse validation retains exact type/pending/mode/echo/
sequence/frequency/integration/endpoint/exposure/value precedence. Ordinary
partial/resource/aggregate/update construction failures translate to the exact
Stage 6.4 typed code with commit-last value rollback, while process-control
exceptions propagate as the identical object.
Task 9's independent review judged production behavior conformant and found
two Important test-evidence gaps plus one Minor oracle gap. The closed matrix
now covers the complete sparse validation chain and every ordinary/process-
control construction failure at pending points one through four, retaining the
complete prefix state on rollback. Every partial length explicitly matches all
twelve frozen center/FWHM source fields to query one, and all five reserved
queries retain that snapshot. Resource expectations now come from an
independent field-by-field arrival recurrence with distinct point atoms and an
included-calibration charged prefix rather than the production helper.
The pushed Task 9 head `4945fff` passed the complete native GitHub Linux x86
matrix on Python 3.11 and Python 3.12 (run 35019024941).

Task 7 adds the exact public `SparseLinewidthCompositeTracker` shell. Reset
enforces declaration-order typed precedence, validates the tracker-held sparse
configuration against its independent constructor snapshot, revalidates the
complete calibration graph, maps calibration epochs, calls the Task 4
prospective geometry validator for all eight identities, checks the charged
starting budget, and constructs every calibration-seeded center/FWHM/live-Q
source before one commit. Initial selection freezes both adjacent fast queries,
replays two atomic charges without multiplication, subtraction, subtotals, or
`sum`, and exposes only query one; the five-atom sequential helper is reserved
for later due scans. Exact ceilings pass, one-ULP-lower ceilings produce an
atomic `budget_exhausted` boundary, repeated selection returns the identical
pending object, and reservation charges no public resource ledger. The
composite never wraps a running Stage 6.3 tracker. The observation-update method
is intentionally a Task 8 placeholder, and no due-scan branch is implemented.
Task 7's independent review found two Important calibration-source boundary
defects. Conditional-free pre-calibration now remains uncharged while retaining
its mapped availability release timestamp instead of inventing a zero release
epoch. Reset also validates the complete nested source graph—including trace,
resources, fit, identity, epoch, availability, and clock facts—without minting
or consuming verified-source authority. Focused corruption witnesses map these
failures to `calibration_mismatch` with value-atomic rollback. The cumulative
re-review approved Task 7 with no Critical or Important findings.

Task 8 planning exposed an impossible private-helper signature: the stateful
sparse-query constructor was required to freeze a configurable integration time
but neither its pure geometry nor its listed arguments carried that value. The
implementation correction adds an explicit private `integration_time_s`
keyword supplied by the tracker-owned immutable sparse configuration. This
preserves non-default policies and keeps scheduler facts out of pure fit
geometry at the cost of extending one private helper signature; the public API
is unchanged.

Task 8 completes the composite tracker's exact fast observation branch and
due-scan selection. Fast pairs preserve Stage 6.3 query order, per-ID parity,
discriminator/gate diagnostics, successful-source refresh, scientific lost-pair
semantics, and arrival-order safe resources while the composite independently
maintains global sequence/time and process-CPU folds. Every validation and
construction failure retains its declared precedence and value-atomic rollback,
including identical process-control exceptions. After exactly eight completed
fast pairs, selection validates current pure geometry before replaying five
affordability atoms, freezes all five queries with the current source echoes,
global acquisition/sequence indices, exact endpoint recurrence, configured
integration time, and nominal exposure, then exposes only point one. Invalid
geometry produces the complete `sparse_geometry_unavailable` diagnostic;
unaffordable due work stops without falling back to a fast pair. Focused RED
witnesses failed on the Task 7 update placeholder, then the GREEN tracker and
complete Stage 6.3 tracker/atomicity gate passed 127 tests. The mandatory gate
passed 1,397 estimator/evaluation/emulator tests and 1,498 repository tests;
Ruff passed and `git diff --check` was silent.
Task 8's independent review found four Important implementation/evidence gaps.
The sparse validation boundary now requires the exact pending query object at
the expected reserved point before sequence validation and rejects stale fast
reservation or partial state without consuming sparse work. A complete
Stage 6.3 projection is compared after every accepted flank across all eight
identities, repeated parity rounds, sources, epochs, ages, partial/query state,
resources, metadata, and success/common-mode/capture/domain/step-limited/
numerical-loss outcomes. Direct stateful-constructor witnesses pin even and odd
per-ID order, non-default integration policy, and every frozen center/FWHM echo
across all five queries. Separate legal due-state scheduler witnesses prove
scan 1 targets r1 and scan 8 returns to r0 with r0's odd second-scan order. The
validation matrix now
covers every precedence row, while identical process-control exceptions are
injected at every public construction stage on both fast sides and retain
value-equal rollback. Final scheduler re-review closure passed 155 Task 8 plus
complete Stage 6.3 tracker/atomicity tests. The fresh combined mandatory gate
passed 1,427 estimator/evaluation/emulator tests and 1,528 repository tests;
Ruff and `git diff --check` passed.

The Task 7 push exposed a native Linux x86/Python 3.11 Stage 6.3 regression:
the instrument's canonical NumPy pseudo-Voigt path and the tracker's private
scalar-libm source model differed in their last bits, producing a false
approximately `5.8e-9 Hz` correction in a static noiseless trace. The tracker
source model now promotes scalar queries through the same canonical
pseudo-Voigt primitive as vector/acquisition evaluation while retaining
baseline-first, optional-offset, literal-source-order, and target-only
semantics. Exact test oracles use the canonical one-element NumPy spectrum;
independent analytic depth/derivative tests remain closed-form. Independent
review is clean. macOS and exact-version Linux container gates pass; the pushed
head `7a6c849` passed the complete native GitHub Linux x86 matrix on Python 3.11
and Python 3.12 (run 35015603657).

Task 6 completes the sparse fit's eight first-applicable scientific gates and
exact diagnostic-presence rows. Only returned nonpositive solver status or a
returned evaluation count at the configured, including non-default, limit is
`optimizer_failed`; malformed or non-finite returned solutions are distinct
from raised programming exceptions. Bound margin, rank cutoff, condition,
resolved amplitude, and normalized-RMSE equality all pass at their closed
limits, with neighboring failing ULPs pinned. One scaled-Jacobian SVD supplies
rank and condition, RMSE retains five-element arrival-order binary64
arithmetic, and successful scan Q preserves finite signed and zero values while
nonrepresentable Q is `nonfinite_solution`. Every completed scientific result
has exact nonnegative process CPU measured from before preparation through its
last applicable gate/Q derivation and sampled before public-record
construction. Raised ordinary exceptions and process-control `BaseException`
values escape unchanged for later transactional translation; no programming
exception is relabeled as a scientific failure.
Task 6's first review found two Important first-applicable-precedence gaps.
Returned scalar/non-finite final predictions and non-finite RMSE now resolve as
`nonfinite_solution` before any SVD call, while all returned solution,
residual, Jacobian, and prediction shapes are checked without iterating a
zero-dimensional array. Preparation now also requires every scaled bound span
to be finite and positive before model or solver work, including the finite
`[-max_float, +max_float]` baseline-offset endpoint witness whose subtraction
overflows. Both fixes retain absent later diagnostics and the original process-
CPU boundary.

Task 5 performs one isolated four-parameter TRF fit in exact dimensionless
center-correction/FWHM/amplitude/baseline-offset coordinates. The canonical
source-bound model retains baseline shape, target eta, and every non-target
line in immutable source order while fitting only the target's local center,
FWHM, amplitude, and constant offset. The solver receives the public initial
guess and intersected bounds once, residuals retain model-minus-observation
arrival order, and validated success results publish the local center, signed
scan Q, RMSE, and scaled-Jacobian diagnostics without mutating source/query/
observation inputs.

Task 4 constructs pure frozen sparse fit geometry and validates every
calibration-seeded identity prospectively. It fixes the exact even five-point
order and odd reversal, checks finite lower then upper frequency envelopes,
intersects scaled optimizer bounds with the source fit, requires a strictly
interior initial guess, and applies empty-bounds, calibration-cell, then
source-domain failures in that order. The geometry retains no acquisition,
sequence, endpoint, exposure, resource, or mutable-clock fact; query scheduling
remains deferred. Task 3 extracts a private, source-bound spectral-model evaluator for the
sparse estimator while preserving the Stage 6.3 scalar calibration path exactly.
It evaluates the frozen baseline once, adds only the caller-provided constant
offset, and subtracts the immutable source tuple in order; only the target
resonance's center, FWHM, and amplitude may vary. The legacy target-center
helper delegates to the same scalar core without a public surface or arithmetic
change. Lorentzian/pseudo-Voigt, affine/quadratic, scalar/vector, boundary, and
source-order regressions pin that differential.
The bound-only entry point now rejects non-finite and non-scalar supplied target
parameters before source access or arithmetic, preserves canonical built-in
float values, requires positive FWHM and non-negative amplitude, and leaves the
legacy Stage 6.3 delegation behavior unchanged.

Stage 6.2 synchronized and CI-green — The causal warm-started sweep estimator,
generated drift regression, documentation, package smoke, and integrated
re-review are on `origin/main`; synchronized CI passed all 797 tests. The Stage
6.3 calibrated two-point tracker design passed its final adversarial re-review
with zero Critical, Important, or Minor findings. Its separate twenty-task,
86-test implementation plan also passed final adversarial re-review with zero
Critical, Important, or Minor findings. Stage 6.3 implementation is complete
and locally release-gated; Stage 6.5 comparative claims remain out of scope.
Task 1 passed its per-task spec and quality re-review and supplies the
frozen/slotted public resource, budget,
identity, fluorescence-provenance, clock, configuration, metadata, and typed
error primitives. Task 2 adds frozen/slotted calibration-source,
per-identity-calibration, and aggregate-calibration contracts. Caller-asserted
sources defensively snapshot declared value data while direct verified
provenance construction is rejected; aggregate calibrations preserve the exact
source object and snapshot tracker configuration and identity records. Task 2
passed per-task spec and quality re-review after adding explicit nested-fit
reconstruction for diagnostics, optional uncertainty, initial guesses, fitted
parameters, and Q values. Task 3 adds the exact frozen/slotted query, partial-
pair, pair-result, identity-estimate, aggregate-estimate, and update records.
Their constructors enforce only locally represented scalar, echo, diagnostic,
history, age, pending-state, and safe-resource invariants; authentication of
reset state, acquisition context, and evaluator resources remains with later
owners. Task 3's per-task reviews found four Important intrinsic-state gaps and
one scalar-subclass capability leak; all received focused RED/GREEN fixes, and
the final spec and quality re-review passed with no findings. Task 4 adds exact
arrival-ordered estimator-safe resource replay and a caller-asserted source
factory with defensive public-trace binding, complete-sweep fit-input facts,
fixed construction-code precedence, and the exact public-midpoint epoch rule.
Task 4 also corrects the Task 2 source constructor to validate stored safe
resources through that same canonical replay, eliminating integer-seeded sums
and endpoint-subtraction regrouping so long sources remain replace-stable. Its
first per-task review found one Important nested-record error-boundary gap; the
factory now reconstructs each exact public record at its assigned precedence
stage so malformed nested values cannot leak bare construction exceptions.
The final review also exposed a code-6 fitted-ID mismatch that could lose to a
later code-7 malformed-fit failure. A focused combined witness now pins the
first-applicable `fit_input_mismatch` result, and the final re-review approved
Task 4 with zero Critical or Important findings. Task 5 adds the canonical
baseline-once, source-order target-only local model, its analytic target-center
derivative, and the public calibration factory. The factory constructs fixed
ordered-difference identity cells, capture-plus-probe insets, target-pair depth,
and positive analytic discriminator slopes cross-checked numerically; it also
enforces calibration-geometry-before-budget precedence and rejects same-run
treatment for caller-asserted sources. Repository-local test directories now
have explicit package markers so direct pytest collection resolves shared test
helpers in a clean Linux checkout; the pushed fix passed the complete GitHub CI
matrix on Python 3.11 and 3.12. Task 5's first review found two Important
regression-coverage gaps:
exact argument-type rejection and factory-owned analytic/numerical slope
agreement. Focused adversarial tests now prove all three constructible
subclasses fail before geometry or budget handling, and a sign-preserving
derivative perturbation fails calibration. The final independent re-review
approved Task 5 with zero Critical, Important, or Minor findings. Task 6 adds
the final slotted tracker shell, atomic reset-time configuration/clock/resource
joins, calibration-seeded identity estimates, exact budget-treatment charging,
sequential two-atom pair affordability, an idempotent first-minus query, and an
atomic boundary-only budget stop. Observation acceptance, second-query
scheduling, completed pairs, identity advancement, and evaluator APIs remain
deferred to their owning tasks. Task 6's first combined review found one
Important reset-boundary representability gap: individually valid metadata and
configuration values could overflow only when the first pair was selected.
Focused adversarial coverage now makes reset prospectively construct the exact
first query and both sequential charged-resource transitions before its atomic
state commit, so accepted reset state cannot fail later selection for nominal,
elapsed, endpoint, or charged-total overflow. The fix is locally verified and
the final independent re-review approved Task 6 with zero Critical, Important,
or Minor findings. Task 7 adds the exact typed observation-acceptance boundary,
including fixed validation-code precedence and value-atomic rejection. A valid
first flank now clears its pending query, commits exactly one arrival-ordered
tracking/charged resource atom and endpoint, retains unchanged centers and pair
counters, and exposes the exact immutable partial pair in its returned update.
All prospective first-side records are constructed before one state replacement;
partial-pair, resource, or aggregate-estimate construction failures chain to
`partial_pair_construction_failed` and preserve every public tracker field. The
already-reserved second query is then issued on the plus side from the frozen
pair center without another affordability check. Completed-pair calculation,
identity advancement, odd-pair alternation, and evaluator APIs remain deferred.
Task 7's first combined review found two Important reservation/precedence
coverage gaps and two Minor test-coverage issues. Reset now evaluates both exact
endpoint recurrences prospectively and rejects a non-finite or non-advancing
first or reserved-second endpoint before replacing any prior run. Exact legal
rounding and overflow witnesses pin that rollback while the ordinary reserved
query still performs no post-acceptance budget check. Combined first/second
validation defects now prove exact-type precedence over no-pending and
no-pending precedence over sequence mismatch. A call-counted fault on the
second public-resource construction proves typed chaining and rollback after
the first resource construction succeeds, and the update-surface test now
checks the typed exception message contract instead of a vacuous class-truth
assertion. The final independent re-review approved Task 7 with zero Critical,
Important, or Minor findings after these locally verified corrections. Task 8
completes the estimator-side pair transition. Adjacent public flank values now
produce the signed hertz discriminator update through the exact six-stage
normalization, numerical, common-mode, capture, fixed-domain, and clipping
policy. Scientific failures commit charged `lost` pairs without refreshing the
active source; successful pairs refresh the center's public reference and
release epochs even for an exact zero step. Every pair advances the r0…r7
round-robin schedule, including failures, and per-identity arrival order flips
from minus/plus to plus/minus on odd pairs. Pair, identity, resource, and
aggregate construction have distinct chained failure codes and one commit-last
state replacement. Inclusive allowed endpoints, one-ULP outward rejection,
active-age equations, inert tracker seeds, and absence of full-resource/truth
paths are covered by focused tests. Evaluator APIs remain deferred.
Task 8's first combined review found four Important and two Minor gaps. The
amendment gives observed-sum validation precedence over current-center model
arithmetic and commits every representability failure reachable from a valid
public calibration as `lost/numerical_failure`. Its initial calibration-center
fallback for unavailable mandatory geometry was subsequently rejected as
scientifically mislabeled. The exact quadratic allowed-endpoint witness now
advances after both positive and zero observed sums. Combined signed gate
defects, every numerical diagnostic prefix, exact
second-arrival resource fields in both budget modes, recursive truth/capability
storage sentinels, the closed tracker surface, seed parity/modulo diversity, and
full retained pair-source epochs now have focused regression coverage.
Task 8's final re-review found three remaining Important contract/evidence
defects. Raised model or derived `ArithmeticError` values now retain the brief's
typed construction-error rollback, while explicitly computed non-finite values
remain committed scientific losses. The approved design and scientific spec
now correct the original mandatory pair-geometry deficiency:
`zero_discriminator` and `discriminator_slope_per_hz` are jointly optional,
absent only when current-pair geometry was not evaluated or unavailable, and
never substituted from the calibration center; available geometry is computed
at the frozen pair center with a finite zero and strictly positive finite slope.
The recursive truth audit also continues past declared dataclass fields through
dynamic dictionaries and inherited slots with de-duplication and a dedicated
dynamic-extra sentinel. The final independent re-review approved Task 8 and
the documented contract correction with zero Critical, Important, or Minor
findings. Task 9 begins the evaluator-owned package without adding runner,
outcome, or resource-record behavior. `ODMRInstrument` now exposes its canonical
nominal photon rate and per-frequency overhead through exact read-only
properties. The evaluator surface contains only the task-owned preflight/start/
state errors, closed resource-mismatch alias, frozen/slotted instrument
configuration and calibration-query request, and an opaque in-process run token.
Ordinary token construction and all subclassing are blocked, as are value
equality, copy, deepcopy, pickle, and JSON serialization. Pure Python still
permits `object.__new__` to allocate an exact base-class object, but it has no
runner-minted identity and Task 13's registry rejects it. All consumers
authenticate exact registry identity and never accept exact class or
`isinstance` membership alone. Evaluator scalars accept supported
NumPy values only by canonicalizing them to built-in `int`/`float`, reject
boolean, complex, array, invalid-domain, and non-finite inputs, and enforce only
their local timing/exposure invariants. Binding, verified acquisition, outcome,
runner state, and full resource accounting remain assigned to later tasks.
The final independent re-review approved Task 9 and its corrected in-process
capability boundary with zero Critical, Important, or Minor findings.
Task 10 adds evaluator-private full-resource zero, single-arrival advance,
arrival-ordered replay, estimator-safe projection, and exact mismatch helpers.
Full replay retains expected photons while projection intentionally omits them;
missing realized counts add an exact integer zero and increment the separate
missing-count field. Every floating total follows the instrument ledger's
left-associated `old + atom` arithmetic, including elapsed time as
`old + (overhead + integration)`, and mismatch reporting compares every full
snapshot field exactly in declaration order without tolerance. The helpers are
not package exports, and estimator modules neither import `ResourceSnapshot`
nor expose full-resource helpers. Calibration outcomes, acquisition records,
resource aggregates, and runner behavior remain assigned to later tasks.
Task 10's final review approved the production implementation with zero
Critical or Important findings. Its order-sensitive replay witness now detects
reversed arrival processing; one Minor limitation in the test-only package AST
scanner is retained for final branch-review triage.
Task 11 adds the exact seven public verified-calibration outcome, tracking-
acquisition, evaluator-pair-timing, and instrument-query-failure names. The six
records are frozen/slotted and enforce only their intrinsic evidence: closed
success/failure/resource-join discriminators; aligned full, safe, and midpoint
tuples; exact full-to-safe observation and aggregate projections; failure-code
specific request, fit, exception, mismatch, and aggregate presence; an exact
authenticated one-observation resource atom; nonempty ordered mismatch fields
without a fabricated unavailable delta; optional midpoint bounds; ordered pair
truth timing and release; and an unchanged atomic query-failure boundary. The
review correction removed a non-invertible midpoint lower bound and a cross-
convention public-reference interval bound: exact binary64 witnesses show both
reject valid instrument traces. Constructors retain only finite, nonnegative,
endpoint-or-release facts, while later runners own exact producer association
and public pair-result equality. A final missing midpoint is now limited to
acquisition-contract or precedence-winning resource-join failures. The expanded
matrix pins exact schemas, nested capability rejection, canonical built-in
strings, both fit polarities, all field-presence directions, adjacent-ULP truth
means, equal-but-distinct failure snapshots, and exact Task 10 helper
delegation. The validators consume the reviewed Task 10 resource primitives
through method-local imports after evaluator type initialization. They add no
runner, acquisition-loop, fit, registry, or resource-builder behavior. The
final independent review approved Task 11 with zero Critical or Important
findings and retained one test-only AST-sentinel Minor for final branch-review
triage.
Task 12 adds the exact ten evaluator resource, abort, runner-state, and typed
step/run outcome names. The eight new records are frozen/slotted and enforce
only locally represented structure: full-observation tuple and snapshot types,
independent zero-or-one incomplete/unaccepted counts, separate accepted charged
prefix and final charged resources, closed abort reasons and exception/acquisition
matrices, equal abort-time tracker estimates, all seven runner phases, exact
successful-calibration identity where both calibration fields are populated,
trace/timing cardinality, terminal abort and instrument-failure placement, and
outcome kind/state compatibility. Resource replay, instrument/run-token
authentication, registry membership, transition execution, and resource-builder
joins remain assigned to later tasks. The final independent review approved
Task 12 with zero Critical or Important findings and retained five test-only
mutation-strength notes for final branch-review triage.
Task 13 introduces the public instrument-owning `TwoPointEvaluatorRunner`,
superseding Task 9's deliberate temporary package-surface absence assertion.
`bind` accepts only an exact clean instrument, snapshots its derived immutable
rate/overhead configuration and zero resource/time boundary, mints a keyed
one-use token, and registers the exact runner/token/instrument/configuration
identities. Verified calibration preflight is complete before the first query
and preserves phase, exact-type, value, frequency-grid, fit/identity, clock,
then clean-boundary precedence without calling the instrument query or fitter.
The acquisition loop retains every exact full observation and safe projection,
computes the actual instrument midpoint before each query with the normative
binary64 association, authenticates both local resource atoms and continuity
between authoritative boundaries, and applies resource-unavailable
classification before frequency, sequence, timing, or nominal-exposure
defects. Only safe values enter the exact `CompleteSweep`; fitting uses a
defensive configuration snapshot and `initial_guess=None`. The keyed verified
source seam checks normalized-instrument provenance, the safe trace, successful
fit and ordered identities, exact evaluator-association physical epoch,
availability, and clock mapping. One-use exact source identity plus the token
registry prevents direct allocations, copied sources, copied outcomes, or
class membership alone from acquiring authority. A complete recursive
structural fingerprint binds all 21 source fields and their nested fit,
configuration, identity, trace, provenance, resource, and clock values to the
one-use construction identity. Attempt-scoped source minting, prospective state
construction, a cloned trusted pre-success registry binding, and unconditional
fresh-token rollback prevent commit-then-raise or post-construction mutation
from leaking authority. Narrow transaction guards also clean up on
`BaseException` and immediately re-raise the identical object without typed
conversion. Post-start resource/time boundaries are captured as indivisible
pairs with two bounded attempts, so transient ordinary faults are safely
rendered and returned observations are retained. Persistent getter failure
terminates as resource-join unavailable with the complete mismatch-field set;
because the closed outcome requires a concrete after-snapshot, it explicitly
retains the last authenticated boundary rather than claiming an unobserved one.
Every ordinary causal query, fit, source,
boundary-snapshot, or final-registry failure after acquisition starts returns
its closed typed outcome with the accepted prefix and moves the runner to
terminal `calibration_failed`; a resource join failure alone omits aggregate
replay. Resource mismatches are deduplicated in declared order, and every
unavailable join retains the authoritative boundary index.
After two RED/GREEN review-fix waves, the final bounded independent re-review
reported zero Critical, Important, or Minor findings; the independent full
repository gate passed all 1,109 tests and Ruff.
The synchronized Task 13 commit also passed the remote GitHub CI matrix on
Python 3.11 and 3.12.
Task 13 intentionally adds no tracking start, tracking step, public aggregate
resource builder, or abort execution, which remain assigned to later tasks.
Task 14 adds the exact `start_tracking` transition and zero-trace tracking
state. A clean ready runner can consume only an exact registry-authenticated
success from another runner under conditional-free-precalibration treatment;
the issuing calibration-succeeded runner can start with either conditional or
included treatment. Included mode additionally joins the original
runner/source/outcome/token/instrument identities, shared clock, source
availability, three-way rate/overhead equality, and exact full-resource
continuity. Every successful start resets the exact tracker once, retains it
for later runner steps, snapshots the tracking boundary, stores the returned
zero-observation estimate, and performs no query. The eight start errors retain
their fixed phase/type/authentication/calibration/provenance/metadata/resource/
reset precedence. Preflight failures are nonmutating; an ordinary reset failure
is chained under `tracker_reset_failed`, including commit-then-raise faults,
while process-control interrupts re-raise unchanged. A fresh review found one
Important rollback gap when a failing reset mutated the tracker's otherwise
immutable configuration slot. The correction now restores both exact tracker
configuration and state slot identities for every failure covered by the
reset/estimate/prospective-state transaction.
The final bounded independent re-review reported zero Critical, Important, or
Minor findings; the independent full repository gate passed all 1,121 tests
and Ruff.
The synchronized Task 14 commit also passed the remote GitHub CI matrix on
Python 3.11 and 3.12.
Task 15 adds the accepted-observation and retriable instrument-failure portions
of `step`. Each accepted flank retains its exact full and estimator-safe views,
pre-query physical midpoint, authoritative before/after resource boundaries,
and one canonical resource atom before the tracker is updated exactly once.
Normal trace snapshots append in arrival order; pair timing appends only after
the second flank, with the evaluator truth reference formed from the two actual
instrument midpoints and the distinct public reference retained from the
tracker result. The pair-3 regression pins the required neighboring binary64
witnesses. An ordinary nonmutating instrument exception preserves the issued
pending query, unchanged trace and estimate, stores its typed diagnostic while
remaining in `tracking`, and allows an explicit retry of that exact query;
acceptance clears the diagnostic. Transaction guards restore both exact tracker
slots when query issuance, update, or prospective runner publication commits
then raises, including identity-preserving process-control rethrow. Public
evaluator resource assembly, budget/external stops, update aborts, resource-
unavailable classification, and run-loop behavior remain deferred to Tasks
16-17.
The final bounded independent review reported zero Critical, Important, or
Minor findings; the independent full repository gate passed all 1,127 tests
and Ruff.
The synchronized Task 15 commit also passed the remote GitHub CI matrix on
Python 3.11 and 3.12.
Task 16 adds the public evaluator resource builder and its private
authenticated-unaccepted assembly seam. The builder accepts only the original
registry-bound runner and exact token, instrument, configuration, verified
outcome, source, calibration, tracker, and estimate context. It reconstructs
the accepted full trace from estimator arrival order, checks every stored
resource and timing boundary, and replays full resource atoms without
subtraction or subtotal regrouping. Calibration remains separately reported in
both treatments; accepted charged resources start with the calibration replay
only for included same-run accounting and must exactly project to the safe
estimate before an optional authenticated abort atom is continued. Expected
photons remain evaluator-only. An authenticated unaccepted observation stays
outside the accepted tuple but contributes to final tracking and charged
resources, while an intrinsically valid unavailable-resource abort validates
its raw record and authoritative final boundary and returns `None` without
constructing an aggregate. No Task 17 stop, abort-transition, or run-loop
behavior is introduced. The five required TDD nodes and the authorized updated
public-surface stage gate are green; the local full repository gate passes all
1,133 tests and Ruff. A fresh review found two Important contextual-authentication
gaps and no Critical or Minor findings. The amended builder now requires a
non-`None` exact unaccepted midpoint whenever the retained observation passes
the established integration/endpoint/resource-time predicate, while preserving
`None` only for timing-invalid raw records. It also authenticates any retained
retriable instrument failure's pending query and equal before/after snapshots
against the exact accepted/current physical boundary, including evaluator-only
expected photons. Focused RED/GREEN witnesses pin valid authenticated and
expected-only-unavailable midpoint erasure plus a one-ULP failure-boundary
divergence. The amended local gate passes all 1,136 tests and Ruff.
The final bounded independent re-review reported zero Critical, Important, or
Minor findings.
Task 17 completes the evaluator runner terminal protocol. Exhausted pair
budgets stop at the unaffordable pair boundary without another instrument
query or tracker update; explicit external stops preserve accepted partial
pairs, retained pending queries, and the last retryable instrument failure.
Authenticated observations rejected by tracker validation, update-record
construction, or another ordinary update exception become one typed terminal
abort with an exact unaccepted resource atom and unchanged pending estimate.
If the returned raw observation cannot join the instrument ledger, the runner
instead retains its full/safe projections, ordered mismatch fields, and the
timing-only midpoint witness while returning no fabricated resource aggregate.
`run_until_event` advances only across accepted observations and returns the
first budget stop, typed abort, or instrument failure without implicit retry.
All five public operations now enforce the seven-phase legality matrix before
calling the instrument, tracker, fitter, resource builder, or provenance
registry. Ordinary `Exception` values at the tracker-update boundary are typed
aborts; process-control `BaseException` values remain transactionally restored
and are re-raised unchanged. The first fresh review found one Important timing
witness defect: a resource-valid raw observation with an invalid endpoint was
incorrectly assigned the expected midpoint, so mandatory abort resource
assembly rejected it. The corrected acquisition seam publishes `None` whenever
integration/endpoint/resource time does not match, independently of resource
authentication, while retaining mandatory full resource accounting and the
authoritative live endpoint. An exact one-ULP endpoint regression pins the
typed validation abort and terminal no-later-call behavior. The 60 Task 17
protocol rows and complete 78-test runner file pass; the full repository gate
passes all 1,196 tests.
The final bounded independent re-review reported zero Critical, Important, or
Minor findings.
Task 18's closed 4,481-point calibration fixture exposed one invalid timing
association in the Task 13 acquisition seam. The instrument clock advances as
`(previous endpoint + overhead) + integration`, while the required resource
ledger independently accumulates `previous elapsed + (overhead + integration)`;
these exact binary64 recurrences first differ by one ULP at the fourth query.
Calibration midpoint authentication now uses the observation endpoint and live
instrument clock only. Resource-ledger integrity remains separately protected
by exact atomic replay and boundary comparison, so removing the cross-
association equality neither introduces a tolerance nor weakens detection of
endpoint or ledger corruption. A focused four-point regression pins both hex
witnesses and the exact instrument midpoint sequence.
The bounded independent review of this owning timing fix reported zero
Critical, Important, or Minor findings.
Task 18 closes the generated scientific acceptance boundary for the calibrated
two-point center tracker. A checked 4,481-point pseudo-Voigt source fit now
anchors exact source endpoint, actual/public midpoint, physical/public epoch,
clock-mapping, and acquisition-resource witnesses. Separate matched fixtures
cover two noiseless static cycles, seeded Poisson reproducibility without an
accuracy claim, thirty cycles of common linear center drift, deterministic r3
contrast loss and recovery, and continuous included-same-run accounting.
Truth scoring owns a separate dynamics instance, occurs exactly once per
completed pair only after release, rejects pre-release and duplicate access,
and pins the pair-3 neighboring-binary64 actual/public reference distinction.
Identity-cycle-safe traversal of the exact tracker object, dataclass fields,
instance dictionaries, inherited slots, and built-in containers confirms no
full observation, snapshot, dynamics, truth oracle, future record, or
evaluator-only expected-photon field/capability is reachable; a forbidden-slot
mutation sentinel proves this test is non-vacuous. The static fixture pins all
32 query frequencies/sides/endpoints plus exact source ages and unchanged
centers. The loss fixture proves pair 11 is the only lost/common-mode-limited
pair, all eight calibration cells (including r3) remain unchanged, and r3
recovers at pair 19.
These are fixed synthetic regression guards, not benchmark results or evidence
that the chosen policy is optimal. Task 18's first independent review found
three Important assertion-strength gaps and no Critical or Minor findings; all
three received isolated test-only corrections. The final bounded independent
re-review reported zero Critical, Important, or Minor findings.
Task 19 publishes the calibrated two-point center-tracking workflow. The
download-free example uses only installed public APIs to acquire a verified
synthetic calibration, start a separate
`conditional_free_precalibration` run, complete one pair per identity, and
print finite public policy, resource, and timing diagnostics without truth,
error, expected-photon, comparison, or superiority output. Researcher guidance
now documents verified versus caller-asserted provenance, both mandatory
budget treatments, fixed identity cells, alternating side order, discriminator
sign, policy-only lock/common-mode semantics, zero-step refresh, partial and
unaccepted resources, token continuity, public versus actual timing, release
and age semantics, terminal aborts, and the inert tracker seed. It states
explicitly that Stage 6.3 provides no Stage 6.5 matched-budget superiority
result. The required public-surface characterization exposed five evaluator
protocol aliases that existed in their defining module but were absent from the
package export; the authorized narrow export and stale exact-`__all__` test
updates make the source and isolated wheel surfaces agree.
The bounded independent Task 19 review reported zero Critical, Important, or
Minor findings.
Task 20's first integrated review wave found three Important cross-task defects
and one documentation Minor. Common-mode and capture gates now retain every
finite discriminator, raw-innovation, requested-step, and candidate-center
diagnostic computed before the gate while applying exactly zero correction and
preserving gate precedence. Tracking midpoint authentication now separates the
instrument endpoint/live clock from the independently associated resource
ledger; an expected-photon-only corruption on the fifth observation retains
the exact physical midpoint even when ledger elapsed and instrument time differ
by one ULP. All foundational Stage 6.3 closed literal fields, including typed
estimator error codes, now store exact built-in strings so subclasses cannot
retain callbacks or payload capabilities, and the recursive isolation witness
traverses non-exact scalar subclasses. Documentation assigns sparse linewidth/Q
estimation to Stage 6.4 and matched-budget comparison to Stage 6.5 without a
performance claim. Focused gates pass 32 tests, the affected estimator/evaluator
suite passes 248 tests, the complete repository passes 1,209 tests, and Ruff
passes.
Task 20's second integrated re-review found two Important transaction and
capability gaps. A post-query live-clock-only divergence is now authenticated
before `tracker.update`: the runner retains the joined full/safe/resource atom
with no physical midpoint, performs no estimator update or normal-trace append,
and enters a typed terminal validation abort whose exact resource totals remain
available. The evaluator resource validator checks the live instrument endpoint
separately from ledger elapsed and accepts that one timing-invalid terminal atom
without weakening exact ledger joins. Foundational nested estimator records now
defensively reconstruct exact identity bindings, fit configurations,
fluorescence provenance, clock mappings, and calibration identities; aggregate
calibration rejects a capable source subclass because source identity is
semantically retained. Focused wave-two coverage passes 3 tests, the affected
estimator/evaluator suite passes 263 tests, and the full repository passes 1,212
tests.
Task 20's third integrated re-review found two Important composition gaps. A
combined resource mismatch and live-clock divergence now terminates through
the resource-first unavailable path: the validator requires state/live clock
identity but requires the observation endpoint only for an authenticated
midpoint, preserving raw evidence, no estimator update, no aggregate, and a
terminal runner. Stage 6.3 fit snapshots now canonicalize nested configuration,
diagnostic, uncertainty, optimizer, and resonance-ID strings to exact built-in
values. The public caller-asserted binding and subsequent calibration/reset
therefore cannot retain callbacks carried by string subclasses. The focused
gate passes 4 tests, affected suites pass 249 tests, and the complete repository
passes 1,213 tests.
The final integrated review range was exactly `9f829ed..1c7fc67` (24 commits).
After three atomic fix waves, independent scientific and software re-reviews
both reported zero Critical and zero Important findings. The final deterministic
gates passed twice: 1,112 estimator/evaluator/emulator tests and 1,213 full
repository tests on each run, with Ruff and diff checks clean. A fresh build
produced exactly one sdist and one wheel; an isolated environment imported the
complete public estimator/evaluator surface under `-I` and ran the public
two-point example from an unrelated working directory. Stage 6.3 therefore
closes the calibrated realtime center-tracking protocol and its fixed synthetic
acceptance boundary, not sparse linewidth/Q estimation or a matched-budget
superiority result.

## Completed work

- Inspected the repository and confirmed it began empty, without prior commits
  or benchmark outputs.
- Defined the scientific observables and unambiguous FWHM/Q conventions.
- Separated causal recorded playback from interactive closed-loop emulation.
- Defined acquisition-resource accounting and matched-budget comparison rules.
- Defined estimator truth-isolation and causal-access requirements.
- Scoped the first end-to-end milestone and its required outputs.
- Created the public GitHub repository and configured it as the `origin` remote:
  `https://github.com/stannum13/nv-odmr-trackbench`.
- Renamed the public repository and distribution to `nv-odmr-trackbench` while
  retaining the `odmr_bench` import package and `odmrbench` CLI.
- Received user approval of the Stage 0 scientific specification.
- Added the installable Python package scaffold and `odmrbench` console entry
  point.
- Added the MIT license, citation metadata, and GitHub Actions CI workflow.
- Added normalized Lorentzian, Gaussian, and FWHM-matched pseudo-Voigt line
  shapes with explicit linewidth conversion helpers and Q calculation.
- Added immutable, validated baseline and resonance parameters using explicit
  Hz-valued fields, canonical Python-float storage, and a reference-centered
  polynomial baseline.
- Added deterministic eight-dip spectrum composition with stable parent IDs
  and caller-supplied, already-realized additive noise.
- Added a YAML-driven script that generates the synthetic eight-resonance
  demonstration plot from reusable package configuration and curve helpers.
- Extended CI to build and install the wheel before testing and smoke-test the
  installed `odmrbench --version` entry point.
- Completed independent task reviews and a clean final senior review for the
  repository-scaffold and spectral-model stages.
- Approved the event-driven virtual-instrument design, including normalized
  fluorescence, photon accounting, truth isolation, and virtual-clock semantics.
- Verified Figshare DOI `10.6084/m9.figshare.28788437.v1` as the first optional
  real-data anchor: 4,693 sweeps by 311 points, CC BY 4.0, checksum-matched.
- Added the task-by-task implementation plan for verified playback, hidden
  dynamics, observation noise/resources, the virtual instrument, and POC CLI.
- Added a checked, explicit-path Figshare registry/loader that preserves the
  raw analog signal and unresolved units, without downloading or redistributing
  external data.
- Added immutable sweep data and causal row-major recorded playback, with
  timestamps unavailable unless a caller explicitly assumes a nominal clock.
- Replaced the deficient generator-only estimator boundary: Python generator
  frames retain the offline dataset and can reveal future samples. Estimator
  playback now uses an evaluator-owned causal callback runner that supplies one
  frozen observation at a time; `iter_playback_for_analysis` is trusted
  evaluator-only tooling.
- Hardened local verified loading by checking and parsing one immutable byte
  snapshot, added full YAML-to-record parity coverage, deterministic verified
  loader success coverage, and non-empty sweep-dimension validation.
- Added frozen hidden `SpectralSnapshot` truth records that require exactly
  eight unique stable physical IDs and positive absolute centers.
- Added runtime-checkable spectral-dynamics protocol plus stationary and
  deterministic common/per-ID linear-center-drift implementations. They accept
  only finite non-negative virtual timestamps and never sort resonance order.
- Added seeded Poisson shot-noise and controlled Gaussian normalized-fluorescence
  models, with photons retained only when the generative model produces counts.
- Added provenance-bearing empirical residual noise with explicit replay,
  independent-sample, and contiguous-block correlation modes.
- Added frozen full and estimator-safe observation records; signal-conditioned
  expected photons remain evaluator-only and are absent from estimator objects.
- Added atomic virtual-acquisition resource accounting for observations,
  integration, nominal/expected/realized photons, uncounted observations, and
  elapsed virtual time.
- Added an event-driven virtual ODMR instrument that evaluates hidden truth at
  the integration midpoint, returns end-of-integration timestamps, advances
  only virtual time, and commits sequence, clock, and resource totals atomically.
- Preserved seeded stochastic reproducibility across failed queries by restoring
  NumPy-generator and stateful-noise state; empirical replay remains usable
  without copying its immutable configuration.
- Added deterministic JSON CLI summaries for checked optional-dataset metadata,
  explicit-local raw playback, and a fixed seeded synthetic-drift scenario.
- Hardened CLI configuration and packaging review findings: the complete query
  schedule is scalar-canonicalized and checked for finite virtual timing before
  instrument construction, expected input failures have concise exit-2 error
  messages, playback streams aggregates, and `bundled:drift` is wheel-packaged
  for arbitrary-working-directory use.
- Added an explicit eight-resonance Poisson drift configuration, a download-free
  in-memory playback/emulation example, and researcher-facing raw-data and
  synthetic-emulation guidance.
- Approved a six-stage estimator sequence and specified Stage 6.1: constrained
  offline Lorentzian/pseudo-Voigt oracle plus an independent repeated-sweep
  baseline.
- Added the task-by-task Stage 6.1 implementation plan covering immutable fit
  contracts, deterministic initialization, constrained fitting/uncertainty,
  and the cold-start repeated full-sweep estimator.
- Corrected the plan after adversarial review by defining the fit-failure state
  machine, full-rank quality gate, scaled-to-public covariance transform, exact
  center bounds, public schemas, initializer formulas, and fixed regressions.
- Corrected the second review findings with fluorescence origin/scale
  parameterization, one-cutoff SVD covariance, an exact baseline-only SSE
  reference, feasible fallback geometry/fixture, a denser scan grid, immutable
  initial-guess provenance, and a typed failure-field matrix. The final review
  closes the remaining numerical gap by evaluating both spectral residuals and
  the baseline-only target in centered fluorescence coordinates.
- Defined pre-initialization `uninformative_sweep` handling for zero-variation
  data and corrected the final smoke-test paths and affine-scaling fixture rules.
- Separated exact identical-input repeatability from physically negligible
  affine-representation roundoff and SciPy termination details.
- Added immutable validated full-sweep, fitting-configuration, initialization,
  uncertainty, fit-result, and per-sweep-estimate contracts for the offline
  oracle. Result records enforce the structured failure state machine, preserve
  attempted-guess provenance, and derive read-only signed Q values from public
  fitted centers and FWHM values without estimator access to hidden truth.
- Added deterministic baseline-aware eight-line initialization using three
  robust rejection updates and a final scaled polynomial trend, prominence-
  ranked dip discovery, actual-frequency separation and width interpolation,
  raw detrended amplitudes, and explicit-only feasible fallback geometry.
  Structured diagnostics preserve candidate scarcity and invalid-window or
  numerical baseline failures without fabricating a detected solution. A
  signal-scaled floating-point discovery floor rejects polynomial/smoothing
  roundoff, while overflow-safe frequency normalization supports extreme finite
  same-sign and opposite-sign endpoints.
- Added deterministic bounded Lorentzian and pseudo-Voigt oracle fitting with
  dimensionless frequency/fluorescence scaling, midpoint-referenced linear or
  quadratic baselines, exact non-crossing center boxes, structured scientific
  failures, raw-unit residual diagnostics, full-rank quality gates, and
  public-unit local-linearized covariance from one shared SVD cutoff. Fixed
  regressions cover clean/noisy recovery, affine fluorescence changes,
  initialization/fallback behavior, and exact baseline-improvement boundaries.
- Hardened the oracle review boundary with nonempty initializer-preflight
  reasons, intentional infinite baseline bounds plus finite feasible resonance
  bounds, explicit public-parameter and rounded center-box checks,
  exponent-aware quadratic scaling, and overflow-safe fluorescence origins. A
  distinct metric-less `quality_failed` state covers nominally successful
  optimizers with any non-finite required output, while an unrepresentable
  public covariance transform preserves rank and can leave an otherwise valid
  fit successful without uncertainty. Covariance tests pin the full public
  layout, strict SVD cutoff, and single-SVD implementation.
- Added `RepeatedFullSweepEstimator`, which passes `initial_guess=None` for
  every completed sweep, retains successful and structured failed attempts in
  immutable evaluator history, advances `latest` on failures, copies only that
  sweep's public completion metadata, and clears all retained state on reset.
- Added a fixed-seed two-sweep generated regression with declared numerical
  tolerances, researcher guidance for the model, initialization, bounds,
  failures, local-linearized uncertainty, ordered-center scope, and recording
  interpretation, plus a download-free synthetic fitting example.
- Addressed the integrated Stage 6.1 findings with a positive finite
  `min_amplitude_significance` configuration (default `5.0`) and an all-line
  model-conditioned local amplitude/standard-error gate derived from the same
  packed covariance as public uncertainty. Fixed noisy seven-line seeds 1, 2,
  and 23 now fail quality before and after direct `+1e6` fluorescence shifts
  instead of promoting a noise-supported eighth component; normal clean/noisy
  eight-component fixtures retain their declared recovery.
- Guarded unrepresentable initializer baseline conversion, strengthened result
  provenance across baseline degree/reference, IDs, diagnostic source, and
  optimizer-attempt state/status/message/evaluation count, restricted the
  uncertainty method to its exact local-Jacobian label, and rejected non-finite
  Q without leaking numerical warnings. The public linearized-error helper now
  rejects complex arrays, bool/non-integral degrees of freedom, and invalid
  scalar inputs without lossy coercion. Its covariance uses square-root SVD
  factors to avoid premature overflow/underflow, and finite scaled baseline
  coefficients that would overflow or underflow in public units fail
  initialization explicitly.
- Added fit-level regressions for zero baseline-only SSE, rank deficiency,
  affine raw diagnostics and public errors, nonuniform grids, covariance
  unavailability, exact-zero amplitude errors, and weak false components near
  the configured significance threshold. A noisy direct-addition regression
  pins classification, IDs, rank, cost, RMSE, every public SE field, and local
  significance behavior under a `+1e6` fluorescence origin shift.
- Specified the Stage 6.2 causal warm-start state machine: successful-prior-only
  seeding, guarded polynomial rebasing, shared sweep/guess preflight, explicit
  warm/cold attempt provenance, cold recovery, stale active estimates with
  distinct age bases, nonoverlapping endpoints, and acquisition-versus-compute
  resource accounting.
- Drafted the complete four-gate Stage 6.2 TDD implementation plan: frozen
  attempt/estimate contracts, shared fitter preparation and exact finite-float
  baseline rebasing, the causal warm-start/recovery/age/CPU state machine, and
  frozen-snapshot drift integration with documentation, example, build, and
  isolated-wheel smoke. After its initial 0-Critical/7-Important/4-Minor review,
  corrected the lost post-preflight rank variable, attempt/source and
  disposition/active invariants, typed compatibility translation/precedence,
  global timer ordering/atomicity, behavioral TDD increments, exact drift
  configuration, wrapper-only constant failure, branch joins, and fail-fast
  wheel/sdist smoke. The revised plan subsequently passed adversarial
  re-review with no findings and preceded feature implementation.
- Added frozen, slotted `SweepFitAttempt` and `WarmSweepEstimate` records with
  closed public literals, immutable attempt tuples, exact warm/cold/preflight
  provenance, disposition and rejection-code matrices, identity-based active
  result selection, explicit stale-age semantics, endpoint/resource domains,
  compound CPU accounting, and derived current-fit, staleness, and evaluation
  totals. Constructors canonicalize supported NumPy scalars without using
  array-valued fit-result equality.
- Extracted the Stage 6.1 sample/variation/origin preflight and initial-guess
  validation into one package-internal preparation path shared by the unchanged
  `fit_spectrum` entry point and future warm-start orchestration. Added exact
  finite-float linear/quadratic baseline rebasing with zero-coefficient and
  exact-cancellation handling, explicit overflow/nonzero-underflow rejection,
  and a guarded successful-prior conversion that deterministically returns one
  of five closed compatibility codes in the specified first-failure order.
- Added `WarmStartedFullSweepEstimator`, which validates causal endpoints before
  preparation, seeds only from the latest selected success, rejects over-age or
  incompatible starts explicitly, retains one eligible same-sweep cold retry,
  exposes separately aged stale active fits, counts acquisition resources once,
  and commits history/endpoints only after globally monotonic process-CPU timing
  and every public record construction succeed.
- Validated the completed-sweep wrapper on one immutable, fixed-seed three-grid
  drift family. Cold and warm estimators receive the identical frozen sweep
  objects, changed-midpoint baseline rebasing remains compatible, ordered IDs
  and fixed-fixture center/FWHM/Q bounds hold, and cumulative observations and
  zero-age source promotion are exact. Separate regressions cover constant-
  sweep preflight staleness, update-age rejection, center-outside-sweep cold
  fallback, and deterministic failed-warm/one-cold recovery without duplicating
  acquisition resources. The download-free example reports source, attempt,
  age, `nfev`, and measured process CPU diagnostics without a speedup claim.
- Synchronized the reviewed Stage 6.2 range to `origin/main`; remote CI passed
  the 797-test suite.
- Drafted the Stage 6.3 calibrated two-point center-tracker design: mandatory
  calibration-budget labeling; one immutable source binding the fit, exact or
  adopted IDs, safe acquisition trace, normalized scale, sweep bounds,
  resources, epochs, availability, and clock mapping; analytic target-only
  discriminator slopes; fixed conservative calibration cells; identity-keyed
  adjacent pairs; reset-bound total ceilings; canonical instrument-ledger
  arithmetic; bounded proportional updates; policy-state/common-mode
  diagnostics; distinct public and actual-instrument pair references; evaluator
  joins; partial/aborted acquisition accounting; truth isolation; and fully
  specified generated regressions. Two adversarial reviews exposed remaining
  causal contract gaps. This corrected revision adds an opaque source-run
  provenance token and exact rate/overhead/ledger continuity, lossless typed
  calibration failures, one evaluator runner with explicit normal and terminal
  transitions, intrinsic-versus-contextual validation ownership, successful
  zero-step source refresh, and signed mapped-reference rules. The latest
  corrections make the pending query part of every ordinary-exception abort
  snapshot, separate malformed raw acquisitions from authenticated resource
  replays, close every public error-code contract, fix caller-asserted epochs to
  the public-midpoint convention, and join the accepted charged prefix before
  an optional abort atom. The final bounded adversarial re-review reported zero
  Critical, Important, or Minor findings. This remains design, not
  implementation.
- Drafted and revised the twenty-task Stage 6.3 implementation plan with
  separately reviewable gates for estimator contracts and atomic resources, caller-
  asserted and verified calibration provenance, analytic calibration and fixed
  cells, pair scheduling and transactional updates, opaque same-run binding,
  typed calibration/runner outcomes, authenticated evaluator joins, closed
  static/Poisson/drift/contrast-loss regressions, public guidance and isolated-
  wheel smoke, and independent scientific/software closeout. Every production
  surface is introduced behind a focused RED, and Stage 6.5 matched-budget
  comparison remains explicitly out of scope. The revision resolves the first
  review's eight Important and three Minor findings by separating every
  dependency, identity owner, and RED/GREEN boundary. A second review's three
  Important findings were then corrected by fixing safe-resource field and
  arithmetic contracts, moving evaluator resource primitives before record
  consumers, and ordering reset/update/example/documentation REDs before their
  owning implementations. The plan is final-re-review-pending; no Stage 6.3
  production implementation has begun. The final bounded plan re-review
  reported zero Critical, Important, or Minor findings.
- Added the six Stage 6.3 estimator-state records with exact public field
  surfaces and exports. Query/partial/pair construction now rejects local
  index, identity, side, observation, arrival, reference, release, and policy-
  diagnostic contradictions. Identity/aggregate/update construction enforces
  signed calibration references, nonnegative pair references/releases/ages,
  active-source/history/counter/pending/partial equations, stopped boundaries,
  safe resource counts, seeds, and accepted-side echoes. Recursive structural
  coverage keeps truth, instrument, full observation/resource, expected-photon,
  callback, evaluator, and future references outside the estimator graph;
  accepted IDs and closed string literals are canonical built-in strings so
  scalar subclasses cannot carry capabilities into that graph.

## Important scientific and design decisions

- Project identity is `nv-odmr-trackbench`; the existing checkout
  directory is retained.
- Internal frequency and linewidth units are Hz. Internal time units are
  seconds. Public APIs must not accept ambiguous unitless physical quantities.
- Each line component uses FWHM directly. Lorentzian HWHM/gamma and Gaussian
  sigma conversions must be explicit and unit-tested.
- Q is defined as resonance center divided by FWHM and is not treated as a
  proxy for magnetometric sensitivity by itself.
- Low-level spectral functions permit finite signed frequency coordinates for
  generality; physical scenarios will require positive absolute resonance
  centers at the instrument-validation boundary without redefining Q.
- The initial benchmark represents eight electronic resonances. Optional
  hyperfine components may later retain parent electronic-resonance identities.
- An offline-oracle success means an eight-component fit passed the declared
  model, candidate-conditioned initializer, and configured quality thresholds.
  It is not calibrated evidence that eight physical resonances are present.
- Scenario truth belongs to the virtual instrument and evaluation harness; an
  estimator receives only observations and permitted public metadata.
- Recorded playback cannot evaluate adaptive frequencies that were not present
  in the recording.
- A callback runner protects the normal estimator API from accidental future
  data access, but it is not a security sandbox against adversarial Python stack
  introspection; use process isolation for that threat model.
- Budget matching and any unavoidable budget mismatch must be explicit in
  machine-readable results and figures.
- Development prioritizes the first falsifiable vertical slice over completing
  every planned abstraction in advance.
- Empirical residual correlation is a declared experimental condition: replay
  preserves supplied order, sample draws independent residuals, and block draws
  seeded contiguous blocks with deterministic wrapping.

## Tests currently passing

- Stage 6.4 Task 20 SW-20-I1 bind-construction transaction regressions: 4
  passed.
- Stage 6.4 Task 20 SW-20-I1 combined Stage 6.3/sparse gate: 811 passed.
- Stage 6.4 Task 20 SW-20-I1 estimator/evaluation/emulator/dynamics affected
  gate: 1,812 passed.
- Stage 6.4 Task 20 SW-20-I1 full repository gate: 1,896 passed.
- Stage 6.4 Task 20 SW-20-I2 lifecycle and stale/live provenance regressions:
  22 passed.
- Stage 6.4 Task 20 SW-20-I2 combined Stage 6.3/sparse gate: 831 passed.
- Stage 6.4 Task 20 SW-20-I2 estimator/evaluation/emulator/dynamics affected
  gate: 1,832 passed.
- Stage 6.4 Task 20 SW-20-I2 full repository gate: 1,916 passed.
- Stage 6.4 Task 20 SCI-20-001/002 truth-isolation and documentation focused
  gate: 63 passed.
- Stage 6.4 Task 20 SCI-20-001/002 integrated scientific gate: 669 passed.
- Stage 6.4 Task 20 SCI-20-001/002 estimator/evaluation/emulator/dynamics
  affected gate: 1,865 passed.
- Stage 6.4 Task 20 SCI-20-001/002 full repository gate: 1,950 passed.
- Stage 6.4 Task 20 SW-20-M1 exact isolated-wheel regression: 1 passed.
- Stage 6.4 Task 20 SW-20-M1 package-focused gate: 10 passed.
- Stage 6.4 Task 20 SW-20-M1 full repository gate: 1,950 passed.
- Stage 6.4 Task 19 review-fix source package/example/guidance gate: 33 passed.
- Stage 6.4 Task 19 review-fix estimator/evaluation/package affected gate:
  1,693 passed.
- Stage 6.4 Task 19 review-fix full repository gate: 1,892 passed.
- Stage 6.4 Task 19 isolated wheel build/install/public-import/example smoke:
  passed.
- Stage 6.4 Task 17 deterministic linewidth-drift fixture: 45 passed.
- Stage 6.4 Task 17 dynamics/sparse-evaluator compatibility gate: 207 passed.
- Stage 6.4 Task 17 model/dynamics/emulator/estimator/evaluation affected gate:
  1,822 passed.
- Stage 6.4 Task 17 full repository gate: 1,868 passed.
- Stage 6.4 Task 17 Ruff and diff-check gates: All checks passed.
- Stage 6.4 Task 18 closed acceptance regressions: 20 passed.
- Stage 6.4 Task 18 regression plus focused sparse gate: 289 passed.
- Stage 6.4 Task 18 dynamics/sparse compatibility gate: 227 passed.
- Stage 6.4 Task 18 model/dynamics/emulator/estimator/evaluation affected gate:
  1,842 passed.
- Stage 6.4 Task 18 full repository gate: 1,888 passed.
- Stage 6.4 Task 18 Ruff and diff-check gates: All checks passed.
- Stage 6.4 Task 16 review-fix sparse evaluator types/resources/runner gate:
  141 passed.
- Stage 6.4 Task 16 review-fix sparse/Stage 6.3 runner-resource compatibility
  gate: 300
  passed.
- Stage 6.4 Task 16 review-fix estimator/evaluation/emulator affected gate:
  1,722 passed.
- Stage 6.4 Task 16 review-fix full repository gate: 1,823 passed.
- Stage 6.4 Task 16 Ruff and diff-check gates: All checks passed.
- Stage 6.4 Task 15 focused sparse runner/resource gate: 69 passed.
- Stage 6.4 Task 15 sparse/Stage 6.3 runner-resource compatibility gate: 229
  passed.
- Stage 6.4 Task 15 estimator/evaluation/emulator affected gate: 1,674 passed.
- Stage 6.4 Task 15 full repository gate: 1,775 passed.
- Stage 6.4 Task 15 Ruff and diff-check gates: All checks passed.
- Stage 6.4 Task 15 review head `806eb64` passed GitHub Actions on Python 3.11
  and Python 3.12 (run 35034957158).
- Stage 6.4 Task 14 review-fix focused resource-builder gate: 13 passed.
- Stage 6.4 Task 14 review-fix sparse/two-point runner-resource compatibility
  gate: 240 passed.
- Stage 6.4 Task 14 review-fix estimator/evaluation/emulator gate: 1,661 passed.
- Stage 6.4 Task 14 review-fix full repository gate: 1,762 passed.
- Stage 6.4 Task 14 review-fix Ruff gate: All checks passed.
- Stage 6.4 Task 14 focused resource-builder gate: 9 passed.
- Stage 6.4 Task 14 sparse evaluator/Stage 6.3 resource compatibility gate: 80
  passed.
- Stage 6.4 Task 14 estimator/evaluation/emulator gate: 1,657 passed.
- Stage 6.4 Task 14 full repository gate: 1,758 passed.
- Stage 6.4 Task 14 Ruff gate: All checks passed.
- Stage 6.4 Task 13 conditional provenance review-fix gate: 12 passed.
- Stage 6.4 Task 13 focused calibration/start contract gate: 39 passed.
- Stage 6.4 Task 13 sparse runner plus complete Stage 6.3 calibration/runner
  compatibility gate: 190 passed.
- Stage 6.4 Task 13 estimator/evaluation/emulator gate: 1,648 passed.
- Stage 6.4 Task 13 full repository gate: 1,749 passed.
- Stage 6.4 Task 13 Ruff gate: All checks passed.
- Stage 6.4 Task 12 focused sparse shell plus complete Stage 6.3 calibration/
  runner compatibility gate: 152 passed.
- Stage 6.4 Task 12 focused sparse value/shell plus complete Stage 6.3
  calibration/runner gate: 176 passed.
- Stage 6.4 Task 12 estimator/evaluation/emulator gate: 1,610 passed.
- Stage 6.4 Task 12 full repository gate: 1,711 passed.
- Stage 6.4 Task 12 Ruff gate: All checks passed.
- Stage 6.4 Task 11 revised-review focused evaluator contract gate: 24 passed.
- Stage 6.4 Task 11 revised-review compatibility contract gate: 185 passed.
- Stage 6.4 Task 11 revised-review estimator/evaluation/emulator gate: 1,599
  passed.
- Stage 6.4 Task 11 revised-review full repository gate: 1,700 passed.
- Stage 6.4 Task 11 revised-review Ruff gate: All checks passed.
- Stage 6.4 Task 11 focused evaluator value-contract gate: 21 passed.
- Stage 6.4 Task 11 sparse/Stage 6.3 evaluator and estimator contract gate: 182
  passed.
- Stage 6.4 Task 11 estimator/evaluation/emulator gate: 1,596 passed.
- Stage 6.4 Task 11 full repository gate: 1,697 passed.
- Stage 6.4 Task 11 Ruff gate: All checks passed.
- Stage 6.4 Task 10 review-fix focused tracker/atomicity files: 230 passed.
- Stage 6.4 Task 10 review-fix complete sparse estimator gate: 454 passed.
- Stage 6.4 Task 10 review-fix estimator/evaluation/emulator gate: 1,575 passed.
- Stage 6.4 Task 10 review-fix full repository gate: 1,676 passed.
- Stage 6.4 Task 10 review-fix Ruff gate: All checks passed.
- Stage 6.4 Task 10 focused types/tracker/atomicity gate: 288 passed.
- Stage 6.4 Task 10 complete sparse estimator gate: 416 passed.
- Stage 6.4 Task 10 estimator/evaluation/emulator gate: 1,537 passed.
- Stage 6.4 Task 10 full repository gate: 1,638 passed.
- Stage 6.4 Task 10 Ruff gate: All checks passed.
- Stage 6.4 Task 9 review-fix focused tracker/atomicity files: 161 passed.
- Stage 6.4 Task 9 review-fix plus complete Stage 6.3 tracker/atomicity gate:
  230 passed.
- Stage 6.4 Task 9 review-fix estimator/evaluator/emulator gate: 1,502 passed.
- Stage 6.4 Task 9 review-fix full repository gate: 1,603 passed.
- Stage 6.4 Task 9 focused tracker/atomicity files: 106 passed.
- Stage 6.4 Task 9 plus complete Stage 6.3 tracker/atomicity gate: 175 passed.
- Stage 6.4 Task 9 estimator/evaluator/emulator gate: 1,447 passed.
- Stage 6.4 Task 9 full repository gate: 1,548 passed.
- Stage 6.4 Task 8 focused tracker/atomicity files: 86 passed.
- Stage 6.4 Task 8 plus complete Stage 6.3 tracker/atomicity gate: 155 passed.
- Stage 6.4 Task 8 estimator/evaluator/emulator gate: 1,427 passed.
- Stage 6.4 Task 8 full repository gate including the numeric repair: 1,528
  passed.
- Canonical numeric-path focused gate: 69 passed on macOS and the exact-version
  Linux/amd64 Python 3.11 container; independent review clean.
- Stage 6.4 Task 7 review-fix focused tracker/atomicity files: 32 passed.
- Stage 6.4 Task 7 review-fix sparse/source compatibility gate: 313 passed.
- Stage 6.4 Task 7 full repository gate: 1,472 passed.
- Stage 6.4 Task 7 Ruff gate: All checks passed.
- Stage 6.4 Task 6 review-fix focused sparse fit file: 81 passed.
- Stage 6.4 Task 6 review-fix estimator/evaluator/emulator integration gate:
  1,339 passed.
- Stage 6.4 Task 6 review-fix full repository gate: 1,440 passed.
- Stage 6.4 Task 5 focused sparse fit file: 21 passed.
- Stage 6.4 Task 5 success/model slice: 3 passed, 18 deselected.
- Stage 6.4 Task 5 estimator/evaluator/emulator integration gate: 1,279 passed.
- Stage 6.4 Task 5 full repository gate: 1,380 passed.
- Stage 6.4 Task 3 source-bound model focused/differential gate: 120 passed.
- Stage 6.4 Task 3 estimator/evaluator/emulator integration gate: 1,313 passed.
- Stage 6.4 Task 3 full repository gate: 1,359 passed.
- Stage 6.4 Task 2 sparse-source regression-closure record contracts: 92 passed.
- Stage 6.4 Task 2 sparse-source regression-closure
  estimator/evaluator/emulator gate: 1,204 passed.
- Stage 6.4 Task 2 sparse-source regression-closure full repository gate:
  1,305 passed.
- Stage 6.4 Task 1 focused sparse primitive contracts: 50 passed.
- Stage 6.4 Task 1 estimator/evaluator/emulator gate: 1,162 passed.
- Stage 6.4 Task 1 full repository gate: 1,263 passed.
- Full repository suite including all three Task 20 fix waves: 1213 passed.
- Task 20 wave-three focused composition gate: 4 passed.
- Task 20 wave-three affected estimator/evaluator gate: 249 passed.
- Task 20 wave-two focused transaction/capability gate: 3 passed.
- Task 20 wave-two affected estimator/evaluator integration gate: 263 passed.
- Task 20 focused scientific/capability regression gate: 32 passed.
- Task 20 affected estimator/evaluator integration gate: 248 passed.
- Task 19 import, out-of-tree example, and guidance nodes: 3 passed.
- Task 18 acceptance file: 7 passed on each of two consecutive complete runs;
  its static exact-schedule node also passed twice independently.
- Dynamics, models, estimators, evaluator, and emulator focused integration
  gate: 1161 passed.
- Task 17 runner file: 78 passed.
- Task 17 named protocol matrix: 60 passed.
- Ruff across every tracked Python file: All checks passed.
- The Task 19 fail-fast package smoke built exactly one
  `nv_odmr_trackbench-0.1.0.tar.gz` and one
  `nv_odmr_trackbench-0.1.0-py3-none-any.whl`, then installed the wheel into a
  fresh environment, imported every planned Stage 6.3 estimator/evaluator
  public name under `-I`, and ran the calibrated two-point example from an
  unrelated working directory.

## Known scientific limitations

- The current dynamics layer provides only stationary and deterministic linear
  center drift; no Hamiltonian model exists yet.
- No external real-data file is bundled or attached, and the recording has no
  verified tracking truth.
- Hyperfine structure, ensemble inhomogeneity, optical power broadening,
  microwave power broadening, temperature coupling, and instrument transfer
  functions are not yet modeled.
- No benchmark results exist, so neither primary nor secondary hypothesis has
  supporting evidence.
- Raw playback retains unresolved analog units and has no measured timestamps;
  its CLI summary labels both its `unknown_analog_signal` quantity and its
  `conflicted_unverified` unit status; it is not a photon-count or timing claim.
- The fixed CLI emulator is synthetic. Its seeded output does not establish
  estimator accuracy, realtime performance, or agreement with the recording.
- The Stage 6.1 model does not resolve arbitrary overlapping, hyperfine-rich,
  asymmetric, or otherwise model-mismatched features. Its local amplitude
  significance is not a calibrated detector or false-discovery guarantee.

## Known software limitations

- End-to-end benchmark reproducibility has not yet been demonstrated beyond the
  installable package, command smoke tests, and deterministic synthetic
  configuration fixtures.
- The proof-of-concept emulator CLI accepts only its explicit Poisson-noise
  schema and fixed query schedule; adaptive estimator orchestration is not yet
  implemented.
- Warm-started fitting operates only after a sweep completes. Its measured
  update-core process CPU interval and optimizer evaluation count are
  machine-dependent descriptive diagnostics and establish neither within-sweep
  realtime utility nor universal computational improvement.
- The Task 10 test-only estimator isolation scanner does not recognize every
  exotic parent-relative import/identifier form and can conservatively flag
  forbidden words in docstrings; the production tree was separately inspected
  and contains no full-resource estimator path.
- The Task 11 test-only Task 10 delegation sentinel counts direct imported
  helper calls but does not yet detect a module-qualified helper call; current
  production uses only the reviewed direct method-local calls.
- Task 12's committed tests do not mutation-pin every schema metadata field,
  legal resource equality/treatment branch, isolated abort join, seven-phase
  over/under-validation branch, or equal-but-distinct outcome identity join;
  independent review found the production contracts conformant.
- The Stage 6.4 sparse evaluator runner now implements clean bind, verified
  calibration acquisition, authenticated tracking start, accepted/retryable
  acquisition transitions, timing retention, clean terminal transitions,
  returned-observation aborts, and lossless terminal resource construction.
  Deterministic linewidth drift and release-gated truth evaluation are available
  only to tests. The complete package export and public generated workflow are
  now available; comparative benchmark orchestration remains Stage 6.5.

## Next actions

1. Complete Stage 6.4 Task 20's integrated scientific/software review gates.
2. Preserve Stage 6.5 for matched-budget comparative benchmarks after the
   linewidth estimator exists.
