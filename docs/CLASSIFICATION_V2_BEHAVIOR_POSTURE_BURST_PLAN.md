# Classification V2 behavior-posture burst plan

## Status

- Version: `classification_v2_behavior_posture_burst_plan.v1`
- Date: `2026-08-01`
- Status: `PREPARATORY_IMPLEMENTATION_SYNTHETIC_PASS`
- Behavior review may continue unchanged.
- Active Behavior decisions must not be read while implementing the preparatory code.
- This plan does not authorize reviewed-data export, final training, or a paper claim.

Implemented without active review access:

- machine-readable behavior/posture and auto-validation contracts;
- safe burst-authority builder with frozen/synthetic Behavior authority gate;
- `UNKNOWN` posture target and mask propagation;
- strict window-authority alignment through immutable anchor keys;
- independent posture-aware auxiliary export and checker;
- conditional hierarchy consistency for `lying/sitting/stand/eat` only;
- audited upright auto-validation with Wilson lower-bound and abstention gates;
- deterministic selective-review scope builder and guarded CLI;
- mandatory review of non-upright, low-confidence, inconsistent and transition
  proposals, plus seeded high-confidence upright controls per stratum;
- synthetic and focused regression tests.

Not yet implemented or executed:

- RGB posture-proposer fitting and grouped calibration on real frozen data;
- selective posture-review GUI on real data;
- real posture authority, reviewed window rebuild, or training.

## Settled semantic contract

The supervised unit is a native annotated burst. Behavior and posture are each
expected to be stable across the frames of that burst. They are stored as two
fields on the same burst, not as a Cartesian compound class and not as unrelated
frame-level decisions.

The direct ten-class `behavior_target` remains the primary output for backward
compatibility. When no more specific action is present, `lying`, `sitting`, and
`stand` remain valid primary behavior states. An active behavior such as
`social-nose` or `fight` takes primary precedence while posture remains visible
through the independent posture target.

Examples:

| Observed burst | `behavior_target` | `posture_target` |
|---|---|---|
| lying without another action | `lying` | `lying` |
| sitting without another action | `sitting` | `sitting` |
| upright and inactive | `stand` | `upright` |
| eating at the fixed feeder | `eat` | `upright` |
| sitting social interaction | `social-nose` | `sitting` |
| lying fight | `fight` | `lying` |
| upright fight | `fight` | `upright` |

Posture V1 has exactly three learned classes:

```text
upright
sitting
lying
```

`UNKNOWN` is an authority state, not a fourth learned class. An unresolved
posture has an empty target and `posture_valid_mask=false`, so it contributes no
posture loss while remaining eligible for behavior supervision.

Body direction, head direction, and keypoint pose are outside V1. They may be
future review-independent model inputs only after a separate contract and
ablation.

### Move/stand temporal annotation rule

`move` and `stand` are episode-level behavior targets. A small displacement is
`move` when the middle burst anchor belongs to a continuous locomotion bout,
including its initiation or deceleration. A short local step, weight shift or
turn remains `stand` when the actor stays locally stationary before and after.

Surrounding frames identify continuous behavior episodes. The current reviewer
may use the following behavior to resolve a burst split across a genuine
transition. That provenance must be preserved; completed decisions must not be
silently reinterpreted as middle-anchor labels.

Before train-ready publication, audit a frozen transition stratum and choose one
contract: terminal-state labeling with matching future model context, or the
recommended middle-anchor label plus `behavior_transition_flag=true`. Do not mix
the two conventions silently. The flag denotes a valid temporal boundary, not
an unresolved label and not an additional behavior class.

Transition metadata remains target-side audit information. It must not enter
model-X, and any downweighting or exclusion requires a frozen paired ablation.
The effective model window must contain the temporal evidence needed by this
annotation rule; reviewer-only future evidence outside model support creates an
unlearnable target and must be resolved before scientific training.

## Current implementation gap

The current auxiliary target builder maps `lying` and `sitting` to themselves
and maps every other behavior to `standing_or_other`. That mapping cannot encode
`social-nose+sitting` or `fight+lying` and must not remain the final posture
authority.

The multitask model, loader, and trainer already contain a posture head and a
masked auxiliary loss path. The main changes are therefore semantic authority,
target construction, hierarchy consistency, balanced-loss composition, and
validation. A broad model rewrite is not justified.

## Where the change enters the pipeline

The change begins after frozen Behavior review and corrected bbox/identity
authority are available, but before the reviewed train-ready snapshot is
published:

```text
frozen corrected source and bbox authority
  -> frozen Behavior authority
  -> independent burst-posture authority
  -> rebuilt review-independent frame features and visual cache
  -> harmonized native units and T6/T8/T12/T16 windows
  -> behavior target plus posture target/mask sidecars
  -> dataset/collate
  -> shared encoder
  -> direct behavior head plus independent posture head
```

The raw XML, original Behavior decisions, and active review ledgers are not
rewritten. Posture is a new versioned sidecar keyed by immutable native-unit
identity.

RGB and spatial caches may be reused only when their source, bbox, frame-order,
resize, feature-schema, and index hashes still match. Bbox/source corrections
invalidate the affected cache rows. A target-only change does not by itself
invalidate numeric model-X caches.

## Phase 1: freeze the executable ontology

Create a machine-readable contract containing:

- canonical ten-class behavior order;
- posture order `upright`, `sitting`, `lying`;
- native-burst target granularity;
- `UNKNOWN` mask semantics;
- behavior/posture examples and forbidden compound labels;
- transition handling;
- authority precedence;
- permitted derivation rules;
- model-X forbidden fields;
- schema and checkpoint compatibility versions.

Initial safe derivations after Behavior review is frozen are:

| Frozen primary behavior | Derived posture | Initial authority |
|---|---|---|
| `lying` | `lying` | `DERIVED_SAFE` |
| `sitting` | `sitting` | `DERIVED_SAFE` |
| `stand` | `upright` | `DERIVED_SAFE` |
| `eat` | `upright` | `DERIVED_SAFE` |

For the current fixed pen and feeder geometry, the feeder permits pigs to eat
only while standing. Therefore a frozen reviewed `eat` burst has authoritative
`upright` posture. This rule must be revalidated if the feeder geometry, pen,
camera authority, or `eat` annotation definition changes.

`move`, `explore`, `drink`, `playwithtoy`, `fight`, and `social-nose` are not
assigned a posture from their behavior name. In particular,
`social-nose -> upright` is a population prior, not a row-level derivation rule.

If posture visibly changes inside a native burst, set
`posture_transition_flag=true` and `posture_valid_mask=false` for V1. Do not
silently select a majority posture.

## Phase 2: create independent posture authority

Create a sidecar with at least:

```text
native_temporal_unit_key
review_unit_id
behavior_target
posture_target
posture_valid_mask
posture_transition_flag
posture_authority
posture_authority_version
posture_proposal_confidence
posture_review_status
source_sha256
code_sha
```

Allowed authority values are:

- `HUMAN_REVIEWED`;
- `DERIVED_SAFE`;
- `AUTO_VALIDATED`;
- `UNRESOLVED`.

Authority, confidence, review status, and behavior labels are target-side audit
metadata. They must never enter model-X.

The authority builder must fail on duplicate keys, missing frozen Behavior
keys, unexpected extra keys, conflicting targets, or target-bearing columns in
the feature whitelist. It must report every unresolved burst rather than drop
it.

## Phase 3: build a high-precision posture proposer

The proposer reduces review effort; it does not become scientific label
authority merely because it emits a high score.

Training seeds come from frozen, safe posture anchors. Its candidate inputs are:

- actor RGB sequence from the native burst;
- valid bbox shape and normalized geometry;
- review-independent motion values with correct validity masks;
- temporal and padding masks.

Forbidden proposer inputs include behavior label, review reason, reviewer ID,
selection rank, source path, video ID as a predictive value, and posture target
derivatives.

Training and calibration must use recording/session/video-grouped partitions.
Random row splits are forbidden. The proposer must emit calibrated probabilities
and an explicit abstention decision.

An auto-carry stratum is eligible for `AUTO_VALIDATED` only after a predeclared
human audit. The proposed default gate is a one-sided 95% lower confidence bound
of at least 0.98 for precision. Freeze the exact gate and sample size before
opening the audit results. If the gate fails, the stratum remains unresolved.

## Phase 4: perform selective posture review

Do not make the operator review every burst again. Build a separate compact
posture review scope containing:

- every uncertain or abstained proposal;
- every proposed `sitting` or `lying` result in active interaction behaviors;
- every detected within-burst transition;
- a stratified random control from proposed `upright` bursts;
- controls stratified by behavior, source provenance, video/session, and
  proposal-confidence band.

Review media should show the complete native burst and enough adjacent context
to distinguish a posture transition from a stable posture. One decision applies
to the native burst. The GUI must not expose Behavior model predictions or alter
the frozen Behavior decision.

The review closes only when every selected key has exactly one resolved posture
decision or an explicit technical exclusion.

## Phase 5: audit temporal and spatial evidence

Posture is a target, not an excuse to inject label-derived values into the 46D
spatial vector. Audit the existing RGB, spatial, ROI, social, and motion evidence
before changing numeric features.

The audit must answer:

- whether bbox aspect, normalized size, and geometry are already present and
  valid;
- whether missing motion is distinguished from stationary motion;
- whether padding and missing-modality masks survive to model forward;
- whether first-frame motion invalidity is masked;
- whether source-specific preprocessing remains after harmonization;
- whether a simple RGB-only posture baseline already separates the three
  postures;
- whether spatial information adds evidence beyond RGB under grouped folds.

Do not change the 46D order for structural symmetry. Add or version a feature
only when the audit and a matched ablation show that the information is missing
and useful.

## Phase 6: preserve native-unit semantics in temporal windows

CVAT native units have six frames and legacy native units may have sixteen. The
claim that behavior and posture are stable applies to one authoritative native
unit, not to an arbitrary window stitched across units.

For every T6/T8/T12/T16 artifact:

- bind the target to one anchor native unit;
- forbid target assignment from a neighboring unit;
- keep all overlapping windows from an anchor unit in one grouped split;
- record whether context extends outside the anchor unit;
- never interpret a longer context window as proof that one posture holds at
  every context timestep;
- fail on mixed target authority without an explicit anchor and target mask.

The model may consume longer context, but the burst target remains scalar. The
loader may broadcast that scalar for a sequence-level loss only; it must not
write fabricated frame-level posture labels.

## Phase 7: version target export and data loading

Preserve `y_behavior.csv` and its canonical ten-class order. Replace the current
derived posture content with authority-backed fields in the auxiliary target
artifact:

```text
window_id
native_temporal_unit_key
behavior_target
posture_target
has_posture_aux_target
posture_authority
```

`has_posture_aux_target` is true only for resolved posture authority. The data
module must align targets by immutable key, not row position alone, and fail
closed on duplicates, reorder mismatch, missing required fields, or unknown
posture strings.

Target and authority fields remain outside `model_inputs`. Collate must preserve
the scalar target and mask without confusing them with padding or spatial
availability masks.

## Phase 8: update model and loss semantics

Keep direct ten-class behavior supervision. Keep one independent three-class
posture head over the shared multimodal embedding. Do not hard-route posture
argmax into the behavior head.

The loss is:

```text
total_loss
  = behavior_weight * balanced_behavior_loss
  + posture_weight * masked_posture_loss
  + bounded_hierarchy_weight * valid_hierarchy_loss
  + other approved auxiliary losses
```

Behavior imbalance policies and posture imbalance policies are fitted from the
training fold only. The newer balanced behavior loss must be composed with the
multitask trainer rather than replacing or bypassing auxiliary supervision.

The current hierarchy consistency mapping must be narrowed:

- `behavior=lying` may constrain posture to `lying`;
- `behavior=sitting` may constrain posture to `sitting`;
- `behavior=stand` may constrain posture to `upright`;
- active behaviors impose no fixed posture constraint;
- unresolved posture contributes no posture or hierarchy loss.

Thus `social-nose+sitting`, `fight+lying`, and `fight+upright` are all legal.

## Phase 9: validation ladder

Add independent expected-value tests for:

1. safe lying, sitting, stand, and fixed-feeder eat anchor mappings;
2. social-nose with each of the three valid postures;
3. fight with lying and upright posture;
4. unresolved posture producing zero posture loss but normal behavior loss;
5. transition burst producing an invalid posture mask;
6. active behavior receiving no forced upright hierarchy target;
7. target and authority columns excluded from model-X;
8. immutable-key alignment across exporter, loader, collate, and forward;
9. T6/T8/T12/T16 anchor-unit target preservation;
10. no native-unit or overlapping-window split leakage;
11. finite forward/backward loss and gradients;
12. behavior-only checkpoint compatibility or explicit version rejection;
13. deterministic repeat under the frozen seed;
14. unchanged behavior targets before and after posture migration.

Then run, in order:

- schema/static checks;
- synthetic target and mask tests;
- one batch for every supported T;
- forward-only smoke;
- one optimizer step;
- deterministic tiny overfit;
- short grouped posture proposer validation;
- selective-review dry run without active ledger access;
- reviewed-data rebuild only after all review authorities are frozen.

## Phase 10: evaluation and acceptance

Behavior metrics remain the direct ten-class metrics. Add posture metrics only
on rows with resolved posture authority:

- macro F1 and balanced accuracy;
- per-posture precision, recall, F1, and support;
- posture confusion matrix;
- behavior metrics stratified by posture where support permits;
- joint behavior-posture support and error table;
- source/video/date stratification;
- calibration and abstention coverage;
- AUTO_VALIDATED versus HUMAN_REVIEWED comparison;
- transition and unresolved counts.

Acceptance requires:

- zero changed frozen Behavior decisions;
- zero target/review fields in model-X;
- zero native-unit or grouped split leakage;
- exact target-key coverage with explicit unresolved rows;
- all legal behavior-posture combinations accepted by the model contract;
- auto-carry precision gate passed per accepted stratum;
- no unexplained behavior metric regression in matched smoke checks;
- finite losses and gradients;
- versioned checkpoint and replay manifests.

## Recompute decision matrix

| Artifact | Default action | Rebuild trigger |
|---|---|---|
| raw XML and original labels | preserve | never from this plan |
| frozen Behavior decisions | preserve | never from this plan |
| posture authority sidecar | create | new or changed posture authority |
| RGB crop cache | reuse by hash | bbox, source frame, resize, or index change |
| spatial/motion cache | reuse by hash | corrected bbox, schema, mask, or ROI change |
| adjusted ROI features | rebuild | required adjusted ROI authority changes inputs |
| T6/T8/T12/T16 manifests | full reviewed rebuild | post-review workflow already requires it |
| behavior target | regenerate and hash-compare | reviewed Behavior apply |
| posture target/mask | regenerate | posture authority or alignment change |
| model checkpoint | new schema | head order, loss, or input contract change |

## Work that may begin while Behavior review continues

The following preparatory implementation is permitted without reading active
decisions:

- ontology and schema contracts;
- synthetic fixtures;
- posture sidecar builder operating on synthetic/frozen inputs;
- proposer interfaces and abstention logic on synthetic data;
- loader/model/loss unit tests;
- dry-run CLI and active-ledger path rejection;
- migration and cache-reuse audit code.

The following must wait for frozen authorities:

- reading reviewed Behavior outcomes;
- deriving posture anchors from reviewed labels;
- fitting or calibrating the real posture proposer;
- constructing the real selective posture review scope;
- applying posture decisions;
- rebuilding reviewed train-ready artifacts;
- any reviewed-data training or scientific metric claim.

## Required implementation artifacts

At minimum produce:

1. `behavior_posture_contract.json`;
2. `posture_burst_authority.csv`;
3. `posture_burst_authority_manifest.json`;
4. `posture_proposal_manifest.json`;
5. `posture_proposal_calibration.json`;
6. `posture_review_scope.csv`;
7. `posture_review_close_authority.json`;
8. updated auxiliary target artifact and audit;
9. temporal target-alignment audit;
10. model-X leakage audit;
11. cache reuse/rebuild report;
12. focused test commands and results;
13. final behavior-posture integration report.

## Stop conditions

Stop on active Behavior ledger access, duplicate or missing authority keys,
behavior decision mutation, behavior-derived posture for active behaviors,
target fields entering model-X, mixed native-unit targets, grouped split leakage,
unvalidated auto-carry, unresolved cache lineage, non-finite loss/gradients, or
an unexplained change to canonical behavior order.

## Selected skills

- `agent-architecture-audit`: trace designed versus effective multitask runtime;
- `plan-orchestrate`: decompose the accepted design into gated execution stages;
- `dataset-contract-leakage-guard`: protect target, temporal unit, split, and
  model-X boundaries;
- `multimodal-sequence-model-builder`: define independent heads, masks, loss,
  forward, and checkpoint contracts;
- `safe-refactor-test-guardian`: keep implementation small, versioned,
  reversible, and focused-test backed;
- `project-state-steward`: reconcile the durable decision and handoff state.
