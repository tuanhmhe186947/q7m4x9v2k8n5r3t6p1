# Classification V2 Spatiotemporal Base-Selection Execution Plan

## 1. Objective

Select the smallest defensible classifier base that can consume the useful
visual, temporal, and later spatial information in `classification_v2`.
Legacy 16f is a screening lane. The final base for full data is selected only
after the finalists are repeated on a frozen mixed-reviewed snapshot.

The plan must answer four separate questions:

1. Does using multiple frames help beyond one center frame?
2. Does content weighting help beyond an unweighted mean?
3. Does ordered temporal modeling help beyond parameter-matched pooling?
4. Which spatial families add value after the temporal base is fixed?

## 2. Claim Boundary

```text
legacy full-development screening
  = all eligible legacy train units plus the fixed development validation role
  != mixed-reviewed full-data confirmation
  != full OOF
  != final Q2 evidence
```

Legacy evidence may eliminate weak designs and nominate at most three
finalists. It cannot by itself set the production or thesis base.

## 3. Frozen Screening Universe

- lineage: `legacy-only-unreviewed-development`;
- native evaluation unit: one complete 16-frame burst;
- train native units: 3,652 for full-development confirmation;
- validation native units: 245 across 33 videos;
- source cache: frozen ResNet18 224x224 frame features;
- direct 10-class supervision and fixed global label order;
- event-balanced CE policy, seed, optimizer, epoch count, and split fixed;
- one model sequence per native unit unless a protocol explicitly says sliding;
- no source media reads and no outer-holdout predictions.

Every semantic configuration must pass synthetic checks and two independent
short processes before a full-development run.

## 4. Stage A: Temporal Base Matrix

Use centered C6 offsets `[5,6,7,8,9,10]` unless stated otherwise.

| ID | Input | Encoder | Hidden | Parameters | Decision isolated |
|---|---|---|---:|---:|---|
| SF128 | offset `[7]` | masked mean | 128 | 68,234 | one versus multiple frames |
| M128 | C6 | masked mean | 128 | 68,234 | fixed basic control |
| A128 | C6 | masked attention | 128 | 68,363 | learned content weighting |
| MW317 | C6 | masked mean | 317 | 167,459 | capacity control for TCN |
| TCN128 | C6 | two-layer masked TCN | 128 | 167,435 | ordered local dynamics |
| MW381 | C6 | masked mean | 381 | 201,059 | capacity control for Transformer |
| TR128 | C6 | one-layer Transformer | 128 | 200,843 | order plus time-position encoding |

Primary controlled pairs are `M128-SF128`, `A128-M128`, `TCN128-MW317`, and
`TR128-MW381`. Operational comparisons against M128 are secondary.

## 5. Stage B: Order And Timing Diagnostics

Run only for an order-aware encoder that survives Stage A.

### TCN diagnostics

- original C6 order;
- deterministic non-temporal permutation with identical frame content;
- inference-only reversal sensitivity;
- same parameters, units, steps, and masks.

### Transformer diagnostics

- real observed deltas;
- zero deltas, which removes time-position information;
- deterministic feature permutation with original temporal positions;
- inference-only reversal sensitivity.

Legacy frames are regularly sampled. Therefore real versus normalized elapsed
time is not a general irregular-timestamp test. True elapsed-time utility must
be confirmed on mixed-reviewed data with observed gaps or source variation.

## 6. Stage C: Sampling View Confirmation

Use the selected Stage A encoder and compare exactly one sequence per native:

| View | Offsets | Isolated variable |
|---|---|---|
| C6 | `5,6,7,8,9,10` | six-frame contiguous control |
| C8 | `4,5,6,7,8,9,10,11` | sequence length |
| S6 | `0,3,6,9,12,15` | temporal span at six frames |

Do not compare these causally with historical T6 sliding unless optimizer
exposure and inference-window count are separately controlled.

## 7. Stage D: Spatial Families

Freeze the selected temporal base before adding spatial information.

| ID | Added information | Primary target groups |
|---|---|---|
| B0 | none | temporal visual control |
| G | geometry | posture and body scale |
| M | motion derivatives | move, explore, transitions |
| R | all-ROI relations | drink, eat, playwithtoy |
| P | pen-boundary context | boundary-conditioned posture and motion |
| S | numeric social relations | fight and social-nose |

Run each branch independently against B0. Do not run all `2^N` combinations.
Only branches that pass their individual gate may enter combinations, in this
order: `G+M`, `G+M+R`, then one context branch at a time. Availability-only and
parameter-matched controls remain mandatory for optional modalities.

## 8. Decision Metrics

Each comparison must report:

- pooled native 10-class macro-F1, weighted-F1, accuracy, macro recall, NLL;
- per-class precision, recall, F1, support, and true-label NLL;
- interaction, ROI-behavior, posture, locomotion, and rare macro-F1;
- confusion pairs and paired correctness transitions;
- 2,000-draw paired video-cluster bootstrap intervals;
- parameter count, optimizer steps, runtime, and peak memory;
- missingness, source, and inference-availability risks.

Point gains without support-aware intervals are descriptive only.

## 9. Promotion And Elimination

A candidate is promoted from legacy screening only when:

1. correctness, determinism, resume, count, and lineage gates pass;
2. the primary controlled-pair macro-F1 delta is positive;
3. the paired interval is not materially negative;
4. rare and target-group macro-F1 do not regress beyond the declared limit;
5. gains are not explained only by parameter count or extra optimizer steps;
6. all inputs exist at inference time;
7. runtime and memory remain feasible on the declared execution profile.

Failing a gate eliminates the candidate from combinations. Ambiguous evidence
retains the simpler model and records the candidate for mixed-reviewed retest.

## 10. Full-Data Base Selection Packet

Every legacy comparison must end with one transfer action:

```text
CARRY = retain as a full-data finalist because the controlled evidence is useful
DROP = eliminate because the controlled hypothesis failed
RETEST = legacy support or acquisition conditions cannot answer the question
```

The decision record must separate:

- evidence expected to transfer across sources, such as parameter-controlled
  order sensitivity and inference-safe tensor contracts;
- legacy-specific evidence, such as effects that depend on regular timing,
  fixed-camera calibration, or weak class support;
- unresolved full-data questions, including CVAT timing, modality missingness,
  source balance, and reviewed rare-class support;
- the exact mixed-reviewed comparison that can confirm or reject transfer.

Full-data promotion requires a positive paired result on the pooled native-unit
metric plus no material regression on either source, the target behavior group,
or the missing-modality strata. A pooled gain driven only by legacy is not
sufficient. When transfer evidence is ambiguous, retain the simpler base.

Legacy screening emits at most three entries:

```text
F0 = simplest valid control
F1 = best temporal finalist
F2 = best individually validated spatiotemporal finalist
```

For each finalist, the packet must bind config, code, cache, fold, checkpoint,
prediction, metric, parameter, runtime, and decision hashes. It must also state
which legacy classes are under-supported and which conclusions require
mixed-reviewed confirmation.

The final base is selected only on the frozen mixed-reviewed snapshot using
the same grouped folds and paired native units. Required final evidence adds
per-source metrics, source/missingness probes, and class-by-fold support. Full
OOF remains blocked until its separate launch gate passes.

### 10.1 Decision utility contract

Every stage packet must answer four questions before it can influence the
full-data base:

1. Which single scientific factor changed, and which controls were matched?
2. Which native units, recording/video clusters, and source strata support the
   result?
3. Is the result a transferable mechanism, a legacy-only observation, or an
   unresolved question requiring a mixed-reviewed retest?
4. What exact candidate and control must be run next on mixed data?

The final mixed-data base decision must include, for each finalist:

- pooled native-unit macro-F1, macro recall, weighted-F1, accuracy, and NLL;
- per-source and source-balanced metrics, with class-by-source support;
- missing-modality strata and context-availability controls;
- rare, interaction, ROI-behavior, posture, and locomotion group metrics;
- paired native-unit correctness transitions and video-cluster intervals;
- parameter count, optimizer exposure, runtime, peak memory, and failures;
- exact config, code, cache, fold, checkpoint, prediction, and metric hashes.

A pooled gain cannot promote a candidate when it is driven only by legacy
units, by one source, by availability flags, or by unsupported classes. The
base is selected only when the candidate beats the registered control under
the declared paired gate without a material source or target-group regression.
If the evidence is ambiguous, retain the simpler control and mark the richer
candidate `RETEST`, rather than treating a legacy point estimate as a final
architecture choice.

The current legacy evidence has these decision roles:

| Evidence | Permitted use | Not permitted |
|---|---|---|
| Stage A temporal matrix | screen encoder complexity | choose the final base |
| Stage C C6/C8/S6 | screen span and sampling | replace the mixed view |
| Pen-boundary branch | test a spatial hypothesis | reject it for mixed data |
| Legacy class/group metrics | identify hypotheses | support a Q2/final claim |

The current mixed-data confirmation set is at least `SF128` versus `A128` on
the same frozen native-unit folds, with C6/fixed observed-time contracts,
source-balanced reports, and missingness controls. A spatial finalist may be
added only after its independent branch passes the same short and paired gates.

## 11. Stage A Execution Record (2026-07-17)

The short gate passed for all seven modes with two fresh, non-overlapping
processes, identical repeat hashes, 80 selected train units, 245 validation
units, and nine optimizer steps. Full development then passed for all modes
with 3,652 train units, 245 validation units, and 345 steps.

The hash-bound transfer packet is the Stage A v3 artifact under
`outputs/classification_v2/agent_experiments/legacy_16f_temporal_base_selection_20260717_v1/`
`paired_decision_stage_a_v3/`.
It reports `SF128` as the only legacy-carried control. `A128` is a conditional
mixed-reviewed retest: its global gain is positive, but its legacy locomotion
group regression and low rare-class support prevent carry. `M128`, `TCN128`,
and `TR128` are dropped from legacy expansion. No order-aware encoder survived
Stage A, so Stage B diagnostics are not authorized yet.

This record is legacy screening evidence only. It does not choose the final
full-data base, authorize full OOF, or support a Q2 claim. Mixed-reviewed
confirmation must compare at least `SF128` and the `A128` retest under the
same grouped native-unit universe, with source and missing-modality strata.

### 11.1 Stage C execution record (2026-07-17)

The one-sequence sampling matrix passed its short repeat and full-development
gates on the same 245 native units and 33 video clusters. The results were:

| View | Macro-F1 | NLL | Transfer role |
|---|---:|---:|---|
| C6 centered contiguous | 0.3708555 | 1.0664057 | retain as legacy working view |
| C8 centered contiguous | 0.3588478 | 1.0379916 | do not promote |
| S6 uniform span-16 | 0.3334808 | 1.0706695 | do not promote |

C8 improves NLL but not macro-F1. S6 has descriptive gains for `drink` and
`move`, but the paired interval crosses zero and `move` has only eight units.
The registered legacy working view is C6. This does not replace the mixed
`fixed6_observed_time` contract or select the final full-data base.

The decision packet is
`outputs/classification_v2/agent_experiments/legacy_16f_temporal_sampling_20260717_v2/`
`paired_decision_v2/temporal_sampling_decision.json` with SHA256
`cdd24a27162ec46bc68214e6820e3aa41aebe86da53acd6903da175bcced2cfa`.

### 11.2 Spatial screening record: pen boundary context

The pen-boundary branch passed correctness, determinism, and short resource
gates. Against the parameter-matched zero control, macro-F1 changed from
`0.2774732` to `0.2773207`; NLL improved from `1.8307485` to `1.8207899`.
The video-cluster interval was `[-0.0183286, 0.0193813]`, so the global
promotion gate failed. A focus diagnostic found a `stand/move/explore` gain,
but this is conditional legacy evidence and does not justify adding the
feature to the base. Reassess the same isolated branch on reviewed mixed data
only if its target-group hypothesis remains important.

The decision is
`DO_NOT_EXPAND_PEN_CONTEXT_FROM_CURRENT_SHORT_EVIDENCE`. Its packet is
`outputs/classification_v2/agent_experiments/legacy_16f_pen_context_20260717_v1/`
`decision/pen_context_short_decision.json` with SHA256
`673ddab840e5d69f984b47c9d832e2415147681f3df6b81448270766ab673e1c`.

Stage B is skipped because no order-aware encoder survived Stage A. Therefore
the next scientific decision is not another legacy architecture expansion:
freeze the mixed-reviewed snapshot, pass the short gate, and compare the
registered finalists under the full-data contract.

### 11.3 Existing spatial transfer ledger

The following legacy L6 branches are already screened. Do not repeat them on
legacy unless a semantic configuration changes and a new short gate is earned:

| Branch | Legacy action | Mixed-data implication |
|---|---|---|
| motion | `DROP` from legacy expansion | reassess only with reviewed support |
| ROI relations | `RETEST` | feeding signal needs mixed support |
| numeric social | `DROP`; top-K deferred | reassess interaction support |
| union crop | `DROP` from current short evidence | reassess after review |
| full-frame context | `DROP` from current short evidence | reassess with source controls |
| pen boundary | `DROP` globally; conditional signal | isolated mixed branch only |

These actions do not mean the information is biologically useless. They mean
the legacy lane did not provide a sufficiently stable promotion signal. A
mixed-data branch may enter the finalist set only with the same parameter-
matched zero/availability controls, source-stratified metrics, and paired
native-unit uncertainty required by Section 10.1.

## 12. Execution Order

```text
Stage A short/full decision
  -> Stage C sampling confirmation (complete)
  -> independent spatial screening (pen branch complete)
  -> mixed-reviewed short gate after human-review handoff
  -> paired mixed-data finalist confirmation
  -> full-data base lock
  -> full OOF launch gate
```

### 12.1 Mixed-finalist preflight

After the operator supplies a `behavior_complete` handoff, create the
comparison contract from
`configs/classification_v2/reviewed_q2_mixed_finalist_contract_template_v1.json`
under the new agent audit root. Run
`check_classification_v2_reviewed_q2_mixed_finalist_preflight.py` before any
paired model smoke. The preflight re-runs the reviewed-Q2 P0 checks, verifies
the same snapshot, native units, folds, preprocessing, seed, loss, sampler,
optimizer exposure, and `fixed6_observed_time` view for `SF128` and `A128`.
It also requires both sources, class and missingness support, the temporal
shortcut audit, and inference-safe inputs. It authorizes only the short paired
gate; it never authorizes development training or full OOF. The pair is a
composite temporal-finalist choice, so it cannot support an attention-only
mechanism claim.
