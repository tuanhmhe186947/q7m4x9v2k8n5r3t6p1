# Tracking Benchmark Metric Recommendation

Date: 2026-07-28

The minimum set below is designed for identity-linked pig behavior profiling.
Each metric must answer a distinct question and have a deterministic,
auditable definition. `Scientific priority` and `implementation status` use the
requested classifications independently.

## Standard core metrics

- `HOTA`: `REQUIRED_PRIMARY`; `AVAILABLE_BUT_NEEDS_VALIDATION`.
  It asks whether detection and identity association are jointly reliable
  across localization thresholds.
- `DetA`: `REQUIRED_PRIMARY`; `AVAILABLE_BUT_NEEDS_VALIDATION`.
  It isolates detection matching from identity association.
- `AssA`: `REQUIRED_PRIMARY`; `AVAILABLE_BUT_NEEDS_VALIDATION`.
  It measures association consistency across a trajectory.
- `LocA`: `REQUIRED_PRIMARY`; `NOT_IMPLEMENTED`.
  It measures localization of true-positive matches.
- `IDF1`: `REQUIRED_PRIMARY`; `ALREADY_AVAILABLE`.
  It measures globally consistent identity detection mass per sequence.
- `ID precision`: `REQUIRED_PRIMARY`; `NOT_IMPLEMENTED`.
  It measures the correctness of predicted identity mass.
- `ID recall`: `REQUIRED_PRIMARY`; `NOT_IMPLEMENTED`.
  It measures how much GT identity mass is recovered.
- `IDSW`: `REQUIRED_PRIMARY`; `ALREADY_AVAILABLE`.
  It counts predicted-identity transition events for GT trajectories.
- `FP`: `REQUIRED_PRIMARY`; `ALREADY_AVAILABLE`.
  It counts unmatched predicted boxes.
- `FN`: `REQUIRED_PRIMARY`; `ALREADY_AVAILABLE`.
  It counts unmatched authoritative GT boxes.
- Detection precision: `REQUIRED_PRIMARY`; `ALREADY_AVAILABLE`.
  It measures the fraction of predictions that are matched.
- Detection recall: `REQUIRED_PRIMARY`; `ALREADY_AVAILABLE`.
  It measures the fraction of authoritative GT boxes that are matched.
- Fragmentation: `REQUIRED_PRIMARY`; `ALREADY_AVAILABLE`.
  It counts matched GT trajectories that break and later resume.

HOTA, DetA, and AssA are present by name but are single-threshold HOTA-style
values. They require a separately authorized correction and reference parity
before standard headline use. FP, FN, precision, recall, IDF1, IDSW, and
fragmentation are implemented, but all match-dependent metrics remain exposed
to the frame-assignment gating defect.

## Project-specific identity-severity diagnostics

Every item in this section must be labeled project-specific in reports.

- Wrong-ID matched frames: `REQUIRED_DIAGNOSTIC`; `ALREADY_AVAILABLE`.
  This measures authoritative matched exposure with the wrong identity.
- Wrong-ID matched seconds: `REQUIRED_DIAGNOSTIC`; `NOT_IMPLEMENTED`.
  This measures wrong-identity exposure in physical time.
- Identity-error episode count: `REQUIRED_DIAGNOSTIC`;
  `AVAILABLE_BUT_NEEDS_VALIDATION`. This counts connected corruptions.
- Median episode duration: `REQUIRED_DIAGNOSTIC`; `NOT_IMPLEMENTED`.
  This measures typical identity-error persistence.
- 95th-percentile episode duration: `REQUIRED_DIAGNOSTIC`;
  `NOT_IMPLEMENTED`. This measures tail persistence.
- Maximum episode duration: `REQUIRED_DIAGNOSTIC`; `NOT_IMPLEMENTED`.
  This records the worst identity corruption duration.
- Permanent swap count: `REQUIRED_DIAGNOSTIC`;
  `AVAILABLE_BUT_NEEDS_VALIDATION`. This counts episodes satisfying the
  declared persistent/permanent contract.
- Terminal swap count: `REQUIRED_DIAGNOSTIC`;
  `AVAILABLE_BUT_NEEDS_VALIDATION`. This counts GT trajectories wrong at
  their final authoritative match.
- Recovery latency: `REQUIRED_DIAGNOSTIC`; `NOT_IMPLEMENTED`.
  This measures time until an authoritative correct match returns.
- GT trajectories with any identity error: `REQUIRED_DIAGNOSTIC`;
  `NOT_IMPLEMENTED`. This measures how broadly animals are affected.
- Percentage of GT trajectories with any identity error:
  `REQUIRED_DIAGNOSTIC`; `NOT_IMPLEMENTED`. This normalizes that population.
- Videos with permanent or terminal swap: `REQUIRED_DIAGNOSTIC`;
  `AVAILABLE_BUT_NEEDS_VALIDATION`. This measures sequence distribution.
- Percentage of videos with permanent or terminal swap:
  `REQUIRED_DIAGNOSTIC`; `AVAILABLE_BUT_NEEDS_VALIDATION`.
- Worst-video identity statistics: `REQUIRED_DIAGNOSTIC`; `NOT_IMPLEMENTED`.
  This exposes catastrophic sequences hidden by an aggregate.
- Median-video identity statistics: `REQUIRED_DIAGNOSTIC`; `NOT_IMPLEMENTED`.
  This describes the typical sequence.
- IDSW per 1,000 authoritative matched GT frames: `REQUIRED_DIAGNOSTIC`;
  `NOT_IMPLEMENTED`. This normalizes switch events by exposure.

Wrong-ID rows are available in remapped identity-event exports. The current
connected episode, permanent, and terminal summaries need correction before
scientific use because span, recovery, GT-primary grouping, and unresolved-GT
policies are not yet valid.

## Optional ablations

- Mostly/partially tracked and mostly lost: `OPTIONAL_ABLATION`;
  `NOT_IMPLEMENTED`. This tests whole-trajectory coverage.
- Per-GT identity purity: `OPTIONAL_ABLATION`; `NOT_IMPLEMENTED`.
  This tests whether one stable predicted identity dominates a GT trajectory.
- Strict versus gap-tolerant IDSW: `OPTIONAL_ABLATION`; `NOT_IMPLEMENTED`.
  This tests sensitivity to switch-gap policy.
- Strict versus gap-tolerant fragmentation: `OPTIONAL_ABLATION`;
  `ALREADY_AVAILABLE`. This tests sensitivity to short misses.
- Hidden versus visible strata: `OPTIONAL_ABLATION`; `NOT_IMPLEMENTED`.
  This tests concentration in labeled hidden intervals.
- Lighting, crowding, and occlusion strata: `OPTIONAL_ABLATION`;
  `NOT_IMPLEMENTED`. These test later evaluation conditions.
- MOTA: `OPTIONAL_ABLATION`; `ALREADY_AVAILABLE`.
  It measures combined CLEAR MOT error burden.
- MOTP IoU: `OPTIONAL_ABLATION`; `ALREADY_AVAILABLE`.
  It measures mean IoU among retained matches at the configured threshold.

The optional strata are justified only when their authority labels are
predeclared and sufficiently reliable. Current Hidden annotations require an
explicit authority caveat.

## Not justified

- Beneficial/harmful repair-event labels without GT: `NOT_JUSTIFIED`;
  `NOT_IMPLEMENTED`. They make a quality claim without evaluation authority.
- Unnormalized composite severity score: `NOT_JUSTIFIED`; `NOT_IMPLEMENTED`.
  It obscures distinct event count, duration, and exposure questions.
- `permanent IDSW`: `NOT_JUSTIFIED`; `NOT_IMPLEMENTED`.
  It incorrectly presents persistence as a standard IDSW subtype.

No additional metric should be added merely because it is available. The
standard core, explicit project-specific severity diagnostics, and a small
number of predeclared ablations are sufficient for the next scientific phase.
