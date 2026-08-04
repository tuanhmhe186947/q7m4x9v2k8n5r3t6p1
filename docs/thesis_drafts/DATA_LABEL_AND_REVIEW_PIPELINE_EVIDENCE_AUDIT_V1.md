# Data, Labeling, and Human-Review Pipeline Evidence Audit V1

> **Scope correction (2026-08-02).** The earlier body of this document was
> too Hidden-centric. Section **O** is the authoritative full-pipeline
> reconstruction requested by the user and supersedes any narrower wording
> above. It covers detection-data construction, tracking, both behavior
> sources, Hidden, review-candidate generation, corrected-source lineage,
> temporal windows, leakage controls, and train-ready status.

**Audit date:** 2026-08-02
**Scope:** read-only reconstruction of the detection, tracking, behavior,
visibility, and human-review pipeline for the thesis
**Primary status:** evidence-backed methodology draft; not a training-data
authorization

## A. Executive verdict

The project contains two different layers that must not be collapsed in the
thesis. The first layer constructs frame-level, identity-bearing observations
from detection and tracking annotations. The second layer derives review
evidence and asks a human reviewer to confirm or correct visibility and behavior
labels. A review candidate is therefore an inspection request, not a new ground
truth label. Only an explicitly resolved decision, followed by the corresponding
source apply and integrity audits, can change the reviewed source.

The most important correction concerns **Hidden**. Hidden is a frame--object
visibility attribute, independent of the behavior target. It is not a native
behavior unit and it is not propagated to an entire six- or sixteen-frame unit
without an explicit span decision. CVAT Hidden values are treated as
tracking-derived and untrusted until current human review. Legacy values may
retain prior-review provenance as metadata, but the active mixed-source lineage
still requires a new two-sided Hidden review before reviewed or train-ready use.

The current project authority records the technical Hidden design but no
verified current human coverage. The versioned technical reference contains a
target-independent review manifest and media checks; its old decision payloads
are explicitly unverified and cannot be cited as completed human review. Thus,
the thesis may describe the protocol as implemented, but must not report Hidden
review completion or a final reviewed dataset until a clean lineage passes the
current authority gates.

## B. Actual end-to-end pipeline

```text
raw RGB video / legacy recovered crops
        |
        v
frame selection and detector-data annotation (historical detection workflow)
        |
        v
detector-assisted tracking pre-annotation
        |
        v
CVAT tracking correction and source-local identity/bounding-box records
        |
        +--> frame-local geometry, ROI, social, motion, timestamp primitives
        |          |
        |          v
        |    target-independent Hidden review manifest
        |          |
        |          v
        |    human Hidden decision: Yes / No / Unclear
        |          |
        |          v
        |    Hidden apply and trust/mask audit
        |          |
        |          v
        |    temporal harmonization (CVAT 6f; legacy 16f)
        |          |
        |          v
        |    native-unit review evidence and behavior review candidates
        |          |
        |          v
        |    human behavior decision: accept / corrected / exclude
        |          |
        |          v
        |    corrected frame source and final exact-view rebuild
        |          |
        |          v
        |    leakage checks, grouped evaluation, and train-ready snapshot
        |
        +--> audit-only review metadata, sampling rationale, and notes
             (never model input)
```

The canonical dependency order is therefore **Hidden review before temporal
harmonization, behavior review after native temporal evidence, and final model
windows only after both reviewed-source stages**. The sequence-window builder
does not define the annotation truth; it consumes the reviewed frame/native
source.

## C. Stage-by-stage authority table

| Stage | Granularity | Evidence and output | Status | Safe for training? |
|---|---|---|---|---|
| Detection frame selection | image/frame | Background and pen mask, activity, perceptual hash, per-video/day balancing; notebooks under `notebooks/01_data_preparation/` | historical implemented workflow | only after a separate detector-data audit |
| Detection annotation | frame-object | Roboflow/CVAT-style box annotations used to train the detector | inherited and historical | detector training only; not behavior authority |
| Detector-assisted tracking | frame-object/trajectory | provisional boxes and track IDs | model-assisted, provisional | no, until corrected/audited |
| CVAT tracking correction | frame-object/trajectory | corrected boxes, source-local track IDs, visibility and behavior attributes | implemented source parser and correction workflow | source input only after source audit |
| Frame-local feature construction | frame-object | geometry, ROI, partner, motion, timestamp, Hidden provenance | implemented | derived evidence, not labels |
| Hidden review selection | frame-object | four disjoint cohorts and target-independent risk/stratum metadata | implemented contract; active clean review pending | no |
| Human Hidden review | frame-object | Yes/No/Unclear plus reviewer provenance | protocol implemented; current verified coverage absent | no until coverage and scientific gate pass |
| Temporal harmonization | native temporal unit | CVAT anchor interval or legacy burst, consistency and Hidden ratios | implemented downstream stage | no before Hidden apply |
| Behavior review selection | native temporal unit | mandatory, high-risk, random-audit, and not-selected cohorts | implemented selection contract | no |
| Human behavior review | native temporal unit | accept, corrected behavior, or technical exclusion | current closure still subject to authority gates | no until apply and audits pass |
| Final view construction | model window/native unit | T6/T8/T12/T16 and declared sparse ablations | implemented downstream | only after reviewed snapshot |

## D. Label granularity and propagation

| Label or field | Actual semantic grain | Propagation rule | Thesis wording |
|---|---|---|---|
| Detection box | frame-object | one box belongs to one decoded frame | frame-level detection |
| Tracking identity | source-local trajectory | continuity is checked within the source video/clip; `pig_id` is annotation-local, not a cross-video biological identity | identity-bearing trajectory |
| CVAT behavior | anchor interval | anchor `k` defines `k..k+5`; raw non-anchor values are retained for audit | six-frame native interval |
| Legacy behavior | recovered burst | one retained unit contains 16 dense frames; legacy anchor offsets may be stored as provenance | sixteen-frame native burst |
| Hidden | frame-object visibility | no automatic six-/sixteen-frame propagation; any span-level use must be explicit | frame-level visibility attribute |
| Behavior review decision | native unit | CVAT apply scope is one six-frame interval; legacy apply scope is one 16-frame burst | human-reviewed behavior unit |
| Model window | derived input | created after review; may contain several native units only when all declared conditions pass | model input window |

## E. Detection, tracking, and source construction

The early detection workflow is documented in the notebooks rather than in the
current classification authority. The implemented frame-selection code uses a
pen mask, an optional background image, frame activity, down-sampled image
differences, perceptual hashing, temporal recency, and balancing across video
leaves. This is evidence for how detector data were assembled, not evidence that
the classifier received the selected frames directly.

The tracking parser reads CVAT interpolation XML, keeps non-outside boxes, and
records `track_id`, `pig_id`, bounding boxes, behavior, and Hidden attributes.
It does not discard a frame merely because fewer than eight pigs are visible;
partial context is recorded explicitly. Interaction rows are marked as requiring
partner context, whereas actor-only non-interaction rows remain valid when only
one pig is present. A provisional detector/ByteTrack output is therefore a
pre-annotation, not tracking ground truth. Corrected CVAT source records are the
spatial and identity basis consumed by later stages.
## F. Hidden visibility semantics and review conditions

This section records the evidence-backed Hidden review semantics.

## G. Legacy-source semantics

Legacy data is retained as a supplementary source within the pooled behavior
dataset. Its value is temporal and day-level diversity, not a separate label
authority. Each retained legacy actor sample is represented as a sixteen-frame
native burst, and its source provenance remains explicit.

## F (authoritative detail). Hidden review semantics

`Hidden` is a frame--object visibility attribute. It describes whether the
annotated actor is visually hidden or sufficiently occluded in one decoded
frame; it is not a behavior class, a posture target, or a native temporal
unit. A value is attached to the actor observation at its own frame index and
is not expanded automatically to all frames in a six-frame CVAT interval or a
sixteen-frame legacy burst. Any interval-level mask is an explicit later
aggregation of row-level decisions.

The active mixed-source population has two source contracts. CVAT tracking XML
contributes six-frame native intervals (`k..k+5`), whereas the recovered
legacy export contributes dense sixteen-frame bursts. Source type, video key,
actor/track key, frame index, and temporal-unit key remain attached to every
visibility observation. `pig_id` is annotation-local and is not a biological
identity shared across videos.

The review design has four disjoint, target-independent cohorts: all untrusted
`Hidden=Yes` rows; a risk-enriched `Hidden=No` cohort; a stratified-random
`Hidden=No` cohort for false-negative estimation with inclusion probabilities
and inverse-probability weights; and low-risk `Hidden=No` clean controls.
Sampling uses source type, review-stratum key, and false-negative risk band.
Behavior labels, target fields, existing review decisions, and manual outcomes
are forbidden as sampling inputs.

The risk score prioritises review candidates rather than relabeling them. Its
evidence includes box clipping, pair overlap and IoU, close-partner distance,
pair contact, adjacent hidden evidence, persistent contact or overlap, abrupt
shape or area changes, and adjacent box instability. Adjacent evidence is
valid only for consecutive frame indices within the same source, video,
actor/track, and temporal unit; sparse annotations are not treated as
contiguous. The high-risk and clean-control thresholds are 0.35 and 0.10.
Neither threshold changes a label automatically.

The Hidden GUI presents the complete frame, actor and partner boxes, an actor
crop when available, source/frame/track metadata, and risk evidence. The
reviewer chooses `Yes`, `No`, or `Unclear`. Resolved Yes/No decisions require
defensible medium or high confidence; `Unclear` remains unresolved. The GUI
writes a resumable decision table and does not modify source XML or CSV. Apply
is allowed only after coverage, key, duplicate, confidence, and row-count
audits pass, and may change Hidden plus declared provenance fields only.

Current authority records verified human Hidden coverage as `0/5,131`.
Technical manifest and media checks exist, but carried or embedded decision
payloads are forensic artifacts and cannot be cited as completed human review.
Hidden apply, temporal rebuild, and training therefore remain blocked.

## H. Behavior-review candidate inventory

The evidence supports candidate generation from native-unit and frame-derived
quality signals, not automatic behavior relabeling. Candidate families include
identity or bounding-box discontinuity, low-visibility burden, missing
interaction-partner evidence, abrupt geometric or motion changes, ROI
disagreement, short runs, neighbouring-label conflicts, and oscillatory or
otherwise inconsistent temporal transitions. A transition such as
`fight`--short interruption--`fight` is a reason to inspect context, not proof
that the interruption is wrong.

The behavior reviewer decides whether a native unit is accepted, corrected, or
technically excluded. Review notes and candidate scores are audit metadata,
not model inputs or additional behavior ground truth. Source corrections are
applied only through the declared corrected-source workflow, followed by
row-count, key, provenance, feature, and leakage checks.

## I. Temporal-consistency rules and limits of evidence

The confirmed temporal rule is conservative adjacency: a previous or next row
contributes temporal evidence only when the frame-index difference is exactly
one and the rows share source, video, actor/track, and temporal-unit keys. This
rule is used for adjacent Hidden evidence, persistent pair contact or overlap,
and bounding-box instability. It prevents sparse annotation anchors from being
mistaken for continuous video observations.

The investigation found no evidence that a temporary behavior between two
`fight` units is automatically relabeled, that ROI disagreement alone changes a
behavior target, or that Hidden is propagated across a native unit without an
explicit aggregation policy. Such patterns may generate review candidates,
but the final label remains a human decision or the original source value when
no review is selected.

## J. Corrections to the working description

| Working statement | Evidence-backed correction |
|---|---|
| Hidden is part of the behavior label. | Hidden is an independent frame/object visibility attribute. |
| A Hidden value describes a whole six- or sixteen-frame unit. | It describes its own frame/object row; span masks are explicit derived metadata. |
| Existing legacy Hidden values prove current review. | Legacy values preserve provenance but still require current two-sided review. |
| Risk scores identify abnormal behavior. | Risk scores prioritise visibility-quality review and are target-independent. |
| Candidate scores create new labels. | Human decisions accept, correct, or technically exclude behavior units. |
| A historical PASS artifact proves completed Hidden review. | Current authority records carried payloads as unverified and coverage as zero. |
| The classifier receives review notes or Hidden risk fields. | They remain audit/mask metadata; only declared model features enter model input. |

## K. Current blockers and unresolved boundaries

The methodology can describe the review protocol and its safeguards, but it
must not claim a completed Hidden-reviewed dataset. The active blockers are the
absence of verified current Hidden decisions, the blocked scientific coverage
gate, and the prohibition on Hidden apply, temporal rebuild, train-ready
snapshot, and model training. Behavior-review closure and corrected-source
evidence remain subject to current authority rather than stale counts in older
drafts.

Two boundaries remain explicit. Source-local track continuity is not a claim of
permanent biological identity across videos. Behavioral deviation screening is
a downstream statistical screening layer without anomaly ground truth; neither
Hidden review nor behavior review constitutes a veterinary diagnosis.

## L. Documentation corrections before English conversion

- Describe Hidden as an independent frame/object quality variable before
  temporal harmonization.
- State that CVAT and legacy sources are pooled only after source-specific
  temporal contracts and provenance are retained.
- Explain the four target-independent Hidden cohorts and separate random
  false-negative estimation from high-risk correction yield.
- State that risk scores select review candidates and never change labels
  automatically.
- Use `0/5,131` verified Hidden coverage as the current status; do not cite
  historical PASS or embedded payload counts as completed review.
- Refer to Figure 4 in the surrounding prose and show separate visibility and
  behavior-review paths; review metadata must remain outside model input.

## M. Thesis-ready corrected description (Vietnamese)

Quy trình gán nhãn và kiểm soát chất lượng được tổ chức theo nhiều mức độ
khác nhau. Các hộp phát hiện thuộc về từng frame, còn định danh theo dõi được
duy trì trong phạm vi video hoặc clip nguồn. Nhãn hành vi của CVAT được biểu
diễn bằng khoảng sáu frame bắt đầu tại anchor `k`, trong khi dữ liệu legacy
được giữ dưới dạng burst mười sáu frame; hai nguồn được hợp nhất sau khi bảo
tồn provenance và quy ước thời gian riêng.

`Hidden` được xử lý như một thuộc tính nhìn thấy của actor ở cấp frame--object,
độc lập với nhãn hành vi. Hệ thống lập các cohort kiểm tra không phụ thuộc vào
target, gồm các trường hợp `Hidden=Yes` chưa tin cậy, nhóm `Hidden=No` có rủi ro
cao, mẫu `Hidden=No` ngẫu nhiên phân tầng và nhóm đối chứng rủi ro thấp. Điểm
rủi ro chỉ dùng để ưu tiên frame cần xem xét; nó không tự động sửa nhãn.
Người đánh giá chọn `Yes`, `No` hoặc `Unclear` trên toàn cảnh có hộp của actor
và các cá thể liên quan. Chỉ quyết định đã được giải quyết, kiểm tra đầy đủ và
áp dụng qua quy trình được phê duyệt mới tạo ra metadata Hidden tin cậy.

Sau bước kiểm tra visibility, các đơn vị hành vi được chọn để human review
dựa trên những dấu hiệu chất lượng như gián đoạn identity, thiếu bằng chứng
đối tác, mâu thuẫn giữa hành vi và vùng chức năng, thay đổi hình học hoặc
chuyển động đột ngột và chuyển trạng thái bất thường. Những dấu hiệu này chỉ
tạo candidate để kiểm tra ngữ cảnh; người đánh giá mới quyết định giữ nguyên,
hiệu chỉnh hoặc loại trừ kỹ thuật. Review vì vậy là cơ chế kiểm soát chất
lượng annotation, không tạo ground truth cho bệnh, stress hay anomaly.

Tại thời điểm audit, manifest và kiểm tra media kỹ thuật đã có, nhưng chưa có
quyết định Hidden hiện hành được xác nhận bởi người dùng; coverage là
`0/5,131`. Phần phương pháp chỉ nên mô tả protocol và điều kiện bảo đảm chất
lượng, chưa trình bày đây là bằng chứng của một tập dữ liệu đã review hoàn tất.

## N. Evidence appendix

| Scope | Current evidence | Status for thesis |
|---|---|---|
| Hidden policy and cohorts | `.agents/memory/09_HIDDEN_REVIEW.md`; runbook | current protocol |
| Current project state | `docs/CLASSIFICATION_V2_CURRENT_STATE.md` | current authority; blocked |
| GUI decisions | `scripts/classification_v2/01_review_units_gui/review_hidden_quality_gui.py` | implemented writer |
| Behavior-review guidance | `docs/CLASSIFICATION_V2_POST_REVIEW_LEARNING_PIPELINE.md`; GUI guide | current support |
| Detection-data history | `notebooks/01_data_preparation/` | historical lineage |
| Technical Hidden reference | `outputs/classification_v2/rebuilds/hidden_review_v6_full_20260714` | manifest/media only; decisions unverified |

This audit is a documentation artifact. It did not open a review GUI, create a
decision, modify source annotations, rebuild model inputs, or run training.

The decision schema also enforces semantic compatibility. A resolved Yes or No
must carry medium or high confidence; `Unclear` must remain low-confidence.
`Yes` cannot be paired with a clearly-visible reason, `No` cannot be justified
only by an occlusion reason, and a resolved Yes/No cannot carry an ambiguous
reason. These checks reject inconsistent payloads; they do not infer a human
decision or promote an unselected source value to trusted metadata.

The scientific gate is separate from candidate ranking. It requires 95%
confidence with 2,000 bootstrap iterations, a random-cohort false-negative
upper bound no greater than 0.05, and a high-risk correction-yield upper bound
no greater than 0.10. Each estimate requires at least 100 reviewed items, 50
native clusters, and five recording clusters. The high-risk yield is reported
as review yield, not as population prevalence.

After a trusted row-level review, the default window policy treats Hidden as a
conservative visual-burden mask. The main limits are a total Hidden ratio of
0.25 and a longest-run ratio of 0.20; robust-only limits are 0.50 and 0.40.
Exceeding a robust limit excludes the window from training and sets its sample
weight to zero. These ratios are audit and mask metadata, never model-X
features, and the no-exclusion option is an explicit ablation rather than the
canonical lineage.

## O. Full-pipeline reconstruction (authoritative correction)

This section answers the complete investigation brief. It separates source
truth, derived evidence, review decisions, and model inputs. A stage marked
`IMPLEMENTED_AND_ACTIVE` means that code or a current contract exists; it does
not mean that the scientific gate has passed or that the resulting artifact is
train-ready.

### O.1 Actual dependency order

The evidence supports the following order. Detection-data selection is an
earlier, historical data-construction pathway. The classification branch then
uses corrected or recovered frame-object sources, preserves source provenance,
and creates native review units before deriving model windows.

```text
RGB source video / recovered legacy crops
        |
        +--> detection-frame selection --> manual detection boxes
        |                                --> detector training
        |                                --> provisional detector/tracker output
        |
        +--> CVAT tracking XML and legacy recovered CSV sources
                     |
                     +--> source-local boxes, track IDs, behavior, Hidden
                     +--> geometry, motion, ROI, social and timestamp evidence
                                      |
                                      +--> target-independent Hidden cohorts
                                      |       --> human Yes / No / Unclear
                                      |       --> trusted visibility mask audit
                                      |
                                      +--> six-frame CVAT or sixteen-frame legacy
                                              native behavior units
                                              --> behavior review candidates
                                              --> human accept / corrected / exclude
                                                      |
                                                      +--> corrected-source audit
                                                              --> leakage checks
                                                                      --> model windows
                                                                              --> X/y/mask/weight export
```

The diagram is a lineage description, not a claim that the final export is
currently available. Current authority blocks the Hidden apply, temporal
rebuild, and training stages because verified current Hidden decisions are
absent.

### O.2 Stage-by-stage authority and training boundary

| Stage | Granularity | Main evidence/output | Status | Changes source truth? | Safe for training now? |
|---|---|---|---|---|---|
| Source-time packaging | video/frame | timestamps, source FPS, 1,800-frame clips | established fact | no | only as provenance |
| Detection frame selection | frame/image | mask, activity, hash, gap and balance fields | historical implemented | no | detector branch only |
| Detection annotation | frame-object | manual Roboflow/CVAT boxes and QC | inherited historical | yes, within detection dataset | detector branch after audit |
| Detector training | image/frame-object | trained detector and prediction artifacts | implemented pathway; result pending | no | not a behavior target |
| Model-assisted tracking | frame-object/trajectory | provisional boxes and local IDs | historical/provisional | no | no |
| CVAT tracking correction | frame-object/trajectory | source-local boxes, IDs, behavior, Hidden | implemented parser/source contract | only through explicit apply | no until GT audit |
| Tracking evaluation authority | trajectory/episode | included videos, GT, metrics and hashes | registered authority; result pending | no | no |
| Source merge | frame-object | canonical mixed-source table and lineage audit | implemented and gated | creates derived table | no before review gates |
| Spatial-temporal evidence | frame-object/native unit | geometry, motion, ROI, social and timestamps | implemented | no | review metadata only |
| Hidden selection/review | frame-object | four target-independent cohorts and decisions | protocol implemented; coverage 0/5,131 | decisions may change Hidden only | no |
| Native behavior review | native unit | accept, corrected or technical exclusion | selector/contract implemented; closure blocked | only explicit decision apply | no |
| Corrected-source rebuild | frame/native unit | overlay, provenance and fixed-point audit | configured but blocked | yes, only via frozen authority | no |
| Temporal windows | model window | 6/8/12/16 and sampled-six manifests | implemented downstream | derived only | no current snapshot |
| Leakage checks | group/window | grouped split, duplicate and overlap audits | contract implemented; final artifact pending | no | no |
| Train-ready snapshot | X/y/mask/weight | run-bound mixed-reviewed export | blocked by current authority | no | no |

### O.3 Detection dataset construction and detector boundary

The notebooks establish a selection workflow rather than arbitrary uniform
sampling. Phase 1 scans source leaves using an empty-pen or background reference,
a valid-pen mask, frame activity and source timestamps. It ranks candidates in
one-second source-time windows, removes near duplicates using a 64-bit average
hash and Hamming thresholds, and applies leaf/video/day gap and quota rules.
Fallback and emergency-fill records must retain their provenance because they
do not necessarily have the same activity distribution as the primary pass.

Phase 2 constructs behavior-oriented burst candidates from motion, foreground,
ROI and trigger fields. Its configured burst length is six frames with offsets
`[-3,-2,-1,0,1,2]`; it also applies per-minute and minimum-gap controls. These
settings describe historical candidate generation, not the final behavior label
contract by themselves.

The annotation notebook contains a historical YOLO-assisted scaffold. The
recorded settings include confidence `0.25`, IoU `0.7`, approximately 1-Hz
sampling, a first-40-video-directory limit, a scene mask, IoU plus HSV
association, `KEEPALIVE_MAX=30`, `MIN_HITS_FOR_ID=5`, and a maximum of eight
IDs per video. Its COCO export and default `Behavior="lying"` are scaffolding
semantics; they are not behavior ground truth. Manual Roboflow bounding-box
annotation and QC remain the detector-data authority to be bound to a final
manifest. No final image/box counts or detector metrics may be copied from
notebook comments.

Depth may exist in the acquisition directory, but the current behavior-model
contract is RGB-derived. The detection notebooks explicitly identify depth
videos and exclude them from the color candidate list. Depth therefore remains
context or future work unless a registered ablation proves its contribution.

### O.4 Tracking, CVAT correction and identity semantics

The detector produces observations; the tracker associates observations into
trajectories. A provisional ByteTrack or hybrid output is therefore
model-assisted pre-annotation, not automatically a tracking ground truth. The
CVAT XML parser keeps non-outside boxes and records `track_id`, `pig_id`, box,
behavior and Hidden fields. It does not drop a frame merely because fewer than
eight pigs are visible; partial visibility is represented as context.

The current identity-adjudication tool is intentionally sidecar-based. It can
record a reviewer-selected actor, a corrected or added box, and frame-scoped
identity attributes. Applying those decisions to an explicitly supplied CSV or
XML source is a separate validated operation. Consequently, the thesis may say
that corrected CVAT/source-local trajectories provide the identity basis, but it
must not imply that every provisional tracker output is already the frozen GT.

`pig_id` and `track_id` are local to the source video/clip contract. They are not
biological identities guaranteed across all six weeks or across unrelated
videos. Missing pigs, re-entry, occlusion and identity switches are separate
tracking phenomena and must be evaluated with tracking metrics, not inferred
from detector precision or behavior labels.

### O.5 Behavior sources, harmonization and label granularity

The current ten-class behavior ontology is retained for both sources. A CVAT
tracking XML row carries an anchor interval whose native contract is six frames,
`k..k+5`. A recovered legacy source carries a dense sixteen-frame native burst.
The two sources are pooled into the mixed classification dataset; legacy is an
additional source of day/video temporal diversity, not a held-out dataset that
is excluded from training. Source type, video key, actor key, frame index,
native-unit key and timestamp remain attached during the merge.

The canonical merge stacks normalized frame-object rows and audits required
columns, source types, behavior vocabulary, bounding-box validity, duplicate
frame-object keys and source-frame clock consistency. The mixed-source lineage
contract additionally binds the expected legacy export, the twelve named CVAT
XML files, file sizes/SHA-256 values and the merged-output hash. A successful
lineage audit is a provenance gate; it is not a human review decision.

### O.6 Independent Hidden visibility review

`Hidden` is a frame-object visibility attribute for one actor observation. It is
not a behavior class, posture target, or automatic six-/sixteen-frame label.
The review builder validates exact source, video, actor, frame and native-unit
keys and rejects malformed native spans. Sampling is target-independent and
uses four disjoint cohorts: untrusted `Hidden=Yes` census, high-risk
`Hidden=No` enrichment, stratified-random `Hidden=No` audit, and low-risk
`Hidden=No` controls. Behavior labels and existing review outcomes are forbidden
sampling inputs.

The reviewer records `Yes`, `No` or `Unclear` with confidence and reason. The
contract rejects incompatible combinations such as resolved `Yes` with
`clearly_visible`, resolved `No` justified only by occlusion, or resolved
decisions carrying an ambiguous reason. Review decisions are written to a
resumable ledger and do not edit source XML/CSV during review. Current verified
coverage is `0/5,131`; carried or embedded historical payloads are forensic
artifacts, not evidence of completed current review. The 95%/2,000-bootstrap
scientific gate, random false-negative bound and high-risk-yield bound therefore
remain open.

### O.7 Behavior-review construction and decision semantics

The review-unit builder creates a native review unit, not a model window. It
maps `legacy_recovered` to `legacy_burst_16` and `cvat_tracking_xml` to
`cvat_interval_6`. A window manifest may be joined only to record affected
windows; it does not redefine annotation truth.

Behavior review candidates are selected from evidence such as identity or box
discontinuity, visibility burden, missing partner context, ROI disagreement,
motion contradiction, posture/shape transition, temporal label conflict and
rare-class coverage. Candidate scores and notes are audit metadata. The human
decision is `accept`, `corrected`, or `exclude`, with explicit correction and
apply scope. No selector predicate automatically changes a label, sample weight
or training action. Corrected-source application is valid only after row-count,
key, source-label, provenance, feature-rebuild and fixed-point audits pass.
The option to include every retained legacy native unit is an explicit review
configuration; it must not be inferred from the fact that legacy rows are
present in the merged training population.

### O.8 Model windows, features and leakage boundary

After a valid reviewed-source stage, downstream builders can construct windows
of length 6, 8, 12 or 16 frames, including the sampled-six configuration at
offsets `0,3,6,9,12,15`. The model receives declared RGB-derived actor imagery,
box geometry, motion, ROI, social and temporal features. Review reason, risk,
Hidden sampling, reviewer notes and other `review_` fields are excluded from
model X by the feature whitelist/blacklist contract. Window masks and sample
weights may use trusted visibility burden only after the Hidden gate; exceeding
the robust Hidden limits can exclude a window, but it does not create a new
behavior label.

Grouped split and leakage checks are intended to cover recording/day/session,
leaf/video overlap, neighboring intervals, duplicate or near-duplicate images,
native-unit overlap and ordered window IDs. The exporter requires the generated
`mixed-reviewed` contract and a declared feature whitelist. Because the current
reviewed-source and Hidden authorities are not closed, no current train-ready
snapshot can be cited as final thesis evidence.

### O.9 Complete review-candidate rule inventory

The primary behavior selector exposes thirteen predicate families. The table
below records the implemented reason families and their scientific effect.

| ID | Family and trigger | Granularity | Status | Effect |
|---|---|---|---|---|
| R01 | ROI label lacks persistent feeder/drinker/toy support | native unit | active | candidate and priority only |
| R02 | another ROI has stronger support than the labeled ROI | native unit | active | candidate and priority only |
| R03 | `explore` has persistent stationary ROI contact | native unit | active | possible-false-negative candidate |
| R04 | `move` has weak motion evidence | native unit | active | candidate only |
| R05 | `stand` has strong motion evidence | native unit | active | candidate only |
| R06 | `explore` has move-like motion | native unit | active | candidate only |
| R07 | posture label occurs during a strong shape transition | native unit | active | candidate only |
| R08 | posture label has strong pixel motion | native unit | active | candidate only |
| R09 | `fight` lacks persistent contact/aggression evidence | native unit | active | candidate only |
| R10 | `social-nose` lacks persistent partner contact | native unit | active | candidate only |
| R11 | `social-nose` has fight-like motion | native unit | active | candidate only |
| R12 | interaction requires partner context that is unavailable | unit/pair | active policy | blocks unsafe interpretation; no relabel |
| R13 | interval has high Hidden burden | native unit | active | review priority; later mask/exclusion only after trusted review |

The selector also records global-mandatory, evidence-insufficiency,
temporal-contradiction, media/actor-authority, rare-class census, risk-triggered
and stratified-low-risk-audit predicates. These are selection design fields,
not additional automatic labels. `playwithtoy` is the configured rare-class
census behavior. Evidence availability alone is explicitly forbidden from
selecting a candidate.

The supporting evidence layer contains active bounding-box and identity checks
such as adjacent box instability, shape/area change, pair overlap/IoU,
duplicate frame-object keys, missing source keys and source-local identity
adjudication cases. These checks generate evidence or an identity-review case;
they do not silently rewrite behavior. Source and sampling audits are therefore
classified as `IMPLEMENTED_AND_ACTIVE` but `AUDIT_ONLY` or `CANDIDATE_ONLY`.

No implemented rule found in the inspected paths automatically changes a
behavior label. Training action or sample weight changes are permitted only in
an explicit human decision or a trusted post-review visibility mask, both with
their own audit contracts. Rules that are configured but not executed in a
current run, and any historical notebook heuristic without a bound manifest,
must be described as `CONFIGURED_BUT_NOT_EXECUTED` or `HISTORICAL`, not as final
ground truth.

### O.10 Temporal-inconsistency inventory

The current temporal audit groups by `source_type`, `dataset_id`, `video_key`
and `object_track_key`, orders native units by frame boundaries, and treats
only contiguous native units as neighbors. A gap is not silently treated as
continuous evidence.

| Rule | Exact trigger | Status | Uses future context? | Output |
|---|---|---|---|---|
| T01 | one non-`fight` unit between two `fight` units | active, HIGH | yes, offline audit | `NON_FIGHT_BURST_BETWEEN_FIGHT` candidate |
| T02 | one `fight` unit between two `social-nose` units | active, MEDIUM | yes, offline audit | `FIGHT_BURST_BETWEEN_SOCIAL_NOSE` candidate |
| T03 | isolated interaction island between matching interaction runs | active, MEDIUM | yes, offline audit | `INTERACTION_SINGLE_BURST_ISLAND` candidate |
| T04 | isolated single-label island in a contiguous run | active, LOW | yes, offline audit | `GENERAL_SINGLE_BURST_LABEL_ISLAND` candidate |
| T05 | post-review unreviewed gap bounded by one effective label | active residual scope | yes, offline only | `POST_REVIEW_SHORT_LABEL_GAP` candidate |
| T06 | post-review gap of up to two units bounded by `fight` | active residual scope | yes, offline only | `POST_REVIEW_NON_FIGHT_GAP_BETWEEN_FIGHT` HIGH candidate |

Thus the exact `fight -> short interruption -> fight` idea does exist, but in a
narrow form: the base audit detects one middle native unit, and the residual
discovery contract permits a maximum gap of two units only for review-informed
post-review residuals. There is no generic rapid-oscillation detector and no
rule that absorbs the interruption into `fight`. The findings are hypotheses
for context review; they do not propose a relabel automatically. Interaction
pair audits additionally report incomplete partner context, fight/social pair
conflict, non-fight group partners and social actors paired with fight partners.
Missing Hidden rows, identity switches and bounding-box gaps are not treated as
proof that the behavior transition is wrong.

### O.11 User-understanding discrepancy table

| User assumption assessed | Evidence-based verdict | Required thesis correction |
|---|---|---|
| Detection frames were selected rather than arbitrary uniform samples | CORRECT | Describe source-time candidate selection and manifests. |
| Background, mask, activity, hash and temporal balancing are used | MOSTLY_CORRECT | Attribute these to historical notebooks and bind final settings to manifests. |
| Roboflow manual boxes are the detector annotation authority | MOSTLY_CORRECT | State manual/QC authority, but do not invent final counts without the export audit. |
| Detector plus ByteTrack produced provisional CVAT tracks | MOSTLY_CORRECT | Call them model-assisted pre-annotations and keep tracker profiles explicit. |
| CVAT correction forms the tracking ground truth | PARTIALLY_CORRECT | Separate corrected source/local trajectories, sidecar adjudication and frozen GT authority. |
| Corrected trajectories support behavior and visibility annotation | MOSTLY_CORRECT | State the downstream identity basis while preserving separate review contracts. |
| Behavior is assigned every six frames | MOSTLY_CORRECT | This is the CVAT native contract; legacy units are sixteen-frame bursts. |
| Hidden follows the same six-frame interval | INCORRECT | Hidden is frame-object-level and is not silently propagated. |
| Six legacy keyframes themselves are the sixteen-frame label authority | PARTIALLY_CORRECT | The native legacy unit is sixteen frames; exact keyframe semantics need a bound manifest. |
| Both sources are harmonized while preserving provenance | MOSTLY_CORRECT | Canonical merge and lineage audit do this, but review order and final snapshot remain gated. |
| Spatio-temporal rules generate suspicious cases for human review | CORRECT | Present rules as candidate generators, not label truth. |
| The fight/interruption and related rules are implemented | MOSTLY_CORRECT | Only the conservative contiguous-unit rules listed in O.10 are evidenced. |
| Final windows are created only after review | INTENDED_BUT_NOT_IMPLEMENTED | This is the declared safe order, but current Hidden/review gates block final export. |
| Recording-day grouping proves detector leakage is absent | NOT_VERIFIABLE | Require the final split and duplicate/overlap audit before claiming it. |

The table contains fourteen assumptions: two `CORRECT`, seven
`MOSTLY_CORRECT`, two `PARTIALLY_CORRECT`, one `INCORRECT`, one
`INTENDED_BUT_NOT_IMPLEMENTED`, and one `NOT_VERIFIABLE`.

### O.12 Blockers, ambiguities and code-defect assessment

Current blockers are scientific-authority blockers, not evidence that the
pipeline stages are absent: verified Hidden human coverage is zero; the Hidden
coverage and yield gates are open; Hidden apply and temporal rebuild are
blocked; behavior review still has targeted and control closure work; the
corrected-source fixed-point authority is not closed; and no final mixed
train-ready snapshot is authorized. Old completed-looking counts or PASS
payloads must not be promoted over current authority.

Important ambiguities that remain for thesis results are the final Roboflow
manifest, detector configuration/metrics, final tracking evaluation population,
the reconciled review-close/fixed-point authority, merged source composition,
posture authority, and final profile/screening thresholds.

The read-only inspection found **no confirmed code defect** against the current
frozen authority. It found documentation risks: the previous report scope was
too Hidden-centric; historical YOLO scaffold semantics could be mistaken for
behavior truth; and review candidates could be described as labels. These are
narrative corrections, not production patches. No source XML/CSV, ledger,
manifest, model weight, GUI decision or training artifact was modified.

### O.13 Thesis-ready corrected narrative (Vietnamese)

Quy trình dữ liệu của nghiên cứu bắt đầu bằng việc chọn các frame theo thời
gian nguồn thay vì lấy mẫu đồng đều một cách độc lập. Quy trình lịch sử sử dụng
ảnh nền và mask vùng chuồng hợp lệ, độ thay đổi giữa frame và nền, lọc gần trùng
nhau bằng average hash, cùng các giới hạn khoảng cách và cân bằng theo video,
ngày ghi hình và leaf. Các frame được chọn sau đó được gán bounding box thủ
công và kiểm tra chất lượng để tạo dữ liệu detector. Một nhánh YOLO-assisted
trước đây được dùng để tạo scaffold và track tạm thời; các kết quả đó không được
xem là ground truth hành vi hoặc identity cuối cùng.

Các nguồn classification hiện hành gồm CVAT tracking XML và dữ liệu legacy
recovered. CVAT giữ một đơn vị hành vi sáu frame bắt đầu tại anchor `k`, còn
legacy giữ burst liên tục mười sáu frame. Hai nguồn được hợp nhất trong schema
chung để bổ sung độ đa dạng theo ngày và video, đồng thời giữ source type,
video, actor, frame, native-unit key và timestamp. `pig_id` và `track_id` chỉ
được diễn giải trong phạm vi source video/clip; chúng không chứng minh identity
sinh học xuyên suốt sáu tuần.

`Hidden` được xử lý riêng như thuộc tính visibility ở cấp frame--object. Các
cohort review của Hidden được chọn độc lập với behavior target và người đánh giá
ghi nhận `Yes`, `No` hoặc `Unclear`. Sau khi tạo evidence không gian--thời gian,
các native behavior unit được đưa vào review nếu có mâu thuẫn ROI, motion,
posture, partner, identity, bounding box hoặc continuity. Những quy tắc này chỉ
tạo candidate; reviewer mới quyết định giữ nguyên, hiệu chỉnh hoặc loại trừ
theo apply scope được ghi lại.

Chỉ sau khi Hidden review, behavior review, corrected-source application,
row/key/provenance audits và grouped leakage checks đạt authority, các cửa sổ
temporal mới được tạo cho mô hình. Hiện tại verified Hidden coverage là `0/5,131`
và các gate tương ứng chưa đóng, nên luận văn chỉ được mô tả protocol và ranh
giới claim, chưa được trình bày một train-ready snapshot cuối cùng. Review
metadata và risk fields là audit information; chúng không đi vào model input.

### O.14 Evidence appendix for the full reconstruction

| Evidence scope | Primary path | Use in thesis |
|---|---|---|
| Current status and blockers | `docs/CLASSIFICATION_V2_CURRENT_STATE.md` | current authority |
| Hidden contract | `.agents/memory/09_HIDDEN_REVIEW.md` | visibility semantics and gates |
| Detection selection | `notebooks/01_data_preparation/video_to_frame_phase_1.ipynb` | historical selection protocol |
| Behavior candidates | `notebooks/01_data_preparation/video_to_frame_phase_2.ipynb` | historical burst-candidate pathway |
| Detection scaffold | `notebooks/01_data_preparation/video_to_frame_annotate.ipynb` | historical YOLO boundary |
| Canonical merge | `src/pig_behavior/classification_v2/merge_sources.py` | schema and row/key audits |
| Mixed lineage | `src/pig_behavior/classification_v2/contracts/merged_source_lineage.py` | file/hash binding |
| CVAT source parser | `src/pig_behavior/classification_v2/sources/cvat_tracking_xml.py` | tracking/local identity semantics |
| Legacy parser | `src/pig_behavior/classification_v2/sources/legacy_recovered_csv.py` | dense burst semantics |
| Review-unit builder | `src/pig_behavior/classification_v2/review/review_unit_builder.py` | native unit boundary |
| Behavior evidence/selection | `src/pig_behavior/classification_v2/review/behavior_evidence.py`; `behavior_review_selection.py` | candidate rules |
| Temporal audit | `src/pig_behavior/classification_v2/review/behavior_consistency_audit.py` | continuity findings |
| Residual discovery | `src/pig_behavior/classification_v2/review/post_review_residual_discovery.py` | bounded post-review gaps |
| Corrected rebuild | `src/pig_behavior/classification_v2/review/reviewed_rebuild.py` | explicit apply/fixed-point contract |
| Train-ready exporter | `scripts/classification_v2/02_train_ready_exports/classification_v2_export_train_ready_windows.py` | X/y/mask/weight boundary |
| Tracking authority | `docs/tracking/reconciliation/FOUR_METHOD_TRACKING_FREEZE_AUTHORITY_20260729.json` | method freeze and evaluation boundary |
| Updated thesis structure | `docs/thesis_drafts/PIG_BEHAVIOR_THESIS_MASTER_OUTLINE_V2.md` | Sections 2.3--2.10 and visual plan |
