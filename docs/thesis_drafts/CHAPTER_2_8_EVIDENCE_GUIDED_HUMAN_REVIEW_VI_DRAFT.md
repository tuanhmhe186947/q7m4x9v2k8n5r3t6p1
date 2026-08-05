# Section 2.8 — Evidence-Guided Human Review

**Draft language:** Vietnamese and English  
**Draft status:** Provisional methodology; review decisions and rebuilt data
remain governed by the registered review authority, while model metrics and
paper-grade comparisons belong to Chapter 3.  
**Approved thesis title:** *An AI Spatio-Temporal Framework for Dynamic
Behavior Profiling and Anomaly Detection in Group-Housed Pigs*

## Vietnamese thesis draft

### Vai trò và ranh giới của human review

Human review được sử dụng như một lớp kiểm soát chất lượng của annotation,
không phải như một bộ phân loại tự động hay một nhánh chẩn đoán bất thường.
Hệ thống trước hết tạo các candidate cần xem lại từ bằng chứng đã có trong
native behavior unit và các quan sát frame–object liên quan. Candidate chỉ
định vị nơi bằng chứng có thể mâu thuẫn, thiếu hoặc không đủ liên tục; nó không
tự thay đổi nhãn hành vi, trọng số huấn luyện hay kết luận về trạng thái bất
thường của con vật.

Review được thực hiện trong phạm vi review unit của nguồn annotation. Phạm vi
này được giữ riêng với model window được tạo ở bước sau. Vì review là một
quy trình hồi cứu, người review có thể sử dụng ngữ cảnh trước và sau unit
trong phạm vi đã khai báo để kiểm tra tính liên tục; ngữ cảnh tương lai chỉ
phục vụ adjudication và không được đưa vào đặc trưng model-facing.

### Sinh candidate từ các nhóm bằng chứng

Candidate generation kết hợp các nhóm bằng chứng bổ sung. Bằng chứng ROI và
nguồn lực kiểm tra mức tiếp xúc hoặc hiện diện bền vững của actor tại feeder,
drinker hoặc toy, đồng thời phát hiện trường hợp một vùng khác có hỗ trợ mạnh
hơn. Bằng chứng động học và hình dạng xem xét chuyển động, trạng thái đứng yên,
biến thiên hộp và chuyển tiếp hình dạng để nhận diện xung đột với các nhãn
`move`, `explore`, `stand`, `lying` hoặc `sitting`. Bằng chứng tương tác dùng
tiếp xúc, độ bền của partner, khoảng cách và chỉ báo chuyển động xã hội để
kiểm tra `fight` và `social-nose` trong ngữ cảnh của các actor liên quan.

Bằng chứng visibility, tính hợp lệ của bounding box và tính đầy đủ của
interval được dùng để phát hiện trường hợp không đủ cơ sở quan sát hoặc không
đủ authority của actor. `Hidden` vẫn là thuộc tính visibility ở cấp
frame–object; nó không trở thành một lớp hành vi và không được dùng như một
nhãn thay thế cho quyết định behavior review. Cuối cùng, temporal evidence
kiểm tra các đơn vị có cùng source, video, actor/track và khóa thời gian, chỉ
coi những lân cận liên tục đã được khai báo là bằng chứng liên tiếp. Khoảng
trống hoặc anchor thưa không được tự động xem như một đoạn chuyển tiếp liên
tục.

Các predicate cụ thể được tổ chức thành những cohort kiểm tra bắt buộc,
nhóm rủi ro cao và một mẫu audit phân tầng cho phần còn lại. Quy tắc
`fight`–đoạn gián đoạn ngắn–`fight` chỉ tạo một candidate cần xem lại trong
ngữ cảnh; nó không chứng minh rằng đoạn gián đoạn sai và không tự động gộp
đoạn đó vào `fight`. Candidate score, reason và sampling metadata chỉ là
thông tin audit.

### Quyết định của reviewer và bảo toàn lineage

Sau khi kiểm tra bằng chứng hình ảnh, hình học, động học, interaction và
temporal context, reviewer có thể giữ nguyên nhãn nguồn, sửa nhãn sang một
behavior hợp lệ hoặc loại unit khỏi việc sử dụng kỹ thuật. Mỗi quyết định
được gắn với review unit, nhãn ban đầu, nhãn sau review, phạm vi áp dụng và
provenance của reviewer. Khi không có quyết định hợp lệ, giá trị nguồn không
được trình bày như một quyết định human-review đã hoàn tất.

Một correction không ghi đè lịch sử annotation. Giá trị trước review vẫn được
giữ để đối chiếu, còn giá trị sau review trở thành authority cho phạm vi đã
được quyết định. Những unit bị loại hoặc chưa đủ điều kiện không được đưa vào
tập huấn luyện chính theo chính sách review tương ứng. Review metadata, risk
reason và candidate score được giữ cho audit nhưng bị loại khỏi model input
X.

### Từ review unit đến dữ liệu model

Sau khi quyết định được chấp nhận và audit coverage, key, duplicate và scope
đạt yêu cầu, pipeline cập nhật corrected source rồi tái tạo các biểu diễn phụ
thuộc vào nó. Phạm vi rebuild bao gồm các đặc trưng hình học và motion, quan
hệ ROI và social, native temporal units và các model windows bị ảnh hưởng.
Các view cũ trong phạm vi correction không được tái sử dụng như thể chúng vẫn
còn hợp lệ. Split và leakage audit chỉ được thực hiện trên biểu diễn đã được
rebuild; một review unit không tự tạo ra ranh giới split hay thay thế cho
model-window policy.

Quy trình này giữ rõ ba vai trò: candidate generation để tìm nơi cần xem,
human decision để xác lập nhãn và phạm vi áp dụng, và corrected-source rebuild
để đưa quyết định đã audit vào dữ liệu học. Vì vậy, human review cải thiện
tính truy nguyên và độ tin cậy của annotation mà không biến thành một nhánh
dự đoán hành vi hoặc một tuyên bố chẩn đoán.

## English academic thesis draft

### Role and boundary of human review

Human review was used as an annotation-quality control layer rather than as an
automatic classifier or an abnormality-diagnosis branch. The pipeline first
generated review candidates from evidence already available in each native
behavior unit and its associated frame–object observations. A candidate marked
where the evidence might be contradictory, incomplete or temporally
insufficient; it did not relabel the behavior, alter a training weight or infer
an abnormal state.

Review was defined on the annotation source's review unit and kept distinct
from the model window constructed downstream. Because the procedure was
retrospective, the reviewer could inspect preceding and following context
within the declared review scope. Future observations were permitted for this
adjudication step only and were excluded from model-facing features.

### Candidate generation from complementary evidence

Candidate generation combined several complementary evidence families. ROI and
resource evidence measured persistent actor contact or presence at the feeder,
drinker or toy and also exposed cases in which another region had stronger
support. Kinematic and shape evidence examined motion, stationary behaviour,
box variation and shape transitions for conflicts with `move`, `explore`,
`stand`, `lying` or `sitting`. Interaction evidence used contact, partner
persistence, proximity and a social-motion indicator to inspect `fight` and
`social-nose` in the context of the associated actors.

Visibility evidence, bounding-box validity and interval completeness identified
cases in which the available observation or actor authority was insufficient.
`Hidden` remained a frame–object visibility attribute; it was not converted
into a behavior class and did not replace a behavior-review decision. Temporal
evidence was restricted to observations sharing the declared source, video,
actor/track and temporal keys. Only declared contiguous neighbours contributed
continuity evidence; sparse anchors or gaps were not silently treated as a
continuous transition.

The predicates were organized into mandatory inspection, high-risk review and
a stratified audit of the residual population. The `fight`–short interruption–
`fight` pattern generated a context-review candidate only. It did not establish
that the interruption was erroneous and did not automatically absorb the
interruption into `fight`. Candidate scores, reasons and sampling metadata were
retained as audit information rather than model inputs.

### Reviewer decisions and lineage preservation

After inspecting the visual, geometric, kinematic, interaction and temporal
evidence, the reviewer could retain the source label, correct it to a valid
behavior class or technically exclude the unit. Each decision was bound to the
review unit, the original label, the post-review label, the apply scope and the
review provenance. When no valid decision had been recorded, the source value
was not presented as a completed human-review decision.

A correction did not erase the annotation history. The pre-review value was
retained for comparison, while the post-review value became authoritative only
within the declared correction scope. Excluded or unresolved units were kept
outside the main training population according to the review policy. Review
metadata, risk reasons and candidate scores remained available for audit but
were excluded from model input $X$.

### From review units to model data

After decision coverage, key, duplicate and scope audits passed, the pipeline
updated the corrected source and rebuilt representations that depended on it.
This rebuild covered affected geometry and motion features, ROI and social
relations, native temporal units and downstream model windows. Stale views from
the corrected scope were not reused as if they remained valid. Split assignment
and leakage auditing were applied to the rebuilt representations; a review unit
did not create an independent split boundary or replace the model-window
policy.

This procedure separates three scientific roles: candidate generation locates
observations requiring inspection, human adjudication establishes the label and
its scope, and corrected-source rebuilding transfers the audited decision into
the learning data. Human review therefore strengthens annotation traceability
and reliability without becoming a behavior-prediction branch or a clinical
diagnostic claim.

## Evidence anchors

- `docs/thesis_drafts/PIG_BEHAVIOR_THESIS_MASTER_OUTLINE_V2.md`, Section 2.8:
  subsection contract, evidence families, review boundary and rebuild scope.
- `src/pig_behavior/classification_v2/review/behavior_evidence.py`:
  review-only motion, ROI, social, posture/shape and temporal evidence, with
  explicit evidence availability handling.
- `src/pig_behavior/classification_v2/review/behavior_review_selection.py`:
  deterministic candidate predicates, mandatory/high-risk/stratified cohorts,
  threshold provenance and audit-only selection metadata.
- `src/pig_behavior/classification_v2/review/review_unit_builder.py`:
  source-specific review-unit construction and the separation between native
  review units and downstream training windows.
- `src/pig_behavior/classification_v2/review/behavior_review_contract.py`:
  decision semantics, decision coverage, apply scope and native-unit contract.
- `scripts/classification_v2/01_review_units_gui/
  classification_v2_apply_review_unit_decisions.py`:
  application of resolved decisions while preserving pre-review behavior and
  recording the reviewed result and training action.
- `src/pig_behavior/classification_v2/review/behavior_consistency_audit.py`:
  interaction-partner checks and conservative temporal-continuity findings,
  including the bounded `fight` interruption candidate.
- `configs/classification_v2/model_research/
  behavior_review_reproduction_contract.yaml`:
  required corrected-source rebuild, before/after comparison and downstream
  reproduction boundary.
- `docs/CLASSIFICATION_V2_CURRENT_STATE.md`:
  current review/rebuild authority and the separation between engineering
  validation and paper-grade model results.

## Visual anchor

**PENDING.** A compact review-lineage diagram or evidence-family table may be
added after the final review authority and Chapter 3 reporting scope are bound.
It should show source annotation → evidence-based candidate → human decision →
corrected-source rebuild → model-window audit. A GUI screenshot is not required
for the scientific explanation and should not be used unless it is tied to a
source-bound example and a reproducible caption.

## Open questions and claim boundary

- The final manuscript should link the review-lineage visual, if retained, to
  the same frozen review and rebuild authority used for Chapter 3.
- Candidate-generation rules describe review eligibility, not a quantitative
  behavior-performance result; correction counts, class distributions and
  model comparisons belong in the registered results artifacts.
- Any paper claim about post-review model performance requires the corresponding
  rebuilt data, split, model, evaluator and artifact lineage; this section does
  not promote engineering smoke results to paper metrics.

## Editorial status

The Vietnamese content states the intended scientific meaning, and the English
text is an original academic rendering rather than a sentence-level software
description. No equation, threshold value or operational count is introduced
without an implementation-bound need; detailed thresholds and activation counts
remain in their manifests, audit reports or results sections.
