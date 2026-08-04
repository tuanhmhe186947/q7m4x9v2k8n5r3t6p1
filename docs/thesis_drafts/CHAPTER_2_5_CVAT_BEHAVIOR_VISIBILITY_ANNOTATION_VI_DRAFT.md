# Section 2.5 — CVAT Behavior and Visibility Annotation

**Draft language:** Vietnamese first  
**Draft status:** Vietnamese meaning accepted; English academic draft added  
**English conversion:** Completed for review  
**Approved thesis title:** *An AI Spatio-Temporal Framework for Dynamic
Behavior Profiling and Anomaly Detection in Group-Housed Pigs*

## Vietnamese thesis draft

Việc gán nhãn hành vi được thực hiện ở cấp cá thể trên các đoạn video có thứ
tự thời gian. Trong nguồn CVAT, mỗi đơn vị gán nhãn liên kết một cá thể với
anchor gồm sáu frame liên tiếp, từ $k$ đến $k+5$, và một nhãn hành vi. Nhãn mô
tả hoạt động của cá thể trong đơn vị đó, không phải tư thế của một frame đơn
lẻ. Native annotation unit được giữ tách khỏi các cửa sổ temporal được tạo
sau khi harmonization.

Hệ thống phân loại gồm mười lớp `drink`, `eat`, `fight`, `social-nose`, `explore`,
`lying`, `stand`, `move`, `sitting` và `playwithtoy`. Các lớp này bao quát sử
dụng tài nguyên, tương tác xã hội, khám phá môi trường, vận động và tư thế.
Bên cạnh mục tiêu hành vi mười lớp, posture được tổ chức như một mục tiêu độc
lập để hỗ trợ phân tích các trạng thái tư thế và các lỗi liên quan. Với các lớp
tương tác, `fight` được gắn cho các cá thể trực tiếp tham gia, còn
`social-nose` chỉ gắn cho actor chủ động.

Trong các annotation CVAT, người gán nhãn đánh dấu `Hidden` ở cấp
frame–object. Nhãn này cho biết việc che khuất hoặc mất quan sát làm suy giảm
bằng chứng cần thiết để xác định hành vi. Một vòng review thứ hai sau đó xem
lại từng phán đoán ban đầu dựa trên bằng chứng hình ảnh và ngữ cảnh theo thời
gian. Người review đưa ra quyết định visibility `Yes` hoặc `No`; nghiên cứu
giữ cả phán đoán ban đầu và quyết định sau review để truy nguyên các thay đổi.

Quyết định sau review cung cấp giá trị visibility được sử dụng trong đồng nhất
hóa theo thời gian và khi tạo temporal window. `Hidden` độc lập với nhãn hành
vi, posture và anomaly. Vòng review này chỉ giải quyết visibility; việc giữ
nguyên, hiệu chỉnh hoặc loại một nhãn hành vi thuộc protocol review hành vi
riêng. Các quyết định vẫn ở cấp frame–object và không tự động gộp thành một
nhãn duy nhất cho toàn bộ temporal window.

Temporal harmonization tổng hợp các quyết định visibility đã review khi tạo
các window của mô hình theo policy trình bày ở Section 2.9. Vì vậy, `Hidden`
không phải là một nhãn window-level được gán trực tiếp trong CVAT.

`Hidden` chỉ phục vụ kiểm soát chất lượng quan sát; mô hình không sử dụng nó
như một đặc trưng đầu vào và không dự đoán nó như một đầu ra. Nhãn này cũng
không thay thế cho nhãn hành vi. Việc sàng lọc sai lệch hành vi ở các bước sau
vẫn dựa trên các dự đoán hành vi và hồ sơ theo thời gian.

## English academic thesis draft

Behavior annotation was performed at the level of individual animals on
temporally ordered video segments. In the CVAT source, each annotation unit
linked one animal to a six-frame anchor interval, $k$ through $k+5$, and a
behavior label. The label described the animal's activity within that unit
rather than a pose in an isolated frame. The native annotation unit was kept
distinct from the temporal windows constructed after harmonization.

The behavior taxonomy comprised ten classes: `drink`, `eat`, `fight`,
`social-nose`, `explore`, `lying`, `stand`, `move`, `sitting` and `playwithtoy`.
Together, these classes represented resource use, social interaction,
environmental exploration, locomotion and posture. Posture was additionally
organized as an independent target to support the analysis of postural states
and posture-related errors. For interaction classes, `fight` was assigned to
directly involved animals, whereas `social-nose` was assigned to the active
actor.

In the CVAT annotations, annotators marked `Hidden` at the frame-object level.
The label indicated whether occlusion or loss of observation reduced the visual
evidence needed to identify the animal's behavior. A second human-review pass
then revisited each initial judgment using the image evidence and its temporal
context. The reviewer issued a binary visibility decision, `Yes` or `No`, and
the study retained both the initial and reviewed judgments so that corrections
could be traced.

The reviewed decision supplied the visibility value used for temporal
harmonization and temporal-window construction. `Hidden` was independent of
behavior, posture and anomaly labels. This review addressed visibility only;
retaining, correcting or excluding a behavior label belonged to a separate
behavior-review protocol. Decisions remained at the frame-object level and
were not automatically collapsed into one label for an entire temporal window.

Temporal harmonization subsequently aggregated the reviewed visibility
decisions when constructing model windows under the policy described in Section
2.9. Thus, `Hidden` was not a window-level annotation assigned directly in
CVAT.

`Hidden` served only to control observation quality; the model did not use it as
an input feature or predict it as an output. The visibility judgment did not
replace the behavior label, and subsequent behavioral-deviation screening
remained based on behavior predictions and their temporal profiles.

## Drafting sources (working note; not manuscript prose)

- `docs/thesis_drafts/PIG_BEHAVIOR_THESIS_MASTER_OUTLINE_V2.md`
- `docs/CLASSIFICATION_V2_CURRENT_STATE.md`
- `docs/CLASSIFICATION_V2_DATA_REBUILD_AND_HUMAN_REVIEW_RUNBOOK.md`
- `docs/CLASSIFICATION_V2_GUI_OPERATOR_GUIDE.md`
- `docs/CLASSIFICATION_V2_THESIS_BLUEPRINT_EVIDENCE_MAP.md`

## Draft status (working note)

- Behavior ontology: `USER-CONFIRMED`.
- Frame-level visibility semantics: `USER-CONFIRMED` and implemented as a
  quality-control and training-selection attribute.
- Hidden review status: `USER-CONFIRMED` complete; reviewed decisions are used
  for temporal-window visibility screening, not as a behavior target.
- Quantitative annotation-quality and inter-reviewer results: `PENDING` until
  a bound evaluation population and report are available.

## Visual status (working note)

Không yêu cầu hình riêng cho mục này ở giai đoạn hiện tại. Các trường
visibility và review decision có thể được trình bày bằng một bảng schema ngắn
nếu cần;
không nên tạo hình minh họa chỉ để lặp lại nội dung của đoạn văn.

## Editorial status (working note)

Bản tiếng Việt đã được chấp nhận và đã được chuyển thành bản tiếng Anh học
thuật theo ý nghĩa, không dịch từng câu.
