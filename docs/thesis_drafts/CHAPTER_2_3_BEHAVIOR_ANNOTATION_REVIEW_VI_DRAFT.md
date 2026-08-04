# Chapter 2, Section 2.3 — Behavior Annotation and Human Review Protocol

**Draft language:** Vietnamese  
**Draft status:** Initial Vietnamese methodology draft  
**English conversion:** Not started  
**Thesis title:** *An AI Spatio-Temporal Framework for Dynamic Behavior
Profiling and Anomaly Detection in Group-Housed Pigs*

> **Superseded section notice.** This file was originally drafted as Section
> 2.3 for behavior annotation and human review. The approved outline places
> detection-data construction in Section 2.3. Use
> `CHAPTER_2_3_DETECTION_DATASET_CONSTRUCTION_VI_EN_DRAFT.md` as the current
> Section 2.3 draft. The behavior-review material is retained as provenance for
> the later human-review section.

## Superseded draft prose (retained for provenance)

Độ tin cậy của nhãn hành vi là điều kiện cần để đánh giá mô hình phân loại và
phân tích hồ sơ hành vi. Vì vậy, luận văn sử dụng một quy trình human review ở
cấp độ đơn vị quan sát gắn với actor track. Người đánh giá không xem một frame
đơn lẻ mà kiểm tra chuỗi ngữ cảnh xung quanh đơn vị được chọn, đồng thời xem
xét sự liên tục của hành vi, vị trí của cá thể và các tương tác liên quan. Kết
quả review được dùng để xác định nhãn có thể giữ nguyên, nhãn cần hiệu chỉnh
hoặc trường hợp không đủ điều kiện kỹ thuật để đưa vào phân tích.

Quy trình review được tổ chức theo nhiều lớp kiểm soát. Lớp đầu tiên rà soát
phạm vi hành vi đã được xác định trước. Lớp tiếp theo kiểm tra tính nhất quán
theo thời gian và các trường hợp tương tác, đặc biệt khi các đơn vị lân cận có
liên quan đến `fight`, `social-nose`, `move` hoặc `explore`. Sau đó, một phạm vi
residual được chọn từ các vùng có dấu hiệu sai lệch ở ranh giới hoặc trong
chuỗi tương tác. Một mẫu control độc lập được lấy trước khi sử dụng kết quả
review để ước lượng mức lỗi còn sót lại trong phần dữ liệu chưa được kiểm tra.
Cuối cùng, fixed-point audit được thực hiện để xác định liệu các hiệu chỉnh đã
tạo ra khoảng trống nhất quán mới trong chuỗi hành vi hay chưa.o

Các quyết định review, ghi chú của người đánh giá, lý do lựa chọn phạm vi, thứ
hạng và các trường chất lượng chỉ phục vụ kiểm soát dữ liệu. Những trường này
không được đưa vào model input. Nhãn dùng cho huấn luyện và đánh giá được lấy
từ source-label authority sau khi các quyết định hiệu chỉnh đã được áp dụng
theo thứ tự và kiểm tra tính toàn vẹn của nguồn. Các thay đổi liên quan đến
identity hoặc bounding box có thể làm mất hiệu lực của đặc trưng không gian và
chuyển động; việc cập nhật nguồn và tái xây dựng các đặc trưng đó được trình
bày ở Section 2.4.

Quy trình trên là một cơ chế kiểm soát chất lượng annotation, không phải một
quy trình gán nhãn bệnh lý hoặc anomaly ground truth. Behavioral deviation chỉ
được suy ra ở tầng phân tích sau khi các dự đoán hành vi đã được sắp xếp theo
thời gian. Do đó, kết quả human review không được diễn giải như bằng chứng
trực tiếp về stress, bệnh, chấn thương hoặc tình trạng phúc lợi của lợn.

## Visual anchor for this section

**Figure 4 — Annotation and review lineage.** The figure should show the flow
from source annotations through primary behavior review, consistency review,
targeted residual review, independent control, and fixed-point audit to the
corrected source authority. Selection metadata and reviewer notes should be
shown as audit-only outputs, separate from model inputs. The number is based on
the current global figure inventory and should be renumbered together with the
other figures after the figure plan is frozen.

## Evidence anchors

- `docs/CLASSIFICATION_V2_POST_REVIEW_LEARNING_PIPELINE.md`
- `docs/CLASSIFICATION_V2_GUI_OPERATOR_GUIDE.md`
- Review-close authority:
  `outputs/classification_v2/review_authority/`
  `review_close_behavior_3243_faee589_20260802_082500_v1/review_close_authority.json`
- Fixed-point audit:
  `outputs/classification_v2/review_authority/`
  `post_review_fixed_point_final_3243_386c304_20260802_081500_v1/`
  `post_review_residual_suspicion_audit.json`
- Review scripts under `scripts/classification_v2/01_review_units_gui/`.

## Evidence status and open items

- Review protocol: `IMPLEMENTED`; the protocol separates label correction from
  selector diagnostics and excludes review metadata from model input.
- Review-close artifact: `FROZEN` in the current authority artifact; final
  thesis counts should be reported only after authority reconciliation.
- Corrected-source lineage: available through the frozen source authority;
  affected spatial and motion features require an audited rebuild.
- Anomaly labels: not available as expert-confirmed ground truth; downstream
  outputs remain behavioral-deviation screening signals.

## Review gate before English conversion

The user should confirm the review-layer ordering, the separation between data
quality control and anomaly labeling, and the boundary between review metadata
and model input. Quantitative review counts will be inserted later in Chapter
3.1 after the final authority and dataset snapshot are reconciled.
