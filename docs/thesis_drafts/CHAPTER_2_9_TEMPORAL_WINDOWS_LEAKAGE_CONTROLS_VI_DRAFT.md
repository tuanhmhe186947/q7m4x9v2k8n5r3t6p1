# Section 2.9 — Temporal Windows and Leakage Controls

**Draft language:** Vietnamese and English  
**Draft status:** Provisional methodology; the final temporal-view and feature
manifests remain required before quantitative results are promoted.  
**Approved thesis title:** *An AI Spatio-Temporal Framework for Dynamic
Behavior Profiling and Anomaly Detection in Group-Housed Pigs*

## Vietnamese thesis draft

### Ranh giới giữa đơn vị chú thích và cửa sổ mô hình

Đơn vị chú thích gốc và cửa sổ đầu vào của mô hình là hai cấp khác nhau. Nguồn
CVAT cung cấp đơn vị neo sáu frame, trong khi nguồn legacy có đơn vị gốc dài
16 frame. Sau khi đồng nhất hóa, mỗi đơn vị vẫn giữ khóa native-unit,
nguồn, video, actor và khoảng frame; cửa sổ mô hình chỉ là một view được tạo
từ các quan sát đó. Vì vậy, một cửa sổ không được xem là một quyết định chú
thích mới và cũng không được dùng để thay thế ranh giới của native unit.

Window builder ghi lại các native-unit cấu thành, khoảng frame được chọn,
timestamp nguồn và trạng thái khả dụng. Các cửa sổ có thể chồng lấn trong cùng
video, nhưng các cửa sổ chồng lấn và native unit liên quan phải giữ cùng vai
trò split. Khi một hiệu chỉnh làm thay đổi quan sát nguồn, các cặp động học,
quan hệ không gian và tổng hợp của mọi cửa sổ bị ảnh hưởng phải được tính lại.

### Cấu hình view và quy tắc neo thời gian

Registry hiện định nghĩa các view mục tiêu liên tiếp có độ dài 6, 8, 12 và 16
frame. Với endpoint \(e\) và độ dài \(L\), tập frame mục tiêu của view liên
tiếp là

\[
\mathcal W_{e,L}=\{e-(L-1),\ldots,e-1,e\}.
\]

Endpoint là thời điểm dự đoán; không có frame nào sau endpoint được đưa vào
view. Cửa sổ T6 liên tiếp vì thế có năm cặp kề nhau, còn các view dài hơn
giữ cùng quy tắc nhưng có nhiều slot hơn. Registry đánh dấu các view liên
tiếp là có thể so sánh giữa hai nguồn, trong khi view chính được báo cáo chỉ
được chốt sau khi manifest temporal cuối cùng được đóng băng.

Ngoài các view liên tiếp, registry còn giữ các nhánh ablation có lịch sử nhân
quả và các view legacy-only. Một view lịch sử gồm target T6 và một đoạn history
đứng hoàn toàn trước target; history chỉ cung cấp ngữ cảnh hình ảnh hoặc không
gian, không cung cấp nhãn cho \(X\). View sparse sáu slot của burst legacy 16
frame chọn đúng các vị trí (0,3,6,9,12,15), còn historical screen là một
nhánh lịch sử không được chuyển thành view chính giữa hai nguồn.

Trong builder hiện tại, cửa sổ legacy liên tiếp được sinh với bước mặc định
ba frame; cửa sổ CVAT được neo theo các interval label với bước một interval.
Do đó, các cửa sổ có thể overlap và một cửa sổ dài có thể bao phủ nhiều native
unit. Sự overlap này được lưu bằng danh sách constituent-unit, không được coi
là thêm các quan sát độc lập khi tính metric. Việc có đưa cửa sổ chuyển tiếp
hoặc mixed vào nhánh robustness hay không phải tuân theo cấu hình đã đóng băng.

Nhãn mục tiêu của cửa sổ phải là một nhãn hành vi đã được giải quyết. Cửa sổ
ổn định một nhãn mới đủ điều kiện cho `main_train`; cửa sổ transition hoặc
mixed có thể được giữ lại như `robust_train_only` khi policy cho phép, nhưng
không được dùng như cửa sổ chính ổn định. Cửa sổ thiếu coverage nhãn hoặc có
nhiều nhãn không được âm thầm gán nhãn chiếm đa số để làm sạch dữ liệu.

### Thời gian nguồn và quyền truy cập nhân quả

Các timestamp được lấy từ đồng hồ nguồn. Với một cặp quan sát hợp lệ,

\[
\Delta f=f_t-f_{t-1},\qquad
\Delta\tau=\tau_t-\tau_{t-1},\qquad
f_{\mathrm{eff}}=\operatorname{median}\!\left(\frac{\Delta f}{\Delta\tau}\right),
\]

trong đó chỉ các cặp có \(\Delta f>0\), \(\Delta\tau>0\) và timestamp hữu hạn
mới được dùng. Duration quan sát được là

\[
D_{\mathrm{obs}}=\tau_{\mathrm{end}}-\tau_{\mathrm{start}},
\]

còn duration của timeline khai báo được suy ra từ khoảng frame và
\(f_{\mathrm{eff}}\) theo đúng manifest. Tốc độ đóng gói 30 fps của tệp phát
lại không thay thế đồng hồ thu nhận 6 fps khi tính thời lượng hoặc tốc độ sinh
học.

Với view nhân quả, mọi offset tương đối endpoint đều không dương. History,
nếu có, phải kết thúc trước frame bắt đầu của target. Nhãn hành vi được lấy từ
target view; history label và các trường review không đi vào model input.
Registry ghi `future_frame_dependence = 0`; đây là điều kiện bắt buộc của cả
view liên tiếp và view có history.

### Visibility, eligibility và các tầng sử dụng cửa sổ

Sau khi quyết định visibility ở cấp frame–object đã được review và áp dụng,
đặt \(H_{u,t}=1\) nếu frame \(t\) của cửa sổ \(u\) là `Hidden=Yes`, và
\(H_{u,t}=0\) nếu là `Hidden=No`. Với \(L_u\) slot của cửa sổ, burden visibility
được tính bởi

\[
r_u=\frac{1}{L_u}\sum_{t=1}^{L_u}H_{u,t},\qquad
\ell_u=\frac{m_u}{L_u},
\]

trong đó \(m_u\) là độ dài run `Hidden` liên tiếp dài nhất trên các frame liền
kề. Window audit phải giữ riêng giá trị Hidden hiện hành, provenance/trust và
review coverage; không trường nào trong số đó trở thành feature dự đoán.

Các ngưỡng visibility được áp dụng theo độ dài view. Main tier cho phép tổng
Hidden và longest run lần lượt không vượt quá các giới hạn sau; robust tier cho
phép giới hạn rộng hơn:

| View | Main: tổng / run | Robust tối đa: tổng / run |
| --- | --- | --- |
| T6 | 1 / 1 | 3 / 2 |
| T8 | 2 / 1 | 4 / 3 |
| T12 | 3 / 2 | 6 / 4 |
| T16 | 4 / 3 | 8 / 6 |

Tương đương, `main_train` yêu cầu \(r_u\leq0.25\) và
\(\ell_u\leq0.20\). Cửa sổ vượt một ngưỡng main nhưng vẫn thỏa
\(r_u\leq0.50\) và \(\ell_u\leq0.40\) chỉ được xếp `robust_train_only`.
Vượt một trong hai ngưỡng robust thì cửa sổ bị `exclude`, giữ lại trong
lineage nhưng không đóng góp vào mục tiêu huấn luyện. So sánh dùng dấu (>),
nên giá trị đúng bằng ngưỡng vẫn ở tầng thấp hơn.

Eligibility còn yêu cầu label coverage đầy đủ, một nhãn ổn định cho main,
bounding box hợp lệ theo ngưỡng cấu hình, spatio-temporal evidence hợp lệ,
không có frame bị review loại và không vi phạm điều kiện review đã khai báo.
Các cửa sổ bị loại nhận training mask không hợp lệ và sample weight bằng 0;
chúng không được diễn giải là quan sát hành vi bằng không.

### Split theo nhóm và các bất biến chống leakage

Split được gán sau khi native unit và cửa sổ đã được tạo, nhưng authority của
split là recording group, không phải từng row hoặc từng frame. Tất cả quan sát
thuộc cùng ngày ghi hình phải giữ cùng vai trò trong một fold; các nguồn CVAT và
legacy cùng ngày không bị tách riêng chỉ vì khác nguồn. Trong validation nội bộ,
session, video hoặc burst-group được dùng làm nhóm con khi protocol yêu cầu.

Với \(g(w)\) là nhóm split của cửa sổ \(w\) và \(K(w)\) là tập native-unit key
cấu thành cửa sổ, điều kiện không rò rỉ được viết là

\[
\operatorname{split}(w)=\operatorname{split}(g(w)),\qquad
\operatorname{split}(k)=\operatorname{split}(w)
\quad\forall k\in K(w).
\]

Do đó, các window overlap, các window có cùng actor-track trong cùng phạm vi
và các interval nguồn lân cận không được tách giữa train và test. Khi tính
metric, các window chồng lấn được quy về native temporal unit trước khi tổng
hợp để không làm phồng kích thước mẫu. `pig_id` chỉ là identity cục bộ của
annotation; nó không được dùng như biological identity xuyên video.

Audit leakage phải kiểm tra tối thiểu: khóa native-unit duy nhất, window ID
duy nhất, group/date purity, object-track purity, không có neighboring source-time
interval qua biên split, không có near-duplicate bị phân bố chéo, thứ tự frame và
timestamp hợp lệ, cùng với `future_frame_dependence = 0`. Random row split và
random window split đều bị loại khỏi protocol.

### Ranh giới giữa dữ liệu audit và model input

Model \(X\) chỉ nhận các trường đã được whitelist và có mặt trong temporal view
được đóng băng. Behavior label, posture label, Hidden, review decision,
reviewer metadata, source type, video, actor/track identifiers, split ID,
đường dẫn tệp và các trường target-derived đều bị loại khỏi \(X\). History
labels chỉ dùng để kiểm tra ranh giới thời gian; chúng không được biến thành
ngữ cảnh học.

Các mask về độ dài, frame quan sát được, tính hợp lệ của hình học và coverage
cặp được truyền như điều khiển khả dụng của sequence. Chúng không được diễn
giải là giá trị hành vi. Các đặc trưng phụ thuộc view phải được tính lại trong
chính view đó sau khi split và temporal sampling đã được xác lập.

### Thứ tự xây dựng và tiêu chí chấp nhận

Quy trình theo thứ tự: (i) xác lập corrected-source authority; (ii) đồng nhất
hóa temporal và tạo native units; (iii) sinh các candidate window theo view;
(iv) tính lại feature, pair và aggregate trong từng view; (v) đánh giá label,
Hidden, geometry và review eligibility; (vi) gán split theo recording group;
(vii) chạy overlap/leakage audit; và (viii) chỉ sau khi mọi audit đạt mới đóng
băng manifest cho training hoặc evaluation.

Một audit chỉ được coi là đạt khi các manifest nguồn và hash khớp nhau, khóa
frame–object/native-unit/window đầy đủ và duy nhất, thứ tự frame và source
clock hợp lệ, các view giữ đúng offset đã khai báo, không có frame tương lai,
không có nhãn hoặc metadata review trong (X), và không có group leakage. Bất
kỳ correction nào sau bước này đều làm mất hiệu lực các window và feature bị
ảnh hưởng, buộc phải rebuild theo đúng lineage mới.

## English academic thesis draft

### Native units and model windows

The native annotation unit and the model-input window are distinct objects. The
CVAT source contributes six-frame anchor units, whereas the legacy source uses
16-frame native bursts. After harmonization, each unit retains its native-unit
key, source, video, actor and frame span. A model window is a declared temporal
view derived from those observations; it is not a new annotation decision and
does not replace the native-unit boundary.

The window builder records the constituent native units, selected frame range,
source timestamps and availability state. Windows may overlap within a video,
but overlapping windows and their native units inherit one split role. A source
correction invalidates the affected motion pairs, spatial relations and window
aggregates, which must then be recomputed from the corrected lineage.

### Temporal-view construction

The registry defines contiguous target views of 6, 8, 12 and 16 frames. For
endpoint \(e\) and target length \(L\),

\[
\mathcal W_{e,L}=\{e-(L-1),\ldots,e-1,e\}.
\]

The endpoint is the prediction time, so no target frame occurs after it. The
contiguous T6 view contains five adjacent pairs; longer views follow the same
causal rule. The registry marks contiguous target views as cross-source
eligible, but the promoted primary view must be bound to the final temporal
manifest rather than inferred from a configuration name alone.

The registry also defines causal-history ablations and legacy-only views. A
history view contains the T6 target followed by a history segment that ends
strictly before the target start. History may provide earlier visual or spatial
context, but history labels never enter \(X\). The legacy sparse six-slot view
selects offsets \(0,3,6,9,12,15\) inside one 16-frame burst. The historical
screen remains legacy-only and is not transferred as the primary cross-source
view.

The current builder uses a default stride of three frames for dense legacy
windows and one label interval for CVAT windows. Consequently, windows can
overlap and longer windows can contain several native units. Their constituent
keys are retained for split and metric audits; overlap is not counted as
independent biological evidence. Transition or mixed windows are retained only
when the frozen robustness policy permits them.

The target must carry one resolved behavior label. A stable single-label window
is eligible for the main tier. A transition or mixed window may be retained as
`robust_train_only` when declared by the policy, but it is not treated as a
stable main-training sample. Missing label coverage or multiple labels are not
silently resolved by assigning the majority label.

### Source time and causal access

Timestamps are taken from the acquisition clock. For a valid ordered pair,

\[
\Delta f=f_t-f_{t-1},\qquad
\Delta\tau=\tau_t-\tau_{t-1},\qquad
f_{\mathrm{eff}}=\operatorname{median}\!\left(\frac{\Delta f}{\Delta\tau}\right),
\]

where only finite pairs with positive frame and time differences contribute.
The observed span is

\[
D_{\mathrm{obs}}=\tau_{\mathrm{end}}-\tau_{\mathrm{start}},
\]

while the declared timeline duration is derived from the declared frame span
and the effective source rate recorded by the manifest. The 30-fps packaging of
the playback file does not replace the 6-fps acquisition clock when biological
duration or rate is computed.

For a causal view, every endpoint-relative offset is non-positive. If history
is present, it ends strictly before the target begins. Behavior labels are
assigned from the target view; history labels and review fields are not model
inputs. The registry records `future_frame_dependence = 0`, which is required
for both contiguous targets and causal-history views.

### Visibility and window eligibility

After frame–object visibility decisions have been reviewed and applied, let
\(H_{u,t}=1\) for `Hidden=Yes` in frame \(t\) of window \(u\), and \(H_{u,t}=0\)
for `Hidden=No`. For a window with \(L_u\) slots,

\[
r_u=\frac{1}{L_u}\sum_{t=1}^{L_u}H_{u,t},\qquad
\ell_u=\frac{m_u}{L_u},
\]

where \(m_u\) is the longest consecutive Hidden run. The audit retains the
applied Hidden value, provenance or trust information, and review coverage as
separate fields; none is a predictive feature.

The visibility contract is length-dependent. The main tier allows the total
Hidden count and longest run shown below, whereas the robustness tier permits
the wider limits:

| View | Main: total / run | Robust maximum: total / run |
| --- | --- | --- |
| T6 | 1 / 1 | 3 / 2 |
| T8 | 2 / 1 | 4 / 3 |
| T12 | 3 / 2 | 6 / 4 |
| T16 | 4 / 3 | 8 / 6 |

Equivalently, `main_train` requires \(r_u\leq0.25\) and
\(\ell_u\leq0.20\). A window above a main limit but within
\(r_u\leq0.50\) and \(\ell_u\leq0.40\) is retained only as
`robust_train_only`. Exceeding either robustness limit yields `exclude`.
The comparison is strict, so an exact boundary value remains in the lower tier.

Eligibility also requires complete label coverage, a stable single label for
the main tier, valid bounding boxes under the frozen quality threshold, valid
spatio-temporal evidence, no review-excluded frame and compliance with the
declared behavior-review requirement. Excluded windows remain in the lineage
with an invalid training mask and zero sample weight; they are not interpreted
as zero behavioral evidence.

### Grouped splits and leakage invariants

Splits are assigned after native units and candidate windows have been formed,
but the authority is a recording group rather than an individual row or frame.
All observations from one recording date retain one role within a fold; CVAT
and legacy observations from the same date are not separated merely because
their source types differ. Internal validation may use session, video or
burst-group keys when required by the registered protocol.

Let (g(w)) denote the split group of window (w), and let (K(w)) be its set
of constituent native-unit keys. The grouping invariant is

\[
\operatorname{split}(w)=\operatorname{split}(g(w)),\qquad
\operatorname{split}(k)=\operatorname{split}(w)
\quad\forall k\in K(w).
\]

Thus, overlapping windows, the same actor-track within one scope and neighboring
source-time intervals cannot cross train and test. Overlapping windows are
collapsed to native temporal units before metric aggregation so that overlap
does not inflate the effective sample size. `pig_id` is annotation-local and is
not treated as a biological identity across videos.

The leakage audit checks, at minimum, unique native-unit keys, unique window IDs,
group/date purity, object-track purity, no neighboring source-time interval
crossing, no cross-split near-duplicate, valid frame order and source clock, and
`future_frame_dependence = 0`. Random row splits and random window splits are
not acceptable substitutes for the grouped protocol.

### Audit fields versus model input

Model (X) receives only whitelisted fields from the frozen temporal view.
Behavior and posture labels, Hidden, review decisions, reviewer metadata, source
type, video, actor or track identifiers, split identifiers, paths and
target-derived fields are excluded from (X). History labels are used only to
check the temporal boundary and are never converted into learned context.

Length masks, observed-frame masks, geometry-validity masks and pair-coverage
masks are sequence-availability controls rather than behavioral values. Any
view-dependent feature must be recomputed within that view after temporal
sampling and split roles have been established.

### Construction order and acceptance criteria

The order is: (i) establish the corrected-source authority; (ii) harmonize
temporal observations and create native units; (iii) generate candidate
windows for each declared view; (iv) recompute view-local pairs, features and
aggregates; (v) evaluate label, Hidden, geometry and review eligibility; (vi)
assign recording-group roles; (vii) run overlap and leakage audits; and (viii)
freeze the manifest for training or evaluation only after every audit passes.

An audit is accepted only when source manifests and hashes agree, frame–object,
native-unit and window keys are complete and unique, frame order and source
clock are valid, view offsets match the registry, no future frame is used, no
label or review metadata enters (X), and no grouped leakage is detected. Any
later correction invalidates the affected windows and features and requires a
rebuild under the new lineage.

## Evidence anchors

- `docs/thesis_drafts/PIG_BEHAVIOR_THESIS_MASTER_OUTLINE_V2.md`, Section 2.9:
  native/model-window distinction, candidate lengths and leakage scope.
- `src/pig_behavior/classification_v2/temporal_views/registry.py` and
  `builder_contract.py`: endpoint-relative offsets, causal history, legacy
  sparse sampling and label/future-frame checks.
- `src/pig_behavior/classification_v2/features/sequence_windows.py`: source-
  specific window generation, Hidden tiers, label eligibility, view-local
  recomputation and source-clock summaries.
- `src/pig_behavior/classification_v2/splits/date_grouped_split.py` and
  `splits/split_audit.py`: grouped split and purity checks.
- `configs/classification_v2/data_contract_v2.json` and
  `feature_semantics_v2.json`: ordered artifact lineage, forbidden model-X
  fields and temporal/mask semantics.
- `docs/CLASSIFICATION_V2_DATA_REBUILD_AND_HUMAN_REVIEW_RUNBOOK.md`: declared
  Hidden thresholds and rebuild order.

## Visual anchor

A compact temporal-view and leakage-invariant table is more informative than a
software-layer diagram. The final table should be bound to the same frozen
temporal-view manifest used for the reported model results.

## Open questions and claim boundary

- The current checkout does not contain the referenced final temporal-view
  manifests. The baseline configuration names `fixed6_observed_time`, while
  the current registry names the comparable cross-source view
  `T6_TARGET_CONTIGUOUS`; the promoted primary must be resolved by one frozen
  manifest rather than by either name alone.
- The runbook describes the conservative applied-Hidden policy in terms of the
  current Hidden column, whereas the present builder also computes a
  trust-filtered Hidden series. The final authority must bind which series
  drives eligibility and preserve the other as audit evidence.
- The current feature-semantics audit reports a contract/export mismatch, so
  this section describes the implemented temporal protocol and its controls,
  not a claim that the train-ready feature snapshot is already final.
- No window counts, split counts or model metrics are asserted here. They belong
  to the frozen manifest and the registered evaluation sections.

## Editorial status

The Vietnamese text states the working scientific meaning, and the English text
is an original academic rendering. This subsection contains no performance or
generalization claim beyond the explicitly bounded temporal and leakage
contracts.
