# Section 2.6 — Legacy Recovery, Source Harmonization and Corrected-Source Lineage

**Draft language:** Vietnamese and English  
**Draft status:** Accepted meaning; academic-language revision completed  
**English conversion:** Original academic prose added  
**Approved thesis title:** *An AI Spatio-Temporal Framework for Dynamic
Behavior Profiling and Anomaly Detection in Group-Housed Pigs*

## Vietnamese thesis draft

Tập dữ liệu phân loại được hình thành từ hai tập annotation được tạo độc lập
trên cùng bộ video. Cả hai tập đều được gán nhãn thủ công bằng CVAT theo cùng
một hệ thống mười lớp hành vi, nhưng khác nhau về phạm vi lựa chọn dữ liệu. Tập
thứ nhất gồm các đoạn hành vi liên tục trong một số video được chọn, còn tập
thứ hai gồm các đoạn ngắn được lấy từ nhiều ngày ghi hình và nhiều video hơn.
Hai nguồn được hợp nhất để vừa giữ diễn tiến hành vi trong từng đoạn, vừa mở
rộng biến thiên theo ngày ghi hình và bối cảnh quan sát; nguồn thứ hai không
được xem là một tập đánh giá tách biệt hay bị loại khỏi dữ liệu huấn luyện chỉ
vì nguồn gốc của nó.

Trước khi hợp nhất, các đơn vị từ hai nguồn được đưa về cùng một biểu diễn
mẫu. Mỗi mẫu vẫn gắn với video nguồn, ngày ghi hình, cá thể, khoảng khung hình,
nhãn hành vi và thời điểm thu nhận. Việc bảo toàn các thông tin này cho phép
truy nguyên mẫu về quan sát ban đầu và phân chia dữ liệu theo nhóm ngày hoặc
video thay vì coi các đoạn có liên quan như những mẫu độc lập. Độ dài đoạn gán
nhãn trong nguồn được giữ riêng với độ dài cửa sổ thời gian do mô hình sử dụng
ở bước sau.

Việc hợp nhất diễn ra ở cấp bản ghi quan sát frame–object. Mỗi bản ghi giữ
nguồn, video, khung hình và cá thể trong khóa truy nguyên, nên hai bản ghi cùng
tham chiếu đến một frame–actor nhưng xuất phát từ hai bộ annotation vẫn được
giữ như hai quan sát có nguồn gốc riêng. Biểu diễn hợp nhất bảo toàn cả hai
bản ghi thay vì cho một nguồn ghi đè nguồn kia; một hiệu chỉnh chỉ có thẩm
quyền trong nguồn và phạm vi khung hình mà quyết định đó đề cập. Ngược lại,
trùng lặp thực sự
trong cùng phạm vi nguồn được xem là lỗi dữ liệu và phải được phát hiện bởi
kiểm tra truy nguyên trước khi xây dựng các biểu diễn tiếp theo.

Khi quá trình đánh giá xác định rằng một nhãn hoặc thông tin không gian của
đơn vị cần được điều chỉnh, quyết định đó được áp dụng trong phạm vi video và
đoạn khung hình tương ứng, đồng thời vẫn giữ lại liên hệ với dữ liệu ban đầu.
Giá trị trước hiệu chỉnh được giữ trong hồ sơ truy nguyên cùng với bằng chứng
trước và sau hiệu chỉnh và bản sao phục hồi; nguồn sau hiệu chỉnh trở thành
căn cứ có thẩm quyền cho các bước tiếp theo. Một hiệu chỉnh vì thế kích hoạt
việc tái tạo mọi đặc trưng
hình học, chuyển động, vùng chức năng và quan hệ xã hội bị ảnh hưởng, cũng như
các đơn vị thời gian và cửa sổ phụ thuộc vào chúng. Các biểu diễn cũ của phạm
vi bị ảnh hưởng không được sử dụng lại như thể chúng vẫn còn hợp lệ.

Do đó, tính nhất quán của dữ liệu được duy trì theo trình tự: quyết định đánh
giá, cập nhật nguồn đã hiệu chỉnh, kiểm tra phạm vi và truy nguyên, tái tạo đặc
trưng, tạo đơn vị và cửa sổ thời gian, xác lập nhóm phân chia, rồi kiểm tra
chồng lấn và rò rỉ trước huấn luyện. Split được gán sau khi tạo cửa sổ nhưng ở
cấp nhóm ghi hình hoặc đơn vị nguồn, không phải độc lập cho từng cửa sổ; mọi
cửa sổ cùng nhóm kế thừa cùng một vai trò. Do đó, split không được gán cho các
bản ghi gốc trước khi harmonization hoàn tất và một cửa sổ không tự tạo một
biên phân chia mới. Một kiểm tra truy nguyên chỉ được chấp nhận khi danh mục
nguồn và dấu vân tay nội dung khớp, khóa frame–object và đơn vị
thời gian là đầy đủ và duy nhất, các biểu diễn cửa sổ giữ nguyên tập khóa và
thứ tự tham chiếu, số bản ghi và liên kết trước–sau được bảo toàn, thứ tự thời
gian và đồng hồ nguồn hợp lệ, các hiệu chỉnh khớp đúng phạm vi, đồng thời không
có trùng lặp hoặc rò rỉ giữa các nhóm phân chia. Trình tự
này tách dữ liệu quan sát ban đầu khỏi dữ liệu sau hiệu chỉnh, đồng thời bảo
đảm rằng nguồn bổ sung về ngày và video làm tăng tính đa dạng của tập huấn
luyện mà không làm mất khả năng kiểm tra nguồn gốc của từng mẫu.

## English academic thesis draft

The classification dataset was assembled from two independently annotated
collections derived from the same video corpus. Both collections were manually
labelled in CVAT using the same ten-class behavior taxonomy, but their sampling
scopes differed. The first collection comprised continuous behavioral
intervals selected from a subset of videos, whereas the second comprised
shorter intervals sampled across a wider range of recording dates and videos.
The collections were combined to preserve behavioral progression within each
interval while increasing variation across recording days and observation
contexts. The second collection therefore broadened temporal and date coverage
within the same pooled training dataset; it was not treated as a
separate evaluation cohort or excluded solely because of its provenance.

Before combination, units from both collections were mapped to a common sample
representation. Each sample retained its source video, recording date, actor,
frame interval, behavior label and acquisition time. These fields enabled
sample-level traceability and grouping by date or video during partitioning,
rather than treating related intervals as independent observations. The source
annotation duration was retained as provenance information and kept distinct
from the temporal window length subsequently used by the model.

The merge operated at the frame–object observation level. Each record retained
its source, video, frame and actor in a source-qualified provenance key. Thus,
records that referred to the same nominal frame–actor but originated from the
two annotation collections remained distinct source observations. The merged
representation preserved both records rather than allowing one collection to
overwrite the other; a correction was authoritative only within the source and
frame scope covered by that decision. A duplicate within one source scope was
instead treated as a data-integrity failure that had to be resolved before
downstream representations were built.

During review, a required correction to a behavior label or spatial annotation
was applied to the corresponding video and frame interval while preserving the
link to the original annotation. The pre-correction value remained in the
lineage with before-and-after evidence and an archived recovery copy, while the
corrected source became the authoritative input for subsequent processing. A
correction therefore required recomputation of every affected geometric,
motion, functional-region and social representation, together with the
temporal units and windows that depended on those observations. Stale
representations from the affected scope were not reused.

The harmonization procedure followed a fixed sequence: review decision,
corrected-source update, scope and provenance audit, source-derived feature
rebuild, native-unit and temporal-window construction, grouped split assignment
and overlap/leakage audit before training. Split roles were assigned after
window construction at the recording-group or source-unit level, not
independently for each window; all windows in one group inherited the same
role. Split roles were therefore not assigned to raw annotation rows before
harmonization, and a window could not define an independent boundary. An audit
was accepted only when source manifests and content hashes matched,
frame–object and temporal-unit keys were complete and unique, derived window
views preserved the reference key set and order, source-row counts and lineage
links were preserved, temporal order and the source clock were valid, the
correction scope was matched exactly, and no duplicate or grouped leakage was
detected. This sequence distinguishes the original annotations
from their corrected representations and allows the additional date and video
coverage to increase training diversity without sacrificing sample-level
traceability.

## Evidence anchors

- `docs/thesis_drafts/CHAPTER_2_2_DATA_SOURCES_NATIVE_UNITS_VI_DRAFT.md`
- `docs/CLASSIFICATION_V2_THESIS_BLUEPRINT_EVIDENCE_MAP.md`
- `docs/CLASSIFICATION_V2_CURRENT_STATE.md`
- User-confirmed description of the two CVAT annotation collections and their
  pooled use in the classification dataset.

## Visual anchor

No separate figure is required for this section. The distinction between the
two annotation collections, their harmonization and the correction sequence can
be presented in prose or, if necessary, in a compact source–provenance table.
A figure should not be added merely to repeat the workflow described above.

## Open questions

- Final counts for each source and recording date will be added in Section 3
  after the evaluation snapshot has been frozen.
- The Vietnamese meaning was confirmed before the English academic rewrite.

## Editorial status

The Vietnamese meaning has been rewritten as academic English rather than
translated sentence by sentence. Final source- and date-level counts remain
reserved for Section 3 until the evaluation snapshot is frozen.
