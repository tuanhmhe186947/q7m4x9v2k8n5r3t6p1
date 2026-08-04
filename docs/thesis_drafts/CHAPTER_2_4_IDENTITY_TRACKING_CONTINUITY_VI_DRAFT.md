# Section 2.4 — Identity Tracking and Continuity

**Draft language:** Vietnamese and English  
**Draft status:** Accepted meaning; academic-language revision completed  
**English conversion:** Original academic prose added  
**Approved thesis title:** *An AI Spatio-Temporal Framework for Dynamic
Behavior Profiling and Anomaly Detection in Group-Housed Pigs*

## Vietnamese thesis draft

Tracking là lớp liên kết các detection qua thời gian để hình thành quỹ đạo
của từng đối tượng trong phạm vi một video. Vì các thống kê hành vi ở các
bước sau được tổng hợp theo cá thể, chất lượng định vị đối tượng và tính liên
tục của định danh được đánh giá như hai khía cạnh riêng biệt.

### 2.4.1. Identity Semantics and Tracking Scope

Nghiên cứu phân biệt năm khái niệm liên quan nhưng không tương đương:
detection, annotation-level pig ID, *video-local track ID*, trajectory và
biological identity. Detection là quan sát của một đối tượng tại một khung
hình; annotation-level pig ID là định danh trong dữ liệu chú thích;
*video-local track ID* là định danh do tracker gán cho chuỗi quan sát trong
một video; trajectory là chuỗi có thứ tự của các hộp giới hạn và thời điểm
tương ứng; còn biological identity là cá thể sinh học thực tế.

Các định danh trong dữ liệu chú thích và tracking chỉ có giá trị trong phạm vi
nguồn dữ liệu hoặc video tương ứng. Chúng không đủ để khẳng định một định danh
sinh học cố định trong toàn bộ sáu tuần ghi hình. Ranh giới này được giữ
nguyên khi tính thời lượng hành vi, bout, chuyển trạng thái và hồ sơ hành vi
theo cá thể.

### 2.4.2. Tracking Ground-Truth Construction

Các detection trước hết được dùng để tạo quỹ đạo ban đầu. Những đoạn có khả
năng mất liên kết, che khuất hoặc thay đổi định danh được đối chiếu với
annotation tracking trong CVAT và hồ sơ giải quyết xung đột. Các hiệu chỉnh
sau đó được áp dụng trở lại dữ liệu nguồn trong phạm vi video–khung hình đã
xác định và được kiểm tra truy nguyên trước khi phiên bản hiệu chỉnh được dùng
làm tracking ground truth.

Tracking ground truth và quỹ đạo do tracker dự đoán được giữ riêng. Quỹ đạo đã
xác minh phục vụ xây dựng dữ liệu hành vi và trích xuất vùng ảnh tập trung vào
cá thể; các quỹ đạo dự đoán được giữ nguyên khi tính chỉ số đánh giá và chỉ
được đối chiếu với ground truth đã xác minh trong bộ đánh giá tương ứng.

### 2.4.3. Tracking Input and Common Evidence Contract

Gọi $v$ là video nguồn, $t$ là thời điểm và $j$ là chỉ số detection. Mỗi bản
ghi đưa vào tracker được biểu diễn bởi

\[
d_{v,t,j}=(v,f_t,\tau_t,b_{t,j},c_{t,j}),
\]

trong đó $f_t$ là chỉ số khung hình, $\tau_t$ là dấu thời gian nguồn,
$b_{t,j}$ là hộp giới hạn và $c_{t,j}$ là confidence. Các bản ghi được sắp
xếp theo thời gian nguồn và xử lý độc lập trong từng video.

Mask hợp lệ của chuồng giới hạn vùng theo dõi. Trạng thái mất quan sát hoặc
kết thúc quỹ đạo được phân biệt với thuộc tính visibility `Hidden` trong
annotation. Đầu ra của tracker giữ lại video, track ID, frame index, dấu thời
gian nguồn, hộp giới hạn và confidence để phục vụ tái tạo và đánh giá.

Theo các cấu hình đã đóng băng, *ByteTrack-Raw* và *Hybrid-ByteTrack* sử dụng
detector ở mọi khung hình, còn *RealTime-Fast* và *RF-Hybrid* sử dụng detector
ở mỗi hai khung hình. Các phương pháp được so sánh như những pipeline hoàn
chỉnh; vì vậy, chênh lệch giữa *ByteTrack-Raw* với *Hybrid-ByteTrack* hoặc
giữa *RealTime-Fast* với *RF-Hybrid* không được quy giản thành một phép đo
riêng cho bước liên kết hoặc hiệu chỉnh hậu xử lý.

### 2.4.4. ByteTrack and RealTime-Fast Causal Cores

*ByteTrack-Raw* sử dụng cơ chế liên kết và vòng đời quỹ đạo nguyên bản của
ByteTrack trong từng video, không áp dụng hiệu chỉnh quỹ đạo sau video. Đây là
baseline không hậu xử lý của nhánh ByteTrack.

*RealTime-Fast* là phương pháp liên kết nhân quả riêng của dự án. Phương
pháp kết hợp độ chồng lấp, độ dịch chuyển vị trí, tính nhất quán về diện tích,
tính liên tục của chuyển động và tín hiệu ngoại hình khi có thể sử dụng. Các
guard và tie-break xác định cách xử lý trường hợp cạnh tranh hoặc mơ hồ; một
track chưa được ghép tiếp tục tồn tại trong một khoảng mất quan sát giới hạn.
Mọi quyết định chỉ sử dụng khung hình hiện tại và lịch sử trước đó.

Các đại lượng dùng trong điểm ghép cặp được tính trực tiếp từ hộp dự đoán của
track và hộp detection. Với hai hộp $a,b$, tâm hộp là $c(\cdot)$, diện tích là
$|\cdot|$, kích thước khung hình là $W\times H$, và $h_a,h_b$ là các histogram
đã chuẩn hoá, các đại lượng này được xác định bởi

\[
\operatorname{IoU}(a,b)=
\frac{|a\cap b|}{|a\cup b|},\qquad
d_c(a,b)=\frac{\lVert c(a)-c(b)\rVert_2}{\sqrt{W^2+H^2}},
\]

\[
d_A(a,b)=\frac{\min\!\left(\left|\log\frac{|b|+10^{-6}}{|a|+10^{-6}}\right|,2\right)}{2},
\qquad
d_{\mathrm{app}}(h_a,h_b)=
\operatorname{clip}\!\left(1-\sum_k\sqrt{h_{a,k}h_{b,k}},0,1\right).
\]

Với track đang được quan sát và track đang ở trạng thái mất quan sát, điểm cơ
sở của một cặp hợp lệ lần lượt là

\[
C_{ij}^{\mathrm{vis}}=0.42(1-\operatorname{IoU}_{ij})
+0.22d_{c,ij}+0.26d_{\mathrm{app},ij}+0.10d_{A,ij},
\]

\[
C_{ij}^{\mathrm{lost}}=0.18(1-\operatorname{IoU}_{ij})
+0.08\min(d_{c,ij},1)+0.52d_{\mathrm{app},ij}+0.12d_{A,ij}.
\]

Đây là điểm cơ sở chưa chuẩn hóa, nên tổng hệ số $0.90$ không biểu thị một
thành phần bị thiếu. Trong nhánh visible, code định nghĩa bán kính

\[
r_i=0.08+0.22\frac{\min(m_i,60)}{60},
\]

với $m_i$ là số frame bị bỏ lỡ hiện tại của track, rồi cộng trực tiếp vào cost
theo

\[
p^{\mathrm{far}}_{ij}=\begin{cases}
1.0, & \operatorname{IoU}_{ij}<0.01\ \text{và}\ d_{c,ij}>r_i,\\
0, & \text{ngược lại}.
\end{cases}
\]

Vì vậy, $+1.0$ là một khoản cộng số thực vào matching cost; nó không làm tăng
bán kính $r_i$ và không phải một cờ Boolean. Khoản này không được tính cho
track đang mất quan sát. Cost cuối cùng còn có thể nhận penalty raw-ID hoặc che
khuất khi điều kiện tương ứng xảy ra; các guard về tính hợp lý có thể loại cặp
trước khi gán, khi đó cặp nhận sentinel $10^6$. Các biểu thức trên ghi lại phép
tính của association core do dự án triển khai. *ByteTrack-Raw* vẫn giữ cơ chế
ByteTrack nguyên bản, còn các bước hiệu chỉnh offline sử dụng quy tắc xem xét
tracklet riêng.

### 2.4.5. Offline Trajectory Repair

*Hybrid-ByteTrack* và *RF-Hybrid* bổ sung hiệu chỉnh quỹ đạo sau khi toàn bộ
video đã được xử lý. *Hybrid-ByteTrack* bắt đầu từ các tracklet do ByteTrack
tạo ra, còn *RF-Hybrid* bắt đầu từ đầu ra causal của *RealTime-Fast*. Hậu xử lý
được thực hiện theo một chuỗi bước đã đóng băng; mỗi bước có điều kiện áp
dụng riêng, nên không tồn tại một repair score chung cho toàn bộ hai phương
pháp.

Một quyết định sửa tính liên tục định danh được mô tả bằng chi phí chuyển tiếp
giữa hai hộp giới hạn,

\[
\delta(a,b)=
\frac{\lVert c(a)-c(b)\rVert_2}{\sqrt{W^2+H^2}}
+0.05\min\left(
\left|\log\frac{|b|+\varepsilon}{|a|+\varepsilon}\right|,2
\right),
\qquad \varepsilon=10^{-6}.
\]

Với hai định danh $i$ và $j$, chi phí giữ nguyên và chi phí hoán đổi tại thời
điểm $t$ lần lượt là

\[
C_{\mathrm{keep}}=
\delta(b_{i,t-1},b_{i,t})+
\delta(b_{j,t-1},b_{j,t}),
\qquad
C_{\mathrm{swap}}=
\delta(b_{i,t-1},b_{j,t})+
\delta(b_{j,t-1},b_{i,t}).
\]

Độ cải thiện được tính bởi $\Delta=C_{\mathrm{keep}}-C_{\mathrm{swap}}$. Candidate
chỉ được chấp nhận khi $\Delta>0.015$ và vượt qua các guard về overlap,
visibility hoặc occlusion; trong trường hợp không có bằng chứng liên tục bổ
sung, code yêu cầu mức cải thiện ít nhất $0.030$. Phép tính này chỉ thuộc
identity-swap guard của các nhánh offline, không phải hàm mục tiêu chung cho
mọi bước.

Trong các phương trình trên, $t-1$ và $t$ lần lượt chỉ hai bản ghi frame liên
tiếp trong chuỗi đầu ra tracking, không chỉ hai thời điểm có detection. Những
quy tắc dùng anchor ở hai phía của một khoảng mất quan sát sử dụng chỉ số riêng,
chẳng hạn $t_-$ và $t_+$, và không được biểu diễn bởi phép tính keep--swap này.

Trong mô tả này, *Hidden* là thuộc tính visibility do tracker sinh ra và gắn
với quan sát ở cấp frame, không phải nhãn visibility được người review xác nhận
độc lập ở cấp annotation. Ở mức quy trình, một episode trước hết được đưa vào
tập ứng viên khi xuất hiện mất liên tục, overlap/occlusion, đoạn kết thúc được
tracker đánh dấu *Hidden*, phân mảnh hoặc sai lệch hình học. Candidate chỉ được
xét tiếp nếu thỏa điều kiện về khoảng frame, độ dài episode, overlap, confidence
và anchor quan sát trước hoặc sau. Bằng chứng được dùng tùy bước gồm dịch chuyển
tâm, thay đổi diện tích, overlap, visibility và các quan sát tương lai được
phép trong chế độ hậu xử lý. Quyết định cuối cùng là giữ nguyên quỹ đạo, trao
đổi định danh giữa các quỹ đạo, hiệu chỉnh bounding box hoặc giữ trạng thái
*Hidden*; các trường hợp không đủ điều kiện được giữ nguyên.

Mỗi thay đổi được ghi theo video, khoảng frame, identity trước và sau, box
trước và sau, bước tạo ra thay đổi, việc sử dụng frame tương lai và hash của
đầu vào–đầu ra. Do đó, *Hybrid-ByteTrack* và *RF-Hybrid* được đánh giá như hai
pipeline hoàn chỉnh, còn chất lượng của từng bước không được suy ra từ một
association cost duy nhất.

### 2.4.6. Development and Configuration Selection

Việc phát triển và lựa chọn cấu hình tracking được thực hiện trên một quần thể
phát triển cố định gồm 13 video. Phân tích tập trung vào che khuất, mất
detection, tái xuất hiện, phân mảnh, identity switch và thay đổi số lượng cá
thể quan sát được. Các chỉ số được khai báo gồm HOTA với DetA và AssA, IDF1,
IDSW, fragmentation, false positives, false negatives và thời lượng chịu sai
định danh; giá trị cụ thể được trình bày trong Section 3.

Cấu hình được đóng băng sau giai đoạn phát triển. Các ngưỡng và logic liên kết
không được điều chỉnh dựa trên video chưa được sử dụng trước đó. Vì vậy, kết
quả trên 13 video chỉ được diễn giải là bằng chứng trên tập phát triển và
không dùng để tuyên bố khả năng chuyển giao hoặc tổng quát hóa nếu chưa có
đánh giá độc lập.

### 2.4.7. Downstream Role and Failure Semantics

Quỹ đạo đã xác minh được dùng để tạo các vùng ảnh tập trung vào cá thể cho
phân loại hành vi, còn quỹ đạo dự đoán được dùng trong suy luận tự động và
đánh giá tracking. Identity continuity ảnh hưởng trực tiếp đến duration, bout,
transition và behavioral profile: fragmentation có thể chia một bout liên tục
thành nhiều đoạn; identity switch có thể chuyển quan sát từ cá thể này sang cá
thể khác; còn failed re-entry có thể tạo track mới cho cùng một cá thể. Các
lỗi này được phân biệt với detection errors, vì một bounding box đúng không tự
chứng minh rằng liên kết định danh cũng đúng.

## English academic thesis draft

Tracking links detections across time to form object trajectories within each
video. Because subsequent behavior statistics are aggregated by animal,
localization quality and identity continuity are evaluated as distinct aspects
of the tracking process.

### 2.4.1. Identity Semantics and Tracking Scope

The study distinguishes five related but non-equivalent concepts: detection,
annotation-level pig ID, *video-local track ID*, trajectory and biological
identity. A detection is an observation of an object in a particular frame;
an annotation-level pig ID is assigned within the corresponding annotation
data; a *video-local track ID* is assigned by the tracker to a linked sequence
within one video; a trajectory is an ordered sequence of bounding boxes and
associated times; and biological identity refers to the physical animal.

Identifiers in the annotation and tracking data are local to their source or
video. They do not establish a fixed biological identity across the six-week
recording period. This boundary is maintained when behavioral duration, bouts,
transitions and individual behavior profiles are computed.

### 2.4.2. Tracking Ground-Truth Construction

Detections were first used to generate initial trajectories. Intervals showing
possible association loss, occlusion or identity change were compared with
CVAT tracking annotations and identity-conflict records. The resulting
corrections were applied to the source data within the declared video/frame
scope and checked for provenance before the corrected version was used as
tracking ground truth.

Tracking ground truth and tracker predictions were kept separate. Verified
trajectories were used to construct behavior data and extract animal-centred
image regions, whereas predicted trajectories remained unaltered for metric
calculation and were compared only with the corresponding verified ground
truth.

### 2.4.3. Tracking Input and Common Evidence Contract

Let $v$ denote the source video, $t$ the observation time and $j$ the
detection index. Each record supplied to the tracker is represented as

\[
d_{v,t,j}=(v,f_t,\tau_t,b_{t,j},c_{t,j}),
\]

where $f_t$ is the frame index, $\tau_t$ the source timestamp, $b_{t,j}$ the
bounding box and $c_{t,j}$ the detection confidence. Records are ordered by
source time and processed independently within each video.

The valid-pen mask defines the tracking region. Missing-observation and
trajectory-termination states are distinguished from the `Hidden` visibility
attribute in the annotations. Tracker outputs retain the video, track ID,
frame index, source timestamp, bounding box and confidence required for
reconstruction and evaluation.

The frozen detector schedules differ across methods: *ByteTrack-Raw* and
*Hybrid-ByteTrack* use detections from every frame, whereas *RealTime-Fast* and
*RF-Hybrid* use detections from every second frame. Each method is evaluated as
a complete pipeline with its own association and post-processing stages.

### 2.4.4. ByteTrack and RealTime-Fast Causal Cores

*ByteTrack-Raw* uses the original ByteTrack association and trajectory
lifecycle within each video and applies no post-video trajectory revision. It
serves as the no-repair baseline for the ByteTrack branch.

*RealTime-Fast* is the project-specific causal association method. It combines
overlap, positional displacement, bounding-box area consistency, motion
continuity and appearance evidence when available. Guards and tie-break rules
determine how competing or ambiguous assignments are handled; an unmatched
track is retained only within a limited missing-observation interval. All
decisions use the current frame and preceding history.

The pairwise association quantities are computed directly from the predicted
track box and the detection box. For boxes $a$ and $b$, with centre $c(\cdot)$,
area $|\cdot|$, frame dimensions $W\times H$, and normalized histograms
$h_a,h_b$, the implementation uses

\[
\operatorname{IoU}(a,b)=
\frac{|a\cap b|}{|a\cup b|},\qquad
d_c(a,b)=\frac{\lVert c(a)-c(b)\rVert_2}{\sqrt{W^2+H^2}},
\]

\[
d_A(a,b)=\frac{\min\!\left(\left|\log\frac{|b|+10^{-6}}{|a|+10^{-6}}\right|,2\right)}{2},
\qquad
d_{\mathrm{app}}(h_a,h_b)=
\operatorname{clip}\!\left(1-\sum_k\sqrt{h_{a,k}h_{b,k}},0,1\right).
\]

For a visible track and a lost track, respectively, the base score of an
admissible pair is

\[
C_{ij}^{\mathrm{vis}}=0.42(1-\operatorname{IoU}_{ij})
+0.22d_{c,ij}+0.26d_{\mathrm{app},ij}+0.10d_{A,ij},
\]

\[
C_{ij}^{\mathrm{lost}}=0.18(1-\operatorname{IoU}_{ij})
+0.08\min(d_{c,ij},1)+0.52d_{\mathrm{app},ij}+0.12d_{A,ij}.
\]

These coefficients form an unnormalised base score, so their sum of $0.90$ does
not indicate a missing term. In the visible-track branch, the implementation
defines

\[
r_i=0.08+0.22\frac{\min(m_i,60)}{60},
\]

where $m_i$ is the track's current number of missed frames, and adds

\[
p^{\mathrm{far}}_{ij}=\begin{cases}
1.0, & \operatorname{IoU}_{ij}<0.01\ \text{and}\ d_{c,ij}>r_i,\\
0, & \text{otherwise}.
\end{cases}
\]

Thus, $+1.0$ is a direct scalar addition to the matching cost; it does not
increase $r_i$ and is not a Boolean flag. This term is not evaluated for lost
tracks. The final cost may also include a conditional raw-identity or
occlusion penalty, while plausibility guards can reject a pair before
assignment and return the sentinel cost $10^6$. The expressions therefore
define the pairwise score used by the project-specific causal association
core. *ByteTrack-Raw* retains the original ByteTrack mechanism, while offline
revision stages apply separate tracklet-review rules.

### 2.4.5. Offline Trajectory Repair

*Hybrid-ByteTrack* and *RF-Hybrid* add trajectory revision after the complete
video has been processed. *Hybrid-ByteTrack* starts from ByteTrack tracklets,
whereas *RF-Hybrid* starts from the causal output of *RealTime-Fast*. The
post-processing steps are applied in a frozen order. Each step has its own
eligibility rule; the two methods therefore do not minimise a shared repair
score.

For an identity-continuity decision, the implementation uses the transition
cost

\[
\delta(a,b)=
\frac{\lVert c(a)-c(b)\rVert_2}{\sqrt{W^2+H^2}}
+0.05\min\left(
\left|\log\frac{|b|+\varepsilon}{|a|+\varepsilon}\right|,2
\right),
\qquad \varepsilon=10^{-6}.
\]

For identities $i$ and $j$ at time $t$, the costs for retaining and exchanging
their assignments are

\[
C_{\mathrm{keep}}=
\delta(b_{i,t-1},b_{i,t})+
\delta(b_{j,t-1},b_{j,t}),
\qquad
C_{\mathrm{swap}}=
\delta(b_{i,t-1},b_{j,t})+
\delta(b_{j,t-1},b_{i,t}).
\]

The improvement is $\Delta=C_{\mathrm{keep}}-C_{\mathrm{swap}}$. A candidate
exchange must satisfy $\Delta>0.015$ and pass the overlap, visibility and
occlusion guards; when no additional continuity evidence is present, the
implementation requires an improvement of at least $0.030$. This expression
belongs to the offline identity-swap guard and is not a common objective for
all post-processing stages.

In these equations, $t-1$ and $t$ denote successive frame records in the
tracking-output sequence rather than successive detector events. Rules that
compare anchors on the two sides of a temporal gap use separate indices, such
as $t_-$ and $t_+$; they are not represented by this keep--swap calculation.

Here, *Hidden* denotes the tracker-generated visibility attribute exported with
each frame-level observation; it is not a separately adjudicated
annotation-level visibility label. Operationally, an episode enters
consideration when it contains a continuity break, overlap or occlusion, a
terminal segment marked *Hidden* by the tracker, fragmentation, or anomalous
box geometry. Eligibility is then determined from frame-gap and episode-length
limits, overlap, confidence, and the availability of an observed anchor before
or after the episode. Depending on the step, the evidence consists of centre
displacement, area change, overlap, visibility, and permitted future anchors.
The resulting action is to retain the trajectory, exchange track identities,
refine a bounding box, or retain the *Hidden* state; candidates that fail the
rules remain unchanged.

Each modification records the video, frame interval, identities and boxes
before and after revision, the responsible revision step, future-frame usage, and
input/output hashes. *Hybrid-ByteTrack* and *RF-Hybrid* are therefore evaluated
as complete pipelines, while no individual post-processing step is interpreted
as a universal association cost.

### 2.4.6. Development and Configuration Selection

Tracking configurations were developed and selected on a fixed development
population of 13 videos. The analysis focused on occlusion, missed detections,
re-entry, fragmentation, identity switches and changes in the number of visible
animals. The declared metrics included HOTA with DetA and AssA, IDF1, IDSW,
fragmentation, false positives, false negatives and the duration exposed to
identity errors; numerical values are reported in Section 3.

The configuration was frozen after development. The 13-video population is
reported as development evidence, whereas claims of transfer or generalisation
require an independent evaluation.

### 2.4.7. Downstream Role and Failure Semantics

Verified trajectories provide the animal-centred image regions used for
behavior classification, whereas predicted trajectories are used for automatic
inference and tracking evaluation. Identity continuity directly affects
duration, bout, transition and behavioral-profile statistics: fragmentation can
split one continuous bout into several segments; an identity switch can assign
observations from one animal to another; and failed re-entry can create a new
track for the same animal. These errors are distinguished from detection
errors, because a correctly localized bounding box does not by itself establish
correct identity association.

## Visual anchor

No separate figure is required for this section at the current stage. A
tracking example should be added only if it provides evidence that cannot be
communicated adequately through the method description or evaluation tables.

## Drafting sources (working note; not manuscript prose)

- `docs/thesis_drafts/PIG_BEHAVIOR_THESIS_MASTER_OUTLINE_V2.md`
- `docs/TRACKING_ID_PIPELINE.md`
- `docs/tracking/reconciliation/FOUR_METHOD_TRACKING_FREEZE_AUTHORITY_20260729.json`
- `docs/tracking/reconciliation/STATE_8_DEVELOPMENT_EVALUATION_AUTHORITY_20260729.json`
- `src/pig_behavior/tracking/`
- Tracking ground truth under `data/annotations/tracking/`

## Draft status (working note)

- Identity semantics, causal/offline distinction and the four-method structure:
  `USER-CONFIRMED` for this revision.
- Tracking ground truth and corrected-source lineage: `PROTOCOL`; quantitative
  claims remain tied to the declared video/frame population and evaluator.
- Development population: 13 videos in the frozen development authority;
  detailed metrics and configuration comparisons belong to Section 3.
- Visual: no separate figure required at this stage; a tracking example is
  conditional on a clear evidentiary need.

## Editorial status (working note)

The accepted Vietnamese structure has been rewritten as original academic
English. The superseded English block was removed so that this section contains
only the current method description. Quantitative tracking values and detailed
configuration comparisons remain reserved for Section 3.
