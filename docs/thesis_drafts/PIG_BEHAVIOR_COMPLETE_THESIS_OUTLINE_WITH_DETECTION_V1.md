# PIG Behavior Project — Consolidated Thesis Outline with Detection

**Ngôn ngữ làm việc:** tiếng Việt; tiêu đề mục tiếng Anh được giữ trong ngoặc
để thuận tiện chuyển sang bản thesis cuối.

**Trạng thái:** outline đề xuất, chưa phải nội dung chương hoàn chỉnh.

**Thesis title:** *An AI Spatio-Temporal Framework for Dynamic Behavior
Profiling and Anomaly Detection in Group-Housed Pigs*

**Phạm vi của tài liệu:** Tài liệu này hợp nhất cấu trúc thesis và phân công
evidence cho detection, tracking, behavior classification, posture và
behavioral-deviation screening. Tài liệu không tạo kết quả mới, không thay thế
blueprint hoặc authority artifact, và không sửa các draft Chapter 2 hiện có.

## A. Quyết định cấu trúc tổng thể (Executive structural decision)

### A.1. Cấu trúc được khuyến nghị

Thesis nên dùng bốn khối chính sau phần đầu: **Introduction**, **Methodology**,
**Experiment**, và **Conclusion and Future Work**, tiếp theo là **References**
và **Appendices**. Cách tổ chức này giữ được trình tự của bài mẫu nhưng điều
chỉnh cho một hệ thống thị giác máy tính nhiều tầng, trong đó behavior
classification vẫn là thành phần khoa học trung tâm.

Detection được đặt trong **Section 2.3 — Detection Dataset Construction and
Pig Detection**, cùng một section với quy trình tạo dữ liệu detection và mô tả
detector. Đây là lựa chọn **Option A**. Evidence hiện có cho thấy frame
selection, annotation và model-assisted pre-annotation đủ quan trọng để có
phương pháp và thực nghiệm riêng, nhưng chưa cần tách thành hai major section
vì sẽ làm đứt mạch từ dữ liệu đầu vào đến detector output. Kết quả detection
được báo cáo riêng ở Section 3.4.

Behavior classification vẫn giữ vị trí trung tâm vì đây là nơi đầu vào RGB của
cá thể, hình học, chuyển động, ROI và ngữ cảnh xã hội được kết hợp thành nhãn
hành vi theo chuỗi thời gian. Detection chỉ locates pigs; tracking duy trì
trajectory và identity continuity; profiling và deviation screening sử dụng
các dự đoán đã có định danh.

Tracking được xem là một upstream scientific component riêng, không phải câu
“mô hình dùng một tracker tiêu chuẩn”. Nếu detection bị bỏ sót hoặc association
gán sai identity, trajectory sẽ bị phân mảnh hoặc đổi cá thể; khi đó behavior
prediction có thể được gán cho sai actor và làm sai duration, frequency, bout,
transition và deviation screening. Vì vậy, thesis phải đánh giá tracking riêng
về identity continuity, long-term stability, causal/offline semantics và cost,
đồng thời phân biệt rõ tracking-generated `track_id` với biological identity.

### A.2. Ranh giới khoa học phải giữ xuyên suốt thesis

- Dữ liệu nguồn có thể gồm RGB và depth, nhưng nhánh hiện hành của behavior
  model chỉ được mô tả với các đầu vào RGB-derived đã được chứng minh. Depth là
  **future work** hoặc một experiment riêng nếu sau này có ablation hợp lệ.
- `behavioral deviation screening` hoặc `potential behavioral anomaly
  detection` là tên phù hợp cho tầng downstream. Đây không phải lớp
  `abnormal` ở cấp frame, không phải supervised anomaly classifier có clinical
  ground truth, và không phải chẩn đoán bệnh, stress, chấn thương hay welfare.
- Posture (`lying`, `sitting`, `standing`) là một experiment chính. Tuy nhiên,
  kết quả chỉ được viết sau khi posture authority và grouped evaluation được
  đăng ký.
- Dữ liệu behavior lịch sử được hợp nhất với nguồn dữ liệu mới để tạo dataset
  đưa vào training. Legacy được gọi là nguồn bổ sung vì mở rộng độ đa dạng theo
  ngày và video, không phải vì bị tách khỏi training dataset. Không được viết
  “mỗi video có hai burst” nếu chưa tính trực tiếp từ manifest.

## B. Mục lục thesis đề xuất (Complete numbered table of contents)

### Front matter

- Acknowledgement
- Abstract
- List of Figures
- List of Tables
- List of Abbreviations

### Chapter 1. Introduction

1.1. Motivation and Research Context (*Motivation and Research Context*)  
1.2. Problem Definition and Scope (*Problem Definition and Scope*)  
1.3. Research Gap and Related Work (*Research Gap and Related Work*)  
1.4. Research Questions (*Research Questions*)  
1.5. Contributions and Contribution Boundaries (*Contributions and Boundaries*)  
1.6. Thesis Organization (*Thesis Organization*)

Các câu hỏi nghiên cứu chính được giữ ở mức có thể kiểm tra bằng evidence của
thesis:

- **RQ1.** How can a reproducible and leakage-safe learning dataset be
  constructed from heterogeneous pig-video annotations?
- **RQ2.** How can individual identity continuity be maintained under occlusion,
  re-entry, and variable visible-pig counts, and how do causal and offline
  tracking modes differ in quality and processing cost?
- **RQ3.** Which visual, geometric, motion, ROI, and social-context signals
  contribute to ten-class behavior recognition?
- **RQ4.** Can an independent posture target improve the interpretation of
  behavior states and support posture-aware error analysis?
- **RQ5.** How can identity-conditioned behavior profiles be used for
  behavioral deviation screening without presenting the result as a supervised
  diagnosis?

### Chapter 2. Methodology

2.1. Overview of the Proposed Framework (*Overview of the Proposed Framework*)  
2.2. Data Sources and Data Representation (*Data Sources and Data Representation*)  
2.3. Detection Dataset Construction and Pig Detection (*Detection Dataset
Construction and Pig Detection*)  
&nbsp;&nbsp;2.3.1. Background Reference and Valid-Pen Mask  
&nbsp;&nbsp;2.3.2. Activity-Guided Candidate Selection  
&nbsp;&nbsp;2.3.3. Candidate Ranking within Source-Time Windows  
&nbsp;&nbsp;2.3.4. Average-Hash Filtering  
&nbsp;&nbsp;2.3.5. Temporal and Video/Day Balancing  
&nbsp;&nbsp;2.3.6. Fallback and Emergency Fill  
&nbsp;&nbsp;2.3.7. Roboflow Annotation and Manual Bounding-Box QC  
&nbsp;&nbsp;2.3.8. Historical YOLO-Assisted Pre-Annotation  
&nbsp;&nbsp;2.3.9. Grouped Split Assignment  
&nbsp;&nbsp;2.3.10. Duplicate and Leakage Audits  
&nbsp;&nbsp;2.3.11. Detector Architecture and Training  
&nbsp;&nbsp;2.3.12. Detector Output Contract  
&nbsp;&nbsp;2.3.13. Detection Failure Categories  
2.4. Identity Tracking (*Identity Tracking*)  
&nbsp;&nbsp;2.4.1. Identity Semantics and Track Lifecycle  
&nbsp;&nbsp;2.4.2. Tracking Ground Truth and Evaluation Population  
&nbsp;&nbsp;2.4.3. Tracking Modes and Association Profiles  
&nbsp;&nbsp;2.4.4. Causal, Bounded-Lag, and Offline Semantics  
&nbsp;&nbsp;2.4.5. Fair Comparison and Detector-Evidence Boundary  
&nbsp;&nbsp;2.4.6. Tracking Baselines and Factorial Comparison  
&nbsp;&nbsp;2.4.7. Tracking Metrics and Identity-Error Episodes  
&nbsp;&nbsp;2.4.8. Long-Term Tracking Stability  
&nbsp;&nbsp;2.4.9. Runtime and Deployment Cost  
&nbsp;&nbsp;2.4.10. Downstream Error Propagation
2.5. Behavior Data Construction, Annotation, and Human Review (*Behavior Data
Construction, Annotation, and Human Review*)  
&nbsp;&nbsp;2.5.1. Current Behavior Source and Ten-Class Ontology  
&nbsp;&nbsp;2.5.2. Historical Burst-Selection Pathway and Legacy Cohort  
&nbsp;&nbsp;2.5.3. Human Review and Residual Controls  
2.6. Corrected-Source and Identity Lineage (*Corrected-Source and Identity
Lineage*)  
2.7. Feature Construction (*Feature Construction*)  
2.8. Temporal Sequence Construction and Leakage Controls (*Temporal Sequence
Construction and Leakage Controls*)  
2.9. Behavior and Posture Model Architecture (*Behavior and Posture Model
Architecture*)  
2.10. Behavior Profiling and Behavioral Deviation Screening (*Behavior
Profiling and Behavioral Deviation Screening*)  
2.11. Reproducibility and Implementation Environment (*Reproducibility and
Implementation Environment*)

### Chapter 3. Experiment

3.1. Dataset Summary and Grouped Split Design (*Dataset Summary and Split Design*)  
3.2. Experimental Settings and Baselines (*Experimental Settings and Baselines*)  
3.3. Evaluation Metrics and Statistical Reporting (*Evaluation Metrics*)  
3.4. Pig Detection Results (*Pig Detection Results*)  
3.5. Tracking Results (*Tracking Results*)  
&nbsp;&nbsp;3.5.1. Evaluation Population and Fairness Checks  
&nbsp;&nbsp;3.5.2. Aggregate and Per-Video Tracking Quality  
&nbsp;&nbsp;3.5.3. Causal/Offline Quality and Processing Cost  
&nbsp;&nbsp;3.5.4. Long-Term Identity Stability  
&nbsp;&nbsp;3.5.5. Tracking Failure and Downstream Propagation
3.6. Behavior Classification Results (*Behavior Classification Results*)  
3.7. Posture Experiment (*Posture Experiment*)  
3.8. Ablation and Robustness Analysis (*Ablation and Robustness Analysis*)  
3.9. Qualitative Examples and Error Analysis (*Qualitative Examples and Error
Analysis*)  
3.10. Long-Term Behavior Profiling and Deviation Screening (*Long-Term
Profiling and Deviation Screening*)

### Chapter 4. Conclusion and Future Work

4.1. Conclusions (*Conclusions*)  
4.2. Limitations (*Limitations*)  
4.3. Future Work (*Future Work*)

### References and Appendices

- References
- Appendix A. Data and annotation schemas
- Appendix B. Review and corrected-source manifests
- Appendix C. Detector and model configuration
- Appendix D. Additional grouped evaluation tables
- Appendix E. Reproducibility checklist

## C. Hợp đồng viết theo từng section (Section-by-section writing contract)

Các trạng thái evidence dùng trong bảng là: `ESTABLISHED_SOURCE_FACT`,
`ESTABLISHED_PROJECT_PROTOCOL`, `PROVISIONAL_RESULT`,
`FINAL_RESULT_REQUIRES_ARTIFACT`, `FUTURE_WORK`, và
`UNSUPPORTED_AND_MUST_BE_REMOVED`.

| Section | Purpose | Must include | Must not include | Evidence | Visual | Cross-reference |
|---|---|---|---|---|---|---|
| Front matter | Cung cấp thông tin thesis và các danh mục | Abstract, figures, tables, abbreviations | Kết quả chưa có authority | Quy định của trường và nội dung final | List of Figures/Tables | Toàn thesis |
| 1.1 | Nêu động lực giám sát lợn nuôi theo nhóm | Tầm quan trọng của identity, temporal behavior và profiling | Tuyên bố chẩn đoán bệnh | Literature và study context | Figure 1 | 2.1, 2.10 |
| 1.2 | Định nghĩa bài toán và giới hạn | Detection, tracking, classification, posture, screening | Cross-farm/general clinical claims | Blueprint, charter | Figure 1 | 2.1, 4.2 |
| 1.3 | Đặt nghiên cứu trong related work | Detection/tracking/behavior literature và hai bài dùng cùng data | Sao chép taxonomy hoặc claim novelty tuyệt đối | DOI-verified sources | Bảng related work nếu cần | 1.5 |
| 1.4 | Chuyển gap thành câu hỏi nghiên cứu | RQ1 dataset/leakage, RQ2 identity tracking, RQ3 multimodal behavior, RQ4 posture, RQ5 profiling/screening | RQ không thể kiểm tra bằng dữ liệu hiện có | Blueprint, tracking 5A, user-confirmed scope | Không bắt buộc | 2.3–2.10, 3.x |
| 1.5 | Nêu đóng góp cá nhân và ranh giới kế thừa | Data workflow, review, tracking, features, model, evaluation | Nhận toàn bộ dữ liệu/taxonomy là mới | Blueprint và lineage | Contribution diagram nếu cần | 2.3–2.11 |
| 1.6 | Hướng dẫn người đọc qua thesis | Logic chapter và quan hệ giữa methodology/experiment | Mô tả kết quả | Cấu trúc đã duyệt | Không bắt buộc | Các chapter |
| 2.1 | Trình bày flow end-to-end ở mức khái quát | RGB video, detection, tracks, sequences, model, two downstream branches | Threshold sampling, final metrics, depth branch hiện hành | Draft 2.1, blueprint | Figure 2 | 2.2–2.10 |
| 2.2 | Mô tả nguồn dữ liệu và quy ước thời gian | Pen/camera, group size biến thiên, 1800 frames, source 6 fps, playback 30 fps | Lặp thuật toán detection hoặc kết quả training | Draft 2.2, dataset papers, timestamps | Figures 3–4; Table 1 | 2.3, 3.1 |
| 2.3 | Giải thích cách tạo detection data và detector | Background/mask, activity, aHash, balancing, annotation, split, YOLO contract | Detector metrics cuối cùng và tracking score | Notebooks, manifests, training artifacts | Figures 6–7; Tables 2–5 | 3.4 |
| 2.3.1 | Xác định vùng nền và vùng chuồng hợp lệ | Empty-pen/background reference, valid-pen mask | Hứa hẹn loại bỏ mọi nhiễu | Notebook và mask artifact | Figure 6 | 2.2 |
| 2.3.2 | Tạo activity score | Frame-to-frame và frame-to-background difference | Gọi difference là hash | Notebook formula/config | Figure 6 | 2.3.3 |
| 2.3.3 | Xếp hạng candidate theo source time | One-second windows và candidate rank | Exact final count khi chưa có manifest | Candidate CSV | Figure 6 | 3.4 |
| 2.3.4 | Lọc gần trùng lặp | 64-bit aHash và Hamming distance | Gọi là pHash nếu code không dùng pHash | Hash implementation/config | Figure 6–7 | 3.1 |
| 2.3.5 | Giữ coverage theo thời gian/video/ngày | Gap, leaf/video cap, trigger/ROI balancing | Suy ra độc lập ngoài split authority | Sampling manifest/seed | Table 2 | 3.1 |
| 2.3.6 | Xử lý thiếu target candidate | Fallback/emergency-fill và provenance | Cho rằng fallback có cùng phân bố | Fill log | Không bắt buộc | 3.4 |
| 2.3.7 | Tạo box ground truth cho detection | Roboflow export, manual correction, QC | Dùng box scaffold như final label | Export/QC manifest | Figure 7 | 3.4 |
| 2.3.8 | Ghi nhận pre-annotation lịch sử | YOLO-assisted scaffold và giới hạn semantics | Gọi default `lying` là behavior truth | Notebook và weight path | Appendix C | 2.5 |
| 2.3.9 | Gán split theo group | Day/session/leaf video policy | Finalize split từ comment notebook | Split manifest | Table 3 | 3.1 |
| 2.3.10 | Kiểm tra leakage | Same burst, neighboring interval, hash, leaf-video overlap | Tuyên bố zero leakage thiếu audit | Leakage report | Table 3 | 3.1 |
| 2.3.11 | Khóa detector training | YOLO version/scale, initialization, augmentation, optimizer | Invent missing hyperparameters | Config, weights, logs | Table 5 | 3.2, 3.4 |
| 2.3.12 | Định nghĩa output dùng cho downstream | Box schema, confidence/NMS, hand-off to tracker | Gộp detector và tracker metric | Pipeline config | Figure 2 | 2.4 |
| 2.3.13 | Định nghĩa error taxonomy | Occlusion, crowding, low density, empty pen, FP/FN | Chọn lỗi không có source frame | Prediction/error manifest | Figure 7 | 3.4, 3.9 |
| 2.4 | Mô tả identity continuity | Detector output, association, lifecycle, causal/offline modes, evaluation and downstream role | Gọi detection score là tracking score hoặc biological identity | Tracking authority, code/config | Figures 8–12; Tables 6–13 | 3.5 |
| 2.4.1 | Định nghĩa các loại identity | Detection index, annotation-local `pig_id`, `track_id`, trajectory và biological identity | Đồng nhất `track_id` với identity sáu tuần | Tracking schema and source metadata | Figure 9 | 2.2 |
| 2.4.2 | Xác định population và ground truth | CVAT tracking annotations, frames, Hidden/occlusion, included/excluded trajectories, hashes | Trộn behavior XML với tracking authority | Tracking GT manifest and video hashes | Table 6 | 3.5.1 |
| 2.4.3 | Mô tả tracking modes | Detector input/cadence, association, motion/appearance, lifecycle, lost/re-entry handling | Promote profile name chưa có final authority | Registered tracker profiles | Table 7 | 3.5.3 |
| 2.4.4 | Phân biệt causal và offline | Current/past-only, bounded lag if any, future evidence and output delay | Gọi offline repair là zero-delay real time | Mode contract and repair audit | Figure 10 | 3.5.3 |
| 2.4.5 | Khóa fairness boundary | Detector, cadence, mask, GT, Hidden policy, evaluator, thresholds, config and code SHA | Gọi pipeline difference là pure association effect | Fairness manifest | Table 8 | 3.5.1 |
| 2.4.6 | Đặt baseline/factorial comparison | Core effect, repair effect and interaction when all arms have authority | Ép ma trận 2×2 khi thiếu arm hợp lệ | Registered comparison design | Table 7 | 3.5.2 |
| 2.4.7 | Định nghĩa tracking metrics | DetA/MOTA/MOTP, HOTA/AssA, IDF1, IDP/IDR, IDSW, fragmentation, episode errors | Chỉ dùng aggregate IDSW | Evaluator contract | Tables 9–10 | 3.5.2 |
| 2.4.8 | Phân tích thời lượng tracking | Short, medium, long concatenated videos; re-entry, swaps, generated IDs | Suy long-term reliability từ clip một phút | Duration-stratified evaluation | Figure 11 | 3.5.4 |
| 2.4.9 | Đo cost triển khai | Latency, throughput, detector/association/repair cost, hardware and repeats | So sánh GPU loads không kiểm soát | Runtime manifest | Figure 12; Table 12 | 3.5.3 |
| 2.4.10 | Đo lan truyền lỗi xuống behavior | Identity authority alternatives and profile error | Giả định tracking errors không ảnh hưởng profiling | Downstream propagation experiment or limitation | Table 13 | 3.5.5 |
| 2.5 | Mô tả behavior source, legacy path và human review | Current review authority, ten classes, historical burst selection, controls | Gộp legacy khi chưa kiểm tra mapping/leakage | CVAT/review artifacts, notebooks | Figure 5; Table 14 | 2.6, 3.1 |
| 2.5.1 | Khóa behavior source hiện hành | Ten-class ontology và actor/time linkage | Dùng legacy label thay current authority | CVAT/review source | Table 13 | 2.6 |
| 2.5.2 | Đặt legacy burst đúng vai trò | Nhiều ngày/video, trigger/ROI, merged training source và temporal diversity | Khẳng định hai burst/video | Phase-2 notebook và legacy manifest | Figure 5; Table 14 | 3.1, 3.8 |
| 2.5.3 | Mô tả human review | Review scope, controls, corrections, residual checks | Đưa review metadata vào model-X | Review-close authority | Figure 5; Table 14 | 2.6 |
| 2.6 | Bảo toàn source và correction lineage | Input label, review decision, corrected source, hashes | Che giấu superseded authority conflict | Review-close and corrected-source artifacts | Figure 5 | 2.5, 2.8 |
| 2.7 | Định nghĩa các feature model nhận vào | RGB actor, geometry, motion, ROI, social, temporal | Depth contribution nếu chưa đánh giá | Feature whitelist/config | Figure 13; Table 18 | 2.9, 3.8 |
| 2.8 | Xây dựng sequence và chống leakage | 6/8/12/16 windows, sampled-six, grouped day/video split | Lặp lịch sử burst như final window semantics | Sequence manifests, split audits | Timeline schematic | 3.1, 3.8 |
| 2.9 | Mô tả behavior và posture model | Heads, fusion, loss, training contract | Kết quả metric chưa đăng ký | Code/config/model authority | Figure 13; Table 5 | 3.6–3.8 |
| 2.10 | Định nghĩa profile và screening | Duration, frequency, bout, transitions, baseline, alert wording | Accuracy/sensitivity disease claim without anomaly GT | Profile contract, source-time rule | Figures 15–16; Table 16 | 3.10, 4.2 |
| 2.11 | Khóa reproducibility | Code SHA, config, seed, data snapshot, environment | Claim reproducibility khi thiếu hash | Run manifests and project rules | Reproducibility checklist | Appendices |
| 3.1 | Báo cáo composition và split thực nghiệm | Counts, recording days, groups, merged new/legacy composition | Exact counts khi snapshot chưa frozen | Final snapshot/split artifacts | Figure 7; Tables 2–3 | 2.2, 2.5, 2.8 |
| 3.2 | Định nghĩa settings/baselines | Detector, tracker, model variants, seeds | So sánh không matched hoặc không cùng split | Registered configs | Table 5 | 3.4–3.8 |
| 3.3 | Định nghĩa metric và uncertainty | Detection, tracking, behavior, posture, grouped intervals | Dùng playback fps làm biological time | Evaluator contract | Metric table | 3.4–3.10 |
| 3.4 | Báo cáo detector | Counts, split-day composition, P/R/mAP, failures | Cross-farm/generalization claim | Predictions, evaluator, manifest, config | PR curve, qualitative panels | 2.3 |
| 3.5 | Báo cáo tracking | Identity metrics, per-video/long-duration quality, causal/offline cost and downstream propagation | Đổ lỗi detector weight hoặc claim perfect identity | Tracking freeze/evaluator | Figures 9–12; Tables 6–13 | 2.4 |
| 3.5.1 | Kiểm tra population/fairness | Source videos, GT, detector evidence, cadence, thresholds and hashes | So sánh không matched | Tracking evaluator and fairness manifest | Table 6; Table 8 | 2.4.2–2.4.5 |
| 3.5.2 | Báo cáo quality | Aggregate, per-video, HOTA/AssA/IDF1, ID errors, fragmentation and worst video | Chỉ báo cáo IDSW aggregate | Registered predictions/evaluator | Figures 11; Tables 9–10 | 2.4.6–2.4.7 |
| 3.5.3 | So sánh causal/offline và cost | Quality, delay, throughput, detector/association/repair timing | Gọi future-frame method real-time | Runtime and mode manifests | Figure 10; Figure 12; Table 12 | 2.4.3–2.4.5, 2.4.9 |
| 3.5.4 | Phân tích long-term stability | Short/medium/long duration, re-entry, fragmentation, terminal swaps | Suy từ one-minute playback | Duration-stratified evaluator | Figure 11; Table 10 | 2.4.8 |
| 3.5.5 | Phân tích failure/downstream effect | Occlusion, re-entry, duplicates, repair errors, profile divergence | Bỏ qua propagation xuống behavior | Qualitative episodes and propagation artifact | Table 13 | 2.4.10, 3.8–3.10 |
| 3.6 | Báo cáo behavior classifier | Overall/per-class, confusion, grouped results | Final claim trước corrected-source rebuild | Registered predictions | Figure 14; Table 15 | 2.9 |
| 3.7 | Báo cáo posture experiment | Three posture classes, confusion, ambiguity | Suy posture tự động từ mọi behavior | Posture authority/evaluator | Figure 14; Table 15 | 2.9, 4.2 |
| 3.8 | Kiểm tra contribution bằng ablation/robustness | One scientific family per ablation, cohort boundary | Cross-pen transfer chưa có authority | Frozen configs and evaluators | Ablation plot | 2.7–2.9 |
| 3.9 | Phân tích lỗi có provenance | Occlusion, crowding, low density, transitions, label ambiguity | Chọn case theo reviewer notes hoặc fabricate | Review-independent predictions | Qualitative panels | 3.4–3.8 |
| 3.10 | Đánh giá application layer | Profiles, baseline, screening signals, time basis | Gọi screening là diagnosis/classification | Profile artifact and rule | Figures 15–16 | 2.10, 4.2 |
| 4.1 | Kết luận theo claim registry | Chỉ claim admitted | Claim vượt single-pen scope | Final registry | Không bắt buộc | Toàn thesis |
| 4.2 | Nêu giới hạn và tác động | Single pen/camera, no anomaly GT, lineage/split limits | Giấu thiếu hụt evidence | Charter, final authority | Table 8 | 2.10, 3.x |
| 4.3 | Đề xuất future work | Depth, cross-pen, anomaly labels, real-time extension | Trình bày kế hoạch như kết quả | Open questions and limitations | Roadmap | 2.11 |
| References | Ghi nguồn đã sử dụng | Dataset papers, method papers, project citations | Citation chưa xác minh | DOI/source records | Không bắt buộc | Các section |
| Appendices | Lưu schema và audit detail | Manifests, configs, checklists | Dùng appendix để thay thế phương pháp chính | Immutable artifacts | Supplemental tables | Relevant sections |

## D. Bản đồ detection chi tiết (Detailed detection map)

| Detection topic | Methodology location | Experiment location | Evidence needed | Claim boundary |
|---|---|---|---|---|
| Raw long-duration videos | 2.2 và 2.3 | 3.1 | Source manifest, video keys, timestamps | Chỉ mô tả population/pen hiện có |
| Empty-pen/background reference | 2.3.1 | 3.9 nếu phân tích lỗi | Notebook code, saved background/mask artifact | Background subtraction là bước chọn dữ liệu, không tự động là novel algorithm |
| Valid-pen or scene mask | 2.3.1 | 3.9 | Mask artifact và code hash | Không khẳng định mask loại bỏ mọi false positive nếu chưa audit |
| Frame-to-frame difference | 2.3.2 | 3.4/3.9 | Notebook cell, config, candidate manifest | Chỉ là một thành phần activity score |
| Frame-to-background difference | 2.3.2 | 3.4/3.9 | Notebook cell, background artifact | Không gọi là image hashing |
| Activity score and source-time windows | 2.3.2 | 3.1/3.4 | Formula, source-time convention, manifest | Không dùng playback 30 fps để diễn giải biological duration |
| Candidate ranking | 2.3.3 | 3.4 | Candidate CSV and ranking parameters | Exact retained count cần manifest cuối |
| Average hashing (aHash) | 2.3.4 | 3.4/3.9 | 64-bit aHash implementation, Hamming thresholds | Không gọi là pHash/perceptual hash nếu code chỉ dùng aHash |
| Temporal-gap and leaf/video gap | 2.3.5 | 3.1 | Config and candidate-selection log | Không suy ra độc lập giữa các ngày nếu split chưa audit |
| Balancing across videos, ROI and triggers | 2.3.5 | 3.1/3.8 | Sampling seed, candidate manifest | Không viết “hai burst/video” trước khi đếm manifest |
| Fallback/emergency fill | 2.3.6 | 3.4 | Emergency-fill log and final manifest | Fallback không chứng minh candidate có cùng diversity |
| Roboflow/manual bbox annotation | 2.3.7 | 3.4 | Export metadata, QC log, label manifest | Exact image/box counts là `FINAL_RESULT_REQUIRES_ARTIFACT` |
| Historical YOLO-assisted pre-annotation | 2.3.8 | 3.4 or Appendix C | Notebook, weight path, conf/IoU settings | Đây là historical scaffold, không phải behavior ground truth cuối |
| Day/session grouping | 2.3.9 | 3.1/3.4 | Split manifest and day map | Chỉ được viết “mỗi recording day thuộc một partition” khi manifest xác nhận |
| Same-burst/neighboring-frame exclusion | 2.3.9 | 3.1 | Source-time overlap audit | Không xem hash-only check là đủ leakage audit |
| Duplicate/near-duplicate cross-split audit | 2.3.10 | 3.1 | Hash audit report and thresholds | Exact “zero leakage” cần artifact, không lấy từ notebook comment |
| Detector architecture and training | 2.3.11 | 3.2/3.4 | YOLO version, scale, weights, config, seed, logs | Không invent version, resolution, augmentation hoặc optimizer |
| Detection output contract | 2.3.12 | 2.4/3.4 | Box schema, confidence/NMS config | Detector score không đồng nghĩa identity continuity |
| Detection failure analysis | 2.3.13 | 3.4/3.9 | Review-independent predictions | Phải tách occlusion, crowded, low-density, empty-pen, FP/FN |
| Downstream detector use | 2.3.12 | 3.5–3.6 | Pipeline config and lineage | Không gộp detector, tracker và classifier thành một metric |

### D.1. Nội dung thực tế của detection-data construction

Section 2.3 nên mô tả pipeline theo thứ tự nhân quả, không theo lịch sử file:

1. Video dài được chia theo source-time windows. Một empty-pen hoặc background
   reference cùng valid-pen mask được dùng để giới hạn vùng có ý nghĩa.
2. Activity score kết hợp khác biệt frame-hiện-tại với frame-trước và khác biệt
   frame-hiện-tại với background. Các candidate được xếp hạng trong từng cửa
   sổ thời gian nguồn.
3. Candidate được lọc bằng 64-bit average hash và khoảng cách Hamming, sau đó
   áp dụng temporal gap, giới hạn theo leaf/video và cân bằng theo video, ROI
   hoặc recording day khi cấu hình đó có trong manifest.
4. Nếu target chưa đạt, emergency-fill được ghi nhận như một stage riêng, vì
   nó có thể có phân bố activity khác với candidate chính.
5. Ảnh giữ lại được annotate/correct trong Roboflow hoặc workflow tương đương;
   manual bounding-box QC và export manifest là ranh giới kết thúc data
   construction.

Các chi tiết về candidate selection là **ESTABLISHED_PROJECT_PROTOCOL** khi
được notebook và manifest hỗ trợ. Chúng không được trình bày như một detector
algorithm mới nếu chưa có baseline hoặc ablation đối chứng.

### D.1.1. Historical model-assisted annotation boundary

Notebook `video_to_frame_annotate` là một nhánh lịch sử để tạo
pre-annotation/tracking scaffold. Nó dùng một YOLO weight tại runtime với
confidence `0.25`, IoU `0.7`, lấy mẫu khoảng 1 Hz, giới hạn ở 40 video directory
đầu tiên của một ngày, global scene mask, và association kết hợp IoU với HSV
appearance histogram. `KEEPALIVE_MAX=30`, `MIN_HITS_FOR_ID=5` và giới hạn tối
đa tám ID/video được dùng trong scaffold; output có COCO annotation và mặc định
gán `Behavior="lying"`, còn track miss được đánh dấu `Hidden`.

Các giá trị này phải được ghi là **historical implementation details**, không
phải cấu hình của detector được promote cuối. Chúng chỉ được chuyển thành
methodology final nếu có training/config artifact xác nhận đúng weight, version,
split, seed và output contract. Đặc biệt, pre-annotation mặc định `lying`
không phải behavior ground truth và không được dùng để chứng minh chất lượng
behavior classifier.

### D.2. Lịch sử tạo behavior burst và vai trò của legacy data

Notebook `video_to_frame_phase_2` thể hiện một pathway lịch sử riêng với các
ngày `pigs101219a`, `pigs101219b` và `pigs111219`, các ROI feeder/drinker/toy,
burst sáu frame, stride ba frame, offsets `[-3,-2,-1,0,1,2]`, cùng các trigger
`roi_enter`, `roi_exit`, `onset`, `offset` và `speed_peak`. Pathway này dùng
foreground/background difference, speed, aspect change, blur, aHash/Hamming,
giới hạn tổng khoảng một burst mỗi phút, gap tối thiểu khoảng 10–12 giây,
post-thinning khoảng 20 giây và balanced sampling với seed 42.

Trong thesis, đây nên được mô tả là **historical behavior-data construction
pathway**. Các burst này được hợp nhất với burst từ nguồn dữ liệu mới để tạo
training dataset; vai trò “supplementary” chỉ nói rằng chúng bổ sung độ đa dạng
theo ngày, video và khoảng thời gian. Số burst thực tế phụ thuộc thời lượng
video, trigger, gap, hash filtering, sleepy policy và manifest. Vì vậy, không
được ghi một con số cố định cho mỗi video trước khi tính lại.

Việc hợp nhất legacy với nguồn mới phải kiểm tra:

- mapping taxonomy và thống nhất mười lớp hiện hành;
- label quality, identity/track provenance và source-time;
- duplicate/near-duplicate với current CVAT/review authority;
- grouped split theo recording day và source video;
- overlap giữa các nguồn trước khi tạo training/evaluation split.

Các kiểm tra này quyết định mapping, chất lượng và grouped split của dataset hợp
nhất; chúng không thay đổi vai trò của legacy như một nguồn dữ liệu được dùng để
tạo training snapshot. Khi báo cáo, composition của nguồn mới và legacy cần
được tách trong provenance/table để người đọc thấy phần đóng góp về temporal
diversity, nhưng metric được tính trên split hợp nhất đã đăng ký.

### D.3. Bản đồ tracking chi tiết (Detailed tracking map)

| Tracking topic | Methodology location | Experiment location | Evidence needed | Claim boundary |
|---|---|---|---|---|
| Identity semantics | 2.4.1 | 3.5.1 | Detection/annotation/tracker schemas | `track_id` không phải biological identity sáu tuần |
| Variable visible-pig counts | 2.4.1–2.4.3 | 3.5.4 | Frame population and visibility strata | Không giả định tám pigs ở mọi frame |
| Track birth, termination, Hidden and occlusion | 2.4.2–2.4.3 | 3.5.2/3.5.5 | Tracking GT and export contract | Semantics phải khớp evaluator |
| Association logic | 2.4.3 | 3.5.2 | Config, code SHA, detector evidence | Không gọi detector score là association quality |
| Causal mode | 2.4.4 | 3.5.3 | Current/past-only contract, delay and runtime | Chỉ claim near-real-time nếu không dùng future frames |
| Bounded-lag mode | 2.4.4 | 3.5.3 | Look-ahead radius and output delay | Không gọi zero-delay |
| Offline hybrid/repair | 2.4.4 | 3.5.3/3.5.5 | Repair manifest, affected ranges, original/corrected IDs, reason and future evidence | Chỉ claim offline long-term refinement |
| Fair detector evidence | 2.4.5 | 3.5.1 | Detector weights, cadence, confidence/NMS, mask, GT and hashes | Separate pure-association from full-pipeline comparison |
| Baseline/factorial design | 2.4.6 | 3.5.2/3.8 | Frozen arms and matched split/evaluator | Không ép 2×2 khi arm thiếu authority |
| Detection-related metrics | 2.4.7 | 3.5.2 | TP, FP, FN, DetA, MOTA, MOTP/localization | Không thay identity metrics |
| Identity-related metrics | 2.4.7 | 3.5.2 | HOTA, AssA, IDF1, IDP/IDR, IDSW, fragmentation, predicted IDs | Không dựa riêng aggregate IDSW |
| Identity-error episodes | 2.4.7 | 3.5.5 | Temporary, recovered, permanent/terminal swaps | Low IDSW không chứng minh error ngắn |
| Long-duration population | 2.4.8 | 3.5.4 | Short/medium/long clip manifests | Không suy long-term từ one-minute clip |
| Runtime/deployment | 2.4.9 | 3.5.3 | Hardware, warm-up, repeats, mean/dispersion/percentiles | Detector-only, tracker-only and full pipeline must be separated |
| Downstream propagation | 2.4.10 | 3.5.5/3.8/3.10 | Profile duration/proportion/bout/transition error, divergence and deviation stability | If unmeasured, state limitation |

Tracking error analysis must use frame sequences or episode strips. Required
categories include dense overlap, partial/full occlusion, similar appearance,
abrupt motion, entry/exit, re-entry, detector miss, duplicate detections,
fragmentation, temporary and permanent swaps, incorrect offline repair, and
empty/low-density scenes.

Where evidence permits, the downstream experiment should compare: (i) ground-
truth boxes and identities, (ii) detected boxes with ground-truth identities,
(iii) predicted tracking with ground-truth behavior labels, and (iv) complete
predicted tracking plus behavior classification. If this factorial propagation
experiment is not completed, the thesis must state that tracking-to-profile
effect remains a limitation rather than assume it is negligible.

## E. Bản đồ chống lặp và cross-reference (Duplication and cross-reference map)

Mỗi khái niệm chỉ có một “home section”. Các section khác chỉ nhắc kết luận cần
thiết và dẫn ngược về home section. Điều này tránh việc methodology trở thành
chuỗi ghi chú lịch sử, đồng thời ngăn results lặp lại thuật toán.

| Khái niệm | Home section | Được nhắc lại ở | Cách cross-reference đề xuất |
|---|---|---|---|
| Toàn bộ pipeline | 2.1 | 1.6, 3.1 | “The complete processing flow is defined in Section 2.1.” |
| Pen, camera, population và modality | 2.2 | 3.1, 4.2 | “The acquisition context and its transfer boundary are described in Section 2.2.” |
| Source-time versus playback-time | 2.2, Figure 4 | 2.8, 3.3, 3.10 | “All biological durations use the source-time convention in Section 2.2.” |
| Detection candidate selection | 2.3 | 3.4, 3.9 | “The selection protocol is described in Section 2.3; this section reports its evaluated output.” |
| Detection split and leakage audit | 2.3.9–2.3.10 | 3.1, 3.4 | “The grouped split and duplicate audit are reported in Section 3.1.” |
| Detector configuration | 2.3.11–2.3.12 | 3.2, 3.4 | “The detector contract is fixed in Section 2.3 and evaluated in Section 3.4.” |
| Identity continuity | 2.4 | 3.5, 3.9 | “Tracking performance is evaluated separately from detector performance in Section 3.5.” |
| Causal/offline tracking semantics | 2.4.4 | 3.5.3, 4.3 | “The processing-mode semantics and delay are defined in Section 2.4.4.” |
| Tracking fairness boundary | 2.4.5 | 3.5.1 | “The matched detector-evidence boundary is reported in Section 3.5.1.” |
| Tracking metrics and identity errors | 2.4.7 | 3.5.2–3.5.5 | “Identity quality is reported with episode and per-video metrics in Section 3.5.” |
| Tracking runtime | 2.4.9 | 3.5.3 | “Runtime is separated into detector, association, repair and end-to-end cost.” |
| Tracking-to-profile propagation | 2.4.10 | 3.5.5, 3.8, 3.10 | “The effect of identity authority on profile statistics is evaluated separately.” |
| Ten behavior definitions | 2.5 | 1.2, 3.6 | “The ten-class ontology is defined in Section 2.5.” |
| Historical legacy pathway | 2.5 | 3.1, 3.8, Appendix B | “Legacy bursts are merged with the new behavior source to form the training dataset; their temporal-diversity contribution is reported separately.” |
| Human review and corrected source | 2.5–2.6, Figure 5 | 3.1, 4.2 | “Review and source lineage are bound to the authority described in Sections 2.5–2.6.” |
| Feature families | 2.7 | 2.9, 3.8 | “The feature whitelist is defined in Section 2.7; family ablations are in Section 3.8.” |
| Temporal windows | 2.8 | 2.9, 3.2, 3.8 | “Window variants are specified once in Section 2.8.” |
| Model architecture | 2.9 | 3.2, 3.6–3.8 | “The architecture is described in Section 2.9 and compared under the contract in Section 3.2.” |
| Profile variables and screening | 2.10 | 3.10, 4.2 | “Screening is an application layer defined in Section 2.10, not a clinical classifier.” |
| Final metric values | Results section for each task | Abstract, conclusion | “Only registered evaluator artifacts may populate the final tables.” |

### E.1. Những lặp lại cần tránh cụ thể

- Không lặp toàn bộ danh sách cửa sổ `6/8/12/16` và sampled-six trong 2.1, 2.8,
  2.9 và 3.2. Section 2.8 là nơi mô tả đầy đủ; nơi khác chỉ dẫn chiếu.
- Không lặp đoạn mô tả nguồn dữ liệu trong 2.2 và 3.1. Chapter 3 chỉ báo cáo
  composition và split thực tế.
- Không lặp thuật toán background/activity/aHash trong 3.4. Results chỉ nêu
  manifest đầu ra và các metric.
- Không đặt câu “same pen implies generalization” ở 2.2 hoặc 3.4. Giới hạn
  transfer được phân tích ở 4.2.
- Không đưa chi tiết review implementation vào 2.9. Model section chỉ nhận
  data contract đã được source authority đóng băng.

## F. Danh mục hình và bảng (Figure and table inventory)

Số hiệu dưới đây là **provisional numbering cho outline hợp nhất**. Các draft
2.1–2.3 hiện có thể còn gọi số hình khác; khi merge vào thesis phải dùng một
master list duy nhất. Đặc biệt, timing được đặt trước review lineage để người
đọc hiểu thời gian trước khi xem luồng correction.

### F.1. Figures

| Figure | Nội dung và chức năng | Nguồn/khả năng tạo | Trạng thái |
|---|---|---|---|
| Figure 1 | Full-scene study example, tracked pigs, short behavior timeline | `data/raw/images_clean/` và bound prediction/profile artifact | `PENDING` |
| Figure 2 | End-to-end framework: RGB video → detection/tracks → windows/features → behavior/posture → profile/screening và causal alerts | Diagram từ blueprint và implemented module names | Có thể dựng; cần chốt tên module |
| Figure 3 | Study pen, camera, feeder, drinkers, enrichment và representative RGB frame | Dataset metadata và real RGB frame; depth chỉ là acquisition context | Có asset ứng viên; cần bind lineage |
| Figure 4 | Source 6 fps, 1800 frames/near five real minutes, MP4 playback 30 fps/near one minute | `times.txt`, source manifest, reproducible timing diagram | Có thể dựng sau khi chốt source-time convention |
| Figure 5 | Annotation/review lineage: raw source → native unit → review/control → corrected source → rebuild/audit | Review scripts, authority artifacts, corrected-source manifest | Có thể dựng; status current authority cần reconcile |
| Figure 6 | Detection-data workflow: video → background/mask → activity → candidate → aHash/Hamming → balancing/fallback → annotation → grouped split | Notebook và manifests | Có thể dựng từ protocol; không được thêm stage không có evidence |
| Figure 7 | Detection qualitative panels: dense correct case, partial occlusion, low density, empty pen, FP/FN | Registered detector predictions and source frames | `PENDING` final predictions |
| Figure 8 | Detection-to-tracking data flow and actor-centred trajectory output | Detector/tracker contract | Có thể dựng từ pipeline authority |
| Figure 9 | Identity continuity through occlusion or re-entry episode | Tracking video, GT and prediction lineage | `PENDING` bound episode |
| Figure 10 | Causal versus offline tracking modes, future evidence and output delay | Mode contract and repair manifest | `PENDING` until profiles are confirmed |
| Figure 11 | Per-video HOTA/IDF1 or identity-error episode plot across duration strata | Registered tracking evaluator | `PENDING` |
| Figure 12 | Runtime breakdown: detection, association, repair and end-to-end | Repeated timing manifest | `PENDING` |
| Figure 13 | Multimodal behavior/posture model architecture without depth branch | Final model/config and feature whitelist | Skeleton; model authority pending |
| Figure 14 | Behavior and posture performance, including confusion matrices | Registered behavior/posture evaluator | `PENDING` |
| Figure 15 | Long-term behavior profiles by individual/time window | Profile builder and review-independent predictions | `PENDING` |
| Figure 16 | Potential behavioral deviation screening signal versus baseline | Baseline/screening artifact; no disease label | `PENDING` |
| Figure 17 | Optional precision–recall curves for detection | Reproducible evaluator output | `PENDING`; omit if no registered curve |

Figure 3 có thể sử dụng ảnh RGB thật trong `data/raw/images_clean/`; actor crop
trong `outputs/legacy_16f_rebuild/legacy_16f_rebuild_20260718_v2/06_full_recovery/crops/`
chỉ là candidate asset cho panel phụ. Mỗi ảnh phải được bind với source video,
frame/burst và lineage trước khi xuất bản. Không dùng crop để suy ra identity
continuity nếu không có authority tương ứng.

### F.2. Tables

| Table | Nội dung | Evidence/status |
|---|---|---|
| Table 1 | Study, acquisition và temporal packaging specifications | `ESTABLISHED_SOURCE_FACT` sau citation/timestamp check |
| Table 2 | Detection frame-selection stages and purpose | `ESTABLISHED_PROJECT_PROTOCOL` từ notebook/config |
| Table 3 | Detection split by recording day/session | `FINAL_RESULT_REQUIRES_ARTIFACT` |
| Table 4 | Images and bounding boxes by split | `FINAL_RESULT_REQUIRES_ARTIFACT` |
| Table 5 | Detector and model configuration | `FINAL_RESULT_REQUIRES_ARTIFACT` |
| Table 6 | Tracking-evaluation video population and trajectory inclusion | Tracking GT, video/frame/hash manifest |
| Table 7 | Tracker configuration and causal/offline mode semantics | Registered tracker profiles and repair contract |
| Table 8 | Detector-evidence fairness matrix | Detector/tracker/evaluator hashes and cadence |
| Table 9 | Aggregate and per-video tracking metrics | Registered tracking predictions/evaluator |
| Table 10 | Identity-error episode summary and duration strata | Episode audit and short/medium/long populations |
| Table 11 | Tracking runtime and latency | Repeated timing manifest and hardware record |
| Table 12 | Downstream profile error under identity authorities | Propagation experiment, or explicit limitation |
| Table 13 | Ten behavior classes and definitions | User-confirmed ontology; citation where inherited |
| Table 14 | Review, correction and merged new/legacy composition | Review-close, corrected-source and merged manifest |
| Table 15 | Behavior/posture metrics and grouped uncertainty | Registered evaluator predictions |
| Table 16 | Profile variables and deviation-screening rules | Profile contract; no anomaly GT |
| Table 17 | Limitations and transfer boundaries | Final scope/authority |
| Table 18 | Feature families and model-X exclusions | Feature whitelist audit and input contract |

Không nên tạo figure hoặc table chỉ để lấp layout. Mỗi visual phải được nhắc
trong đoạn văn trước hoặc sau nó, có caption tự đủ, có đơn vị và có source-time
basis. Các plot định lượng chỉ được tạo từ artifact có hash và evaluator.

## G. Bản đồ claim–evidence (Claim–evidence status)

Bảng này là bản đồ viết, không phải claim registry mới. Trạng thái `may write`
chỉ có nghĩa là có thể viết ở mức protocol hoặc giới hạn; nó không cho phép
đưa một metric chưa được đăng ký vào abstract, conclusion hoặc contribution
claim.

| Planned claim | Authority/evidence path | Status | May write now? |
|---|---|---|---|
| Framework gồm detection, tracking, behavior classification và downstream profiling | `CLASSIFICATION_V2_THESIS_BLUEPRINT_EVIDENCE_MAP.md`, draft 2.1 | `ESTABLISHED_PROJECT_PROTOCOL` | Có, ở mức kiến trúc |
| Behavior classification là trung tâm thesis | Blueprint và user-confirmed narrative | `ESTABLISHED_PROJECT_PROTOCOL` | Có |
| Dataset được ghi ở source khoảng 6 fps, clip 1800 frames, playback đóng gói 30 fps | Dataset papers, `times.txt`, draft 2.2 | `ESTABLISHED_SOURCE_FACT` cần đối chiếu citation | Có, nhưng phải dùng source-time |
| Không phải mọi frame đều chứa đủ tám pigs | User-confirmed study context và source metadata | `ESTABLISHED_SOURCE_FACT` | Có |
| Depth đã được thu nhưng chưa chứng minh tác dụng trong current model | Current feature/model authority và figure plan | `ESTABLISHED_PROJECT_PROTOCOL` | Có, như giới hạn/future work |
| Detection candidate selection dùng background/mask, activity, aHash/Hamming và temporal gaps | `notebooks/01_data_preparation/video_to_frame_phase_1.ipynb` và run manifest nếu có | `ESTABLISHED_PROJECT_PROTOCOL` | Có, không gọi là novel algorithm |
| Historical behavior pathway dùng các ngày và trigger/ROI đa dạng | `video_to_frame_phase_2.ipynb`, candidate/manifests | `ESTABLISHED_PROJECT_PROTOCOL` | Có, như provenance |
| Legacy cohort có đúng hai burst cho mỗi video | Không có evidence; code phụ thuộc trigger/gap/manifest | `UNSUPPORTED_AND_MUST_BE_REMOVED` | Không |
| Legacy data được hợp nhất với nguồn mới để bổ sung temporal diversity cho training dataset | Legacy manifests, mapping, leakage audit, merged snapshot | `ESTABLISHED_PROJECT_PROTOCOL` với snapshot final còn pending | Có thể viết ở mức pipeline; metric chờ merged evaluator |
| Tất cả recording day nằm độc quyền trong một split | Detection split manifest và audit cần kiểm tra | `FINAL_RESULT_REQUIRES_ARTIFACT` | Chưa viết như fact |
| Exact detection image/box counts | Final detection manifest | `FINAL_RESULT_REQUIRES_ARTIFACT` | Chưa |
| Detector precision/recall/mAP hoặc superiority | Prediction artifact, evaluator, config, code SHA | `FINAL_RESULT_REQUIRES_ARTIFACT` | Chỉ để placeholder |
| Detector performance cao do fixed pen/camera/background và controlled selection | Study design và final split evidence | `PROVISIONAL_RESULT` | Có ở phần interpretation, không suy ra transfer |
| Tracking là cơ chế biến detection thành actor-centred trajectories và identity-conditioned behavior data | Tracking contract, pipeline lineage and model-input contract | `ESTABLISHED_PROJECT_PROTOCOL` | Có |
| High detector accuracy implies high identity continuity | Không có authority; task definitions tách hai tầng | `UNSUPPORTED_AND_MUST_BE_REMOVED` | Không |
| Causal tracking dùng current/past evidence và offline repair dùng future evidence | Mode profiles, repair manifest and runtime contract | `ESTABLISHED_PROJECT_PROTOCOL` nếu profile được đăng ký | Viết semantics, chưa viết superiority |
| Tracking quality được chứng minh bằng aggregate, per-video và identity-error episodes | Tracking evaluator, GT population and fairness manifest | `FINAL_RESULT_REQUIRES_ARTIFACT` | Chưa điền metric |
| Tracking mode có trade-off quality–processing cost | Matched mode comparison and repeated runtime manifest | `FINAL_RESULT_REQUIRES_ARTIFACT` | Chỉ viết research question/protocol |
| Tracking error ảnh hưởng duration, bout, transition và deviation profile | Downstream propagation experiment | `FINAL_RESULT_REQUIRES_ARTIFACT` hoặc `FUTURE_WORK` | Không giả định bằng không |
| Detection performance implies perfect tracking | Không có authority; contradicted by task separation | `UNSUPPORTED_AND_MUST_BE_REMOVED` | Không |
| Primary reviewed behavior labels are final and train-ready | Review-close, fixed-point, corrected-source and current-state authorities conflict | `FINAL_RESULT_REQUIRES_ARTIFACT` | Chỉ viết status/provenance |
| Posture is a main experiment with three classes | User-confirmed design; posture authority/evaluator pending | `ESTABLISHED_PROJECT_PROTOCOL` plus pending result | Viết design, chưa viết metric |
| Screening output is a potential deviation signal, not diagnosis | Blueprint, charter, user-confirmed anomaly definition | `ESTABLISHED_PROJECT_PROTOCOL` | Có |
| Anomaly classifier has supervised clinical ground truth | No anomaly labels in current data | `UNSUPPORTED_AND_MUST_BE_REMOVED` | Không |
| Cross-farm, cross-camera hoặc cross-pen generalization | No registered transfer experiment | `FUTURE_WORK` | Chỉ viết giới hạn/future work |

### G.1. Các artifact cần bind trước khi viết kết quả

Đối với detection, cần một manifest cuối có image IDs, box IDs, source video,
recording day/session, split, hash và annotation version; prediction artifact;
evaluator version; detector code/config/weights; và leakage audit. Đối với
behavior, cần một train-ready snapshot sau khi review-close, corrected-source
rebuild, feature/window manifest và grouped evaluator. Những điều kiện này
không được thay bằng con số trong draft cũ.

Có một điểm cần ghi rõ trong outline: latest review-close artifact và current
decision/memory đang thể hiện các trạng thái review khác nhau (bao gồm frozen
review-close và residual/fixed-point history). Đây là **authority reconciliation
item**, không được giải quyết bằng cách chọn con số “mới hơn” trong hội thoại.
Chapter 2 chỉ mô tả workflow; Chapter 3 chỉ điền composition sau khi authority
được thống nhất.

## H. Câu hỏi còn bỏ ngỏ (Unresolved questions)

Chỉ các câu hỏi dưới đây cần trả lời trước khi khóa outline thành manuscript:

1. Detector nào là promoted final method: YOLO version, model scale, input
   resolution, pretrained initialization, augmentation, optimizer, epochs,
   confidence/NMS và selection criterion?
2. Detection manifest cuối có bao nhiêu images và bounding boxes theo từng
   split, và split-day/session map thực tế là gì? Active notebook comments không
   đủ để suy ra map cuối.
3. Roboflow export nào là authority, và manual bounding-box QC có manifest/hash
   độc lập hay chỉ nằm trong workspace history?
4. Leakage audit có kiểm tra cùng leaf video, neighboring source-time interval,
   duplicate/near-duplicate hash và source-burst overlap giữa các split không?
5. Review-close/fixed-point/corrected-source artifacts nào là current authority
   cho primary behavior labels sau conflict giữa current-state và latest
   review artifacts?
6. Legacy và nguồn mới sau khi hợp nhất có bao nhiêu actor/burst, trải trên bao
   nhiêu ngày/video, và merged training snapshot được phân chia thế nào?
7. Taxonomy mapping legacy → current ten classes có trường hợp không tương
   thích hoặc label ambiguous cần loại trước khi tạo merged snapshot không?
8. Temporal windows `6/8/12/16` và sampled-six đã có train-ready manifests sau
   corrected-source rebuild chưa, hay mới chỉ là experiment design?
9. Posture authority, transition-stratum policy và grouped posture evaluator đã
   được freeze chưa?
10. Profile window, baseline construction, threshold/outlier rule và online
    causal alert policy cuối cùng là gì?
11. Có prediction/profile artifact review-independent nào đủ provenance để làm
    Figure 1, Figure 9 và Figure 10 không?
12. Institution yêu cầu citation style, data-availability, ethics, funding,
    authorship và AI-disclosure ở mức nào?
13. Tracking authority cuối cùng gồm những mode/profile nào, và profile nào
    được phép gọi là causal, bounded-lag hoặc offline hybrid?
14. Tracking evaluation population gồm video nào, frame range nào, short/medium/
    long strata nào, và Hidden/occlusion semantics được mã hóa ra sao?
15. Ground-truth tracking hashes, source-video hashes, included/excluded
    trajectories và development/validation/untouched-test roles đã được bind
    trong manifest nào?
16. Fair comparison có cố định detector weights, cadence, confidence/NMS, mask,
    GT, evaluator, threshold, code SHA và runtime environment không?
17. Có đủ arm hợp lệ để báo cáo factorial core × offline-repair comparison hay
    cần dùng một design khác theo authority hiện hành?
18. Runtime measurements đã kiểm soát warm-up, GPU load, repeats và percentile
    latency chưa; detector, tracker, repair và full pipeline được tách chưa?
19. Downstream propagation experiment đã đo duration/proportion/bout/transition
    error và deviation-score stability chưa, hay phải ghi là limitation?

## I. Trình tự viết thực tế (Recommended drafting order)

1. Dùng outline này để supervisor xác nhận cấu trúc và ranh giới claim; chưa
   chuyển sang English prose.
2. Khóa Section 2.1 và 2.2 ở dạng tiếng Việt, chỉ sửa các câu làm rõ flow,
   timing và RGB-only current branch.
3. Viết Section 2.3 detection methodology từ notebook và manifest; đánh dấu
   mọi setting chưa có artifact là `PENDING`.
4. Viết Section 2.5–2.6 behavior source, legacy provenance, human review và
   corrected-source lineage; báo cáo riêng composition nguồn legacy trong merged
   training dataset nhưng vẫn giữ nguồn này trong tập dữ liệu dùng cho training.
5. Sau authority reconciliation, hoàn thiện 2.7–2.9 về feature, windows và
   model; posture giữ ở mức design cho tới khi evaluator tồn tại.
6. Viết Chapter 3 protocol trước, rồi mới điền metrics vào 3.4–3.10 khi các
   artifacts đạt gate.
7. Chọn ảnh thật và tạo diagrams/plots song song với đoạn văn tương ứng. Mỗi
   hình phải được gọi tên trong prose và có caption nêu nó chứng minh điều gì.
8. Viết Chapter 4, Abstract và front matter cuối cùng, chỉ dùng claim được
   claim registry hoặc human/source authority thừa nhận.

## J. Ghi chú chuyển sang English thesis

Khi technical meaning của outline được xác nhận, bản English phải được viết lại
theo meaning chứ không dịch từng câu. Dùng văn phong học thuật trực tiếp, tránh
liệt kê dày đặc trong prose, tránh mở đầu kiểu “This section will discuss”, và
giữ các thuật ngữ ổn định: `identity tracking`, `native behavioral burst`,
`source time`, `behavioral profile`, `behavioral deviation screening`.

Không đưa các cột “Evidence”, “Status” hay “Open question” vào manuscript
trừ khi supervisor muốn một bảng phương pháp. Chúng là công cụ biên tập để
kiểm tra rằng một câu, một hình và một kết quả đều có nguồn hợp lệ.

## K. Authority và tài liệu tham chiếu

- `docs/CLASSIFICATION_V2_THESIS_BLUEPRINT_EVIDENCE_MAP.md`
- `docs/CLASSIFICATION_V2_CURRENT_STATE.md` (đọc cùng authority index; có thể
  chứa trạng thái cũ cần reconcile)
- `docs/CLASSIFICATION_V2_DATA_REBUILD_AND_HUMAN_REVIEW_RUNBOOK.md`
- `docs/thesis_drafts/CHAPTER_2_1_OVERVIEW_FRAMEWORK_VI_DRAFT.md`
- `docs/thesis_drafts/CHAPTER_2_2_DATA_SOURCES_NATIVE_UNITS_VI_DRAFT.md`
- `docs/thesis_drafts/CHAPTER_2_3_BEHAVIOR_ANNOTATION_REVIEW_VI_DRAFT.md`
- `docs/thesis_drafts/THESIS_FIGURE_AND_TABLE_PLAN.md`
- `docs/tracking/reconciliation/FOUR_METHOD_TRACKING_FREEZE_AUTHORITY_20260729.json`
- `notebooks/01_data_preparation/video_to_frame_phase_1.ipynb`
- `notebooks/01_data_preparation/video_to_frame_phase_2.ipynb`
- `notebooks/01_data_preparation/video_to_frame_annotate.ipynb`
- `outputs/classification_v2/review_authority/` và các manifest detection/legacy
  khi được xác nhận là current authority

**Kết luận của outline:** detection cần một methodology/result path riêng vì nó
có quy trình chọn và làm sạch dữ liệu có thể tái lập; behavior classification
vẫn là trọng tâm; historical legacy bursts được hợp nhất với nguồn mới để tạo
training dataset và được báo cáo như phần bổ sung temporal diversity. Grouped
leakage, taxonomy và label quality vẫn phải được kiểm tra trước khi snapshot
hợp nhất được dùng làm authority cho kết quả cuối.
> **Superseded outline.** This historical planning file is retained for
> provenance. The active master outline is
> `docs/thesis_drafts/PIG_BEHAVIOR_THESIS_MASTER_OUTLINE_V2.md`, which
> corrects the thesis title, current review status, section balance and
> evidence boundaries.
