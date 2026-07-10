# Classification V2: Research protocol hướng tới chuẩn bài báo Q1/Q2

## Document control

| Trường | Giá trị |
|---|---|
| Protocol version | 1.2 |
| Companion roadmap | `CLASSIFICATION_V2_SPATIOTEMPORAL_IMAGE_TRAINING_ROADMAP.md`, version 2.2 |
| Machine-readable gate | `configs/classification_v2/paper_grade_protocol_v1.json` |
| Study type | Retrospective computational study trên video hành vi lợn |
| Current status | Protocol design, chưa preregister, chưa full training |
| Publication readiness | Chưa sẵn sàng |
| Primary statistical unit dự kiến | Canonical recording date/session |
| Primary prediction unit dự kiến | Temporal annotation unit/review unit |
| Primary endpoint dự kiến | Event-level macro-F1 trên nested out-of-fold predictions |

Mục đích của protocol này là biến roadmap kỹ thuật thành một nghiên cứu có câu hỏi kiểm chứng được, đơn vị mẫu đúng, baseline công bằng, uncertainty rõ và claim không vượt quá bằng chứng. Q1/Q2 là mức nghiêm ngặt hướng tới, không phải nhãn chất lượng tự gán hoặc bảo đảm được nhận đăng.

## 0. Scope đã khóa theo quyết định nghiên cứu

Protocol này khóa hướng paper ở mức Q2 mạnh:

- Đóng góp trung tâm: framework multimodal spatio-temporal bbox + ROI all-class + social/partner context, nằm trong pipeline dữ liệu/review có audit và leakage-safe validation.
- Claim chính cho paper: cải thiện nhận diện hành vi lợn dưới session/video-safe validation so với baseline unimodal công bằng.
- Không claim Q1 generalization/cross-farm/cross-camera/cross-cohort nếu chưa có external cohort/farm/camera độc lập.
- `pig_id` trong annotation không được dùng như biological identity xuyên video nếu chưa có metadata xác nhận.
- Review và target construction dùng `review_unit_id`; evaluation chính dùng temporal unit/native review unit.
- Full-frame/partner context được dùng cho interaction vì fight/social-nose cần ngữ cảnh xã hội; quy tắc actor-only/bystander phải giữ trong label policy.

## 1. Kết luận đánh giá lại

Roadmap version 1 tốt ở cấp engineering nhưng chưa đủ ở cấp scholarly protocol. Những điểm mạnh đã có:

- Data lineage từ frame đến temporal interval, sequence window và review unit rõ.
- Quy tắc leakage ở feature-level tốt; model input có whitelist.
- Có split manifest, mask, sample weight, audit JSON và smoke baseline.
- Đã nhận diện train-serving mismatch, padding mask, class imbalance và confusion groups.
- Có ablation path từ tabular đến spatial, image và multimodal fusion.

Các blocker khoa học chính:

1. Chưa có research question, falsifiable hypothesis và smallest effect size of interest.
2. Không có literature search/citation matrix, nên chưa thể claim novelty hoặc SOTA.
3. Split hiện tại group theo video key nhưng 13/13 recording dates vẫn xuất hiện ở nhiều split.
4. 160,740 windows không phải 160,740 mẫu độc lập; test theo window sẽ gây pseudoreplication.
5. CVAT chỉ có 12 video thuộc 3 ngày; metadata cá thể/cohort/farm/camera chưa đủ để claim external generalization.
6. Human review mới có 3 decision rows; chưa có inter-rater reliability hoặc gold test adjudication.
7. Chưa có prospective statistical analysis plan, power/precision analysis, multiplicity control và locked-test policy.
8. Source và label có association đáng kể, tạo nguy cơ shortcut learning.

## 2. Scholar evaluation của roadmap version 1

Artifact được đánh giá là research proposal/engineering roadmap, không phải manuscript hoàn chỉnh.

| Dimension | Điểm v1/5 | Bằng chứng | Ưu tiên sửa |
|---|---:|---|---|
| Problem and research question | 2 | Có mục tiêu 10-class nhưng chưa có RQ/hypothesis định lượng | Critical |
| Literature and context | 1 | Chưa có search log, citation hay closest-work matrix | Critical |
| Methodology | 3 | Pipeline/mask/ablation tốt, nhưng split và prediction unit chưa publication-safe | Critical |
| Data and evidence | 2 | Có row counts; thiếu biological/session metadata và review reliability | Critical |
| Analysis | 2 | Có metrics/slices; thiếu statistical tests, CI design, power và multiplicity | Critical |
| Results and interpretation | N/A | Chưa có full experiment | Sau training |
| Limitations and validity | 3 | Đã nêu leakage/label noise; chưa định lượng session/source confounding | High |
| Writing and structure | 4 | Rõ và đầy đủ nhưng dài, engineering và research còn trộn | Medium |
| Citations | 1 | Không có citation support | Critical |

Overall applicable score của version 1: khoảng `2.3/5`. Sau khi áp dụng protocol này, chất lượng thiết kế mục tiêu là `>=4/5`; điểm citations chỉ tăng sau khi literature review thật sự hoàn tất.

## 3. Evidence registry hiện có

### 3.1. Quy mô quan sát

| Đơn vị | Số lượng |
|---|---:|
| Reviewed frame rows | 245,664 |
| Temporal intervals/review units | 33,354 |
| Sequence windows | 160,740 |
| Main-train valid windows | 152,704 |
| CVAT temporal intervals | 28,800 |
| Legacy temporal intervals | 4,554 |
| CVAT recordings | 12 |
| Legacy clip keys | 668 |
| Canonical recording dates | 13 |
| Exposed pig-ID tokens | 8 |
| Human decisions loaded | 3 |

`Exposed pig-ID tokens=8` không tự chứng minh có 8 hay nhiều hơn 8 biological subjects qua toàn bộ ngày/source. Metadata acquisition phải xác nhận số cá thể, cohort, farm, pen, camera và khả năng cùng cá thể xuất hiện qua nhiều ngày.

### 3.2. Evidence về non-independence

- 13/13 canonical dates xuất hiện ở nhiều split hiện tại.
- 14/16 source-specific session tokens xuất hiện ở nhiều split.
- Mỗi temporal interval có thể sinh nhiều window length và các window lân cận dùng chung frame.
- Legacy clip key thường là burst/clip trong một recording date, không phải một biological replicate độc lập.

Kết luận: split hiện tại chỉ dùng để smoke pipeline. Mọi metric publication-facing phải được tái tạo bằng session-safe protocol.

### 3.3. Evidence về source confounding

Interval-level contingency giữa `behavior` và `source_type` cho `Cramér's V=0.356` (`n=33,354`). Một số ví dụ:

| Behavior | CVAT intervals | Legacy intervals |
|---|---:|---:|
| fight | 2,196 | 17 |
| lying | 1,661 | 1,552 |
| sitting | 10,237 | 1,379 |
| playwithtoy | 101 | 42 |
| social-nose | 1,174 | 112 |

Source style có thể trở thành proxy cho label. Source phải là audit stratum, không phải model input.

## 4. Phạm vi claim và đóng góp nghiên cứu

### 4.1. Candidate central claim

Candidate claim nên tập trung vào một câu chuyện chính:

> Một framework multimodal spatio-temporal kết hợp actor bbox appearance, local/full-frame context khi cần, geometry-motion features, all-class ROI relations và social/partner context cải thiện nhận diện hành vi lợn dưới session/video-safe validation so với các baseline unimodal công bằng, trong một pipeline dữ liệu/review có audit và kiểm soát leakage.

Đây mới là candidate claim. Chỉ được giữ nếu literature review xác nhận khoảng trống và experiment session-safe hỗ trợ.

### 4.2. Candidate contributions

1. Một auditable review-to-training pipeline cho CVAT 6-frame intervals và legacy 16-frame bursts, giữ nguyên row và decision lineage.
2. Một representation spatio-temporal có mask tách biệt cho padding, missing observation và feature quality.
3. Một multimodal actor/context fusion model dùng ROI all-class và social/partner channels không phụ thuộc ground-truth label.
4. Một evaluation protocol session/video-safe, event-level, có source/shortcut controls và uncertainty theo recording cluster.
5. Một artifact contract tái lập cho dataset snapshot, split, model input, predictions, metrics và claim evidence.

Không nên claim cả năm đóng góp là novel ngang nhau. Sau literature review phải chọn một primary methodological contribution và tối đa hai supporting contributions.

### 4.3. Claim boundaries

- Không claim cross-farm/cross-cohort generalization nếu không có external cohort.
- Không claim unseen-animal generalization nếu identity metadata không chứng minh animal-disjoint split.
- Không claim SOTA nếu dataset/split/label ontology khác prior work hoặc baseline không được tái tạo công bằng.
- Không suy diễn attention/Grad-CAM thành cơ chế sinh học nhân quả.
- Không dùng window count làm số mẫu độc lập.
- Không gọi output là fully human-reviewed khi chỉ một phần unit có manual decision.

## 5. Research questions và hypotheses

Các endpoint và threshold dưới đây là provisional. Chúng phải được freeze trong experiment registry trước outer evaluation.

### RQ1, primary

**RQ1:** Multimodal fusion có cải thiện event-level behavior recognition trên unseen recording sessions so với strongest unimodal baseline không?

**H1:** Proposed fusion model có paired improvement dương về pooled out-of-fold macro-F1 so với strongest unimodal baseline, với:

- 95% recording-cluster bootstrap CI của delta không chứa 0.
- Provisional smallest effect size of interest, `SESOI = +0.03` absolute macro-F1.
- Không có critical-class regression lớn hơn non-inferiority margin đã khóa.

SESOI phải được domain expert hoặc cost-of-error rationale xác nhận; không được thay đổi sau khi xem outer-test result.

### RQ2, secondary

**RQ2:** Local context và explicit ROI/social relations có cải thiện nhóm ROI-intent và interaction so với actor-only sequence không?

**H2:** So với actor-only model:

- ROI-group macro-F1 và interaction-group macro-F1 tăng ít nhất provisional `0.03`.
- Posture-group macro-F1 không giảm quá provisional non-inferiority margin `0.02`.
- Bystander fight false-positive rate không tăng.

### RQ3, methodological

**RQ3:** Video-key split hiện tại ước lượng performance lạc quan đến mức nào so với session-safe nested evaluation?

Endpoint là chênh lệch macro-F1, calibration error và focus-pair confusion giữa hai protocol. RQ3 định lượng bias của evaluation; không dùng split yếu để chọn final model.

### RQ4, exploratory

**RQ4:** Hiệu năng thay đổi ra sao theo source, recording date, hidden/quality tier, window duration và tracking noise?

RQ4 là exploratory. Kết quả phải ghi rõ là hypothesis-generating và dùng multiplicity-aware reporting.

### Secondary model questions

- Multi-task heads có giảm confusion có cấu trúc không?
- Temporal order có đóng góp signal thật hay model chủ yếu dùng appearance/background?
- Graph social branch có cải thiện interaction đủ để bù complexity/latency không?

Các câu hỏi này chỉ vào paper chính nếu protocol được preregister trước khi chạy; nếu không, đưa vào supplementary/exploratory section.

## 6. Đơn vị nghiên cứu và target construction

### 6.1. Hierarchy bắt buộc

| Cấp | Định nghĩa | Vai trò |
|---|---|---|
| Biological unit | Pig/cohort thật | External-validity claim |
| Recording cluster | Farm-camera-date-session | Split và statistical resampling |
| Video/clip | File video hoặc burst source | Data loading, nested trong cluster |
| Temporal unit | CVAT 6f hoặc legacy 16f | Primary prediction/evaluation unit |
| Window | 6/8/12/16 sequence | Training augmentation/secondary analysis |
| Frame | Bbox/image observation | Feature extraction, không phải sample độc lập |

### 6.2. Primary analysis dataset

Primary paper analysis nên dùng đúng 33,354 temporal units:

- CVAT: native 6-frame interval từ anchor `k` đại diện `k..k+5`.
- Legacy: native 16-frame reviewed burst.
- Variable-length model dùng `length_mask` và `observed_mask`.
- Mỗi temporal unit tạo đúng một primary prediction.

Các 6/8/12/16 sequence windows vẫn có thể dùng cho training augmentation hoặc sensitivity analysis, nhưng test predictions phải collapse theo temporal unit bằng rule khóa trước. Không tính bốn window lengths như bốn test observations độc lập.

### 6.3. Canonical recording group

Tạo `recording_group_id` từ metadata có thứ tự ưu tiên:

1. Farm/facility.
2. Cohort/pen.
3. Camera/view.
4. Recording date.
5. Session/block liên tục.

Filename-derived date chỉ là fallback audit. Trước paper phải có mapping table được con người xác nhận, bao gồm alias giữa CVAT và legacy. Cùng recording date/source scene không được cắt qua folds.

## 7. Annotation quality protocol

### 7.1. Gold evaluation labels

- Outer-test/external-test temporal units phải được review độc lập với model prediction.
- Reviewer không xem predicted class/confidence khi xác nhận gold label.
- Disagreement được adjudicate bằng guideline versioned.
- `pending` không trở thành corrected label.
- `exclude` giữ row nhưng mask khỏi supervised main loss/evaluation theo protocol.

### 7.2. Inter-rater reliability

Xây một stratified double-review sample theo:

- 10 behavior classes.
- CVAT/legacy source.
- Hidden/quality tier.
- Confusion-focused pairs.
- Recording dates.

Sample size được xác định theo target CI width của agreement statistic, không chọn tùy tiện theo phần trăm. Báo cáo:

- Raw agreement.
- Cohen's kappa.
- Gwet's AC1 hoặc statistic bền hơn với prevalence imbalance.
- Per-class agreement/confusion.
- 95% CI theo recording cluster.

Rare classes nên oversample cho reliability analysis nhưng phải dùng weight đúng khi ước lượng overall agreement.

### 7.3. Label uncertainty

- Main confirmatory analysis dùng adjudicated hard labels.
- Label strength/reviewer uncertainty chỉ dùng weight hoặc sensitivity analysis.
- Soft-label experiment là secondary; không thay primary target sau khi thấy result.
- Annotation guideline phải định nghĩa actor/bystander, social-nose actor-only và direct fight involvement.

## 8. Literature review và novelty validation

Hiện chưa có literature evidence trong roadmap. Vì vậy mọi novelty statement đang ở trạng thái `UNVERIFIED`.

### 8.1. Databases

Tối thiểu tìm kiếm trên:

- Scopus hoặc Web of Science.
- IEEE Xplore.
- PubMed nếu paper liên quan precision livestock/animal health.
- Google Scholar để snowball citation, không dùng làm nguồn metadata duy nhất.
- Một agricultural database phù hợp nếu có quyền truy cập.

### 8.2. Search concepts

Kết hợp các nhóm từ khóa:

```text
(pig OR swine OR livestock)
AND (behavior recognition OR activity recognition OR social behavior)
AND (video OR computer vision OR deep learning)
AND (temporal OR sequence OR transformer OR TCN OR LSTM)
```

Các search phụ:

- Pig interaction/fighting detection.
- ROI-aware feeding/drinking recognition.
- Multi-animal graph/social behavior recognition.
- Multimodal image plus trajectory/geometry fusion.
- Domain shift và cross-farm validation trong precision livestock farming.
- Annotation harmonization, weak labels và temporal label propagation.

### 8.3. Inclusion/exclusion

Inclusion:

- Primary empirical work có video-based animal behavior classification.
- Methods có temporal modeling hoặc multimodal context liên quan trực tiếp.
- Báo cáo dataset, split và metric đủ để đánh giá.
- Foundational methods được dùng làm baseline dù không chuyên về pig.

Exclusion:

- Chỉ detection/tracking, không behavior classification, trừ khi là dependency quan trọng.
- Không mô tả split hoặc annotation unit đủ để so sánh.
- Review/opinion được dùng để tìm nguồn nhưng không thay primary evidence.

### 8.4. Literature matrix bắt buộc

Mỗi paper ghi:

- Citation/DOI, publication status, year.
- Species, farm/cohort/camera và number of animals/sessions.
- Behavior ontology và annotation granularity.
- Input modality/crop/context.
- Model/backbone/temporal encoder.
- Split unit và có external validation hay không.
- Primary metric, macro/per-class metrics và uncertainty.
- Code/data availability.
- Điểm giống/khác với proposed contribution.

Novelty statement chỉ được viết sau khi matrix có các closest works và được cập nhật tới search-freeze date. Quartile của journal phải kiểm tra theo năm submission, không suy từ tên venue hoặc dữ liệu cũ.

## 9. Experimental design

### 9.1. Publication-safe split

Với dữ liệu hiện có, protocol ưu tiên:

1. Outer leave-one-recording-date/session-out evaluation trên 13 canonical date groups.
2. Inner grouped cross-validation chỉ trên outer-train groups để chọn hyperparameter/model.
3. Aggregate out-of-fold predictions để tính primary event-level metrics.
4. Giữ một external cohort/farm/camera set hoàn toàn khóa nếu có thể thu thập.

Nếu một outer fold thiếu class, không tự xóa fold. Báo support và tính pooled OOF metric trên toàn bộ event predictions; per-session metric chỉ tính trên classes có support và phải ghi rõ denominator.

### 9.2. Model selection firewall

- Outer fold labels không dùng chọn architecture, crop size, loss, threshold hoặc early stopping.
- Hyperparameter budget giống nhau giữa candidate models.
- Test/external predictions được tạo bằng command/config đã khóa.
- Mọi failed/null run vẫn ghi trong experiment registry.
- Không đổi primary endpoint sau khi xem result.

### 9.3. Seed policy

- Smoke: 1 seed.
- Development ablation: tối thiểu 3 seeds.
- Confirmatory candidate: 5 seeds nếu compute cho phép.
- Report seed variance riêng với recording-cluster uncertainty.
- Primary model comparison dùng cùng seeds/folds để giữ pairing.

### 9.4. Data preprocessing firewall

Mọi statistic học từ dữ liệu phải fit trên outer-train hoặc inner-train tương ứng:

- Scaler/normalization.
- Percentile clipping.
- Imputation.
- Class weights.
- Calibration/temperature.
- Decision thresholds.

ROI geometry cố định từ scene annotation có thể áp dụng toàn video nếu thực sự là metadata có sẵn tại inference, nhưng version/checksum phải ghi rõ.

## 10. Baselines và candidate models

### 10.1. Baseline ladder

| ID | Model | Mục đích |
|---|---|---|
| B0 | Majority/prior classifier | Sanity floor |
| B1 | Whitelisted logistic/SGD | Linear tabular baseline |
| B2 | Gradient-boosted tabular model | Strong nonlinear tabular control |
| B3 | Single-frame actor CNN | Appearance-only control |
| B4 | Actor CNN + temporal mean/GRU | Standard sequence baseline |
| B5 | Spatial masked TCN | Engineered spatiotemporal baseline |
| B6 | Established video baseline từ literature review | Closest published control |
| P1 | Actor image + spatial + tabular fusion | Primary proposed model |
| P2 | P1 + local context | Context hypothesis |
| P3 | P2 + multi-task heads | Secondary candidate |
| P4 | P2 + social graph | Exploratory candidate |

B6 chỉ được chốt sau literature review. Không chọn một yếu baseline để tạo gain giả.

### 10.2. Fair-comparison rules

- Cùng outer/inner folds và label set.
- Cùng image resolution/augmentation khi architecture cho phép.
- Cùng pretrained backbone family cho ablation image/context.
- Parameter count, FLOPs, training time và inference latency được báo cáo.
- Hyperparameter search budget tương đương.
- Early stopping dùng cùng primary validation objective.

### 10.3. Confirmatory ablations

Predeclare tối đa các ablation chính:

1. Image-only versus spatial-only versus fusion.
2. Fusion minus motion.
3. Fusion minus all-class ROI.
4. Fusion minus social context.
5. Actor-only versus actor plus local context.
6. Correct mask versus deliberately unmasked padding control.

Multi-task, graph, pose và optical flow là secondary/exploratory trừ khi được preregister như paper chính.

## 11. Statistical analysis plan

### 11.1. Primary endpoint

- Event-level macro-F1 trên pooled nested out-of-fold temporal-unit predictions.
- Một prediction cho mỗi temporal unit.
- Label order cố định 10 classes.
- Missing-class support được báo cáo, không silently drop khỏi global class list.

### 11.2. Secondary endpoints

- Per-class precision, recall, F1 và one-vs-rest AUPRC.
- Balanced accuracy và multiclass MCC.
- ROI-group, interaction-group, motion-group, posture-group macro-F1.
- Focus-pair directional confusion rates.
- Bystander fight false-positive rate.
- NLL, Brier score và ECE với binning rule khóa trước.
- Latency, throughput, peak RAM/VRAM, parameters và artifact size.

### 11.3. Uncertainty

- 95% CI bằng paired recording-cluster bootstrap, resample recording groups chứ không resample windows.
- Dùng cùng bootstrap draws cho hai model để ước lượng CI của delta.
- Với chỉ 13 groups, báo rõ CI có thể rộng và thực hiện sensitivity với leave-one-group-out influence.
- Report mean/std qua seeds; không trộn seed variance với sampling uncertainty thành một con số mơ hồ.

### 11.4. Hypothesis tests và effect size

- Primary model delta dùng paired session-level randomization/sign-flip test nếu assumptions phù hợp; với 13 groups có thể enumerate `2^13=8192` sign patterns.
- Luôn báo absolute delta, relative delta, 95% CI và SESOI; p-value không thay effect size.
- Secondary confirmatory endpoints dùng Holm correction.
- Exploratory endpoints báo raw CI/p-value và đánh dấu exploratory.
- Không dùng post-hoc power; trước confirmatory run thực hiện prospective precision/power simulation dựa trên train/inner-CV cluster variance.

### 11.5. Sensitivity analyses

- Temporal-unit primary dataset versus multi-scale window aggregation.
- Adjudicated-only versus all trusted labels.
- CVAT-only, legacy-only và combined-source.
- Hidden-trusted included versus stricter visible subset.
- Equal-event weighting versus current window weighting.
- Different session grouping assumptions nếu metadata còn bất định.

## 12. Shortcut, robustness và failure analysis

### 12.1. Shortcut controls

- Train source classifier từ candidate embeddings/features; high source separability là evidence domain encoding.
- Background-only/local-context với actor masked.
- Actor-only với background giảm mạnh.
- Temporal shuffle.
- Repeat một frame cho toàn sequence.
- Remove ROI/social channels.
- Compare full image với grayscale/color-jitter controls.

Nếu background-only hoặc temporal-shuffle giữ performance cao, claim về temporal behavior mechanism phải giảm hoặc model/data phải sửa.

### 12.2. Runtime robustness

- Bbox jitter theo detector-error distribution thật.
- Frame drop/gap và variable FPS.
- Partial occlusion/hidden tiers.
- Tracking ID switch episodes.
- ROI annotation perturbation.
- Lighting/compression/domain shift.

Đánh giá riêng oracle GT boxes và runtime tracker boxes. Không quy lỗi behavior model cho detector nếu hai setting chưa tách.

### 12.3. Error taxonomy

Mỗi major error cluster gắn với một action:

- Label ambiguity -> re-review/guideline.
- Missing visual signal -> context/pose/flow.
- Source shortcut -> split/domain correction.
- Temporal boundary -> target/window policy.
- Tracking contamination -> quality mask/noise augmentation.
- Model capacity/calibration -> architecture/loss/threshold.

Preserve representative failures thành regression set, không chỉ hình minh họa chọn lọc.

## 13. Reproducibility và artifact governance

Mỗi run lưu:

- Dataset snapshot ID, SHA256 và row counts.
- Split/fold manifest hash.
- Code commit SHA và dirty-worktree state.
- Full config, seed, package lock, OS/GPU.
- Feature schema và normalization state.
- Crop/temporal/mask contract.
- Training log, best/last checkpoint.
- OOF predictions và per-fold metrics.
- Calibration artifact.
- Runtime benchmark.

Artifact tối thiểu cho paper:

```text
research_protocol.yaml
literature_search_log.csv
literature_comparison_matrix.csv
dataset_snapshot.json
recording_group_manifest.csv
annotation_guideline.md
inter_rater_reliability.json
experiment_registry.csv
fold_predictions.parquet
statistical_analysis.json
model_card.md
dataset_card.md
reproduction_commands.md
```

Generated data/model artifacts không nhất thiết commit vào Git, nhưng manifest/checksum/config phải versioned.

## 14. Ethics, governance và reporting

Trước submission cần document:

- Nguồn video, mục đích thu thập và observational/interventional status.
- Institutional animal ethics approval hoặc lý do waiver/not applicable.
- Farm/facility permission, data ownership và redistribution rights.
- Animal welfare risk khi dùng prediction; model không thay thế veterinary judgment.
- Privacy của facility/personnel có thể xuất hiện trong video.
- Applicability của ARRIVE 2.0 hoặc guideline animal-research phù hợp; không claim compliance trước khi checklist được kiểm.
- Model limitations theo cohort, camera, age/weight, housing và behavior ontology.

Nếu raw video không thể công bố, cần cung cấp tối đa metadata, split IDs, annotation schema, code và synthetic/sample artifacts trong giới hạn quyền dữ liệu.

## 15. Manuscript architecture

### 15.1. Logical argument

```text
Problem: heterogeneous temporal labels + group context + source shift
  -> Gap: existing actor-only/window-random evaluation may miss context and overestimate generalization
  -> Method: native-unit harmonization + masked multimodal fusion
  -> Test: nested session-safe paired evaluation
  -> Evidence: primary delta + ablations + shortcut controls + uncertainty
  -> Boundary: same-domain/session claims unless external cohort validates broader use
```

### 15.2. Main figures

1. Dataset lineage và unit hierarchy.
2. Distribution theo behavior, source, date/session và review status.
3. Proposed multimodal architecture và masks.
4. Nested session-safe evaluation design.
5. Primary model comparison với cluster CIs.
6. Confusion/calibration/source robustness.
7. Qualitative successes/failures, chọn theo rule chứ không cherry-pick.

### 15.3. Main tables

1. Dataset composition theo independent clusters, không chỉ frames/windows.
2. Closest prior work và evaluation protocol.
3. Baseline/proposed performance với 95% CIs.
4. Predeclared ablations.
5. Per-class/source/session robustness.
6. Compute/latency/model-size tradeoff.

### 15.4. Supplementary material

- Full label/source/date counts.
- Hyperparameter search space và budget.
- All seeds/folds.
- Inter-rater results.
- Extended confusion pairs.
- Null/failed ablations.
- Detailed data and model cards.

## 16. Execution phases và go/no-go gates

| Phase | Deliverable | Go condition | No-go/blocker |
|---|---|---|---|
| R0 | Metadata + literature audit | Biological/session mapping và closest-work matrix | Không biết independent units hoặc novelty |
| R1 | Frozen protocol | RQs, SESOI, endpoints, split, stats khóa | Protocol đổi sau test access |
| R2 | Label-quality package | Gold test review + reliability CI | Review coverage quá thấp/ambiguous |
| R3 | Session-safe data package | 0 recording-group leakage, native temporal units | Same session cross folds |
| R4 | Baseline study | B0-B6 cùng folds/budget | Baseline yếu hoặc unfair |
| R5 | Proposed model study | P1/P2 + core ablations + OOF predictions | Gain không vượt SESOI/CI |
| R6 | Robustness/statistics | Shortcut controls, cluster CI, sensitivity | Shortcut hoặc unstable result |
| R7 | External validation | Independent cohort/farm/camera | Chỉ claim in-domain nếu thiếu |
| R8 | Manuscript/release | Claim-evidence map, reproducibility package | Unsupported novelty/generalization |

## 17. Claim-to-evidence matrix

| Candidate claim | Evidence bắt buộc | Không đủ nếu chỉ có |
|---|---|---|
| Pipeline không mất dữ liệu | Row-preservation audits + checksums | Một output CSV tồn tại |
| Model tốt hơn baseline | Paired session-safe OOF delta + CI + fair budget | Accuracy trên smoke split |
| Context giúp interaction | Predeclared actor-only comparison + bystander FP | Một vài qualitative examples |
| Temporal model dùng motion | Temporal-shuffle/repeat-frame controls | Attention heatmap |
| Generalizes to unseen sessions | Session-disjoint outer folds | Random/video-key split |
| Generalizes to new farm/cohort | Locked external validation | CVAT/legacy source split |
| Novel/SOTA | Literature matrix + comparable baselines/protocol | Không tìm thấy paper giống trong search sơ bộ |

## 18. Tiêu chí Q1-oriented và Q2-oriented

### Q1-oriented readiness

Thông thường cần đồng thời:

- Novelty được literature review xác nhận.
- Session-safe nested evaluation.
- External cohort/farm/camera validation hoặc một methodological contribution đủ mạnh với claim rất chặt.
- Gold labels và inter-rater evidence.
- Strong published baselines, ablation, robustness và uncertainty.
- Reproducibility package và ethics/data governance đầy đủ.

### Q2-oriented readiness

Có thể chấp nhận in-domain study hơn nếu:

- Research question tập trung và contribution rõ.
- Không pseudoreplication/session leakage.
- Baselines/ablations/stats đúng.
- Limitations và generalization boundaries trung thực.
- Không claim cross-farm hoặc SOTA không có bằng chứng.

Quartile phụ thuộc journal, năm và subject category; protocol tốt không bảo đảm quartile hoặc acceptance.

## 19. Next actions theo thứ tự bắt buộc

1. Tạo canonical biological/cohort/farm/camera/date/session metadata table.
2. Xây lại publication split theo recording group và chứng minh leakage bằng 0.
3. Chốt primary temporal-unit dataset, không dùng window làm independent test unit.
4. Thực hiện literature search và closest-work matrix trước khi khóa contribution.
5. Thiết kế double-review sample và inter-rater precision target.
6. Freeze RQ1/RQ2, SESOI, endpoints, ablations và statistical plan.
7. Chạy full tabular/spatial baselines trên nested session-safe folds.
8. Chỉ sau đó train image và multimodal candidates.
9. Thực hiện shortcut/robustness controls trước khi viết claim.
10. Thu thập hoặc khóa external validation set nếu mục tiêu là cross-domain/Q1-strength claim.

## 20. Machine-readable gate status

Protocol version 1.2 bổ sung một gate kiểm tự động:

- Config: `configs/classification_v2/paper_grade_protocol_v1.json`.
- Checker: `scripts/dev_tools/check_classification_v2_paper_grade_protocol.py`.
- Audit: `outputs/classification_v2/paper_grade_protocol/paper_grade_protocol_audit.json`.

Gate này không thay thế peer review khoa học hoặc literature review. Nó kiểm rằng claim boundary đang khóa ở mức Q2 mạnh, các tài liệu bắt buộc tồn tại, training snapshot/trainer contract/source-domain/native-OOF artifacts pass, confusion pairs và ablation ladder đủ tối thiểu. Nếu gate fail, kết quả model chỉ được xem là engineering smoke, không được dùng làm claim paper.
