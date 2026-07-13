# Classification V2 Core Scientific Execution Goal Prompt

Use the prompt below as the authoritative execution request for the next
`classification_v2` model-development goal.

```text
Bạn là senior machine-learning, computer-vision và scientific-audit agent cho:

C:\Users\ironh\Downloads\PIG_Behavior_Project

Mục tiêu là triển khai classifier hành vi lợn 10 lớp mạnh, tái lập được,
leakage-safe và đủ chặt chẽ cho claim Q2 trong cùng acquisition domain.

Luôn chạy project command trong CMD:

cd /d C:\Users\ironh\Downloads\PIG_Behavior_Project
set PYTHONPATH=%CD%\src
set PY=C:\Users\ironh\anaconda3\envs\pig_project\python.exe

Không sửa dữ liệu dưới `data\`. Không chạy full training hoặc full OOF trước
khi toàn bộ gate tương ứng PASS và authorization khớp exact hashes.

============================================================
I. AUTHORITY VÀ PRECEDENCE
============================================================

Đọc trước khi làm việc:

1. `AGENTS.md`.
2. `.agents/memory/01_PROJECT_MEMORY_SHORT.md`.
3. `.agents/memory/02_CURRENT_DECISION.md`.
4. `.agents/memory/03_PROJECT_RULES.md`.
5. `.agents/memory/08_WORKFLOW.md`.
6. `docs/CLASSIFICATION_V2_CURRENT_STATE.md`.
7. `docs/CLASSIFICATION_V2_DATA_REBUILD_AND_HUMAN_REVIEW_RUNBOOK.md`.
8. `plans/classification_v2-core-classifier-roadmap.md`.
9. `plans/classification_v2-scientific-performance-upgrade-roadmap.md`.

Precedence bắt buộc:

- Core roadmap P0-P8 là canonical critical path.
- Scientific roadmap M0-M12 là research reference mở rộng.
- Khi hai roadmap xung đột, core roadmap thắng.
- P9, strict five-class, paper reproduction và publication không chặn P0-P8.
- Current-state và current-decision thắng tài liệu lịch sử.
- Data rebuild runbook là authority cho source, review và train-ready lineage.

============================================================
II. GOAL VÀ TRẠNG THÁI HIỆN TẠI
============================================================

Goal objective:

Triển khai đầy đủ critical path P0-P8 của core classifier roadmap; dùng M0-M12
để tăng độ chặt chẽ khi tương thích; xây model 10 lớp tốt hơn baseline cũ bằng
paired recording/session-safe native-unit validation; hoàn thiện code, tests,
audit, lineage và model package mà không vượt qua human-review/full-run gates.

Mười lớp authority:

`drink, eat, fight, social-nose, explore, lying, stand, move, sitting,
playwithtoy`.

Trạng thái bắt đầu phải được xác nhận lại, không hard-code làm kết quả mới:

- technical smoke hiện có 688 frame row, 63 native/review unit và 438 window;
- Hidden v5 mới có 30/5.171 decision resolved;
- behavior review mới có 3/4.670 decision, còn một pending;
- current reviewed lineage chưa train-ready;
- full OOF cũ chỉ là historical engineering baseline;
- user cho phép full có điều kiện, không phải authorization vô điều kiện.

Không tạo fake decision, không auto-accept pending, không dùng old reviewed CSV
để né blocker và không gọi artifact kỹ thuật là human-reviewed final.

============================================================
III. SCIENTIFIC INVARIANTS
============================================================

1. Không sửa, xóa, đổi tên hoặc overwrite raw files dưới `data\`.
2. Không silent-drop hoặc silent-relabel frame, unit, event hay window.
3. Không đưa label, ID, path, fold, policy, `manual_*` hoặc `review_*` vào X.
4. Không tự chọn mọi numeric column; dùng exact feature whitelist có hash.
5. Mọi normalization, prior, class weight, threshold và calibration là fold-local.
6. Không split random theo row/frame/window; group theo recording/video/session.
7. Một native unit và overlapping windows của nó chỉ thuộc một split role.
8. `pig_id` là annotation-local, không phải biological identity xuyên video.
9. Primary metric là pooled OOF native-unit 10-class macro-F1.
10. Mọi paired comparison dùng cùng fold manifest và eligible native units.
11. Partner/ROI/context routing không được đọc target behavior.
12. `social-nose` actor-only; `fight` chỉ direct participants, không bystander.
13. Missing modality luôn có mask nhưng mask không mặc định là behavior evidence.
14. Actor crop dùng letterbox, không square-stretch hình lợn.
15. Training full chỉ đọc reusable audited cache, không seek/crop video lặp lại.
16. Mỗi experiment chỉ thay một principal scientific variable family.
17. Outer OOF prediction không được quay lại tune architecture hay threshold.
18. Claim chỉ là internal known-domain Q2; không claim external generalization.

Identifier blocker phải được giải quyết trước P0 PASS: code hiện dùng
`frame_uid` như scene-frame key. Hoặc migrate rõ thành `scene_frame_uid` cộng
object-level `frame_uid`, hoặc version schema/composite key đầy đủ và giữ
backward-compatible reader. Không deduplicate object rows chỉ bằng `frame_uid`.

============================================================
IV. KHI HUMAN REVIEW CHƯA HOÀN TẤT
============================================================

Human review là dependency thật nhưng không phải lý do dừng mọi engineering.
Trong khi chờ người dùng review, tiếp tục các hạng mục độc lập sau:

1. Versioned path/data-contract plumbing, không fallback canonical.
2. Identifier migration/composite-key tests và row-preservation audit.
3. Exact video resolver assertion cho case `000231_30fps.mp4`.
4. Hidden design-based/clustered uncertainty và predeclared threshold checker.
5. Tách target-informed Hidden enrichment khỏi prevalence estimate.
6. Fixed-6 temporal-view manifest, loader và source/length shortcut probes.
7. Fold-local preprocessing, sampler, losses và lineage registry bằng fixtures.
8. Configurable model factory, masks, forward and missing-modality tests.
9. Pretrained backbone interfaces không cần tải weight trong unit tests.
10. One-batch và tiny synthetic/native-event smoke, không full dataset training.
11. Evaluation/native-collapse/paired-metric code bằng synthetic predictions.
12. Documentation, audit schema và rollback contract cho từng milestone.

Các gate phụ thuộc human data phải giữ `BLOCKED`, không được giả lập PASS:

- Hidden decision coverage/apply;
- behavior decision coverage/apply;
- final reviewed windows/native units;
- immutable active training snapshot;
- active-lineage one-fold pilot và full OOF.

Khi gặp blocker này, chuyển sang task độc lập kế tiếp. Chỉ báo blocked cho toàn
goal khi không còn task code/test/documentation độc lập có thể tiến hành.

============================================================
V. CANONICAL EXECUTION P0-P8
============================================================

P0 - Freeze data, baseline và shortcut evidence:

- sửa code blockers trước snapshot;
- tạo một versioned reviewed lineage sau khi human gates PASS;
- khóa source hashes, fold manifest, X whitelist và temporal views;
- reconcile baseline cũ chỉ như engineering control;
- xuất class/source/fold/context support và source/missingness probes;
- không promote canonical mixed-lineage artifacts.

P0 chỉ PASS khi snapshot cùng `RUN_ID`, không missing/duplicate/silent loss,
reviewed rows bằng enhanced rows và mọi contract/checker fail-closed.

P1 - Strong visual baseline:

- V0: pretrained ResNet18, 160x160;
- V1: pretrained ResNet18, 224x224;
- V2: pretrained ResNet34, 224x224;
- V0 -> V1 chỉ đổi resolution;
- V1 -> V2 chỉ đổi backbone;
- tìm architecture bằng ResNet18; chỉ finalist mới trả chi phí ResNet34.

P2 - Temporal baseline trên cùng visual setup:

- T0 masked mean hoặc attention pooling;
- T1 masked TCN;
- T2 small Transformer chỉ khi T1 chưa đủ;
- primary view là `fixed6_observed_time`;
- `fixed6_normalized_phase` là shortcut diagnostic;
- native 6/16 chỉ là ablation, không tự động là primary protocol.

P3 - Geometry, motion và ROI:

- actor-temporal baseline + geometry;
- sau đó + motion;
- sau đó + all-class ROI continuous relations;
- ROI được encode đối xứng, không chọn feeder/drinker/toy theo y;
- augmentation phải transform đồng bộ image/bbox/ROI/relation tensors;
- thêm parameter-matched actor-only wider-MLP control.

P4 - Social và interaction context theo tầng:

- S1 numeric partner relations;
- S2 top-K label-independent partner set encoder;
- S3 actor-partner union crop;
- S4 full-frame context chỉ khi S3 chưa đủ;
- luôn chạy actor-only, availability-only, real-context và matched-subset controls;
- dùng label-independent modality dropout và báo missingness theo fold/source.

P5 - Chọn đúng một imbalance policy:

- L0 event-balanced standard CE;
- L1 effective-number CE;
- L2 Balanced Softmax với training-fold prior;
- không ghép focal, aggressive sampler và class weight cùng lúc;
- deferred reweighting, focal và logit adjustment chỉ sang optional research
  khi L0-L2 đã có bằng chứng chưa đủ;
- balancing unit là unique native event, không phải overlapping window count.

P6 - Confusion-driven hierarchy:

- chỉ triển khai sau strong P3/P4 baseline và development error analysis;
- review shortlist theo confusion/source/quality strata;
- auxiliary targets có confidence và mask;
- final 10-class head luôn direct-supervised;
- không đưa auxiliary argmax vào final head và không hard cascade mặc định;
- so sánh no-hierarchy, auxiliary-only và soft-fusion bằng paired units.

P7 - Candidate lock:

- capacity-confirm retained ResNet18 branches bằng ResNet34;
- full-OOF shortlist tối đa F0 actor-temporal, F1 spatial/ROI, F2 final
  multimodal/social và F2-no-hierarchy khi cần;
- khóa config, folds, seeds, loss, augmentation, temporal view và metric policy;
- tạo immutable finalist lock, runtime audit và exact authorization packet.

P8 - Full grouped OOF và model package:

- chỉ chạy sau static, one-batch, tiny-overfit, resume, runtime và one-fold PASS;
- mỗi fold ghi indices, priors, weights, event coverage, checkpoint, prediction,
  runtime, VRAM và toàn bộ hashes;
- collapse window probabilities về native unit bằng frozen policy;
- prediction count phải khớp manifest, mỗi native unit đúng một OOF prediction;
- báo pooled global, rare, interaction, ROI, posture và locomotion macro-F1;
- báo per-class/source/video/recording, calibration và paired uncertainty;
- package checkpoint inference-compatible kèm preprocessing và label order.

P9 - Optional, không chặn P0-P8:

- strict five-class ResNet18/ResNet34 comparison;
- Bergamini-inspired hybrid khi calibration/metadata đủ;
- paper package, causal deployment và integration experiments;
- graph/pose/self-supervised branches sau khi core candidate đã khóa.

Không dùng M7 five-class hoặc M12 publication làm prerequisite cho classifier
10 lớp. Không chạy cả native LORO và Q2 outer/inner rồi chọn metric tốt hơn.

============================================================
VI. CÁCH DÙNG M0-M12 MÀ KHÔNG LỆCH CORE ROADMAP
============================================================

- M0 hỗ trợ P0 baseline reconciliation, không promote lineage cũ.
- M1 ontology/metadata hỗ trợ P0; five-class contract chuyển sang P9.
- M2 hỗ trợ P5 nhưng initial loss scope chỉ L0-L2 như core roadmap.
- M3 hỗ trợ P1 cache/backbone controls.
- M4 hỗ trợ P2, nhưng fixed-6 là primary và native length là ablation.
- M5 chỉ chạy ở P6 sau confusion-driven review evidence.
- M6 hỗ trợ P3/P4 theo staged social-context ladder.
- M7 là P9 optional, không nằm trên critical path.
- M8 là verification ladder bắt buộc trước mọi pilot.
- M9 phải dùng core controlled matrix, không đổi backbone/resolution/modality/loss
  trong cùng một unexplained jump.
- M10 tương ứng P7 lock/authorization.
- M11 tương ứng P8 full grouped OOF.
- M12 chỉ chạy sau P8 và không được sửa confirmatory labels/model selection.

Không triển khai đồng thời toàn bộ loss, hierarchy, full-frame CNN, Transformer
và ResNet34. Mỗi milestone phải chứng minh gain hoặc correctness riêng; nhánh
không có gain, không ổn định hoặc dùng shortcut phải dừng và lưu negative result.

============================================================
VII. IMPLEMENTATION VÀ REPOSITORY DISCIPLINE
============================================================

1. Đọc file, import graph, schema và tests trước khi patch.
2. Sửa module chính dưới `src/pig_behavior/classification_v2`, không output hack.
3. Giữ scripts trong numbered workflow `00_*` đến `09_*`; không thêm wrapper.
4. Public function/nontrivial model module có docstring ngắn nêu input, output,
   masks, key alignment và leakage assumption.
5. Dùng config/model factory chung cho ablations, không copy-paste model runners.
6. Mọi modality branch có encoded tensor, availability mask, quality mask và
   shape validation; model vẫn chạy khi branch thiếu.
7. Checkpoint schema bind model config, snapshot, cache, whitelist và fold hashes.
8. Mỗi fold ghi output riêng để hỗ trợ remote/rented GPU và merge có audit.
9. Local RTX 3050 dùng compile, forward, tiny-overfit và one-fold smoke.
10. Remote GPU được phép cho ResNet34/full OOF sau cùng gates và authorization.
11. Không tải pretrained weight trong unit tests; mock interface hoặc dùng
    explicit no-download path. Pilot thật phải ghi exact weight enum/hash.
12. Không refactor lớn cùng lúc với thêm thuật toán.

Trước mỗi edit, gửi update ngắn nêu file và mục đích. Dùng `apply_patch` cho
manual edits. Không dùng redirect, heredoc, temporary overwrite hoặc whole-file
rewrite. Không revert thay đổi người dùng và không dùng `git reset --hard`.

============================================================
VIII. VALIDATION LADDER VÀ FULL-RUN GATE
============================================================

Mỗi semantic change chạy theo đúng thứ tự:

1. overlong-line scan và `git diff --check`;
2. `py_compile` changed files và `compileall` related package;
3. import test và focused pytest;
4. synthetic schema/shape/leakage fixtures;
5. one-batch forward/backward;
6. deterministic repeat và checkpoint resume;
7. overfit 16-64 unique native events;
8. one fold, one epoch với class/source coverage;
9. runtime, peak VRAM, cache-only I/O và prediction-schema audit;
10. representative development folds;
11. finalist lock;
12. exact preflight, authorization và full OOF.

Short PASS không tự động cho phép full ở stage kế tiếp. Semantic config, data,
resize, temporal view, split, loss hoặc model thay đổi thì phải chạy lại các gate
liên quan. Không dùng full run để dò lỗi.

Full OOF chỉ được chạy khi:

- human Hidden và behavior decisions đã complete/fail-closed PASS;
- versioned reviewed snapshot và caches đã hash-lock;
- P0-P7 đều PASS;
- one-fold measured runtime chứng minh command hợp lý;
- exact config/code/data/cache/fold hashes nằm trong launch packet;
- user authorization áp đúng signature đó;
- execution gate trả `allowed=true`.

============================================================
IX. EVALUATION VÀ PROMOTION
============================================================

Primary evaluation là pooled OOF native-unit macro-F1 theo global 10-class
order. Fold-level macro-F1 chỉ dùng supported classes và luôn công bố support.

Required reports:

- accuracy, balanced accuracy, weighted-F1, macro recall;
- NLL, Brier score và ECE;
- per-class precision, recall, F1 và support;
- global/rare/interaction/ROI/posture/locomotion macro-F1;
- confusion matrix và predeclared confusion-pair metrics;
- per-source/video/recording/session/quality slices;
- class x fold và source x fold support;
- parameter count, throughput, runtime và peak VRAM;
- paired cluster bootstrap hoặc paired recording/native-unit uncertainty.

Promotion cần đồng thời:

- global macro-F1 tăng so với paired baseline;
- rare-class macro-F1 không giảm;
- ít nhất một confusion group mục tiêu cải thiện;
- real context tốt hơn availability-only trên matched subset;
- không có severe per-class recall collapse;
- gain không biến mất trên source-balanced/matched analysis;
- runtime/VRAM nằm trong declared compute profile.

Ngưỡng effect/guardrail phải predeclare sau P0 reconciliation và trước xem
confirmatory outer predictions. Negative result vẫn phải đăng ký và giữ audit.

============================================================
X. LINEAGE, ARTIFACT VÀ COMMIT POLICY
============================================================

Mỗi run lưu tối thiểu:

- run ID, experiment, phase, fold và seed;
- code SHA và dirty-worktree status;
- dataset/snapshot/cache/fold/whitelist/config hashes;
- architecture, pretrained enum, resolution, temporal view và modalities;
- loss, sampler, optimizer và augmentation;
- environment, GPU, VRAM, AMP, runtime và peak memory;
- checkpoint, prediction, metrics, audit paths, status và failure reason.

Mỗi milestone có:

1. starting SHA và small plan-ledger entry `IN_PROGRESS`;
2. focused implementation;
3. checker/audit JSON;
4. tests và diff summary;
5. one achievement commit sau PASS;
6. rollback instruction;
7. ledger update `PASS`, `FAIL` hoặc `BLOCKED`.

Trước mọi code commit chạy:

`rg -n "^.{101,}$" <changed-files>`, `git diff --check`, compile và focused tests.

Không commit large cache, checkpoint hoặc predictions trừ khi repository policy
yêu cầu. Chỉ stage file thuộc achievement hiện tại; không stage unrelated user
changes. Không sửa số liệu/audit cũ để làm model mới trông tốt hơn.

============================================================
XI. STOP CONDITIONS
============================================================

Dừng stage hiện tại ngay khi có một trong các lỗi:

- raw `data\` thay đổi;
- row/native-unit/prediction loss không có reason;
- duplicate hoặc cross-split native unit;
- target/review/path/source ID lọt vào X;
- fold-local transform đọc validation/test;
- checkpoint/resume hash mismatch;
- cache key/tensor/index không one-to-one;
- source/length/availability gần như suy ra label mà chưa có mitigation;
- NaN/Inf, constant logits, nondecreasing tiny-overfit loss hoặc OOM chưa audit;
- outer OOF được dùng để tune;
- agent chuẩn bị chạy full khi human/P0-P7/authorization chưa PASS.

Không che lỗi bằng broad `try/except`, row drop, fake decision, stale canonical
fallback hoặc đổi threshold sau khi xem confirmatory result.

============================================================
XII. FIRST EXECUTION ORDER
============================================================

Ngay khi nhận prompt này:

1. Đọc authority files và báo chính xác file đã đọc.
2. Kiểm `git status`, active goal và current-state gates.
3. Không sửa/revert unrelated worktree changes.
4. Reconcile hai roadmap thành một phase ledger P0-P8 với M mapping.
5. Audit code hiện có so với P0 deliverables và lập PASS/FAIL bằng evidence.
6. Chọn blocker code độc lập human review có dependency sớm nhất.
7. Commit ledger/start marker riêng nếu project policy yêu cầu.
8. Implement, test và commit đúng một achievement.
9. Cập nhật phase ledger rồi tiếp tục achievement kế tiếp tự động.
10. Không dừng ở plan nếu còn task độc lập an toàn có thể làm.

Nếu goal tool khả dụng và không có unfinished goal, tạo goal bằng objective ở
mục II. Nếu đã có paused/active unfinished goal, không mark complete giả; báo
goal conflict và tiếp tục latest user-authorized scope theo project plan.

============================================================
XIII. COMPLETION REPORT
============================================================

Sau mỗi achievement báo:

1. phase/milestone và PASS/FAIL/BLOCKED;
2. scientific hypothesis hoặc contract vừa xử lý;
3. file đã đọc, tạo và sửa;
4. behavior trước/sau và behavior cố ý không đổi;
5. commands/tests đã chạy và kết quả;
6. row/key/schema/hash/leakage/runtime evidence liên quan;
7. commit SHA;
8. rollback instruction;
9. dependency tiếp theo;
10. human action còn cần, nếu có.

Chỉ mark toàn goal complete khi P0-P8 đều thực sự PASS, full native-unit OOF và
model package hoàn tất, không còn required work. Nếu human review ngăn P0/P8,
giữ đúng trạng thái incomplete/blocked sau khi mọi task độc lập đã cạn; không
đổi definition of done để kết thúc sớm.

Bắt đầu thực hiện ngay theo mục XII.
```

## End
