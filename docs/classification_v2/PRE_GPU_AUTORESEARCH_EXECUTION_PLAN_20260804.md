# Classification V2 — kế hoạch thực thi trước GPU trả phí và autoresearch

Ngày lập: 2026-08-04  
Phạm vi: Classification V2, dữ liệu đã review; không chạm Behavior ledger/GUI
đang hoạt động.  
Trạng thái khi lập kế hoạch: "PAUSE_BEFORE_LEARNED_A12_AND_NATIVE_OOF".

## 1. Mục tiêu và ranh giới

Mục tiêu không phải chỉ làm cho một lệnh train chạy. Mục tiêu là một pipeline
mà mỗi kết luận truy được tới code, dữ liệu, split, seed, checkpoint và phép
đo; sau đó mới cho phép thuê GPU và chạy autoresearch.

Ba mốc phải tách biệt:

1. "LOCAL_MODEL_RUNNING": mô hình thật chạy trên RTX 3050, có loss, gradient,
   validation prediction, checkpoint và resume; chưa phải kết quả khoa học.
2. "LOCAL_SCIENTIFICALLY_SCREENABLE": learned A12, native-unit evaluation,
   one-fold pilot, resume/profile và đủ điều kiện sàng lọc ablation giới hạn.
3. "PAID_GPU_AUTHORIZED": mọi hard gate trước GPU trả phí PASS, có permit mới
   gắn hash; sau đó mới chạy remote pilot rồi claim-grade OOF.

Không làm trong kế hoạch này:

- không mở GUI, không ghi quyết định review, không sửa annotation/XML;
- không đưa source_type, reviewer, review status, ID, path hoặc target-derived
  fields vào X;
- không random split, không chọn model bằng outer-test/OOF;
- không chạy full OOF, full autoresearch hay thuê GPU ở giai đoạn hiện tại;
- không xem fixed camera là lỗi leakage; đó là giới hạn external validity;
- không đưa Hidden trực tiếp vào X hoặc biến Hidden thành nhãn hành vi.

## 2. Authority hiện tại phải được bind ở mọi run

Các giá trị sau lấy từ audit hiện hành, không suy ra từ tên file cũ:

| Hạng mục | Authority hiện tại |
|---|---|
| Main SHA | d4fb797b6a07e5cfea1812d9df9f2c725ffc8533 |
| Classification V2 code SHA | 0d1a3f1219b8963df752b15306c7ceabd9be8812 |
| Classification tree hash | c021e1a654e9b15a2be5f610523932f9c045cf6e |
| Audit package | pre_gpu_autoresearch_q2_47103f6_20260804_133801 (path below) |
| Reviewed snapshot SHA-256 | ab86e2e04267cfdc8248f9bdb8774615479d67a3589f7a25844bb1a4c93a639e |
| Effective window index SHA-256 | hash listed below |
| Native/frame/window counts | 33,355 / 245,680 / 165,305 |
| Trainable/excluded windows | 159,410 / 5,895 |
| Views | T6, T8, T12, T16 |
| Predictive spatial width | 46 = 4 bbox + 2 shape + 12 motion + 18 ROI + 10 social |
| Motion authority | classification_v2.motion_tensor.v2, width 12 |
| Outer split | four-fold date-grouped; no random window split |

Các file trong plan_model vẫn là authority cho hypothesis, effect threshold và
tên experiment nếu không mâu thuẫn; SHA và trạng thái provisional cũ không được
dùng thay current authority. Mỗi run phải ghi rõ disposition của tài liệu cũ.

Audit directory đầy đủ:
`outputs/classification_v2/model_readiness_audit/`
`pre_gpu_autoresearch_q2_47103f6_20260804_133801`.

Effective window index SHA-256:
`810071b311ebc008d420c5873fea11f2369f9ec8edfc3b2dc2958635b30ac7f1`.

## 3. Gate hiện tại

Đã PASS: G01–G09 và G11.

Còn BLOCKED: G10 learned source-shortcut, G12 paired social, G13 BALANCED,
G14 native OOF/calibration, G15 real resume, G16 complete compute profile,
G17 reviewed autoresearch, G18 scientific result package và G19 paid-GPU
permit.

Smoke B0–B3 hiện tại chỉ là contract evidence; không phải behavior metric và
không tự mở khóa GPU trả phí.

## 4. Thứ tự phase và tiêu chí PASS

Chỉ một phase được IN_PROGRESS. Sau mỗi phase phải lưu artifact hash và kết quả
checker độc lập.

### P0 — re-freeze run envelope

Mục đích là ngăn dirty main hoặc output cũ đi vào thí nghiệm.

Thực hiện:

- tạo worktree riêng từ đúng Classification authority;
- ghi branch, main SHA, code SHA, tree hash, Python/PyTorch/CUDA/GPU;
- hash snapshot, window index, RGB cache, spatial schema, split và event weights;
- tạo run ID duy nhất và thư mục output rỗng;
- ghi run_manifest.json trước khi load model.

PASS:

- mọi hash resolve và không đổi trước/sau run;
- tracked code trong worktree sạch;
- không collision output;
- model, view, resolution, modalities, loss, seed và pretrained enum đều
  explicit;
- forbidden-feature, causality và training-mass audit chạy lại và PASS.

STOP khi có authority conflict, hash thiếu, tracked code dirty, collision hoặc
input thay đổi.

### P1 — learned A12

Đây là blocker reasoning-heavy kế tiếp. Feature-only source decoder hiện
decodable mạnh (pooled balanced accuracy khoảng 0.911; fold khoảng 0.896 và
0.916), nhưng chưa chứng minh learned behavior model dựa vào source.

Controls bắt buộc, dùng cùng fold/preprocessing/seed:

1. source-balanced date-safe train/eval cho mọi baseline được giữ;
2. source-stratified metrics cho CVAT và legacy, kèm support;
3. feature-only source decoder từ đúng X whitelist;
4. source probe trên representation/gate-weight, không có source field;
5. mean-only/simple-proxy controls cho border, padding, resolution;
6. shuffled-source negative control; probe phải ở chance trong CI đã khai báo;
7. train-one-source/test-other ở nơi date structure cho phép;
8. audit xóa direct source identifiers và filename/overlay/cache confound.

A12 PASS:

- mọi control hoàn thành;
- forbidden source field trong X bằng 0;
- shuffle control cho kết quả chance;
- source-balanced date-safe không đảo ranking;
- gain còn practical (macro-F1 tối thiểu +0.02 hoặc target recall +0.03);
- source gap và gate probe được báo cáo kèm uncertainty;
- không credit modality gain nếu biến mất dưới source balancing.

A12 FAIL:

- source metadata vào X; hoặc shuffle control không chance; hoặc headline gain
  biến mất material dưới source-balanced date-safe. Khi đó chỉ được giữ như
  source-sensitive engineering observation, không được claim contribution.

A12 INCONCLUSIVE:

- support không đủ cho cross-source hoặc probe không ổn định. Không thuê GPU;
  mở rộng bounded control hoặc thu hẹp claim.

Outputs tối thiểu: a12_control_manifest.json, source-stratified/balanced
predictions, shuffled-source report, feature/gate probe report, image-cache
audit, paired comparison và quyết định PASS/FAIL/INCONCLUSIVE.

### P2 — one-fold real local pilot

Chỉ chạy sau P0 và contract của P1. Đây là lần đầu tiên có thể nói “model chạy
local”, nhưng chưa được gọi là kết quả paper.

Giữ T6 target semantics, event/native-unit mass correction và Hidden
exclusion/zero-weight semantics hiện hành. Chạy B0 -> B1 -> B2 -> B3, một seed,
một development fold; batch phải có cả CVAT và legacy nếu dữ liệu cho phép.

PASS LOCAL_MODEL_RUNNING:

- real CVAT và legacy batch load được;
- logits, loss, gradient và optimizer state finite;
- không NaN/Inf, shape drift, forbidden X hoặc future frame;
- cùng seed/config cho deterministic repeat trong tolerance đã khai báo;
- checkpoint reload tái tạo prediction và optimizer state;
- validation predictions link đúng fold/config/checkpoint;
- tiny-overfit cải thiện loss và đạt target diagnostic đã predeclare.

Kết quả one-fold là diagnostic. Không dùng để tuyên bố model A tốt hơn model B.

### P3 — real interrupted-run/resume

Một fixed fold/seed/config; dừng tại global step định trước, rồi resume vào output
directory mới.

PASS G15:

- checkpoint có model, optimizer, scheduler, AMP scaler, epoch/step, sampler
  state và mọi RNG state;
- resume kiểm tra code/config/data/schema/fold hashes;
- uninterrupted và interrupted+resumed có cùng step và prediction ordering;
- FP32 max parameter/logit/loss delta <= 1e-6; AMP dùng tolerance đã ghi và
  luôn finite;
- checkpoint/artifact hashes và registry links đầy đủ;
- collision hoặc hash mismatch fail closed.

Synthetic resume không còn đủ để đóng G15 sau phase này.

### P4 — bounded compute/memory profile

Dùng B3 (hoặc model nhỏ nhất được giữ) trên RTX 3050 Laptop 4-GB class device.
Tách thời gian loader/cache, host-to-device, forward, backward, optimizer,
model-only và end-to-end. Đo FP32/AMP agreement, bounded batch search,
gradient accumulation và OOM recovery có kiểm soát.

PASS G16:

- peak allocated/reserved VRAM dưới safety margin khai báo (mặc định 85% VRAM
  vật lý hoặc giới hạn chặt hơn);
- không memory growth, leak hoặc OOM không phục hồi;
- AMP finite và phù hợp FP32 trong tolerance;
- effective batch = batch x accumulation được ghi rõ;
- timing có warm-up, p50/p95, evaluation và checkpoint overhead;
- disk estimate gồm cache, checkpoint, logs và failed-run retention;
- remote runtime/cost dựa trên số đo thật, không dựa projected parameters.

### P5 — native-unit OOF, calibration và uncertainty

Dùng immutable date-grouped folds. Collapse overlapping windows thành đúng một
prediction cho mỗi native temporal unit trước scoring.

PASS G14:

- mỗi eligible native unit có đúng một prediction hoặc exclusion có audit;
- bốn outer folds dùng cùng manifest và class order;
- outer predictions không dùng để tune architecture/loss/threshold;
- báo pooled ten-class macro-F1, supported-fold macro-F1, macro recall,
  balanced accuracy, per-class metrics, confusion pairs, rare PR-AUC, NLL,
  Brier, ECE, reliability, confidence-coverage và selective risk;
- CI dùng recording-date cluster bootstrap; paired comparison dùng cùng held-out
  units;
- support thiếu được báo, không impute hoặc lặng lẽ drop.

Một-fold local chỉ diagnostic; claim-grade OOF cần đủ bốn fold, finalist lock,
reviewed authority phù hợp và authorization rõ ràng.

### P6 — screening ladder

Mỗi trial đổi đúng một principal family, giữ fold, preprocessing, mass rule và
seed policy giống nhau. Giữ cả kết quả dương và âm.

1. B0/B1/B2/B3: geometry cho lying/sitting; motion cho stand/move/explore.
2. L0–L7: freeze B3; priors/weights chỉ từ train-fold native units; không đổi
   loss và sampler đồng thời trong so sánh đầu.
3. ROI R0/R1/R2: scene maps hiện hành; target ROI fields không vào X; phân biệt
   map unavailable, feature invalid, no target ROI và actor not near ROI.
4. History H0/H6/H12/H24: T6 target cố định; history label không vào X;
   future-frame bằng 0; báo ALL_ELIGIBLE và COMMON_MATCHED_COHORT.
5. Social S0/S1/S2: S3 GAT chỉ mở nếu S2 thắng paired gate; test permutation và
   missing partner.
6. Fusion F0/F1/F2/F3: gate chỉ dùng quality/availability, không source/reviewer.
7. BALANCED: chỉ lắp module đã retained; không mặc định “bật tất cả”.

Promotion chỉ khi correctness, tiny-overfit, resume, native-unit pairing, A12,
runtime và hypothesis gate PASS; practical effect tối thiểu là macro-F1 +0.02,
target recall +0.03, hoặc stability/calibration win đã predeclare. Nếu không,
ghi DROP hoặc INCONCLUSIVE.

### P7 — posture/quality branch (chỉ khi authority đủ)

Live B0–B3 contract hiện không tự bao gồm posture head. Nếu cần so behavior-only
với behavior+posture, phải đăng ký một scientific family riêng sau P5:

- behavior head vẫn được supervise trực tiếp;
- posture chỉ là auxiliary target/head khi reviewed authority và mask rõ;
- không hard-cascade posture argmax vào behavior;
- cùng fold/seed/preprocess; so paired native-unit behavior, calibration,
  runtime và missing-posture handling;
- Hidden vẫn là observation quality/weight, không phải behavior uncertainty và
  không phải predictive shortcut.

Nếu posture authority/loader/split chưa PASS thì branch BLOCKED, không thêm vào
autoresearch order.

### P8 — autoresearch eligibility

Harness hiện đã fail-closed và bind SHA/tree, nhưng chỉ diagnostic cho tới khi
P1–P5 PASS.

Trước khi bật execution phải xác minh:

- policy/candidate manifests immutable và hash-bound;
- semantic diff chỉ cho một family;
- timeout/resource/output isolation/failure retention hoạt động;
- không auto-promotion; mỗi promotion có decision đọc được;
- candidate qua correctness, A12, paired native-unit và runtime gate;
- outer OOF không dùng tune trial kế tiếp.

Search order bị khóa:
B0–B3 -> L0–L7 -> R0–R2 -> H0/H6/H12/H24 ->
S0/S1/S2 (S3 gated) -> F0–F3.

### P9 — paid-GPU permit, remote pilot và claim-grade OOF

G19 chỉ PASS khi G01–G10, G14, G15, G16, G17, G18 PASS và label authority
được đánh dấu phù hợp với loại claim. Permit mới phải bind code SHA/tree, data/
schema hashes, fold manifest, finalist config, seeds, environment, hardware,
output roots và budget.

Sau permit:

1. remote pilot một fold/seed;
2. verify hash, resume và prediction count;
3. mới chạy isolated outer-fold jobs;
4. merge chỉ immutable fold outputs;
5. chạy calibration/statistics từ prediction set frozen;
6. promote checkpoint sau khi result package PASS.

Thay đổi bất kỳ permit input nào phải invalidate permit và quay lại P0.

## 5. Record bắt buộc của autoresearch

Mỗi trial ghi: run ID, parent, một family thay đổi, code/data/schema/split/weight
hash, fold/seed/view/history/modalities/loss/pretrained enum, source controls,
native prediction hash, support/confusion/calibration/date CI, params/MACs/VRAM/
latency, failure reason và decision RETAIN/DROP/INCONCLUSIVE. Không reuse run ID,
không overwrite fold, không resume với hash khác.

## 6. Điều kiện cho báo cáo Q2

Result package phải có authority/lineage, phân phối dữ liệu/class, support theo
date/fold, native predictions, paired ablations, A12, calibration, uncertainty,
complexity/latency/VRAM/cost, negative results, review/Hidden sensitivity,
limitations và command tái dựng.

Headline claim chỉ hợp lệ khi:

- label authority hiện hành phù hợp paper-grade use;
- A12 source-balanced date-safe PASS;
- native-unit OOF và calibration PASS;
- forbidden X/future/split/mass đều PASS;
- practical effect và date stability PASS;
- post-review/reproduction status được ghi;
- không dựa vào một rare-class fold hoặc một seed.

Single-camera/single-pen là external-validity limitation, không trình bày như
cross-camera generalization.

## 7. Final machine checklist

Trước khi nói “được phép thuê GPU”, tất cả phải PASS:

~~~
CODE_AUTHORITY_ESTABLISHED
DATA_AUTHORITY_ESTABLISHED
MOTION_DIMENSION_CONTRACT_PASS
SPATIAL_SCHEMA_CONTRACT_PASS
PAIR_VALIDITY_CONTRACT_PASS
LOADER_TENSOR_CONTRACT_PASS
FORBIDDEN_FEATURES_PRESENT_IN_X=0
FUTURE_FRAME_DEPENDENCE=0
CROSS_LABEL_WINDOWS=0
SPLIT_PURITY_PASS
TRAINING_MASS_AUDIT_PASS
LEARNED_A12_PASS
NATIVE_UNIT_OOF_AND_CALIBRATION_PASS
REAL_RESUME_PASS
COMPUTE_PROFILE_PASS
AUTORESEARCH_REVIEWED_CAMPAIGN_PASS
SCIENTIFIC_RESULT_PACKAGE_PASS
PAID_GPU_LAUNCH_PERMIT=PASS
~~~

Giá trị hiện tại:

~~~
READY_FOR_LOCAL_SMOKE=PASS
READY_FOR_LOCAL_MODEL_RUNNING=NOT_YET_MEASURED
READY_FOR_PILOT_TRAINING=BLOCKED
READY_FOR_SCREENING_TRAINING=BLOCKED
READY_FOR_CLAIM_GRADE_TRAINING=BLOCKED
READY_FOR_PAID_GPU=BLOCKED
~~~

## 8. Immediate next action

Chưa launch training từ tài liệu này. Bước kế tiếp được phép là thiết kế và
implement bounded learned-A12 control runner trong isolated worktree, thêm
independent checker, rồi chạy lại P0 contract checks. Sau khi A12 có report
PASS/FAIL/INCONCLUSIVE mới chạy one-fold local pilot, real resume và profile.
Chỉ readiness package mới có thể đổi trạng thái paid-GPU.
