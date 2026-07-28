• Kế Hoạch Tương Lai

## Confirmatory unseen method set frozen - 2026-07-28

- `realtime_fast` is the frozen primary same-barn unseen candidate.
- `bytetrack_raw` is the frozen confirmatory technical baseline.
- `hybrid_bytetrack` and `rf_hybrid_offline` are development-only ablations;
  neither is authorized for unseen execution.
- The future R0-minus-B0 contrast is whole-pipeline and includes detector
  cadence. It is not a pure association-core comparison.
- Unseen work must preserve the frozen HOTA/IDF1 co-primary hierarchy and the
  predeclared identity-severity diagnostics, using video as the statistical
  unit and session/day as a secondary cluster.
- Next gate: freeze the unseen-data authority in a separate task. Unseen file
  access, execution, tuning, method reselection, evaluation, and promotion
  remain unauthorized until their later explicit gates.

## Development tracking 2x2 Standard-V2 complete - 2026-07-28

- Frozen B0/B1/R0 metric authorities were reused and frozen R1 was evaluated
  twice under the same Standard-V2 include-Hidden contract.
- R0 remains strongest overall. R1 lowers IDSW to `18`, but loses HOTA and
  IDF1 and increases wrong-ID exposure and terminal episodes relative to R0.
- ByteTrack repair is `MIXED_TRADEOFF`; RF repair is `BROADLY_HARMFUL`.
- The interaction is metric-level and includes profile-specific detector
  cadence. It is not a pure association-core interaction.
- B1 event attribution is unavailable from its frozen authority. R1 event
  attribution remains diagnostic only; cross-core event-count comparison is
  forbidden.
- Next gate: a separate unseen-method freeze decision. Unseen execution,
  promotion, runtime benchmarking, and method changes remain unauthorized.

## R1 frozen prediction authority — 2026-07-28

- R1 `rf_hybrid_offline` đã hoàn tất độc lập trên đúng 13 video development,
  dùng frozen R0 even-frame cache và không gọi live detector.
- Vì R0 public export thiếu lifecycle/provenance nội bộ bắt buộc, topology là
  `EXACT_R1_PROFILE_EXECUTION`. Raw RF core đã PASS parity với frozen R0 trước
  adapter và offline repair cho cả 13 video.
- Đã đóng băng 13 XML, 187.200 prediction objects, raw-core snapshots và
  deterministic repair ledgers tại
  `frozen_predictions_standard_v2_20260728_retry1/R1_rf_hybrid_offline`.
- R1 artifact authority:
  `40052f992871d50984fc4c0c839c4933b772bca2bfcaaaacafcde40d0e8a1800`.
- Không chạy detector inference, evaluator, quality comparison, unseen data
  hoặc MP4; frozen R0 không bị sửa.
- Task tiếp theo có thể evaluate development 2x2 B0/B1/R0/R1 bằng
  Standard-V2 trong worktree riêng. Unseen evaluation, runtime và promotion
  vẫn chưa được phép.

## B0/B1/R0 Standard-V2 authority - 2026-07-28

- Frozen B0, B1, and R0 predictions passed two complete
  `TRACKING_EVALUATOR_STANDARD_V2` runs with `include_hidden=true`.
- Corrected HOTA: B0 `0.849511403`, B1 `0.849873389`,
  R0 `0.888187232`.
- Corrected IDF1: B0 `0.920646368`, B1 `0.914081197`,
  R0 `0.971892400`.
- Corrected `IDSW_STANDARD`: B0 `84`, B1 `64`, R0 `29`.
- The old B1 > R0 > B0 headline ranking is retired. Standard-V2 values now
  govern development baseline reporting.
- B1 minus B0 is still the matched-cadence offline-repair contrast.
  R0 contrasts include detector cadence and cannot support a pure
  association-core claim.
- R1 prediction generation is the next authorized tracking task under its
  frozen profile and even-frame detector contract.
- Complete development 2x2 evaluation, unseen evaluation, runtime testing,
  and promotion remain unauthorized.

## B0/B1 frozen prediction authority — 2026-07-28

- B0 `bytetrack_raw` và B1 `hybrid_bytetrack` đã chạy độc lập trên đúng 13
  video development, dùng full-frame cache và không gọi detector inference.
- Mỗi arm đã đóng băng 13 XML, 23.400 processed frames và 187.200 prediction
  objects trong `frozen_predictions_standard_v2_20260728_retry1`.
- B0/B1/R0 đã PASS fairness ở mức authority. Cross-core comparison chỉ được
  diễn giải là whole-pipeline effect gồm detector cadence; không được claim
  pure association-core effect.
- Bước tiếp theo là task riêng để re-evaluate frozen B0/B1/R0 bằng
  `TRACKING_EVALUATOR_STANDARD_V2` và `IDENTITY_ERROR_EPISODES_V2`.
- Chưa được chạy development 2x2, unseen evaluation, runtime funnel hoặc
  promotion.

## Full-frame detector cache gate — 2026-07-28

- Detector cache full-frame cho đúng 13 video development đã khóa đã PASS:
  23.400 frame records, gồm 11.700 R0 even records giữ nguyên và 11.700 odd
  records mới.
- `EVEN_SUBSET_PARITY=PASS`; không inference lại frame chẵn, không chạy tracker,
  evaluator, unseen data hoặc sinh MP4.
- Có 12.100 physical odd-frame calls vì 400 retry cùng frame/cùng detector
  authority sau lỗi khóa heartbeat tạm thời trên Windows; số odd records duy
  nhất vẫn là 11.700.
- Bước tiếp theo được phép lập task riêng để regenerate và freeze prediction
  B0/B1 từ full-frame cache. R0 tiếp tục dùng frozen even subset và không rerun.
- So sánh cross-core chỉ được diễn giải là whole-pipeline effect gồm detector
  cadence; không được claim pure association-core effect.
- Chưa được chạy Standard-V2 re-evaluation, development 2×2, unseen evaluation,
  runtime funnel hoặc promotion.

## B0/B1 frozen prediction gate — 2026-07-28

- Không chạy B0/B1 với cache R0 hiện tại: cache chỉ có frame chẵn
  `0,2,...,1798`, còn hai profile yêu cầu detector evidence ở mọi frame.
- Quyết định hiện tại:
  `FAIL_COMMON_DETECTOR_REPLAY_CONTRACT`.
- Bước tiếp theo phải là một task riêng cho phép tạo detector evidence
  full-frame trên đúng 13 video development đã khóa.
- Chưa được chạy Standard-V2 re-evaluation, development 2×2, unseen
  evaluation, runtime funnel, hoặc promotion.

## Override registry tracking ngay 2026-07-28

- Registry active chi con `bytetrack_raw`, `realtime_fast`, va
  `hybrid_bytetrack`.
- `realtime` da retire va khong redirect ngam; lenh hien tai phai dung ro
  `realtime_fast`.
- `realtime_balanced`, `realtime_quality_delayed`, va
  `realtime_fast_h1_r2` chi con la ten lich su, khong duoc active execution.
- Cac bang chung, manifest, va decision H1/H2 cu duoc giu nguyen. Cache/replay,
  telemetry chung, evaluator, va repair chung van duoc bao toan.
- Override nay thay the cac chi dan cu ve viec giu nhieu realtime profile trong
  registry active. No khong cho phep implement `rf_hybrid_offline`.

## Goal mới: realtime online challenger — 2026-07-20

- Mục tiêu vận hành realtime chuyển sang một challenger online-only; chỉ dùng
  frame hiện tại và state/lịch sử quá khứ hữu hạn, không dùng future frame.
- Giữ `realtime_fast` và `realtime_balanced` làm control bất biến. Fast là
  identity reference; Balanced là quality/coverage reference. Không chạy lại
  raw quality authority.
- Không coi `realtime_quality_delayed` hiện tại là ứng viên realtime: delay
  `-1`/global graph không đúng semantic online. Giữ nó làm upper-bound paper và
  bằng chứng âm; challenger mới thay vai trò realtime quality nếu thắng gate.
- Có thể học các cơ chế Hybrid đã chứng minh nhưng chỉ port phần causal:
  hidden-owner/re-entry guard, gap-over-bad-match và motion prediction. Không
  port suffix repair, full-video smoothing, future-anchor hay global graph.
- Challenger phải giải quyết cả identity và vận hành: `include_hidden=true`,
  IDSW/HOTA/IDF1/FP/FN/fragments, causal delay `0`, FPS, p50/p95, backlog,
  output age, memory, repeatability và lineage. Native 30 FPS hoặc drop policy
  được khai báo là gate bắt buộc; không gọi profile dưới gate là realtime winner.
- Funnel cố định: hard event windows -> một video khó -> hard set ít nhất ba
  video -> full-13 chỉ sau khi hard-set PASS. Không sinh MP4, preview, overlay
  hay event clip ở bất kỳ bước nào.
- Mỗi candidate chỉ đổi một family, có parent/candidate/repeat manifest riêng.
  Base cũ là control để so sánh, không phải nền bắt buộc; nếu challenger thắng
  đa mục tiêu và không vi phạm guardrail, được phép thay thế profile cũ bằng
  commit thuật toán và commit profile promotion tách biệt.
- Classification/model nằm ngoài phạm vi goal này.

## Override chọn realtime winner ngày 2026-07-19

### Tiêu chí chọn realtime: Pareto đa mục tiêu

- Không chọn realtime chỉ theo HOTA, IDF1 hoặc IDSW. Candidate phải qua các
  hard gate về causal/fixed-delay, prefix invariance, repeatability, lineage,
  memory và zero-MP4.
- Sau khi đủ điều kiện, so đồng thời identity, HOTA/IDF1, FP/FN, fragments,
  effective FPS, loop-FPS, p50/p95, stage timing, delay và chi phí triển khai.
- Một cấu hình chỉ được gọi là winner khi không bị candidate khác trội trên
  toàn bộ các chiều và trade-off phù hợp use case realtime đã công bố.
  Nếu không có cấu hình trội tuyệt đối, phải báo rõ mặt đánh đổi còn lại.
- Đo tốc độ là tiêu chí bắt buộc, nhưng không được tuyên bố nhanh hơn nếu
  primary/repeat chưa dùng cùng harness và chưa qua runtime gate.
- Raw authority cùng contract đã PASS: IDSW `145`, HOTA `88.91%`, IDF1
  `88.47%`, loop-FPS `22.65/27.03`, repeatability PASS và `mp4_count=0`.
- Fast hiện là causal reference (IDSW `69`) chứ chưa phải winner cuối cùng:
  `000302` đang có IDSW `6` so với ceiling `2`, và speed claim chờ
  common-harness runtime audit.

### Quyền chọn của realtime Quality (bắt buộc)

- `realtime_quality` (profile hiện có: `realtime_quality_delayed`) là ứng viên
  realtime chính thức, có cùng quyền thắng Pareto như Fast và Balanced.
- Không được loại Quality chỉ vì bảng paper cuối cùng có thể chỉ giữ một
  profile realtime. Việc bỏ bớt profile là quyết định trình bày sau khi đã
  sàng lọc đầy đủ, không phải lý do để bỏ qua thí nghiệm Quality.
- Nếu một bản Quality causal hoặc fixed-delay vượt qua prefix invariance,
  identity, HOTA/IDF1, runtime, repeatability, lineage và zero-MP4, rồi thắng
  Pareto, nó thay thế Fast/Balanced trong chuỗi paper:
  `bytetrack_raw -> realtime_quality -> hybrid_bytetrack_best`.
- Quality delay `-1` hiện tại chỉ là upper bound post-video; không được gọi là
  realtime winner, nhưng phải giữ làm bằng chứng chất lượng và mốc so sánh.
- Vì các candidate finite-delay hiện tại chưa qua runtime gate, bước raw
  authority kế tiếp không xóa Quality khỏi kế hoạch; chỉ mở lại một family
  Quality mới khi có giả thuyết và funnel riêng được khai báo trước.

- `hybrid_bytetrack_best` đã hoàn tất trước với full-13 IDSW `0`, HOTA
  `98.3506%` và IDF1 `99.1490%`.
- Sau Hybrid, Fast, Balanced và Quality là ba ứng viên trong cùng một phép
  chọn Pareto; thứ tự tên không phải thứ hạng kết quả. Fast là mốc causal vận
  hành, còn Quality là challenger bắt buộc và được chọn nếu thắng hợp lệ.
- Quality chỉ đủ tư cách thắng khi causal hoặc finite-delay, qua prefix
  invariance, chất lượng, FPS, p95, memory, repeatability và zero-MP4.
- RQ1-RQ4 đã sàng lọc finite-delay Quality công bằng nhưng chưa có candidate
  hợp lệ. RQ4 giữ cải tiến clone output-equivalent, song fail runtime nên
  không mở later window, full video, hard set hoặc full-13.
- Quality delay `-1` dùng toàn video chỉ là post-video upper bound, không được
  xếp làm realtime winner.
- Bảng Pareto Fast/Balanced/Quality đã được khóa: Fast được chọn tạm thời vì
  IDSW thấp nhất trong các authority causal hợp lệ; Balanced vẫn là quality
  reference, còn Quality chưa qua finite-delay runtime gate.
- Bước kế tiếp là tạo raw authority cùng contract, rồi mới so: raw -> Fast ->
  Hybrid. Nếu Quality có candidate causal/fixed-delay mới thắng Pareto, được
  phép thay Fast theo đúng gate.
- Mọi thử nghiệm tiếp tục dùng `include_hidden=true`, rule
  `iou0_area0_condarea0_merge0` và tuyệt đối không sinh MP4.

## Cập nhật H4 sang H5 ngày 2026-07-18

- H4 đã sửa đúng family bbox Hidden vùng xa của `000328`: full-video IDSW
  `4 -> 0`, đồng thời HOTA, IDF1, matches, FP và FN đều tốt hơn.
- Hard set bốn video không có regression và aggregate IDSW `8 -> 4`, nhưng
  chỉ một video khó cải thiện. Vì gate đã khóa cần ít nhất hai video, không
  chạy full-13 và không promote profile chỉ với H4.
- Giữ H4 làm component đã có bằng chứng và chuyển sang H5 cho conflict
  identity payload của `000233`.
- Sau khi H5 qua window và full target, đánh giá H4+H5 trên cùng hard set.
  Chỉ khi ít nhất hai video khó tốt hơn mới được mở full-13.
- Các lane realtime vẫn đóng; sau khi hybrid hoàn tất mới lấy fast làm mốc
  vận hành và balanced làm target thực dụng.

## Bổ sung chủ động 2026-07-16: tối ưu tracking không hồi quy

Phần này là kế hoạch đang có hiệu lực. Nội dung cũ bên dưới được giữ làm lịch
sử, nhưng không được dùng để bỏ qua các gate mới.

### -2. Hiệu chỉnh hybrid-residual-first ngày 2026-07-18

- Một authority promote candidate chỉ đóng experiment đó, không có nghĩa lane
  `hybrid_bytetrack` đã hoàn tất.
- Authority hiện tại còn tám IDSW: `000233` có bốn tại frames `1111-1114`,
  `000328` có bốn tại frames `1347-1355`.
- Xử lý hai cụm này như hai family riêng. `000328` là lỗi geometry của bbox
  Hidden vùng xa bên phải; `000233` là lỗi identity payload.
- Tiếp tục funnel hard-window, full target video, hard set ít nhất ba video,
  rồi mới full-13 và repeat. Không mở lane realtime chỉ vì near-wall candidate
  đã được promote.
- Chỉ chuyển công nghệ sang realtime sau một decision riêng xác nhận hybrid
  đã qua residual audit và stop gate. Khi đó fast là mốc vận hành và balanced
  phải chứng minh đủ ổn định để dùng thực tế so với fast.

### -1. Thứ tự critical path khóa ngày 2026-07-18

- Hoàn thiện và khóa authority cho `hybrid_bytetrack` trước khi chuyển bất kỳ
  cơ chế nào sang realtime. Không tối ưu hai lane trong cùng một experiment.
- Near-wall Hidden bbox geometry đã được promote vào `hybrid_bytetrack_best`:
  IDSW giữ `8`, FP/FN `1630 -> 1622`, HOTA `98.31% -> 98.32%`, raw IDF1 tăng,
  primary/repeat khớp và `mp4_count=0`.
- Lane realtime tiếp theo dùng `realtime_fast` làm mốc vận hành. Balanced chỉ
  được công nhận khi qua gate identity-stability và latency khai báo trước,
  đồng thời tạo giá trị thực so với fast; tốt hơn balanced cũ là chưa đủ.
- Authority và bằng chứng âm được khóa tại
  `docs/TRACKING_PROMOTION_DECISION_20260718_HYBRID_NEAR_WALL_GEOMETRY.json`.

### 0. Mốc phải khóa trước khi tối ưu

- Hybrid chuẩn là `hybrid_bytetrack_best` với rule
  `iou0_area0_condarea0_merge0`.
- Run `20260707_230230` là mốc tương thích exclude-Hidden: 13 video đều có
  `remapped_idsw=0`, HOTA `97.26%`, IDF1 `98.58%`, FP `2911`, FN `2346`,
  và gap-tolerant fragments `55`. Không dùng nó làm authority cho promotion.
- Baseline chính theo GT đúng là prediction
  `20260717_ec67d27_full13_primary_hybrid_best`, replay tại
  `20260717_e55973f_includehidden_full13_baseline_ec67d27_v1` với
  `include_hidden=true`: IDSW `10`, HOTA `98.31%`, IDF1 `99.13%`, FP/FN
  `1628/1628`, và gap-tolerant fragments `5`.
- GT được tracker cũ sinh trước rồi con người sửa bbox/ID. Khoảng 1.930 giá trị
  `Hidden` có thể là lỗi tracker cũ, không phải visibility đã được xác nhận.
  Vì vậy bbox/ID là authority; `Hidden` không phải target tối ưu.
- Không dùng hàng `ALL` của `codex_visible_suffix_gate_full`: artifact đó đang
  đọc sai/stale GT `000216` và báo `remapped_idsw=33`.
- Run `outputs/eval/mode_compare/20260709_040751` đã gồm raw, hybrid và đủ ba
  profile realtime, nhưng dùng `include_hidden=false`. Giữ nó để tương thích;
  phải replay hoặc tái tạo hash-bound với `include_hidden=true` trước promotion.
- `realtime_fast`: IDSW `56`, HOTA `93.10%`, IDF1 `92.91%`, FP/FN
  `2367/1019`.
- `realtime_balanced`: IDSW `75`, HOTA `92.77%`, IDF1 `93.12%`, FP/FN
  `2320/1055`.
- `realtime_quality_delayed`: IDSW `21`, HOTA `96.60%`, IDF1 `97.02%`,
  FP/FN `2320/1055`.
- Các video quality-delayed còn yếu là `000114=2`, `000231=6`, `000233=9`,
  `000263=2`, `000327=2`.

Các số realtime ở trên đều thuộc hợp đồng exclude-Hidden lịch sử. Không dùng
chúng để chọn hoặc promote candidate cho đến khi có baseline include-Hidden.

Replay hash-bound theo hợp đồng chính `include_hidden=true` tại commit
`9a2979d` đã khóa lại ba baseline:

- `realtime_fast`: IDSW `87`, HOTA `93.8897%`, IDF1 `93.2109%`, FP/FN
  `564/688`.
- `realtime_balanced`: IDSW `133`, HOTA `93.9254%`, IDF1 `93.7140%`, FP/FN
  `449/587`.
- Parent `realtime_quality_delayed` với simple gain `0.005`: IDSW `168`,
  HOTA `97.6610%`, IDF1 `97.5782%`, FP/FN `449/587`.

Candidate quality-delayed đã promote trước đó với simple gain `0.003` được giữ
lại sau khi sửa contract. Primary và repeat đều cho IDSW `166`; chỉ `000263`
cải thiện `44 -> 42`, không video nào tăng IDSW, FP/FN và fragmentation không
đổi, HOTA/IDF1 không giảm. Repeatability lock kiểm tra 26 prediction XML, 72
artifact và `mp4_count=0`. Profile này vẫn là `post_video_global_graph` với
delay `-1`, không phải causal hoặc fixed-delay. Không dùng biến động FPS tuyệt
đối giữa các run để tuyên bố tăng tốc; chỉ candidate repeatability và memory
gate đã PASS.

Mỗi baseline lock phải ghi SHA commit, hash detector weight, danh sách video,
ánh xạ video-GT, hash GT/XML, semantic config và output root mới. Sai stem,
trùng video, thiếu file hoặc output root đã tồn tại phải fail closed.

### 1. Tách hai lane độc lập

- Ba profile realtime đã được triển khai trong code và đã benchmark tại
  `20260709_040751`. Giữ nguyên tên/profile; không tạo bản trùng lặp.
- `hybrid_bytetrack`: offline/high-quality; không video nào được tăng IDSW so
  với baseline include-Hidden cùng contract. Chỉ yêu cầu đúng `0` khi baseline
  cùng contract của video đó bằng `0`.
- `realtime_fast` và `realtime_balanced`: causal, không sửa output đã phát.
- `realtime_quality_delayed`: chỉ được gọi fixed-delay sau khi chứng minh mọi
  quyết định dùng buffer hữu hạn và frame tương lai ngoài buffer không thể đổi
  output đã commit.
- Không chuyển suffix repair, hidden-suffix repair hoặc full-video graph từ
  hybrid sang realtime.

### 1.1. Output contract: cấm sinh MP4 khi thử nghiệm

- Video `.mp4` trong `data/videos` chỉ là input read-only.
- Mọi benchmark, probe, ablation, optimizer và hard-scene analysis không được
  tạo `.mp4` mới dưới `outputs/` hoặc thư mục tạm.
- Chỉ giữ artifact cần cho phân tích: prediction XML, CSV, JSON, Markdown và
  log nhỏ. Không render tracked preview, overlay video hoặc event clip.
- Run `20260709_040751` chỉ còn CSV/JSON/Markdown trong eval root; đây là dạng
  artifact cần giữ cho các run sau.

Code hiện tại chưa bảo đảm contract này: `run_tracker_for_pair` gọi
`run_tracking`, còn `run_tracking` luôn gọi `render_annotation_video` và mặc
định tạo `tracked_pigs_with_ids.mp4`. Vì vậy trước benchmark tiếp theo phải:

1. Thêm cờ fail-closed kiểu `write_output_video=false` cho evaluation.
2. Evaluation/optimizer/hard-scene mặc định tắt preview, overlay và clip.
3. Chỉ flow tracking phục vụ người dùng mới được render khi bật rõ ràng.
4. Cho phép `TrackingSummary.output_video` không tồn tại khi video bị tắt.
5. Sau mỗi run, scan đệ quy output root; thấy bất kỳ `.mp4` mới thì run FAIL.

Test bắt buộc phải chứng minh khi video output tắt:

- `render_annotation_video` không được gọi;
- prediction XML và metric vẫn được sinh đầy đủ;
- config/CLI truyền đúng cờ qua mọi mode;
- không có file `.mp4` trong output root.

### 2. Metric contract và điều kiện promote

Hybrid chỉ được promote khi toàn bộ điều kiện sau cùng đúng:

- Baseline, candidate và repeat đều dùng `include_hidden=true`; báo cáo loại
  Hidden chỉ dùng để đối chiếu tương thích.
- Không video nào tăng remapped IDSW so với baseline cùng contract; video có
  baseline IDSW bằng `0` phải giữ bằng `0`.
- Aggregate HOTA và IDF1 không giảm; không video nào giảm HOTA quá `0.10`
  điểm phần trăm hoặc IDF1 quá `0.05` điểm phần trăm.
- FP, FN và gap-tolerant fragments không tăng ngoài sai số đã khai báo.
- Có cải thiện thật: aggregate HOTA tăng ít nhất `0.10` điểm phần trăm, hoặc
  tổng FP+FN giảm ít nhất `1%` mà fragmentation không tăng.
- Repeat độc lập cho cùng config phải cho metric và semantic hash giống nhau.

Realtime chỉ được promote khi:

- Không video nào tăng remapped IDSW, kể cả `000302`, so với baseline
  include-Hidden của chính profile đó.
- Tổng IDSW giảm ít nhất `2`, hoặc HOTA tăng ít nhất `0.20` điểm phần trăm
  trong khi IDSW và IDF1 không xấu đi.
- FP và FN mỗi loại không tăng quá `0.5%`.
- Báo cáo đủ FPS, frame-time p50/p95, detector/association/postprocess time,
  buffer-delay frames/ms và peak memory.
- `realtime_fast` không chậm hơn baseline; `realtime_balanced` có p95 không
  quá `110%` baseline. Quality-delayed phải khai báo delay cứng.

Hybrid weak set hiện tại được suy ra từ baseline, không gắn cứng vào optimizer:

- `000225`: HOTA `93.94%`, FP `652`, gap-tolerant fragments `12`.
- `000233`: HOTA `94.38%`, FN `644`, gap-tolerant fragments `7`.
- `000231`: gap-tolerant fragments `14`.
- `000216`: HOTA `96.75%`, FP/FN `255/221`, nhưng phải khóa đúng GT trước.

Sau mỗi baseline mới, weak set phải được tính lại bằng metric, không chọn video
theo kết quả thuận lợi của candidate.

### 3. Trình tự thực hiện bắt buộc

P0 - Khóa lineage, chưa đổi thuật toán:

- Tạo manifest 13 video với đường dẫn và SHA256 của video, GT, weight.
- Fail closed nếu `video_stem`, GT stem và row trong report không khớp.
- Ghi commit SHA, profile semantic diff, seed, CLI và hash output.
- Output mỗi run là thư mục versioned mới; cấm ghi đè baseline.
- Khóa `20260707_230230` cho hybrid và `20260709_040751` cho mode comparison.
- Hoàn thành no-MP4 gate trước bất kỳ run tracker/evaluation mới nào.

P1 - Thêm test và telemetry, prediction phải byte-identical:

- Test profile tách biệt raw/realtime/hybrid và default flag không đổi.
- Test evaluator không nhận GT sai stem, video trùng hoặc universe thiếu.
- Test evaluation tạo XML/metric nhưng không gọi renderer và không sinh MP4.
- Ghi timing từng stage, FPS, delay và peak memory.
- Ghi counter theo phase association, lý do reject/hold và repair trigger.
- Nếu prediction hoặc metric đổi ở P1 thì dừng, không gọi là instrumentation.

P2 - Gate nhỏ cho từng candidate:

- Static/config semantic diff, compile, import và focused unit tests.
- Synthetic crossing, occlusion, low-confidence, long-gap và dense-crowd.
- Một video target, sau đó target cộng guardrail `000302`.
- Hard set realtime:
  `000114/000231/000233/000263/000327/000302`.
- Hard set hybrid được tính từ bottom metric của baseline.

P3 - Chỉ sau khi P2 pass:

- Chạy full 13-video cùng manifest và weight.
- Chạy repeat độc lập vào output root khác.
- So paired per-video, không chỉ hàng `ALL`.
- Chỉ promote ở commit riêng sau khi promotion report PASS.
- Kế hoạch này không tự cấp quyền chạy benchmark dài; chỉ chạy khi người dùng
  yêu cầu giai đoạn thực thi.

### 4. Các họ thí nghiệm, mỗi lần chỉ đổi một họ

Hybrid H1 - Matching theo confidence:

- Giả thuyết: `hybrid_bytetrack` ghép `all_detection_indices` quá sớm làm tăng
  FP/FN hoặc gap dù ID sau repair đã đúng.
- Thử high-confidence visible/reid trước, low-confidence recovery sau bằng một
  flag opt-in. Không đổi owner guard, post-processing hoặc detector.

Hybrid H2 - Geometry-only refinement:

- Khóa toàn bộ ID payload và repair quyết định identity.
- Chỉ ablate smoothing/refinement bbox để tăng DetA/MOTP/HOTA.
- Trước full/repeat, so toàn XML và reject nếu track ID, shape key, Behavior,
  `Hidden`, `occluded` hoặc non-geometry payload đổi. Bbox thay đổi không được
  phép kích hoạt một family hậu xử lý visibility khác.

Hybrid H3 - Visibility và short-gap recovery:

- Chẩn đoán gap theo ID ở `000225/000233/000231/000216`.
- Chỉ fill gap ngắn có motion/visibility support; không relabel ID.
- Không bật `condarea` mặc định và không chạy detector-only sweep khi chưa có
  bằng chứng detection threshold là nguyên nhân.

Realtime R0 - Sửa đúng semantic trước tối ưu:

- Hiện `stabilize_realtime_motion_pairs` chạy sau vòng xử lý video và dùng
  planned-edge graph của toàn bộ chuỗi; chưa chứng minh fixed-delay causal.
- `local_pair_swap_repair=true` cũng chưa chạy trong profile realtime vì nằm
  sau gate `enable_offline_smoothing=false`.
- Trước tiên phải thêm causality test và reclassify profile hoặc viết rolling
  fixed-lag implementation; chưa được quảng bá nó là realtime thuần.

Realtime R1 - Association causal:

- Dùng state quá khứ hữu hạn cho competitor/reentry conflict còn lại.
- Không port suffix repair hay hybrid post-processing; mỗi guard là một ablation.

Realtime R2 - Rolling short buffer:

- Thử một delay cố định trong `12/15/30` frame cho local swap, hidden island và
  short gap; frame đã flush là immutable.
- Test bắt buộc: thêm frame tương lai ngoài buffer không đổi output đã flush.

Realtime R3 - Speed:

- Chỉ sau R1/R2, ablate `detect_every_n_frames` và LK/motion prediction như một
  họ riêng; không đổi cadence cùng association.

### 5. Chuỗi commit an toàn và có thể rollback

Mỗi commit chỉ có một vai trò:

1. `docs(tracking): lock benchmark and promotion contract`.
2. `test(tracking): add lineage, causality and regression gates`.
3. `feat(tracking): disable video rendering for experiment pipelines`.
4. `chore(tracking): add prediction-invariant telemetry`.
5. `feat(tracking): add one opt-in candidate family`.
6. `docs(tracking): record paired promotion or rejection evidence`.
7. `feat(tracking): promote validated profile` chỉ sau full-13 và repeat PASS.

Không bật candidate mặc định trong commit triển khai. Commit promote phải tách
riêng để rollback bằng một `git revert` mà không xóa implementation/test.

Trước mỗi commit:

- Ghi starting SHA và `git status`; không stage thay đổi ngoài tracking.
- Đặc biệt không stage các file classification và diagnostic đang dirty.
- Xem `git diff -- <file>` cho từng file.
- Chạy `git diff --check` và scan `^.{101,}$` trên mọi file thay đổi.
- Chạy compile/import, Ruff và focused tracking tests.
- Tối thiểu gồm `test_tracking_pipeline.py`,
  `test_tracking_improvements.py`, `test_tracking_profiles.py` và
  `test_tracking_eval_config_overrides.py`.
- So semantic config diff; reject nếu đổi nhiều họ ngoài khai báo.
- Ghi exact command, exit code, test counts và artifact hashes.

Không dùng commit metric nếu output bị ghi đè, manifest khác baseline, thiếu
per-video row, repeat không deterministic hoặc có regression chưa giải thích.

### 6. Sổ thí nghiệm và stop rule

Mỗi run phải cập nhật các artifact versioned:

- `tracking_experiment_matrix.csv`;
- `tracking_ablation_registry.csv`;
- `tracking_promotion_decisions.json`;
- `tracking_rejected_experiments.json`;
- `tracking_finalist_lock.json`.

Ma trận khởi đầu:

- `MC0`: mode comparison `20260709_040751`, đủ năm presentation mode.
- `HB0`: hybrid `20260707_230230`, baseline khóa.
- `H1`: staged high/low-confidence matching, parent `HB0`.
- `H2`: geometry-only refinement, parent `HB0`.
- `H3`: visibility/short-gap recovery, parent `HB0`.
- `RF0`: `realtime_fast` từ `MC0`.
- `RB0`: `realtime_balanced` từ `MC0`, causal baseline.
- `RQ0`: quality-delayed từ `MC0`, ghi rõ post-video/global-graph.
- `R1`: causal association memory, parent `RB0`.
- `R2-D12`, `R2-D15`, `R2-D30`: rolling fixed-delay, parent `RB0`.
- `R3`: cadence/LK speed ablation, parent của finalist R1 hoặc R2.

Stop ngay khi xảy ra một trong các điều kiện:

- Baseline, GT mapping, weight hash hoặc semantic config chưa khóa.
- Một diff đổi hơn một họ thuật toán.
- Infrastructure-only commit làm prediction thay đổi.
- Bất kỳ bước phân tích nào sinh `.mp4` mới trong output root.
- Candidate hybrid làm tăng per-video IDSW so với baseline include-Hidden cùng
  contract, hoặc làm video baseline-zero xuất hiện IDSW.
- Realtime dùng future ngoài delay đã khai báo hoặc sửa frame đã flush.
- Candidate chỉ thắng hàng `ALL` nhưng thua guardrail/per-video.
- Focused test, full-13, repeat hoặc latency gate không PASS.

Bước thực thi kế tiếp hợp lệ là P0 lineage lock cùng no-MP4 test/implementation;
chưa sửa association/refinement và chưa chạy benchmark dài trong lần lập kế
hoạch này. Các mục lịch sử bên dưới về việc tạo ba profile realtime đã hoàn
thành và không được thực hiện lại.

  Kết quả mới sau khi sửa GT 216 cho thấy cấu hình hiện tại đã rất mạnh về
  identity: 13 video đều remapped_idsw=0. Vì vậy hướng tiếp theo không nên tiếp
  tục “vá IDSW” bừa bãi nữa, mà nên tách dự án thành 3 chế độ rõ ràng theo mục
  tiêu sử dụng.

  1. Phân Rõ 3 Mode

  bytetrack_raw

  Đây nên là baseline kỹ thuật: dùng ByteTrack/Ultralytics gần như nguyên bản,
  không dùng các guard/repair/smoothing nâng cao của project. Mục
  đích là làm mốc so sánh khoa học: “nếu không có logic cải tiến thì kết quả thế nào”.

  Trong code hiện tại, mode đúng nên gọi là bytetrack_raw. Lưu ý bytetrack đang
  là legacy alias và bị map về hybrid_bytetrack, nên trong báo
  cáo/README nên tránh gọi bytetrack nếu muốn nói baseline thô.

  hybrid_bytetrack

  Đây là mode chất lượng cao nhất. Nó nên là mode chính cho xuất CVAT, đánh giá
  khoa học, tạo annotation, và kết quả tracking cuối cùng.

  Nó có thể dùng toàn bộ logic hiện tại:

  - association guards
  - hidden owner hold
  - reentry/occlusion guards
  - smoothing/refinement
  - overlap suppression
  - suffix repairs
  - hidden suffix repair

  Mode này không cần đúng nghĩa realtime. Nó ưu tiên metric và tính ổn định ID.

  realtime

  Đây nên là mode streaming/low-latency đúng nghĩa. Nó không nên bê nguyên toàn
  bộ hybrid_bytetrack sang, vì nhiều phần của hybrid cần nhìn
  toàn video hoặc nhìn suffix dài phía sau.

  Realtime nên dùng thiết kế “causal only”: chỉ dùng frame hiện tại và quá khứ,
  hoặc một buffer rất ngắn nếu chấp nhận delay.

  2. Có Lấy Gì Từ Hybrid Sang Realtime Được Không?

  Có, nhưng phải chia làm 3 nhóm.

  Có thể chuyển gần như trực tiếp:

  - hidden_owner_guard
  - hidden_owner_guard_hold_assignment
  - occlusion_reid_prefer_gap_over_bad_match
  - raw-owner / unowned-raw mismatch guards
  - motion prediction / LK prediction khi skip detection
  - telemetry/debug assignment events
  - adaptive confidence filter nếu giới hạn số candidate rõ ràng

  Những logic này chỉ cần trạng thái hiện tại/quá khứ, nên phù hợp realtime.

  Có thể chuyển nhưng cần viết lại thành bản online/buffer ngắn:

  - temporal smoothing
  - hidden overlap island stabilization
  - overlap small-box suppression
  - local pair repair

  Các phần này có thể chạy với buffer 5-15 frame, chấp nhận delay nhỏ. Ví dụ
  camera realtime vẫn hiển thị trễ 0.2-0.5 giây để đổi lấy ID ổn
  định hơn.

  Không nên chuyển trực tiếp:

  - suffix_pair_swap_repair
  - hidden_suffix_id_swap_repair
  - long suffix repair
  - các repair cần nhìn hàng trăm frame tương lai

  Đây là logic hậu xử lý offline. Nếu đưa thẳng vào realtime sẽ sai bản chất
  “realtime”. Nếu muốn dùng, phải đổi thành delayed_realtime với
  buffer dài, không gọi là realtime thuần.

  3. Thiết Kế Realtime Đề Xuất

  Tách realtime thành 3 profile:

  realtime_fast

  Mục tiêu: tốc độ cao nhất.

  - detect mỗi N frame, ví dụ detect_every_n_frames=3 hoặc 5
  - giữa các frame dùng LK/motion prediction
  - không offline repair
  - không suffix repair
  - chỉ dùng guard nhẹ trong association
  - metric có thể thấp hơn hybrid nhưng latency thấp

  realtime_balanced

  Mục tiêu: dùng được thực tế.

  - detect mỗi 2-3 frame
  - dùng causal guards từ hybrid
  - dùng overlap suppression dạng per-frame
  - dùng short buffer 5-10 frame để ổn định hidden/occlusion
  - không dùng suffix repair dài

  realtime_quality_delayed

  Mục tiêu: gần hybrid hơn nhưng có delay nhỏ.

  - buffer 15-30 frame
  - smoothing online trong buffer
  - local pair conflict repair trong buffer
  - vẫn không dùng các suffix repair cần 600-1500 frame
  - phù hợp dashboard/monitoring nếu chấp nhận hiển thị trễ

  4. Hướng Cải Thiện Tiếp Cho Hybrid

  Với hybrid hiện tại, không nên ưu tiên IDSW nữa vì đã đạt 0 trên 13 video.
  Nếu cải thiện tiếp thì chuyển mục tiêu sang:

  - giảm FP/FN
  - giảm fragment
  - tăng HOTA/IDF1
  - giữ remapped_idsw=0 làm hard guardrail

  Các video nên ưu tiên:

  - 000225: HOTA thấp nhất, FP cao
  - 000233: FN cao
  - 000231: fragment gap cao
  - 000216: fragment nhiều nhưng ID ổn

  Hướng kỹ thuật:

  - cải thiện detector confidence/visibility handling
  - giảm hidden false positive/false negative
  - kiểm tra gap lớn theo từng ID
  - thêm quality report theo gap dài thay vì chỉ IDSW

  5. Kế Hoạch Thử Nghiệm

  Giai đoạn 1: đóng mốc hiện tại

  - Ghi 20260707_230230 là current best 13-video identity baseline.
  - Lưu rõ config stack.
  - Không promote thêm thay đổi nếu chưa qua full 13-video.

  Giai đoạn 2: benchmark 3 mode

  Chạy cùng 13 video, cùng weight, cùng smooth_det020_loose nếu áp dụng được:

  - bytetrack_raw: baseline thô
  - hybrid_bytetrack: best quality
  - realtime: mode hiện tại

  Bảng cần báo cáo:

  - IDSW
  - IDF1
  - HOTA
  - FP/FN
  - fragments
  - FPS trung bình
  - latency p50/p95 nếu có log

  Giai đoạn 3: realtime causal guard port

  Port từng phần từ hybrid sang realtime theo thứ tự:

  - hidden owner guard
  - occlusion bad-match guard
  - small-box suppression per-frame
  - short-buffer smoothing

  Mỗi lần chỉ bật một nhóm và test lại:

  - single 000216
  - nhóm yếu 000225/000233/000231/000216
  - full 13 video
  - latency

  Giai đoạn 4: định nghĩa chuẩn mode

  Cập nhật README/config naming:

  - bytetrack_raw: raw baseline
  - hybrid_bytetrack: offline/high-quality tracking
  - realtime: low-latency streaming
  - tùy chọn thêm realtime_delayed nếu dùng buffer

  6. Kết Luận Kỹ Thuật

  Cấu hình tốt nhất hiện tại nên được giữ làm hybrid_bytetrack quality
  baseline, không nên ép nó thành realtime. Nhưng realtime có thể học từ
  hybrid ở phần association/guard causal.

  • Các bước nâng cấp realtime nên đi theo hướng không copy nguyên
  hybrid_bytetrack, mà tách phần nào dùng được online, phần nào phải bỏ hoặc
  viết lại bằng short-buffer.

  Mục tiêu realtime

  realtime nên có 3 profile rõ:

  1. realtime_fast
     Chạy nhanh nhất, detect mỗi N frame, motion/LK predict giữa các frame, ít guard.

  2. realtime_balanced
     Dùng các guard online đã chứng minh từ hybrid_bytetrack, ưu tiên giảm IDSW
     nhưng vẫn giữ latency thấp.

  3. realtime_quality_delayed
     Cho phép delay ngắn 15-30 frame để sửa local conflict, nhưng không dùng
     repair suffix dài như offline.

  Bước 1: Đóng băng baseline realtime hiện tại

  Chạy benchmark hiện tại trên cùng bộ video đã dùng cho hybrid_bytetrack:

  C:\Users\ironh\anaconda3\envs\pig_project\python.exe scripts\evaluate_tracking.py ^
    --eval-config smooth_det020_loose ^
    -a ^
    --mode realtime ^
    --rule-combo iou0_area0_condarea0_merge0 ^
    --output-root outputs\eval\realtime\baseline_current ^
    --prediction-root outputs\pred\realtime\baseline_current

  Cần ghi lại: IDSW, IDF1, HOTA, FP/FN, fragments, FPS/elapsed time.

  Bước 2: Phân loại logic hybrid

  Có thể chuyển sang realtime gần như trực tiếp:

  - hidden_owner_guard
  - hidden_owner_guard_hold_assignment
  - reentry_unowned_raw_mismatch_episode_reject nếu episode window chỉ nhìn quá khứ
  - occlusion_reid_prefer_gap_over_bad_match
  - overlap_small_box_suppression
  - motion prediction / hidden motion model
  - association diagnostics

  Không chuyển trực tiếp:

  - suffix_pair_swap_repair
  - hidden_suffix_id_swap_repair
  - mọi repair cần nhìn suffix dài/tương lai dài
  - offline smoothing toàn chuỗi

  Cần viết lại bằng short-buffer nếu muốn dùng realtime-delayed:

  - temporal smoothing
  - local pair repair
  - hidden island stabilization

  Bước 3: Tạo profile config realtime

  Thêm preset/profile kiểu:

  realtime_fast
  realtime_balanced
  realtime_quality_delayed

  Trong đó realtime_balanced nên bật thử trước:

  detect_every_n_frames=1 hoặc 2
  enable_offline_smoothing=false
  smooth_boxes=false
  refine_boxes=false
  hidden_owner_guard=true
  hidden_owner_guard_hold_assignment=true
  occlusion_reid_prefer_gap_over_bad_match=true
  overlap_small_box_suppression=true

  Bước 4: Port từng nhóm, không port một lần

  Thứ tự nên là:

  1. Motion prediction + giữ track khi mất detection ngắn.
  2. Hidden owner guard.
  3. Occlusion reid prefer-gap-over-bad-match.
  4. Overlap small-box suppression.
  5. Short-buffer smoothing nếu cần.
  6. Short-buffer local repair nếu latency cho phép.

  Sau mỗi bước chạy:

  - single weak video: 000233, 000263, 000216, 000225
  - guardrail: 000302
  - full 13 video

  Bước 5: Thêm đo latency/FPS thật

  Realtime không chỉ nhìn IDSW. Cần thêm report:

  avg_fps
  median_frame_time_ms
  p95_frame_time_ms
  detector_time_ms
  association_time_ms
  postprocess_time_ms
  buffer_delay_frames
  buffer_delay_ms

  Nếu không đo latency thì chưa thể gọi là realtime có căn cứ.

  Bước 6: Thiết kế short-buffer

  Với realtime_quality_delayed, dùng buffer nhỏ:

  buffer_size = 15 hoặc 30 frame

  Trong buffer này chỉ được sửa:

  - bbox jitter ngắn
  - local swap 2 con trong vài frame
  - hidden/unhidden island ngắn
  - gap-fill ngắn

  Không được sửa kiểu “swap từ frame 193 đến cuối video” như offline suffix repair.

  Bước 7: Báo cáo khoa học theo 3 chế độ

  Bảng nên có:

  bytetrack_raw
  realtime_fast
  realtime_balanced
  realtime_quality_delayed
  hybrid_bytetrack

  So sánh:

  IDSW
  IDF1
  HOTA
  FP
  FN
  fragments
  FPS
  latency
  causal/offline

  Kết luận hợp lý sẽ là:

  - hybrid_bytetrack: chất lượng cao nhất cho offline annotation/research.
  - realtime_balanced: ứng dụng camera trực tiếp.
  - realtime_quality_delayed: ứng dụng gần realtime, chấp nhận delay ngắn để tăng ổn định ID.

  Bước tiếp theo thực tế nên làm là benchmark realtime hiện tại trước, rồi mới
  port từng guard từ hybrid_bytetrack.
