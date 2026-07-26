# Báo cáo Audit & Kế hoạch cải tiến — Causal Realtime Tracking (RF_ACC23)

Ngày lập: 2026-07-26 · Phạm vi: chỉ tracking · Trạng thái: audit-only, **chưa sửa code**, chưa chạy benchmark dài, chưa full-13, không sinh MP4.

---

## 0. Authority đã đọc và skill đã chọn (bắt buộc ghi rõ)

Memory/authority đã đọc đầy đủ (staged read-only từ máy):

- `AGENTS.md` (root) và `.agents/AGENTS.md` (legacy)
- `.agents/memory/00_README.md`, `01_PROJECT_MEMORY_SHORT.md`, `02_CURRENT_DECISION.md`, `03_PROJECT_RULES.md`, `04_PROJECT_MEMORY_MEDIUM.md`, `05_PROJECT_MEMORY_LONG.md`, `06_BENCHMARK_NOTES.md`, `07_LEGACY_DIFF_NOTES.md`, `08_WORKFLOW.md`, `09_HIDDEN_REVIEW.md`
- `Kế Hoạch Tương Lai.md` (bao gồm addendum 2026-07-16 và override 2026-07-19/07-20)

Skill đã đọc (SKILL.md): `tracking-experiment-guardian`, `experiment-lineage-reproducibility`, `scientific-ablation-controller`, `computer-vision-opencv`, `safe-refactor-test-guardian`. Đã tuân thủ ordered procedure của `tracking-experiment-guardian` (đọc AGENTS + memory 01/02/03/08 + 04–07, đọc addendum trong Kế Hoạch, ghi commit gốc/semantic config/skill).

Source tracking đã đọc trực tiếp: `profiles/{realtime,hybrid_bytetrack,bytetrack_raw,__init__}.py`, `runner.py`, `detections.py`, `telemetry.py`, `config.py`, `association.py` (~2300 dòng), `occlusion.py`, `tracks.py`, `schemas.py`, `masks.py`, `geometry.py`, `constants.py`. Git metadata: `.git/HEAD`, `.git/logs/HEAD` (reflog), `.git/config`.

---

## 1. Trạng thái main & semantic diff chính xác từng mode

### 1.1. Xác minh git (Giai đoạn A — audit parent)

Đã xác minh trực tiếp từ reflog, **khớp đúng prompt**:

- `HEAD → refs/heads/main → 5fa23de` = `perf(classification-v2): cache source preflight verification`. ✔
- Commit tracking gần nhất: `d925c900…` = **`feat(tracking): promote RF_ACC23 as realtime default`** (≈ 2026-07-23). ✔ (đây là `d925c90` trong prompt)
- RF_ACC23 được ráp từ một stack cherry-pick ngay trước `d925c900`:
  `batch skipped-frame LK points` → `promote batched LK for realtime Fast` → `add foreground-core appearance descriptor` → `add core unassigned tiebreak` → `defer core histogram aggregation` → `guard core tiebreak by confidence` → `add causal pairwise core tiebreak` → `score pairwise core evidence jointly` → `share core tiebreak computations`.
- Giữa `d925c900` và `5fa23de`, **mọi commit đều là classification-v2** (không có token tracking/realtime nào trong reflog) → **tracking không đổi kể từ `d925c90`**. ✔
- `git status --porcelain -- src/pig_behavior/tracking` = rỗng (exit 0) → **worktree tracking sạch**, RF_ACC23 đã commit đúng, không có file tracking dirty. ✔

Kết luận Giai đoạn A: **semantic config RF_ACC23 trên main = đúng như code đọc được** (xem 1.3), causal delay 0 (xem 1.4), đã xác minh committed clean.

### 1.2. ⚠️ Cảnh báo lineage quan trọng (ảnh hưởng cách ghi số liệu)

Các chuỗi `RF_ACC23`, `d925c90`, `5fa23de`, `Hard6`, và các số RF_ACC23 (IDSW 53, HOTA ~97.044%, IDF1 ~97.077%, wrong-ID 8579→5219, Hard6 55→49) **không xuất hiện trong bất kỳ file `.md` memory nào** trong snapshot này. Bản memory tracking mới nhất dừng ở **2026-07-20**: `realtime_fast @ 74cad2b / 62f140b`, full-13 **IDSW 59**. Phần classification tiếp tục cập nhật tới 2026-07-26, còn phần tracking bị "đóng băng" ở 07-20.

Giải thích (đã kiểm chứng bằng reflog): RF_ACC23 promote ~07-23, **sau** mốc memory tracking 07-20; công việc tracking đã chuyển sang worktree/branch `PIG_task_tracking / task/update-tracking` (theo `03_PROJECT_RULES.md`), nên các số RF_ACC23 nằm ở lineage đó, chưa được ghi lại vào memory của repo này.

→ Theo đúng chỉ dẫn của prompt: **không tuyên bố số RF_ACC23 là kết quả mới của HEAD `5fa23de`**. Trong bảng ở mục 2, số RF_ACC23 được đánh dấu "lineage-reported, chưa tái xác minh trên repo này"; còn số `realtime_fast IDSW 59` là mức memory-verified của bản 07-20 ("Realtime Fast cũ" trong prompt). Code trên main = RF_ACC23; **con số** RF_ACC23 cần khóa lại từ artifact gốc (xem mục 8).

### 1.3. Semantic diff từng mode (đọc từ code, không suy đoán)

| Mode / profile | Detector call | detect_every_n | Association đặc thù | Post-processing | Timing contract |
|---|---|---|---|---|---|
| `bytetrack_raw` | `model.track(persist=True)` + `tracker.yaml` (ByteTrack, `with_reid:false`) | 1 | Tắt toàn bộ guard dự án; dùng thẳng ByteTrack ID | Không | online raw (baseline) |
| `realtime` → `realtime_fast` = **RF_ACC23** | `model.predict()` (raw_id = None) | **2** | batched LK; core-unassigned tiebreak (confidence-guarded); core-pairwise tiebreak; visible close/better competitor prefer+guard (far-right x≥0.67) | Không (offline smoothing OFF) | **causal_framewise, delay 0** |
| `realtime_balanced` | `model.predict()` | 1 (mặc định) | causal_hidden_detection_reservation ON; low-conf recovery guard; visible better/close competitor | Không | causal_framewise, delay 0 |
| `realtime_quality_delayed` | `model.predict()` | 1 | kế thừa balanced (reservation OFF) + local_pair_swap_repair + **realtime_motion_pair_stabilizer** | Motion-pair stabilizer chạy **sau vòng video** (đồ thị toàn cục) | **post_video_global_graph, delay −1** |
| `hybrid_bytetrack` → `hybrid_bytetrack_best` | `model.track(persist=True)` | 1 | ByteTrack ID + raw-owner guards + occlusion-reid + reentry episode reject | offline smoothing/refine + suffix/long/episode/local swap repair + near-wall & **far-camera (dùng future gap)** geometry | post_video_offline, delay −1 |

Chi tiết RF_ACC23 (`profiles/realtime.py` REALTIME_FAST_CONFIG, đã đối chiếu default trong `config.py`):
`det_conf=0.25`, `detect_every_n_frames=2`, `realtime_lk_point_batching=True`, `max_raw_detections=32`, `occlusion_aware_matching=False`; core-unassigned tiebreak (`max_cost_delta=0.01`, `min_appearance_gain=0.01`, `min_detection_iou=0.30`, `max_selected_cost=0.40`, `require_score_nondecrease=True`); core-pairwise tiebreak (`max_total_cost_increase=0.05`, `min_total_appearance_gain=0.10`, `min_detection_iou=0.30`); visible close-competitor guard (`margin=0.08`, `max_cost=0.40`, `min_center_x_ratio=0.67`, `min_hits=3`); visible better-competitor **prefer** ON, **reject** OFF; `causal_hidden_detection_reservation=False`.

### 1.4. Xác minh causal / prefix (Giai đoạn A #2)

- `resolve_output_timing_contract()` (telemetry.py): với `realtime` + `realtime_motion_pair_stabilizer=False` + `enable_offline_smoothing=False` ⇒ trả về `("causal_framewise", 0)`. RF_ACC23 rơi đúng nhánh này ⇒ **delay 0**. ✔
- Vòng chạy (`runner.py`) hoàn toàn tuần tự: `prev_frame` = đúng frame trước (`prev_frame = frame.copy()` cuối mỗi frame); LK chỉ dùng `prev_gray→curr_gray`; tiebreak/guard/reservation chỉ đọc detection frame hiện tại + trạng thái quá khứ. **Không đọc frame tương lai.** ✔
- Lưu ý R0 trong Kế Hoạch (motion-pair stabilizer chạy post-video; `local_pair_swap_repair` bị chặn sau gate `enable_offline_smoothing=false`) **chỉ ảnh hưởng `realtime_quality_delayed`**, KHÔNG ảnh hưởng RF_ACC23 (cả hai cờ đó đều OFF ở RF_ACC23). RF_ACC23 là causal delay-0 thật.
- Test/audit đã có sẵn để chạy prefix-invariance: `scripts/audit_tracking_prefix_invariance.py`, `tests/test_tracking_prefix_invariance.py`. Khuyến nghị chạy lại đúng lineage RF_ACC23 khi mở experiment (xem mục 8).

---

## 2. Bảng authority hiện có (kèm scope & giới hạn claim)

Trừ khi ghi khác: scope = **full-13, `include_hidden=true`, `iou0_area0_condarea0_merge0`, mp4_count=0**. Ưu tiên identity theo thứ tự hard-guard: permanent/terminal swap → IDSW → wrong-ID duration → HOTA/IDF1/FP-FN/fragments → (realtime) latency/throughput/delay.

| Mode | Scope | Timing | IDSW | HOTA | IDF1 | FP/FN | Frag | Độ tin cậy |
|---|---|---|---|---|---|---|---|---|
| `bytetrack_raw` | full-13, inc-hidden | online raw | **145** | 88.91% | 88.47% | — | — | Authority khóa (immutable). Không chạy lại nếu input/code contract không đổi. |
| `realtime_fast` **cũ** (74cad2b) | full-13, inc-hidden | causal, 0 | **59** | 95.63% | 95.37% | 486/610 | 107 | Memory-verified (07-20). Là parent trực tiếp của RF_ACC23. |
| **RF_ACC23** (d925c90, đang bật) | full-13, inc-hidden | causal, 0 | **53** | ~97.044% | ~97.077% | 486/610 | 107 | ⚠️ Lineage-reported (PIG_task_tracking), **chưa tái xác minh trên repo này**. Code = verified; số = cần khóa lại từ artifact. |
| RF_ACC23 — Hard6 | 6 video khó¹, inc-hidden | causal, 0 | 55→**49** | 91.317→**94.487%** | 90.593→**94.289%** | — | — | ⚠️ Lineage-reported (parent→RF_ACC23). |
| `realtime_balanced` | full-13, inc-hidden | causal, 0 | **121** | 95.68% | 95.76% | 448/586 | 127 | Memory-verified. Không thắng Fast về identity (miss target ≤86). Quality/coverage reference. |
| `realtime_quality_delayed` | full-13, inc-hidden | post-video, −1 | **166** | 97.66% | 97.58% | 449/587 | — | Memory-verified. Effective FPS ~12.09. **Không phải realtime candidate hợp lệ** (delay −1). Chỉ delayed-quality evidence. |
| `hybrid_bytetrack_best` | full-13, inc-hidden | offline, −1 | **0** | 98.3506% | 99.1490% | 1593/1593 | strict 426 / gap-tol 6 | Authority khóa (stop-gate đã đạt). Offline upper-bound. Không tối ưu thêm. |

¹ "Hard6" = realtime hard set trong memory: `000114, 000231, 000233, 000263, 000327, 000302` (cần xác nhận đây đúng là tập RF_ACC23 đã dùng — mục 8).

Giới hạn claim quan trọng:
- Số RF_ACC23 (IDSW 53 / wrong-ID 8579→5219 / Hard6 55→49) **chưa đủ tư cách authority trên repo này** cho tới khi khóa được run manifest + hash artifact của lineage d925c90. Chú ý: FP/FN (486/610) và fragments (107) **giống hệt** bản 59 → nhất quán với việc RF_ACC23 chỉ sửa **identity** (2 tiebreak core-hist), không đổi tập bbox; đây là bằng chứng gián tiếp mạnh nhưng không thay cho việc tái đo.
- **Runtime chưa có winner native**: theo memory (07-20), cả Fast (~27.34/27.61 FPS) và Balanced (~28.72/28.89 FPS) **chưa đạt gate 30 FPS**. Đây là số bản 07-20; RF_ACC23 thêm batched LK (giảm overhead LK) nhưng **chưa có đo runtime công bằng nào cho RF_ACC23**.
- Quality và runtime phải là **2 bảng riêng**; kết luận không được dựa trên 1 lần chạy FPS. Bảng paper mục tiêu: `bytetrack_raw → RF_ACC23 (hoặc successor causal) → hybrid_bytetrack_best`.

---

## 3. Residual weaknesses còn lại của RF_ACC23

Phân tích từ code (association.py/occlusion.py) + evidence memory. Phân loại theo taxonomy Giai đoạn A #3.

R1 — **Identity theft khi crowding/occlusion (điểm yếu kiến trúc lớn nhất)**. RF_ACC23 đặt `occlusion_aware_matching=False` ⇒ `build_occlusion_context` trả context rỗng: `occluded_track_ids`, `detection_competitors`, `active_detection_owners` đều rỗng ⇒ `occlusion_assignment_penalty ≡ 0`, `assignment_is_occlusion_ambiguous ≡ False`. Nghĩa là **khi hai con chồng nhau, matcher Hungarian không có bất kỳ occlusion cost shaping nào** — chỉ còn IoU+center+full-hist+area. Đây là nơi dễ sinh long wrong-ID segment và (xấu nhất) permanent swap. Event types: overlapping/crowding, fight (000263 f193/195, 792/846/865; 000233 f1111–1242, 1424+).

R2 — **Không có reservation: visible track "cướp" detection của hidden/occluded track**. `causal_hidden_detection_reservation=False` ở RF_ACC23 (chỉ balanced bật). Code reservation ĐÃ tồn tại và causal (call site trong pha `visible_high_conf`), nhưng RF_ACC23 early-return. Hệ quả: khi một track đang hold/occluded, một visible track có thể chiếm mất detection đúng của nó ⇒ mất ID và có thể swap. Đây đúng là lỗ hổng "uncertainty/hold thay vì cướp bbox".

R3 — **Appearance memory thô sơ**. `mean_hist()`/`mean_core_hist()` = trung bình cộng **không trọng số** trên tối đa 80 frame (deque), **không EMA, không confidence-weighted**. Một detection far-right mờ (score 0.26) đóng góp ngang một detection gần nét (score 0.95). Làm yếu re-ID và tiebreak đúng ở vùng khó nhất. Event types: far-right/small/blur, lost/re-entry, long wrong-ID.

R4 — **Cost chính vẫn dùng `full_hist`, core-hist chỉ dùng ở tiebreak**. `appearance_hist_foreground_core=False` ⇒ `det.hist = full_hist` cho Hungarian; `core_hist` (đã trừ nền, elip foreground) chỉ dùng ở 2 tiebreak sau LAP. Vậy lợi ích "chống lẫn màu nền/hàng xóm" của core-hist bị giới hạn ở các cặp gần-hòa, không áp vào bước match chính khi crowding.

R5 — **Camera prior chỉ là 1 gate cứng, không phải cost term**. Logic camera-aware duy nhất trong realtime là gate `min_center_x_ratio=0.67` của close-competitor guard (chỉ kích hoạt ở 1/3 phải khung). Không có size-by-position prior, không có cost liên tục theo hình học camera. Vùng far-right (xa/nhỏ/mờ) và wall-interaction không được prior hóa trong association. (Các cơ chế `near_wall_/far_camera_hidden_geometry` là **offline/hybrid** và có cái dùng future gap → không port thẳng.)

R6 — **Frame skipping (detect_every_n=2)**. Ở frame skip chỉ có LK motion prediction, không detect. Nếu occlusion/chuyển động lớn xảy ra đúng frame skip, tracker chỉ dựa LK (kém khi occlusion). Là đánh đổi throughput↔identity ngay tại thời điểm khó. Cần tách như một family tốc độ riêng (R3 trong Kế Hoạch), không trộn với association.

R7 — **Video/sự kiện yếu đã biết** (evidence memory, phần lớn là era cũ nhưng khả năng còn): `000233` (video realtime yếu nhất; xung đột ID_1/ID_8 dài), `000263` (reid switch trong fight; legacy IDSW ≈2 vs hiện ≈6 — nghi `association.py` raw/lost-reid), `000216` (far-camera ID 5/8 — **phải khóa GT trước** vì memory cảnh báo GT stale), `000328` (far-right hidden bbox đè visible, f1347–1355), `000231` (gain-concentration fragility), `000302` (đã về 0 nhờ far-right guard — là **guardrail phải giữ**). IDSW 53 của RF_ACC23 phân bố trên các video này; **chưa có breakdown per-video** trong repo (mục 8).

R8 — **Ràng buộc coupling coverage↔occlusion**. Lịch sử realtime từng FN/recall rất tệ (recall ~60.58% ở một baseline), được cứu bằng chính `occlusion_aware_matching=false`. Vì vậy bất kỳ cải tiến nào bật lại occlusion-aware cost phải chứng minh **không** làm regress FN — đây là guardrail cứng cho R1/R2.

---

## 4. Tối đa 3 hướng cải tiến (đã xếp hạng)

Tất cả đều: causal (past-only), delay 0, đổi **đúng một scientific family**, không hard-code tên video/GT/frame, không sinh MP4, tách commit thuật toán vs commit promote profile.

Xếp hạng: **H1 (reservation/hold) > H2 (appearance memory) > H3 (camera prior)**.

### H1 — Causal hidden-detection reservation + "hold thay vì cướp" cho RF_ACC23 ⭐ ưu tiên
- Family: **association — reservation/hold** (một family).
- Cơ chế: bật cơ chế `apply_causal_hidden_detection_reservation` (ĐÃ có, đã proven causal ở balanced) trong một biến thể `realtime_fast_reservation`, **retune ngưỡng cho `det_conf=0.25`** (ngưỡng hiện set cho balanced `det_conf=0.20`); và/hoặc thêm nhánh "uncertainty-hold" trong accept loop dùng primitive `freeze_area_occluded_track` sẵn có khi top-2 cost của một detection nằm trong margin bất định.
- Giả thuyết falsifiable: *"Bật causal reservation/hold trong RF_ACC23 giảm full-13 IDSW và wrong-ID matched-animal frames trên sự kiện crowding/occlusion, với FP/FN tăng ≤0.5%, không có permanent/terminal swap regression, giữ delay 0."* → Bác bỏ nếu IDSW không giảm ≥2, hoặc xuất hiện permanent swap mới, hoặc 000302 vượt trần frozen.
- Event types mục tiêu: overlapping/crowding, occlusion onset, hidden-track detection contention (000263, 000233).
- Rủi ro permanent/terminal swap: **Thấp–TB**. Reservation *hold* thay vì commit → giảm terminal swap. Rủi ro là over-hold (từ chối detection đúng tạm thời), nhưng hold hồi phục được. Chặn bằng `min_iom`/`min_gain`/`max_alternative_cost` (giữ ≤0.25 như balanced đã học để tránh sập 000216 ở 0.30).
- Runtime overhead dự kiến: **Thấp**. Chỉ re-solve LAP khi có claim hidden (hiếm). Đã đo ở balanced, không tụt FPS đáng kể.
- Gates (funnel): static/synthetic (crossing/occlusion/low-conf/long-gap) → frozen windows từ **parent RF_ACC23** (000263 f193/195, 000233 f1111) → ≥2 independent positive episodes → 1 full difficult video (000233 hoặc 000263) → Hard6 + guardrail 000302 → full-13 chỉ khi ≥2 video khó cải thiện → exact repeat finalist.
- Stop rule & rollback: **Stop** nếu có permanent/terminal swap, 000302 vượt trần, FP/FN >0.5%, hoặc frozen window regress. **Rollback** = tắt cờ trong `profiles/realtime.py` (một dòng config; thuật toán tái dùng code có sẵn) → reversible tức thì. Không đụng `realtime_fast` đã promote (tạo biến thể riêng).

### H2 — Confidence-weighted / EMA causal appearance memory
- Family: **appearance/cost model** (một family). *Không* trộn với H2b bên dưới trong cùng experiment.
- Cơ chế: thay trung bình cộng 80-frame trong `schemas.py` (`mean_hist`/`mean_core_hist`, append ở `update_detected`) bằng **running mean có trọng số theo `det.score`** (hoặc EMA). Cost `track_detection_cost` đã gọi `mean_hist()` nên không đổi call-site. (H2b tùy chọn tách riêng: chuyển cost chính sang `core_hist` qua cờ `appearance_hist_foreground_core=True` — đường dẫn đã tồn tại.)
- Giả thuyết: *"Template ngoại hình có trọng số theo confidence giảm wrong-ID duration và lỗi re-ID ở far-right/small/blur và re-entry sau occlusion, cải thiện HOTA/IDF1 và IDSW, không regress FP/FN, delay 0."*
- Event types: far-right/small/blur, lost/re-entry, long wrong-ID segment.
- Rủi ro permanent/terminal swap: **TB** — vì template ảnh hưởng **mọi** match (tác động rộng). Template xấu có thể sinh swap mới. Giảm thiểu: giữ là reweighting thuần (khoảng cách vẫn [0,1]), ablate riêng scheme trọng số, screening window nghiêm.
- Runtime overhead: **Không đáng kể** (cùng phép histogram, chỉ đổi trọng số tổng hợp).
- Gates: như funnel H1; vì tác động rộng, bắt buộc Hard6 + repeat và soi kỹ per-video. H2 và H2b là **hai experiment tách rời**.
- Stop rule & rollback: **Stop** nếu có permanent swap mới hoặc HOTA/IDF1 tụt ngoài budget. **Rollback** = revert hàm tổng hợp trong `schemas.py` (một hàm cô lập).

### H3 — Causal camera-aware size/spatial prior từ hình học mask
- Family: **cost prior** (một family — thêm một additive cost term).
- Cơ chế: thêm helper `apply_camera_size_prior_to_costs(...)` song song `apply_directional_y_prior_to_costs` (chèn ngay sau khi dựng cost matrix), phạt các gán mà **kích thước detection không hợp lý theo vị trí x** (phải=xa=nhỏ, trái=gần=lớn), dùng bbox mask (`masks.py`) + thống kê size per-track quá khứ. Thay thế causal cho cơ chế far-camera offline (vốn dùng future gap).
- Giả thuyết: *"Prior kích-thước-theo-vị-trí camera (causal) giảm lỗi identity vùng far-right/small và swap khi tương tác tường, không regress video near-field, delay 0."*
- Event types: far-right/small/blur, wall interaction (000328, 000216-sau-khi-khóa-GT).
- Rủi ro permanent/terminal swap: **Thấp–TB** (penalty mềm, bounded như directional_y ~0.12). Rủi ro: phạt nhầm khi con vật lại gần camera (size tăng hợp lệ) → chặn bằng cho phép thay đổi size theo vận tốc/độ mượt.
- Runtime overhead: **Thấp** (một phép cộng cost vector hóa/frame).
- Gates: static/synthetic (far-right crossing) → frozen far-right windows (000328 f1347; 000216 chỉ sau khi khóa GT) → ≥2 episodes → full 000328 → Hard6 + guardrail → full-13. **Ablate prior độc lập**, tuyệt đối không hard-code video.
- Stop rule & rollback: **Stop** nếu near-field regress hoặc phát hiện hard-code video/frame. **Rollback** = gỡ call-site (helper cô lập).

---

## 5. Đề xuất benchmark runtime công bằng (Giai đoạn B — chuẩn hóa trước khi tối ưu tốc độ)

Nguyên tắc: **quality table và runtime table tách riêng**; kết luận không dựa 1 lần FPS; đo trên phần cứng đã khai báo.

Ba tầng (đúng yêu cầu, dùng cùng cached `model.predict()` observations):

1. Tracker-only (cô lập association — cái ta thay đổi): chạy `model.predict()` **một lần**/video, cache tensor detection (box, score, mask, full_hist, core_hist, raw_id=None) ra đĩa kèm hash; sau đó chỉ chạy `match_and_update_tracks` + hold/LK trên cache. Baseline RF_ACC23 và candidate **đọc cùng cache** ⇒ công bằng tuyệt đối cho phần identity.
2. Detector-only: chạy riêng `model.predict()` (không association) để đo latency/throughput/VRAM detector — chi phí dùng chung cho mọi realtime profile.
3. End-to-end core realtime: full path RF_ACC23 (detect mỗi 2 frame + LK skip + association) đo effective FPS, p50/p95 frame latency, backlog, output age, deadline-miss vs 30 FPS, VRAM, RSS.

Kỷ luật đo (bắt buộc):
- Warm-up N frame trước khi tính giờ; **≥5 valid repeats**; baseline↔candidate chạy **paired và hoán đổi thứ tự** (chống drift nhiệt/clock).
- Dùng **CUDA Events** hoặc `torch.cuda.synchronize()` quanh vùng đo (hiện code dùng `time.perf_counter` wall-time — cần bổ sung GPU-accurate timing).
- Ghi **GPU util, VRAM, temperature, power, clocks, P-state** (NVML/pynvml lấy mẫu định kỳ) + **CPU/RAM** (psutil).
- **Đánh dấu contaminated run** (thermal throttle, tải GPU lạ, util bất thường) — không âm thầm đưa vào trung bình. Báo cáo **median + IQR** trên ≥5 repeat valid, paired.
- Không dùng `raw_id` ByteTrack làm detector output. Muốn đo ByteTrack tracker-only phải viết adapter riêng feed cached detector tensors vào ByteTrack (tùy chọn, thứ cấp).
- No-MP4: `write_output_video=False`, fresh output root, recursive `mp4_count=0` sau mỗi run.

Tái dùng hạ tầng đã có (không viết lại từ đầu):
- `telemetry.py` đã có schema: detector/association/postprocess/frame time (total/mean/p50/p95), `effective_fps`, `tracking_loop_effective_fps`, `realtime_factor`, backlog (`max_backlog_frames`), `output_age_ms_*`, `frame_deadline_miss_*`, `peak_process_rss_bytes`, `peak_cuda_memory_*`. → chỉ cần bổ sung CUDA-event timing + NVML env sampling.
- `scripts/benchmarks/` (thư mục đã tồn tại — **cần soi trước**, có thể chính là "common GPU harness" ở commit `7c9179e` memory nhắc), `scripts/run_tracking_mode.py`, `scripts/audit_tracking_prefix_invariance.py`, `scripts/audit_tracking_repeatability.py`.

Gate promote runtime (từ Kế Hoạch/workflow): repeat effective-FPS ≥ 90% primary; repeat peak RSS/CUDA ≤ 110% primary; **native 30 FPS hoặc drop policy khai báo rõ**; báo cáo đầy đủ p50/p95, stage timing, delay, backlog, output age.

---

## 6. Danh sách file dự kiến sửa/thêm & vai trò

Chưa sửa gì ở bước này. Dự kiến khi được duyệt:

Thuật toán (theo hướng chọn):
- `src/pig_behavior/tracking/profiles/realtime.py` — thêm **biến thể experiment** (vd `realtime_fast_reservation`); **không** đổi `realtime_fast` (RF_ACC23) đã promote. Commit riêng cho profile.
- `src/pig_behavior/tracking/association.py` — H1: retune/mở reservation cho realtime + (tùy) nhánh uncertainty-hold trong accept loop; H3: thêm helper `apply_camera_size_prior_to_costs` + 1 call-site sau khi dựng cost.
- `src/pig_behavior/tracking/schemas.py` — H2: đổi tổng hợp appearance (`mean_hist`/`mean_core_hist`/`update_detected`) sang confidence-weighted/EMA.
- `src/pig_behavior/tracking/detections.py` — H2b (nếu tách): bật đường `appearance_hist_foreground_core` cho cost chính.
- `src/pig_behavior/tracking/config.py` — thêm field cờ/ngưỡng mới cho biến thể (kèm validate range).

Hạ tầng benchmark/telemetry (prediction-invariant, commit `chore` riêng):
- `src/pig_behavior/tracking/telemetry.py` — CUDA-event timing option + hook NVML env sampling.
- `scripts/benchmarks/<harness 3 tầng>` (mở rộng cái có sẵn) + module cache detection mới (dưới `scripts/benchmarks/` hoặc `tracking/`).

Tests (commit `test` riêng, trước khi bật candidate):
- `tests/test_tracking_profiles.py`, `test_tracking_prefix_invariance.py`, `test_tracking_telemetry.py`, `test_tracking_improvements.py` (regression), `test_association_lost_reacquire.py`, `test_tracking_no_mp4.py`, `test_tracking_repeatability.py`.

Tài liệu/quyết định (markdown/JSON, standing approval cho .md):
- `docs/TRACKING_*_DECISION_*.json` (lineage/promotion), cập nhật `.agents/memory/06_BENCHMARK_NOTES.md` khi có evidence RF_ACC23.

Thứ tự commit (theo Kế Hoạch §5): docs lock → tests → disable-video → telemetry → **1 family candidate (opt-in)** → docs evidence → promote profile (chỉ sau full-13 + repeat PASS).

---

## 7. Thông tin còn thiếu cần xác minh

1. **Số RF_ACC23 chưa khóa trên repo này**: cần run manifest + hash artifact của lineage `d925c90` (output root, `tracking_metrics.csv` per-video) để chứng minh IDSW 53 / HOTA 97.044 / IDF1 97.077 / wrong-ID 8579→5219 / Hard6 55→49. Hiện chỉ có "code = RF_ACC23" được xác minh; **số cần tái đo hoặc import artifact**.
2. **Breakdown per-video của IDSW 53**: video nào đang giữ switch — không có trong memory; cần CSV full-13 của RF_ACC23 để chọn frozen windows đúng.
3. **Đo runtime RF_ACC23**: chưa có benchmark công bằng nào cho RF_ACC23 (số 27.3/28.7 FPS là bản 07-20 trước batched-LK). Cần chạy harness 3 tầng (khi được duyệt).
4. **Path chính xác của common-GPU harness** dưới `scripts/benchmarks/` (chưa mở) — soi trước khi mở rộng, tránh trùng.
5. **Xác nhận "Hard6"** = `000114/000231/000233/000263/000327/000302` đúng là tập RF_ACC23 đã dùng.
6. **Khóa GT `000216`** trước mọi việc far-camera (memory cảnh báo GT stale ở một run).
7. **detect_every_n=2 và claim identity**: xác nhận IDSW 53 đo tại `detect_every_n_frames=2` (để tách ảnh hưởng frame-skip khỏi identity).
8. **`mask.png`**: xác nhận path/tồn tại cho hướng H3 (camera prior) và tham số hình học.
9. **Final unbiased sessions**: tập test cuối phải chọn **trước** khi tối ưu (stratified theo độ khó/camera/session), giữ cả easy/typical/hard; current-13 là development set, không phải final test.
10. **Chạy lại prefix-invariance/repeatability audit** đúng lineage RF_ACC23 (`audit_tracking_prefix_invariance.py`, `audit_tracking_repeatability.py`) để đóng Giai đoạn A trước khi mở candidate.

---

## 8. Khuyến nghị làm trước & vì sao

Làm **H1 (causal reservation/hold) trước**, vì:

1. Rủi ro thấp nhất, reversible nhất: tái dùng code reservation **đã committed & đã chứng minh causal** ở balanced; mở qua một biến thể profile (một dòng), không đụng `realtime_fast` đã promote.
2. Đánh trúng residual giá trị cao nhất (R1+R2): RF_ACC23 hiện **không có occlusion cost shaping** và **không có reservation**, nên "visible cướp detection của hidden track" là nguồn swap dài/permanent lớn nhất.
3. "Hold thay vì cướp" đúng với thứ tự hard-guard #1 (tránh permanent/terminal swap) — giảm rủi ro thay vì thêm.
4. Khớp port order người dùng đã ghi trong Kế Hoạch (motion/hold + hidden-owner guard trước) và kỷ luật one-family của `tracking-experiment-guardian`.

Nhưng **trước cả H1**, chạy tiền đề bắt buộc (không đổi thuật toán, không benchmark dài):
- P0/P1: khóa lineage 13-video (path+SHA256 video/GT/weight/mask), đóng gate no-MP4 (`write_output_video=False` + test), thêm test causality/prefix, **khóa số RF_ACC23** từ artifact (mục 7.1–7.2).
- Sau đó mở H1 theo funnel: static/synthetic → frozen windows từ parent → ≥2 episodes → full difficult video → Hard6 + 000302 guardrail → full-13 (chỉ khi ≥2 video khó cải thiện) → exact repeat.

Thứ tự tổng thể đề xuất: **P0/P1 (lineage+no-MP4+tests+khóa số) → H1 → H2 → H3**, mỗi bước một family, tách commit thuật toán vs promote, không full-13 và không MP4 cho tới khi gate được duyệt.

---

## Phụ lục — trạng thái đã xác minh vs cần xác minh

- ✔ Verified (code+git): HEAD `5fa23de`; RF_ACC23 promote `d925c90`; tracking unchanged since `d925c90`; worktree tracking sạch; semantic config từng mode; RF_ACC23 = causal delay 0; RF_ACC23 = batched LK + foreground-core descriptor + core unassigned tiebreak (confidence-guarded) + causal pairwise core tiebreak + close/better competitor guard.
- ✔ Memory-verified (đóng băng 07-20): bytetrack_raw 145; realtime_fast(cũ) 59; balanced 121; quality_delayed 166; hybrid 0; scope/timing contract; runtime chưa đạt 30 FPS (bản 07-20).
- ⚠️ Lineage-reported, **chưa tái xác minh trên repo này**: RF_ACC23 IDSW 53 / HOTA 97.044 / IDF1 97.077 / wrong-ID 8579→5219 / Hard6 55→49.
