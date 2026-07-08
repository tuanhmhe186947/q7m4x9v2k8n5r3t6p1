• Kế Hoạch Tương Lai

  Kết quả mới sau khi sửa GT 216 cho thấy cấu hình hiện tại đã rất mạnh về identity: 13 video đều remapped_idsw=0. Vì vậy hướng tiếp theo
  không nên tiếp tục “vá IDSW” bừa bãi nữa, mà nên tách dự án thành 3 chế độ rõ ràng theo mục tiêu sử dụng.

  1. Phân Rõ 3 Mode

  bytetrack_raw

  Đây nên là baseline kỹ thuật: dùng ByteTrack/Ultralytics gần như nguyên bản, không dùng các guard/repair/smoothing nâng cao của project. Mục
  đích là làm mốc so sánh khoa học: “nếu không có logic cải tiến thì kết quả thế nào”.

  Trong code hiện tại, mode đúng nên gọi là bytetrack_raw. Lưu ý bytetrack đang là legacy alias và bị map về hybrid_bytetrack, nên trong báo
  cáo/README nên tránh gọi bytetrack nếu muốn nói baseline thô.

  hybrid_bytetrack

  Đây là mode chất lượng cao nhất. Nó nên là mode chính cho xuất CVAT, đánh giá khoa học, tạo annotation, và kết quả tracking cuối cùng.

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

  Đây nên là mode streaming/low-latency đúng nghĩa. Nó không nên bê nguyên toàn bộ hybrid_bytetrack sang, vì nhiều phần của hybrid cần nhìn
  toàn video hoặc nhìn suffix dài phía sau.

  Realtime nên dùng thiết kế “causal only”: chỉ dùng frame hiện tại và quá khứ, hoặc một buffer rất ngắn nếu chấp nhận delay.

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

  Các phần này có thể chạy với buffer 5-15 frame, chấp nhận delay nhỏ. Ví dụ camera realtime vẫn hiển thị trễ 0.2-0.5 giây để đổi lấy ID ổn
  định hơn.

  Không nên chuyển trực tiếp:

  - suffix_pair_swap_repair
  - hidden_suffix_id_swap_repair
  - long suffix repair
  - các repair cần nhìn hàng trăm frame tương lai

  Đây là logic hậu xử lý offline. Nếu đưa thẳng vào realtime sẽ sai bản chất “realtime”. Nếu muốn dùng, phải đổi thành delayed_realtime với
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

  Với hybrid hiện tại, không nên ưu tiên IDSW nữa vì đã đạt 0 trên 13 video. Nếu cải thiện tiếp thì chuyển mục tiêu sang:

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

  Cấu hình tốt nhất hiện tại nên được giữ làm hybrid_bytetrack quality baseline, không nên ép nó thành realtime. Nhưng realtime có thể học từ
  hybrid ở phần association/guard causal.

  • Các bước nâng cấp realtime nên đi theo hướng không copy nguyên hybrid_bytetrack, mà tách phần nào dùng được online, phần nào phải bỏ hoặc
  viết lại bằng short-buffer.

  Mục tiêu realtime

  realtime nên có 3 profile rõ:

  1. realtime_fast
     Chạy nhanh nhất, detect mỗi N frame, motion/LK predict giữa các frame, ít guard.

  2. realtime_balanced
     Dùng các guard online đã chứng minh từ hybrid_bytetrack, ưu tiên giảm IDSW nhưng vẫn giữ latency thấp.

  3. realtime_quality_delayed
     Cho phép delay ngắn 15-30 frame để sửa local conflict, nhưng không dùng repair suffix dài như offline.

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

  Bước tiếp theo thực tế nên làm là benchmark realtime hiện tại trước, rồi mới port từng guard từ hybrid_bytetrack.