# Scripts README

Chay lenh tu thu muc goc repo:

```cmd
cd C:\Users\ironh\Downloads\PIG_Behavior_Project
```

Neu dung moi truong ao:

```cmd
.venv\Scripts\python.exe scripts\<script_name>.py ...
```

Neu Python he thong da dung moi truong du an:

```cmd
python scripts\<script_name>.py ...
```

## 1. Entrypoint Chinh

- `track_videos.py`: chay tracking va xuat prediction/XML.
- `evaluate_tracking.py`: chay tracking + danh gia voi GT XML.
- `optimize_tracking_metrics.py`: optimizer tu dong tim cau hinh tracking tot.
- `benchmark_tracking_weights.py`: so sanh nhieu detector weights.
- `benchmark_tracking_modes.py`: so sanh tracking modes.
- `evaluate_best3_roboflow.py`: benchmark 3 video co dinh.
- `eval_hard_scenes.py`: chan doan identity tren scene kho.
- `detect_single_frame.py`: debug detector tren mot frame.

Compatibility wrappers cu van con:

- `eval_pipeline.py` -> dung thay bang `evaluate_tracking.py`.
- `run_tracking.py` -> dung thay bang `track_videos.py`.
- `run_best3_yolov8_roboflow.py` -> dung thay bang `evaluate_best3_roboflow.py`.
- `run_weight_tracking_gt_benchmark.py` -> dung thay bang `benchmark_tracking_weights.py`.
- `evaluate_recall_ablation.py` -> dung thay bang `optimize_tracking_metrics.py`.

## 2. Weight Naming

Khong dao nguoc hai file nay:

- `models\detector\pig_detector_yolov8.pt`: weight moi nhat, dang uu tien.
- `models\detector\pig_detector_yolov8_roboflow.pt`: weight cu da doi ten.

## 3. Tracking Thuong

Moi truong Python dang dung cho tracking/eval hien tai:

```cmd
C:\Users\ironh\anaconda3\envs\pig_project\python.exe
```

Mode tracking dang dung:

- `hybrid_bytetrack`: mode chinh cho tracking chat luong cao, gom cac guard/repair da duoc validate.
- `bytetrack_raw`: baseline ByteTrack thuan, dung de doi chieu khoa hoc.
- `realtime`: mode streaming/low-latency, khong mac dinh ke thua cac offline repair dai.

Eval-config realtime hien chia thanh 3 profile:

- `realtime_fast`: probe toc do, detect thua hon va it guard.
- `realtime_balanced`: probe chinh hien tai, causal-only, khong bat offline smoothing/refine/suffix repair.
- `realtime_quality_delayed`: probe quality-delayed, bat local finite-window repair
  va `realtime_motion_pair_stabilizer` short-memory; khong dung suffix repair dai.

`realtime_balanced` gom `smooth_det020_loose` detector/recovery settings voi cac
guard online da co tin hieu tot (`occlusion_aware_matching=false`,
`realtime_visible_close_competitor_guard=true`,
`realtime_visible_better_competitor_reject=true`,
`realtime_visible_better_competitor_prefer=true`,
`realtime_low_conf_recovery_guard=true`). Khong dung cac repair offline/suffix.

`realtime_quality_delayed` ke thua `realtime_balanced` va them
`realtime_motion_pair_stabilizer=true`. Stabilizer nay dung short-memory motion
voi component gate hai ID de giam IDSW sau tracking, vi vay phu hop che do
quality-delayed hon la low-latency streaming thuan.

Khong dung `--mode bytetrack` nua. Alias legacy nay da bi go bo de tranh nham voi `hybrid_bytetrack`; neu can pipeline tot nhat thi truyen ro `--mode hybrid_bytetrack`.

Chay tracking mot video voi base/preset thuong:

```cmd
C:\Users\ironh\anaconda3\envs\pig_project\python.exe scripts\track_videos.py ^
  -v Pigs291119_000263_30fps ^
  --mode hybrid_bytetrack ^
  --eval-config smooth_det020_loose
```

Chay tracking tat ca video trong path config:

```cmd
C:\Users\ironh\anaconda3\envs\pig_project\python.exe scripts\track_videos.py ^
  -a ^
  --mode hybrid_bytetrack ^
  --eval-config smooth_det020_loose
```

Chay tracking voi weight cu neu can doi chieu:

```cmd
C:\Users\ironh\anaconda3\envs\pig_project\python.exe scripts\track_videos.py ^
  -v Pigs291119_000263_30fps ^
  --mode hybrid_bytetrack ^
  --eval-config smooth_det020_loose ^
  --weights models\detector\pig_detector_yolov8_roboflow.pt
```

Giu bbox tracker noi suy nhung khong tu danh dau `Hidden=Yes` de dua len CVAT label lai:

```cmd
C:\Users\ironh\anaconda3\envs\pig_project\python.exe scripts\track_videos.py ^
  -v Pigs291119_000263_30fps ^
  --mode hybrid_bytetrack ^
  --eval-config smooth_det020_loose ^
  --no-emit-hidden-tracks
```

Chay tracking-only voi candidate opt-in hien tot nhat tren hard 5-video:

```cmd
C:\Users\ironh\anaconda3\envs\pig_project\python.exe scripts\track_videos.py ^
  -v Pigs291119_000233_30fps ^
  --mode hybrid_bytetrack ^
  --eval-config smooth_det020_loose ^
  --profile-override hidden_owner_guard=true ^
  --profile-override hidden_owner_guard_hold_assignment=true ^
  --profile-override reentry_unowned_raw_mismatch_episode_reject=true ^
  --profile-override reentry_unowned_raw_mismatch_episode_action=hold ^
  --profile-override reentry_unowned_raw_mismatch_episode_max_events=8 ^
  --profile-override reentry_unowned_raw_mismatch_episode_min_missed=1 ^
  --profile-override reentry_unowned_raw_mismatch_episode_max_missed=20 ^
  --profile-override reentry_unowned_raw_mismatch_episode_max_cost=0.36 ^
  --profile-override occlusion_reid_prefer_gap_over_bad_match=true ^
  --profile-override occlusion_reid_bad_match_action=reject ^
  --profile-override occlusion_reid_bad_match_same_raw_only=false ^
  --profile-override occlusion_reid_bad_match_raw_mismatch_only=true ^
  --profile-override occlusion_reid_bad_match_unowned_raw_only=true ^
  --profile-override occlusion_reid_bad_match_occlusion_hold_only=true ^
  --profile-override occlusion_reid_bad_match_min_missed=7 ^
  --profile-override occlusion_reid_bad_match_max_missed=12 ^
  --profile-override occlusion_reid_bad_match_min_cost=0.55 ^
  --profile-override occlusion_reid_bad_match_max_cost=0.70 ^
  --profile-override suffix_pair_swap_repair=true ^
  --profile-override overlap_small_box_suppression=true ^
  --profile-override hidden_suffix_id_swap_repair=true
```

Ghi nho luong hien tai:

- `track_videos.py` la wrapper batch; script nay goi `python -m pig_behavior.tracking.cli`.
- `track_videos.py --eval-config <name>` dung chung preset voi `evaluate_tracking.py`, roi truyen sang CLI bang cac `--profile-override KEY=VALUE`.
- `pig_behavior.tracking.cli` phai co entrypoint `if __name__ == "__main__": SystemExit(main())`; neu thieu, `python -m pig_behavior.tracking.cli` chi import module roi thoat 0.
- `track_videos.py` chu dong them `src` vao `PYTHONPATH` cho subprocess de CLI chay duoc khi project chua duoc editable-install.
- `--no-emit-hidden-tracks` van xuat bbox tracker noi suy, nhung ghi attribute `Hidden=No` de nguoi label danh lai tren CVAT; no khong tat tracking state noi bo, association, smoothing, hay occlusion logic.

## 4. Evaluate Tracking

Evaluate mot video voi base/preset thuong:

```cmd
C:\Users\ironh\anaconda3\envs\pig_project\python.exe scripts\evaluate_tracking.py ^
  --eval-config smooth_det020_loose ^
  -v Pigs291119_000263_30fps ^
  --rule-combo iou0_area0_condarea0_merge0
```

Evaluate tat ca GT video voi base/preset thuong:

```cmd
C:\Users\ironh\anaconda3\envs\pig_project\python.exe scripts\evaluate_tracking.py ^
--eval-config smooth_det020_loose ^
-a ^
--rule-combo iou0_area0_condarea0_merge0
```

Realtime balanced 5-video guard set:

```cmd
C:\Users\ironh\anaconda3\envs\pig_project\python.exe scripts\evaluate_tracking.py ^
--eval-config realtime_balanced ^
-v "Pigs291119_000231_30fps,Pigs291119_000233_30fps,Pigs291119_000263_30fps,Pigs301119_000328_30fps,Pigs291119_000302_30fps" ^
--mode realtime ^
--rule-combo iou0_area0_condarea0_merge0 ^
--output-root outputs\eval\realtime\realtime_balanced_5video ^
--prediction-root outputs\pred\realtime\realtime_balanced_5video
```

Realtime single-video probe co debug assignment:

```cmd
C:\Users\ironh\anaconda3\envs\pig_project\python.exe scripts\evaluate_tracking.py ^
--eval-config realtime_balanced ^
-v "Pigs291119_000263_30fps" ^
--mode realtime ^
--rule-combo iou0_area0_condarea0_merge0 ^
--output-root outputs\eval\realtime\probe_realtime_263_debug ^
--prediction-root outputs\pred\realtime\probe_realtime_263_debug ^
--profile-override association_debug=true
```

Realtime fast / delayed profile probe:

```cmd
C:\Users\ironh\anaconda3\envs\pig_project\python.exe scripts\evaluate_tracking.py ^
--eval-config realtime_fast ^
-v "Pigs291119_000233_30fps" ^
--mode realtime ^
--rule-combo iou0_area0_condarea0_merge0
```

```cmd
C:\Users\ironh\anaconda3\envs\pig_project\python.exe scripts\evaluate_tracking.py ^
--eval-config realtime_quality_delayed ^
-v "Pigs291119_000263_30fps" ^
--mode realtime ^
--rule-combo iou0_area0_condarea0_merge0
```

Evaluate full 12-video voi candidate opt-in hien tot nhat:

```cmd
C:\Users\ironh\anaconda3\envs\pig_project\python.exe scripts\evaluate_tracking.py ^
  --eval-config smooth_det020_loose ^
  -a ^
  --rule-combo iou0_area0_condarea0_merge0 ^
  --output-root outputs\eval\hybrid_bytetrack\visible_suffix_gate_full ^
  --prediction-root outputs\pred\hybrid_bytetrack\visible_suffix_gate_full ^
  --profile-override hidden_owner_guard=true ^
  --profile-override hidden_owner_guard_hold_assignment=true ^
  --profile-override reentry_unowned_raw_mismatch_episode_reject=true ^
  --profile-override reentry_unowned_raw_mismatch_episode_action=hold ^
  --profile-override reentry_unowned_raw_mismatch_episode_max_events=8 ^
  --profile-override reentry_unowned_raw_mismatch_episode_min_missed=1 ^
  --profile-override reentry_unowned_raw_mismatch_episode_max_missed=20 ^
  --profile-override reentry_unowned_raw_mismatch_episode_max_cost=0.36 ^
  --profile-override occlusion_reid_prefer_gap_over_bad_match=true ^
  --profile-override occlusion_reid_bad_match_action=reject ^
  --profile-override occlusion_reid_bad_match_same_raw_only=false ^
  --profile-override occlusion_reid_bad_match_raw_mismatch_only=true ^
  --profile-override occlusion_reid_bad_match_unowned_raw_only=true ^
  --profile-override occlusion_reid_bad_match_occlusion_hold_only=true ^
  --profile-override occlusion_reid_bad_match_min_missed=7 ^
  --profile-override occlusion_reid_bad_match_max_missed=12 ^
  --profile-override occlusion_reid_bad_match_min_cost=0.55 ^
  --profile-override occlusion_reid_bad_match_max_cost=0.70 ^
  --profile-override suffix_pair_swap_repair=true ^
  --profile-override overlap_small_box_suppression=true ^
  --profile-override hidden_suffix_id_swap_repair=true
```

Evaluate hard 5-video voi candidate opt-in hien tot nhat:

```cmd
C:\Users\ironh\anaconda3\envs\pig_project\python.exe scripts\evaluate_tracking.py ^
  --eval-config smooth_det020_loose ^
  -v "Pigs291119_000231_30fps,Pigs291119_000233_30fps,Pigs291119_000263_30fps,Pigs301119_000328_30fps,Pigs291119_000302_30fps" ^
  --rule-combo iou0_area0_condarea0_merge0 ^
  --output-root outputs\eval\hybrid_bytetrack\hidden_suffix_id_swap_5video_rerun ^
  --prediction-root outputs\pred\hybrid_bytetrack\hidden_suffix_id_swap_5video_rerun ^
  --profile-override hidden_owner_guard=true ^
  --profile-override hidden_owner_guard_hold_assignment=true ^
  --profile-override reentry_unowned_raw_mismatch_episode_reject=true ^
  --profile-override reentry_unowned_raw_mismatch_episode_action=hold ^
  --profile-override reentry_unowned_raw_mismatch_episode_max_events=8 ^
  --profile-override reentry_unowned_raw_mismatch_episode_min_missed=1 ^
  --profile-override reentry_unowned_raw_mismatch_episode_max_missed=20 ^
  --profile-override reentry_unowned_raw_mismatch_episode_max_cost=0.36 ^
  --profile-override occlusion_reid_prefer_gap_over_bad_match=true ^
  --profile-override occlusion_reid_bad_match_action=reject ^
  --profile-override occlusion_reid_bad_match_same_raw_only=false ^
  --profile-override occlusion_reid_bad_match_raw_mismatch_only=true ^
  --profile-override occlusion_reid_bad_match_unowned_raw_only=true ^
  --profile-override occlusion_reid_bad_match_occlusion_hold_only=true ^
  --profile-override occlusion_reid_bad_match_min_missed=7 ^
  --profile-override occlusion_reid_bad_match_max_missed=12 ^
  --profile-override occlusion_reid_bad_match_min_cost=0.55 ^
  --profile-override occlusion_reid_bad_match_max_cost=0.70 ^
  --profile-override suffix_pair_swap_repair=true ^
  --profile-override overlap_small_box_suppression=true ^
  --profile-override hidden_suffix_id_swap_repair=true
```

Probe rieng `Pigs291119_000263_30fps` voi cung candidate opt-in:

```cmd
C:\Users\ironh\anaconda3\envs\pig_project\python.exe scripts\evaluate_tracking.py ^
  --eval-config smooth_det020_loose ^
  -v Pigs291119_000263_30fps ^
  --rule-combo iou0_area0_condarea0_merge0 ^
  --output-root outputs\eval\hybrid_bytetrack\probe_263_visible_suffix_gate_rerun ^
  --prediction-root outputs\pred\hybrid_bytetrack\probe_263_visible_suffix_gate_rerun ^
  --profile-override hidden_owner_guard=true ^
  --profile-override hidden_owner_guard_hold_assignment=true ^
  --profile-override reentry_unowned_raw_mismatch_episode_reject=true ^
  --profile-override reentry_unowned_raw_mismatch_episode_action=hold ^
  --profile-override reentry_unowned_raw_mismatch_episode_max_events=8 ^
  --profile-override reentry_unowned_raw_mismatch_episode_min_missed=1 ^
  --profile-override reentry_unowned_raw_mismatch_episode_max_missed=20 ^
  --profile-override reentry_unowned_raw_mismatch_episode_max_cost=0.36 ^
  --profile-override occlusion_reid_prefer_gap_over_bad_match=true ^
  --profile-override occlusion_reid_bad_match_action=reject ^
  --profile-override occlusion_reid_bad_match_same_raw_only=false ^
  --profile-override occlusion_reid_bad_match_raw_mismatch_only=true ^
  --profile-override occlusion_reid_bad_match_unowned_raw_only=true ^
  --profile-override occlusion_reid_bad_match_occlusion_hold_only=true ^
  --profile-override occlusion_reid_bad_match_min_missed=7 ^
  --profile-override occlusion_reid_bad_match_max_missed=12 ^
  --profile-override occlusion_reid_bad_match_min_cost=0.55 ^
  --profile-override occlusion_reid_bad_match_max_cost=0.70 ^
  --profile-override suffix_pair_swap_repair=true ^
  --profile-override overlap_small_box_suppression=true ^
  --profile-override hidden_suffix_id_swap_repair=true
```

Probe rieng `Pigs291119_000233_30fps` voi cung candidate opt-in:

```cmd
C:\Users\ironh\anaconda3\envs\pig_project\python.exe scripts\evaluate_tracking.py ^
  --eval-config smooth_det020_loose ^
  -v Pigs291119_000233_30fps ^
  --rule-combo iou0_area0_condarea0_merge0 ^
  --output-root outputs\eval\hybrid_bytetrack\probe_233_hidden_suffix_id_swap_rerun ^
  --prediction-root outputs\pred\hybrid_bytetrack\probe_233_hidden_suffix_id_swap_rerun ^
  --profile-override hidden_owner_guard=true ^
  --profile-override hidden_owner_guard_hold_assignment=true ^
  --profile-override reentry_unowned_raw_mismatch_episode_reject=true ^
  --profile-override reentry_unowned_raw_mismatch_episode_action=hold ^
  --profile-override reentry_unowned_raw_mismatch_episode_max_events=8 ^
  --profile-override reentry_unowned_raw_mismatch_episode_min_missed=1 ^
  --profile-override reentry_unowned_raw_mismatch_episode_max_missed=20 ^
  --profile-override reentry_unowned_raw_mismatch_episode_max_cost=0.36 ^
  --profile-override occlusion_reid_prefer_gap_over_bad_match=true ^
  --profile-override occlusion_reid_bad_match_action=reject ^
  --profile-override occlusion_reid_bad_match_same_raw_only=false ^
  --profile-override occlusion_reid_bad_match_raw_mismatch_only=true ^
  --profile-override occlusion_reid_bad_match_unowned_raw_only=true ^
  --profile-override occlusion_reid_bad_match_occlusion_hold_only=true ^
  --profile-override occlusion_reid_bad_match_min_missed=7 ^
  --profile-override occlusion_reid_bad_match_max_missed=12 ^
  --profile-override occlusion_reid_bad_match_min_cost=0.55 ^
  --profile-override occlusion_reid_bad_match_max_cost=0.70 ^
  --profile-override suffix_pair_swap_repair=true ^
  --profile-override overlap_small_box_suppression=true ^
  --profile-override hidden_suffix_id_swap_repair=true
```

`--output-root` va `--prediction-root` chi doi thu muc ghi ket qua; chung khong doi logic tracking hay chi so neu config/code/video/weight giong nhau. Phan lam thay doi ket qua la cac dong `--profile-override`.

Mac dinh `evaluate_tracking.py` do dung mot tracking config. Day la lenh dung de bao cao chi so khoa hoc cho mot moc tracking cu the. Neu can xem preset config co san:

```cmd
C:\Users\ironh\anaconda3\envs\pig_project\python.exe scripts\evaluate_tracking.py --list-eval-configs
```

Preset `smooth_det020_loose` la ten rut gon de thay cho `iou0_area0_condarea0_merge0_smooth_det020_loose_motion`. Alias cu van duoc giu lai de khong vo lenh da dung truoc day.

Neu can tai lap benchmark-compatible matrix cu, bat ro:

```cmd
C:\Users\ironh\anaconda3\envs\pig_project\python.exe scripts\evaluate_tracking.py ^
  -a ^
  --mode hybrid_bytetrack ^
  --benchmark-compatible
```

Evaluate tat ca GT video va tat smooth:

```cmd
C:\Users\ironh\anaconda3\envs\pig_project\python.exe scripts\evaluate_tracking.py ^
  -a ^
  --mode hybrid_bytetrack ^
  --no-smooth
```

Evaluate mot video va tat smooth:

```cmd
C:\Users\ironh\anaconda3\envs\pig_project\python.exe scripts\evaluate_tracking.py ^
  -v Pigs291119_000263_30fps ^
  --mode hybrid_bytetrack ^
  --no-smooth
```
## 5. Tracking Optimizer

Nguyen tac hien tai:

- `evaluate_tracking.py` mac dinh do dung mot tracking config; dung `--benchmark-compatible` neu can tai lap matrix cu.
- `optimize_tracking_metrics.py` thay profile override cho tung candidate, sau do chay cung pipeline evaluate.
- Moi candidate optimizer chay trong process rieng de tranh state leak giua cac candidate.
- `-a/--all-videos` cua optimizer chi lay cac video co GT XML, giong evaluate.

Script:

```cmd
python scripts\optimize_tracking_metrics.py ...
```

Mac dinh optimizer:

- Dung `hybrid_bytetrack`.
- Chi toi uu rule combo `iou0_area0_condarea0_merge0`.
- Test ca `nosmooth` va `smooth`.
- Xem `smooth` la baseline chat luong hien tai; `nosmooth` chi la nhanh doi chieu.
- Neu khong truyen `--target-video`, script tu dong lay cac video yeu nhat tu baseline:
  `outputs\eval\hybrid_bytetrack\Tracking má»›i táº¯t smooth\yolov8\iou0_area0_condarea0_merge0\tracking_metrics.csv`.
- Ghi output vao `outputs\eval\hybrid_bytetrack\<run-name>\optimizer`.
- Co resume mac dinh, neu cung `--run-name` thi bo qua candidate da co `tracking_metrics.csv`.

Chay nhanh de kiem tra luong:

```cmd
python scripts\optimize_tracking_metrics.py -a --scope quick --dry-run
```

Chay probe ngan de xem smooth/no-smooth va cac video yeu nhat auto-discover:

```cmd
python scripts\optimize_tracking_metrics.py -a --scope quick --smooth-mode both --rank-by identity --run-name optimizer_probe_auto_targets
```

Chay toi uu mac dinh tren tat ca GT:

```cmd
python scripts\optimize_tracking_metrics.py -a --run-name optimizer_iou0
```

Chay balanced chi voi smooth de uu tien baseline hien tai:

```cmd
python scripts\optimize_tracking_metrics.py -a --scope balanced --smooth --rank-by identity --run-name optimizer_balanced_smooth
```

Chay `full` tracking-focused theo dung search space hien da duoc khai bao trong
`optimize_tracking_metrics.py`. Detector van anh huong truc tiep den tracking qua
confidence gate, missed detection, false positive, overlap/NMS va raw detection
budget. Tuy nhien, artifact `overnight_iou0` cho thay cac detector-only preset da
thu (`det_conf`/`nms`/raw budget thuan) cho metric y het `base`, nen khong de
nhom nay trong scope mac dinh de tranh ton thoi gian va kho doc nguyen nhan.

```cmd
python scripts\optimize_tracking_metrics.py -a --scope full --run-name overnight_iou0
```

Luu y: lenh tren du de dao huong tracking hien tai, nhung chua phai joint
detector+tracking optimization day du. Chi probe detector-only khi can xac minh
lai gia thuyet detector:

```cmd
python scripts\optimize_tracking_metrics.py -a --scope detector_probe --run-name detector_probe_iou0
```

Neu muon doc ro tac dong rieng, chay theo 2 tang thay vi tron ngay tu dau:

```cmd
python scripts\optimize_tracking_metrics.py -a --scope full --smooth --rank-by identity --run-name overnight_iou0_tracking_focused
python scripts\optimize_tracking_metrics.py -a --scope detector_probe --smooth --rank-by identity --run-name detector_probe_iou0
```

Chay `full` theo search space hien da duoc khai bao cho combo `iou1_area0_condarea0_merge0`:

```cmd
python scripts\optimize_tracking_metrics.py -a --scope full --rule-scope iou --run-name overnight_iou1
```

Chi chay no-smooth:

```cmd
python scripts\optimize_tracking_metrics.py -a --scope full --no-smooth --run-name overnight_iou0_nosmooth
```

Chi chay smooth:

```cmd
python scripts\optimize_tracking_metrics.py -a --scope full --smooth --run-name overnight_iou0_smooth
```

So sanh toi thieu base voi mot preset smooth cu the:

```cmd
python scripts\optimize_tracking_metrics.py -a --tracking-mode hybrid_bytetrack --scope full --rule-scope baseline --smooth --rank-by identity --preset base --preset smooth_responsive --run-name clean_base_vs_responsive_isolated --no-resume
```

Chay tren mot video de debug truoc:

```cmd
python scripts\optimize_tracking_metrics.py -v Pigs291119_000263_30fps --scope quick --no-smooth --run-name debug_000263
```

Chi dinh ro video muc tieu trong ranking/report:

```cmd
python scripts\optimize_tracking_metrics.py -a --scope balanced --smooth --target-video Pigs291119_000263_30fps --target-video Pigs291119_000226_30fps --target-video Pigs301119_000327_30fps --target-video Pigs301119_000328_30fps --run-name optimizer_targets
```

Xem preset optimizer:

```cmd
python scripts\optimize_tracking_metrics.py --list-presets
```

Preset hien tai khong chi quet `det_conf`. Search space duoc chia theo family:

- `baseline`: moc so sanh `base`.
- `detection`: `det_conf`, `max_raw_detections`, `nms_iou`; nhom nay anh huong tracking nhung chi nam trong `--scope detector_probe` vi artifact hien tai chua cho thay tin hieu tot hon `base`.
- `association`: `track_high_conf`, `track_match_iou`, `motion_gate`, `reid`, low-conf motion.
- `lifecycle`: `max_missing_frames`, `max_lost_frames`.
- `occlusion_identity`: identity guard, occlusion penalties, hidden motion.
- `smoothing`: smoothing alpha, refinement gap.

Quan trong:

- `--scope full` chi mo rong het cac preset da duoc script khai bao.
- `--scope full` mac dinh khong con chay detector-only preset vi artifact `overnight_iou0` da cho thay nhom nay khong co tin hieu trong cau hinh hien tai.
- Dung `--scope detector_probe` hoac `--preset <ten_preset>` neu muon chay lai detector-only.
- Khong dien giai `detector_probe` la "detector khong lien quan tracking"; day chi la cach tach thuc nghiem de doc nguyen nhan ro hon.
- Hien tai chua co lenh san nao de quet "full moi chi so" trong toan bo `src\pig_behavior\tracking\constants.py`.
- Muon co lenh do, can tiep tuc mo rong `optimize_tracking_metrics.py` de dua them cac tham so chua duoc expose vao preset/override.

Output quan trong:

- `tracking_optimizer_ranked.csv`: bang xep hang de chon cau hinh.
- `tracking_optimizer_summary.csv`: tat ca candidate, delta voi baseline, stability, Pareto.
- `tracking_optimizer_detailed_metrics.csv`: metric tung video va ALL.
- `tracking_optimizer_report.md`: bao cao top results.
- `tracking_optimizer_manifest.json`: cau hinh chay va search plan.

Cot nen xem dau tien:

- `selection_score`: diem tong hop can bang.
- `is_pareto_optimal`: candidate khong bi candidate khac ap dao tren nhieu muc tieu.
- `preset_family`: candidate thuoc nhom detection / association / lifecycle / occlusion_identity / smoothing nao.
- `remapped_hota_pct`, `remapped_idf1_pct`, `remapped_idsw`.
- `target_total_idsw`, `target_min_hota_pct`: tong IDSW va HOTA xau nhat tren cac video muc tieu.
- `target_<video_key>_*`: cac cot chi tiet cho tung video muc tieu. Video key duoc rut gon tu stem, vi du `000263`, `000226`, `000327`, `000328`.
- `fn`, `fp`, `fragments`.
- `worst_video_hota_pct`, `max_video_idsw`, `hota_std`.
- `delta_*`: chenh lech so voi preset `base` cung smooth/rule combo.
- Trong `tracking_optimizer_report.md`, xem them 2 bang:
  `Baseline Diagnostics` va `Best By Preset Family`.

## 6. Chon Rule Combo

Mac dinh:

```cmd
python scripts\optimize_tracking_metrics.py -a --scope full --run-name overnight_iou0
```

Tuong ung:

```text
iou0_area0_condarea0_merge0
USE_IOU_FALLBACK=False
USE_AREA_OCCLUSION_FREEZE=False
USE_CONDITIONAL_AREA_OCCLUSION_FREEZE=False
USE_MERGED_BOX_SPLIT=False
```

Neu muon test combo gan tuong duong nhung co IoU fallback:

```cmd
python scripts\optimize_tracking_metrics.py -a --scope full --rule-scope iou --run-name overnight_iou1
```

Tuong ung:

```text
iou1_area0_condarea0_merge0
USE_IOU_FALLBACK=True
USE_AREA_OCCLUSION_FREEZE=False
USE_CONDITIONAL_AREA_OCCLUSION_FREEZE=False
USE_MERGED_BOX_SPLIT=False
```

## 7. Smooth / No-Smooth

No-smooth trong optimizer gom:

```text
enable_offline_smoothing=False
identity_swap_guard=False
smooth_boxes=False
refine_boxes=False
```

Smooth trong optimizer gom:

```text
enable_offline_smoothing=True
identity_swap_guard=True
smooth_boxes=True
refine_boxes=True
```

Nen doc ket qua theo thu tu:

- `smooth`: baseline chat luong hien tai can bao ve.
- `nosmooth`: chi de biet loi/xau den tu post-processing hay tu tracking truoc smoothing.

Lenh chi chay no-smooth:

```cmd
python scripts\optimize_tracking_metrics.py -a --scope full --no-smooth --run-name overnight_nosmooth
```

Lenh chi chay smooth:

```cmd
python scripts\optimize_tracking_metrics.py -a --scope full --smooth --run-name overnight_smooth
```

## 8. Benchmark Weight

`evaluate_tracking.py` mac dinh do dung mot tracking config. Dung
`--benchmark-compatible` chi khi can tai lap detector/rule matrix cu; khong dung
matrix nay lam chi so cho mot config tracking don le.

De chay nhanh cac config co ten san, dung `--eval-config`. Cac ten ngan nen uu
tien:

```text
base
smooth_conservative
smooth_responsive
smooth_det020_loose
realtime_fast
realtime_balanced
realtime_quality_delayed
smooth_responsive_det020
```

Lenh cu `iou0_area0_condarea0_merge0_smooth_det020_loose_motion` van duoc giu
nhu alias tuong thich nguoc, nhung nen chuyen sang `smooth_det020_loose` de de
go va de nho hon.

Benchmark tracking voi weight mac dinh:

```cmd
python scripts\benchmark_tracking_weights.py --mode hybrid_bytetrack
```

Benchmark mode voi weight cu:

```cmd
python scripts\benchmark_tracking_modes.py --weights models\detector\pig_detector_yolov8_roboflow.pt --video data\videos\Pigs291119_000263_30fps.mp4
```

## 9. Best3 Roboflow

Chay benchmark 3 video co dinh:

```cmd
python scripts\evaluate_best3_roboflow.py --tag best3-roboflow
```

## 10. Debug

Debug detector mot frame:

```cmd
python scripts\detect_single_frame.py --video data\videos\Pigs291119_000263_30fps.mp4 --frame 547
```

Chan doan identity scene kho:

```cmd
python scripts\eval_hard_scenes.py
```

## 11. Thu Muc Script Phu

- `scripts\_internal`: tien ich noi bo cho agent/repo.
- `scripts\_legacy`: script cu, khong phai workflow chinh.
- `scripts\_shortcuts`: file `.bat` tien loi tren Windows.
