# Scripts README

Chay cac lenh tu thu muc goc repo:

```cmd
cd C:\Users\ironh\Downloads\PIG_Behavior_Project
```

Python khuyen dung:

```cmd
C:\Users\ironh\anaconda3\envs\pig_project\python.exe
```

## Main Entrypoints

- `run_tracking_mode.py`: lenh gon de chay/so sanh cac mode trinh bay.
- `track_videos.py`: chi chay tracking va xuat prediction/XML.
- `evaluate_tracking.py`: chay tracking neu can, sau do danh gia voi GT XML.
- `optimize_tracking_metrics.py`: tim cau hinh tracking bang nhieu preset.

Thu muc phu:

- `scripts\benchmarks`: benchmark detector/tracking.
- `scripts\diagnostics`: debug frame, hard-scene identity.
- `scripts\integrations`: tich hop ngoai nhu Roboflow.
- `scripts\behavior_review_tools`: tooling review behavior/classification.
- `scripts\dev_tools`: tool phat trien/pilot, khong phai workflow chinh.
- `scripts\_legacy`: wrapper cu de tham chieu/tuong thich.
- `scripts\_shortcuts`: file `.bat` tien loi tren Windows.

## Mode vs Eval Config

`--mode` va `--eval-config` khac nhau:

- `--mode`: runtime engine/duong chay tracker.
- `--eval-config`: bo override `TrackingConfig` da dat ten trong `src\pig_behavior\tracking\profiles\*.py`.

Gia tri runtime `--mode` trong tracking/eval truc tiep:

- `bytetrack_raw`
- `realtime`
- `hybrid_bytetrack`

Vi du:

```cmd
--mode hybrid_bytetrack --eval-config hybrid_bytetrack_best
```

Nghia la chay engine `hybrid_bytetrack`, roi nap bo override `hybrid_bytetrack_best`.

`run_tracking_mode.py --mode` la lop goi gon de trinh bay. No map ten trinh bay thanh cap runtime mode + eval-config:

- `bytetrack_raw` -> `--mode bytetrack_raw --eval-config bytetrack_raw`
- `realtime` -> `--mode realtime --eval-config realtime_quality_delayed`
- `realtime_fast` -> `--mode realtime --eval-config realtime_fast`
- `realtime_quality_delayed` -> `--mode realtime --eval-config realtime_quality_delayed`
- `hybrid_bytetrack` -> `--mode hybrid_bytetrack --eval-config hybrid_bytetrack_best`

## Eval Configs

`--eval-config` hien gom cac nhom sau.

Hybrid configs tu `src\pig_behavior\tracking\profiles\hybrid_bytetrack.py`:

- `base`: hybrid base, bat offline smoothing/refine/identity guard co ban.
- `smooth_conservative`: hybrid base voi smoothing cham/on dinh hon.
- `smooth_responsive`: hybrid base voi smoothing nhanh/nhay hon.
- `smooth_det020_loose`: hybrid base voi `det_conf=0.20`, recovery loose hon.
- `smooth_responsive_det020`: responsive smoothing + `det_conf=0.20`.
- `hybrid_bytetrack_best`: best hybrid hien tai, gom cac guard/repair da validate.
- `iou0_area0_condarea0_merge0_smooth_det020_loose_motion`: alias cu cho `smooth_det020_loose`.

Realtime configs tu `src\pig_behavior\tracking\profiles\realtime.py`:

- `realtime_fast`: uu tien toc do/latency, co frame skipping, khong dung delayed repair.
- `realtime_balanced`: realtime day du detection hon, co cac guard runtime can bang.
- `realtime_quality_delayed`: realtime chat luong cao nhat hien tai, co short-delay motion-pair stabilizer.

Raw ByteTrack config tu `src\pig_behavior\tracking\profiles\bytetrack_raw.py`:

- `bytetrack_raw`: baseline ByteTrack thuan, tat cac guard/repair rieng cua du an.

Voi `track_videos.py` hoac `evaluate_tracking.py`, neu muon dung realtime quality-delayed thi ghi ro:

```cmd
--mode realtime --eval-config realtime_quality_delayed
```

Neu chay `run_tracking_mode.py --mode realtime`, wrapper tu map sang cap tren. No khong tu dong chay ca 3 realtime variants.

## `run_tracking_mode.py`

Dung khi can lenh gon de trinh bay hoac so sanh mode.

Option chinh:

- `--mode`: chon mode trinh bay cua wrapper, vi du `hybrid_bytetrack`, `realtime`, `bytetrack_raw`.
- `--compare-modes`: danh sach mode cho `--task compare`; mac dinh `bytetrack_raw,realtime,hybrid_bytetrack`.
- `--task track`: chi tao prediction/XML.
- `--task eval`: danh gia mot mode voi GT; mac dinh de `evaluate_tracking.py` track neu prediction thieu.
- `--task compare`: chay eval tren nhieu mode, sau do ghi `mode_comparison_summary.csv` va `mode_comparison_summary.md`.
- `--eval-existing`: voi `--task eval`, chi danh gia prediction da co san, khong track them.
- `-v`, `--video`: chon 1 hoac nhieu video, cach nhau bang dau phay.
- `-a`, `--all-videos`: chay tat ca video trong path config.
- `--rule-combo`: chon rule combo eval, mac dinh `iou0_area0_condarea0_merge0`.
- `--all-rule-combos`: khong ep rule combo mac dinh, cho evaluation chay full rule matrix.
- `--dry-run`: in command se chay, khong thuc thi.
- `--compare-output-root`: thu muc output cho compare task.
- `--compare-prediction-root`: thu muc prediction cho compare task.
- `--path-profile`: path profile, forward thanh `--profile` cho script ben duoi.
- `--path-config`: file path config tuy chinh.
- `--list-modes`: xem cac mode co san.

Xem mode:

```cmd
C:\Users\ironh\anaconda3\envs\pig_project\python.exe scripts\run_tracking_mode.py --list-modes
```

Chay tracking + eval cho 1 video:

```cmd
C:\Users\ironh\anaconda3\envs\pig_project\python.exe scripts\run_tracking_mode.py ^
  --mode hybrid_bytetrack ^
  --task eval ^
  -v "Pigs291119_000263_30fps"
```

Chi tracking:

```cmd
C:\Users\ironh\anaconda3\envs\pig_project\python.exe scripts\run_tracking_mode.py ^
  --mode bytetrack_raw ^
  --task track ^
  -v "Pigs291119_000263_30fps"
```

Chi eval prediction da co san:

```cmd
C:\Users\ironh\anaconda3\envs\pig_project\python.exe scripts\run_tracking_mode.py ^
  --mode hybrid_bytetrack ^
  --task eval ^
  --eval-existing ^
  -a
```

So sanh 3 mode chinh:

```cmd
C:\Users\ironh\anaconda3\envs\pig_project\python.exe scripts\run_tracking_mode.py ^
  --task compare ^
  -v "Pigs291119_000263_30fps"
```

So sanh 3 realtime variants:

```cmd
C:\Users\ironh\anaconda3\envs\pig_project\python.exe scripts\run_tracking_mode.py ^
  --task compare ^
  --compare-modes realtime_fast,realtime_balanced,realtime_quality_delayed ^
  -v "Pigs291119_000263_30fps"
```

Sau khi `--task compare` chay xong, summary nam trong compare output root:

- `mode_comparison_summary.csv`
- `mode_comparison_summary.md`
- `mode_runtime_summary.csv`
- `mode_runtime_summary.md`
- `mode_scientific_summary.csv`
- `mode_scientific_summary.md`

Compare summary gom nhieu nhom chi so de tranh ket luan chi dua tren IDSW/HOTA:

- Protocol/mode semantics: `baseline_role`, `causality_level`,
  `uses_offline_smoothing`, `uses_identity_repair`, `uses_delayed_repair`,
  `detect_every_n_frames`, `latency_window_frames`.
- Chat luong detection/tracking: `gt_detections`, `pred_detections`, `matches`,
  `fp`, `fn`, `precision_pct`, `recall_pct`, `mota_pct`, `motp_iou_pct`,
  `idf1_pct`, `hota_pct`.
- Chat luong identity sau remap: `remapped_idsw`, `remapped_mota_pct`,
  `remapped_idf1_pct`, `remapped_hota_pct`, `remapped_assa_pct`,
  `idmap_coverage_pct`.
- Do lien tuc track: `fragments`, `remapped_fragments`,
  `gap_tolerant_fragments`, `remapped_gap_tolerant_fragments`, `tracklets`,
  `remapped_tracklets`.
- Toc do/thuc te: `compare_elapsed_sec`, `compare_evaluated_fps`,
  `video_duration_sec`, `compare_realtime_factor`.

`compare_elapsed_sec` la thoi gian cua subprocess compare cho tung mode, gom ca
tracking neu prediction thieu va phan evaluation/report. `compare_realtime_factor`
lon hon `1.0` nghia la nhanh hon realtime tren tong thoi luong video da evaluate.

`mode_scientific_summary.csv` gom mot dong moi mode, phu hop de dua vao bang
paper: total metrics lay tu dong `ALL`, con mean/std/median tinh tren tung video
va bo qua dong `ALL`. Cach nay tranh viec mot video dai lan at toan bo ket luan.

Ghi chu khi viet paper:

- `bytetrack_raw` la raw ByteTrack baseline trong cung detector/input pipeline:
  tat offline smoothing, identity guard, hidden motion, local/suffix repair,
  overlap suppression, hidden suffix repair va realtime stabilizer. Khong nen
  goi la benchmark chinh thuc cua moi bien the ByteTrack ben ngoai repo.
- `realtime_fast` va `realtime_balanced` la realtime/online candidates vi tat
  offline smoothing. `realtime_quality_delayed` la short-delay realtime candidate
  vi co window repair/stabilizer; can bao cao `latency_window_frames`.
- `hybrid_bytetrack` voi `hybrid_bytetrack_best` la offline quality profile vi
  bat `enable_offline_smoothing`, `smooth_boxes`, `refine_boxes` va cac repair.
  Neu profile nay cho ket qua tot nhat, viet nhu mot offline/post-processed upper
  bound hoac quality-oriented mode. Dieu nay khong lam giam gia tri bai bao neu
  duoc noi ro; nguoc lai, no lam trade-off accuracy/latency minh bach hon.

## `track_videos.py`

Dung khi chi can tao prediction/XML, khong can tinh metric.

Option chinh:

- `-v`, `--video`: video name/key/path, co the truyen nhieu video bang dau phay.
- `-a`, `--all-videos`: track tat ca video trong path config.
- `--mode`: runtime tracking engine, vi du `hybrid_bytetrack`, `realtime`, `bytetrack_raw`.
- `--eval-config`: bo override `TrackingConfig` da dat ten, vi du `hybrid_bytetrack_best`, `smooth_det020_loose`, `realtime_quality_delayed`, `bytetrack_raw`.
- `-p`, `--profile`: path profile name, khong phai tracking mode.
- `--path-config`: file config duong dan video/weight/mask/output.
- `--list-eval-configs`: liet ke cac eval config.

Moi option khong nam trong wrapper se duoc forward xuong tracking CLI, vi du:

- `--weights`: detector weight.
- `--output-dir`: thu muc output tracking.
- `--det-conf`: nguong detection confidence.
- `--max-raw-detections`: so detection toi da moi frame.
- `--detect-every-n-frames`: skip frame cho realtime/fast mode.
- `--profile-override KEY=VALUE`: override truc tiep field cua `TrackingConfig`.
- `--no-emit-hidden-tracks`: khi export CVAT, khong gan label Hidden cho track noi suy.

Tracking 1 video voi best hybrid:

```cmd
C:\Users\ironh\anaconda3\envs\pig_project\python.exe scripts\track_videos.py ^
  -v "Pigs291119_000263_30fps" ^
  --mode hybrid_bytetrack ^
  --eval-config hybrid_bytetrack_best
```

## `evaluate_tracking.py`

Dung khi can metric voi GT XML. Script nay co the tu goi tracker neu prediction thieu.

Option chinh:

- `-v`, `--video`: chon video.
- `-a`, `--all-videos`: chay tat ca video co GT.
- `--mode`: runtime tracking engine: `bytetrack_raw`, `realtime`, hoac `hybrid_bytetrack`.
- `--eval-config`: bo override `TrackingConfig` da dat ten. Co the lap lai nhieu lan de chay nhieu config trong cung command.
- `--rule-combo`: chi chay rule combo cu the, vi du `iou0_area0_condarea0_merge0`.
- `--benchmark-rules`: mo rong rule combos cho selected run.
- `--benchmark-detectors`: mo rong detector configs cho selected run.
- `--benchmark-compatible`: chay matrix cu day du detector/rule de tai lap benchmark output.
- `--smooth` / `--no-smooth`: bat/tat offline smoothing/refinement cho hybrid.
- `--skip-missing-gt`: bo qua video khong co GT XML.
- `--fail-missing-gt`: gap video thieu GT thi fail.
- `--prediction-root`: doi thu muc prediction.
- `--output-root`: doi thu muc eval output.
- `--no-run-missing-tracker`: chi eval prediction co san, khong track neu thieu.
- `--force-track`: ep track lai.
- `--profile-override KEY=VALUE`: override config co anh huong ket qua.
- `--list-eval-configs`: liet ke eval configs.

Eval 1 video voi best hybrid:

```cmd
C:\Users\ironh\anaconda3\envs\pig_project\python.exe scripts\evaluate_tracking.py ^
  -v "Pigs291119_000263_30fps" ^
  --mode hybrid_bytetrack ^
  --eval-config hybrid_bytetrack_best ^
  --rule-combo iou0_area0_condarea0_merge0
```

Full rule benchmark cho mot config:

```cmd
C:\Users\ironh\anaconda3\envs\pig_project\python.exe scripts\evaluate_tracking.py ^
  -a ^
  --mode hybrid_bytetrack ^
  --eval-config hybrid_bytetrack_best ^
  --benchmark-rules
```

## `optimize_tracking_metrics.py`

Dung khi muon thu nhieu preset va xep hang theo metric.

Option chinh:

- `-v`, `--video`: chon video target.
- `-a`, `--all-videos`: chay tat ca video co GT.
- `--mode`, `--tracking-mode`: runtime tracking mode de tao prediction.
- `--scope`: muc do tim kiem `quick`, `balanced`, `full`, `detector_probe`.
- `--preset`: chi chay preset cu the, co the lap lai.
- `--rule-scope`: `baseline` cho `iou0_area0_condarea0_merge0`, `iou` cho `iou1_area0_condarea0_merge0`.
- `--smooth-mode`: `nosmooth`, `smooth`, hoac `both`.
- `--smooth` / `--no-smooth`: ep smoothing on/off cho tat ca run.
- `--rank-by`: cot uu tien xep hang.
- `--target-video`: video can hien thi rieng trong diagnostics.
- `--run-name`: ten folder output on dinh de resume.
- `--resume` / `--no-resume`: tiep tuc hoac chay lai tu dau.
- `--fail-fast`: dung ngay khi mot run loi.
- `--dry-run`: in ke hoach, khong chay tracking.
- `--list-presets`: liet ke preset optimizer.
- `--top-k`: so dong top ranking can hien thi.

Optimizer nhanh tren 1 video:

```cmd
C:\Users\ironh\anaconda3\envs\pig_project\python.exe scripts\optimize_tracking_metrics.py ^
  -v "Pigs291119_000263_30fps" ^
  --mode hybrid_bytetrack ^
  --scope quick ^
  --rule-scope baseline
```

## Notes

- `--output-root` va `--prediction-root` chi doi thu muc ghi output, khong tu lam doi logic tracking.
- `--profile-override KEY=VALUE` moi la nhom option truc tiep lam thay doi ket qua.
- `--profile` trong `track_videos.py` / `evaluate_tracking.py` la path profile, khong phai tracking mode.
- Khong dung `--mode bytetrack`; dung ro `bytetrack_raw` hoac `hybrid_bytetrack`.
- De so sanh khoa hoc, nen ghi lai command day du, output folder, mode, eval-config, rule-combo, detector weight va commit.
