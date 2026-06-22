$video = "data\videos\Pigs291119_000226_30fps.mp4"
$outRoot = "outputs\id_tracking\no_gt_assoc_eval"

$presets = @(
  @{name="base"; det=0.25; jump=0.08; stationary=0.045; gain=0.015; iom=0.10; detection_iom=0.30; growth=1.50},
  @{name="strict_assoc_1"; det=0.25; jump=0.06; stationary=0.040; gain=0.020; iom=0.10; detection_iom=0.30; growth=1.50},
  @{name="strict_assoc_2"; det=0.25; jump=0.05; stationary=0.035; gain=0.020; iom=0.15; detection_iom=0.30; growth=1.40},
  @{name="strict_assoc_3"; det=0.30; jump=0.05; stationary=0.035; gain=0.020; iom=0.15; detection_iom=0.30; growth=1.40},
  @{name="conservative"; det=0.30; jump=0.05; stationary=0.030; gain=0.030; iom=0.15; detection_iom=0.40; growth=1.30},
  @{name="soft_detection"; det=0.25; jump=0.06; stationary=0.035; gain=0.020; iom=0.15; detection_iom=0.20; growth=1.40}
)

foreach ($p in $presets) {
  Write-Host "Running preset: $($p.name)"
  .\.venv\Scripts\python.exe src\pig_behavior\data_preparation\tracking_annotation.py `
    --video $video `
    --output-dir "$outRoot\$($p.name)" `
    --det-conf $p.det `
    --low-conf-max-center-jump $p.jump `
    --bbox-sanity-max-center-jump $p.jump `
    --occlusion-stationary-max-center-jump $p.stationary `
    --identity-swap-min-gain $p.gain `
    --identity-swap-iom-threshold $p.iom `
    --occlusion-detection-iom-threshold $p.detection_iom `
    --merged-box-growth-ratio $p.growth `
    --use-conditional-area-occlusion-freeze `
    --use-merged-box-split
}
