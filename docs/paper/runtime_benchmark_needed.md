# Runtime and Computational Resource Benchmark Status

## 1. Tracking Runtime Evidence: COMPLETE (PASS)
- Full physical telemetry for all three tracking systems (`bytetrack_raw`,
  `realtime_fast`, `hybrid_bytetrack`) is available from `outputs/tracking_fresh/20260812_
  fresh_main_01/primary/eval/*/iou0_area0_condarea0_merge0/tracking_runtime_telemetry.csv`
  .
- Metrics include mean frame time, 50th/95th percentiles, effective loop FPS, peak
  allocated CUDA memory, and host process RSS across 13 full development videos.

## 2. Behavior Model Runtime Evidence: PARTIAL (NEEDS_GPU_BENCHMARK for Dedicated Latency Profiles)
- Parameter counts, trainable parameters, and checkpoint file sizes are verified directly
  from physical PyTorch checkpoints.
- If dedicated isolated milliseconds-per-window GPU latency benchmarks (batch size 1 vs
  32, cold vs warm cache) are required for the final manuscript camera-ready table, they
  must be executed on an authorized cloud GPU Studio:
  - **Command**: `python scripts/classification_v2/benchmark_inference_latency.py --models
ModelA ModelJ EnsembleEJ --batch-sizes 1 8 32 --device cuda:0`
  - **Status**: Execution deferred; Studio was not started during this task in adherence
to project rules.
