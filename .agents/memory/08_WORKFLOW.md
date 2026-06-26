# Workflow

## Before every coding task

1. Read root `AGENTS.md`.
2. Read `.agents/memory/01_PROJECT_MEMORY_SHORT.md`.
3. Read `.agents/memory/02_CURRENT_DECISION.md`.
4. Read `.agents/memory/03_PROJECT_RULES.md`.
5. State which memory files were read.

## Audit task

If the user asks to audit/check/diff:

- Do not modify code.
- Do not run tracking/evaluation/inference.
- Use read-only commands only.
- Report findings with file/function/behavior/risk.

## Patch task

If the user asks to patch:

- Modify only requested scope.
- Prefer one small patch at a time.
- State exact files changed.
- State behavior changed.
- State behavior intentionally not changed.
- Do not run long benchmarks unless requested.

## Verification task

When user allows verification, run in this order:

1. Static/syntax/import check.
2. Single video `Pigs291119_000263_30fps`.
3. Single video `Pigs291119_000302_30fps`.
4. 3-video common set.
5. 7-video full set.

## Preserved legacy workflow notes from previous `.agents/WORKFLOW.md`

- ROI definitions live in `data/annotations/roi/ROI_annotations.coco.json` with related scene background and mask assets.
- CVAT XML annotations are processed into classification datasets and feature tables.
- Detection and tracking previously centered around `pig-track-for-annotation` workflows with GPU fallback to CPU.
- RGB-D occlusion handling used depth calibration files such as `depth_scale.npy`, `inverse_intrinsic.npy`, and `rot.npy`.
- Behavior training, export, inference, and FastAPI dashboard flows remain documented in `.agents/WORKFLOW.md`.
- CI/quality gates previously documented there remain preserved as legacy workflow guidance.
