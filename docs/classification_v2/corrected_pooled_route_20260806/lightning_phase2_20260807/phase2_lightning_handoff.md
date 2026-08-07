# Phase 2 Lightning CPU preflight checkpoint

Local prerequisites pass: VS Code, OpenSSH, Git, and the official Remote SSH
extension are available. No non-secret Lightning SSH target or local Lightning
configuration exists. No Studio, CPU resource, transfer, GPU allocation, or E0
execution has occurred.

The former executable blocker is resolved locally. The canonical authority is
`next_phase_20260806_r2/e0_execution_authority.json` (SHA-256
`20914be7b71f9e43dca3293e9d3a50b19da590065a6017e3d1b37b03c58de3c5`),
with the explicit `classification_v2_run_e0_inner_only.py` wrapper. It fixes
B3/T6/FOLD_3/seed 20260804 with actor RGB, geometry 6D, and motion 12D; it
disables ROI, social, interaction context, visual context, history, posture,
availability controls, and quality controls. The resolved command does not use
`--variants full`, and the wrapper blocks the held-out FOLD_3 test role before
data use.

The remote transfer inventory now contains only E0 code, authority, environment
lock, reviewed T6 metadata, actor RGB cache, B3 geometry/motion arrays, FOLD_3
role map, and the train-only event-weight authority. Its estimated transfer is
below 15 GiB; H5, posture, raw video, old outputs, and unrelated caches remain
excluded.

The E0 environment lock is the canonical staged CRLF artifact at
`next_phase_20260806_r2/e0_environment/uv.lock` (SHA-256
`6b783d5296094e0be94b0e553e3c83376a462eec3278285b076b35761bc103ca`).
Transfer it verbatim as the package-root `uv.lock`; the repository-root LF lock
is development-only and is not a remote E0 substitute.

After pre-GPU main canonicalization, resume at the existing Lightning UI/SSH
checkpoint. Do not create a Studio in this phase and do not allocate a GPU or
run E0.
