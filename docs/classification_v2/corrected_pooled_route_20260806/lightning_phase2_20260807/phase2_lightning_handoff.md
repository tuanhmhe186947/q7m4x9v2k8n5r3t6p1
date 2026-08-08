# Phase 2 Lightning and E0 handoff

## Release anchor

Use the immutable main ref `classification-v2-pre-gpu-authority-20260808-r3`.
Resolve it with `git rev-parse refs/tags/classification-v2-pre-gpu-authority-20260808-r3^{commit}`
and work from that detached checkout only. No temporary worktree SHA is current.

## Ready-to-transfer package

The package contract is `pre_gpu_e0_transfer_package.json`, backed by
`remote_e0_transfer_inventory.json`. It contains only the canonical wrapper,
E0 authority/handoff, Python source, `pyproject.toml`, staged lock, the
inventory-bound reconciliation control, and the hash-bound external data
selection. Its expected transfer is below 15 GiB;
H5, posture, raw video, historical worktrees, unrelated caches, and secrets
are excluded.

Its r3 descriptor SHA-256 is
`c2529f23c4b0f16fd2aef0d3ceadcf67e278b8924963d4765a25e8ad5aa6e567`.

The required remote input root is
`/teamspace/studios/this_studio/pig_e0_r3/inputs`, bound by
`e0_remote_input_bindings.json`. Its spatial projection includes only the
existing hash-bound B3 geometry and motion shards; it does not add ROI, social,
H5, or posture inputs.

Copy `next_phase_20260806_r2/e0_environment/uv.lock` verbatim as package-root
`uv.lock` (SHA-256 `6b783d5296094e0be94b0e553e3c83376a462eec3278285b076b35761bc103ca`).
The repository-root lock is development-only and cannot substitute for it.

## Morning procedure after explicit Lightning access

1. Check out the tag, stage exactly the transfer inventory, and verify every
   declared file/hash before data use.
2. From the staged package root, realize the frozen E0 environment:

   ```bash
   uv sync --frozen --python 3.11 --extra pt
   ```

3. Run CPU/config-only preflight; it must report the canonical B3/T6/FOLD_3
   authority and block outer-test access:

   ```bash
   E0_ROUTE=docs/classification_v2/corrected_pooled_route_20260806
   E0_AUTHORITY="$E0_ROUTE/next_phase_20260806_r2/e0_execution_authority.json"
   E0_BINDINGS="$E0_ROUTE/lightning_phase2_20260807/e0_remote_input_bindings.json"
   uv run --frozen python \
     scripts/classification_v2/04_baselines_smokes/classification_v2_run_e0_inner_only.py \
     --authority "$E0_AUTHORITY" \
     --mode inspect --report "$E0_OUT/inspection.json"
   uv run --frozen python \
     scripts/classification_v2/04_baselines_smokes/classification_v2_run_e0_inner_only.py \
     --authority "$E0_AUTHORITY" \
     --mode assert-outer-blocked
   ```

4. Only after separate paid-execution authorization, use one L4 24 GB and the
   canonical command below. It is a 16-step engineering validation, not model
   selection:

   ```bash
   uv run --frozen python \
     scripts/classification_v2/04_baselines_smokes/classification_v2_run_e0_inner_only.py \
     --authority "$E0_AUTHORITY" \
     --data-bindings "$E0_BINDINGS" --output-dir "$E0_OUT" --device cuda \
     --mode train --execution-authorization REQUIRED
   ```

Stop immediately for a tag/hash mismatch, a lock mutation request, an outer
test path, missing FOLD_3 role binding, non-finite loss, failed checkpoint or
prediction export, or missing explicit paid-execution authorization. Preserve
the final and step checkpoints, `predictions.csv`, `download_hash_manifest.json`,
and all registered run/inspection reports. E0 passes only when its declared
16-step endpoint, checkpoint/resume, prediction export, hash audit, and
outer-test block all pass; it never selects an S1/final behavior model.
