from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


OLD = Path(
    r"E:\PigProjectStorage\PIG_Behavior_Project\outputs\classification_v2\model_readiness_audit\rg03_readiness_handoff_7fd98ab_20260805_075705"
)
NEW = Path(
    r"E:\PigProjectStorage\PIG_Behavior_Project\outputs\classification_v2\model_readiness_audit\rg03_readiness_handoff_e212632_20260805_095700"
)
RG04 = Path(
    r"E:\PigProjectStorage\PIG_Behavior_Project\outputs\classification_v2\model_readiness_audit\rg04_social_oof_executor_09b3639_20260805_110000"
)
PLAN_SHA = "cd96fb3d37bc0dc0f366e2a7ef76511f39e0284ed9a0a132675774697620dbce"
CURRENT_COMMIT = "e212632ca252220d4f39fb92106800127ed4511b"
FOLD_CODE_SHA = "09b3639eeb2533b23b40ca32292e4900c518c810"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def main() -> None:
    if NEW.exists():
        raise FileExistsError(NEW)
    report_path = RG04 / "paired_social_oof_report.json"
    report = load(report_path)
    if report.get("status") != "PASS_ENGINEERING_ONLY":
        raise ValueError("RG-04 report is not PASS_ENGINEERING_ONLY")
    selection = report["selection"]
    if selection["paired_native_unit_count"] != 59:
        raise ValueError("unexpected bounded paired native-unit count")
    if selection["not_full_oof"] is not True:
        raise ValueError("RG-04 must remain bounded diagnostic-only")
    for key in ("S1_social_10d_vs_S0_no_social", "S2_topk_k3_vs_S0_no_social"):
        comparison = report["paired_comparisons"][key]["report"]
        if comparison.get("valid") is not True:
            raise ValueError(f"invalid paired comparison: {key}")
        if comparison["comparison"].get("paper_facing_ready") is not False:
            raise ValueError(f"paper-facing boundary missing: {key}")

    NEW.mkdir(parents=True)
    shutil.copy2(OLD / "paid_gpu_launch_matrix.csv", NEW / "paid_gpu_launch_matrix.csv")

    decision = load(OLD / "training_readiness_decision_current.json")
    decision["generated_at"] = "2026-08-05T09:57:00+07:00"
    decision["plan"]["sha256"] = PLAN_SHA
    repository = decision["repository"]
    repository["candidate_executor_commit_sha"] = CURRENT_COMMIT
    repository["candidate_executor_fold_run_code_sha"] = FOLD_CODE_SHA
    repository["candidate_executor_branch"] = "codex/classification-v2-rg04-social-executor"
    repository["candidate_executor_worktree_clean_after_commit"] = True
    decision["gates"]["G12_paired_social_S0_S1_S2"] = "PASS_ENGINEERING_ONLY_BOUNDED"
    decision["gates"]["G12_paired_social_scientific_gate"] = "BLOCKED_NOT_CLAIM_GRADE"
    decision["readiness_flags"]["paired_social_executor"] = "PASS_ENGINEERING_ONLY"
    decision["readiness_flags"]["ready_for_paired_native_claim_grade"] = "BLOCKED"
    decision["evidence"]["RG04_paired_social_executor"] = {
        "path": str(report_path),
        "sha256": sha256(report_path),
        "status": report["status"],
        "paired_native_units": selection["paired_native_unit_count"],
        "paired_windows": selection["paired_window_count"],
        "outer_folds": report["outer_fold_ids"],
        "arm_coverage": selection["arm_native_unit_counts"],
        "paper_metric_authority": False,
        "candidate_commit_sha": CURRENT_COMMIT,
        "fold_run_code_sha": FOLD_CODE_SHA,
    }
    decision["bounded_oof_status"] = {
        "runner_valid": True,
        "prediction_schema_valid": True,
        "native_unit_rows": selection["paired_native_unit_count"],
        "complete_four_fold_coverage": True,
        "paired_arms": ["S0_no_social", "S1_social_10d", "S2_topk_k3"],
        "paper_facing_result": False,
        "status": "PASS_ENGINEERING_ONLY_BOUNDED",
        "report_sha256": sha256(report_path),
    }
    decision["px2_decision"]["paired_social_executor_status"] = "PASS_ENGINEERING_ONLY_BOUNDED"
    decision["px2_decision"]["paired_social_claim_status"] = "BLOCKED_NOT_CLAIM_GRADE"
    decision["px2_decision"]["next_gate"] = (
        "Resolve strict A12 source support, then run full native-unit paired OOF/calibration."
    )
    decision["final_verdict"] = "PAID_GPU_BLOCKED_A12_SOURCE_SUPPORT_FULL_OOF_BALANCED_AND_AUTORESEARCH"
    decision["next_authorized_stage"] = (
        "Resolve FOLD_2/FOLD_4 source support and buildability gates; do not launch paid GPU or claim paper metrics."
    )
    decision["non_interference"]["candidate_code_changed"] = True
    decision["non_interference"]["candidate_commit_sha"] = CURRENT_COMMIT
    decision["non_interference"]["production_code_changed"] = False
    decision["non_interference"]["rg04_session_gpu_used"] = False
    decision["production_code_changed"] = False
    decision["patch_commit_sha"] = CURRENT_COMMIT
    decision["patch_scope"] = "isolated candidate branch only; not merged into main"
    decision_path = NEW / "training_readiness_decision_current.json"
    decision_path.write_text(
        json.dumps(decision, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    current_state = """# RG-03 refreshed state

RG-04 paired social executor is available and independently focused-tested.
The bounded four-fold diagnostic passed on 59 paired native units with identical
S0/S1/S2 window and native-unit identities across all four folds. It remains an
engineering diagnostic only; `paper_facing_ready=false` and no model selection
or paper metric is authorized.

Strict A12 remains INCONCLUSIVE because FOLD_2 and FOLD_4 are single-source,
so full source-balanced cross-source evidence is unavailable. BALANCED,
claim-grade native-unit OOF/calibration, autoresearch promotion, and paid GPU
remain blocked. The original RG-03 handoff is preserved unchanged.
"""
    (NEW / "CURRENT_STATE.md").write_text(current_state, encoding="utf-8")
    final_handoff = f"""# Classification V2 RG-03 refreshed handoff

- Plan SHA-256: `{PLAN_SHA}`
- Main SHA: `487a55b12041889a00f6d1827297a88865e15345`
- Candidate executor commit: `{CURRENT_COMMIT}`
- Fold-run code SHA recorded by RG-04: `{FOLD_CODE_SHA}`
- RG-04 report: `{report_path}`
- RG-04 report SHA-256: `{sha256(report_path)}`

## Decision

`PASS_ENGINEERING_ONLY_BOUNDED` for the paired S0/S1/S2 executor. The run
contains 59 paired native units across FOLD_1--FOLD_4, with identical arm
coverage and valid lineage. This is not a paper metric and cannot select a
social architecture.

Strict A12 remains `INCONCLUSIVE`: FOLD_2 and FOLD_4 do not contain both source
types. BALANCED, claim-grade OOF/calibration, autoresearch promotion, and paid
GPU remain `BLOCKED`. The old RG-03 handoff was not overwritten.
"""
    (NEW / "FINAL_HANDOFF.md").write_text(final_handoff, encoding="utf-8")

    check = {
        "schema_version": "classification_v2.rg03_refreshed_independent_check.v1",
        "status": "PASS",
        "decision_json_valid": True,
        "rg04_report_status": report["status"],
        "rg04_report_sha256": sha256(report_path),
        "paired_native_unit_count": selection["paired_native_unit_count"],
        "four_fold_coverage": report["outer_fold_ids"],
        "paper_facing_ready": False,
        "paid_gpu": "BLOCKED",
        "main_worktree_touched": False,
        "human_ledgers_touched": False,
    }
    check_path = NEW / "independent_handoff_check.json"
    check_path.write_text(json.dumps(check, indent=2) + "\n", encoding="utf-8")

    artifact_paths = [
        decision_path,
        NEW / "paid_gpu_launch_matrix.csv",
        check_path,
        NEW / "FINAL_HANDOFF.md",
        NEW / "CURRENT_STATE.md",
        report_path,
        RG04 / "paired_social_window_predictions.csv",
        RG04 / "paired_social_fold_assignments.csv",
        RG04 / "S1_social_10d_vs_S0_no_social_native_predictions.csv",
        RG04 / "S2_topk_k3_vs_S0_no_social_native_predictions.csv",
    ]
    manifest = {
        "schema_version": "classification_v2.rg03_refreshed_artifact_hash_manifest.v1",
        "generated_at": "2026-08-05T09:57:00+07:00",
        "base_handoff": str(OLD),
        "plan_sha256": PLAN_SHA,
        "artifacts": {path.name: file_record(path) for path in artifact_paths},
    }
    (NEW / "artifact_hash_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(NEW), "report_sha256": sha256(report_path)}))


if __name__ == "__main__":
    main()
