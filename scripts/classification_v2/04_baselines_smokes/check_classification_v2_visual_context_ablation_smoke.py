from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Check paired visual-context/no-visual OOF wiring smoke artifacts.")
    parser.add_argument(
        "--full-audit",
        type=Path,
        default=Path("outputs/classification_v2/model_smoke/visual_context_packed_oof/full_multimodal_oof_audit.json"),
    )
    parser.add_argument(
        "--no-visual-audit",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_smoke/no_visual_context_packed_oof/full_multimodal_oof_audit.json"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/classification_v2/model_design/visual_context_ablation_smoke_audit.json"),
    )
    args = parser.parse_args()
    full = _read_json(args.full_audit)
    no_visual = _read_json(args.no_visual_audit)
    errors: list[str] = []
    full_fold = _one_fold(full, "full", errors)
    no_visual_fold = _one_fold(no_visual, "no_visual", errors)
    _expect(full.get("config", {}).get("ablation_variant") == "full", "full_variant_mismatch", errors)
    _expect(
        no_visual.get("config", {}).get("ablation_variant") == "no_visual_context",
        "no_visual_variant_mismatch",
        errors,
    )
    _expect(full_fold.get("instantiated_branches", {}).get("visual_context") is True, "full_visual_absent", errors)
    _expect(
        no_visual_fold.get("instantiated_branches", {}).get("visual_context") is False,
        "no_visual_branch_still_instantiated",
        errors,
    )
    for key in ["train_indices_sha256", "eval_indices_sha256"]:
        _expect(full_fold.get(key) == no_visual_fold.get(key), f"paired_index_hash_mismatch={key}", errors)
    visual_load = full.get("visual_context_load_audit", {})
    _expect(int(visual_load.get("packed_cache_hits", 0)) > 0, "full_visual_has_no_packed_hits", errors)
    _expect(int(visual_load.get("packed_cache_misses", -1)) == 0, "full_visual_packed_misses", errors)
    _expect(int(visual_load.get("individual_cache_loads", -1)) == 0, "full_visual_individual_fallback", errors)
    for name, audit in [("full", full), ("no_visual", no_visual)]:
        _expect(audit.get("prediction_schema_valid") is True, f"{name}_prediction_schema_invalid", errors)
        _expect(audit.get("paper_facing_result") is False, f"{name}_smoke_marked_paper_facing", errors)
        _expect(audit.get("run_mode") == "pilot", f"{name}_run_mode_not_pilot", errors)
    result: dict[str, Any] = {
        "schema_version": "classification_v2_visual_context_ablation_smoke_audit_v1",
        "full_audit": str(args.full_audit),
        "no_visual_audit": str(args.no_visual_audit),
        "paired_train_indices_sha256": full_fold.get("train_indices_sha256"),
        "paired_eval_indices_sha256": full_fold.get("eval_indices_sha256"),
        "full_trainable_parameter_count": full_fold.get("trainable_parameter_count"),
        "no_visual_trainable_parameter_count": no_visual_fold.get("trainable_parameter_count"),
        "full_visual_packed_hits": visual_load.get("packed_cache_hits"),
        "metric_interpretation": "wiring_smoke_only_not_statistical_ablation_evidence",
        "errors": errors,
        "valid": not errors,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _one_fold(audit: dict[str, Any], name: str, errors: list[str]) -> dict[str, Any]:
    folds = audit.get("fold_audits", [])
    if len(folds) != 1:
        errors.append(f"{name}_expected_one_fold={len(folds)}")
        return {}
    return folds[0]


def _expect(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


if __name__ == "__main__":
    main()
