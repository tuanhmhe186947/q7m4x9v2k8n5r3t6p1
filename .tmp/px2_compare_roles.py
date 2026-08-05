from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


OLD = Path(
    r"outputs/classification_v2/model_readiness_audit/"
    r"pre_gpu_autoresearch_q2_59684f7_20260805_011800/"
    r"four_fold_roles_47103f6_20260804_153800/native_unit_outer_roles.csv"
)
NEW = Path(
    r"E:/PigProjectStorage/PIG_Behavior_Project/.codex_tmp/worktrees/"
    r"classification_v2_pre_gpu_autoresearch_q2/outputs/classification_v2/"
    r"model_readiness_audit/px2_rebind_59684f7_20260805_020000/"
    r"four_fold_roles_inner4/native_unit_outer_roles.csv"
)
WEIGHT_OLD = Path(
    r"outputs/classification_v2/model_readiness_audit/"
    r"pre_gpu_autoresearch_q2_59684f7_20260805_011800/"
    r"fold_event_weights_47103f6_20260804_154500/fold_event_weight_manifest.csv"
)
WEIGHT_NEW = Path(
    r"E:/PigProjectStorage/PIG_Behavior_Project/.codex_tmp/worktrees/"
    r"classification_v2_pre_gpu_autoresearch_q2/outputs/classification_v2/"
    r"model_readiness_audit/px2_rebind_59684f7_20260805_020000/"
    r"fold_event_weights_inner4/fold_event_weight_manifest.csv"
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_roles(path: Path) -> pd.DataFrame:
    columns = [
        "outer_fold_id",
        "temporal_unit_key",
        "role",
        "inner_fold_id",
        "validation_inner_fold_id",
        "behavior_label",
        "primary_t6_model_eligible",
        "primary_t6_eligibility_reason",
    ]
    return pd.read_csv(path, usecols=columns, dtype=str)


def load_weights(path: Path) -> pd.DataFrame:
    columns = [
        "outer_fold_id",
        "window_id",
        "role",
        "fold_event_sample_weight",
        "fold_event_class_sample_weight",
    ]
    return pd.read_csv(path, usecols=columns, dtype=str)


def main() -> None:
    old = load_roles(OLD)
    new = load_roles(NEW)
    key = ["outer_fold_id", "temporal_unit_key"]
    joined = old.merge(
        new,
        on=key,
        how="outer",
        suffixes=("_old", "_new"),
        indicator=True,
    )
    role_columns = [
        "role",
        "inner_fold_id",
        "validation_inner_fold_id",
        "behavior_label",
        "primary_t6_model_eligible",
        "primary_t6_eligibility_reason",
    ]
    role_diffs = {
        column: int(
            joined[f"{column}_old"].fillna("")
            .ne(joined[f"{column}_new"].fillna(""))
            .sum()
        )
        for column in role_columns
    }
    result = {
        "old": {"path": str(OLD), "rows": len(old), "sha256": sha(OLD)},
        "new": {"path": str(NEW), "rows": len(new), "sha256": sha(NEW)},
        "key_counts": {
            "both": int((joined["_merge"] == "both").sum()),
            "old_only": int((joined["_merge"] == "left_only").sum()),
            "new_only": int((joined["_merge"] == "right_only").sum()),
            "old_duplicate_keys": int(old.duplicated(key).sum()),
            "new_duplicate_keys": int(new.duplicated(key).sum()),
        },
        "role_differences": role_diffs,
        "old_role_counts": old["role"].value_counts().to_dict(),
        "new_role_counts": new["role"].value_counts().to_dict(),
        "ordered_row_equality": bool(old.equals(new)),
    }
    old_weight = load_weights(WEIGHT_OLD)
    new_weight = load_weights(WEIGHT_NEW)
    weight_key = ["outer_fold_id", "window_id"]
    weight_join = old_weight.merge(
        new_weight,
        on=weight_key,
        how="outer",
        suffixes=("_old", "_new"),
        indicator=True,
    )
    result["weights"] = {
        "old": {"rows": len(old_weight), "sha256": sha(WEIGHT_OLD)},
        "new": {"rows": len(new_weight), "sha256": sha(WEIGHT_NEW)},
        "key_counts": {
            "both": int((weight_join["_merge"] == "both").sum()),
            "old_only": int((weight_join["_merge"] == "left_only").sum()),
            "new_only": int((weight_join["_merge"] == "right_only").sum()),
            "old_duplicate_keys": int(old_weight.duplicated(weight_key).sum()),
            "new_duplicate_keys": int(new_weight.duplicated(weight_key).sum()),
        },
        "role_diff": int(
            weight_join["role_old"].fillna("").ne(weight_join["role_new"].fillna("")).sum()
        ),
        "event_weight_diff": int(
            weight_join["fold_event_sample_weight_old"].fillna("").ne(
                weight_join["fold_event_sample_weight_new"].fillna("")
            ).sum()
        ),
        "event_class_weight_diff": int(
            weight_join["fold_event_class_sample_weight_old"].fillna("").ne(
                weight_join["fold_event_class_sample_weight_new"].fillna("")
            ).sum()
        ),
    }
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
