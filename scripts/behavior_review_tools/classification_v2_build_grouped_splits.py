from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_MANIFEST = Path("outputs/classification_v2/sequence_features_reviewed/sequence_window_manifest.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/classification_v2/train_ready_windows")
LABEL_COL = "behavior_window_label"
VALID_COL = "window_valid_for_main_train"
SPLITS = ["train", "val", "test"]


def _json_default(value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    return value


def _parse_ratios(text: str) -> dict[str, float]:
    parts = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(parts) != 3:
        raise SystemExit("--ratios must contain exactly train,val,test values, e.g. 0.70,0.15,0.15")
    total = sum(parts)
    if total <= 0:
        raise SystemExit("--ratios sum must be positive")
    return {name: value / total for name, value in zip(SPLITS, parts)}


def _stable_group_key(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in ["source_type", "dataset_id", "video_key"] if c in df.columns]
    if not cols:
        raise SystemExit("Manifest has no source/video grouping columns.")
    return df[cols].fillna("").astype(str).agg("|".join, axis=1)


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def _label_counts(rows: pd.DataFrame) -> Counter[str]:
    return Counter(rows[LABEL_COL].fillna("__missing__").astype(str).tolist())


def _counter_add(a: Counter[str], b: Counter[str]) -> Counter[str]:
    out = Counter(a)
    out.update(b)
    return out


def _projected_label_score(
    split_name: str,
    candidate_counts: Counter[str],
    target_labels: dict[str, dict[str, float]],
) -> float:
    label_score = 0.0
    for label, target in target_labels[split_name].items():
        candidate = candidate_counts.get(label, 0)
        label_score += ((candidate - target) / max(1.0, target)) ** 2
    return label_score


def assign_grouped_splits(df: pd.DataFrame, ratios: dict[str, float]) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = ["window_id", LABEL_COL, VALID_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Manifest missing required split columns: {missing}")

    work = df.copy()
    work["split_group_key"] = _stable_group_key(work)
    work["_valid_for_split_balance"] = _as_bool(work[VALID_COL])
    balance_rows = work[work["_valid_for_split_balance"]].copy()

    group_stats: list[dict[str, Any]] = []
    for group_key, group in work.groupby("split_group_key", sort=True):
        balance_group = balance_rows[balance_rows["split_group_key"].eq(group_key)]
        counts = _label_counts(balance_group) if not balance_group.empty else Counter()
        group_stats.append(
            {
                "split_group_key": group_key,
                "rows": int(len(group)),
                "balance_rows": int(len(balance_group)),
                "label_counts": counts,
            }
        )

    total_balance_rows = sum(g["balance_rows"] for g in group_stats)
    total_label_counts = Counter()
    for g in group_stats:
        total_label_counts.update(g["label_counts"])
    for g in group_stats:
        rarity_score = 0.0
        for label, count in g["label_counts"].items():
            rarity_score += count / max(1, total_label_counts[label])
        g["rarity_score"] = rarity_score

    target_rows = {split: total_balance_rows * ratio for split, ratio in ratios.items()}
    target_labels = {
        split: {label: count * ratios[split] for label, count in total_label_counts.items()} for split in SPLITS
    }

    assigned_rows = {split: 0 for split in SPLITS}
    assigned_labels = {split: Counter() for split in SPLITS}
    group_to_split: dict[str, str] = {}

    # Rare-label-heavy groups first gives val/test a chance to receive hard
    # classes under strict video/session grouping. Fill-ratio assignment below
    # still keeps the requested row ratios close.
    ordered_groups = sorted(
        group_stats,
        key=lambda g: (-g["rarity_score"], -g["balance_rows"], -g["rows"], g["split_group_key"]),
    )
    for group in ordered_groups:
        best_split = min(
            SPLITS,
            key=lambda split: (
                assigned_rows[split] / max(1.0, target_rows[split]),
                max(0.0, (assigned_rows[split] + group["balance_rows"] - target_rows[split]) / max(1.0, target_rows[split])),
                _projected_label_score(
                    split,
                    _counter_add(assigned_labels[split], group["label_counts"]),
                    target_labels,
                ),
                split,
            ),
        )
        group_to_split[group["split_group_key"]] = best_split
        assigned_rows[best_split] += group["balance_rows"]
        assigned_labels[best_split].update(group["label_counts"])

    out = work[["window_id", "split_group_key"]].copy()
    out["split"] = out["split_group_key"].map(group_to_split)
    passthrough = [
        "source_type",
        "dataset_id",
        "video_key",
        "object_track_key",
        "pig_id",
        "track_id",
        "window_length_frames",
        "window_start_frame",
        "window_end_frame",
        LABEL_COL,
        "sequence_label_status",
        VALID_COL,
        "window_sample_weight",
    ]
    passthrough = [c for c in passthrough if c in work.columns]
    out = out.merge(work[["window_id", *passthrough]], on="window_id", how="left", validate="one_to_one")

    group_assignment = pd.DataFrame(
        [{"split_group_key": key, "split": split} for key, split in sorted(group_to_split.items())]
    )
    leakage_groups = (
        group_assignment.groupby("split_group_key")["split"].nunique().loc[lambda s: s > 1].index.astype(str).tolist()
    )

    audit = {
        "rows": int(len(out)),
        "groups": int(len(group_assignment)),
        "ratios": ratios,
        "target_train_balance_rows": target_rows,
        "split_rows": out["split"].value_counts(dropna=False).to_dict(),
        "split_valid_rows": out.loc[_as_bool(out[VALID_COL]), "split"].value_counts(dropna=False).to_dict(),
        "split_label_counts": {
            split: out.loc[out["split"].eq(split), LABEL_COL].value_counts(dropna=False).to_dict() for split in SPLITS
        },
        "split_valid_label_counts": {
            split: out.loc[out["split"].eq(split) & _as_bool(out[VALID_COL]), LABEL_COL]
            .value_counts(dropna=False)
            .to_dict()
            for split in SPLITS
        },
        "group_counts_by_split": group_assignment["split"].value_counts(dropna=False).to_dict(),
        "leakage_group_count": len(leakage_groups),
        "leakage_groups": leakage_groups[:50],
    }
    return out, audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-safe grouped splits for reviewed sequence windows.")
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ratios", default="0.70,0.15,0.15", help="train,val,test ratios")
    args = parser.parse_args()

    ratios = _parse_ratios(args.ratios)
    df = pd.read_csv(args.manifest_csv, low_memory=False)
    split_manifest, audit = assign_grouped_splits(df, ratios)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_path = args.output_dir / "split_manifest.csv"
    audit_path = args.output_dir / "split_audit.json"
    split_manifest.to_csv(split_path, index=False)
    audit.update(
        {
            "manifest_csv": str(args.manifest_csv),
            "split_manifest_csv": str(split_path),
            "split_audit_json": str(audit_path),
        }
    )
    audit_path.write_text(json.dumps(audit, indent=2, default=_json_default), encoding="utf-8")

    print(json.dumps(audit, indent=2, default=_json_default))
    if audit["leakage_group_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
