from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pandas as pd

VALID_BEHAVIORS = {
    "drink",
    "eat",
    "fight",
    "social-nose",
    "explore",
    "lying",
    "stand",
    "move",
    "sitting",
    "playwithtoy",
}

ROOT = Path(".")
ENHANCED = Path(
    r"outputs\classification_v2\frame_features"
    r"\spatiotemporal_frame_features_enhanced.csv"
)
ENH_AUDIT = Path(r"outputs\classification_v2\audits\enhanced_spatiotemporal_audit.json")
INTERVALS = Path(r"outputs\classification_v2\sequence_features\temporal_label_intervals.csv")
WINDOWS = Path(r"outputs\classification_v2\sequence_features\sequence_window_manifest.csv")
SEQ_AUDIT = Path(r"outputs\classification_v2\sequence_features\sequence_window_audit.json")
UNITS = Path(r"outputs\classification_v2\review_units\review_unit_manifest.csv")
UNIT_AUDIT = Path(r"outputs\classification_v2\review_units\review_unit_audit.json")
INTERACTION_TEMPLATE = Path(
    r"outputs\classification_v2\review_units"
    r"\interaction_review_unit_template.csv"
)
INTERACTION_SHORTLIST = Path(
    r"outputs\classification_v2\review_units"
    r"\interaction_review_unit_shortlist.csv"
)
OUT_AUDIT = Path(r"outputs\classification_v2\audits\review_unit_contract_check.json")
LEGACY_CROP_ROOT = Path(
    os.environ.get(
        "CLASSIFICATION_V2_LEGACY_CROP_ROOT",
        "outputs/legacy_16f_rebuild/"
        "legacy_16f_rebuild_20260718_v2/06_full_recovery/crops",
    )
)


def raw_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return next(csv.reader(f))


def require_file(path: Path, errors: list[str]) -> None:
    if not path.exists():
        errors.append(f"missing file: {path}")


def check_no_duplicate_header(path: Path, errors: list[str]) -> None:
    if not path.exists():
        return
    h = raw_header(path)
    dup = sorted({c for c in h if h.count(c) > 1})
    if dup:
        errors.append(f"{path} has duplicate header columns: {dup}")


def require_columns(df: pd.DataFrame, cols: list[str], name: str, errors: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        errors.append(f"{name} missing columns: {missing}")


def parse_display_indices(value) -> list[int]:
    if pd.isna(value):
        return []
    out = []
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.append(int(float(token)))
        except Exception:
            return []
    return out


def resolve_legacy_crop_path(value: object) -> Path | None:
    """Resolve stale exported crop paths through the immutable raw crop root."""
    path = Path(str(value))
    if path.exists():
        return path
    parts = list(path.parts)
    if "crops" not in parts:
        return None
    relative = Path(*parts[parts.index("crops") + 1 :])
    candidate = LEGACY_CROP_ROOT / relative
    return candidate if candidate.exists() else None


def load_audit(path: Path, errors: list[str], name: str) -> dict:
    if not path.exists():
        errors.append(f"missing audit: {path}")
        return {}
    try:
        audit = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"cannot read audit {path}: {exc}")
        return {}
    audit_errors = audit.get("errors", [])
    if audit_errors:
        errors.append(f"{name} audit errors not empty: {audit_errors}")
    return audit


def main() -> None:
    errors: list[str] = []
    warnings: list[str] = []
    summary: dict = {}

    for p in [
        ENHANCED,
        ENH_AUDIT,
        INTERVALS,
        WINDOWS,
        SEQ_AUDIT,
        UNITS,
        UNIT_AUDIT,
        INTERACTION_TEMPLATE,
        INTERACTION_SHORTLIST,
    ]:
        require_file(p, errors)
        if p.suffix.lower() == ".csv":
            check_no_duplicate_header(p, errors)

    if errors:
        print("EARLY FAIL:")
        for e in errors:
            print("-", e)
        raise SystemExit(1)

    load_audit(ENH_AUDIT, errors, "enhanced")
    load_audit(SEQ_AUDIT, errors, "sequence")
    load_audit(UNIT_AUDIT, errors, "review_unit")

    enh = pd.read_csv(ENHANCED, low_memory=False)
    intervals = pd.read_csv(INTERVALS, low_memory=False)
    windows = pd.read_csv(WINDOWS, low_memory=False)
    units = pd.read_csv(UNITS, low_memory=False)
    interaction = pd.read_csv(INTERACTION_TEMPLATE, low_memory=False)
    shortlist = pd.read_csv(INTERACTION_SHORTLIST, low_memory=False)

    require_columns(
        enh,
        [
            "source_type",
            "dataset_id",
            "video_key",
            "frame_index",
            "pig_id",
            "behavior",
            "label_anchor_frame_index",
            "label_window_start",
            "label_window_end",
            "temporal_unit_key",
        ],
        "enhanced",
        errors,
    )
    require_columns(
        intervals,
        [
            "temporal_unit_key",
            "source_type",
            "dataset_id",
            "video_key",
            "object_track_key",
            "pig_id",
            "track_id",
            "label_window_start",
            "label_window_end",
            "behavior_temporal_final",
            "temporal_consistency_status",
        ],
        "intervals",
        errors,
    )
    require_columns(
        windows,
        [
            "window_id",
            "source_type",
            "dataset_id",
            "video_key",
            "object_track_key",
            "pig_id",
            "window_length_frames",
            "window_start_frame",
            "window_end_frame",
            "behavior_window_label",
            "sequence_label_status",
            "window_valid_for_main_train",
        ],
        "windows",
        errors,
    )
    require_columns(
        units,
        [
            "review_unit_id",
            "review_unit_type",
            "source_type",
            "dataset_id",
            "video_key",
            "pig_id",
            "unit_start_frame",
            "unit_end_frame",
            "unit_frame_count",
            "display_frame_indices",
            "display_frame_count",
            "behavior_label",
            "apply_scope",
        ],
        "review units",
        errors,
    )
    require_columns(
        shortlist,
        ["review_unit_id", "review_unit_type", "source_type", "behavior_label", "review_reason"],
        "shortlist",
        errors,
    )

    if errors:
        print("SCHEMA FAIL:")
        for e in errors:
            print("-", e)
        raise SystemExit(1)

    # Identifier rules.
    if "window_uid" in windows.columns:
        errors.append("sequence_window_manifest.csv contains forbidden window_uid")
    if "window_uid" in units.columns:
        errors.append("review_unit_manifest.csv contains forbidden window_uid")
    if "window_uid" in shortlist.columns:
        errors.append("interaction_review_unit_shortlist.csv contains forbidden window_uid")
    if windows["window_id"].duplicated().any():
        duplicate = int(windows["window_id"].duplicated().sum())
        errors.append(f"duplicate window_id count = {duplicate}")
    if units["review_unit_id"].duplicated().any():
        duplicate = int(units["review_unit_id"].duplicated().sum())
        errors.append(f"duplicate review_unit_id count = {duplicate}")
    if shortlist["review_unit_id"].duplicated().any():
        duplicate = int(shortlist["review_unit_id"].duplicated().sum())
        errors.append(
            f"duplicate shortlist review_unit_id count = {duplicate}"
        )

    # Behavior labels.
    for name, df, col in [
        ("enhanced", enh, "behavior"),
        ("intervals", intervals, "behavior_temporal_final"),
        ("windows", windows, "behavior_window_label"),
        ("units", units, "behavior_label"),
        ("shortlist", shortlist, "behavior_label"),
    ]:
        vals = set(df[col].dropna().astype(str)) - {"", "nan"}
        bad = sorted(vals - VALID_BEHAVIORS)
        if bad:
            errors.append(f"{name} has non-canonical labels in {col}: {bad}")

    # CVAT frame mapping in enhanced.
    cvat = enh[enh["source_type"].astype(str).eq("cvat_tracking_xml")].copy()
    if cvat.empty:
        errors.append("enhanced has no cvat_tracking_xml rows")
    else:
        fi = pd.to_numeric(cvat["frame_index"], errors="coerce")
        ws = pd.to_numeric(cvat["label_window_start"], errors="coerce")
        we = pd.to_numeric(cvat["label_window_end"], errors="coerce")
        anchor = pd.to_numeric(cvat["label_anchor_frame_index"], errors="coerce")
        bad_inside = cvat[~((fi >= ws) & (fi <= we))]
        if len(bad_inside):
            errors.append(
                "CVAT enhanced rows where frame_index is outside label window: "
                f"{len(bad_inside)}"
            )
        bad_anchor = cvat[anchor.notna() & ((anchor % 6) != 0)]
        if len(bad_anchor):
            errors.append(f"CVAT label_anchor_frame_index not divisible by 6: {len(bad_anchor)}")
        mods = sorted((fi.dropna().astype(int) % 6).unique().tolist())
        summary["cvat_frame_mod6"] = mods
        if set(mods) != {0, 1, 2, 3, 4, 5}:
            errors.append(f"CVAT enhanced frame_index mod 6 should include 0..5, got {mods}")

    # Interval length rules.
    intervals["_len"] = (
        pd.to_numeric(intervals["label_window_end"], errors="coerce")
        - pd.to_numeric(intervals["label_window_start"], errors="coerce")
        + 1
    )
    legacy_intervals = intervals[intervals["source_type"].astype(str).eq("legacy_recovered")]
    cvat_intervals = intervals[intervals["source_type"].astype(str).eq("cvat_tracking_xml")]
    if legacy_intervals.empty:
        errors.append("temporal_label_intervals has no legacy_recovered rows")
    else:
        bad = legacy_intervals[legacy_intervals["_len"] != 16]
        if len(bad):
            errors.append(f"legacy intervals not length 16: {len(bad)}")
    if cvat_intervals.empty:
        errors.append("temporal_label_intervals has no cvat_tracking_xml rows")
    else:
        bad = cvat_intervals[cvat_intervals["_len"] != 6]
        if len(bad):
            errors.append(f"CVAT intervals not length 6: {len(bad)}")

    # Window rules.
    lengths = set(
        pd.to_numeric(windows["window_length_frames"], errors="coerce")
        .dropna()
        .astype(int)
        .tolist()
    )
    summary["window_lengths"] = sorted(lengths)
    if not {6, 8, 12, 16}.issubset(lengths):
        errors.append(f"sequence windows missing one of 6,8,12,16. got {sorted(lengths)}")
    allowed_status = {"stable", "uncertain", "transition", "incomplete"}
    status_values = set(windows["sequence_label_status"].dropna().astype(str))
    bad_status = sorted(status_values - allowed_status)
    if bad_status:
        errors.append(f"unexpected sequence_label_status values: {bad_status}")

    # Review unit rules.
    units["_display_count_calc"] = units["display_frame_indices"].map(
        lambda value: len(parse_display_indices(value))
    )
    legacy_units = units[units["source_type"].astype(str).eq("legacy_recovered")].copy()
    cvat_units = units[units["source_type"].astype(str).eq("cvat_tracking_xml")].copy()

    if legacy_units.empty:
        errors.append("review_unit_manifest has no legacy_recovered units")
    else:
        bad_type = legacy_units[legacy_units["review_unit_type"].astype(str) != "legacy_burst_16"]
        if len(bad_type):
            errors.append(f"legacy review units not legacy_burst_16: {len(bad_type)}")
        bad_count = legacy_units[
            (pd.to_numeric(legacy_units["display_frame_count"], errors="coerce") != 16)
            | (legacy_units["_display_count_calc"] != 16)
        ]
        if len(bad_count):
            errors.append(f"legacy review units not displaying 16 frames: {len(bad_count)}")
        bad_scope = legacy_units[
            legacy_units["apply_scope"].astype(str) != "whole_legacy_burst_16f"
        ]
        if len(bad_scope):
            errors.append(f"legacy review units wrong apply_scope: {len(bad_scope)}")

    if cvat_units.empty:
        errors.append("review_unit_manifest has no cvat_tracking_xml units")
    else:
        bad_type = cvat_units[cvat_units["review_unit_type"].astype(str) != "cvat_interval_6"]
        if len(bad_type):
            errors.append(f"CVAT review units not cvat_interval_6: {len(bad_type)}")
        bad_count = cvat_units[
            (pd.to_numeric(cvat_units["display_frame_count"], errors="coerce") != 6)
            | (cvat_units["_display_count_calc"] != 6)
        ]
        if len(bad_count):
            errors.append(f"CVAT review units not displaying 6 frames: {len(bad_count)}")
        bad_scope = cvat_units[cvat_units["apply_scope"].astype(str) != "cvat_interval_6f"]
        if len(bad_scope):
            errors.append(f"CVAT review units wrong apply_scope: {len(bad_scope)}")

    # Interaction shortlist rules.
    allowed_interaction = {"fight", "social-nose"}
    shortlist_labels = set(shortlist["behavior_label"].dropna().astype(str))
    bad_interaction_labels = sorted(shortlist_labels - allowed_interaction)
    if bad_interaction_labels:
        warnings.append(f"shortlist contains non-interaction labels: {bad_interaction_labels}")
    if "window_id" in shortlist.columns:
        warnings.append(
            "shortlist contains window_id; it is metadata only and review must "
            "still use review_unit_id"
        )

    # Legacy crop_path preflight for first shortlist legacy units.
    if "crop_path" in enh.columns:
        legacy_short = shortlist[
            shortlist["source_type"].astype(str).eq("legacy_recovered")
        ].head(10)
        missing_direct_crop = 0
        unresolved_crop = 0
        for _, unit in legacy_short.iterrows():
            f = enh[
                enh["source_type"].astype(str).eq("legacy_recovered")
                & enh["dataset_id"].astype(str).eq(str(unit["dataset_id"]))
                & enh["video_key"].astype(str).eq(str(unit["video_key"]))
                & enh["pig_id"].astype(str).eq(str(unit["pig_id"]))
                & enh["temporal_unit_key"].astype(str).eq(str(unit["temporal_unit_key"]))
            ].copy()
            for cp in f["crop_path"].dropna().astype(str).head(16):
                if not Path(cp).exists():
                    missing_direct_crop += 1
                if resolve_legacy_crop_path(cp) is None:
                    unresolved_crop += 1
        summary["legacy_shortlist_direct_crop_paths_missing_first10_units"] = int(
            missing_direct_crop
        )
        summary["legacy_shortlist_unresolved_crop_paths_first10_units"] = int(unresolved_crop)
        if unresolved_crop:
            warnings.append(
                "legacy crop paths unresolved after raw-root fallback in first "
                f"10 units: {unresolved_crop}"
            )
    else:
        warnings.append(
            "enhanced CSV has no crop_path column; legacy GUI may need raw-root fallback"
        )

    summary.update(
        {
            "enhanced_rows": int(len(enh)),
            "interval_rows": int(len(intervals)),
            "window_rows": int(len(windows)),
            "review_unit_rows": int(len(units)),
            "interaction_template_rows": int(len(interaction)),
            "interaction_shortlist_rows": int(len(shortlist)),
            "source_counts_units": units["source_type"].value_counts(dropna=False).to_dict(),
            "unit_type_counts": units["review_unit_type"].value_counts(dropna=False).to_dict(),
            "display_count_by_unit_type": pd.crosstab(
                units["review_unit_type"], units["display_frame_count"]
            ).to_dict(),
            "warnings": warnings,
            "errors": errors,
        }
    )

    OUT_AUDIT.parent.mkdir(parents=True, exist_ok=True)
    OUT_AUDIT.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))

    if errors:
        print("\nCONTRACT CHECK FAILED")
        for e in errors:
            print("-", e)
        raise SystemExit(1)

    print("\nCONTRACT CHECK PASSED")


if __name__ == "__main__":
    main()
