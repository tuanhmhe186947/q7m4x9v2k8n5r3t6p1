"""Build recording-group metadata for leakage-safe classification_v2 splits.

The helpers in this module intentionally avoid treating annotation pig IDs as
biological identities across videos. Filename-derived metadata is only an audit
fallback; a manually curated metadata table can override it when available.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from datetime import date
from math import isfinite
from pathlib import PureWindowsPath
from typing import Any

import pandas as pd

from pig_behavior.classification_v2.schema import VALID_BEHAVIORS

SPLITS = ("train", "val", "test")

_PIG_DATE_RE = re.compile(r"pigs?([0-3]\d[01]\d\d{2})([a-z]?)", re.IGNORECASE)
_CLIP_RE = re.compile(r"(?:^|[_\\/])(\d{6})(?:[_\\/]|$)")


@dataclass(slots=True)
class RecordingGroupTables:
    manifest: pd.DataFrame
    audit: dict[str, Any]


@dataclass(slots=True)
class PublicationSplitTables:
    split_manifest: pd.DataFrame
    audit: dict[str, Any]


def build_recording_group_manifest(
    rows: pd.DataFrame,
    *,
    manual_metadata: pd.DataFrame | None = None,
    group_level: str = "recording_date",
) -> RecordingGroupTables:
    """Return one metadata row per source/dataset/video key.

    Supported group levels:
    - ``recording_date``: strictest default for current data; all clips/videos
      from the same inferred date stay in one fold.
    - ``recording_session``: date plus session suffix and clip/session token.
    - ``video``: exact canonical video key; useful for engineering smoke only.
    """
    required = ["source_type", "dataset_id", "video_key"]
    missing = [c for c in required if c not in rows.columns]
    if missing:
        raise ValueError(f"Missing metadata input columns: {missing}")
    if group_level not in {"recording_date", "recording_session", "video"}:
        raise ValueError(f"Unsupported group_level={group_level!r}")

    source_rows = _normalized_key_frame(
        rows,
        required,
        context="recording rows",
    )
    base = source_rows[required].drop_duplicates().copy()
    base = base.sort_values(required, kind="stable").reset_index(drop=True)
    inferred = [_infer_metadata(row, group_level=group_level) for row in base.to_dict("records")]
    manifest = pd.concat([base, pd.DataFrame(inferred)], axis=1)

    if manual_metadata is not None and not manual_metadata.empty:
        manifest = _apply_manual_metadata(manifest, manual_metadata, group_level=group_level)

    _validate_recording_metadata_values(manifest, group_level=group_level)
    unknown_dates = manifest["canonical_recording_date"].eq("unknown_date")
    if group_level != "video" and unknown_dates.any():
        sample = manifest.loc[unknown_dates, "video_key"].astype(str).head(10)
        raise ValueError(
            "recording date/session grouping requires known recording dates: "
            f"count={int(unknown_dates.sum())} sample={sample.tolist()}"
        )
    audit = _recording_group_audit(
        source_rows,
        manifest,
        group_level=group_level,
    )
    return RecordingGroupTables(manifest=manifest, audit=audit)


def assign_publication_splits(
    rows: pd.DataFrame,
    group_manifest: pd.DataFrame,
    *,
    ratios: dict[str, float],
    id_col: str = "window_id",
    label_col: str = "behavior_window_label",
    valid_col: str = "window_valid_for_main_train",
) -> PublicationSplitTables:
    """Assign rows to train/val/test with zero recording-group overlap."""
    _validate_ratios(ratios)
    required_rows = [id_col, "source_type", "dataset_id", "video_key", label_col, valid_col]
    missing_rows = [c for c in required_rows if c not in rows.columns]
    if missing_rows:
        raise ValueError(f"Missing split input columns: {missing_rows}")
    required_meta = ["source_type", "dataset_id", "video_key", "recording_group_id"]
    missing_meta = [c for c in required_meta if c not in group_manifest.columns]
    if missing_meta:
        raise ValueError(f"Missing recording-group manifest columns: {missing_meta}")

    key_columns = ["source_type", "dataset_id", "video_key"]
    work = _normalized_key_frame(
        rows,
        [id_col, *key_columns],
        context="publication split rows",
    )
    if work[id_col].duplicated().any():
        duplicate_rows = int(work[id_col].duplicated(keep=False).sum())
        raise ValueError(f"duplicate split {id_col} rows={duplicate_rows}")
    meta = _normalized_key_frame(
        group_manifest,
        required_meta,
        context="recording-group manifest",
    )[required_meta].copy()
    duplicate_meta = int(meta.duplicated(key_columns, keep=False).sum())
    if duplicate_meta:
        raise ValueError(
            f"duplicate recording-group key rows={duplicate_meta}"
        )
    input_row_count = len(work)
    work = work.merge(
        meta,
        on=key_columns,
        how="left",
        validate="many_to_one",
    )
    if len(work) != input_row_count:
        raise RuntimeError(
            "publication split merge changed row count: "
            f"before={input_row_count} after={len(work)}"
        )
    missing_group_rows = int(work["recording_group_id"].isna().sum())
    if missing_group_rows:
        raise ValueError(f"Rows without recording_group_id: {missing_group_rows}")

    valid_mask, invalid_validity = _strict_bool_series(work[valid_col])
    if invalid_validity:
        raise ValueError(
            f"invalid {valid_col} values: {invalid_validity}"
        )
    raw_labels = work[label_col].fillna("").astype(str)
    labels = raw_labels.str.strip()
    padded_labels = int(raw_labels.ne(labels).sum())
    if padded_labels:
        raise ValueError(
            f"surrounding whitespace in {label_col}: {padded_labels}"
        )
    invalid_valid_labels = int(
        (valid_mask & ~labels.isin(VALID_BEHAVIORS)).sum()
    )
    if invalid_valid_labels:
        raise ValueError(
            f"invalid valid-row {label_col} values: {invalid_valid_labels}"
        )
    work[label_col] = labels
    balance_rows = work.loc[valid_mask].copy()
    if balance_rows.empty:
        raise ValueError("No valid rows available for split balancing.")

    group_stats = []
    for group_id, group in work.groupby("recording_group_id", sort=False):
        balance_group = balance_rows.loc[balance_rows["recording_group_id"].eq(group_id)]
        label_counts = Counter(balance_group[label_col].fillna("__missing__").astype(str).tolist())
        group_stats.append(
            {
                "recording_group_id": str(group_id),
                "rows": int(len(group)),
                "valid_rows": int(len(balance_group)),
                "label_counts": label_counts,
            }
        )

    total_valid_rows = sum(g["valid_rows"] for g in group_stats)
    total_labels: Counter[str] = Counter()
    for group in group_stats:
        total_labels.update(group["label_counts"])
    for group in group_stats:
        rarity_score = 0.0
        for label, count in group["label_counts"].items():
            rarity_score += count / max(1, total_labels[label])
        group["rarity_score"] = rarity_score

    target_rows = {split: total_valid_rows * ratios[split] for split in SPLITS}
    target_labels = {
        split: {
            label: count * ratios[split]
            for label, count in total_labels.items()
        }
        for split in SPLITS
    }
    assigned_rows = {split: 0 for split in SPLITS}
    assigned_labels = {split: Counter() for split in SPLITS}
    group_to_split: dict[str, str] = {}

    ordered_groups = sorted(
        group_stats,
        key=lambda g: (-g["rarity_score"], -g["valid_rows"], -g["rows"], g["recording_group_id"]),
    )
    for group in ordered_groups:
        best_split = min(
            SPLITS,
            key=lambda split: (
                assigned_rows[split] / max(1.0, target_rows[split]),
                _projected_label_score(
                    _counter_add(assigned_labels[split], group["label_counts"]),
                    target_labels[split],
                ),
                split,
            ),
        )
        group_to_split[group["recording_group_id"]] = best_split
        assigned_rows[best_split] += group["valid_rows"]
        assigned_labels[best_split].update(group["label_counts"])

    split_manifest = work[
        [
            id_col,
            "source_type",
            "dataset_id",
            "video_key",
            "recording_group_id",
            label_col,
            valid_col,
        ]
    ].copy()
    split_manifest["split"] = split_manifest["recording_group_id"].map(group_to_split)

    group_assignment = pd.DataFrame(
        [
            {"recording_group_id": group, "split": split}
            for group, split in sorted(group_to_split.items())
        ]
    )
    leakage_groups = (
        split_manifest.groupby("recording_group_id")["split"]
        .nunique()
        .loc[lambda s: s > 1]
        .index.astype(str)
        .tolist()
    )
    video_split_counts = split_manifest.groupby(key_columns)["split"].nunique()
    leakage_videos = [
        "|".join(map(str, key))
        for key in video_split_counts.loc[lambda series: series > 1]
        .index.tolist()
    ]
    positive_splits = [split for split in SPLITS if ratios[split] > 0]
    empty_positive_splits = [
        split
        for split in positive_splits
        if not (
            split_manifest["split"].eq(split)
            & valid_mask
        ).any()
    ]
    output_ids = set(split_manifest[id_col].astype(str))
    input_ids = set(work[id_col].astype(str))
    missing_output_ids = sorted(input_ids - output_ids)
    extra_output_ids = sorted(output_ids - input_ids)
    audit_errors = []
    if leakage_groups:
        audit_errors.append("recording_group_leakage_detected")
    if leakage_videos:
        audit_errors.append("video_split_leakage_detected")
    if empty_positive_splits:
        audit_errors.append(
            f"positive_ratio_split_without_valid_rows={empty_positive_splits}"
        )
    if missing_output_ids or extra_output_ids:
        audit_errors.append(
            "split_id_set_mismatch="
            f"missing:{len(missing_output_ids)},extra:{len(extra_output_ids)}"
        )
    audit = {
        "rows": int(len(split_manifest)),
        "input_rows": int(len(rows)),
        "row_count_delta": int(len(split_manifest) - len(rows)),
        "id_col": id_col,
        "valid_rows": int(valid_mask.sum()),
        "invalid_validity_values": invalid_validity,
        "invalid_valid_label_rows": invalid_valid_labels,
        "recording_group_count": int(len(group_assignment)),
        "ratios": ratios,
        "target_valid_rows": target_rows,
        "split_rows": split_manifest["split"].value_counts(dropna=False).to_dict(),
        "split_valid_rows": split_manifest.loc[valid_mask, "split"]
        .value_counts(dropna=False)
        .to_dict(),
        "split_label_counts": {
            split: split_manifest.loc[split_manifest["split"].eq(split), label_col]
            .value_counts(dropna=False)
            .to_dict()
            for split in SPLITS
        },
        "split_valid_label_counts": {
            split: split_manifest.loc[
                split_manifest["split"].eq(split) & valid_mask,
                label_col,
            ]
            .value_counts(dropna=False)
            .to_dict()
            for split in SPLITS
        },
        "recording_groups_by_split": group_assignment["split"].value_counts(dropna=False).to_dict(),
        "leakage_group_count": int(len(leakage_groups)),
        "leakage_groups": leakage_groups[:50],
        "leakage_video_count": int(len(leakage_videos)),
        "leakage_video_sample": leakage_videos[:50],
        "missing_output_id_sample": missing_output_ids[:20],
        "extra_output_id_sample": extra_output_ids[:20],
        "empty_positive_ratio_splits": empty_positive_splits,
        "errors": audit_errors,
        "warnings": [],
    }
    return PublicationSplitTables(split_manifest=split_manifest, audit=audit)


def parse_ratios(text: str) -> dict[str, float]:
    parts = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(parts) != 3:
        raise ValueError("--ratios must contain exactly train,val,test values, e.g. 0.70,0.15,0.15")
    if any(not isfinite(value) for value in parts):
        raise ValueError("--ratios values must be finite")
    total = sum(parts)
    if total <= 0:
        raise ValueError("--ratios must sum to a positive value")
    normalized = {
        name: value / total
        for name, value in zip(SPLITS, parts, strict=True)
    }
    _validate_ratios(normalized)
    return normalized


def json_default(value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _infer_metadata(row: dict[str, Any], *, group_level: str) -> dict[str, Any]:
    source_type = _clean_token(row.get("source_type"))
    dataset_id = _clean_token(row.get("dataset_id"))
    video_key = _clean_token(row.get("video_key"))
    source_text = "|".join([source_type, dataset_id, video_key])
    normalized_video_key = _normalize_video_key(video_key)
    date_token, session_suffix = _extract_date_token(source_text)
    canonical_date = _date_token_to_iso(date_token) if date_token else "unknown_date"
    clip_token = _extract_clip_token(source_text)
    recording_session_token = "|".join(
        x for x in [canonical_date, session_suffix or "session_unknown", clip_token] if x
    )

    if group_level == "recording_date":
        recording_group_id = f"date={canonical_date}"
    elif group_level == "recording_session":
        recording_group_id = f"session={recording_session_token}"
    else:
        recording_group_id = f"video={source_type}|{dataset_id}|{normalized_video_key}"

    metadata_quality = "filename_inferred" if date_token else "unknown_date"
    return {
        "canonical_video_key": normalized_video_key,
        "canonical_recording_date": canonical_date,
        "recording_session_token": recording_session_token,
        "source_alias_group": (
            f"{source_type}|{canonical_date}|"
            f"{clip_token or normalized_video_key}"
        ),
        "recording_group_id": recording_group_id,
        "recording_group_level": group_level,
        "metadata_quality": metadata_quality,
        "farm_id": "unknown",
        "cohort_id": "unknown",
        "pen_id": "unknown",
        "camera_id": "unknown",
        "biological_subject_scope_known": False,
        "biological_subject_note": (
            "annotation pig_id is not treated as a cross-video biological "
            "identity"
        ),
    }


def _apply_manual_metadata(
    manifest: pd.DataFrame,
    manual_metadata: pd.DataFrame,
    *,
    group_level: str,
) -> pd.DataFrame:
    keys = ["source_type", "dataset_id", "video_key"]
    missing_keys = [c for c in keys if c not in manual_metadata.columns]
    if missing_keys:
        raise ValueError(f"Manual metadata missing key columns: {missing_keys}")
    manual = _normalized_key_frame(
        manual_metadata,
        keys,
        context="manual recording metadata",
    )
    duplicate_manual = int(manual.duplicated(keys, keep=False).sum())
    if duplicate_manual:
        raise ValueError(
            f"duplicate manual metadata key rows: {duplicate_manual}"
        )
    manifest_keys = set(map(tuple, manifest[keys].itertuples(index=False, name=None)))
    manual_keys = set(map(tuple, manual[keys].itertuples(index=False, name=None)))
    unmatched_manual = sorted(manual_keys - manifest_keys)
    if unmatched_manual:
        raise ValueError(
            "manual metadata contains unmatched source/video keys: "
            f"count={len(unmatched_manual)} sample={unmatched_manual[:10]}"
        )
    merged = manifest.merge(
        manual,
        on=keys,
        how="left",
        suffixes=("", "_manual"),
        validate="one_to_one",
    )
    has_manual_override = pd.Series(False, index=merged.index)
    override_cols = [
        "canonical_recording_date",
        "recording_session_token",
        "source_alias_group",
        "farm_id",
        "cohort_id",
        "pen_id",
        "camera_id",
        "biological_subject_scope_known",
        "biological_subject_note",
    ]
    for col in override_cols:
        manual_col = f"{col}_manual"
        if manual_col in merged.columns:
            mask = merged[manual_col].notna()
            if col == "biological_subject_scope_known":
                parsed, invalid = _strict_bool_series(merged.loc[mask, manual_col])
                if invalid:
                    raise ValueError(
                        "invalid manual biological_subject_scope_known values: "
                        f"{invalid}"
                    )
                merged.loc[mask, col] = parsed.to_numpy()
                has_manual_override = has_manual_override | mask
                merged = merged.drop(columns=[manual_col])
                continue
            text = merged.loc[mask, manual_col].astype(str).str.strip()
            if text.eq("").any():
                raise ValueError(f"blank manual override values for {col}")
            has_manual_override = has_manual_override | mask
            merged.loc[mask, col] = text
            merged = merged.drop(columns=[manual_col])
    explicit_group = pd.Series(False, index=merged.index)
    explicit_group_values = pd.Series("", index=merged.index, dtype=object)
    if "recording_group_id_manual" in merged.columns:
        mask = merged["recording_group_id_manual"].notna()
        text = merged.loc[mask, "recording_group_id_manual"].astype(str).str.strip()
        if text.eq("").any():
            raise ValueError("blank manual override values for recording_group_id")
        explicit_group.loc[mask] = True
        explicit_group_values.loc[mask] = text
        has_manual_override = has_manual_override | mask
        merged = merged.drop(columns=["recording_group_id_manual"])
    merged["recording_group_id"] = merged.apply(
        lambda row: _group_id_from_row(row, group_level),
        axis=1,
    )
    merged.loc[explicit_group, "recording_group_id"] = explicit_group_values.loc[
        explicit_group
    ]
    _validate_recording_metadata_values(merged, group_level=group_level)
    merged.loc[has_manual_override, "metadata_quality"] = "manual"
    return merged


def _group_id_from_row(row: pd.Series, group_level: str) -> str:
    if group_level == "recording_date":
        return f"date={row['canonical_recording_date']}"
    if group_level == "recording_session":
        return f"session={row['recording_session_token']}"
    return f"video={row['source_type']}|{row['dataset_id']}|{row['canonical_video_key']}"


def _recording_group_audit(
    rows: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    group_level: str,
) -> dict[str, Any]:
    by_video = rows[["source_type", "dataset_id", "video_key"]].drop_duplicates()
    date_counts = manifest["canonical_recording_date"].value_counts(dropna=False).to_dict()
    unknown_dates = (
        manifest.loc[
            manifest["canonical_recording_date"].eq("unknown_date"),
            "video_key",
        ]
        .astype(str)
        .tolist()
    )
    biological_known, invalid_biological_known = _strict_bool_series(
        manifest["biological_subject_scope_known"]
    )
    errors = []
    if invalid_biological_known:
        errors.append(
            "invalid_biological_subject_scope_known="
            f"{invalid_biological_known}"
        )
    return {
        "input_rows": int(len(rows)),
        "unique_source_dataset_video_rows": int(len(by_video)),
        "manifest_rows": int(len(manifest)),
        "group_level": group_level,
        "recording_group_count": int(manifest["recording_group_id"].nunique(dropna=False)),
        "canonical_recording_date_counts": date_counts,
        "metadata_quality_counts": (
            manifest["metadata_quality"]
            .value_counts(dropna=False)
            .to_dict()
        ),
        "source_type_counts": rows["source_type"].value_counts(dropna=False).to_dict(),
        "unknown_date_video_count": int(len(unknown_dates)),
        "unknown_date_video_sample": unknown_dates[:50],
        "biological_subject_scope_known": bool(biological_known.any()),
        "warnings": [
            "biological_subject_identity_unknown_do_not_use_pig_id_cross_video",
            *([] if not unknown_dates else ["some_video_keys_have_unknown_recording_date"]),
        ],
        "errors": errors,
    }


def _normalize_video_key(value: Any) -> str:
    text = _clean_token(value).replace("\\", "/")
    if not text:
        return "unknown_video"
    path = PureWindowsPath(text)
    name = path.stem if path.suffix else path.name
    name = name.lower()
    if name == "color":
        parent = path.parent.name.lower()
        grandparent = path.parent.parent.name.lower()
        return f"{grandparent}_{parent}_color"
    for suffix in ("_30fps", ".mp4"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return re.sub(r"[^a-z0-9]+", "_", name).strip("_") or "unknown_video"


def _extract_date_token(text: str) -> tuple[str | None, str | None]:
    match = _PIG_DATE_RE.search(text)
    if not match:
        return None, None
    return match.group(1), match.group(2).lower() or None


def _date_token_to_iso(token: str) -> str:
    day = int(token[:2])
    month = int(token[2:4])
    year = 2000 + int(token[4:6])
    try:
        return date(year, month, day).isoformat()
    except ValueError as exc:
        raise ValueError(f"Invalid recording date token: {token!r}") from exc


def _extract_clip_token(text: str) -> str | None:
    matches = _CLIP_RE.findall(text.replace("\\", "/"))
    return matches[-1] if matches else None


def _validate_ratios(ratios: dict[str, float]) -> None:
    missing = [split for split in SPLITS if split not in ratios]
    if missing:
        raise ValueError(f"Missing split ratios for: {missing}")
    if any(not isfinite(ratios[split]) for split in SPLITS):
        raise ValueError(f"Split ratios must be finite: {ratios}")
    if any(ratios[split] < 0 for split in SPLITS):
        raise ValueError(f"Split ratios must be non-negative: {ratios}")
    total = sum(ratios[split] for split in SPLITS)
    if total <= 0:
        raise ValueError(f"Split ratios must sum to a positive value: {ratios}")
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must be normalized to one: {ratios}")


def _strict_bool_series(series: pd.Series) -> tuple[pd.Series, int]:
    """Parse explicit booleans without silently converting unknown text."""

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool), int(series.isna().sum())
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    truthy = {"true", "1", "yes", "y", "t"}
    falsy = {"false", "0", "no", "n", "f"}
    invalid = int((~normalized.isin(truthy | falsy)).sum())
    return normalized.isin(truthy), invalid


def _normalized_key_frame(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    context: str,
) -> pd.DataFrame:
    """Copy and validate stable keys without silently trimming malformed IDs."""

    out = frame.copy()
    errors: list[str] = []
    for column in columns:
        raw = out[column].fillna("").astype(str)
        normalized = raw.str.strip()
        blank = int(normalized.eq("").sum())
        padded = int(raw.ne(normalized).sum())
        if blank:
            errors.append(f"blank_{column}={blank}")
        if padded:
            errors.append(f"surrounding_whitespace_{column}={padded}")
        out[column] = normalized
    if errors:
        raise ValueError(f"{context} key contract failed: " + "; ".join(errors))
    return out


def _validate_recording_metadata_values(
    manifest: pd.DataFrame,
    *,
    group_level: str,
) -> None:
    """Validate manual/inferred metadata before it can define split groups."""

    errors: list[str] = []
    dates = manifest["canonical_recording_date"].fillna("").astype(str).str.strip()
    for value in sorted(set(dates)):
        if value == "unknown_date":
            continue
        try:
            date.fromisoformat(value)
        except ValueError:
            errors.append(f"invalid_canonical_recording_date={value!r}")
    group_ids = manifest["recording_group_id"].fillna("").astype(str).str.strip()
    if group_ids.eq("").any():
        errors.append(
            f"blank_recording_group_id={int(group_ids.eq('').sum())}"
        )
    if group_level == "recording_session":
        sessions = (
            manifest["recording_session_token"]
            .fillna("")
            .astype(str)
            .str.strip()
        )
        if sessions.eq("").any():
            errors.append(
                "blank_recording_session_token="
                f"{int(sessions.eq('').sum())}"
            )
    if errors:
        raise ValueError(
            "recording metadata value contract failed: " + "; ".join(errors)
        )


def _counter_add(a: Counter[str], b: Counter[str]) -> Counter[str]:
    out = Counter(a)
    out.update(b)
    return out


def _projected_label_score(
    candidate_counts: Counter[str],
    target_counts: dict[str, float],
) -> float:
    if not target_counts:
        return 0.0
    return sum(
        abs(candidate_counts.get(label, 0) - target) / max(1.0, target)
        for label, target in target_counts.items()
    )


def _clean_token(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()
