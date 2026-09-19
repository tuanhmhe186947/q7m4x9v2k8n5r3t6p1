"""Build review-only evidence from leakage-safe Pig-STRENet artifacts.

The output is one row per native temporal unit.  Every derived column starts
with ``review_pig_`` so it cannot enter model-X.  Behavior labels in the pair
manifest are deliberately ignored; this layer describes evidence, not a
target-conditioned decision.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

ROI_CLASSES = ("feeder", "drinker", "toy")

PIG_REVIEW_EVIDENCE_COLUMNS: tuple[str, ...] = (
    "review_pig_evidence_available",
    "review_pig_pair_count",
    "review_pig_history_available_ratio",
    "review_pig_history_transition_available",
    "review_pig_history_duration_sec",
    "review_pig_target_duration_sec",
    "review_pig_history_start_frame",
    "review_pig_history_end_frame",
    "review_pig_history_display_frame_indices",
    "review_pig_motion_transition_score",
    "review_pig_stationary_to_motion_score",
    "review_pig_motion_to_stationary_score",
    "review_pig_history_motion_burstiness_score",
    "review_pig_social_phase_score",
    "review_pig_contact_persistence_score",
    "review_pig_partner_change_score",
    "review_pig_shape_transition_score",
    "review_pig_topk_overlap_score",
    "review_pig_topk_contact_ratio",
    "review_pig_pair_motion_energy_p90",
    "review_pig_pair_motion_source_percentile",
    "review_pig_social_valid_ratio",
    "review_pig_diff_valid_ratio",
    "review_pig_diff_active_pixel_ratio",
    "review_pig_diff_inner_mean",
    "review_pig_diff_boundary_mean",
    "review_pig_roi_feeder_phase_score",
    "review_pig_roi_feeder_valid_ratio",
    "review_pig_roi_feeder_motion_inside_p90",
    "review_pig_roi_drinker_phase_score",
    "review_pig_roi_drinker_valid_ratio",
    "review_pig_roi_drinker_motion_inside_p90",
    "review_pig_roi_toy_phase_score",
    "review_pig_roi_toy_valid_ratio",
    "review_pig_roi_toy_motion_inside_p90",
)


def load_pig_strenet_review_evidence(
    artifact_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a standard artifact directory and build native-unit evidence."""

    root = Path(artifact_dir)
    required = {
        "pair_manifest": root / "pair_manifest.csv",
        "history_features": root / "history_features.csv",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise ValueError(f"Pig-STRENet review artifacts missing files={missing}")
    optional = {
        "roi_dynamics": root / "roi_dynamics.csv",
        "social_edges": root / "social_edges.csv",
        "difference_summary": root / "stabilized_difference_summary.csv",
    }
    tables = {
        name: pd.read_csv(path, low_memory=False)
        for name, path in {**required, **optional}.items()
        if path.exists()
    }
    evidence, audit = build_pig_strenet_review_evidence(
        tables["pair_manifest"],
        tables["history_features"],
        roi_dynamics=tables.get("roi_dynamics"),
        social_edges=tables.get("social_edges"),
        difference_summary=tables.get("difference_summary"),
    )
    audit["artifact_dir"] = str(root)
    audit["input_sha256"] = {
        name: _sha256(path)
        for name, path in {**required, **optional}.items()
        if path.exists()
    }
    return evidence, audit


def build_pig_strenet_review_evidence(
    pair_manifest: pd.DataFrame,
    history_features: pd.DataFrame,
    *,
    roi_dynamics: pd.DataFrame | None = None,
    social_edges: pd.DataFrame | None = None,
    difference_summary: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Collapse pair artifacts to one transparent row per temporal unit."""

    _require_columns(
        pair_manifest,
        [
            "pair_id",
            "temporal_unit_key",
            "source_type",
            "video_key",
            "history_expected_frame_count",
            "history_available_ratio",
            "history_complete",
            "target_complete",
            "history_window_start_frame",
            "history_window_end_frame",
            "history_duration_sec",
            "target_duration_sec",
        ],
        "pair_manifest",
    )
    _require_columns(history_features, ["pair_id"], "history_features")
    _require_unique(pair_manifest, "pair_id", "pair_manifest")
    _require_unique(history_features, "pair_id", "history_features")

    ignored_target_columns = {
        "behavior_label_audit_only",
        "label_propagation_policy",
        "source_lineage_review_complete",
    }
    base_columns = [
        column
        for column in pair_manifest.columns
        if column not in ignored_target_columns
    ]
    pair = pair_manifest[base_columns].merge(
        history_features,
        on="pair_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_history"),
    )
    if len(pair) != len(pair_manifest):
        raise RuntimeError("Pig-STRENet review merge changed pair count")

    history_complete = _bool_series(pair["history_complete"])
    target_complete = _bool_series(pair["target_complete"])
    declared_transition = _bool_series(
        pair.get(
            "history_target_transition_available",
            history_complete & target_complete,
        )
    )
    transition_valid = history_complete & target_complete & declared_transition

    out = pair[
        ["pair_id", "temporal_unit_key", "source_type", "video_key"]
    ].copy()
    out["review_pig_evidence_available"] = target_complete
    out["review_pig_history_available_ratio"] = _numeric(
        pair["history_available_ratio"]
    )
    out["review_pig_history_transition_available"] = transition_valid
    out["review_pig_history_duration_sec"] = _numeric(
        pair["history_duration_sec"]
    )
    out["review_pig_target_duration_sec"] = _numeric(pair["target_duration_sec"])
    out["review_pig_history_start_frame"] = _numeric(
        pair["history_window_start_frame"]
    )
    out["review_pig_history_end_frame"] = _numeric(
        pair["history_window_end_frame"]
    )
    out["review_pig_history_display_frame_indices"] = [
        _frame_range(start, end, available)
        for start, end, available in zip(
            out["review_pig_history_start_frame"],
            out["review_pig_history_end_frame"],
            out["review_pig_history_available_ratio"],
            strict=True,
        )
    ]

    motion_transition = _row_max(
        pair,
        [
            "stationary_per_second_to_motion_score",
            "motion_to_stationary_per_second_score",
        ],
    )
    social_phase = _row_max(
        pair,
        [
            "approach_per_second_to_contact_score",
            "contact_persistence_score",
            "contact_to_separation_per_second_score",
        ],
    )
    out["review_pig_motion_transition_score"] = _masked(
        motion_transition,
        transition_valid,
    )
    out["review_pig_stationary_to_motion_score"] = _masked(
        _numeric(
            pair.get("stationary_per_second_to_motion_score", 0.0)
        ).clip(0.0, 1.0),
        transition_valid,
    )
    out["review_pig_motion_to_stationary_score"] = _masked(
        _numeric(
            pair.get("motion_to_stationary_per_second_score", 0.0)
        ).clip(0.0, 1.0),
        transition_valid,
    )
    out["review_pig_history_motion_burstiness_score"] = _masked(
        _numeric(
            pair.get("history_motion_burstiness_per_second", 0.0)
        ).div(2.0).clip(0.0, 1.0),
        history_complete,
    )
    out["review_pig_social_phase_score"] = _masked(social_phase, transition_valid)
    out["review_pig_contact_persistence_score"] = _masked(
        _numeric(pair.get("contact_persistence_score", 0.0)).clip(0.0, 1.0),
        transition_valid,
    )
    out["review_pig_partner_change_score"] = _masked(
        _numeric(pair.get("partner_change_count", 0.0)).div(3.0).clip(0.0, 1.0),
        transition_valid,
    )
    out["review_pig_shape_transition_score"] = _masked(
        _numeric(pair.get("shape_change_history_to_target", 0.0))
        .div(0.20)
        .clip(0.0, 1.0),
        transition_valid,
    )
    for roi_class in ROI_CLASSES:
        out[f"review_pig_roi_{roi_class}_phase_score"] = _masked(
            _row_max(
                pair,
                [
                    f"{roi_class}_approach_n_per_second_to_engagement",
                    f"{roi_class}_engagement_to_departure_n_per_second",
                ],
            ),
            transition_valid,
        )

    out = out.merge(
        _social_summary(pair_manifest, social_edges),
        on="pair_id",
        how="left",
        validate="one_to_one",
    )
    out = out.merge(
        _roi_summary(roi_dynamics),
        on="pair_id",
        how="left",
        validate="one_to_one",
    )
    out = out.merge(
        _difference_summary(pair_manifest, difference_summary),
        on="pair_id",
        how="left",
        validate="one_to_one",
    )
    out = _fill_review_defaults(out)
    out["review_pig_pair_motion_source_percentile"] = (
        out.groupby(["source_type", "video_key"], dropna=False)[
            "review_pig_pair_motion_energy_p90"
        ]
        .rank(method="average", pct=True)
        .fillna(0.0)
    )
    collapsed = _collapse_native_units(out)
    audit = {
        "schema_version": "classification_v2.pig_strenet_review_evidence.v2",
        "primary_motion_time_basis": "source_frame_timestamp_seconds",
        "pair_rows": int(len(pair_manifest)),
        "review_unit_rows": int(len(collapsed)),
        "transition_valid_pairs": int(transition_valid.sum()),
        "transition_invalid_pairs": int((~transition_valid).sum()),
        "roi_available": roi_dynamics is not None,
        "social_available": social_edges is not None,
        "difference_available": difference_summary is not None,
        "behavior_columns_ignored": sorted(ignored_target_columns),
        "errors": [],
        "warnings": [],
        "valid": True,
    }
    return collapsed, audit


def attach_pig_strenet_review_evidence(
    temporal_units: pd.DataFrame,
    evidence: pd.DataFrame,
) -> pd.DataFrame:
    """Attach review evidence without changing rows, keys, or labels."""

    _require_columns(temporal_units, ["temporal_unit_key"], "temporal_units")
    _require_columns(evidence, ["temporal_unit_key"], "review_evidence")
    _require_unique(evidence, "temporal_unit_key", "review_evidence")
    original_columns = list(temporal_units.columns)
    protected = [
        column
        for column in temporal_units.columns
        if "behavior" in column.lower() or "label" in column.lower()
    ]
    before = temporal_units[protected].copy(deep=True)
    overlap = sorted(
        set(original_columns).intersection(PIG_REVIEW_EVIDENCE_COLUMNS)
    )
    if overlap:
        raise ValueError(f"temporal units already contain Pig review columns={overlap}")
    out = temporal_units.merge(
        evidence,
        on="temporal_unit_key",
        how="left",
        validate="one_to_one",
    )
    if len(out) != len(temporal_units):
        raise RuntimeError("Pig-STRENet review attach changed temporal-unit count")
    if not out[protected].equals(before):
        raise RuntimeError("Pig-STRENet review attach changed label columns")
    return _fill_review_defaults(out)


def _social_summary(
    pairs: pd.DataFrame,
    edges: pd.DataFrame | None,
) -> pd.DataFrame:
    pair_ids = pairs[["pair_id", "history_expected_frame_count"]].copy()
    if edges is None or edges.empty:
        return pair_ids[["pair_id"]]
    _require_columns(
        edges,
        [
            "pair_id",
            "slot_index",
            "edge_available",
            "pair_overlap_ratio",
            "pair_contact",
            "pair_motion_energy_n_per_second2",
        ],
        "social_edges",
    )
    work = edges.merge(pair_ids, on="pair_id", how="inner", validate="many_to_one")
    target = work[
        _numeric(work["slot_index"]).ge(
            _numeric(work["history_expected_frame_count"])
        )
    ].copy()
    target["edge_available"] = _bool_series(target["edge_available"])
    target["pair_contact"] = _bool_series(target["pair_contact"])
    slot = target.groupby(["pair_id", "slot_index"], sort=False).agg(
        slot_available=("edge_available", "max"),
        slot_overlap=("pair_overlap_ratio", "max"),
        slot_contact=("pair_contact", "max"),
        slot_motion=("pair_motion_energy_n_per_second2", "max"),
    ).reset_index()
    return slot.groupby("pair_id", sort=False).agg(
        review_pig_topk_overlap_score=("slot_overlap", "max"),
        review_pig_topk_contact_ratio=("slot_contact", "mean"),
        review_pig_pair_motion_energy_p90=(
            "slot_motion",
            lambda values: float(pd.to_numeric(values).quantile(0.90)),
        ),
        review_pig_social_valid_ratio=("slot_available", "mean"),
    ).reset_index()


def _roi_summary(roi: pd.DataFrame | None) -> pd.DataFrame:
    if roi is None or roi.empty:
        return pd.DataFrame({"pair_id": pd.Series(dtype="object")})
    _require_columns(
        roi,
        [
            "pair_id",
            "slot_role",
            "roi_class",
            "available",
            "motion_inside_n_per_second",
        ],
        "roi_dynamics",
    )
    target = roi[roi["slot_role"].astype(str).eq("target")].copy()
    target["available"] = _bool_series(target["available"])
    rows: list[pd.DataFrame] = []
    for roi_class in ROI_CLASSES:
        part = target[target["roi_class"].astype(str).eq(roi_class)]
        if part.empty:
            continue
        summary = part.groupby("pair_id", sort=False).agg(
            **{
                f"review_pig_roi_{roi_class}_valid_ratio": (
                    "available",
                    "mean",
                ),
                f"review_pig_roi_{roi_class}_motion_inside_p90": (
                    "motion_inside_n_per_second",
                    lambda values: float(pd.to_numeric(values).quantile(0.90)),
                ),
            }
        ).reset_index()
        rows.append(summary)
    if not rows:
        return pd.DataFrame({"pair_id": pd.Series(dtype="object")})
    result = rows[0]
    for part in rows[1:]:
        result = result.merge(part, on="pair_id", how="outer", validate="one_to_one")
    return result


def _difference_summary(
    pairs: pd.DataFrame,
    difference: pd.DataFrame | None,
) -> pd.DataFrame:
    pair_ids = pairs[["pair_id", "history_expected_frame_count"]].copy()
    if difference is None or difference.empty:
        return pair_ids[["pair_id"]]
    _require_columns(
        difference,
        [
            "pair_id",
            "pair_slot_index",
            "pair_valid",
            "diff_active_pixel_ratio",
            "diff_inner_mean",
            "diff_boundary_mean",
        ],
        "difference_summary",
    )
    work = difference.merge(
        pair_ids,
        on="pair_id",
        how="inner",
        validate="many_to_one",
    )
    target = work[
        _numeric(work["pair_slot_index"]).ge(
            _numeric(work["history_expected_frame_count"])
        )
    ].copy()
    target["pair_valid"] = _bool_series(target["pair_valid"])
    return target.groupby("pair_id", sort=False).agg(
        review_pig_diff_valid_ratio=("pair_valid", "mean"),
        review_pig_diff_active_pixel_ratio=("diff_active_pixel_ratio", "max"),
        review_pig_diff_inner_mean=("diff_inner_mean", "max"),
        review_pig_diff_boundary_mean=("diff_boundary_mean", "max"),
    ).reset_index()


def _collapse_native_units(pair_evidence: pd.DataFrame) -> pd.DataFrame:
    aggregations: dict[str, Any] = {
        "review_pig_evidence_available": "max",
        "review_pig_history_available_ratio": "max",
        "review_pig_history_transition_available": "max",
        "review_pig_history_duration_sec": "max",
        "review_pig_target_duration_sec": "max",
        "review_pig_history_start_frame": "min",
        "review_pig_history_end_frame": "max",
        "review_pig_history_display_frame_indices": _join_frame_lists,
    }
    for column in PIG_REVIEW_EVIDENCE_COLUMNS:
        if column in aggregations or column in {
            "review_pig_pair_count",
        }:
            continue
        aggregations[column] = "max"
    grouped = pair_evidence.groupby("temporal_unit_key", sort=False).agg(
        **{column: (column, reducer) for column, reducer in aggregations.items()}
    ).reset_index()
    counts = pair_evidence.groupby("temporal_unit_key", sort=False)["pair_id"].nunique()
    grouped["review_pig_pair_count"] = grouped["temporal_unit_key"].map(counts).astype(int)
    return grouped[["temporal_unit_key", *PIG_REVIEW_EVIDENCE_COLUMNS]]


def _fill_review_defaults(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    text_columns = {"review_pig_history_display_frame_indices"}
    bool_columns = {
        "review_pig_evidence_available",
        "review_pig_history_transition_available",
    }
    for column in PIG_REVIEW_EVIDENCE_COLUMNS:
        if column not in out:
            out[column] = "" if column in text_columns else False if column in bool_columns else 0.0
        if column in text_columns:
            out[column] = out[column].fillna("").astype(str).replace("nan", "")
        elif column in bool_columns:
            out[column] = _bool_series(out[column])
        else:
            out[column] = _numeric(out[column]).fillna(0.0)
    return out


def _row_max(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    values = []
    for column in columns:
        if column in frame:
            values.append(_numeric(frame[column]).rename(column))
        else:
            values.append(pd.Series(0.0, index=frame.index, name=column))
    return pd.concat(values, axis=1).max(axis=1).clip(0.0, 1.0)


def _masked(values: pd.Series, valid: pd.Series) -> pd.Series:
    return _numeric(values).where(_bool_series(valid), 0.0).fillna(0.0)


def _numeric(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.Series(values if isinstance(values, list) else [values], dtype="float64")


def _bool_series(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        source = values
    else:
        source = pd.Series(values if isinstance(values, list) else [values])
    if pd.api.types.is_bool_dtype(source):
        return source.fillna(False).astype(bool)
    return source.fillna("").astype(str).str.strip().str.lower().isin(
        {"1", "true", "yes", "y"}
    )


def _frame_range(start: Any, end: Any, available_ratio: Any) -> str:
    try:
        if float(available_ratio) <= 0:
            return ""
        first = int(float(start))
        last = int(float(end))
    except (TypeError, ValueError):
        return ""
    return ",".join(str(value) for value in range(first, last + 1))


def _join_frame_lists(values: pd.Series) -> str:
    frames: set[int] = set()
    for value in values.fillna("").astype(str):
        for token in value.split(","):
            try:
                frames.add(int(token.strip()))
            except ValueError:
                continue
    return ",".join(str(value) for value in sorted(frames))


def _require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} missing columns={missing}")


def _require_unique(frame: pd.DataFrame, column: str, name: str) -> None:
    if frame[column].astype(str).duplicated().any():
        raise ValueError(f"{name} has duplicate {column}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "PIG_REVIEW_EVIDENCE_COLUMNS",
    "attach_pig_strenet_review_evidence",
    "build_pig_strenet_review_evidence",
    "load_pig_strenet_review_evidence",
]
