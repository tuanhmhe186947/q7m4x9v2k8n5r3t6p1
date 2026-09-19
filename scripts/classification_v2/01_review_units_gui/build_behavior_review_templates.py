from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROI_BEHAVIOR_TO_TARGET = {
    "eat": "feeder",
    "drink": "drinker",
    "playwithtoy": "toy",
}
ROI_DOMINANT = set(ROI_BEHAVIOR_TO_TARGET)
AGGRESSION_SOCIAL = {"fight", "social-nose"}
ROI_AMBIGUOUS_NON_TARGET = {"stand", "explore", "move"}
MOTION_STATE = {"move", "explore", "stand"}
POSTURE_STATE = {"lying", "sitting", "stand"}
VALID_STRENGTH = {"strong", "medium", "weak", "boundary"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build behavior review templates and automatic label-strength attributes "
            "for the Pig Behavior classification_v2 frame-object CSV."
        )
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--scope",
        choices=["critical", "roi", "all_review", "all_rows"],
        default="critical",
        help=(
            "critical: ROI conflicts + social/aggression + boundaries + motion ambiguity; "
            "roi: ROI-label consistency only; all_review: all rows flagged for review; "
            "all_rows: full table in the review template."
        ),
    )
    parser.add_argument("--max-review-rows", type=int, default=None)
    parser.add_argument("--write-full-annotated", action="store_true")
    parser.add_argument("--motion-low-threshold", type=float, default=0.18)
    parser.add_argument("--motion-strong-threshold", type=float, default=0.75)
    parser.add_argument("--boundary-frame-gap", type=int, default=12)
    parser.add_argument("--window-radius", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv, low_memory=False)
    annotated = add_review_attributes(
        df,
        motion_low_threshold=args.motion_low_threshold,
        motion_strong_threshold=args.motion_strong_threshold,
        boundary_frame_gap=args.boundary_frame_gap,
        window_radius=args.window_radius,
    )

    review = select_review_rows(annotated, scope=args.scope)
    if args.max_review_rows is not None:
        review = review.head(args.max_review_rows).copy()

    template = make_review_template(review)
    template_path = args.output_dir / "behavior_strength_review_template.csv"
    template.to_csv(template_path, index=False)

    priority_template = template.sort_values(
        [
            "review_priority",
            "ambiguity_group_auto",
            "behavior",
            "video_key",
            "frame_index",
        ],
        kind="mergesort",
    )
    priority_template.to_csv(
        args.output_dir / "behavior_strength_review_template_priority.csv",
        index=False,
    )

    if args.write_full_annotated:
        annotated.to_csv(args.output_dir / "frame_features_with_auto_review_attrs.csv", index=False)

    audit = build_audit(annotated, template, args)
    (args.output_dir / "behavior_strength_review_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"input rows: {len(df)}")
    print(f"review rows: {len(template)}")
    print(f"template: {template_path}")
    print(f"audit: {args.output_dir / 'behavior_strength_review_audit.json'}")


def add_review_attributes(
    df: pd.DataFrame,
    *,
    motion_low_threshold: float,
    motion_strong_threshold: float,
    boundary_frame_gap: int,
    window_radius: int,
) -> pd.DataFrame:
    out = df.copy()
    required_motion_scope = {
        "temporal_unit_key",
        "timestamp_sec",
        "frame_index",
    }
    missing_scope = sorted(required_motion_scope.difference(out.columns))
    if missing_scope:
        raise ValueError(
            "behavior template requires native-unit physical-time scope: "
            f"missing={missing_scope}"
        )
    out["behavior"] = out.get("behavior", "").astype(str).str.strip()

    ensure_columns(out)
    out["review_key"] = make_review_keys(out)
    out["entity_key"] = make_entity_keys(out)

    add_motion_and_boundary_features(
        out,
        boundary_frame_gap=boundary_frame_gap,
        window_radius=window_radius,
    )

    attrs = out.apply(
        lambda row: infer_row_review_attrs(
            row,
            motion_low_threshold=motion_low_threshold,
            motion_strong_threshold=motion_strong_threshold,
        ),
        axis=1,
        result_type="expand",
    )

    for col in attrs.columns:
        out[col] = attrs[col]

    return out


def ensure_columns(df: pd.DataFrame) -> None:
    defaults: dict[str, Any] = {
        "source_type": "",
        "dataset_id": "",
        "video_key": "",
        "source_video_key": "",
        "frame_uid": "",
        "frame_index": np.nan,
        "relative_frame_index": np.nan,
        "track_id": "",
        "track_label": "",
        "pig_id": "",
        "behavior": "",
        "x1": np.nan,
        "y1": np.nan,
        "x2": np.nan,
        "y2": np.nan,
        "cx_n": np.nan,
        "cy_n": np.nan,
        "bbox_valid": True,
        "roi_target_class": "",
        "roi_target_contact": False,
        "roi_target_near": False,
        "roi_target_available": False,
        "roi_context_quality": "",
        "roi_feeder_near": False,
        "roi_feeder_contact": False,
        "roi_drinker_near": False,
        "roi_drinker_contact": False,
        "roi_toy_near": False,
        "roi_toy_contact": False,
        "interaction_partner_count": 0,
        "social_feature_quality": "",
        "local_context_quality": "",
        "training_tier": "",
        "qa_status": "",
        "sample_weight": 1.0,
        "crop_path": "",
        "image_name": "",
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    for col in [
        "frame_index",
        "relative_frame_index",
        "timestamp_sec",
        "cx_n",
        "cy_n",
        "interaction_partner_count",
        "sample_weight",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")


def make_review_keys(df: pd.DataFrame) -> pd.Series:
    key_cols = [
        "source_type",
        "dataset_id",
        "video_key",
        "frame_index",
        "relative_frame_index",
        "track_id",
        "pig_id",
        "behavior",
        "x1",
        "y1",
        "x2",
        "y2",
    ]

    def row_hash(row: pd.Series) -> str:
        text = "|".join(str(row.get(c, "")) for c in key_cols)
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

    return df.apply(row_hash, axis=1)


def make_entity_keys(df: pd.DataFrame) -> pd.Series:
    track = df.get("track_id", "").astype(str)
    pig = df.get("pig_id", "").astype(str)
    entity = np.where(track.str.len() > 0, "track=" + track, "pig=" + pig)
    return (
        df.get("source_type", "").astype(str)
        + "::"
        + df.get("dataset_id", "").astype(str)
        + "::"
        + df.get("video_key", "").astype(str)
        + "::"
        + pd.Series(entity, index=df.index).astype(str)
    )


def add_motion_and_boundary_features(
    df: pd.DataFrame,
    *,
    boundary_frame_gap: int,
    window_radius: int,
) -> None:
    df.sort_values(
        [
            "entity_key",
            "temporal_unit_key",
            "frame_index",
            "relative_frame_index",
        ],
        kind="mergesort",
        inplace=True,
    )

    grouped = df.groupby(
        ["entity_key", "temporal_unit_key"],
        sort=False,
        group_keys=False,
    )
    df["prev_behavior_auto"] = grouped["behavior"].shift(1)
    df["next_behavior_auto"] = grouped["behavior"].shift(-1)
    df["prev_frame_index_auto"] = grouped["frame_index"].shift(1)
    df["next_frame_index_auto"] = grouped["frame_index"].shift(-1)
    df["prev_cx_n_auto"] = grouped["cx_n"].shift(1)
    df["prev_cy_n_auto"] = grouped["cy_n"].shift(1)
    df["prev_timestamp_sec_auto"] = grouped["timestamp_sec"].shift(1)

    dx = df["cx_n"] - df["prev_cx_n_auto"]
    dy = df["cy_n"] - df["prev_cy_n_auto"]
    displacement = np.sqrt(dx * dx + dy * dy)
    delta_frame = df["frame_index"] - df["prev_frame_index_auto"]
    delta_seconds = df["timestamp_sec"] - df["prev_timestamp_sec_auto"]
    adjacent_pair_valid = delta_frame.eq(1) & delta_seconds.gt(0)
    df["adjacent_motion_pair_valid_auto"] = adjacent_pair_valid
    df["step_speed_n_per_second_auto"] = (
        displacement.div(delta_seconds).where(adjacent_pair_valid, 0.0)
    )

    frame_gap_prev = (df["frame_index"] - df["prev_frame_index_auto"]).abs()
    frame_gap_next = (df["next_frame_index_auto"] - df["frame_index"]).abs()
    prev_changed = df["prev_behavior_auto"].notna() & (
        df["prev_behavior_auto"].astype(str) != df["behavior"].astype(str)
    )
    next_changed = df["next_behavior_auto"].notna() & (
        df["next_behavior_auto"].astype(str) != df["behavior"].astype(str)
    )
    df["label_boundary_auto"] = (prev_changed & frame_gap_prev.le(boundary_frame_gap)) | (
        next_changed & frame_gap_next.le(boundary_frame_gap)
    )

    # Review/debug rolling evidence remains inside one native temporal unit.
    def rolling_motion(s: pd.Series) -> pd.Series:
        return s.rolling(window=2 * window_radius + 1, min_periods=1, center=True).mean()

    df["window_speed_mean_n_per_second_auto"] = grouped[
        "step_speed_n_per_second_auto"
    ].transform(rolling_motion)
    df["window_speed_max_n_per_second_auto"] = grouped[
        "step_speed_n_per_second_auto"
    ].transform(
        lambda s: s.rolling(window=2 * window_radius + 1, min_periods=1, center=True).max()
    )

    df.sort_index(inplace=True)


def infer_row_review_attrs(
    row: pd.Series,
    *,
    motion_low_threshold: float,
    motion_strong_threshold: float,
) -> pd.Series:
    behavior = norm(row.get("behavior"))
    reasons: list[str] = []
    groups: list[str] = []
    review_required = False
    priority = 99
    strength = "strong"
    suggested_action = "main_train"

    if not bool_value(row.get("bbox_valid", True)):
        reasons.append("invalid_bbox")
        groups.append("bbox_quality")
        strength = "weak"
        review_required = True
        priority = min(priority, 1)
        suggested_action = "exclude_until_fixed"

    if bool_value(row.get("label_boundary_auto", False)):
        reasons.append("near_label_transition_boundary")
        groups.append(group_for_behavior(behavior))
        strength = min_strength(strength, "boundary")
        review_required = True
        priority = min(priority, 2)
        suggested_action = "exclude_main_or_low_weight"

    if behavior in ROI_DOMINANT:
        groups.append("roi_based")
        target = ROI_BEHAVIOR_TO_TARGET[behavior]
        target_class = norm(row.get("roi_target_class")) or target
        contact = bool_value(row.get("roi_target_contact"))
        near = bool_value(row.get("roi_target_near"))
        available = bool_value(row.get("roi_target_available"))
        center_inside = bool_value(row.get("roi_target_center_inside"))

        if target_class and target_class != target:
            reasons.append(f"target_roi_class_mismatch_expected_{target}_got_{target_class}")
            review_required = True
            priority = min(priority, 1)
            strength = min_strength(strength, "weak")
        if contact:
            reasons.append("target_roi_contact")
            strength = min_strength(strength, "strong")
        elif near or center_inside:
            reasons.append("target_roi_near_but_no_contact")
            review_required = True
            priority = min(priority, 2)
            strength = min_strength(strength, "medium")
            suggested_action = "main_train_medium_weight_after_review"
        elif available:
            reasons.append("target_roi_far_despite_roi_dominant_label")
            review_required = True
            priority = min(priority, 1)
            strength = min_strength(strength, "weak")
            suggested_action = "review_or_exclude_roi_training"
        else:
            reasons.append("target_roi_unavailable")
            review_required = True
            priority = min(priority, 1)
            strength = min_strength(strength, "weak")
            suggested_action = "review_or_exclude_roi_training"

    if behavior in AGGRESSION_SOCIAL:
        groups.append("aggression_social")
        partners = safe_float(row.get("interaction_partner_count"), 0.0)
        social_quality = norm(row.get("social_feature_quality"))
        motion = safe_float(
            row.get("window_speed_mean_n_per_second_auto"),
            np.nan,
        )

        review_required = True
        priority = min(priority, 1 if behavior == "fight" else 2)
        if behavior == "fight":
            if partners <= 0 or social_quality in {
                "missing_context",
                "missing_partner",
                "unknown",
                "",
            }:
                reasons.append("fight_without_clear_partner_context")
                strength = min_strength(strength, "weak")
            elif not np.isnan(motion) and motion >= motion_strong_threshold:
                reasons.append("fight_with_partner_and_motion_evidence")
                strength = min_strength(strength, "medium")
            else:
                reasons.append("fight_event_but_window_may_show_pause")
                strength = min_strength(strength, "weak")
            suggested_action = "manual_strength_required"
        else:
            if partners <= 0 or social_quality in {
                "missing_context",
                "missing_partner",
                "unknown",
                "",
            }:
                reasons.append("social_nose_without_clear_partner_context")
                strength = min_strength(strength, "weak")
            else:
                reasons.append("social_nose_requires_intent_review")
                strength = min_strength(strength, "medium")
            suggested_action = "manual_strength_required"

    if behavior in ROI_AMBIGUOUS_NON_TARGET:
        near_roi = any(
            bool_value(row.get(col))
            for col in [
                "roi_feeder_near",
                "roi_feeder_contact",
                "roi_drinker_near",
                "roi_drinker_contact",
                "roi_toy_near",
                "roi_toy_contact",
            ]
        )
        if near_roi:
            groups.append("roi_based")
            reasons.append(f"{behavior}_near_roi_possible_eat_drink_toy_confusion")
            review_required = True
            priority = min(priority, 3)
            strength = min_strength(strength, "medium")
            suggested_action = "review_if_class_balance_allows"

    if behavior in MOTION_STATE:
        groups.append("motion_state")
        motion = safe_float(
            row.get("window_speed_mean_n_per_second_auto"),
            np.nan,
        )
        if behavior == "move":
            if np.isnan(motion):
                reasons.append("move_motion_unavailable")
                strength = min_strength(strength, "medium")
            elif motion < motion_low_threshold:
                reasons.append("move_label_with_low_window_displacement")
                review_required = True
                priority = min(priority, 2)
                strength = min_strength(strength, "weak")
                suggested_action = "review_or_low_weight"
            elif motion >= motion_strong_threshold:
                reasons.append("move_with_clear_displacement")
                strength = min_strength(strength, "strong")
            else:
                reasons.append("move_with_medium_displacement")
                strength = min_strength(strength, "medium")
        elif behavior == "stand" and not bool_value(row.get("label_boundary_auto", False)):
            strength = min_strength(strength, "medium")
            reasons.append("stand_is_posture_not_intent")
        elif behavior == "explore":
            strength = min_strength(strength, "medium")
            reasons.append("explore_is_broad_fallback_like_class")

    if behavior in POSTURE_STATE:
        groups.append("posture")
        if behavior == "sitting" and not bool_value(row.get("label_boundary_auto", False)):
            strength = min_strength(strength, "medium")
            reasons.append("sitting_is_intermediate_posture")

    if not groups:
        groups.append("general")
    if not reasons:
        reasons.append("auto_clean")

    priority = priority if priority != 99 else 5
    review_reason = ";".join(unique_keep_order(reasons))
    ambiguity_group = "+".join(unique_keep_order(groups))

    sample_weight = weight_for_strength(strength)
    if suggested_action.startswith("exclude"):
        sample_weight = 0.0

    return pd.Series(
        {
            "ambiguity_group_auto": ambiguity_group,
            "label_strength_auto": strength,
            "review_required_auto": bool(review_required),
            "review_priority": int(priority),
            "review_reason_auto": review_reason,
            "training_action_suggested": suggested_action,
            "sample_weight_suggested": float(sample_weight),
            "roi_consistency_status_auto": roi_consistency(row, behavior),
        }
    )


def select_review_rows(df: pd.DataFrame, *, scope: str) -> pd.DataFrame:
    if scope == "all_rows":
        return df.copy()

    mask = df["review_required_auto"].astype(bool)
    if scope == "roi":
        mask &= df["ambiguity_group_auto"].astype(str).str.contains("roi_based", regex=False)
    elif scope == "critical":
        mask &= df["review_priority"].le(3)
    elif scope == "all_review":
        pass
    else:
        raise ValueError(scope)

    return df[mask].copy()


def make_review_template(df: pd.DataFrame) -> pd.DataFrame:
    first_cols = [
        "review_key",
        "review_priority",
        "ambiguity_group_auto",
        "label_strength_auto",
        "review_reason_auto",
        "training_action_suggested",
        "sample_weight_suggested",
        "roi_consistency_status_auto",
        "source_type",
        "dataset_id",
        "video_key",
        "frame_uid",
        "frame_index",
        "relative_frame_index",
        "track_id",
        "track_label",
        "pig_id",
        "behavior",
        "prev_behavior_auto",
        "next_behavior_auto",
        "label_boundary_auto",
        "window_speed_mean_n_per_second_auto",
        "window_speed_max_n_per_second_auto",
        "roi_target_class",
        "roi_target_min_dist_n",
        "roi_target_contact",
        "roi_target_near",
        "roi_context_quality",
        "interaction_partner_count",
        "interaction_partner_ids",
        "social_feature_quality",
        "local_context_quality",
        "crop_path",
        "image_name",
        "x1",
        "y1",
        "x2",
        "y2",
    ]
    manual_cols = [
        "manual_review_decision",
        "manual_label_strength",
        "manual_corrected_behavior",
        "manual_ambiguity_group",
        "manual_training_action",
        "manual_sample_weight",
        "manual_note",
    ]
    cols = [c for c in first_cols if c in df.columns]
    out = df[cols].copy()
    for col in manual_cols:
        out[col] = ""
    return out


def build_audit(
    annotated: pd.DataFrame,
    template: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "input_csv": str(args.input_csv),
        "rows": int(len(annotated)),
        "review_rows": int(len(template)),
        "scope": args.scope,
        "review_required_auto": value_counts(annotated, "review_required_auto"),
        "review_priority": value_counts(annotated, "review_priority"),
        "label_strength_auto": value_counts(annotated, "label_strength_auto"),
        "ambiguity_group_auto": value_counts(annotated, "ambiguity_group_auto"),
        "roi_consistency_status_auto": value_counts(annotated, "roi_consistency_status_auto"),
        "behavior": value_counts(annotated, "behavior"),
        "review_behavior": value_counts(template, "behavior"),
        "review_reasons_top30": value_counts(template, "review_reason_auto", top=30),
    }


def roi_consistency(row: pd.Series, behavior: str) -> str:
    if behavior not in ROI_DOMINANT:
        return "not_required"
    if bool_value(row.get("roi_target_contact")):
        return "target_roi_contact"
    if bool_value(row.get("roi_target_near")) or bool_value(row.get("roi_target_center_inside")):
        return "target_roi_near_no_contact"
    if bool_value(row.get("roi_target_available")):
        return "target_roi_far"
    return "target_roi_unavailable"


def group_for_behavior(behavior: str) -> str:
    if behavior in AGGRESSION_SOCIAL:
        return "aggression_social"
    if behavior in ROI_DOMINANT or behavior in ROI_AMBIGUOUS_NON_TARGET:
        return "roi_based"
    if behavior in MOTION_STATE:
        return "motion_state"
    if behavior in POSTURE_STATE:
        return "posture"
    return "general"


def min_strength(a: str, b: str) -> str:
    order = {"strong": 0, "medium": 1, "weak": 2, "boundary": 3}
    return a if order.get(a, 0) >= order.get(b, 0) else b


def weight_for_strength(strength: str) -> float:
    return {"strong": 1.0, "medium": 0.75, "weak": 0.35, "boundary": 0.0}.get(strength, 0.5)


def bool_value(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if pd.isna(v):
        return False
    return str(v).strip().lower() in {"true", "1", "yes", "y"}


def safe_float(v: Any, default: float) -> float:
    try:
        if pd.isna(v):
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def norm(v: Any) -> str:
    if pd.isna(v):
        return ""
    return str(v).strip().lower()


def unique_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def value_counts(df: pd.DataFrame, col: str, top: int | None = None) -> dict[str, int]:
    if col not in df.columns:
        return {}
    counts = df[col].fillna("").astype(str).value_counts(dropna=False)
    if top is not None:
        counts = counts.head(top)
    return {str(k): int(v) for k, v in counts.items()}


if __name__ == "__main__":
    main()
