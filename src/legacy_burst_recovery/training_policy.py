from __future__ import annotations

from pathlib import Path

import pandas as pd

MANUAL_REVIEW_TEMPLATE_COLUMNS = [
    "tracklet_id",
    "group_id",
    "sample_id",
    "pig_id",
    "behavior",
    "frame_indices_review",
    "tracking_status_summary",
    "qa_status",
    "qa_notes",
    "debug_visuals_dir",
    "manual_decision",
    "manual_reason",
    "include_in_training",
]

TRACKLET_POLICY_COLUMNS = [
    "auto_qa_status",
    "manual_decision",
    "manual_reason",
    "include_in_training",
    "training_tier",
]

MANUAL_REVIEW_APPLY_AUDIT_COLUMNS = [
    "manual_row_index",
    "match_key_used",
    "manual_sample_id",
    "manual_group_id",
    "manual_pig_id",
    "matched_tracklet_id",
    "matched_sample_id",
    "matched_group_id",
    "matched_pig_id",
    "manual_decision",
    "applied",
    "reason",
]


def _coerce_bool(value: object) -> bool | None:
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _join_unique(values: pd.Series, sep: str = ";") -> str:
    seen: list[str] = []
    for value in values.fillna("").astype(str):
        item = value.strip()
        if item and item not in seen:
            seen.append(item)
    return sep.join(seen)


def _max_consecutive_true(values: list[bool]) -> int:
    longest = 0
    current = 0
    for value in values:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _key_value(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def load_manual_review_decisions(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(columns=MANUAL_REVIEW_TEMPLATE_COLUMNS)

    df = pd.read_csv(path)
    has_sample_key = "sample_id" in df.columns
    has_group_pig_key = {"group_id", "pig_id"}.issubset(df.columns)
    has_tracklet_key = "tracklet_id" in df.columns
    if not (has_sample_key or has_group_pig_key or has_tracklet_key):
        raise ValueError("--manual-review-csv must include sample_id, group_id+pig_id, or tracklet_id columns")

    for column in MANUAL_REVIEW_TEMPLATE_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA

    df = df.copy()
    for column in ["tracklet_id", "group_id", "sample_id", "pig_id"]:
        df[column] = df[column].map(_key_value)
    df["manual_decision"] = df["manual_decision"].fillna("").astype(str).str.strip().str.lower()
    df["manual_reason"] = df["manual_reason"].fillna("").astype(str)
    df["include_in_training"] = df["include_in_training"].map(_coerce_bool)
    return df


def _compute_tracklet_policy(group: pd.DataFrame) -> dict[str, object]:
    ordered = group.sort_values("frame_index").copy()
    tracking_status = ordered["tracking_status"].fillna("").astype(str)
    qa_status = ordered["qa_status"].fillna("").astype(str)
    is_interpolated = ordered["is_interpolated"].fillna(False).astype(bool)
    det_confidence = (
        pd.to_numeric(ordered["det_confidence"], errors="coerce")
        if "det_confidence" in ordered
        else pd.Series([pd.NA] * len(ordered), index=ordered.index)
    )
    track_confidence = (
        pd.to_numeric(ordered["track_confidence"], errors="coerce").fillna(0.0)
        if "track_confidence" in ordered
        else pd.Series([0.0] * len(ordered), index=ordered.index)
    )
    legacy_gt_mode = (
        str(ordered["legacy_gt_mode"].dropna().iloc[0])
        if "legacy_gt_mode" in ordered and ordered["legacy_gt_mode"].notna().any()
        else "single_anchor"
    )
    legacy_gt_support_series = (
        pd.to_numeric(ordered["legacy_gt_support_count"], errors="coerce")
        if "legacy_gt_support_count" in ordered
        else pd.Series([pd.NA] * len(ordered), index=ordered.index)
    )
    legacy_gt_support_count = (
        int(legacy_gt_support_series.dropna().max()) if legacy_gt_support_series.notna().any() else 0
    )
    id_switch_risk_score = (
        pd.to_numeric(ordered["id_switch_risk_score"], errors="coerce").fillna(0.0)
        if "id_switch_risk_score" in ordered
        else pd.Series([0.0] * len(ordered), index=ordered.index)
    )
    detector_disagrees = (
        ordered["detector_disagrees_with_legacy_gt"].fillna(False).astype(bool)
        if "detector_disagrees_with_legacy_gt" in ordered
        else pd.Series(False, index=ordered.index)
    )

    has_failed_frames = tracking_status.eq("failed").any()
    has_needs_review_frames = qa_status.isin(["review", "needs_review"]).any()
    has_low_confidence_frames = tracking_status.eq("low_confidence").any()
    has_id_switch_risk_frames = bool(
        (
            (tracking_status.eq("low_confidence") | qa_status.isin(["review", "needs_review"]))
            & id_switch_risk_score.ge(0.5)
        ).any()
    )
    max_consecutive_interpolated = _max_consecutive_true(is_interpolated.tolist())
    det_confidence_coverage = float(det_confidence.notna().mean()) if len(ordered) else 0.0
    track_confidence_min = float(track_confidence.min()) if len(ordered) else 0.0

    if legacy_gt_mode == "multi_anchor":
        has_missing_legacy_gt_support = legacy_gt_support_count < 3
        include_in_training = all(
            [
                not has_failed_frames,
                not has_needs_review_frames,
                not has_id_switch_risk_frames,
                not has_missing_legacy_gt_support,
                max_consecutive_interpolated <= 2,
                track_confidence_min >= 0.50,
            ]
        )
    else:
        has_missing_legacy_gt_support = False
        include_in_training = all(
            [
                not has_failed_frames,
                not has_needs_review_frames,
                not has_low_confidence_frames,
                max_consecutive_interpolated <= 2,
                det_confidence_coverage >= 0.90,
                track_confidence_min >= 0.50,
            ]
        )
    auto_qa_status = "ok" if include_in_training else "review"

    if legacy_gt_mode == "multi_anchor" and has_missing_legacy_gt_support:
        tracking_status_summary = "legacy_gt_missing"
        training_tier = "legacy_gt_review"
    elif max_consecutive_interpolated > 2:
        tracking_status_summary = "long_occlusion"
        training_tier = "hard_occlusion"
    elif has_failed_frames:
        tracking_status_summary = "failed_frames"
        training_tier = "review"
    elif legacy_gt_mode == "multi_anchor" and has_id_switch_risk_frames:
        tracking_status_summary = "id_switch_risk"
        training_tier = "review"
    elif has_low_confidence_frames:
        tracking_status_summary = "low_confidence"
        training_tier = "review"
    elif has_needs_review_frames:
        tracking_status_summary = "needs_review"
        training_tier = "review"
    else:
        tracking_status_summary = "ok"
        training_tier = "clean"

    review_mask = (
        qa_status.isin(["review", "needs_review"])
        | is_interpolated
        | tracking_status.isin(["failed", "low_confidence"])
    )
    frame_indices_review = "|".join(map(str, ordered.loc[review_mask, "frame_index"].astype(int).tolist()))
    notes = _join_unique(ordered["qa_notes"]) if "qa_notes" in ordered else ""
    if (
        legacy_gt_mode == "multi_anchor"
        and detector_disagrees.any()
        and "detector_disagrees_with_legacy_gt" not in notes
    ):
        notes = ";".join([part for part in [notes, "detector_disagrees_with_legacy_gt"] if part])
    if legacy_gt_mode == "multi_anchor" and has_missing_legacy_gt_support and "missing_legacy_gt_support" not in notes:
        notes = ";".join([part for part in [notes, "missing_legacy_gt_support"] if part])

    return {
        "tracklet_id": str(ordered["tracklet_id"].iloc[0]),
        "group_id": _key_value(ordered["group_id"].iloc[0]) if "group_id" in ordered else "",
        "sample_id": _key_value(ordered["sample_id"].iloc[0]) if "sample_id" in ordered else "",
        "pig_id": _key_value(ordered["pig_id"].iloc[0]) if "pig_id" in ordered else "",
        "frame_indices_review": frame_indices_review,
        "tracking_status_summary": tracking_status_summary,
        "auto_qa_status": auto_qa_status,
        "include_in_training": bool(include_in_training),
        "training_tier": training_tier,
        "qa_notes": notes,
        "max_consecutive_interpolated_frames": int(max_consecutive_interpolated),
        "det_confidence_coverage": det_confidence_coverage,
        "track_confidence_min": track_confidence_min,
    }


def _manual_review_matches(
    policy_df: pd.DataFrame,
    manual_review_df: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manual_df = (
        manual_review_df.copy()
        if manual_review_df is not None
        else pd.DataFrame(columns=MANUAL_REVIEW_TEMPLATE_COLUMNS)
    )
    decision_columns = ["tracklet_id", "manual_decision", "manual_reason", "manual_include_in_training"]
    if manual_df.empty:
        return pd.DataFrame(columns=decision_columns), pd.DataFrame(columns=MANUAL_REVIEW_APPLY_AUDIT_COLUMNS)

    for column in MANUAL_REVIEW_TEMPLATE_COLUMNS:
        if column not in manual_df.columns:
            manual_df[column] = pd.NA
    for column in ["tracklet_id", "group_id", "sample_id", "pig_id"]:
        manual_df[column] = manual_df[column].map(_key_value)
    manual_df["manual_decision"] = manual_df["manual_decision"].fillna("").astype(str).str.strip().str.lower()
    manual_df["manual_reason"] = manual_df["manual_reason"].fillna("").astype(str)
    manual_df["include_in_training"] = manual_df["include_in_training"].map(_coerce_bool)

    by_sample_id = {
        _key_value(row["sample_id"]): row
        for _, row in policy_df.iterrows()
        if _key_value(row.get("sample_id", ""))
    }
    by_group_pig = {
        (_key_value(row["group_id"]), _key_value(row["pig_id"])): row
        for _, row in policy_df.iterrows()
        if _key_value(row.get("group_id", "")) and _key_value(row.get("pig_id", ""))
    }
    by_tracklet_id = {
        _key_value(row["tracklet_id"]): row
        for _, row in policy_df.iterrows()
        if _key_value(row.get("tracklet_id", ""))
    }

    audit_rows: list[dict[str, object]] = []
    matched_rows: list[dict[str, object]] = []
    for manual_row_index, manual_row in manual_df.iterrows():
        manual_sample_id = _key_value(manual_row.get("sample_id", ""))
        manual_group_id = _key_value(manual_row.get("group_id", ""))
        manual_pig_id = _key_value(manual_row.get("pig_id", ""))
        manual_tracklet_id = _key_value(manual_row.get("tracklet_id", ""))
        manual_decision = _key_value(manual_row.get("manual_decision", "")).lower()

        match_key_used = ""
        matched = None
        reason = ""
        if manual_sample_id:
            match_key_used = "sample_id"
            matched = by_sample_id.get(manual_sample_id)
            if matched is None:
                reason = "no_match_for_sample_id"
        elif manual_group_id and manual_pig_id:
            match_key_used = "group_id+pig_id"
            matched = by_group_pig.get((manual_group_id, manual_pig_id))
            if matched is None:
                reason = "no_match_for_group_id_pig_id"
        elif manual_tracklet_id:
            match_key_used = "tracklet_id"
            matched = by_tracklet_id.get(manual_tracklet_id)
            if matched is None:
                reason = "no_match_for_tracklet_id"
        else:
            match_key_used = "none"
            reason = "missing_match_key"

        audit_match = matched
        conflict_columns: list[str] = []
        if matched is not None and match_key_used == "tracklet_id":
            for column, manual_value in [
                ("sample_id", manual_sample_id),
                ("group_id", manual_group_id),
                ("pig_id", manual_pig_id),
            ]:
                matched_value = _key_value(matched.get(column, ""))
                if manual_value and matched_value and manual_value != matched_value:
                    conflict_columns.append(column)
            if conflict_columns:
                reason = "stable_identifier_conflict:" + "|".join(conflict_columns)
                matched = None

        applied = matched is not None
        if applied:
            reason = "applied"
            matched_rows.append(
                {
                    "tracklet_id": _key_value(matched["tracklet_id"]),
                    "manual_decision": manual_decision,
                    "manual_reason": _key_value(manual_row.get("manual_reason", "")),
                    "manual_include_in_training": manual_row.get("include_in_training"),
                }
            )

        audit_rows.append(
            {
                "manual_row_index": int(manual_row_index),
                "match_key_used": match_key_used,
                "manual_sample_id": manual_sample_id,
                "manual_group_id": manual_group_id,
                "manual_pig_id": manual_pig_id,
                "matched_tracklet_id": "" if audit_match is None else _key_value(audit_match.get("tracklet_id", "")),
                "matched_sample_id": "" if audit_match is None else _key_value(audit_match.get("sample_id", "")),
                "matched_group_id": "" if audit_match is None else _key_value(audit_match.get("group_id", "")),
                "matched_pig_id": "" if audit_match is None else _key_value(audit_match.get("pig_id", "")),
                "manual_decision": manual_decision,
                "applied": applied,
                "reason": reason,
            }
        )

    matched_df = pd.DataFrame(matched_rows, columns=decision_columns)
    if not matched_df.empty:
        matched_df = matched_df.drop_duplicates(subset=["tracklet_id"], keep="last")
    return matched_df, pd.DataFrame(audit_rows, columns=MANUAL_REVIEW_APPLY_AUDIT_COLUMNS)


def apply_training_policy(
    dense_df: pd.DataFrame,
    manual_review_df: pd.DataFrame | None = None,
    *,
    return_manual_review_audit: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    if dense_df.empty:
        enriched = dense_df.copy()
        for column in ["tracking_status_summary", "frame_indices_review", *TRACKLET_POLICY_COLUMNS]:
            if column not in enriched.columns:
                enriched[column] = pd.Series(dtype="object")
        if return_manual_review_audit:
            return enriched, pd.DataFrame(columns=MANUAL_REVIEW_APPLY_AUDIT_COLUMNS)
        return enriched

    policy_df = pd.DataFrame(
        _compute_tracklet_policy(group) for _, group in dense_df.groupby("tracklet_id", sort=False)
    )

    manual_df, manual_review_apply_audit = _manual_review_matches(policy_df, manual_review_df)

    policy_df = policy_df.merge(manual_df, on="tracklet_id", how="left")
    policy_df["manual_decision"] = policy_df["manual_decision"].fillna("").astype(str).str.strip().str.lower()
    policy_df["manual_reason"] = policy_df["manual_reason"].fillna("").astype(str)

    for idx, row in policy_df.iterrows():
        decision = row["manual_decision"]
        if decision == "accept":
            policy_df.at[idx, "include_in_training"] = True
            policy_df.at[idx, "training_tier"] = "clean"
        elif decision == "accept_with_note":
            policy_df.at[idx, "include_in_training"] = True
            policy_df.at[idx, "training_tier"] = "warning"
        elif decision == "reject":
            policy_df.at[idx, "include_in_training"] = False
            policy_df.at[idx, "training_tier"] = "rejected"
        elif decision == "hard_occlusion":
            policy_df.at[idx, "include_in_training"] = False
            policy_df.at[idx, "training_tier"] = "hard_occlusion"
        else:
            manual_include = _coerce_bool(row["manual_include_in_training"])
            if manual_include is not None:
                policy_df.at[idx, "include_in_training"] = manual_include
                if manual_include and row["training_tier"] == "review":
                    policy_df.at[idx, "training_tier"] = "clean"

    enriched = dense_df.copy()
    enriched["tracklet_id"] = enriched["tracklet_id"].astype(str)
    enriched = enriched.merge(
        policy_df[
            [
                "tracklet_id",
                "frame_indices_review",
                "tracking_status_summary",
                "auto_qa_status",
                "manual_decision",
                "manual_reason",
                "include_in_training",
                "training_tier",
                "qa_notes",
            ]
        ].rename(columns={"qa_notes": "policy_qa_notes"}),
        on="tracklet_id",
        how="left",
    )

    long_occlusion_mask = enriched["tracking_status_summary"].eq("long_occlusion")
    enriched.loc[long_occlusion_mask, "qa_status"] = "review"
    legacy_gt_review_mask = enriched["training_tier"].eq("legacy_gt_review")
    enriched.loc[legacy_gt_review_mask, "qa_status"] = "review"
    existing_notes = enriched["qa_notes"].fillna("").astype(str)
    note_mask = long_occlusion_mask & existing_notes.eq("")
    append_mask = (
        long_occlusion_mask & ~existing_notes.eq("") & ~existing_notes.str.contains("long_occlusion", regex=False)
    )
    enriched.loc[note_mask, "qa_notes"] = "long_occlusion"
    enriched.loc[append_mask, "qa_notes"] = existing_notes[append_mask] + ";long_occlusion"
    policy_notes = enriched["policy_qa_notes"].fillna("").astype(str)
    existing_notes = enriched["qa_notes"].fillna("").astype(str)
    add_policy_mask = pd.Series(
        [
            bool(policy and policy not in existing)
            for policy, existing in zip(policy_notes, existing_notes, strict=False)
        ],
        index=enriched.index,
    )
    empty_notes = add_policy_mask & existing_notes.eq("")
    append_policy = add_policy_mask & ~existing_notes.eq("")
    enriched.loc[empty_notes, "qa_notes"] = policy_notes[empty_notes]
    enriched.loc[append_policy, "qa_notes"] = existing_notes[append_policy] + ";" + policy_notes[append_policy]
    enriched = enriched.drop(columns=["policy_qa_notes"])
    if return_manual_review_audit:
        return enriched, manual_review_apply_audit
    return enriched


def build_manual_review_template(dense_df: pd.DataFrame, output_root: Path) -> pd.DataFrame:
    if dense_df.empty:
        return pd.DataFrame(columns=MANUAL_REVIEW_TEMPLATE_COLUMNS)

    rows: list[dict[str, object]] = []
    for _, group in dense_df.groupby("tracklet_id", sort=False):
        ordered = group.sort_values("frame_index")
        first = ordered.iloc[0]
        rows.append(
            {
                "tracklet_id": str(first["tracklet_id"]),
                "group_id": first.get("group_id", ""),
                "sample_id": first.get("sample_id", ""),
                "pig_id": first.get("pig_id", ""),
                "behavior": first.get("behavior", ""),
                "frame_indices_review": first.get("frame_indices_review", ""),
                "tracking_status_summary": first.get("tracking_status_summary", ""),
                "qa_status": first.get("auto_qa_status", first.get("qa_status", "")),
                "qa_notes": _join_unique(ordered["qa_notes"]) if "qa_notes" in ordered else "",
                "debug_visuals_dir": str(output_root / "debug_visuals" / str(first["tracklet_id"])),
                "manual_decision": first.get("manual_decision", ""),
                "manual_reason": first.get("manual_reason", ""),
                "include_in_training": first.get("include_in_training", False),
            }
        )

    return pd.DataFrame(rows, columns=MANUAL_REVIEW_TEMPLATE_COLUMNS)
