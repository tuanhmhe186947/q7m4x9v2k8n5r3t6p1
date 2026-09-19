from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import VALID_BEHAVIORS

REQUIRED_COLUMNS = {
    "video_final",
    "day_final",
    "group_id",
    "sample_id",
    "img_name",
    "frames",
    "pig_id",
    "x1",
    "y1",
    "x2",
    "y2",
    "behavior",
    "hidden",
    "center_frame_from_img",
    "center_frame_final",
    "frame_mismatch",
    "match_source",
}


def parse_frames(value: object) -> list[int]:
    if value is None or pd.isna(value):
        return []
    frames: list[int] = []
    for part in str(value).replace(",", "|").split("|"):
        part = part.strip()
        if not part:
            continue
        try:
            frames.append(int(float(part)))
        except ValueError:
            return []
    return frames


def validate_legacy_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    df = df.copy()
    df["parsed_frames"] = df["frames"].apply(parse_frames)
    df["behavior"] = df["behavior"].astype(str).str.strip()
    df["row_rejection_reason"] = ""

    invalid_behavior = ~df["behavior"].isin(VALID_BEHAVIORS)
    invalid_frames = df["parsed_frames"].apply(len).eq(0)
    invalid_bbox = pd.Series(False, index=df.index)
    for col in ["x1", "y1", "x2", "y2"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        invalid_bbox |= df[col].isna()
    invalid_bbox |= df["x2"].le(df["x1"]) | df["y2"].le(df["y1"])

    df.loc[invalid_behavior, "row_rejection_reason"] += "unknown_behavior;"
    df.loc[invalid_frames, "row_rejection_reason"] += "invalid_frames;"
    df.loc[invalid_bbox, "row_rejection_reason"] += "invalid_bbox;"

    rejected = df[df["row_rejection_reason"].ne("")].copy()
    accepted = df[df["row_rejection_reason"].eq("")].copy()
    return accepted, rejected


def load_legacy_csv(input_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    return validate_legacy_dataframe(pd.read_csv(input_csv))

