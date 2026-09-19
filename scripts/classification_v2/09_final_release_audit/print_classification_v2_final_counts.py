from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path("outputs/classification_v2")


def vc(df: pd.DataFrame, col: str) -> dict[str, int]:
    return {str(k): int(v) for k, v in df[col].fillna("<NA>").astype(str).value_counts().items()}


def main() -> None:
    reviewed = pd.read_csv(
        ROOT / "review_policy/reviewed_frame_features.csv",
        usecols=["behavior", "behavior_before_review", "source_type"],
        low_memory=False,
    )
    windows = pd.read_csv(
        ROOT / "sequence_features_reviewed/sequence_window_manifest.csv",
        usecols=[
            "source_type",
            "sequence_label_status",
            "window_valid_for_main_train",
            "review_excluded_frame_count_window",
        ],
        low_memory=False,
    )
    print("label_distribution_before_review =", vc(reviewed, "behavior_before_review"))
    print("label_distribution_after_review  =", vc(reviewed, "behavior"))
    print("reviewed_source_distribution    =", vc(reviewed, "source_type"))
    print("window_source_distribution      =", vc(windows, "source_type"))
    print("sequence_label_status           =", vc(windows, "sequence_label_status"))
    print("window_valid_for_main_train     =", vc(windows, "window_valid_for_main_train"))
    print("review_excluded_frame_count_win =", vc(windows, "review_excluded_frame_count_window"))


if __name__ == "__main__":
    main()
