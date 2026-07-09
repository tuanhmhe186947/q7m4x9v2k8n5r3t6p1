from __future__ import annotations

from pathlib import Path

import pandas as pd

INTERVALS_PATH = Path(r"outputs\classification_v2\sequence_features\temporal_label_intervals.csv")


def main() -> None:
    intervals = pd.read_csv(INTERVALS_PATH, low_memory=False)
    cvat = intervals[intervals["source_type"].astype(str).eq("cvat_tracking_xml")].copy()

    print("CVAT intervals =", len(cvat))
    print("\nCVAT temporal_consistency_status:")
    print(cvat["temporal_consistency_status"].value_counts(dropna=False).to_string())
    print("\nCVAT behavior_temporal_final:")
    print(cvat["behavior_temporal_final"].fillna("").value_counts(dropna=False).head(20).to_string())

    if "anchor_behavior_in_interval" in cvat.columns:
        mismatch = cvat[
            cvat["behavior_temporal_final"].fillna("").astype(str)
            != cvat["anchor_behavior_in_interval"].fillna("").astype(str)
        ]
        print("\nfinal != anchor mismatch rows =", len(mismatch))

    bad = cvat[
        cvat["temporal_consistency_status"].astype(str).eq("mixed")
        & cvat["behavior_temporal_final"].fillna("").astype(str).ne("")
    ]
    print("mixed non-empty behavior =", len(bad))

    legacy = intervals[intervals["source_type"].astype(str).eq("legacy_recovered")].copy()
    print("\nLegacy temporal_consistency_status:")
    print(legacy["temporal_consistency_status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
