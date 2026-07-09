import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

VIDEO_KEY = "Pigs281119_000085_30fps"
PIG_ID = "ID_4"
ANCHOR = 1020
END = 1025

xml_candidates = [
    Path(r"data\annotations\classification\Pigs281119_000085_30fps.xml"),
    Path(r"data\annotations\classification\Tracking_annotation_Pigs281119_000085_30fps.xml"),
]

xml_path = None
for p in xml_candidates:
    if p.exists():
        xml_path = p
        break

print("=== XML GT CHECK ===")
print("xml_path =", xml_path)

xml_anchor_behavior = None

if xml_path is not None:
    root = ET.parse(xml_path).getroot()

    rows = []
    for tr in root.findall(".//track"):
        for box in tr.findall("box"):
            f = int(box.attrib["frame"])
            if not (ANCHOR - 2 <= f <= END + 1):
                continue
            attrs = {a.attrib.get("name"): (a.text or "") for a in box.findall("attribute")}
            if attrs.get("ID") == PIG_ID:
                rows.append(
                    {
                        "xml_track_id": tr.attrib.get("id"),
                        "track_label": tr.attrib.get("label"),
                        "frame": f,
                        "behavior": attrs.get("Behavior"),
                        "hidden": attrs.get("Hidden"),
                        "pig_id": attrs.get("ID"),
                    }
                )

    xdf = pd.DataFrame(rows).sort_values("frame")
    print(xdf.to_string(index=False))

    a = xdf[xdf["frame"].eq(ANCHOR)]
    if not a.empty:
        xml_anchor_behavior = str(a.iloc[0]["behavior"])
        print("\nXML anchor behavior =", xml_anchor_behavior)
else:
    print("[WARN] Local XML file not found. Put latest XML in data\\annotations\\classification first.")


def read_csv(path):
    p = Path(path)
    if not p.exists():
        print("[MISSING]", p)
        return None
    return pd.read_csv(p, low_memory=False)


print("\n=== ENHANCED CSV CHECK ===")
enh = read_csv(r"outputs\classification_v2\frame_features\spatiotemporal_frame_features_enhanced.csv")
enh_anchor_behavior = None

if enh is not None:
    q = enh[
        enh["source_type"].astype(str).eq("cvat_tracking_xml")
        & enh["video_key"].astype(str).eq(VIDEO_KEY)
        & enh["pig_id"].astype(str).eq(PIG_ID)
    ].copy()

    for c in ["frame_index", "label_window_start", "label_window_end", "label_anchor_frame_index"]:
        if c in q.columns:
            q[c] = pd.to_numeric(q[c], errors="coerce")

    show = q[
        (q["frame_index"].between(ANCHOR - 2, END + 1))
        | (q["label_window_start"].eq(ANCHOR))
        | (q["label_anchor_frame_index"].eq(ANCHOR))
    ].copy()

    cols = [
        "source_type",
        "video_key",
        "pig_id",
        "track_id",
        "track_label",
        "frame_index",
        "behavior",
        "label_anchor_frame_index",
        "label_window_start",
        "label_window_end",
        "temporal_label_mode",
        "temporal_unit_key",
    ]
    cols = [c for c in cols if c in show.columns]
    print(show[cols].sort_values(["frame_index"]).to_string(index=False))

    a = show[show["frame_index"].eq(ANCHOR)]
    if not a.empty:
        enh_anchor_behavior = str(a.iloc[0]["behavior"])
        print("\nEnhanced anchor behavior =", enh_anchor_behavior)


print("\n=== TEMPORAL INTERVAL CHECK ===")
itv = read_csv(r"outputs\classification_v2\sequence_features\temporal_label_intervals.csv")
interval_behavior = None

if itv is not None:
    q = itv[
        itv["source_type"].astype(str).eq("cvat_tracking_xml")
        & itv["video_key"].astype(str).eq(VIDEO_KEY)
        & itv["pig_id"].astype(str).eq(PIG_ID)
    ].copy()

    for c in ["label_window_start", "label_window_end", "label_anchor_frame_index"]:
        if c in q.columns:
            q[c] = pd.to_numeric(q[c], errors="coerce")

    q = q[q["label_window_start"].eq(ANCHOR)].copy()

    cols = [
        "temporal_unit_key",
        "source_type",
        "video_key",
        "pig_id",
        "track_id",
        "label_anchor_frame_index",
        "label_window_start",
        "label_window_end",
        "behavior_temporal_final",
        "temporal_consistency_status",
    ]
    cols = [c for c in cols if c in q.columns]
    print(q[cols].to_string(index=False))

    if not q.empty and "behavior_temporal_final" in q.columns:
        interval_behavior = str(q.iloc[0]["behavior_temporal_final"])
        print("\nInterval behavior =", interval_behavior)


print("\n=== REVIEW UNIT CHECK ===")
units = read_csv(r"outputs\classification_v2\review_units\review_unit_manifest.csv")
unit_behavior = None
unit_ids = []

if units is not None:
    q = units[
        units["source_type"].astype(str).eq("cvat_tracking_xml")
        & units["video_key"].astype(str).eq(VIDEO_KEY)
        & units["pig_id"].astype(str).eq(PIG_ID)
    ].copy()

    for c in ["unit_start_frame", "unit_end_frame", "label_anchor_frame_index"]:
        if c in q.columns:
            q[c] = pd.to_numeric(q[c], errors="coerce")

    q = q[q["unit_start_frame"].eq(ANCHOR)].copy()

    cols = [
        "review_unit_id",
        "review_unit_type",
        "review_template",
        "review_reason",
        "source_type",
        "video_key",
        "pig_id",
        "track_id",
        "unit_start_frame",
        "unit_end_frame",
        "behavior_label",
        "temporal_consistency_status",
    ]
    cols = [c for c in cols if c in q.columns]
    print(q[cols].to_string(index=False))

    if not q.empty:
        unit_behavior = str(q.iloc[0]["behavior_label"])
        unit_ids = q["review_unit_id"].astype(str).tolist()
        print("\nReview unit behavior =", unit_behavior)


print("\n=== TEMPLATE MEMBERSHIP CHECK ===")
template_paths = [
    r"outputs\classification_v2\review_units\interaction_review_unit_template.csv",
    r"outputs\classification_v2\review_units\motion_review_unit_template.csv",
    r"outputs\classification_v2\review_units\posture_review_unit_template.csv",
    r"outputs\classification_v2\review_units\roi_review_unit_template.csv",
    r"outputs\classification_v2\review_units\full_review_unit_manifest.csv",
    r"outputs\classification_v2\review_units\balanced_gui_pilots\motion_balanced_gui_pilot.csv",
    r"outputs\classification_v2\review_units\balanced_gui_pilots\interaction_balanced_gui_pilot.csv",
]

for path in template_paths:
    p = Path(path)
    if not p.exists():
        continue
    df = pd.read_csv(p, low_memory=False)
    if "review_unit_id" not in df.columns:
        continue
    hit = df[df["review_unit_id"].astype(str).isin(unit_ids)].copy()
    print("\nFILE =", p)
    print("hit rows =", len(hit))
    if len(hit):
        cols = ["review_unit_id", "review_template", "behavior_label", "review_reason"]
        cols = [c for c in cols if c in hit.columns]
        print(hit[cols].to_string(index=False))


print("\n=== DIAGNOSIS ===")
print("xml_anchor_behavior      =", xml_anchor_behavior)
print("enhanced_anchor_behavior =", enh_anchor_behavior)
print("interval_behavior        =", interval_behavior)
print("review_unit_behavior     =", unit_behavior)

if xml_anchor_behavior and enh_anchor_behavior and xml_anchor_behavior != enh_anchor_behavior:
    print(
        "\nLIKELY PROBLEM: enhanced CSV was built from stale/wrong XML, "
        "or CVAT parser imported the wrong label at anchor."
    )
elif enh_anchor_behavior and interval_behavior and enh_anchor_behavior != interval_behavior:
    print("\nLIKELY PROBLEM: temporal harmonization is not using anchor behavior correctly for this unit.")
elif interval_behavior and unit_behavior and interval_behavior != unit_behavior:
    print("\nLIKELY PROBLEM: review units/templates are stale or review_unit_builder used old intervals.")
elif unit_behavior == "social-nose":
    print("\nDATA LOOKS CORRECT NOW. GUI/balanced pilot may be stale; rebuild balanced pilots and open with --fresh.")
else:
    print("\nNeed inspect printed tables above.")
