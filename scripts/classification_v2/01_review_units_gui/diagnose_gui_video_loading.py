import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

try:
    import cv2
except Exception as e:
    cv2 = None
    print("[ERROR] Cannot import cv2:", repr(e))


VIDEO_KEY = "Pigs291119_000231"
PIG_ID = "ID_4"
TRACK_ID = "3"
START = 678
END = 683

VIDEO_ROOT = Path(r"data\videos")
XML_ROOT = Path(r"data\annotations\classification")
ENHANCED_CSV = Path(r"outputs\classification_v2\frame_features\spatiotemporal_frame_features_enhanced.csv")

print("=== TARGET ===")
print("video_key =", VIDEO_KEY)
print("pig_id =", PIG_ID)
print("track_id =", TRACK_ID)
print("frames =", START, "-", END)

print("\n=== VIDEO ROOT ===")
print("video_root =", VIDEO_ROOT)
print("exists =", VIDEO_ROOT.exists())

print("\n=== DIR SEARCH ===")
matches = []
if VIDEO_ROOT.exists():
    for p in VIDEO_ROOT.rglob("*"):
        if p.is_file() and VIDEO_KEY.lower() in p.name.lower():
            matches.append(p)

print("matches =", len(matches))
for p in matches[:50]:
    print(" ", p, "size=", p.stat().st_size)

print("\n=== COMMON CANDIDATES ===")
common = []
for ext in [".mp4", ".MP4", ".avi", ".AVI", ".mov", ".MOV", ".mkv", ".MKV"]:
    common.append(VIDEO_ROOT / f"{VIDEO_KEY}{ext}")
    common.append(VIDEO_ROOT / f"{VIDEO_KEY}_30fps{ext}")

for p in common:
    if p.exists():
        print("[EXISTS]", p)

print("\n=== XML SOURCE CHECK ===")
xml_candidates = [
    XML_ROOT / f"{VIDEO_KEY}.xml",
    XML_ROOT / f"{VIDEO_KEY}_30fps.xml",
    XML_ROOT / f"Tracking_annotation_{VIDEO_KEY}.xml",
    XML_ROOT / f"Tracking_annotation_{VIDEO_KEY}_30fps.xml",
]

for xp in xml_candidates:
    print(xp, "exists =", xp.exists())
    if xp.exists():
        try:
            root = ET.parse(xp).getroot()
            source = root.findtext(".//meta/task/source")
            size = root.findtext(".//meta/task/size")
            print("  xml source =", source)
            print("  xml size =", size)
        except Exception as e:
            print("  [XML READ ERROR]", repr(e))

print("\n=== ENHANCED CSV ROW CHECK ===")
if not ENHANCED_CSV.exists():
    print("[MISSING]", ENHANCED_CSV)
    raise SystemExit(1)

df = pd.read_csv(ENHANCED_CSV, low_memory=False)

q = df[
    df["source_type"].astype(str).eq("cvat_tracking_xml")
    & df["video_key"].astype(str).eq(VIDEO_KEY)
    & df["pig_id"].astype(str).eq(PIG_ID)
    & pd.to_numeric(df["frame_index"], errors="coerce").between(START, END)
].copy()

print("rows =", len(q))

cols = [
    "source_type",
    "dataset_id",
    "video_key",
    "pig_id",
    "track_id",
    "track_label",
    "frame_index",
    "behavior",
    "label_anchor_frame_index",
    "label_window_start",
    "label_window_end",
    "x1",
    "y1",
    "x2",
    "y2",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "xtl",
    "ytl",
    "xbr",
    "ybr",
]
cols = [c for c in cols if c in q.columns]
print(q[cols].to_string(index=False))

print("\n=== BBOX COLUMN DETECTION ===")
bbox_sets = [
    ("x1", "y1", "x2", "y2"),
    ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"),
    ("xtl", "ytl", "xbr", "ybr"),
]
bbox_cols = None
for s in bbox_sets:
    if all(c in q.columns for c in s):
        bbox_cols = s
        print("bbox_cols =", bbox_cols)
        break

if bbox_cols is None:
    print("[WARN] No bbox column set detected.")
else:
    tmp = q.copy()
    for c in bbox_cols:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
    bbox_ok = (
        tmp[list(bbox_cols)].notna().all(axis=1)
        & (tmp[bbox_cols[2]] > tmp[bbox_cols[0]])
        & (tmp[bbox_cols[3]] > tmp[bbox_cols[1]])
    )
    print("bbox_ok:")
    print(bbox_ok.value_counts(dropna=False).to_string())

print("\n=== OPENCV READ TEST ===")
if cv2 is None:
    raise SystemExit(1)

video_candidates = []
video_candidates.extend([p for p in common if p.exists()])
video_candidates.extend(matches)

# Deduplicate preserving order
seen = set()
dedup = []
for p in video_candidates:
    rp = str(p.resolve()).lower()
    if rp not in seen:
        seen.add(rp)
        dedup.append(p)

if not dedup:
    print("[FAIL] No video candidate found for", VIDEO_KEY)
    raise SystemExit(0)

for vp in dedup[:10]:
    print("\nVIDEO =", vp)
    cap = cv2.VideoCapture(str(vp))
    print("opened =", cap.isOpened())
    if not cap.isOpened():
        continue

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print("frame_count =", frame_count)
    print("width,height =", width, height)
    print("fps =", fps)

    for f in range(START, END + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        print(" frame", f, "read_ok =", bool(ok), "shape =", None if not ok else frame.shape)

    cap.release()
