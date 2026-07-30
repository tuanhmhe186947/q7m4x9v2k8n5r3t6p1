"""Tkinter human audit GUI and strict validate-only preflight."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pig_behavior.tracking.gt_audit_review import (
    DECISIONS,
    ExactFrameReader,
    append_event,
    atomic_write_json,
    load_rows,
    parse_cvat,
    render_boxes,
    sha256,
    validate_decision,
)

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "docs/tracking/gt_audit_gui/TRACKING_GT_AUDIT_REVIEW_MANIFEST_20260730.csv"
AUTH = (
    ROOT
    / "docs/tracking/development_evidence_defense"
    / "DEVELOPMENT_EVIDENCE_INPUT_AUTHORITY_20260730.json"
)
OUT = (
    ROOT / "outputs/tracking/gt_audit_gui_preflight_20260730/TRACKING_GT_AUDIT_MEDIA_PREFLIGHT.json"
)


def preflight(path=MANIFEST):
    rows = load_rows(path)
    checks = {
        "MEDIA_MISSING": 0,
        "VIDEO_HASH_MISMATCH": 0,
        "GT_HASH_MISMATCH": 0,
        "PREDICTION_HASH_MISMATCH": 0,
        "FRAME_RANGE_ERRORS": 0,
        "OVERLAY_RENDER_ERRORS": 0,
        "UNSEEN_PATH_REFERENCES": 0,
        "SOURCE_ITEMS_UNMAPPED": 0,
        "decode_errors": [],
    }
    hash_cache = {}
    decoded = set()
    for r in rows:
        for _kind, p, expected, key in (
            ("video", r["video_path"], r["video_sha256"], "VIDEO_HASH_MISMATCH"),
            ("gt", r["GT_path"], r["GT_sha256"], "GT_HASH_MISMATCH"),
            (
                "prediction",
                r["prediction_path"],
                r["prediction_sha256"],
                "PREDICTION_HASH_MISMATCH",
            ),
        ):
            if "unseen" in p.lower() or "locked" in p.lower() and "development" not in p.lower():
                checks["UNSEEN_PATH_REFERENCES"] += 1
            if not os.path.exists(p):
                checks["MEDIA_MISSING"] += 1
                continue
            actual = hash_cache.setdefault(p, sha256(p))
            if actual != expected:
                checks[key] += 1
        if int(r["event_start_frame"]) < 0 or int(r["event_end_frame"]) >= 1800:
            checks["FRAME_RANGE_ERRORS"] += 1
        if os.path.exists(r["video_path"]) and r["video_path"] not in decoded:
            try:
                import cv2

                cap = cv2.VideoCapture(r["video_path"])
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(r["anchor_frame"]))
                if cap.read()[0] is False:
                    checks["decode_errors"].append(r["review_unit_id"])
                cap.release()
                decoded.add(r["video_path"])
            except Exception as exc:
                checks["decode_errors"].append(f"{r['review_unit_id']}:{exc}")
    checks["OVERLAY_RENDER_ERRORS"] = len(checks["decode_errors"])
    checks["status"] = (
        "PASS"
        if not any(
            checks[k]
            for k in (
                "MEDIA_MISSING",
                "VIDEO_HASH_MISMATCH",
                "GT_HASH_MISMATCH",
                "PREDICTION_HASH_MISMATCH",
                "FRAME_RANGE_ERRORS",
                "OVERLAY_RENDER_ERRORS",
                "UNSEEN_PATH_REFERENCES",
                "SOURCE_ITEMS_UNMAPPED",
            )
        )
        else "FAIL"
    )
    checks["review_units"] = len(rows)
    checks["source_items_mapped"] = sum(int(r["source_item_count"]) for r in rows)
    atomic_write_json(OUT, checks)
    return checks


def run_gui(rows, read_only=False, run_root=None):
    import tkinter as tk
    from tkinter import messagebox, ttk

    import cv2
    from PIL import Image, ImageTk

    if not rows:
        raise SystemExit("No review units selected")
    root = tk.Tk()
    root.title("Tracking GT Audit — neutral review")
    root.geometry("1500x950")
    root.minsize(1100, 750)
    idx = 0
    frame = int(rows[0]["anchor_frame"])
    method_revealed = False
    context_revealed = False
    playing = False
    reader = ExactFrameReader()
    gt_index = {}
    pred_index = {}
    status = tk.StringVar(value="AUDIT_TARGET_METHOD")
    info = tk.StringVar()
    decision = tk.StringVar()
    confidence = tk.StringVar(value="MEDIUM")
    comment = tk.StringVar()
    run_root = Path(
        run_root or ROOT / "human_review_workspace" / "tracking_gt_audit" / "REVIEW_RUN_ID"
    )
    run_root.mkdir(parents=True, exist_ok=True)

    header = ttk.Label(root, textvariable=info, font=("Segoe UI", 11, "bold"))
    header.pack(fill="x", padx=8, pady=(6, 2))
    view_grid = ttk.Frame(root)
    view_grid.pack(fill="both", expand=True, padx=8, pady=4)
    view_grid.columnconfigure(0, weight=1)
    view_grid.columnconfigure(1, weight=1)
    view_grid.rowconfigure(0, weight=1)
    view_grid.rowconfigure(1, weight=1)
    image_labels = {}
    for position, name in enumerate(("Clean source", "GT", "Prediction", "Combined")):
        panel = ttk.LabelFrame(view_grid, text=name)
        panel.grid(row=position // 2, column=position % 2, sticky="nsew", padx=3, pady=3)
        label = ttk.Label(panel, anchor="center")
        label.pack(fill="both", expand=True)
        image_labels[name] = label

    timeline = tk.Scale(root, from_=0, to=1, orient="horizontal", showvalue=False)
    timeline.pack(fill="x", padx=12)
    controls = ttk.Frame(root)
    controls.pack(fill="x", padx=8, pady=3)
    review = ttk.Frame(root)
    review.pack(fill="x", padx=8, pady=(3, 8))

    def load_unit() -> None:
        nonlocal frame, gt_index, pred_index, method_revealed, context_revealed
        row = rows[idx]
        reader.open(row["video_path"])
        gt_index = parse_cvat(row["GT_path"])
        pred_index = parse_cvat(row["prediction_path"])
        frame = int(row["anchor_frame"])
        method_revealed = False
        context_revealed = False
        status.set("AUDIT_TARGET_METHOD")
        timeline.configure(
            from_=int(row["context_start_frame"]),
            to=int(row["context_end_frame"]),
        )

    def photo(frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        scale = min(690 / width, 335 / height)
        size = (max(1, int(width * scale)), max(1, int(height * scale)))
        resized = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        return ImageTk.PhotoImage(Image.fromarray(resized))

    def show():
        nonlocal frame
        row = rows[idx]
        frame = max(
            int(row["context_start_frame"]),
            min(frame, int(row["context_end_frame"])),
        )
        source = reader.read(frame)
        gt_objects = gt_index.get(frame, [])
        pred_objects = pred_index.get(frame, [])
        gt_view = render_boxes(source, gt_objects, row["GT_identity"], (0, 210, 0), "GT")
        pred_view = render_boxes(
            source,
            pred_objects,
            row["predicted_identity"],
            (0, 140, 255),
            "PRED",
        )
        combined = render_boxes(
            gt_view,
            pred_objects,
            row["predicted_identity"],
            (0, 140, 255),
            "PRED",
        )
        for name, image in (
            ("Clean source", source),
            ("GT", gt_view),
            ("Prediction", pred_view),
            ("Combined", combined),
        ):
            tk_image = photo(image)
            image_labels[name].configure(image=tk_image)
            image_labels[name].image = tk_image
        fps = float(row["FPS"])
        relative = frame - int(row["event_start_frame"])
        info.set(
            f"{row['video_id']}  frame {frame}  t={frame / fps:.3f}s  "
            f"event-relative={relative:+d}  unit {idx + 1}/{len(rows)}  "
            f"{status.get()}"
        )
        timeline.set(frame)

    def step(delta):
        nonlocal frame
        frame += delta
        show()

    def seek(value):
        nonlocal frame
        frame = int(float(value))
        show()

    def jump(which):
        nonlocal frame
        frame = int(rows[idx][which])
        show()

    def change_unit(delta):
        nonlocal idx
        idx = max(0, min(len(rows) - 1, idx + delta))
        load_unit()
        show()

    def toggle_play():
        nonlocal playing
        playing = not playing
        if playing:
            play_tick()

    def play_tick():
        nonlocal frame, playing
        if not playing:
            return
        if frame >= int(rows[idx]["context_end_frame"]):
            playing = False
            return
        frame += 1
        show()
        delay = max(1, round(1000 / float(rows[idx]["FPS"])))
        root.after(delay, play_tick)

    def reveal_method():
        nonlocal method_revealed
        method_revealed = True
        status.set(rows[idx]["primary_method_id"])
        append_event(
            run_root / "tracking_gt_audit_decision_events.jsonl",
            {"event_type": "METHOD_REVEALED", "review_unit_id": rows[idx]["review_unit_id"]},
        )
        show()

    def reveal_context():
        nonlocal context_revealed
        context_revealed = True
        row = rows[idx]
        messagebox.showinfo(
            "Audit context",
            f"Category: {row['error_category']}\n"
            f"Selection: {row['selection_reasons']}\n"
            f"Contribution: {row['metric_contributions']}",
        )
        append_event(
            run_root / "tracking_gt_audit_decision_events.jsonl",
            {"event_type": "AUDIT_CONTEXT_REVEALED", "review_unit_id": row["review_unit_id"]},
        )

    def save():
        if read_only:
            return
        row = rows[idx].copy()
        row.update(
            {
                "reviewer": os.environ.get("USERNAME", "HUMAN_REVIEWER"),
                "decision": decision.get(),
                "confidence": confidence.get(),
                "reviewer_comment": comment.get(),
                "reviewed_context_start": row["context_start_frame"],
                "reviewed_context_end": row["context_end_frame"],
                "method_revealed_before_decision": "YES" if method_revealed else "NO",
                "audit_context_revealed_before_decision": "YES" if context_revealed else "NO",
            }
        )
        errors = validate_decision(row)
        if errors:
            messagebox.showerror("Invalid decision", ", ".join(errors))
            return
        target = run_root / "tracking_gt_audit_decisions.csv"
        fields = list(row)
        existing = []
        if target.exists():
            existing = load_rows(target)
            existing = [x for x in existing if x.get("review_unit_id") != row["review_unit_id"]]
        with target.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(existing + [row])
        append_event(
            run_root / "tracking_gt_audit_decision_events.jsonl",
            {
                "event_type": "DECISION_CREATED",
                "review_unit_id": row["review_unit_id"],
                "decision": row["decision"],
            },
        )
        messagebox.showinfo("Saved", f"Saved {row['review_unit_id']}")

    buttons = (
        ("Prev unit", lambda: change_unit(-1)),
        ("-1s", lambda: step(-round(float(rows[idx]["FPS"])))),
        ("-10", lambda: step(-10)),
        ("-1", lambda: step(-1)),
        ("Play/Pause", toggle_play),
        ("+1", lambda: step(1)),
        ("+10", lambda: step(10)),
        ("+1s", lambda: step(round(float(rows[idx]["FPS"])))),
        ("Event start", lambda: jump("event_start_frame")),
        ("Anchor", lambda: jump("anchor_frame")),
        ("Event end", lambda: jump("event_end_frame")),
        ("Next unit", lambda: change_unit(1)),
        ("Reveal method", reveal_method),
        ("Audit context", reveal_context),
        ("Save", save),
    )
    for text, cmd in buttons:
        ttk.Button(controls, text=text, command=cmd).pack(side="left", padx=3)
    timeline.configure(command=seek)
    ttk.Label(review, text="Decision").pack(side="left")
    ttk.Combobox(
        review,
        textvariable=decision,
        values=sorted(DECISIONS),
        width=30,
        state="readonly",
    ).pack(side="left", padx=4)
    ttk.Label(review, text="Confidence").pack(side="left")
    ttk.Combobox(
        review,
        textvariable=confidence,
        values=["HIGH", "MEDIUM", "LOW"],
        width=8,
        state="readonly",
    ).pack(side="left", padx=4)
    ttk.Label(review, text="Comment").pack(side="left")
    ttk.Entry(review, textvariable=comment, width=55).pack(
        side="left", fill="x", expand=True, padx=4
    )
    root.bind("<space>", lambda _event: toggle_play())
    root.bind("<Left>", lambda _event: step(-1))
    root.bind("<Right>", lambda _event: step(1))
    root.bind("<Shift-Left>", lambda _event: step(-10))
    root.bind("<Shift-Right>", lambda _event: step(10))
    root.bind("<Home>", lambda _event: jump("event_start_frame"))
    root.bind("<End>", lambda _event: jump("event_end_frame"))
    root.bind("<Prior>", lambda _event: change_unit(-1))
    root.bind("<Next>", lambda _event: change_unit(1))
    root.protocol("WM_DELETE_WINDOW", lambda: (reader.close(), root.destroy()))
    load_unit()
    show()
    root.mainloop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--headless-smoke", action="store_true")
    ap.add_argument("--read-only", action="store_true")
    ap.add_argument("--max-items", type=int)
    ap.add_argument("--review-unit-id")
    ap.add_argument("--run-root")
    args = ap.parse_args()
    rows = load_rows(MANIFEST)
    if args.max_items:
        rows = rows[: args.max_items]
    if args.review_unit_id:
        rows = [r for r in rows if r["review_unit_id"] == args.review_unit_id]
    if args.validate_only:
        result = preflight()
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "PASS" else 2
    if args.headless_smoke:
        print(json.dumps({"status": "PASS", "review_units": len(rows)}))
        return 0
    run_gui(rows, args.read_only, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
