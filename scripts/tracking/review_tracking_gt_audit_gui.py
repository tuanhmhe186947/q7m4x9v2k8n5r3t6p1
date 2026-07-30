"""Tkinter human audit GUI and strict validate-only preflight."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pig_behavior.tracking.gt_audit_review import (
    DECISIONS,
    DecisionLedger,
    ExactFrameReader,
    actor_crop,
    append_event,
    atomic_write_json,
    bbox_iou,
    identity_tokens,
    load_rows,
    parse_cvat,
    render_boxes,
    sha256,
    timeline_state,
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
    import cv2

    rows = load_rows(path)
    authority = json.loads(AUTH.read_text(encoding="utf-8"))
    allowed_gt = {
        (item["video_id"], str(Path(item["path"])), item["sha256"])
        for item in authority["ground_truth_authorities"]
    }
    authority_root = Path(authority["ground_truth_authorities"][0]["path"]).parents[3]
    allowed_predictions = {
        (method_id, str(authority_root / item["relative_path"]), item["sha256"])
        for method_id, method in authority["prediction_authorities"].items()
        for item in method["files"]
    }
    source_audit = (
        ROOT
        / "docs/tracking/development_evidence_defense"
        / "DEVELOPMENT_GT_ERROR_AUDIT_ITEMS_20260730.csv"
    )
    checks = {
        "MEDIA_MISSING": 0,
        "VIDEO_HASH_MISMATCH": 0,
        "GT_HASH_MISMATCH": 0,
        "PREDICTION_HASH_MISMATCH": 0,
        "FRAME_RANGE_ERRORS": 0,
        "OVERLAY_RENDER_ERRORS": 0,
        "UNSEEN_PATH_REFERENCES": 0,
        "SOURCE_ITEMS_UNMAPPED": 0,
        "IDENTITY_REFERENCE_ERRORS": 0,
        "FPS_ERRORS": 0,
        "FRAME_COUNT_ERRORS": 0,
        "AUTHORITY_PATH_ERRORS": 0,
        "AUTHORITY_METHOD_ERRORS": 0,
        "SOURCE_ARTIFACTS_WRITABLE": 0,
        "decode_errors": [],
    }
    hash_cache = {}
    xml_cache = {}
    reader = ExactFrameReader(cache_size=2)
    if tuple(authority.get("active_methods", [])) != (
        "bytetrack_raw",
        "hybrid_bytetrack",
        "realtime_fast",
        "rf_hybrid",
    ):
        checks["AUTHORITY_METHOD_ERRORS"] += 1
    for r in rows:
        if (r["video_id"], r["GT_path"], r["GT_sha256"]) not in allowed_gt:
            checks["AUTHORITY_PATH_ERRORS"] += 1
        if (
            r["primary_method_id"],
            r["prediction_path"],
            r["prediction_sha256"],
        ) not in allowed_predictions:
            checks["AUTHORITY_PATH_ERRORS"] += 1
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
            if os.access(p, os.W_OK):
                checks["SOURCE_ARTIFACTS_WRITABLE"] += 1
            actual = hash_cache.setdefault(p, sha256(p))
            if actual != expected:
                checks[key] += 1
        if int(r["event_start_frame"]) < 0 or int(r["event_end_frame"]) >= 1800:
            checks["FRAME_RANGE_ERRORS"] += 1
        if float(r["FPS"]) <= 0:
            checks["FPS_ERRORS"] += 1
        if os.path.exists(r["video_path"]):
            try:
                cap = cv2.VideoCapture(r["video_path"])
                video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                video_fps = float(cap.get(cv2.CAP_PROP_FPS))
                cap.release()
                if video_frames <= int(r["context_end_frame"]):
                    checks["FRAME_COUNT_ERRORS"] += 1
                if abs(video_fps - float(r["FPS"])) > 0.01:
                    checks["FPS_ERRORS"] += 1
                reader.open(r["video_path"])
                source = reader.read(int(r["anchor_frame"]))
                gt_index = xml_cache.setdefault(r["GT_path"], parse_cvat(r["GT_path"]))
                pred_index = xml_cache.setdefault(
                    r["prediction_path"], parse_cvat(r["prediction_path"])
                )
                anchor = int(r["anchor_frame"])
                gt_objects = gt_index.get(anchor, [])
                pred_objects = pred_index.get(anchor, [])
                render_boxes(source, gt_objects, r["GT_identity"], (0, 210, 0), "GT")
                render_boxes(
                    source,
                    pred_objects,
                    r["predicted_identity"],
                    (0, 140, 255),
                    "PRED",
                )
                all_gt_ids = {
                    obj["id"] for frame_objects in gt_index.values() for obj in frame_objects
                }
                if not identity_tokens(r["GT_identity"]).issubset(all_gt_ids):
                    checks["IDENTITY_REFERENCE_ERRORS"] += 1
                all_pred_ids = {
                    obj["id"] for frame_objects in pred_index.values() for obj in frame_objects
                }
                if not identity_tokens(r["predicted_identity"]).issubset(all_pred_ids):
                    checks["IDENTITY_REFERENCE_ERRORS"] += 1
            except Exception as exc:
                checks["decode_errors"].append(f"{r['review_unit_id']}:{exc}")
    reader.close()
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
                "IDENTITY_REFERENCE_ERRORS",
                "FPS_ERRORS",
                "FRAME_COUNT_ERRORS",
                "AUTHORITY_PATH_ERRORS",
                "AUTHORITY_METHOD_ERRORS",
            )
        )
        else "FAIL"
    )
    checks["review_units"] = len(rows)
    checks["source_items_mapped"] = sum(int(r["source_item_count"]) for r in rows)
    source_count = len(load_rows(source_audit))
    checks["SOURCE_ITEMS_UNMAPPED"] = max(0, source_count - checks["source_items_mapped"])
    atomic_write_json(OUT, checks)
    return checks


def run_gui(rows, read_only=False, run_root=None, reviewer="HUMAN_REVIEWER", resume=False):
    import tkinter as tk
    from tkinter import messagebox, ttk

    import cv2
    from PIL import Image, ImageTk

    if not rows:
        raise SystemExit("No review units selected")
    root = tk.Tk()
    root.title("Tracking GT Audit - neutral review")
    root.geometry("1500x950")
    root.minsize(1100, 750)
    idx = 0
    frame = int(rows[0]["anchor_frame"])
    method_revealed = False
    context_revealed = False
    playing = False
    unit_started = time.monotonic()
    reader = ExactFrameReader()
    gt_index = {}
    pred_index = {}
    status = tk.StringVar(value="AUDIT_TARGET_METHOD")
    info = tk.StringVar()
    decision = tk.StringVar()
    confidence = tk.StringVar(value="MEDIUM")
    comment = tk.StringVar()
    banner = tk.StringVar(value="Inspect clean, GT, prediction, temporal context, then decide.")
    playback_speed = tk.DoubleVar(value=1.0)
    view_visible = {"GT": True, "Prediction": True, "Combined": True}
    hidden_metadata_visible = True
    crop_visible = True
    context_bounds = {
        row["review_unit_id"]: [
            int(row["context_start_frame"]),
            int(row["context_end_frame"]),
        ]
        for row in rows
    }
    run_root = Path(
        run_root or ROOT / "human_review_workspace" / "tracking_gt_audit" / "REVIEW_RUN_ID"
    )
    ledger = None
    if not read_only:
        gui_code_sha = sha256(Path(__file__))
        ledger = DecisionLedger(run_root, MANIFEST, gui_code_sha, reviewer)
        if resume:
            reviewed_ids = set(ledger.current())
            idx = next(
                (
                    position
                    for position, row in enumerate(rows)
                    if row["review_unit_id"] not in reviewed_ids
                ),
                0,
            )
    reviewed_ids = set(ledger.current()) if ledger else set()

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

    crop_frame = ttk.Frame(root)
    crop_frame.pack(fill="x", padx=8, pady=2)
    crop_labels = {}
    for name in ("GT actor context", "Prediction actor context"):
        panel = ttk.LabelFrame(crop_frame, text=name)
        panel.pack(side="left", fill="both", expand=True, padx=3)
        label = ttk.Label(panel, anchor="center")
        label.pack(fill="both", expand=True)
        crop_labels[name] = label

    timeline = tk.Scale(root, from_=0, to=1, orient="horizontal", showvalue=False)
    timeline.pack(fill="x", padx=12)
    timeline_info = tk.StringVar()
    ttk.Label(root, textvariable=timeline_info).pack(fill="x", padx=12)
    controls = ttk.Frame(root)
    controls.pack(fill="x", padx=8, pady=3)
    review = ttk.Frame(root)
    review.pack(fill="x", padx=8, pady=(3, 8))
    ttk.Label(root, textvariable=banner, foreground="#145a32").pack(fill="x", padx=12)

    def load_unit() -> None:
        nonlocal frame, gt_index, pred_index, method_revealed, context_revealed
        nonlocal unit_started
        row = rows[idx]
        reader.open(row["video_path"])
        gt_index = parse_cvat(row["GT_path"])
        pred_index = parse_cvat(row["prediction_path"])
        frame = int(row["anchor_frame"])
        method_revealed = False
        context_revealed = False
        unit_started = time.monotonic()
        decision.set("")
        comment.set("")
        status.set("AUDIT_TARGET_METHOD")
        start, end = context_bounds[row["review_unit_id"]]
        timeline.configure(
            from_=start,
            to=end,
        )

    def photo(frame_bgr, max_width=690, max_height=335):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        scale = min(max_width / width, max_height / height)
        size = (max(1, int(width * scale)), max(1, int(height * scale)))
        resized = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        return ImageTk.PhotoImage(Image.fromarray(resized))

    def show():
        nonlocal frame
        row = rows[idx]
        context_start, context_end = context_bounds[row["review_unit_id"]]
        frame = max(
            context_start,
            min(frame, context_end),
        )
        source = reader.read(frame)
        gt_objects = gt_index.get(frame, [])
        pred_objects = pred_index.get(frame, [])
        gt_target = next(
            (obj for obj in gt_objects if obj["id"] in identity_tokens(row["GT_identity"])),
            None,
        )
        pred_target = next(
            (
                obj
                for obj in pred_objects
                if obj["id"] in identity_tokens(row["predicted_identity"])
            ),
            None,
        )
        iou = (
            bbox_iou(gt_target["bbox"], pred_target["bbox"]) if gt_target and pred_target else None
        )
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
            if name in view_visible and not view_visible[name]:
                image = source
            tk_image = photo(image)
            image_labels[name].configure(image=tk_image)
            image_labels[name].image = tk_image
        gt_crop = actor_crop(source, gt_objects, row["GT_identity"])
        pred_crop = actor_crop(source, pred_objects, row["predicted_identity"])
        for name, image in (
            ("GT actor context", gt_crop),
            ("Prediction actor context", pred_crop),
        ):
            if not crop_visible:
                image = source
            tk_image = photo(image, max_width=650, max_height=150)
            crop_labels[name].configure(image=tk_image)
            crop_labels[name].image = tk_image
        fps = float(row["FPS"])
        relative = frame - int(row["event_start_frame"])
        reviewed = len(reviewed_ids)
        category_display = row["error_category"] if context_revealed else "COLLAPSED"
        match_display = (
            row.get("matching_eligibilities", "NOT_AVAILABLE") if context_revealed else "COLLAPSED"
        )
        timeline_marker = (
            "EVENT"
            if timeline_state(row, frame, gt_objects, pred_objects)["event_active"]
            else "context"
        )
        hidden_display = row["Hidden_status"] if hidden_metadata_visible else "HIDDEN_METADATA_OFF"
        info.set(
            f"{row['video_id']}  frame {frame}  t={frame / fps:.3f}s  "
            f"event-relative={relative:+d}  unit {idx + 1}/{len(rows)}  "
            f"reviewed={reviewed} unresolved={len(rows) - reviewed}  "
            f"speed={playback_speed.get():.2f}x  {status.get()}"
        )
        timeline_info.set(
            f"context={context_start}-{context_end} | "
            f"event={row['event_start_frame']}-{row['event_end_frame']} | "
            f"anchor={row['anchor_frame']} | GT={row['GT_identity']} | "
            f"PRED={row['predicted_identity']} | "
            f"Hidden={row['Hidden_status']} | category={category_display} | "
            f"match={match_display} | "
            f"IoU={'NA' if iou is None else f'{iou:.3f}'} | "
            f"{timeline_marker} | Hidden={hidden_display}"
        )
        timeline.set(frame)

    def toggle_view(name):
        view_visible[name] = not view_visible[name]
        show()

    def toggle_hidden_metadata():
        nonlocal hidden_metadata_visible
        hidden_metadata_visible = not hidden_metadata_visible
        show()

    def toggle_crop():
        nonlocal crop_visible
        crop_visible = not crop_visible
        show()

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

    def jump_context(which):
        nonlocal frame
        frame = context_bounds[rows[idx]["review_unit_id"]][which]
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
        if frame >= context_bounds[rows[idx]["review_unit_id"]][1]:
            playing = False
            return
        frame += 1
        show()
        delay = max(
            1,
            round(1000 / (float(rows[idx]["FPS"]) * playback_speed.get())),
        )
        root.after(delay, play_tick)

    def reveal_method():
        nonlocal method_revealed
        method_revealed = True
        status.set(rows[idx]["primary_method_id"])
        if ledger:
            append_event(
                ledger.events_path,
                {
                    "event_type": "METHOD_REVEALED",
                    "review_unit_id": rows[idx]["review_unit_id"],
                    "human_initiated": True,
                },
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
        if ledger:
            append_event(
                ledger.events_path,
                {
                    "event_type": "AUDIT_CONTEXT_REVEALED",
                    "review_unit_id": row["review_unit_id"],
                    "human_initiated": True,
                },
            )

    def extend_context(seconds):
        row = rows[idx]
        delta = round(float(row["FPS"]) * seconds)
        bounds = context_bounds[row["review_unit_id"]]
        bounds[0] = max(0, bounds[0] - delta)
        bounds[1] = min(1799, bounds[1] + delta)
        timeline.configure(from_=bounds[0], to=bounds[1])
        if ledger:
            append_event(
                ledger.events_path,
                {
                    "event_type": "CONTEXT_EXTENDED",
                    "review_unit_id": row["review_unit_id"],
                    "reviewed_context_start": bounds[0],
                    "reviewed_context_end": bounds[1],
                    "human_initiated": True,
                },
            )
        show()

    def next_unresolved():
        nonlocal idx
        positions = list(range(idx + 1, len(rows))) + list(range(0, idx + 1))
        idx = next(
            (
                position
                for position in positions
                if rows[position]["review_unit_id"] not in reviewed_ids
            ),
            idx,
        )
        load_unit()
        show()

    def undo():
        nonlocal reviewed_ids
        if not ledger:
            return
        uid = ledger.undo_latest()
        reviewed_ids = set(ledger.current())
        banner.set(f"Undid latest decision: {uid}" if uid else "No decision to undo.")
        show()

    def save():
        nonlocal reviewed_ids
        if read_only or ledger is None:
            return
        row = rows[idx].copy()
        if not messagebox.askyesno(
            "Confirm decision",
            f"Save {decision.get() or '<none>'} with {confidence.get()} confidence?",
        ):
            return
        try:
            context_start, context_end = context_bounds[row["review_unit_id"]]
            ledger.save(
                row,
                decision.get(),
                confidence.get(),
                comment.get(),
                context_start,
                context_end,
                method_revealed,
                context_revealed,
                time.monotonic() - unit_started,
            )
        except ValueError as exc:
            messagebox.showerror("Invalid decision", str(exc))
            return
        banner.set(f"Saved {row['review_unit_id']} - advancing to next unresolved unit.")
        reviewed_ids = set(ledger.current())
        next_unresolved()

    buttons = (
        ("Prev unit", lambda: change_unit(-1)),
        ("-1s", lambda: step(-round(float(rows[idx]["FPS"])))),
        ("-10", lambda: step(-10)),
        ("-1", lambda: step(-1)),
        ("Play/Pause", toggle_play),
        ("+1", lambda: step(1)),
        ("+10", lambda: step(10)),
        ("+1s", lambda: step(round(float(rows[idx]["FPS"])))),
        ("Context start", lambda: jump_context(0)),
        ("Event start", lambda: jump("event_start_frame")),
        ("Anchor", lambda: jump("anchor_frame")),
        ("Event end", lambda: jump("event_end_frame")),
        ("Context end", lambda: jump_context(1)),
        ("Extend ±3s", lambda: extend_context(3)),
        ("Next unit", lambda: change_unit(1)),
        ("Next unresolved", next_unresolved),
        ("Undo", undo),
        ("Reveal method", reveal_method),
        ("Audit context", reveal_context),
        ("Save", save),
    )
    for text, cmd in buttons:
        ttk.Button(controls, text=text, command=cmd).pack(side="left", padx=3)
    ttk.Label(controls, text="Speed").pack(side="left", padx=(8, 2))
    ttk.Combobox(
        controls,
        textvariable=playback_speed,
        values=[0.25, 0.5, 1.0, 1.5, 2.0, 4.0],
        width=5,
        state="readonly",
    ).pack(side="left")
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
    ttk.Label(
        root,
        text=(
            "Shortcuts: Space play/pause | Left/Right +/-1 | Shift+Left/Right +/-10 | "
            "Ctrl+Left/Right +/-1s | Home/End event | PgUp/PgDn units | "
            "N next unresolved | G/P/C toggle views | H Hidden | Z crops | U undo | S save"
        ),
    ).pack(fill="x", padx=12, pady=(0, 5))
    root.bind("<space>", lambda _event: toggle_play())
    root.bind("<Left>", lambda _event: step(-1))
    root.bind("<Right>", lambda _event: step(1))
    root.bind("<Shift-Left>", lambda _event: step(-10))
    root.bind("<Shift-Right>", lambda _event: step(10))
    root.bind(
        "<Control-Left>",
        lambda _event: step(-round(float(rows[idx]["FPS"]))),
    )
    root.bind(
        "<Control-Right>",
        lambda _event: step(round(float(rows[idx]["FPS"]))),
    )
    root.bind("<Home>", lambda _event: jump("event_start_frame"))
    root.bind("<End>", lambda _event: jump("event_end_frame"))
    root.bind("<Prior>", lambda _event: change_unit(-1))
    root.bind("<Next>", lambda _event: change_unit(1))
    root.bind("<n>", lambda _event: next_unresolved())
    root.bind("<g>", lambda _event: toggle_view("GT"))
    root.bind("<p>", lambda _event: toggle_view("Prediction"))
    root.bind("<c>", lambda _event: toggle_view("Combined"))
    root.bind("<h>", lambda _event: toggle_hidden_metadata())
    root.bind("<z>", lambda _event: toggle_crop())
    root.bind("<u>", lambda _event: undo())
    root.bind("<s>", lambda _event: save())
    root.protocol("WM_DELETE_WINDOW", lambda: (reader.close(), root.destroy()))
    load_unit()
    show()
    root.mainloop()


def headless_smoke(rows):
    with tempfile.TemporaryDirectory(prefix="tracking_gt_audit_smoke_") as temp:
        ledger = DecisionLedger(temp, MANIFEST, sha256(Path(__file__)), "SYNTHETIC_REVIEWER")
        row = rows[0]
        ledger.save(
            row,
            "NO_MATERIAL_ISSUE_CONFIRMED",
            "HIGH",
            "",
            int(row["context_start_frame"]),
            int(row["context_end_frame"]),
            False,
            False,
            1.0,
        )
        assert row["review_unit_id"] in ledger.current()
        assert ledger.undo_latest() == row["review_unit_id"]
        assert not ledger.current()
    return {"status": "PASS", "review_units": len(rows), "temporary_decisions": 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--headless-smoke", action="store_true")
    ap.add_argument("--read-only", action="store_true")
    ap.add_argument("--max-items", type=int)
    ap.add_argument("--review-unit-id")
    ap.add_argument("--run-root")
    ap.add_argument("--reviewer")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--fresh", action="store_true")
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
        print(json.dumps(headless_smoke(rows)))
        return 0
    if not args.read_only and (not args.run_root or not args.reviewer):
        raise SystemExit("--run-root and --reviewer are required for writable review")
    if args.max_items and not args.read_only:
        marker = str(args.run_root).upper()
        if "PILOT" not in marker and "SMOKE" not in marker:
            raise SystemExit("--max-items requires a separate PILOT or SMOKE run root")
    if args.fresh and args.run_root and Path(args.run_root).exists():
        raise SystemExit("--fresh refuses to overwrite an existing review root; use a new RUN_ID")
    result = preflight()
    if result["status"] != "PASS":
        raise SystemExit("TRACKING_GT_AUDIT_MEDIA_PREFLIGHT_FAILED")
    run_gui(rows, args.read_only, args.run_root, args.reviewer, args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
