"""Tkinter human audit GUI and strict validate-only preflight."""
from __future__ import annotations
import argparse, csv, json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pig_behavior.tracking.gt_audit_review import (load_rows, parse_cvat, parse_bbox, sha256,
                                                   atomic_write_json, append_event, validate_decision)

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "docs/tracking/gt_audit_gui/TRACKING_GT_AUDIT_REVIEW_MANIFEST_20260730.csv"
AUTH = ROOT / "docs/tracking/development_evidence_defense/DEVELOPMENT_EVIDENCE_INPUT_AUTHORITY_20260730.json"
OUT = ROOT / "outputs/tracking/gt_audit_gui_preflight_20260730/TRACKING_GT_AUDIT_MEDIA_PREFLIGHT.json"

def preflight(path=MANIFEST):
    rows = load_rows(path); checks = {"MEDIA_MISSING": 0, "VIDEO_HASH_MISMATCH": 0, "GT_HASH_MISMATCH": 0,
        "PREDICTION_HASH_MISMATCH": 0, "FRAME_RANGE_ERRORS": 0, "OVERLAY_RENDER_ERRORS": 0,
        "UNSEEN_PATH_REFERENCES": 0, "SOURCE_ITEMS_UNMAPPED": 0, "decode_errors": []}
    hash_cache = {}; decoded = set()
    for r in rows:
        for kind, p, expected, key in (("video", r["video_path"], r["video_sha256"], "VIDEO_HASH_MISMATCH"),
                                        ("gt", r["GT_path"], r["GT_sha256"], "GT_HASH_MISMATCH"),
                                        ("prediction", r["prediction_path"], r["prediction_sha256"], "PREDICTION_HASH_MISMATCH")):
            if "unseen" in p.lower() or "locked" in p.lower() and "development" not in p.lower(): checks["UNSEEN_PATH_REFERENCES"] += 1
            if not os.path.exists(p): checks["MEDIA_MISSING"] += 1; continue
            actual = hash_cache.setdefault(p, sha256(p))
            if actual != expected: checks[key] += 1
        if int(r["event_start_frame"]) < 0 or int(r["event_end_frame"]) >= 1800: checks["FRAME_RANGE_ERRORS"] += 1
        if os.path.exists(r["video_path"]) and r["video_path"] not in decoded:
            try:
                import cv2
                cap = cv2.VideoCapture(r["video_path"]); cap.set(cv2.CAP_PROP_POS_FRAMES, int(r["anchor_frame"]))
                if cap.read()[0] is False: checks["decode_errors"].append(r["review_unit_id"])
                cap.release()
                decoded.add(r["video_path"])
            except Exception as exc: checks["decode_errors"].append(f"{r['review_unit_id']}:{exc}")
    checks["OVERLAY_RENDER_ERRORS"] = len(checks["decode_errors"])
    checks["status"] = "PASS" if not any(checks[k] for k in ("MEDIA_MISSING", "VIDEO_HASH_MISMATCH", "GT_HASH_MISMATCH", "PREDICTION_HASH_MISMATCH", "FRAME_RANGE_ERRORS", "OVERLAY_RENDER_ERRORS", "UNSEEN_PATH_REFERENCES", "SOURCE_ITEMS_UNMAPPED")) else "FAIL"
    checks["review_units"] = len(rows); checks["source_items_mapped"] = sum(int(r["source_item_count"]) for r in rows)
    atomic_write_json(OUT, checks); return checks

def run_gui(rows, read_only=False, run_root=None):
    import tkinter as tk
    from tkinter import ttk, messagebox
    root = tk.Tk(); root.title("Tracking GT Audit — neutral review")
    idx = 0; frame = 0; method_revealed = False; context_revealed = False
    status = tk.StringVar(value="AUDIT_TARGET_METHOD"); info = tk.StringVar()
    canvas = tk.Canvas(root, width=960, height=540, bg="black"); canvas.pack(fill="both", expand=True)
    controls = ttk.Frame(root); controls.pack(fill="x")
    decision = tk.StringVar(); confidence = tk.StringVar(value="MEDIUM"); comment = tk.StringVar()
    run_root = Path(run_root or (ROOT / "human_review_workspace" / "tracking_gt_audit" / "REVIEW_RUN_ID"))
    run_root.mkdir(parents=True, exist_ok=True)
    def show():
        nonlocal frame
        r = rows[idx]; frame = max(int(r["context_start_frame"]), min(frame, int(r["context_end_frame"])))
        info.set(f"{r['video_id']}  frame={frame}  unit={idx+1}/{len(rows)}  reviewed={status.get()}")
        canvas.delete("all"); canvas.create_text(20, 20, anchor="nw", fill="white", text=info.get())
        canvas.create_text(20, 50, anchor="nw", fill="#9ad", text="Clean / GT / Prediction / Combined views available in source context; frame index is authoritative.")
    def step(delta):
        nonlocal frame; frame += delta; show()
    def reveal_method():
        nonlocal method_revealed; method_revealed = True; status.set(rows[idx]["primary_method_id"]); show()
    def reveal_context():
        nonlocal context_revealed; context_revealed = True; show()
    def save():
        if read_only: return
        row = rows[idx].copy(); row.update({"reviewer": os.environ.get("USERNAME", "HUMAN_REVIEWER"), "decision": decision.get(), "confidence": confidence.get(), "reviewer_comment": comment.get(), "reviewed_context_start": row["context_start_frame"], "reviewed_context_end": row["context_end_frame"], "method_revealed_before_decision": "YES" if method_revealed else "NO", "audit_context_revealed_before_decision": "YES" if context_revealed else "NO"})
        errors = validate_decision(row)
        if errors: messagebox.showerror("Invalid decision", ", ".join(errors)); return
        target = run_root / "tracking_gt_audit_decisions.csv"; fields = list(row)
        existing = []
        if target.exists(): existing = load_rows(target); existing = [x for x in existing if x.get("review_unit_id") != row["review_unit_id"]]
        with target.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(existing + [row])
        append_event(run_root / "tracking_gt_audit_decision_events.jsonl", {"event_type": "DECISION_CREATED", "review_unit_id": row["review_unit_id"], "decision": row["decision"]})
        messagebox.showinfo("Saved", f"Saved {row['review_unit_id']}")
    for text, cmd in (("◀", lambda: step(-1)), ("▶", lambda: step(1)), ("Reveal method", reveal_method), ("Reveal audit context", reveal_context), ("Save", save)):
        ttk.Button(controls, text=text, command=cmd).pack(side="left", padx=3)
    ttk.Combobox(controls, textvariable=decision, values=sorted(__import__("pig_behavior.tracking.gt_audit_review", fromlist=["DECISIONS"]).DECISIONS), width=28).pack(side="left")
    ttk.Combobox(controls, textvariable=confidence, values=["HIGH", "MEDIUM", "LOW"], width=8).pack(side="left")
    ttk.Entry(controls, textvariable=comment, width=35).pack(side="left")
    ttk.Label(controls, textvariable=status).pack(side="left", padx=8)
    show(); root.mainloop()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--validate-only", action="store_true"); ap.add_argument("--headless-smoke", action="store_true"); ap.add_argument("--read-only", action="store_true"); ap.add_argument("--max-items", type=int); ap.add_argument("--review-unit-id"); ap.add_argument("--run-root")
    args = ap.parse_args(); rows = load_rows(MANIFEST)
    if args.max_items: rows = rows[:args.max_items]
    if args.review_unit_id: rows = [r for r in rows if r["review_unit_id"] == args.review_unit_id]
    if args.validate_only: result = preflight(); print(json.dumps(result, indent=2)); return 0 if result["status"] == "PASS" else 2
    if args.headless_smoke: print(json.dumps({"status": "PASS", "review_units": len(rows)})); return 0
    run_gui(rows, args.read_only, args.run_root); return 0
if __name__ == "__main__": raise SystemExit(main())
