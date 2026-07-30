# Tracking GT/error audit GUI runbook

This tool reviews the frozen development audit population only. It records human
questions about GT, predictions and evaluator matching; it never edits GT,
predictions, or metrics. Reviewers should inspect the temporal span, identity
continuity, bbox ownership, Hidden state, and gaps before choosing one decision.

Decisions are `PREDICTION_ERROR_CONFIRMED`, `GT_IDENTITY_QUESTION`,
`GT_BBOX_QUESTION`, `HIDDEN_LABEL_QUESTION`, `EVALUATOR_MATCHING_QUESTION`,
`FRAGMENTATION_ONLY`, `NO_MATERIAL_ISSUE_CONFIRMED`, `AMBIGUOUS_UNRESOLVED`, or
`OTHER_REVIEW_QUESTION`. Use HIGH/MEDIUM/LOW confidence. Comments are required
for questions, ambiguity, and decisions contradicting the source category.

## Commands (Windows CMD)

```text
cd /d C:\Users\ironh\Downloads\PIG_Behavior_Project
python scripts\tracking\build_tracking_gt_audit_review_manifest.py
python scripts\tracking\review_tracking_gt_audit_gui.py --validate-only
python scripts\tracking\review_tracking_gt_audit_gui.py --headless-smoke
python scripts\tracking\review_tracking_gt_audit_gui.py --max-items 5 --read-only
python scripts\tracking\review_tracking_gt_audit_gui.py --read-only
python scripts\tracking\check_tracking_gt_audit_coverage.py --manifest docs\tracking\gt_audit_gui\TRACKING_GT_AUDIT_REVIEW_MANIFEST_20260730.csv --decisions human_review_workspace\tracking_gt_audit\RUN_ID\tracking_gt_audit_decisions.csv --events human_review_workspace\tracking_gt_audit\RUN_ID\tracking_gt_audit_decision_events.jsonl --output human_review_workspace\tracking_gt_audit\RUN_ID\tracking_gt_audit_coverage.json
python scripts\tracking\summarize_tracking_gt_audit_decisions.py --manifest docs\tracking\gt_audit_gui\TRACKING_GT_AUDIT_REVIEW_MANIFEST_20260730.csv --decisions human_review_workspace\tracking_gt_audit\RUN_ID\tracking_gt_audit_decisions.csv --coverage human_review_workspace\tracking_gt_audit\RUN_ID\tracking_gt_audit_coverage.json --output-json human_review_workspace\tracking_gt_audit\RUN_ID\tracking_gt_audit_summary.json --output-csv human_review_workspace\tracking_gt_audit\RUN_ID\tracking_gt_audit_summary.csv
```

The coding agent does not open the production GUI or create decisions. A human
operator should use a fresh `RUN_ID`, review from the anchor through the full
event, extend context when needed, save one decision per unit, and leave GT or
prediction corrections to a separate scientific reconciliation task.

Shortcuts: Space play/pause; Left/Right one frame; Shift+Left/Right ten frames;
Ctrl+Left/Right one second; Home/End event bounds; PageUp/PageDown units; N next
unresolved; G/P/C overlays; H Hidden metadata; Z crop; U undo; S save. The
current implementation is intentionally read-only at handoff; decision-ledger
integration is validated by the standalone checker and must be enabled only by
the human review workflow.

Completion requires coverage PASS for all review units. `AMBIGUOUS_UNRESOLVED`
counts as complete review coverage but remains a scientific ambiguity.
