# Tracking GT/error audit GUI runbook

This tool reviews the frozen 13-video development audit population only. The
reviewer decides whether visual evidence supports a prediction error, GT
question, Hidden-label question, evaluator question, fragmentation, ambiguity,
or no material issue. The reviewer does not correct GT, alter predictions,
recalculate metrics, rank methods, or make a tracker-promotion decision.

For each unit, begin at the anchor, inspect the clean frame, then inspect GT and
prediction separately. Play before and after the event. Follow the physical pig,
not merely the displayed identity text. A wrong owner persists on another pig;
fragmentation is a gap or termination without convincing transfer to another
pig. Flag GT identity when GT changes owner, GT bbox when the rectangle covers
the wrong actor, Hidden when visibility status conflicts with the image, and
evaluator matching when both annotations appear defensible but the match does
not.

Decisions are `PREDICTION_ERROR_CONFIRMED`, `GT_IDENTITY_QUESTION`,
`GT_BBOX_QUESTION`, `HIDDEN_LABEL_QUESTION`, `EVALUATOR_MATCHING_QUESTION`,
`FRAGMENTATION_ONLY`, `NO_MATERIAL_ISSUE_CONFIRMED`, `AMBIGUOUS_UNRESOLVED`, or
`OTHER_REVIEW_QUESTION`. Use HIGH/MEDIUM/LOW confidence. Comments are required
for GT, Hidden, evaluator, ambiguous and other questions. Do not guess when the
visual evidence is insufficient; use `AMBIGUOUS_UNRESOLVED`.

The full frame is authoritative. Actor crops are supplemental and include
context margin. The timeline displays context, event, anchor, identities,
Hidden state and error category. The method remains `AUDIT_TARGET_METHOD`
unless the reviewer explicitly reveals it. Aggregate performance is never
shown.

## Commands (Windows CMD)

```text
cd /d C:\Users\ironh\Downloads\PIG_Behavior_Project
set MANIFEST=docs\tracking\gt_audit_gui\TRACKING_GT_AUDIT_REVIEW_MANIFEST_20260730.csv
set AUTHDIR=docs\tracking\development_evidence_defense
set AUDIT=%AUTHDIR%\DEVELOPMENT_GT_ERROR_AUDIT_ITEMS_20260730.csv
set AUTH=%AUTHDIR%\DEVELOPMENT_EVIDENCE_INPUT_AUTHORITY_20260730.json
set RUNROOT=human_review_workspace\tracking_gt_audit\HUMAN_20260730_R1
python scripts\tracking\review_tracking_gt_audit_gui.py --validate-only
python scripts\tracking\review_tracking_gt_audit_gui.py --headless-smoke
python scripts\tracking\review_tracking_gt_audit_gui.py --max-items 5 ^
  --reviewer REVIEWER_NAME ^
  --run-root human_review_workspace\tracking_gt_audit\PILOT_20260730
python scripts\tracking\review_tracking_gt_audit_gui.py ^
  --reviewer REVIEWER_NAME ^
  --run-root human_review_workspace\tracking_gt_audit\HUMAN_20260730_R1
python scripts\tracking\review_tracking_gt_audit_gui.py --resume ^
  --reviewer REVIEWER_NAME ^
  --run-root human_review_workspace\tracking_gt_audit\HUMAN_20260730_R1
python scripts\tracking\check_tracking_gt_audit_coverage.py ^
  --manifest %MANIFEST% ^
  --source-audit %AUDIT% ^
  --input-authority %AUTH% ^
  --decisions %RUNROOT%\tracking_gt_audit_decisions.csv ^
  --events %RUNROOT%\tracking_gt_audit_decision_events.jsonl ^
  --expected-gui-code-sha 2224b587374beea3477d2e325949a20d70d181c8c91031d4c52650db93a38cb8 ^
  --output %RUNROOT%\tracking_gt_audit_coverage.json
python scripts\tracking\summarize_tracking_gt_audit_decisions.py ^
  --manifest %MANIFEST% ^
  --decisions %RUNROOT%\tracking_gt_audit_decisions.csv ^
  --coverage %RUNROOT%\tracking_gt_audit_coverage.json ^
  --output-json %RUNROOT%\tracking_gt_audit_summary.json ^
  --output-csv %RUNROOT%\tracking_gt_audit_summary.csv
```

The coding agent does not open the production GUI or create decisions. Use a
new `RUN_ID` for the first review. Existing roots resume and are never silently
overwritten. Review from the anchor through the full event, extend context when
identity remains unclear, select one decision and confidence, then confirm the
save. GT questions remain review evidence only; a later scientific
reconciliation task decides whether any correction or metric rerun is allowed.

Shortcuts: Space play/pause; Left/Right one frame; Shift+Left/Right ten frames;
Ctrl+Left/Right one second; Home/End event bounds; PageUp/PageDown units; N next
unresolved; G/P/C toggle GT/prediction/combined views; U undo; S opens the
visible save confirmation. Playback speed is adjustable in the control bar.
The GUI shows exact frame, event-relative frame, timestamp, current/total,
reviewed/unresolved counts and current speed.

Completion requires coverage PASS for all review units. `AMBIGUOUS_UNRESOLVED`
counts as complete review coverage but remains a scientific ambiguity.

Final handoff should report the review run ID, reviewer, manifest hash, GUI
code hash, coverage status, reviewed/unresolved counts, decision totals,
scientific ambiguities, GT/evaluator questions, and the location of the
decision CSV, event log, coverage JSON and summary files. Never modify source
videos, GT XML, prediction XML, evaluator code, method registry, or locked
unseen data during review.
