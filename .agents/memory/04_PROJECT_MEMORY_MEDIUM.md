# Project Memory - Medium Term

## Lifecycle contract

- Scope: paused/dormant work, blockers, hypotheses, and temporary decisions
  that must survive beyond the active short-memory working set.
- Do not copy an active managed task here merely because the date changed.
  Admit it only when explicitly paused and remove its active short capsule so
  one task never has two current execution-state authorities.
- Every active item needs `status`, `opened`, `next_action`, `evidence`, and
  an exit condition.
- Review active entries at each session closeout and at least every seven days.
  The cadence detects neglect; it is not evidence of correctness or permanence.
- When resolved, register reusable knowledge in `21_MEMORY_MATURITY.json`.
  Promote only after typed evidence, deliberate acceptance, source disposition,
  and revalidation gates pass. Completion or elapsed inactivity is insufficient.
- Archive or delete completed operational detail that has no future reuse value.
- Never copy raw chat, command logs, or an error without a validated correction.

## Active cross-day entries

- `thesis_writing.classification_v2`
  - Status: `VIETNAMESE_SECTION_2_2_DRAFTED`
  - Opened: `2026-08-02`.
  - Next action: user review of the Vietnamese Chapter 2, Section 2.2 draft;
    then draft Section 2.3 before beginning the English conversion pass.
  - Evidence: user-confirmed thesis title and project narrative; blueprint
    `docs/CLASSIFICATION_V2_THESIS_BLUEPRINT_EVIDENCE_MAP.md`; draft
    `docs/thesis_drafts/CHAPTER_2_1_OVERVIEW_FRAMEWORK_VI_DRAFT.md`.
  - Scope: three-layer pipeline, ten-class behavior classification,
    long-term behavioral deviation screening, and offline/near-real-time use.
  - Visual requirement: every major subsection must have a real study image,
    pipeline diagram, reproducible plot, table, or explicit pending visual;
    inventory is `docs/thesis_drafts/THESIS_FIGURE_AND_TABLE_PLAN.md`.
  - Writing workflow: use `.agents/skills/thesis-evidence-writing/` for the
    Vietnamese meaning pass, user confirmation, original English rewrite, and
    evidence/visual alignment; comparison is recorded in
    `docs/THESIS_SKILL_COMPARISON_EXTERNAL_REPOSITORIES.md`.
  - Structural revision: Section 2.1 was rewritten as a connected narrative;
    `Figure 2` is cited in the prose and its visual role is stated explicitly.
    English conversion remains blocked on user confirmation of the meaning.
  - User refinement: the overview must describe what enters the model rather
    than explain the annotation-rebuild workflow. The current revision removes
    CVAT/native-burst/harmonization detail from Section 2.1 and states the
    temporal input configurations (6/8/12/16 frames plus sampled six-frame
    inputs) and RGB-derived feature families directly.
  - Section 2.2 initial Vietnamese draft now covers the user-confirmed study
    setting, RGB/depth acquisition boundary, 6 fps to 30 fps source-time
    conversion, and the distinction between native units and model windows.
    Figures 3 and 5 are its visual anchors; final source manifests remain
    pending.
  - Reviewer feedback was accepted with scope corrections: define CVAT
    six-frame and legacy 16-frame bursts, shorten screening language, and label
    posture/near-real-time branches as planned or experimental where evidence is
    not yet complete.
  - User correction: the camera recorded RGB and depth, but depth is not used
    by the current validated behavior-classification pipeline. Figure 2 must
    show RGB-derived current inputs only; depth belongs to acquisition context
    and future work unless a separate contribution experiment is registered.
    Real RGB scene frames are available under `data/raw/images_clean/`, and
    actor crops are available under the legacy recovery `crops/` directory.
  - Boundary: anomaly has no separate ground truth and must not be presented as
    supervised diagnosis; posture is a planned main experiment pending its
    authority and evaluation.
  - Exit: user approves the Vietnamese content and all evidence anchors are
    bound before English conversion.

- `classification_v2.reviewed_lineage`
  - Status: `REVIEWED_ENGINEERING_SNAPSHOT_FROZEN_AWAITING_NEXT_EXECUTION`.
- Current review-close authority: `3,243` reviewed units; the two fixed-point
  `explore` decisions remain unchanged by explicit user decision.
- Opened: `2026-07-12`.
- Next action: choose a separate authorized execution path: materialize the
  original-label-only replay sidecar, freeze posture authority, or launch a
  bounded reviewed-data training phase. Do not access the final test.
- Evidence: reviewed rebuild root
  `C:\pig_runs\classification_v2_reviewed_rebuild_20260802_v1`, commits
  `d0fe99f`, `879cecc`, `6c83bd7`, low-memory commit `a034440e`, and final
  contract commit `e666d85`. Snapshot `reviewed_engineering_4c430dfae2d193dc`
  binds the 24-shard memmap, bounded smoke/checkpoint evidence, and split hash
  `557156a7eb6cceeb6a91f667f7c51dcb286e3111f35f414970fa7431acc7e63b`.
- Source: current classification authority and user-approved project goal.
- Valid from: `2026-07-12`.
- Review after: `2026-08-07`.
- Supersedes: earlier unreviewed-lineage execution assumptions.
- Invalidation condition: any reviewed rebuild, split, ROI, mask-contract, or
  smoke hash fails validation.
- Exit: a separately authorized replay, posture-ablation, or reviewed-data
  training run is selected and registered without opening the final test.

- `autoresearch.tracking_campaign`
  - Status: `BLOCKED` after harness `DEV_PASS`.
  - Opened: `2026-07-31`.
- Next action: register a separate campaign method, freeze its baseline and
  acceptance gates, then have an independent reviewer issue one bound permit.
- Evidence: `02_CURRENT_DECISION.md`, `13_METHOD_STATE.md`, and the focused
  autoresearch regression suite.
- Source: expired `01_PROJECT_MEMORY_SHORT.md` handoff.
- Valid from: `2026-07-31`.
- Review after: `2026-08-07`.
- Supersedes: the unconstrained editable-template execution flow.
- Invalidation condition: campaign authority, policy, or evaluator contract
  changes before registration.
- Exit: the first authorized trial completes or the campaign is closed.

- `governance.live_agent_baseline`
  - Status: `PENDING`.
  - Opened: `2026-07-31`.
- Next action: run the pinned live-agent task and judge pair at least three
  times without changing either definition.
- Evidence: fixture suite passed, but fixture evidence is explicitly not
  live-agent capability evidence.
- Source: expired `01_PROJECT_MEMORY_SHORT.md` handoff and
  `02_CURRENT_DECISION.md`.
- Valid from: `2026-07-31`.
- Review after: `2026-08-07`.
- Supersedes: none.
- Invalidation condition: the pinned task or judge definition changes.
- Exit: a reproducible live-agent baseline is recorded with run lineage.

Sections below preserve prior medium-term context. Where they conflict, the
active entry above and `02_CURRENT_DECISION.md` have precedence.

## 2026-07-12 active classification_v2 focus

The current medium-term project focus is the `classification_v2` Q2 behavior
recognition roadmap, not tracking ablation.

Current state:

- The data/review/train-ready path has been upgraded toward a multimodal
  spatio-temporal design using bbox actor images, ROI relations, motion,
  social/partner context, and leakage-safe tabular features.
- The technical reference has 245,664 enhanced rows and a target-independent
  5,131-item Hidden v6 design. Old decision payloads are unverified.
- Clean human coverage starts at 0/5,131 Hidden and 0/4,670 behavior units under
  a new `human_review_workspace/classification_v2/<RUN_ID>` root.
- Therefore no reviewed train-ready snapshot is currently valid for new model
  experiments. Complete review and freeze new hashes before model smoke.
- A full 13-fold engineering OOF run exists at commit `18d6692`, but it belongs
  to the previous unreviewed lineage and is not the final Q2 result.
- Canonical actor cache remains letterboxed. Rebuild it or verify its hash
  against the future reviewed snapshot before reuse.
- Current status authority is `docs/CLASSIFICATION_V2_CURRENT_STATE.md`.

The tracking notes below are preserved because they still matter if the user
returns to tracking. They are not the active workstream.

## Historical tracking memory

This week the main focus is recovering and tuning the tracking pipeline after
the architecture was split from legacy `tracking_engine.py` into
`src/pig_behavior/tracking/*`.

Key points:

1. Legacy 21/06 had one one-way tracking flow and no `cfg.mode`.
2. Current code has multiple modes, and `hybrid_bytetrack` is not fully equivalent to legacy.
3. Current best baseline is `hybrid_bytetrack + iou0_area0_condarea0_merge0`.
4. `Pigs291119_000302_30fps` improved mainly because of the new detector weight.
5. `Pigs291119_000263_30fps` increased IDSW from ≈2 to ≈6 with both old and
   new weights on current code, so the cause is code/pipeline behavior.
6. Main suspect: `association.py`, especially raw_id logic and `all_detection_indices` matching.
7. Secondary suspects: forced post-processing by mode and detection filtering
   differences from legacy.
