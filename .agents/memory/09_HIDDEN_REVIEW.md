# Classification V2 Hidden Review Memory

## Current reviewed-rebuild authority (2026-08-03)

The original Hidden GUI review is complete in the current reviewed rebuild.
The scientific coverage gate records `5,233/5,233` resolved items, zero missing,
duplicate, or unresolved items, and status `PASS`. Hidden apply consumes the
same manifest and decision hashes, preserves all `245,680` frame rows, and is
the declared input to temporal harmonization.

Hidden notes recorded later during Behavior review are a complementary burst
exclusion layer, not a second competing Hidden ledger. The current unified
window audit records `252` Hidden-note rows, zero note rows left unexcluded,
`84` propagated exclusion rows, `336` excluded frame rows, and a valid audit.

Therefore `HIDDEN_REVIEW_COMPLETE=PASS` for the current reviewed engineering
lineage. This does not authorize paper-grade training by itself and does not
promote Hidden to model X or behavior-label uncertainty.

## Active mixed source boundary (2026-07-20)

The active review population is the mixed legacy 16f P0-P10 export plus the
12 behavior XML files under `data/annotations/classification`. The historical
statement that legacy is outside the main source manifest is superseded for
this rebuild target; it remains a historical boundary for prior experiments.
Both source types require the same current two-sided Hidden review before
reviewed/train-ready use.

## Historical pre-review decision

Operator lineage `c2v2_human_review_20260721_reviewer01_v2` is frozen at the
failed complete-unit Hidden smoke. Its 704-row/64-unit input was valid, but the
builder at code SHA `150b2b9929b412d3882ebc118bc2432185e0987b` incorrectly
used absence of row caps as a full-support semantic switch. The partial smoke
manifest/context CSVs are failure evidence, not authority; v2 cannot resume
after the semantic patch.

The active builder contract requires explicit
`--design-scope {smoke,full}`. Row caps only bound debug input. Smoke keeps all
structural checks and does not require final-support quotas; full enforces the
predeclared quotas and rejects bounded input. Canonical output files are
published only as one validated transaction. A failed build may leave only a
failure audit declaring `no_outputs_published=true`. The next operator lineage
must use a new code SHA and new versioned RUN_ID.

Hidden review is required after enhanced frame features and before temporal
harmonization. Hidden is a frame/object visibility attribute, not a behavior
target and not a native-unit decision.

CVAT Hidden values are tracking-derived and untrusted until current human
review. Legacy values may retain prior provenance as audit metadata, but the
canonical 16f P0-P10 PASS does not establish current human trust. The standalone
legacy 16f lineage still requires two-sided frame/object Hidden review before
reviewed or train-ready use.

Legacy 16f is part of the active mixed rebuild target through the canonical
P0-P10 export identified in README. Existing legacy Hidden metadata does not
satisfy current review coverage; the new mixed lineage must bind and review
its own selected source population.

Hidden smoke and full authority use disjoint roots. Smoke review and apply use
`%HSM%` plus `%HSMDEC%`; full review uses `%HREV%` plus `%HDEC%`. Never direct a
smoke manifest into `%HDEC%`, and never count `%HSMDEC%` toward full coverage.

## Review design

Use four disjoint cohorts:

- census every untrusted `Hidden=Yes` and stratified-audit trusted Yes;
- label-independent high-risk `Hidden=No` enrichment;
- stratified-random `Hidden=No` for false-negative estimation;
- low-risk clean negative controls.

Random rows must store stratum population, inclusion probability and inverse
sampling weight. Report the post-stratified random false-negative estimate
separately from high-risk correction yield.

Temporal proximity is conservative. A previous or next sorted row contributes
adjacent Hidden evidence only when absolute frame-index delta equals 1.
Persistent pair contact/overlap and adjacent bbox instability use the same
valid adjacency gate. Sparse CVAT annotations are not silently treated as
contiguous, and no behavior or target label participates in this risk score.

## Operator contract

Use `review_hidden_quality_gui.py`. It shows full-frame bbox context and a
letterboxed actor crop, then writes decisions only. Do not use the old
direct-source GUI for a new lineage because it can write corrected XML/CSV.

Default coverage rejects missing, duplicate, pending and unclear decisions.
Apply writes `hidden_reviewed_frame_features.csv`, preserves row count and
changes only Hidden plus declared provenance fields.

Hidden review is deliberately one-action: the reviewer chooses Yes, No, or
Unclear. The GUI automatically writes `hidden_review_confidence=high` for a
resolved Yes/No and `low` for Unclear. It also supplies a compatible default
reason. Confidence is compatibility provenance, not an extra human task, and
does not alter sample weight, training inclusion, availability masks, or model
inputs. Blank, invalid, or manually created resolved-low payloads fail coverage.

Do not collect blur, small/distant subject, low light, weak bbox, or frame-edge
flags in the Hidden GUI: they are not model inputs and slow this review. Use
Unclear when image quality prevents a defensible Hidden decision. Occlusion is
the reviewed target; detailed pig/scene reasons remain optional audit metadata.

Unselected CVAT No remains `untrusted_tracking_derived`. Do not silently promote
it to visible trusted metadata. After frame/object apply and harmonization, the
canonical window policy uses current row-level Hidden as conservative visual
burden: main thresholds are total ratio `0.25` and longest-run ratio `0.20`;
robust-only limits are `0.50` and `0.40`; exceeding either robust limit excludes
the window from training and forces its sample weight to zero. This policy is
enabled by default for every T6/T8/T12/T16 view and forces a frame-derived
rebuild instead of fast window reuse.

Hidden ratios, trust, review coverage and policy tier are audit/mask metadata,
never model-X. `--no-exclude-high-hidden-from-main` is an explicit ablation only
and cannot authorize a canonical reviewed lineage.

## Historical evidence and status

The existing technical reference is
`outputs/classification_v2/rebuilds/hidden_review_v6_full_20260714`. Its 5,131
items contain 4,122 Yes confirmations, 384 high-risk No, 601 random No, and 24
clean controls. Behavior is descriptive metadata only; target-derived risk and
sampling fields are absent.

The manifest SHA256 is
`3e4fec14c466a89370a1e20d913cb024bd1dda1fa8db9c1fabdf8a51fa31072e`.
The predeclared design has adequate planned random/high-risk clustered support.
Commits `32eaa2b` and `aaf8460` preserve 30 old payload rows through identifier
v2 and bind migration artifacts technically. The user confirms no review has
started, so those rows have unverified provenance and are not human evidence.

Commit `f2179e3` upgrades media evidence to schema v2 with exact manifest and
frame-context hashes. The 24-item dual-source smoke resolves 12 video and 12
crop items; the full gate resolves 4,613 video and 518 crop items. Both report
zero missing or unknown-source media.

Verified human coverage is 0/5,131. The clean authority must be rebuilt under
`human_review_workspace/classification_v2/<RUN_ID>` with zero carried rows.
The scientific gate is `BLOCKED_INCOMPLETE_OR_INSUFFICIENT_REVIEW`; apply,
temporal rebuild, snapshot, and training remain blocked. Follow section 8A of
`docs/CLASSIFICATION_V2_DATA_REBUILD_AND_HUMAN_REVIEW_RUNBOOK.md`.
