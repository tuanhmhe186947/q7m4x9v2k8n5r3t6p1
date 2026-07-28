# Unseen Confirmatory Claim Contract — 2026-07-28

## Authority

- `UNSEEN_ROLE=CONFIRMATORY`
- `UNSEEN_METHOD_SELECTION_COMPLETE=YES`
- `UNSEEN_TUNING_AUTHORIZED=NO`
- `UNSEEN_METHOD_RESELECTION_AUTHORIZED=NO`
- Primary method: frozen `realtime_fast`.
- Technical baseline: frozen `bytetrack_raw`.

This contract is frozen before any unseen-data access. It does not authorize
unseen preparation, execution, evaluation, tuning, promotion, or method
reselection.

## Authorized future unseen questions

1. Does frozen `realtime_fast` retain acceptable tracking quality on unseen
   same-barn sessions?
2. Does frozen `realtime_fast` outperform or remain competitive with the raw
   `bytetrack_raw` technical baseline on the unseen population?
3. Are the development identity advantages of `realtime_fast` preserved across
   unseen lighting, session, and crowding conditions?
4. What is the distribution of performance across unseen videos and sessions?

## Questions not authorized

- Whether `hybrid_bytetrack` can be rescued on unseen data.
- Whether `rf_hybrid_offline` becomes beneficial in a favorable unseen
  subgroup.
- Which repair stage should be tuned.
- Whether a different detector cadence would improve `realtime_fast`.
- Whether tracking thresholds should be changed.
- Whether development methods should be redesigned after unseen results are
  visible.

## Interpretation boundary

The authorized `realtime_fast - bytetrack_raw` contrast is a whole-pipeline
comparison including profile-specific detector cadence. `realtime_fast`
detects every two frames; `bytetrack_raw` detects every frame. This contrast
does not isolate a pure association-core effect.

The future statistical unit is video, with session or day as a secondary
cluster. Frames are not independent statistical units. No unseen video may be
selected or dropped based on its result.
