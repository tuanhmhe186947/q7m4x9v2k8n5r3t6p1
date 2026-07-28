# Development tracking 2×2 Standard-V2 conclusion

The frozen development 2×2 metric authority is complete under
`TRACKING_EVALUATOR_STANDARD_V2`, with `include_hidden=true`.

R0 has the best HOTA (`0.8881872315857098`) and IDF1
(`0.9718924002607702`). R1 has the lowest IDSW (`18`), while R0 has the
lowest wrong-ID exposure (`11,893` matched frames) and the fewest terminal
identity-error episodes (`12`). R0 therefore remains the strongest overall
development arm under the predeclared hierarchy.

The ByteTrack repair effect is `MIXED_TRADEOFF`. B1 slightly improves HOTA and
reduces IDSW and terminal episodes, but lowers IDF1 and increases wrong-ID
frames and persistent pairwise swaps. The frozen repair stack is not broadly
beneficial for ByteTrack under Standard V2.

The RF repair effect is `BROADLY_HARMFUL` under the predeclared deterministic
rule. Relative to R0, R1 lowers HOTA by `0.0099064555078777` and IDF1 by
`0.0140110506684905`, increases wrong-ID exposure by `2,622` frames, and adds
two terminal episodes. R1 does reduce IDSW by `11` and persistent pairwise
swaps by one, but those supporting improvements do not override the
co-primary and identity-severity regressions. R1 does not provide a
scientifically defensible offline-quality benefit over R0 on development.

Repair behavior differs across the complete pipelines, but the interaction is
metric-specific and must not be collapsed into a single score. The HOTA
interaction is `-0.0102684414790773`, the IDF1 interaction is
`-0.0074458797283192`, the IDSW interaction is `9`, the wrong-ID-frame
interaction is `2,343`, and the terminal-episode interaction is `3`.

Cross-core contrasts and interactions are whole-pipeline statements. They
include the profile-specific detector cadence: every frame for B0/B1 and every
two frames for R0/R1. They are not pure association-core effects. R1 is a
post-video offline method that uses future frames and cannot support a
realtime claim.

B1 event-level attribution is unavailable because its promoted frozen
authority exposes neither a raw pre-repair artifact nor a repair-event ledger.
This prevents repair-stage or event-level comparison across cores, but it does
not prevent prediction-level Standard-V2 comparison. The 2×2 interaction
remains a metric-level interaction. R1 event attribution is diagnostic only
and was not used for repair-effect classification or interaction.

Future generation authorities should expose immutable raw pre-repair outputs
and deterministic repair ledgers at generation time. The present result does
not promote or remove a method. B1's active-promotion status requires a
separate authority decision informed by this corrected Standard-V2 result.
