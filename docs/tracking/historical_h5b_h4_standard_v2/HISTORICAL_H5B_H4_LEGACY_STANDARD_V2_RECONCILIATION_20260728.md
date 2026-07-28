# Historical H5b/H4 Legacy–Standard-V2 Reconciliation

- Legacy evaluator: `TRACKING_EVALUATOR_LEGACY_V1`
- Legacy HOTA: `0.9835062270290739`
- Standard-V2 HOTA: `0.9002906560906596`
- Legacy IDF1: `0.9914903846153846`
- Standard-V2 IDF1: `0.9915010683760683`
- Legacy IDSW: `0`
- Standard-V2 IDSW: `0`

The prediction bytes are unchanged. The corrected values differ only because
Standard V2 uses pre-assignment eligibility, video-isolated sequences,
Hidden-inclusive matching, the frozen 19-alpha HOTA set, and standard
last-match IDSW semantics. Therefore this is evaluator reconciliation, not a
tracker regression.

Legacy IDSW=0 survives Standard V2: `YES`.

The legacy HOTA is close to the Standard-V2 alpha-0.05 diagnostic
(`0.983527569`) rather than the required 19-alpha mean (`0.900290656`).
The historical arm still leads current B0, B1, and R0 on HOTA and IDF1, so
the legacy ranking survives even though the legacy absolute HOTA is not a
Standard-V2 value.

The earlier strong visual impression is consistent with the corrected
identity metrics: there are no Standard-V2 ID switches, terminal episodes, or
persistent pairwise swaps. The lower 19-alpha HOTA reflects the full
localization-threshold integration and does not contradict that identity
stability.
