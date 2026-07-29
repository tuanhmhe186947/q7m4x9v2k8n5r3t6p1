# Tracking paper table authority

Table A contains only the two methods with current execution authority and
runtime-benchmark eligibility: `bytetrack_raw` and `realtime_fast`. Their
comparison is a complete-method comparison. It is not a pure
association-core ablation because detector cadence, producer semantics, and
stage topology differ.

Table B separates development references and ablations from executable
primary methods. The exact historical numerical runtime for
`hybrid_bytetrack` is unavailable. Its surviving 13 historical XML files
remain the development prediction authority, while the recovered accepted
lineage remains the algorithmic authority.

The standardized B1 artifact is not `hybrid_bytetrack` and must not replace
the historical XML authority. It remains forensic-only.

Neither `rf_hybrid` v1 nor the rejected v2 candidate improves overall
tracking quality relative to `realtime_fast`. V1 reduces
`IDSW_STANDARD` while degrading HOTA, IDF1, and wrong-identity exposure. V2
improves several identity-event measures but fails the predeclared HOTA and
IDF1 non-regression gates. IDSW alone is therefore insufficient for ranking
identity quality.

`rf_hybrid` v2 is not a fifth active scientific method. The canonical active
`rf_hybrid` remains version 1 as a frozen mixed transfer ablation.
