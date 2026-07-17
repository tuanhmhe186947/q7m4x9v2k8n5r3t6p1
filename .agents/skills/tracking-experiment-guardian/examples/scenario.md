# Tracking improvement scenario

Evaluate one isolated association candidate against the locked
`hybrid_bytetrack` parent and the three existing realtime profiles. Bind the
same videos, GT, detector, semantic config, and hardware policy. Use fresh
prediction and report roots, emit no MP4, record raw and canonical prediction
hashes, and evaluate corrected bbox/ID with `include_hidden=true`. Do not make
tracker-derived `Hidden` a target. Reject a geometry-only candidate if any ID,
shape key, Behavior, `Hidden`, `occluded`, or other non-geometry payload changes.
Repeat only after every per-video identity and runtime guardrail passes. Run
`scripts/audit_tracking_repeatability.py` with current input rehashing and
same-contract per-video IDSW guards. Only a fresh immutable JSON lock with
`status=PASS` may become authority evidence.
