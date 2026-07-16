# Tracking improvement scenario

Evaluate one isolated association candidate against the locked
`hybrid_bytetrack` parent and the three existing realtime profiles. Bind the
same videos, GT, detector, semantic config, and hardware policy. Use fresh
prediction and report roots, emit no MP4, record raw and canonical prediction
hashes, repeat the candidate, and reject it if any declared per-video identity
or runtime guardrail regresses. Run `scripts/audit_tracking_repeatability.py`
with current input rehashing and the profile's per-video IDSW guards. Only a
fresh immutable JSON lock with `status=PASS` may become authority evidence.
