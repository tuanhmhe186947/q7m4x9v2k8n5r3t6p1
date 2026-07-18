# Tracking improvement scenario

Evaluate one isolated association candidate against the locked
`hybrid_bytetrack` parent and the three existing realtime profiles. Bind the
same videos, GT, detector, semantic config, and hardware policy. Use fresh
prediction and report roots, emit no MP4, record raw and canonical prediction
hashes, and evaluate corrected bbox/ID with `include_hidden=true`. Do not make
tracker-derived `Hidden` a target. Reject a geometry-only candidate if any ID,
shape key, Behavior, `Hidden`, `occluded`, or other non-geometry payload changes.
Freeze difficult windows from parent switch events. Run each window with a
past-only warm-up interval that is excluded from scoring, then advance through
one full target video and a hard set containing at least three difficult videos.
Run full-13 only after the hard-set aggregate improves across at least two
difficult videos and every critical guardrail passes. Declare any allowed local
regression budget before execution and report all per-video trade-offs. Repeat
only after the full-13 quality and runtime gates pass. Run
`scripts/audit_tracking_repeatability.py` with current input rehashing and
same-contract aggregate and critical guardrail checks. Only a fresh immutable
JSON lock with `status=PASS` may become authority evidence.
