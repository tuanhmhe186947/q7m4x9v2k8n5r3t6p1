# Tracking improvement scenario

First evaluate one isolated candidate against the locked `hybrid_bytetrack`
parent. Do not tune or compare realtime profiles in the same experiment. Bind
the same videos, GT, detector, semantic config, and hardware policy. Use fresh
prediction and report roots, emit no MP4, record raw and canonical prediction
hashes, and evaluate corrected bbox/ID with `include_hidden=true`. Do not make
tracker-derived `Hidden` a target. Reject a geometry-only candidate if any ID,
shape key, Behavior, `Hidden`, `occluded`, or other non-geometry payload changes.
Freeze difficult windows from parent switch events. Run each window with a
past-only warm-up interval that is excluded from scoring, then advance through
one full target video and a hard set containing at least three difficult videos.
If a strictly post-video geometry pass cannot reproduce the parent's state
after tracker reset, replay it from the hash-matched parent shapes JSON and XML.
Require equal shape keys and non-geometry payload, and score the frozen window
before evaluating the full replay. Audit that pair with
`--post-video-geometry-replay`: every video must explicitly declare
`telemetry_available=false`, prediction runtime artifacts must be absent, and
tracker FPS/RSS guardrails must be recorded as `NOT_APPLICABLE`.
Run full-13 only after the hard-set aggregate improves across at least two
difficult videos and every critical guardrail passes. Declare any allowed local
regression budget before execution and report all per-video trade-offs. Repeat
only after the full-13 quality and runtime gates pass. Run
`scripts/audit_tracking_repeatability.py` with current input rehashing and
same-contract aggregate and critical guardrail checks. Only a fresh immutable
JSON lock with `status=PASS` may become authority evidence.
A candidate authority closes only that experiment. Recompute the residual
hybrid events and continue isolated hybrid work until a separate, predeclared
lane-completion gate passes. Only then may a proven mechanism enter a separate
realtime experiment. Use `realtime_fast` as the operational reference and
require balanced to pass a predeclared identity-stability/latency gate relative
to fast, not merely improve over an older balanced result.
