# H2-CDSP validation policy

## Selected policy

Option 2 is selected: freeze a new untouched H2 validation population before
production implementation.

The existing H1 validation windows are not assumed suitable. Their roles were
defined for hidden-owner contention, while H2 targets causal state loss,
uncertainty growth, and state availability at re-entry. They remain untouched
and are not part of H2 development or validation.

## Development status

The six historical positive events and four controls in
`H2_CDSP_SHADOW_DEVELOPMENT_MANIFEST.csv` are mechanism-discovery and future
shadow-development evidence. They have influenced H2 and are permanently
ineligible for independent validation.

Historical `b0d9009` outcomes do not establish current-main prevalence. A
current-main shadow run requires separate authorization and remains development
evidence even if it passes.

## Required validation separation

Before implementation authorization, an independent role assigner must freeze
new H2 validation windows that are disjoint from all H2 development by:

- video key;
- recording date and session;
- source video hash;
- preferably pen/session when reliable metadata exists.

The assigner may use source video, GT, baseline parent evidence, and pre-H2
metadata. The assigner must not inspect H2 outputs, state-quality traces,
activation density, candidate comparisons, or threshold screening.

Each future row must bind:

- window ID and immutable boundaries;
- source video and GT SHA-256;
- recording date/session and pen/session where available;
- positive state-loss or control role;
- concise causal-evidence rationale;
- assigner identity and timestamp;
- assignment-artifact SHA-256.

Ambiguous authority is excluded rather than forced. Event `000216` is excluded
unless its GT authority is independently repaired and reauthorized before
population freeze.

## Blindness and execution

No H2 output may be executed, rendered, scored, or inspected on validation
before separate authorization. Thresholds, decay constants, state transitions,
eligibility, and invalidation rules must be frozen before validation begins and
cannot change after any validation output is viewed.

Validation execution must use shared hashed detector evidence, causal prefix
warmup, delay 0, no future frames, and zero MP4 output.

## Current status

- New H2 validation population frozen: `NO`;
- H2 validation outputs exist: `NO`;
- Development/validation overlap: `NONE`;
- Validation execution authorized: `NO`;
- Production implementation authorized: `NO`.

The absence of a frozen H2 validation population blocks production
implementation authorization, but it does not block a separately authorized
current-main shadow prerequisite because that prerequisite is development only.
