# Phase 4 human review checklist

- [ ] Confirm all 17 exact contract stages and dependency edges.
- [ ] Confirm all 17 semantic domains and authority mappings.
- [ ] Reproduce canonical hash golden cases independently.
- [ ] Confirm mapped production changes alter stage code hashes.
- [ ] Confirm docs/tests/audits do not invalidate production artifacts.
- [ ] Inspect the bounded inventory and preserved stopped-lineage reasons.
- [ ] Confirm current rebuild start is `stage.frame_local_primitives`.
- [ ] Confirm Hidden carry-forward requires exact visual/key authority.
- [ ] Confirm Behavior carry-forward requires exact unit/task authority.
- [ ] Confirm interrupted promotion leaves no authoritative partial output.
- [ ] Confirm every release authorization is false before sign-off.
- [ ] Record reviewer, date, decision, and exact implementation SHA.

Proposed decision token:
`ACCEPT_PHASE_4_IMPLEMENTATION_AND_AUTHORIZE_FRAME_LOCAL_REBUILD_PLANNING`

Acceptance does not itself run a rebuild or authorize later stages.
