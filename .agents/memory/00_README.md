# Codex Project Memory

## V2 bootstrap and authority

For new material work, read `.agents/memory/00_AGENT_BOOTSTRAP.md` and run the
manager `.agents/skills/project-state-steward/scripts/manage_agent_governance.py`
with the
V2 manager lifecycle (`bootstrap`, `create`, `confirm-plan`, skill-read receipt,
`permit`, `advance`/`amend`, `review-outcome`, `close`). The canonical skill
authority is `.agents/skills/skill_inventory.json`; lifecycle is tracked in
`.agents/memory/22_WORKTREE_LIFECYCLE.json`; registry, portfolio, and
README files are generated/supporting views. Existing V1 capsules use the V1
manager compatibility path until explicitly migrated.

## Memory lifecycle

- `01_PROJECT_MEMORY_SHORT.md`: daily state, active managed resume capsules,
  and one-day closeout.
- `04_PROJECT_MEMORY_MEDIUM.md`: paused/dormant work and live hypotheses, never
  a duplicate of an active short-memory capsule.
- `05_PROJECT_MEMORY_LONG.md`: stable goals, contracts, and complete project facts.
- `17_PROJECT_MEMORY_SHORT_ARCHIVE_2026-07-31.md`: migration history.
- Short memory must declare `Opened` and `Expires`.
- Planned material prompts use the atomic short-memory manager before their
  first effect; managed task blocks are never patched manually.
- Checkpoint `DONE` before the next step's first effect. Recover interrupted
  `IN_PROGRESS` work by verifying evidence, never by rerunning it blindly.
- At expiry, atomically retain nonterminal managed tasks byte-for-byte, reduce
  terminal tasks to a one-day closeout, and reset only daily state. Move a task
  to medium only when it is explicitly paused or removed from the active set.
- Promote medium to long only through evidence maturity, explicit review, and
  project-wide acceptance. Elapsed inactivity never proves durability.
- Historical material belongs in an archive, not active short memory.

Long-lived governance authorities:

- `12_PROJECT_CHARTER.md`
- `13_METHOD_STATE.md`
- `14_CLAIM_REGISTRY.md`
- `15_AGENT_REGRESSION.md`
- `16_HALT_CONDITIONS.md`
- `18_AUTHORITY_INDEX.md` and `18_AUTHORITY_INDEX.json`
- `19_REASONING_ROUTING.md`
- `21_MEMORY_MATURITY.md` and `21_MEMORY_MATURITY.json`

After checking short-memory expiry, use `18_AUTHORITY_INDEX.json` to select one
scope-specific current authority. Read archive files only for provenance or a
declared contradiction audit.

This folder contains project-specific memory and rules for `PIG_Behavior_Project`.

Read order for normal tasks:

1. `01_PROJECT_MEMORY_SHORT.md`
2. `02_CURRENT_DECISION.md`
3. `03_PROJECT_RULES.md`
4. `08_WORKFLOW.md`

Read order for larger architecture/tracking tasks:

5. `04_PROJECT_MEMORY_MEDIUM.md`
6. `05_PROJECT_MEMORY_LONG.md`
7. `06_BENCHMARK_NOTES.md`
8. `07_LEGACY_DIFF_NOTES.md`

For classification data/review work, also read:

9. `09_HIDDEN_REVIEW.md`
10. `docs/CLASSIFICATION_V2_CURRENT_STATE.md`

Important:

- Current classification status is authoritative in
  `docs/CLASSIFICATION_V2_CURRENT_STATE.md`.
- The top current section of `02_CURRENT_DECISION.md` overrides later historical
  sections in that file.
- A dated model report is authoritative only for its own data/config/code hashes.
- `.agents/skills/` is for Codex skills.
- `.agents/memory/` is for project memory/rules.
- Root `AGENTS.md` is the main Codex instruction entrypoint.
