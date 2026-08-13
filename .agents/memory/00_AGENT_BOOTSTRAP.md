# Agent Bootstrap

Read this bounded entrypoint before loading project history.

1. Run `manage_agent_governance.py bootstrap` from the current registered
   worktree.
2. If the requested task is V2, run `inspect --task-id <ID>` and retrieve only
   the authority receipts named by that record.
3. If the requested task is a legacy V1 capsule, run
   `manage_short_memory.py inspect --task-id <ID>`; do not read every capsule.
4. Resolve current scope authority through `18_AUTHORITY_INDEX.json` and record
   exact receipts before effects.
5. For a material new task, create one V2 record, communicate its plan, confirm
   the plan digest, admit one worktree, and obtain an action permit.
6. At each phase boundary, use atomic `advance` or `amend-plan`. Do not append
   hidden work after a terminal step.
7. Close only after integration or failure-evidence extraction, one learning
   disposition, skill-impact review, and a worktree disposition.

Full history remains available but is cold context:

- `01_PROJECT_MEMORY_SHORT.md`: legacy active capsules and daily handoff;
- `04_PROJECT_MEMORY_MEDIUM.md`: paused or dormant work;
- `05_PROJECT_MEMORY_LONG.md`: accepted durable knowledge;
- archives and rollout summaries: provenance only.

Unknown authority, ownership, evidence, or cleanup state is protected and
fails closed.
