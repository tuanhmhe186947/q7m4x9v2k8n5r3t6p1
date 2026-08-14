# G-4 Owner Reconciliation

## Immutable comparison inputs

- Base: canonical `main` at `9dc32b5a4c73d251b8c0dbd547d6a2a5997f847e`.
- Owner: the canonical working tree captured before reconciliation.
- Governance: the validated dedicated repair worktree.
- Owner snapshot: `execution_model_repair_g4_owner_work_preintegration_snapshot.json`
  (`SHA256=1dfdcc49ce15ac11c5e6e9267d27cfc8dd144a4c3c342127ab720954ca82a51b`).
- The snapshot records all 23 dirty/untracked paths, HEAD/current hashes, exact
  diffs, and full patches for both overlapping files. It remains protected
  evidence in this dedicated worktree and is intentionally not staged.

## Semantic hunk audit

| File | Hunk | Class | Owner intent | Governance intent | Resolution |
| --- | --- | --- | --- | --- | --- |
| `manage_agent_governance.py` | MG-1 helpers near `_worktree_identity` | `GOVERNANCE_ONLY` | None | Bound scope and descendant-lineage checks | Keep in release candidate. |
| `manage_agent_governance.py` | MG-2 create path-scope rule and provenance fields | `GOVERNANCE_ONLY` | None | Separate immutable base from accepted/actual task state | Keep in release candidate. |
| `manage_agent_governance.py` | MO-1 administrative rebaseline, head-rebind, and history-reconcile methods | `OWNER_ONLY` | Explicit one-off owner recovery | None | Retain only in canonical owner working tree. |
| `manage_agent_governance.py` | MG-3 expired-permit classification and accepted-progress refresh | `GOVERNANCE_ONLY` | None | Normal, scope-bound same-task progression | Keep in release candidate. |
| `manage_agent_governance.py` | MO-2 permit-renewal method | `OWNER_ONLY` | Extend a still-valid permit administratively | None | Retain only in canonical owner working tree. |
| `manage_agent_governance.py` | MG-4 `advance` accepted-state update | `GOVERNANCE_ONLY` | None | Preserve monotonic accepted state after typed evidence | Keep in release candidate. |
| `manage_agent_governance.py` | MO-3 parser and CLI dispatch for owner recovery commands | `OWNER_ONLY` | Expose owner-only recovery mechanisms | None | Retain only in canonical owner working tree. |
| `test_agent_governance_v2.py` | TO-1 recovery helper, rebaseline, and permit-renewal tests | `OWNER_ONLY` | Cover owner-only command behavior | None | Retain only in canonical owner working tree. |
| `test_agent_governance_v2.py` | TG-1 scoped expired-permit progression regression block | `GOVERNANCE_ONLY` | None | Prove the automatic progression model and fail-closed controls | Keep in release candidate. |
| `test_agent_governance_v2.py` | TC-1 `timedelta` import and terminal test-file append point | `COMPATIBLE_OVERLAP` | Existing owner tests require the original import context and end-of-file tests | New regressions require `timedelta` and append at the same logical boundary | Three-way result keeps both; no conflict marker was produced. |

`EQUIVALENT_DUPLICATE`: none. `INCOMPATIBLE_OVERLAP`: none.

## Combined candidate evidence

`git merge-file -p` produced conflict-free combined candidates for both
overlapping files. The isolated candidate suite passed `65` tests on
2026-08-14. The candidate contains the owner recovery utilities/tests plus the
automatic base/accepted/actual progression model/tests; neither safety surface
was weakened.

## Commit separation

The release-candidate commit stages only the dedicated-worktree governance
repair paths. Owner-only source and tests are intentionally left unstaged in
canonical main. Canonical integration will use the conflict-free combined
working-tree result, then stage only the base-to-governance delta.
