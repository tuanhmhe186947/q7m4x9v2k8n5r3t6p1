# Agent Storage Guard

Agent Storage Guard is a local-only dashboard for reviewing agent sessions,
temporary files, and regenerable caches before moving selected items into the
Windows Recycle Bin.

## Run

From the project root:

```powershell
$env:PYTHONPATH = "$PWD\src"
C:\Users\ironh\anaconda3\envs\pig_project\python.exe `
  -m pig_behavior.storage_cleanup
```

Then open `http://127.0.0.1:8765`.

The installed package also exposes:

```powershell
pig-storage-cleaner
```

## Project-aware recommendations

Every row explains both the suggested action and its likely effect on the pig
project:

- `Dọn trước`: regenerable package/Python cache and old user temp.
- `Cần xem`: Codex chat history, agent scratch, orphan worktrees, or project
  temp that may still contain a patch or lineage evidence.
- `Giữ`: active/recent items, registered worktrees, models, datasets, and
  scientific outputs.

Names containing markers such as `classification_v2`, `tracking`,
`hidden_review`, `checkpoint`, `manifest`, or `oof` are explicitly routed to
lineage review instead of being recommended as disposable cache.

## Safety contract

- The server binds only to `127.0.0.1`.
- A scan is read-only and is limited to registered locations.
- No item is selected automatically.
- Codex sessions changed within 24 hours are protected.
- Session labels use the first real user question after injected project
  instructions and environment context.
- Cache roots changed within one hour are protected.
- Registered Git worktrees are protected. Only unregistered worktree
  directories older than seven days can become selectable.
- Links and Windows reparse points are never selectable.
- Large scientific/project artifacts are review-only.
- The browser sends opaque scan tokens, never arbitrary deletion paths.
- Drill-down children receive new server-side tokens and inherit the approved
  root and protection policy of their parent.
- Selecting a directory automatically supersedes selections inside it; the
  backend also rejects overlapping parent/child selections.
- A preview is bound to an exact item fingerprint and expires after five
  minutes.
- Normal items require the generated confirmation phrase. Protected or
  review-required items remain unselected by default; the local owner may
  explicitly enable the override, review the warning, and type `DELETE`.
  Reparse points and links remain non-overridable.
- Files changed after scanning are rejected.
- The tool has no permanent-delete operation and never empties Recycle Bin.

Codex can retain a stale title in its session index after a session file is
recycled. The tool intentionally does not edit Codex databases, indexes,
authentication files, or configuration.
