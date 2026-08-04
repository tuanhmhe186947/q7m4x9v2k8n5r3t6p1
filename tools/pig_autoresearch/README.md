# Project Autoresearch Harness

This directory adapts the experiment loop from
[`karpathy/autoresearch`](https://github.com/karpathy/autoresearch) at pinned
commit `228791fb499afffb54b46200aca536f79142f117`. It keeps one editable
surface, one bounded trial, and a `keep`/`discard`/`crash` ledger while
applying this project's scientific authority and lineage gates.

## Control Boundary

The agent may edit only `candidate.json`. It must not edit `train.py`,
`harness.py`, `policy.json`, source, evaluators, or authority files during
a campaign. Every candidate changes exactly one declared parameter.

`train.py` is an immutable launcher. `harness.py` validates the candidate,
builds a shell-free command, enforces the fixed budget, consumes a bound
single-use authorization, checks worktree drift and forbidden MP4 artifacts,
and writes structured observations. It never resets the repository.

## Workflow

1. Read the project authority required by `AGENTS.md`.
2. Edit only `candidate.json`; use a new `run_tag` and one parameter.
3. Run `python tools/pig_autoresearch/prepare.py` or
   `python tools/pig_autoresearch/train.py --dry-run`.
4. Stop if `authorization_eligible` is false. Register or transition the
   campaign method through the project method-state process; do not edit the
   registry as a shortcut.
5. A human or separately authorized reviewer copies the printed bindings into
   a new permit based on `authorization.example.json`, then stores it below
   `.agents/authorizations/autoresearch/`.
6. Execute once:

```powershell
python tools/pig_autoresearch/train.py --execute `
  --authorization .agents/authorizations/autoresearch/<permit>.json
```

The permit binds the candidate, policy, Git SHA, complete dirty-worktree
fingerprint, method state, stage, budget, and tracking baseline. A create-only
claim file prevents reuse. Any candidate, policy, code, or untracked-file
change after review invalidates the permit.

## Authority-Aware Modes

- Tracking is currently `FROZEN`. The frozen baseline cannot be optimized.
  Execution requires a separately registered campaign method in an allowed
  state, a hash-bound baseline CSV, and aggregate IDSW, per-video IDSW, and
  aggregate HOTA gates. A `keep` result is trial evidence, not promotion.
- Classification is currently `BLOCKED`. The only allowed action is
  `synthetic_preflight`. Its result is always `diagnostic`, never `keep`
  or model-performance evidence.

## Artifacts

Each run writes below `outputs/autoresearch/<run_tag>/`:

- `run_manifest.json`: exact command, bindings, candidate, and Git state.
- `run_state.json`: heartbeat and terminal status.
- `run.log`: child process output.
- `run_result.json`: decision, guardrails, failure, and claim boundary.

The append-only campaign ledger is `outputs/autoresearch/results.jsonl`.
Output directories must be fresh. Invalid metrics, extra/missing videos,
duplicate rows, non-finite values, MP4 artifacts, timeout, nonzero exit, or
worktree drift produce `crash`/error rather than evidence.

## Environment

The harness uses the current Python interpreter and sets project `src/` on
`PYTHONPATH` for child processes. Install the normal project dependencies;
there is no separate autoresearch environment. Preflight is read-only and
does not run tracking, training, or evaluation.
