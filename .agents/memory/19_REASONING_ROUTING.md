# Mandatory Reasoning Routing

## Gate

For a nontrivial task, select the required reasoning group before a code or
domain skill. Record selected skills and purpose in the plan or run manifest.
A code skill cannot satisfy this gate by itself.

| Task class | Required reasoning skills |
| --- | --- |
| Architecture or goal drift | `agent-architecture-audit`, `plan-orchestrate` |
| Agent behavior debugging | `agent-introspection-debugging` |
| Handoff quality | `agent-self-evaluation` |
| Agent task evaluation | `agent-eval`, `eval-harness` |
| Context or memory | `iterative-retrieval`, `knowledge-ops` |
| Action, tool, observation | `agent-harness-construction` |

`plan-orchestrate` is generative only. Reading it satisfies architecture
coverage, but it must not be represented as having executed implementation.

## Enforcement

`11_SKILL_PORTFOLIO.json` is the machine-readable routing authority. The
governance validator rejects missing task classes, missing required skills, or
a project skill record without version/hash, dependencies, review/use dates,
proof task, stale signal, and next action.
