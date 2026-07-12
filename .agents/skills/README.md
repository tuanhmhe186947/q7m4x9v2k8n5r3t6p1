# classification_v2 Project Skills

This is the native project-local skill root discovered by Codex. Each active
skill may be invoked implicitly when its description matches the task. Future
skills require explicit invocation because their `openai.yaml` policy disables
implicit use.

## Routing Index

| Order | Skill | Status | Invoke for | Do not invoke for |
|---:|---|---|---|---|
| 1 | `safe-refactor-test-guardian` | active | classification code changes | read-only discussion |
| 2 | `dataset-contract-leakage-guard` | active | data, folds, X schema | unrelated tracking |
| 3 | `experiment-lineage-reproducibility` | active | model run/resume | prose planning |
| 4 | `scientific-ablation-controller` | active | ablation/promotion | fixed smoke |
| 5 | `multimodal-sequence-model-builder` | active | model modules and modes | dataset-only audit |
| 6 | `grouped-cv-evaluation` | active | OOF or prediction comparison | model construction |
| 7 | `imbalanced-classification-evaluator` | future | post-pilot loss audit | pre-pilot work |
| 8 | `gpu-training-profiler` | future | post-pilot GPU profiling | CPU-only data audit |

## Dependency Order

Use the first six skills in routing order for a new classifier change. Evaluation
depends on the frozen data contract and lineage metadata. The model builder
depends on the refactor guard, leakage guard, lineage contract, and ablation
decision. Future skills may be invoked explicitly only after a one-fold pilot.

Shared deterministic resources:

- [templates](templates)
- [read-only checks](checks)
- [synthetic examples](examples)
- [machine routing contract](skill_registry.json)

No skill authorizes full OOF, edits raw data, or changes the scientific target.
