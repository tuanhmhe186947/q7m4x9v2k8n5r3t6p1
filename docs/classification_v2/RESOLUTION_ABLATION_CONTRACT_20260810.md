# Resolution Ablation Contract — Experiment Pending

Status: engineering-capable only. This contract does not select a resolution,
activate a backbone, or establish a performance result.

## Scientific binding

One scientific RGB observation is the immutable source media/frame, actor or
local track scope, authoritative exact bounding box, source/video and frame
identity, crop provenance, split role, and temporal observation identity. Its
identity hash excludes resolution. `input_resolution` is a runtime realization
field restricted to `64`, `160`, or `224`, with a separate realization hash.

The transform remains exact actor crop → aspect-preserving letterbox → requested
square resolution. No margin, partner context, augmentation, H5, social/ROI or
motion input is introduced by this infrastructure.

## Future matched resolution ablation

The single changed variable is input spatial resolution: `64`, `160`, `224`.
Keep the smoke CNN, temporal view, feature inputs, seed, optimizer,
4164-optimizer-step budget, train/validation population, event weights, and
evaluator identical. Do not use outer predictions to choose a setting and do
not combine this comparison with a backbone or pretraining change.

The later encoder/pretraining ablation is separate: hold one selected resolution
constant while comparing smoke CNN with one explicitly registered pretrained
backbone. Its preprocessing and pretrained weight enum need their own binding.

## Interpretation boundary

`H_SPATIAL_FIDELITY`: higher resolution may preserve actor-local detail that
64×64 removes.

`H_INTERACTION_CONTEXT`: an exact actor box can omit a partner or relation needed
for fight. Increasing 64→160/224 cannot restore pixels outside that crop.

Neither hypothesis implies a performance result. Future experiments must keep
them causally separate.
