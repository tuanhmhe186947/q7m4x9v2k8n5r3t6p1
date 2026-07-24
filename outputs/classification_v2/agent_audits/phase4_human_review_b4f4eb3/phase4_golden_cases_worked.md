# Phase 4 worked golden cases

## Canonical semantic hash

`SHA256(UTF8(canonical_json(payload_without_ephemeral_fields)))`

Dictionary insertion order and JSON whitespace do not change the bytes.
Changing threshold `0.08` to `0.081` changes the semantic hash. Swapping
ordered motion features changes the hash.

## Stage execution fingerprint

The fingerprint hashes stage ID/version, mapped production-code blob hash,
stage semantic-domain hash, upstream artifact fingerprint, and schema hashes.
Tests, caches, generated audits, and non-authoritative documentation are not
production-code blobs.

## Current rebuild derivation

Changed accepted domains are mapped to their direct stages and the dependency
closure is traversed. The earliest topological direct stage is
`stage.frame_local_primitives`. This is computed, not injected as an artifact
status.

## Human decisions

An identical stable key is necessary but insufficient: identity, span, visual
media, review task, and decision schemas must also match. Changed visual
authority requires human revalidation. New-only units are never auto-accepted.

## Release preflight

Missing manifests, hash mismatch, stale semantics, stopped lineages,
diagnostic-only evidence, audit-only evidence, or missing Phase 4 sign-off
leave every official authorization false.
