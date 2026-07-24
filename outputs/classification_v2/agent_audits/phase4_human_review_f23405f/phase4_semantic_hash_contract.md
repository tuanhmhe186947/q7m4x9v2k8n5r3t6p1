# Phase 4 semantic hash contract

This package is non-official audit evidence. It authorizes no rebuild.

- Canonicalization: UTF-8 canonical JSON, sorted object keys, compact
  separators, normalized newlines and repo-relative `/` paths.
- Ordered schema lists preserve order. Declared unordered collections sort
  by canonical JSON bytes.
- NaN and Infinity are rejected; negative zero is normalized.
- Generated timestamps and self-hash fields do not affect semantic hashes.
- Algorithm: SHA-256.
- Semantic bundle: `bundle.classification_v2.phase1_4`
- Bundle version: `classification_v2.semantic_bundle.v4`
- Bundle hash: `4e2f03edd7b685d0fc7811027fa787ef0a2903d7c865f4bdafd22565cebb0a4b`
- Whole-repository Git SHA is provenance, not the only stage code authority.
- Official outputs require validated bytes and a matching manifest promoted
  last.
