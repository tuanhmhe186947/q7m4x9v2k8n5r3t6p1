"""Fail-closed Temporal-v2 model-input authority loader.

Temporal membership is emitted by the frozen release.  S1 and resolution
consumers may choose only corpus, view, target, and spatial resolution.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MATCHED = "MATCHED_TEMPORAL_ABLATION"
FULL = "FULL_NONOVERLAP_VIEW_POOL"
VIEWS = frozenset({"T6", "T8", "T12", "T16"})
RESOLUTIONS = frozenset({64, 128, 160})
_HASH_VERIFICATION_CACHE: dict[Path, tuple[tuple[str, int, int], ...]] = {}


class TemporalV2ConsumerError(ValueError):
    """Raised when canonical Temporal-v2 input authority is unsafe."""


@dataclass(frozen=True, slots=True)
class TemporalV2Target:
    """One immutable model-facing target, independent of image resolution."""

    target_id: str
    corpus: str
    view: str
    frames: tuple[int, ...]
    source_type: str
    dataset_id: str
    video_key: str
    actor: str
    behavior: str
    split: str
    group: str
    observed_mask: tuple[bool, ...]
    boundary_reset: bool
    temporal_features: dict[str, str]
    provenance: dict[str, str]
    matched_support_id: str | None


@dataclass(frozen=True, slots=True)
class TemporalV2ConsumerInput:
    """Final loader input for S1 or a resolution arm.

    This object deliberately carries emitted membership rather than a start,
    stride, or native-window parameter.  Consumers must pass it unchanged to
    their model-input assembly layer.
    """

    consumer: str
    input_resolution: int | None
    target: TemporalV2Target
    membership_source: str = "emitted:selected_frame_indices"
    historical_selectors_reachable: bool = False


def load_temporal_v2_target(
    authority_root: Path,
    *,
    corpus: str,
    view: str,
    target_id: str,
    verify_hashes: bool = True,
    frame_offset_index: Path | None = None,
) -> TemporalV2Target:
    """Load emitted membership and associated model authority for one target."""

    root = Path(authority_root).resolve()
    if corpus not in {MATCHED, FULL}:
        raise TemporalV2ConsumerError(f"unsupported corpus={corpus}")
    if view not in VIEWS:
        raise TemporalV2ConsumerError(f"unsupported view={view}")
    _verify_authority(root, verify_hashes=verify_hashes)
    if frame_offset_index is not None:
        _verify_frame_offset_index_authority(root, frame_offset_index)
    manifest = root / (
        "matched_temporal_window_manifest_release.csv"
        if corpus == MATCHED
        else "full_temporal_window_manifest_release.csv"
    )
    row = _one_row(manifest, "target_id", target_id)
    _require(row, "pool", corpus)
    _require(row, "view_id", view)
    frames = _frames(row["selected_frame_indices"], int(row["target_length"]))
    split = _one_row(root / "target_split_roles.csv", "target_id", target_id)
    frame_rows = list(
        _rows_for_target(
            root / "sequence_frame_features.csv",
            target_id,
            frame_offset_index=frame_offset_index,
        )
    )
    if len(frame_rows) != len(frames):
        raise TemporalV2ConsumerError("frame authority count does not match emitted membership")
    observed = tuple(_strict_bool(item.get("observed_mask")) for item in frame_rows)
    emitted = tuple(_strict_int(item.get("frame_index")) for item in frame_rows)
    if emitted != frames:
        raise TemporalV2ConsumerError("frame authority order differs from emitted membership")
    if not _strict_bool(frame_rows[0].get("window_first_step_reset")):
        raise TemporalV2ConsumerError("first frame does not reset window boundary state")
    _require_frame_agreement(frame_rows, row, split)
    features = _one_row(root / "window_temporal_feature_summary.csv", "target_id", target_id)
    return TemporalV2Target(
        target_id=target_id,
        corpus=corpus,
        view=view,
        frames=frames,
        source_type=row["source_type"],
        dataset_id=row["dataset_id"],
        video_key=row["video_key"],
        actor=row["object_track_key"],
        behavior=row["behavior"],
        split=split["split"],
        group=split["outer_fold_id"],
        observed_mask=observed,
        boundary_reset=True,
        temporal_features=features,
        provenance={
            "authority_root": str(root),
            "authority_sha256": _sha256(root / "temporal_semantics_authority_v2.json"),
            "artifact_manifest_sha256": _sha256(root / "temporal_v2_artifact_hash_manifest.json"),
        },
        matched_support_id=_optional(row, "matched_support_id"),
    )


def build_s1_temporal_v2_input(**kwargs: Any) -> TemporalV2ConsumerInput:
    """Build the only canonical S1 temporal input before model assembly."""

    return _consumer_input(consumer="S1", input_resolution=None, **kwargs)


def build_resolution_temporal_v2_input(
    *, input_resolution: int, **kwargs: Any
) -> TemporalV2ConsumerInput:
    """Build the only canonical R64/R128/R160 input before model assembly."""

    if input_resolution not in RESOLUTIONS:
        raise TemporalV2ConsumerError(f"unsupported resolution={input_resolution}")
    return _consumer_input(
        consumer="RESOLUTION",
        input_resolution=input_resolution,
        **kwargs,
    )


def load_s1_temporal_v2_target(**kwargs: Any) -> TemporalV2Target:
    """Compatibility target accessor backed by the final S1 consumer path."""

    return build_s1_temporal_v2_input(**kwargs).target


def load_resolution_temporal_v2_target(
    *, input_resolution: int, **kwargs: Any
) -> TemporalV2Target:
    """Compatibility target accessor backed by the final resolution path."""

    return build_resolution_temporal_v2_input(
        input_resolution=input_resolution,
        **kwargs,
    ).target


def audit_resolution_parity(targets: Iterable[TemporalV2Target]) -> dict[str, Any]:
    """Prove all resolution arms received one identical canonical target."""

    values = list(targets)
    if len(values) != 3:
        raise TemporalV2ConsumerError("parity audit requires R64, R128, and R160 targets")
    first = values[0]
    fields = (
        "target_id", "corpus", "view", "frames", "behavior", "split",
        "group", "observed_mask",
    )
    if any(
        any(getattr(item, field) != getattr(first, field) for field in fields)
        for item in values[1:]
    ):
        raise TemporalV2ConsumerError("resolution consumer temporal parity failure")
    return {"status": "PASS", "target_id": first.target_id, "frames": list(first.frames)}


def audit_release_counts(
    authority_root: Path,
    *,
    verify_hashes: bool = True,
) -> dict[str, dict[str, int]]:
    """Reconcile emitted manifest counts against the frozen publication receipt."""

    root = Path(authority_root).resolve()
    _verify_authority(root, verify_hashes=verify_hashes)
    receipt = _read_json(root / "temporal_v2_publication_receipt.json")
    expected = receipt.get("release_counts")
    if not isinstance(expected, dict):
        raise TemporalV2ConsumerError("publication receipt lacks frozen release counts")
    actual: dict[str, dict[str, int]] = {MATCHED: {}, FULL: {}}
    for corpus, filename in (
        (MATCHED, "matched_temporal_window_manifest_release.csv"),
        (FULL, "full_temporal_window_manifest_release.csv"),
    ):
        for row in _read_rows(root / filename):
            _require(row, "pool", corpus)
            view = row.get("view_id", "")
            if view not in VIEWS:
                raise TemporalV2ConsumerError(f"manifest has unsupported view={view}")
            actual[corpus][view] = actual[corpus].get(view, 0) + 1
    if actual != expected:
        raise TemporalV2ConsumerError("frozen release counts disagree with manifests")
    return actual


def audit_matched_support(targets: Iterable[TemporalV2Target]) -> dict[str, str]:
    """Prove one matched support spans exactly the four canonical views."""

    values = list(targets)
    supports = {value.matched_support_id for value in values}
    if (
        len(values) != 4
        or None in supports
        or len(supports) != 1
        or {value.view for value in values} != VIEWS
        or any(value.corpus != MATCHED for value in values)
    ):
        raise TemporalV2ConsumerError("matched support is not a four-view relation")
    return {"status": "PASS", "matched_support_id": next(iter(supports)) or ""}


def build_target_frame_offset_index(
    authority_root: Path,
    *,
    output_path: Path,
) -> dict[str, Any]:
    """Create a seek index over existing canonical frame rows without copying them."""

    root = Path(authority_root).resolve()
    _verify_authority(root, verify_hashes=False)
    source = root / "sequence_frame_features.csv"
    if Path(output_path).exists():
        raise TemporalV2ConsumerError(f"frame offset index already exists={output_path}")
    offsets: dict[str, dict[str, int]] = {}
    with source.open("rb") as handle:
        header = handle.readline()
        columns = next(csv.reader([header.decode("utf-8").rstrip("\r\n")]))
        try:
            target_column = columns.index("target_id")
        except ValueError as exc:
            raise TemporalV2ConsumerError("frame authority lacks target_id") from exc
        prior_target = None
        while True:
            offset = handle.tell()
            row = handle.readline()
            if not row:
                break
            values = next(csv.reader([row.decode("utf-8").rstrip("\r\n")]))
            if len(values) <= target_column:
                raise TemporalV2ConsumerError("frame authority row is truncated")
            target = values[target_column]
            if target != prior_target:
                if target in offsets:
                    raise TemporalV2ConsumerError("frame authority target rows are not contiguous")
                offsets[target] = {"offset": offset, "count": 0}
                prior_target = target
            offsets[target]["count"] += 1
    stat = source.stat()
    manifest = _read_json(root / "temporal_v2_artifact_hash_manifest.json")
    payload = {
        "schema_version": "classification_v2.temporal_v2_frame_offset_index.v1",
        "authority_sha256": _sha256(root / "temporal_semantics_authority_v2.json"),
        "artifact_manifest_sha256": _sha256(root / "temporal_v2_artifact_hash_manifest.json"),
        "sequence_frame_features": {
            "expected_sha256": manifest["artifacts"]["sequence_frame_features.csv"]["sha256"],
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        },
        "target_offsets": offsets,
    }
    _write_json_atomic(Path(output_path), payload)
    return {"status": "PASS", "target_count": len(offsets), "path": str(output_path)}


def verify_registered_canonical_authority(
    authority_root: Path,
    *,
    mapping_path: Path,
) -> dict[str, str]:
    """Accept the I-2 full-hash receipt for a hard-linked canonical release.

    This is the bounded runtime alternative to rehashing the 8-GB frame
    authority for every target.  It still binds the current authority and
    artifact-manifest bytes to the approved source release before loading.
    """

    root = Path(authority_root).resolve()
    mapping = _read_json(Path(mapping_path).resolve())
    if (
        mapping.get("schema_version")
        != "classification_v2.temporal_v2_canonical_authority_mapping.v1"
        or mapping.get("verification", {}).get("status") != "PASS"
    ):
        raise TemporalV2ConsumerError("canonical authority mapping is not verified")
    source = mapping.get("source", {})
    authority_hash = _sha256(root / "temporal_semantics_authority_v2.json")
    artifact_hash = _sha256(root / "temporal_v2_artifact_hash_manifest.json")
    if (
        source.get("authority_sha256") != authority_hash
        or source.get("artifact_manifest_sha256") != artifact_hash
    ):
        raise TemporalV2ConsumerError("registered canonical authority hash mismatch")
    _verify_authority(root, verify_hashes=False)
    return {
        "authority_sha256": authority_hash,
        "artifact_manifest_sha256": artifact_hash,
    }


def _verify_authority(root: Path, *, verify_hashes: bool) -> None:
    authority = _read_json(root / "temporal_semantics_authority_v2.json")
    receipt = _read_json(root / "temporal_v2_publication_receipt.json")
    if authority.get("schema_version") != "classification_v2.temporal_semantics_authority.v2":
        raise TemporalV2ConsumerError("Temporal-v2 authority version mismatch")
    if receipt.get("schema_version") != "classification_v2.temporal_v2_publication_receipt.v1":
        raise TemporalV2ConsumerError("Temporal-v2 publication version mismatch")
    if (
        authority.get("gate", {}).get("status") != "PASS"
        or receipt.get("status") != "PASS"
    ):
        raise TemporalV2ConsumerError("Temporal-v2 publication gate is not PASS")
    manifest = _read_json(root / "temporal_v2_artifact_hash_manifest.json")
    required = {
        "full_temporal_window_manifest_release.csv",
        "matched_temporal_window_manifest_release.csv",
        "sequence_frame_features.csv",
        "target_split_roles.csv",
        "window_temporal_feature_summary.csv",
    }
    missing = required.difference(manifest.get("artifacts", {}))
    if missing:
        raise TemporalV2ConsumerError(f"artifact hash manifest incomplete={sorted(missing)}")
    if verify_hashes:
        fingerprint = _artifact_fingerprint(root, manifest)
        if _HASH_VERIFICATION_CACHE.get(root) == fingerprint:
            return
        for name, record in manifest.get("artifacts", {}).items():
            path = root / name
            if not path.is_file() or _sha256(path) != record.get("sha256"):
                raise TemporalV2ConsumerError(f"artifact hash mismatch={name}")
        _HASH_VERIFICATION_CACHE[root] = fingerprint


def _one_row(path: Path, key: str, value: str) -> dict[str, str]:
    found = [row for row in _read_rows(path) if row.get(key) == value]
    if len(found) != 1:
        raise TemporalV2ConsumerError(f"authority target lookup is not unique={path.name}:{value}")
    return found[0]


def _rows_for_target(
    path: Path,
    target_id: str,
    *,
    frame_offset_index: Path | None,
) -> Iterable[dict[str, str]]:
    if frame_offset_index is not None:
        yield from _indexed_rows_for_target(path, target_id, frame_offset_index)
        return
    found = False
    for row in _read_rows(path):
        if row.get("target_id") == target_id:
            found = True
            yield row
        elif found:
            return


def _indexed_rows_for_target(
    path: Path,
    target_id: str,
    index_path: Path,
) -> Iterable[dict[str, str]]:
    index = _read_json(Path(index_path).resolve())
    if index.get("schema_version") != "classification_v2.temporal_v2_frame_offset_index.v1":
        raise TemporalV2ConsumerError("frame offset index version mismatch")
    source = index.get("sequence_frame_features", {})
    stat = path.stat()
    if source.get("size_bytes") != stat.st_size or source.get("mtime_ns") != stat.st_mtime_ns:
        raise TemporalV2ConsumerError("frame offset index source identity mismatch")
    record = index.get("target_offsets", {}).get(target_id)
    if not isinstance(record, dict):
        raise TemporalV2ConsumerError("frame offset index lacks target")
    with path.open("rb") as handle:
        header = handle.readline().decode("utf-8")
        handle.seek(int(record["offset"]))
        lines = [handle.readline().decode("utf-8") for _ in range(int(record["count"]))]
    rows = list(csv.DictReader(io.StringIO(header + "".join(lines))))
    if len(rows) != int(record["count"]) or any(
        row.get("target_id") != target_id for row in rows
    ):
        raise TemporalV2ConsumerError("frame offset index target disagreement")
    yield from rows


def _verify_frame_offset_index_authority(root: Path, index_path: Path) -> None:
    index = _read_json(Path(index_path).resolve())
    if (
        index.get("authority_sha256")
        != _sha256(root / "temporal_semantics_authority_v2.json")
        or index.get("artifact_manifest_sha256")
        != _sha256(root / "temporal_v2_artifact_hash_manifest.json")
    ):
        raise TemporalV2ConsumerError("frame offset index authority mismatch")


def _read_rows(path: Path) -> Iterable[dict[str, str]]:
    if not path.is_file():
        raise TemporalV2ConsumerError(f"authority artifact missing={path}")
    with path.open(encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise TemporalV2ConsumerError(f"authority artifact missing={path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _frames(raw: str, length: int) -> tuple[int, ...]:
    values = tuple(int(value) for value in json.loads(raw))
    if len(values) != length or len(set(values)) != length or values != tuple(sorted(values)):
        raise TemporalV2ConsumerError("invalid emitted physical frame membership")
    return values


def _require(row: dict[str, str], key: str, expected: str) -> None:
    if row.get(key) != expected:
        raise TemporalV2ConsumerError(f"authority {key} mismatch")


def _consumer_input(
    *,
    consumer: str,
    input_resolution: int | None,
    **kwargs: Any,
) -> TemporalV2ConsumerInput:
    return TemporalV2ConsumerInput(
        consumer=consumer,
        input_resolution=input_resolution,
        target=load_temporal_v2_target(**kwargs),
    )


def _require_frame_agreement(
    frame_rows: list[dict[str, str]],
    target: dict[str, str],
    split: dict[str, str],
) -> None:
    """Reject label, source, and split disagreement in frame-local authority."""

    expected = {
        "source_type": target.get("source_type"),
        "dataset_id": target.get("dataset_id"),
        "video_key": target.get("video_key"),
        "object_track_key": target.get("object_track_key"),
        "behavior": target.get("behavior"),
        "pool": target.get("pool"),
        "view_id": target.get("view_id"),
        "split": split.get("split"),
        "outer_fold_id": split.get("outer_fold_id"),
    }
    for row in frame_rows:
        for key, value in expected.items():
            if key in row and value is not None and row[key] != value:
                raise TemporalV2ConsumerError(f"frame authority {key} disagreement")
        if "target_id" in row and row["target_id"] != split.get("target_id"):
            raise TemporalV2ConsumerError("frame authority target disagreement")


def _optional(row: dict[str, str], key: str) -> str | None:
    value = row.get(key)
    return value if value else None


def _artifact_fingerprint(
    root: Path,
    manifest: dict[str, Any],
) -> tuple[tuple[str, int, int], ...]:
    """Cache a completed full hash audit only while every artifact is unchanged."""

    values: list[tuple[str, int, int]] = []
    for name in sorted(manifest.get("artifacts", {})):
        path = root / name
        if not path.is_file():
            raise TemporalV2ConsumerError(f"authority artifact missing={path}")
        stat = path.stat()
        values.append((name, stat.st_size, stat.st_mtime_ns))
    return tuple(values)


def _strict_bool(value: object) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise TemporalV2ConsumerError("invalid authority boolean")


def _strict_int(value: object) -> int:
    try:
        return int(str(value))
    except ValueError as exc:
        raise TemporalV2ConsumerError("invalid authority physical frame") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as handle:
        json.dump(payload, handle, sort_keys=True)
        temporary = Path(handle.name)
    os.replace(temporary, path)


__all__ = [
    "FULL", "MATCHED", "RESOLUTIONS", "VIEWS", "TemporalV2ConsumerError",
    "TemporalV2ConsumerInput", "TemporalV2Target", "audit_matched_support",
    "audit_release_counts", "audit_resolution_parity",
    "build_target_frame_offset_index",
    "build_resolution_temporal_v2_input", "build_s1_temporal_v2_input",
    "load_resolution_temporal_v2_target", "load_s1_temporal_v2_target",
    "load_temporal_v2_target", "verify_registered_canonical_authority",
]
