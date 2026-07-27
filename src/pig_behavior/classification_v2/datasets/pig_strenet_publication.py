"""Crash-safe publication and recovery for Pig-STRENet media evidence."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

MEDIA_MANIFEST_SCHEMA = "classification_v2.pig_strenet_media_manifest.v1"
PUBLICATION_CHECKPOINT_SCHEMA = (
    "classification_v2.pig_strenet_media_publication_checkpoint.v1"
)
ProgressCallback = Callable[[str, int | None, int | None], None]


class MediaPublicationError(RuntimeError):
    """Raised when durable media publication cannot be proven."""


def publish_media_manifest(
    path: Path,
    *,
    video_root: Path,
    legacy_crop_root: Path,
    video_index_aliases: int,
    usage: Mapping[str, Mapping[str, Any]],
    status_counts: Mapping[str, int],
    runtime_counts: Mapping[str, int],
    rejected_scene_candidates: Sequence[str],
    checkpoint_path: Path,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Publish exact resolver usage with resumable per-file hashing."""

    store = _MediaPublicationStore(checkpoint_path)
    try:
        store.replace_usage(usage)
        return store.publish(
            path,
            video_root=video_root,
            legacy_crop_root=legacy_crop_root,
            video_index_aliases=video_index_aliases,
            status_counts=status_counts,
            runtime_counts=runtime_counts,
            rejected_scene_candidates=rejected_scene_candidates,
            usage_count_semantics="resolver_mark_used_calls",
            progress_callback=progress_callback,
        )
    finally:
        store.close()


def recover_media_manifest(
    path: Path,
    *,
    video_root: Path,
    legacy_crop_root: Path,
    video_index_aliases: int,
    provenance_paths: Sequence[Path],
    checkpoint_path: Path,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Recover media authority only from completed production provenance."""

    store = _MediaPublicationStore(checkpoint_path)
    try:
        store.replace_usage_from_provenance(
            provenance_paths,
            progress_callback=progress_callback,
        )
        payload = store.publish(
            path,
            video_root=video_root,
            legacy_crop_root=legacy_crop_root,
            video_index_aliases=video_index_aliases,
            status_counts={
                "recovered_provenance_rows": store.provenance_rows()
            },
            runtime_counts={"publication_recovery": 1},
            rejected_scene_candidates=[],
            usage_count_semantics="durable_provenance_rows",
            progress_callback=progress_callback,
        )
        payload["provenance_audit"] = store.provenance_audit()
        _atomic_write_json(path, payload)
        return payload
    finally:
        store.close()


def checkpointed_sha256(path: Path, *, checkpoint_path: Path) -> str:
    """Hash one immutable file and reuse the digest while stat identity matches."""

    store = _MediaPublicationStore(checkpoint_path)
    try:
        stat = path.stat()
        return store.cached_hash(path, stat.st_size, stat.st_mtime_ns)
    finally:
        store.close()


class _MediaPublicationStore:
    """SQLite-backed usage and hash checkpoint for publication."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self._create_schema()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS usage (
                path TEXT NOT NULL,
                source_kind TEXT NOT NULL,
                use_count INTEGER NOT NULL,
                PRIMARY KEY (path, source_kind)
            );
            CREATE TABLE IF NOT EXISTS frames (
                path TEXT NOT NULL,
                frame_index INTEGER NOT NULL,
                PRIMARY KEY (path, frame_index)
            );
            CREATE TABLE IF NOT EXISTS hashes (
                path TEXT PRIMARY KEY,
                size INTEGER NOT NULL,
                mtime_ns INTEGER NOT NULL,
                sha256 TEXT NOT NULL
            );
            """
        )
        self._set_metadata("schema_version", PUBLICATION_CHECKPOINT_SCHEMA)
        self.connection.commit()

    def close(self) -> None:
        """Commit and close the durable checkpoint."""

        self.connection.commit()
        self.connection.close()

    def replace_usage(
        self,
        usage: Mapping[str, Mapping[str, Any]],
    ) -> None:
        self._clear_usage()
        with self.connection:
            for path_text, record in sorted(usage.items()):
                counts = record.get("source_kind_counts", {})
                for source_kind, count in sorted(dict(counts).items()):
                    self._upsert_usage(path_text, str(source_kind), int(count))
                for frame_index in sorted(record.get("frame_indices", ())):
                    self.connection.execute(
                        "INSERT OR IGNORE INTO frames VALUES (?, ?)",
                        (path_text, int(frame_index)),
                    )
            self._set_metadata("usage_complete", "resolver")

    def replace_usage_from_provenance(
        self,
        paths: Sequence[Path],
        *,
        progress_callback: ProgressCallback | None,
    ) -> None:
        signatures = [_file_signature(path) for path in paths]
        expected = json.dumps(signatures, sort_keys=True)
        if self._metadata("provenance_signature") == expected:
            if self._metadata("usage_complete") == "provenance":
                return
        self._clear_usage()
        total_bytes = sum(int(item["size"]) for item in signatures)
        completed_bytes = 0
        total_rows = 0
        audit: list[dict[str, Any]] = []
        for source_path in paths:
            if not source_path.is_file():
                raise MediaPublicationError(
                    f"MEDIA_PROVENANCE_MISSING:{source_path}"
                )
            columns = list(pd.read_csv(source_path, nrows=0).columns)
            required_column = next(
                (
                    name
                    for name in (
                        "frame_available",
                        "pixel_geometry_expected",
                    )
                    if name in columns
                ),
                None,
            )
            usecols = [
                "pixel_available",
                "pixel_source_kind",
                "pixel_media_path",
                "pixel_frame_index",
            ]
            if required_column is not None:
                usecols.append(required_column)
            file_rows = 0
            required_rows = 0
            available_rows = 0
            missing_required_rows = 0
            for chunk in pd.read_csv(
                source_path,
                usecols=usecols,
                dtype="string",
                chunksize=100_000,
                low_memory=False,
            ):
                total_rows += int(len(chunk))
                file_rows += int(len(chunk))
                available = _bool_series(chunk["pixel_available"])
                available_rows += int(available.sum())
                if required_column is not None:
                    required = _bool_series(chunk[required_column])
                    required_rows += int(required.sum())
                    missing_required_rows += int((required & ~available).sum())
                self._ingest_provenance_chunk(chunk)
                if progress_callback is not None:
                    progress_callback(
                        "recover_media_provenance",
                        completed_bytes,
                        total_bytes,
                    )
            completed_bytes += int(source_path.stat().st_size)
            audit.append(
                {
                    "path": str(source_path),
                    "rows": file_rows,
                    "required_rows": required_rows,
                    "available_rows": available_rows,
                    "missing_required_rows": missing_required_rows,
                }
            )
            if progress_callback is not None:
                progress_callback(
                    "recover_media_provenance",
                    completed_bytes,
                    total_bytes,
                )
        with self.connection:
            self._set_metadata("provenance_signature", expected)
            self._set_metadata("provenance_rows", str(total_rows))
            self._set_metadata(
                "provenance_audit",
                json.dumps(audit, sort_keys=True),
            )
            self._set_metadata("usage_complete", "provenance")

    def _ingest_provenance_chunk(self, chunk: pd.DataFrame) -> None:
        work = chunk.copy()
        work["pixel_media_path"] = (
            work["pixel_media_path"].fillna("").astype(str).str.strip()
        )
        work = work[work["pixel_media_path"].ne("")]
        if work.empty:
            return
        work["pixel_source_kind"] = (
            work["pixel_source_kind"].fillna("unknown").astype(str)
        )
        counts = (
            work.groupby(
                ["pixel_media_path", "pixel_source_kind"],
                dropna=False,
                sort=False,
            )
            .size()
            .reset_index(name="use_count")
        )
        frames = work[["pixel_media_path", "pixel_frame_index"]].copy()
        frames["pixel_frame_index"] = pd.to_numeric(
            frames["pixel_frame_index"],
            errors="coerce",
        )
        frames = frames.dropna().drop_duplicates()
        with self.connection:
            for row in counts.itertuples(index=False):
                self._upsert_usage(
                    str(row.pixel_media_path),
                    str(row.pixel_source_kind),
                    int(row.use_count),
                )
            self.connection.executemany(
                "INSERT OR IGNORE INTO frames VALUES (?, ?)",
                (
                    (str(row.pixel_media_path), int(row.pixel_frame_index))
                    for row in frames.itertuples(index=False)
                ),
            )

    def publish(
        self,
        path: Path,
        *,
        video_root: Path,
        legacy_crop_root: Path,
        video_index_aliases: int,
        status_counts: Mapping[str, int],
        runtime_counts: Mapping[str, int],
        rejected_scene_candidates: Sequence[str],
        usage_count_semantics: str,
        progress_callback: ProgressCallback | None,
    ) -> dict[str, Any]:
        rows = self.connection.execute(
            "SELECT DISTINCT path FROM usage ORDER BY path"
        ).fetchall()
        total = len(rows)
        sources: list[dict[str, Any]] = []
        for index, (path_text,) in enumerate(rows, start=1):
            source = self._source_record(str(path_text))
            sources.append(source)
            if progress_callback is not None and (
                index == total or index % 100 == 0
            ):
                progress_callback("hash_media_manifest", index, total)
        payload = {
            "schema_version": MEDIA_MANIFEST_SCHEMA,
            "publication_checkpoint_schema": PUBLICATION_CHECKPOINT_SCHEMA,
            "video_root": str(video_root),
            "legacy_crop_root": str(legacy_crop_root),
            "video_index_aliases": int(video_index_aliases),
            "source_file_count": len(sources),
            "sources": sources,
            "status_counts": dict(sorted(status_counts.items())),
            "runtime_counts": dict(sorted(runtime_counts.items())),
            "usage_count_semantics": usage_count_semantics,
            "rejected_static_scene_candidates": sorted(
                str(value) for value in rejected_scene_candidates
            ),
            "background_as_temporal_scene_used": False,
            "valid": all(
                source["exists"] and source["sha256"] for source in sources
            ),
        }
        _atomic_write_json(path, payload)
        return payload

    def _source_record(self, path_text: str) -> dict[str, Any]:
        path = Path(path_text)
        exists = path.is_file()
        if not exists:
            return {
                "path": path_text,
                "exists": False,
                "size": None,
                "sha256": None,
                "source_kind_counts": self._source_kind_counts(path_text),
                "frame_index_count": 0,
                "frame_index_min": None,
                "frame_index_max": None,
                "frame_indices_sha256": _ordered_values_sha256([]),
            }
        stat = path.stat()
        digest = self._cached_hash(path, stat.st_size, stat.st_mtime_ns)
        frames = [
            int(row[0])
            for row in self.connection.execute(
                "SELECT frame_index FROM frames WHERE path = ? "
                "ORDER BY frame_index",
                (path_text,),
            )
        ]
        return {
            "path": path_text,
            "exists": True,
            "size": int(stat.st_size),
            "sha256": digest,
            "source_kind_counts": self._source_kind_counts(path_text),
            "frame_index_count": len(frames),
            "frame_index_min": frames[0] if frames else None,
            "frame_index_max": frames[-1] if frames else None,
            "frame_indices_sha256": _ordered_values_sha256(frames),
        }

    def _cached_hash(self, path: Path, size: int, mtime_ns: int) -> str:
        cached = self.connection.execute(
            "SELECT size, mtime_ns, sha256 FROM hashes WHERE path = ?",
            (str(path),),
        ).fetchone()
        if cached is not None:
            if int(cached[0]) == int(size) and int(cached[1]) == int(mtime_ns):
                return str(cached[2])
        digest = _sha256_file_with_retry(path)
        with self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO hashes VALUES (?, ?, ?, ?)",
                (str(path), int(size), int(mtime_ns), digest),
            )
        return digest

    def cached_hash(self, path: Path, size: int, mtime_ns: int) -> str:
        """Return the durable digest for one exact file identity."""

        return self._cached_hash(path, size, mtime_ns)

    def _source_kind_counts(self, path_text: str) -> dict[str, int]:
        rows = self.connection.execute(
            "SELECT source_kind, use_count FROM usage WHERE path = ? "
            "ORDER BY source_kind",
            (path_text,),
        ).fetchall()
        return {str(kind): int(count) for kind, count in rows}

    def provenance_rows(self) -> int:
        return int(self._metadata("provenance_rows") or 0)

    def provenance_audit(self) -> list[dict[str, Any]]:
        payload = self._metadata("provenance_audit") or "[]"
        value = json.loads(payload)
        return list(value) if isinstance(value, list) else []

    def _clear_usage(self) -> None:
        with self.connection:
            self.connection.execute("DELETE FROM usage")
            self.connection.execute("DELETE FROM frames")
            self._set_metadata("usage_complete", "")
            self._set_metadata("provenance_signature", "")
            self._set_metadata("provenance_rows", "0")
            self._set_metadata("provenance_audit", "[]")

    def _upsert_usage(self, path: str, kind: str, count: int) -> None:
        self.connection.execute(
            """
            INSERT INTO usage(path, source_kind, use_count)
            VALUES (?, ?, ?)
            ON CONFLICT(path, source_kind)
            DO UPDATE SET use_count = use_count + excluded.use_count
            """,
            (path, kind, int(count)),
        )

    def _metadata(self, key: str) -> str | None:
        row = self.connection.execute(
            "SELECT value FROM metadata WHERE key = ?",
            (key,),
        ).fetchone()
        return None if row is None else str(row[0])

    def _set_metadata(self, key: str, value: str) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO metadata VALUES (?, ?)",
            (key, value),
        )


def _sha256_file_with_retry(path: Path, attempts: int = 3) -> str:
    last_error: OSError | None = None
    for attempt in range(attempts):
        try:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()
        except OSError as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(0.25 * (2**attempt))
    raise MediaPublicationError(
        f"MEDIA_HASH_READ_FAILED:{path}:{last_error}"
    ) from last_error


def _ordered_values_sha256(values: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(int(value)).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _file_signature(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise MediaPublicationError(f"MEDIA_PROVENANCE_MISSING:{path}")
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _bool_series(values: pd.Series) -> pd.Series:
    return (
        values.fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
        .isin({"1", "true", "yes", "y"})
    )


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


__all__ = [
    "MEDIA_MANIFEST_SCHEMA",
    "MediaPublicationError",
    "checkpointed_sha256",
    "publish_media_manifest",
    "recover_media_manifest",
]
