"""Crash-safe, non-authoritative checkpoints for Pig-STRENet artifact builds."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

CHECKPOINT_SCHEMA_VERSION = "classification_v2.pig_strenet_checkpoint.v1"


class PigSTRENetCheckpointError(RuntimeError):
    """Raised when a checkpoint is absent, corrupt, or authority-incompatible."""


class PigSTRENetCheckpointStore:
    """Persist exact phase tables and bounded social-graph chunks atomically."""

    def __init__(
        self,
        root: Path,
        *,
        identity: Mapping[str, Any],
        resume: bool,
        social_chunk_pairs: int = 250,
    ) -> None:
        if social_chunk_pairs <= 0:
            raise ValueError("social_chunk_pairs must be positive")
        self.root = root
        self.identity = _jsonable(dict(identity))
        self.identity_hash = _json_hash(self.identity)
        self.social_chunk_pairs = int(social_chunk_pairs)
        self._identity_path = self.root / "checkpoint_identity.json"
        if resume:
            self._validate_identity()
        else:
            if self.root.exists() and any(self.root.iterdir()):
                raise PigSTRENetCheckpointError(
                    f"checkpoint root is not empty: {self.root}"
                )
            self.root.mkdir(parents=True, exist_ok=True)
            self._write_json(
                self._identity_path,
                {
                    "schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "identity": self.identity,
                    "identity_hash": self.identity_hash,
                },
            )

    def load_phase(
        self,
        phase: str,
        table_names: tuple[str, ...],
    ) -> dict[str, pd.DataFrame] | None:
        """Load a complete phase or return ``None`` when it was not checkpointed."""

        manifest_path = self.root / f"{phase}.checkpoint.json"
        if not manifest_path.is_file():
            return None
        manifest = self._read_json(manifest_path)
        self._validate_manifest_identity(manifest, manifest_path)
        tables = manifest.get("tables")
        if not isinstance(tables, dict) or set(tables) != set(table_names):
            raise PigSTRENetCheckpointError(
                f"checkpoint table set mismatch: {manifest_path}"
            )
        loaded: dict[str, pd.DataFrame] = {}
        for name in table_names:
            record = tables[name]
            path = self.root / str(record["file"])
            self._validate_file(path, str(record["sha256"]))
            frame = pd.read_pickle(path, compression="gzip")
            if len(frame) != int(record["rows"]):
                raise PigSTRENetCheckpointError(
                    f"checkpoint row count mismatch: {path}"
                )
            loaded[name] = frame
        return loaded

    def save_phase(
        self,
        phase: str,
        tables: Mapping[str, pd.DataFrame],
    ) -> None:
        """Atomically publish all tables for one completed phase."""

        records: dict[str, dict[str, Any]] = {}
        for name, frame in tables.items():
            filename = f"{phase}.{name}.pkl.gz"
            path = self.root / filename
            self._write_pickle(path, frame)
            records[name] = {
                "file": filename,
                "rows": int(len(frame)),
                "columns": list(frame.columns),
                "sha256": _sha256(path),
            }
        self._write_json(
            self.root / f"{phase}.checkpoint.json",
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "identity_hash": self.identity_hash,
                "phase": phase,
                "tables": records,
            },
        )

    def social_resume_index(self, total_pairs: int) -> int:
        """Return the exact contiguous social-graph pair prefix on disk."""

        manifest = self._social_manifest(total_pairs)
        return int(manifest["completed_pairs"])

    def save_social_chunk(
        self,
        *,
        start_pair: int,
        end_pair: int,
        total_pairs: int,
        nodes: pd.DataFrame,
        edges: pd.DataFrame,
    ) -> None:
        """Append one contiguous, immutable social-graph checkpoint chunk."""

        manifest = self._social_manifest(total_pairs)
        completed = int(manifest["completed_pairs"])
        if start_pair != completed or end_pair <= start_pair:
            raise PigSTRENetCheckpointError(
                "social checkpoint chunks must be positive and contiguous"
            )
        stem = f"social_graph.{start_pair:08d}_{end_pair:08d}"
        node_path = self.root / f"{stem}.nodes.pkl.gz"
        edge_path = self.root / f"{stem}.edges.pkl.gz"
        self._write_pickle(node_path, nodes)
        self._write_pickle(edge_path, edges)
        manifest["chunks"].append(
            {
                "start_pair": start_pair,
                "end_pair": end_pair,
                "nodes_file": node_path.name,
                "nodes_rows": int(len(nodes)),
                "nodes_sha256": _sha256(node_path),
                "edges_file": edge_path.name,
                "edges_rows": int(len(edges)),
                "edges_sha256": _sha256(edge_path),
            }
        )
        manifest["completed_pairs"] = end_pair
        self._write_json(self.root / "social_graph.chunks.json", manifest)

    def load_social_chunks(
        self,
        total_pairs: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load and concatenate a complete ordered social checkpoint."""

        manifest = self._social_manifest(total_pairs)
        if int(manifest["completed_pairs"]) != total_pairs:
            raise PigSTRENetCheckpointError(
                "social checkpoint is not complete"
            )
        node_chunks: list[pd.DataFrame] = []
        edge_chunks: list[pd.DataFrame] = []
        expected_start = 0
        for record in manifest["chunks"]:
            if int(record["start_pair"]) != expected_start:
                raise PigSTRENetCheckpointError(
                    "social checkpoint chunk sequence is not contiguous"
                )
            expected_start = int(record["end_pair"])
            node_chunks.append(
                self._load_chunk_table(record, prefix="nodes")
            )
            edge_chunks.append(
                self._load_chunk_table(record, prefix="edges")
            )
        return (
            pd.concat(node_chunks, ignore_index=True),
            pd.concat(edge_chunks, ignore_index=True),
        )

    def cleanup(self) -> None:
        """Remove only this validated temporary checkpoint namespace."""

        self._validate_identity()
        paths = list(self.root.iterdir())
        if any(not path.is_file() for path in paths):
            raise PigSTRENetCheckpointError(
                f"checkpoint root contains an unexpected directory: {self.root}"
            )
        for path in paths:
            path.unlink()
        self.root.rmdir()

    def _load_chunk_table(
        self,
        record: Mapping[str, Any],
        *,
        prefix: str,
    ) -> pd.DataFrame:
        path = self.root / str(record[f"{prefix}_file"])
        self._validate_file(path, str(record[f"{prefix}_sha256"]))
        frame = pd.read_pickle(path, compression="gzip")
        if len(frame) != int(record[f"{prefix}_rows"]):
            raise PigSTRENetCheckpointError(
                f"social checkpoint row count mismatch: {path}"
            )
        return frame

    def _social_manifest(self, total_pairs: int) -> dict[str, Any]:
        path = self.root / "social_graph.chunks.json"
        if not path.is_file():
            return {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "identity_hash": self.identity_hash,
                "phase": "social_graph_chunks",
                "total_pairs": int(total_pairs),
                "completed_pairs": 0,
                "chunks": [],
            }
        manifest = self._read_json(path)
        self._validate_manifest_identity(manifest, path)
        if int(manifest.get("total_pairs", -1)) != int(total_pairs):
            raise PigSTRENetCheckpointError(
                f"social checkpoint pair total mismatch: {path}"
            )
        return manifest

    def _validate_identity(self) -> None:
        if not self._identity_path.is_file():
            raise PigSTRENetCheckpointError(
                f"checkpoint identity is missing: {self._identity_path}"
            )
        payload = self._read_json(self._identity_path)
        if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise PigSTRENetCheckpointError("checkpoint schema mismatch")
        if payload.get("identity_hash") != self.identity_hash:
            raise PigSTRENetCheckpointError("checkpoint identity mismatch")
        if payload.get("identity") != self.identity:
            raise PigSTRENetCheckpointError("checkpoint authority mismatch")

    def _validate_manifest_identity(
        self,
        manifest: Mapping[str, Any],
        path: Path,
    ) -> None:
        if manifest.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise PigSTRENetCheckpointError(
                f"checkpoint schema mismatch: {path}"
            )
        if manifest.get("identity_hash") != self.identity_hash:
            raise PigSTRENetCheckpointError(
                f"checkpoint identity mismatch: {path}"
            )

    @staticmethod
    def _validate_file(path: Path, expected_sha256: str) -> None:
        if not path.is_file() or _sha256(path) != expected_sha256:
            raise PigSTRENetCheckpointError(
                f"checkpoint file hash mismatch: {path}"
            )

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise PigSTRENetCheckpointError(
                f"checkpoint JSON is invalid: {path}"
            ) from error
        if not isinstance(payload, dict):
            raise PigSTRENetCheckpointError(
                f"checkpoint JSON must be an object: {path}"
            )
        return payload

    @staticmethod
    def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
        temporary = path.with_name(f"{path.name}.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    @staticmethod
    def _write_pickle(path: Path, frame: pd.DataFrame) -> None:
        temporary = path.with_name(f"{path.name}.tmp")
        frame.to_pickle(
            temporary,
            compression={"method": "gzip", "compresslevel": 1},
        )
        temporary.replace(path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _json_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
