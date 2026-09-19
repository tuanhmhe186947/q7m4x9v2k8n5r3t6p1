from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Build resumable per-file SHA256 integrity for an image cache.")
    parser.add_argument(
        "--cache-manifest",
        type=Path,
        default=Path("outputs/classification_v2/image_cache_v2_letterbox/manifest.csv"),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    audit = build_integrity_manifest(
        cache_manifest=args.cache_manifest,
        workers=args.workers,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    print(json.dumps(audit, indent=2))
    if not audit["valid"]:
        raise SystemExit(2)


def build_integrity_manifest(
    *,
    cache_manifest: Path,
    workers: int,
    checkpoint_every: int,
    resume: bool,
    overwrite: bool,
) -> dict[str, Any]:
    """Hash cache files once, using bounded threads and append-only checkpoints."""

    if workers <= 0:
        raise ValueError("workers must be positive")
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")
    source = pd.read_csv(cache_manifest, usecols=["image_context_id", "cache_path"], low_memory=False)
    if source["image_context_id"].duplicated().any():
        raise ValueError("cache manifest contains duplicate image_context_id rows")
    output_dir = cache_manifest.parent
    partial_path = output_dir / "image_cache_integrity_manifest.partial.csv"
    output_path = output_dir / "image_cache_integrity_manifest.csv"
    partial_audit_path = output_dir / "image_cache_integrity_audit.partial.json"
    audit_path = output_dir / "image_cache_integrity_audit.json"
    if overwrite:
        for path in [partial_path, output_path, partial_audit_path, audit_path]:
            path.unlink(missing_ok=True)
    completed_ids: set[str] = set()
    if resume and partial_path.exists():
        partial = pd.read_csv(partial_path, usecols=["image_context_id"], low_memory=False)
        if partial["image_context_id"].duplicated().any():
            raise ValueError("partial integrity manifest contains duplicate image_context_id rows")
        completed_ids = set(partial["image_context_id"].astype(str))
    elif partial_path.exists():
        raise FileExistsError(f"partial integrity manifest exists; use --resume or --overwrite: {partial_path}")

    pending = source[~source["image_context_id"].astype(str).isin(completed_ids)].copy()
    records = [(str(row.image_context_id), str(row.cache_path)) for row in pending.itertuples(index=False)]
    written_this_run = 0
    buffer: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="cache-sha256") as executor:
        for result in executor.map(lambda item: _hash_cache_entry(output_dir, *item), records, chunksize=64):
            buffer.append(result)
            if len(buffer) >= checkpoint_every:
                _append_checkpoint(partial_path, buffer)
                written_this_run += len(buffer)
                buffer.clear()
                _write_partial_audit(
                    partial_audit_path,
                    source_rows=len(source),
                    resumed_rows=len(completed_ids),
                    written_this_run=written_this_run,
                    workers=workers,
                )
    if buffer:
        _append_checkpoint(partial_path, buffer)
        written_this_run += len(buffer)

    integrity = pd.read_csv(partial_path, low_memory=False).sort_values("image_context_id", kind="mergesort")
    duplicate_rows = int(integrity["image_context_id"].duplicated().sum())
    source_ids = set(source["image_context_id"].astype(str))
    integrity_ids = set(integrity["image_context_id"].astype(str))
    missing_context_rows = int(len(source_ids.difference(integrity_ids)))
    unexpected_context_rows = int(len(integrity_ids.difference(source_ids)))
    failed_rows = int(integrity["integrity_status"].ne("ok").sum())
    integrity.to_csv(output_path, index=False)
    audit = {
        "schema_version": "classification_v2_image_cache_integrity_audit_v1",
        "cache_manifest": str(cache_manifest),
        "integrity_manifest": str(output_path),
        "source_rows": int(len(source)),
        "integrity_rows": int(len(integrity)),
        "resumed_rows": int(len(completed_ids)),
        "written_this_run": int(written_this_run),
        "workers": int(workers),
        "duplicate_rows": duplicate_rows,
        "missing_context_rows": missing_context_rows,
        "unexpected_context_rows": unexpected_context_rows,
        "failed_rows": failed_rows,
        "total_cache_bytes": int(integrity.loc[integrity["integrity_status"].eq("ok"), "cache_size_bytes"].sum()),
        "valid": bool(
            len(integrity) == len(source)
            and duplicate_rows == 0
            and missing_context_rows == 0
            and unexpected_context_rows == 0
            and failed_rows == 0
        ),
    }
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def _hash_cache_entry(base: Path, context_id: str, cache_value: str) -> dict[str, Any]:
    """Return one deterministic checksum record without loading NumPy arrays."""

    path = Path(cache_value)
    if not path.is_absolute():
        path = base / path
    if not path.exists():
        return {
            "image_context_id": context_id,
            "cache_path": cache_value,
            "cache_size_bytes": 0,
            "cache_sha256": "",
            "integrity_status": "missing",
        }
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "image_context_id": context_id,
        "cache_path": cache_value,
        "cache_size_bytes": int(path.stat().st_size),
        "cache_sha256": digest.hexdigest(),
        "integrity_status": "ok",
    }


def _append_checkpoint(path: Path, rows: list[dict[str, Any]]) -> None:
    """Append a bounded completed batch so interruption loses little work."""

    pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def _write_partial_audit(
    path: Path,
    *,
    source_rows: int,
    resumed_rows: int,
    written_this_run: int,
    workers: int,
) -> None:
    audit = {
        "schema_version": "classification_v2_image_cache_integrity_partial_audit_v1",
        "source_rows": int(source_rows),
        "resumed_rows": int(resumed_rows),
        "written_this_run": int(written_this_run),
        "completed_rows": int(resumed_rows + written_this_run),
        "workers": int(workers),
        "complete": False,
    }
    path.write_text(json.dumps(audit, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
