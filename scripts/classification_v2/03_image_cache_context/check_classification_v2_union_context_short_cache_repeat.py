from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

SCHEMA_VERSION = "classification_v2.legacy_development_l6.union_context_short_cache_config.v1"
RESULT_SCHEMA = "classification_v2.legacy_development_l6.union_context_short_cache_repeat_gate.v1"
LINEAGE_SCOPE = "legacy-only-unreviewed-development"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check byte-identical primary/repeat union-context caches."
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_union_context_short_cache_repeat(args.config)
    print(json.dumps(result, indent=2))
    if not result["valid"]:
        raise SystemExit(2)


def evaluate_union_context_short_cache_repeat(
    config_path: Path,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    payload, repo_root = _load_config(config_path)
    _validate_bound_implementation(payload, repo_root)
    guard = _git_guard(payload, repo_root)
    packets: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for replica in ("primary", "repeat"):
        root = _bound_path(repo_root, payload["outputs"][f"{replica}_root_relative_path"])
        packet = _read_packet(root, payload)
        packets[replica] = packet
        errors.extend(f"{replica}:{error}" for error in packet["errors"])
    semantic = _semantic_comparison(packets)
    artifacts = _artifact_comparison(packets, payload)
    errors.extend(semantic["errors"])
    errors.extend(artifacts["errors"])
    errors.extend(guard["errors"])
    gate_path = _bound_path(repo_root, payload["outputs"]["repeat_gate_relative_path"])
    if gate_path.exists():
        errors.append(f"repeat gate already exists: {gate_path}")
    valid = not errors
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": (
            "PASS_LEGACY_DEVELOPMENT_L6_UNION_CONTEXT_SHORT_CACHE_REPEAT"
            if valid
            else "FAIL_LEGACY_DEVELOPMENT_L6_UNION_CONTEXT_SHORT_CACHE_REPEAT"
        ),
        "lineage_scope": LINEAGE_SCOPE,
        "canonical_source_name": str(payload["canonical_source_name"]),
        "source_type": str(payload["source_type"]),
        "dataset_id": str(payload["dataset_id"]),
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "config_path": str(config_path),
        "config_sha256": _file_sha256(config_path),
        "primary": _packet_summary(packets.get("primary", {})),
        "repeat": _packet_summary(packets.get("repeat", {})),
        "semantic_comparison": semantic,
        "artifact_comparison": artifacts,
        "separate_output_roots": (
            packets.get("primary", {}).get("root")
            != packets.get("repeat", {}).get("root")
        ),
        "source_media_reads": 0,
        "outer_holdout_rows": 0,
        "git_guard": guard,
        "errors": errors,
        "valid": valid,
    }
    if valid:
        gate_path.parent.mkdir(parents=True, exist_ok=True)
        gate_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _packet_summary(packet: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in packet.items()
        if key not in {"manifest", "build_audit", "run_manifest"}
    }


def _read_packet(root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    manifest_path = root / "visual_context_manifest.csv"
    audit_path = root / "visual_context_cache_audit.json"
    run_path = root / "run_manifest.json"
    if not root.is_dir():
        return {"root": str(root), "errors": [f"missing_root={root}"]}
    if not manifest_path.is_file():
        errors.append(f"missing_manifest={manifest_path}")
    if not audit_path.is_file():
        errors.append(f"missing_build_audit={audit_path}")
    if not run_path.is_file():
        errors.append(f"missing_run_manifest={run_path}")
    if errors:
        return {"root": str(root), "errors": errors}
    manifest = pd.read_csv(manifest_path, low_memory=False)
    build_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    run_manifest = json.loads(run_path.read_text(encoding="utf-8"))
    required = {
        "visual_context_id",
        "image_context_id",
        "visual_context_available",
        "cache_path",
        "image_size",
        "resize_policy",
        "lineage_scope",
        "human_review_complete",
    }
    missing = sorted(required.difference(manifest.columns))
    if missing:
        errors.append(f"manifest_missing_columns={missing}")
    available = _to_bool(manifest.get("visual_context_available", pd.Series()))
    if manifest.get("visual_context_id", pd.Series()).astype(str).duplicated().any():
        errors.append("manifest_duplicate_visual_context_id")
    expected_rows = int(payload["cache_contract"]["selected_rows"])
    expected_ready = int(payload["cache_contract"]["expected_ready_rows"])
    if len(manifest) != expected_rows:
        errors.append(f"manifest_rows={len(manifest)}!={expected_rows}")
    if int(available.sum()) != expected_ready:
        errors.append(f"available_rows={int(available.sum())}!={expected_ready}")
    if build_audit.get("valid") is not True:
        errors.append("build_audit_invalid")
    if run_manifest.get("valid") is not True:
        errors.append("run_manifest_invalid")
    if run_manifest.get("selection_csv_sha256") != _selection_hash(payload):
        errors.append("run_selection_hash_mismatch")
    if build_audit.get("selection_sha256") != _selection_hash(payload):
        errors.append("build_selection_hash_mismatch")
    if build_audit.get("lineage_scope") != LINEAGE_SCOPE:
        errors.append("build_lineage_scope_mismatch")
    if build_audit.get("human_review_complete") is not False:
        errors.append("build_review_claim_mismatch")
    if set(manifest["lineage_scope"].astype(str)) != {LINEAGE_SCOPE}:
        errors.append("manifest_lineage_scope_mismatch")
    if set(_to_bool(manifest["human_review_complete"])) != {False}:
        errors.append("manifest_review_claim_mismatch")
    return {
        "root": str(root),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _file_sha256(manifest_path),
        "build_audit_path": str(audit_path),
        "build_audit_sha256": _file_sha256(audit_path),
        "run_manifest_path": str(run_path),
        "run_manifest_sha256": _file_sha256(run_path),
        "manifest_rows": int(len(manifest)),
        "available_rows": int(available.sum()),
        "manifest": manifest,
        "build_audit": build_audit,
        "run_manifest": run_manifest,
        "errors": errors,
    }


def _semantic_comparison(packets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    left = packets.get("primary", {})
    right = packets.get("repeat", {})
    if left.get("errors") or right.get("errors"):
        return {"compared": [], "errors": errors, "valid": False}
    fields = [
        "selection_sha256",
        "frame_context_sha256",
        "selected_rows",
        "available_rows",
        "unavailable_rows",
        "status_counts",
        "image_size",
        "padding_ratio",
        "resize_policy",
        "lineage_scope",
        "human_review_complete",
        "video_decode_count",
        "video_seek_count",
        "video_frame_reuse_count",
    ]
    compared: dict[str, Any] = {}
    for field in fields:
        left_value = left["build_audit"].get(field)
        right_value = right["build_audit"].get(field)
        compared[field] = {"primary": left_value, "repeat": right_value}
        if left_value != right_value:
            errors.append(f"build_audit_mismatch={field}")
    if left["run_manifest"].get("code_sha") != right["run_manifest"].get("code_sha"):
        errors.append("run_code_sha_mismatch")
    return {"compared": compared, "errors": errors, "valid": not errors}


def _artifact_comparison(
    packets: dict[str, dict[str, Any]],
    payload: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    left = packets.get("primary", {})
    right = packets.get("repeat", {})
    if left.get("errors") or right.get("errors"):
        return {"manifest_byte_identical": False, "errors": errors, "valid": False}
    if left["manifest_sha256"] != right["manifest_sha256"]:
        errors.append("manifest_byte_hash_mismatch")
    tensor_hashes = {}
    for name, packet in (("primary", left), ("repeat", right)):
        tensor_hashes[name] = _tensor_content_hash(
            Path(packet["root"]),
            packet["manifest"],
            expected_rows=int(payload["cache_contract"]["expected_ready_rows"]),
        )
        if tensor_hashes[name]["errors"]:
            errors.extend(f"{name}:{error}" for error in tensor_hashes[name]["errors"])
    if tensor_hashes.get("primary", {}).get("digest") != tensor_hashes.get(
        "repeat", {}
    ).get("digest"):
        errors.append("tensor_content_hash_mismatch")
    return {
        "manifest_byte_identical": left["manifest_sha256"] == right["manifest_sha256"],
        "tensor_content_hashes": tensor_hashes,
        "errors": errors,
        "valid": not errors,
    }


def _tensor_content_hash(
    root: Path,
    manifest: pd.DataFrame,
    *,
    expected_rows: int,
) -> dict[str, Any]:
    available = _to_bool(manifest["visual_context_available"])
    rows = manifest.loc[available].sort_values("image_context_id", kind="mergesort")
    errors: list[str] = []
    if len(rows) != expected_rows:
        errors.append(f"tensor_rows={len(rows)}!={expected_rows}")
    digest = hashlib.sha256()
    checked = 0
    for row in rows.itertuples(index=False):
        path = root / str(row.cache_path)
        if not path.is_file():
            errors.append(f"missing_tensor={path}")
            continue
        tensor_sha = _file_sha256(path)
        digest.update(str(row.image_context_id).encode("utf-8"))
        digest.update(b"\n")
        digest.update(tensor_sha.encode("ascii"))
        digest.update(b"\n")
        checked += 1
    return {
        "digest": digest.hexdigest() if not errors else "",
        "checked_tensors": checked,
        "errors": errors,
    }


def _load_config(path: Path) -> tuple[dict[str, Any], Path]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("union-context repeat config schema mismatch")
    if payload.get("lineage_scope") != LINEAGE_SCOPE:
        raise ValueError("union-context repeat lineage scope mismatch")
    return payload, path.parents[2]


def _validate_bound_implementation(payload: dict[str, Any], root: Path) -> None:
    for name, entry in payload["implementation"].items():
        path = _bound_path(root, entry["path"])
        if entry["sha256"] == "TO_BE_FILLED":
            raise ValueError(f"repeat implementation hash missing: {name}")
        if _file_sha256(path) != entry["sha256"]:
            raise ValueError(f"repeat implementation hash mismatch: {name}")


def _selection_hash(payload: dict[str, Any]) -> str:
    return str(payload["inputs"]["selection_csv"]["sha256"])


def _git_guard(payload: dict[str, Any], root: Path) -> dict[str, Any]:
    guard = payload["execution_guard"]
    entries = [
        line
        for line in _git(
            root,
            "status",
            "--porcelain",
            "--untracked-files=all",
        ).splitlines()
        if line.strip()
    ]
    observed = sorted(_status_path(line) for line in entries)
    allowed = sorted(
        str(path).replace("\\", "/") for path in guard["allowed_dirty_paths"]
    )
    unexpected = sorted(set(observed) - set(allowed))
    required = [
        str(path).replace("\\", "/")
        for path in guard["required_tracked_paths"]
    ]
    untracked = [
        path
        for path in required
        if subprocess.run(
            ["git", "-C", str(root), "ls-files", "--error-unmatch", "--", path],
            capture_output=True,
            check=False,
            text=True,
        ).returncode
        != 0
    ]
    errors: list[str] = []
    if unexpected:
        errors.append(f"unexpected_dirty_paths={unexpected}")
    if untracked:
        errors.append(f"required_paths_untracked={untracked}")
    return {
        "errors": errors,
        "valid": not errors,
        "code_sha": _git(root, "rev-parse", "HEAD").strip(),
    }


def _bound_path(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"path escapes project root: {value}") from error
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _to_bool(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(dtype=bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _status_path(line: str) -> str:
    value = line[3:].strip()
    if " -> " in value:
        value = value.rsplit(" -> ", maxsplit=1)[1]
    return value.replace("\\", "/")


if __name__ == "__main__":
    main()
