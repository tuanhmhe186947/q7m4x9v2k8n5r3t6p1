"""Load and validate the operational Classification V2 lineage config."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

CONFIG_RELATIVE_PATH = Path("configs/classification_v2/lineage_rebuild_v1.yaml")
SOURCE_VERIFICATION_CACHE_RELATIVE_PATH = Path(
    "source_verification/source_bundle_verification_cache.json"
)
SOURCE_VERIFICATION_CACHE_SCHEMA = (
    "classification_v2.source_bundle_verification_cache.v1"
)
STALE_SOURCE_TOKENS = (
    "legacy_full_multigt_masked_nodup_16f",
    "data/raw/legacy_full_multigt_masked_nodup_16f",
)


def repository_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / CONFIG_RELATIVE_PATH).is_file():
            return candidate
    raise FileNotFoundError(CONFIG_RELATIVE_PATH)


def _yaml_load(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise RuntimeError("PyYAML is required to load lineage config") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("lineage config must contain a mapping")
    return payload


def load_config(path: Path | None = None) -> tuple[Path, dict[str, Any]]:
    root = repository_root()
    config_path = (path or root / CONFIG_RELATIVE_PATH).resolve()
    config = _yaml_load(config_path)
    if config.get("schema_version") != "classification_v2.lineage_rebuild.v1":
        raise ValueError("unsupported lineage config version")
    return root, config


def resolve_run_root(root: Path, config: dict[str, Any]) -> Path:
    env_name = str(config["run_root_env"])
    raw = os.environ.get(env_name, str(config["run_root_default"]))
    return Path(os.path.expandvars(raw)).expanduser().resolve()


def resolve_source_path(root: Path, config: dict[str, Any], key: str) -> Path:
    return (root / str(config["source"][key])).resolve()


def canonical_legacy_crop_root() -> Path:
    root, config = load_config()
    return resolve_source_path(root, config, "legacy_crop_root")


def canonical_video_root() -> Path:
    root, config = load_config()
    return resolve_source_path(root, config, "video_root")


def canonical_roi_path() -> Path:
    root, config = load_config()
    return resolve_source_path(root, config, "roi")


def resolve_stage_path(
    root: Path,
    config: dict[str, Any],
    stage: str,
    key: str,
) -> Path:
    relative = str(config["stages"][stage][key])
    return resolve_run_root(root, config) / relative


def stable_stage_ids(config: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(value) for value in config["stage_order"])


def reject_stale_path(value: str | Path) -> None:
    normalized = str(value).replace("\\", "/").lower()
    if any(token in normalized for token in STALE_SOURCE_TOKENS):
        raise ValueError(f"STALE_SOURCE_PATH:{value}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _aggregate_fingerprint(paths: list[Path], base: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(base).as_posix().encode())
        digest.update(b"\0")
        digest.update(_sha256(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _json_fingerprint(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _tree_snapshot(base: Path) -> tuple[list[Path], dict[str, Any]]:
    paths: list[Path] = []
    records: list[dict[str, Any]] = []

    def visit(directory: Path) -> None:
        with os.scandir(directory) as scanner:
            entries = sorted(scanner, key=lambda entry: entry.name)
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                visit(Path(entry.path))
            elif entry.is_file(follow_symlinks=False):
                stat = entry.stat(follow_symlinks=False)
                path = Path(entry.path)
                paths.append(path)
                records.append(
                    {
                        "path": path.relative_to(base).as_posix(),
                        "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                    }
                )

    visit(base)
    paths.sort()
    records.sort(key=lambda record: str(record["path"]))
    return paths, {
        "file_count": len(records),
        "metadata_fingerprint": _json_fingerprint(records),
    }


def _source_authority_paths(
    root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    source = config["source"]
    return {
        "legacy_export": str(
            resolve_source_path(root, config, "legacy_export")
        ),
        "legacy_completion_audit": str(
            resolve_source_path(root, config, "legacy_completion_audit")
        ),
        "legacy_crop_root": str(
            resolve_source_path(root, config, "legacy_crop_root")
        ),
        "cvat_behavior_xml": [
            str((root / value).resolve())
            for value in source["cvat_behavior_xml"]
        ],
        "roi": str(resolve_source_path(root, config, "roi")),
        "video_root": str(
            resolve_source_path(root, config, "video_root")
        ),
        "pen_mask": str(resolve_source_path(root, config, "pen_mask")),
    }


def source_verification_cache_path(
    root: Path,
    config: dict[str, Any],
) -> Path:
    return resolve_run_root(root, config) / SOURCE_VERIFICATION_CACHE_RELATIVE_PATH


def _config_fingerprints(
    root: Path,
    config: dict[str, Any],
    config_path: Path | None,
) -> dict[str, str | None]:
    resolved_path = (
        config_path.resolve()
        if config_path is not None
        else (root / CONFIG_RELATIVE_PATH).resolve()
    )
    return {
        "semantic": _json_fingerprint(config),
        "file_sha256": (
            _sha256(resolved_path) if resolved_path.is_file() else None
        ),
    }


def _source_report_is_valid(
    report: dict[str, Any],
    source: dict[str, Any],
) -> bool:
    return (
        report["legacy_csv_sha256"] == source["expected_legacy_sha256"]
        and report["legacy_csv_rows"] == source["expected_legacy_rows"]
        and report["crop_file_count"] == source["expected_legacy_crop_files"]
        and report["crop_fingerprint"] == source["expected_crop_fingerprint"]
        and report["cvat_xml_count"] == source["expected_cvat_xml_count"]
        and report["cvat_box_rows"] == source["expected_cvat_box_rows"]
        and report["cvat_xml_fingerprint"]
        == source["expected_cvat_xml_fingerprint"]
        and report["roi_sha256"] == source["expected_roi_sha256"]
        and report["pen_mask_sha256"] == source["expected_pen_mask_sha256"]
        and report["completion_audit_sha256"]
        == source["expected_completion_audit_sha256"]
        and report["video_fingerprint"]
        == source["expected_video_fingerprint"]
        and report["projected_mixed_rows"] == source["expected_mixed_rows"]
        and report["completion_audit_status"] == "PASS"
        and report["bundle_fingerprint"]
        == source["expected_bundle_fingerprint"]
    )


def _build_source_report(
    *,
    root: Path,
    config: dict[str, Any],
    crop_files: list[Path],
    video_files: list[Path],
    crop_fingerprint: str | None,
    video_fingerprint: str | None,
) -> dict[str, Any]:
    source = config["source"]
    legacy = resolve_source_path(root, config, "legacy_export")
    audit = resolve_source_path(root, config, "legacy_completion_audit")
    crops = resolve_source_path(root, config, "legacy_crop_root")
    xmls = [root / value for value in source["cvat_behavior_xml"]]
    roi = resolve_source_path(root, config, "roi")
    pen_mask = resolve_source_path(root, config, "pen_mask")
    report: dict[str, Any] = {
        "legacy_csv_path": str(legacy),
        "legacy_csv_exists": legacy.is_file(),
        "legacy_csv_sha256": _sha256(legacy) if legacy.is_file() else None,
        "legacy_csv_rows": (
            max(sum(1 for _ in legacy.open("rb")) - 1, 0)
            if legacy.is_file()
            else None
        ),
        "completion_audit_path": str(audit),
        "completion_audit_exists": audit.is_file(),
        "completion_audit_sha256": _sha256(audit) if audit.is_file() else None,
        "completion_audit_status": (
            json.loads(audit.read_text(encoding="utf-8")).get("status")
            if audit.is_file()
            else None
        ),
        "crop_file_count": len(crop_files) if crops.is_dir() else None,
        "crop_fingerprint": crop_fingerprint,
        "cvat_xml_count": len(xmls),
        "cvat_xml_fingerprint": (
            _aggregate_fingerprint(xmls, root / source["cvat_behavior_root"])
            if all(path.is_file() for path in xmls)
            else None
        ),
        "cvat_box_rows": (
            sum(len(ET.parse(path).findall(".//box")) for path in xmls)
            if all(path.is_file() for path in xmls)
            else None
        ),
        "roi_sha256": _sha256(roi) if roi.is_file() else None,
        "pen_mask_sha256": _sha256(pen_mask) if pen_mask.is_file() else None,
        "video_fingerprint": video_fingerprint,
    }
    report["projected_mixed_rows"] = (
        (report["legacy_csv_rows"] or 0) + (report["cvat_box_rows"] or 0)
        if report["legacy_csv_rows"] is not None
        and report["cvat_box_rows"] is not None
        else None
    )
    fingerprint_fields = {
        key: report[key]
        for key in (
            "legacy_csv_sha256",
            "legacy_csv_rows",
            "completion_audit_sha256",
            "crop_file_count",
            "crop_fingerprint",
            "cvat_xml_count",
            "cvat_xml_fingerprint",
            "cvat_box_rows",
            "roi_sha256",
            "pen_mask_sha256",
            "video_fingerprint",
            "projected_mixed_rows",
        )
    }
    report["bundle_fingerprint"] = _json_fingerprint(fingerprint_fields)
    report["valid"] = _source_report_is_valid(report, source)
    return report


def _cache_record_checksum(record: dict[str, Any]) -> str:
    unsigned = dict(record)
    unsigned.pop("record_sha256", None)
    return _json_fingerprint(unsigned)


def _write_verification_cache(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(record)
    payload["record_sha256"] = _cache_record_checksum(payload)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _verification_failure(
    cache_path: Path,
    errors: list[str],
) -> dict[str, Any]:
    return {
        "valid": False,
        "verification_mode": "fast",
        "verification_cache_path": str(cache_path),
        "verification_cache_used": False,
        "verification_errors": errors,
    }


def source_bundle_report(
    root: Path,
    config: dict[str, Any],
    *,
    verification_mode: str = "fast",
    write_cache: bool = False,
    config_path: Path | None = None,
) -> dict[str, Any]:
    if verification_mode not in {"fast", "full"}:
        raise ValueError(f"UNKNOWN_SOURCE_VERIFICATION_MODE:{verification_mode}")
    started = perf_counter()
    source = config["source"]
    crops = resolve_source_path(root, config, "legacy_crop_root")
    videos = resolve_source_path(root, config, "video_root")
    cache_path = source_verification_cache_path(root, config)
    config_fingerprints = _config_fingerprints(root, config, config_path)
    authority_paths = _source_authority_paths(root, config)
    cache: dict[str, Any] | None = None
    if verification_mode == "fast":
        if not cache_path.is_file():
            return _verification_failure(
                cache_path,
                ["SOURCE_VERIFICATION_CACHE_MISSING:RUN_FULL_SOURCE_VERIFY"],
            )
        try:
            loaded_cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return _verification_failure(
                cache_path,
                ["SOURCE_VERIFICATION_CACHE_MALFORMED:RUN_FULL_SOURCE_VERIFY"],
            )
        errors = []
        if not isinstance(loaded_cache, dict):
            errors.append("SOURCE_VERIFICATION_CACHE_NOT_A_MAPPING")
        else:
            cache = loaded_cache
            if cache.get("record_sha256") != _cache_record_checksum(cache):
                errors.append("SOURCE_VERIFICATION_CACHE_CHECKSUM_MISMATCH")
            if cache.get("schema_version") != SOURCE_VERIFICATION_CACHE_SCHEMA:
                errors.append("SOURCE_VERIFICATION_CACHE_SCHEMA_STALE")
            if cache.get("bundle_id") != source["bundle_id"]:
                errors.append("SOURCE_VERIFICATION_CACHE_BUNDLE_ID_MISMATCH")
            if (
                cache.get("expected_bundle_fingerprint")
                != source["expected_bundle_fingerprint"]
            ):
                errors.append(
                    "SOURCE_VERIFICATION_CACHE_BUNDLE_FINGERPRINT_MISMATCH"
                )
            if cache.get("config_fingerprints") != config_fingerprints:
                errors.append("SOURCE_VERIFICATION_CACHE_CONFIG_MISMATCH")
            if cache.get("source_authority_paths") != authority_paths:
                errors.append("SOURCE_VERIFICATION_CACHE_PATH_MISMATCH")
            if not isinstance(cache.get("verified_report"), dict):
                errors.append("SOURCE_VERIFICATION_CACHE_REPORT_MISSING")
        if errors:
            return _verification_failure(cache_path, errors)

    crop_files, crop_inventory = (
        _tree_snapshot(crops) if crops.is_dir() else ([], None)
    )
    video_files, video_inventory = (
        _tree_snapshot(videos) if videos.is_dir() else ([], None)
    )
    inventory_finished = perf_counter()

    if verification_mode == "full":
        heavy_hash_started = perf_counter()
        report = _build_source_report(
            root=root,
            config=config,
            crop_files=crop_files,
            video_files=video_files,
            crop_fingerprint=(
                _aggregate_fingerprint(crop_files, crops)
                if crops.is_dir()
                else None
            ),
            video_fingerprint=(
                _aggregate_fingerprint(video_files, videos)
                if videos.is_dir()
                else None
            ),
        )
        heavy_hash_finished = perf_counter()
        _, crop_inventory_after = (
            _tree_snapshot(crops) if crops.is_dir() else ([], None)
        )
        _, video_inventory_after = (
            _tree_snapshot(videos) if videos.is_dir() else ([], None)
        )
        errors = []
        if crop_inventory != crop_inventory_after:
            errors.append("CROP_TREE_CHANGED_DURING_FULL_VERIFICATION")
        if video_inventory != video_inventory_after:
            errors.append("VIDEO_TREE_CHANGED_DURING_FULL_VERIFICATION")
        report.update(
            verification_mode="full",
            verification_cache_path=str(cache_path),
            verification_cache_used=False,
            verification_errors=errors,
            verification_phase_seconds={
                "tree_inventory": round(inventory_finished - started, 3),
                "exact_hash_and_small_authority": round(
                    heavy_hash_finished - heavy_hash_started,
                    3,
                ),
                "stability_inventory": round(
                    perf_counter() - heavy_hash_finished,
                    3,
                ),
            },
        )
        report["valid"] = bool(report["valid"] and not errors)
        if write_cache and report["valid"]:
            record = {
                "schema_version": SOURCE_VERIFICATION_CACHE_SCHEMA,
                "bundle_id": source["bundle_id"],
                "expected_bundle_fingerprint": source[
                    "expected_bundle_fingerprint"
                ],
                "config_fingerprints": config_fingerprints,
                "source_authority_paths": authority_paths,
                "crop_inventory": crop_inventory_after,
                "video_inventory": video_inventory_after,
                "verified_report": {
                    key: value
                    for key, value in report.items()
                    if not key.startswith("verification_")
                },
                "verified_at_utc": datetime.now(timezone.utc).isoformat(),
                "verification_mode": "full_byte_hash",
            }
            _write_verification_cache(cache_path, record)
            report["verification_cache_written"] = True
        else:
            report["verification_cache_written"] = False
        return report

    errors = []
    assert cache is not None
    if cache.get("crop_inventory") != crop_inventory:
        errors.append("SOURCE_VERIFICATION_CACHE_CROP_METADATA_MISMATCH")
    if cache.get("video_inventory") != video_inventory:
        errors.append("SOURCE_VERIFICATION_CACHE_VIDEO_METADATA_MISMATCH")
    cached_report = cache["verified_report"]
    if errors:
        return _verification_failure(cache_path, errors)
    report = _build_source_report(
        root=root,
        config=config,
        crop_files=crop_files,
        video_files=video_files,
        crop_fingerprint=cached_report.get("crop_fingerprint"),
        video_fingerprint=cached_report.get("video_fingerprint"),
    )
    if report.get("bundle_fingerprint") != cached_report.get(
        "bundle_fingerprint"
    ):
        errors.append("SOURCE_VERIFICATION_CACHE_REPORT_DRIFT")
    report.update(
        verification_mode="fast",
        verification_cache_path=str(cache_path),
        verification_cache_used=True,
        verification_errors=errors,
        verification_phase_seconds={
            "tree_inventory": round(inventory_finished - started, 3),
            "cache_and_small_authority": round(
                perf_counter() - inventory_finished,
                3,
            ),
        },
    )
    report["valid"] = bool(report["valid"] and not errors)
    return report


def source_paths(config: dict[str, Any]) -> list[str]:
    paths = [
        config["source"]["legacy_run"],
        config["source"]["legacy_export"],
        config["source"]["legacy_crop_root"],
        config["source"]["legacy_completion_audit"],
        config["source"]["cvat_behavior_root"],
        *config["source"]["cvat_behavior_xml"],
        config["source"]["roi"],
        config["source"]["video_root"],
    ]
    return [str(value) for value in paths]


def current_git_sha(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def expand_template(value: str, run_root: Path) -> str:
    return re.sub(r"\$\{CLASSIFICATION_V2_RUN_ROOT\}", str(run_root), value)


__all__ = [
    "CONFIG_RELATIVE_PATH",
    "canonical_legacy_crop_root",
    "canonical_roi_path",
    "canonical_video_root",
    "current_git_sha",
    "expand_template",
    "load_config",
    "reject_stale_path",
    "resolve_run_root",
    "resolve_source_path",
    "resolve_stage_path",
    "source_bundle_report",
    "source_paths",
    "source_verification_cache_path",
    "stable_stage_ids",
]
