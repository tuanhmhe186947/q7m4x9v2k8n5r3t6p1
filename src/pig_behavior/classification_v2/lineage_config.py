"""Load and validate the operational Classification V2 lineage config."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

CONFIG_RELATIVE_PATH = Path("configs/classification_v2/lineage_rebuild_v1.yaml")
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


def source_bundle_report(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    source = config["source"]
    legacy = resolve_source_path(root, config, "legacy_export")
    audit = resolve_source_path(root, config, "legacy_completion_audit")
    crops = resolve_source_path(root, config, "legacy_crop_root")
    xmls = [root / value for value in source["cvat_behavior_xml"]]
    roi = resolve_source_path(root, config, "roi")
    videos = resolve_source_path(root, config, "video_root")
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
        "crop_file_count": (
            sum(1 for path in crops.rglob("*") if path.is_file())
            if crops.is_dir()
            else None
        ),
        "crop_fingerprint": (
            _aggregate_fingerprint(
                [path for path in crops.rglob("*") if path.is_file()],
                crops,
            )
            if crops.is_dir()
            else None
        ),
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
        "video_fingerprint": (
            _aggregate_fingerprint(
                [path for path in videos.rglob("*") if path.is_file()],
                videos,
            )
            if videos.is_dir()
            else None
        ),
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
    report["bundle_fingerprint"] = hashlib.sha256(
        json.dumps(
            fingerprint_fields,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    report["valid"] = (
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
    "stable_stage_ids",
]
