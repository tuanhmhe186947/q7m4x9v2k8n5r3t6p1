"""Launch Classification V2 operator GUIs from one versioned path profile."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROFILE = (
    PROJECT_ROOT / "configs" / "classification_v2" / "gui_operator_profile_v1.json"
)
PROFILE_SCHEMA = "classification_v2.gui_operator_profile.v1"
GUI_DIR = PROJECT_ROOT / "scripts" / "classification_v2" / "01_review_units_gui"
BEHAVIOR_GUI = GUI_DIR / "review_final_behavior_gui_v1.py"
MINI_CVAT_GUI = GUI_DIR / "review_identity_continuity_gui_v2.py"
SESSION_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class LauncherError(ValueError):
    """Raised when the operator profile or requested launch is unsafe."""


def _require_mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LauncherError(f"profile_field_must_be_object={name}")
    return value


def _require_text(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise LauncherError(f"profile_field_missing_or_empty={key}")
    return value.strip()


def _require_path_value(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    if (
        isinstance(value, list)
        and value
        and all(isinstance(part, str) and part.strip() for part in value)
    ):
        return str(Path(value[0]).joinpath(*value[1:]))
    raise LauncherError(f"profile_path_missing_or_empty={key}")


def load_profile(path: Path) -> dict[str, Any]:
    """Load and validate only the stable top-level launcher contract."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise LauncherError(f"profile_not_found={path}") from exc
    except json.JSONDecodeError as exc:
        raise LauncherError(f"profile_invalid_json={path}:{exc}") from exc
    profile = _require_mapping(payload, "root")
    if profile.get("schema_version") != PROFILE_SCHEMA:
        raise LauncherError(
            "profile_schema_mismatch="
            f"{profile.get('schema_version')!r};expected={PROFILE_SCHEMA}"
        )
    common = _require_mapping(profile.get("common"), "common")
    behavior = _require_mapping(profile.get("behavior"), "behavior")
    mini_cvat = _require_mapping(profile.get("mini_cvat"), "mini_cvat")
    for key in (
        "review_units_csv",
        "frame_features_csv",
        "video_root",
        "raw_root",
        "roi_coco_json",
    ):
        _require_path_value(common, key)
    _require_text(profile, "default_reviewer")
    _require_text(behavior, "output_dir")
    _require_text(mini_cvat, "output_root")
    return profile


def resolve_profile_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _common_paths(profile: dict[str, Any]) -> dict[str, Path]:
    common = _require_mapping(profile["common"], "common")
    return {
        key: resolve_profile_path(_require_path_value(common, key)) for key in common
    }


def _behavior_command(
    profile: dict[str, Any],
    *,
    max_items: int,
    start_review_unit_id: str,
    prepare_frame_cache_only: bool,
) -> tuple[list[str], list[Path]]:
    common = _common_paths(profile)
    behavior = _require_mapping(profile["behavior"], "behavior")
    output_dir = resolve_profile_path(_require_text(behavior, "output_dir"))
    command = [
        sys.executable,
        str(BEHAVIOR_GUI),
        "--review-units-csv",
        str(common["review_units_csv"]),
        "--frame-features-csv",
        str(common["frame_features_csv"]),
        "--output-dir",
        str(output_dir),
        "--video-root",
        str(common["video_root"]),
        "--raw-root",
        str(common["raw_root"]),
        "--roi-coco-json",
        str(common["roi_coco_json"]),
    ]
    if max_items > 0:
        command.extend(("--max-items", str(max_items)))
    if start_review_unit_id:
        command.extend(("--start-review-unit-id", start_review_unit_id))
    if prepare_frame_cache_only:
        command.append("--prepare-frame-cache-only")
    required = [
        BEHAVIOR_GUI,
        common["review_units_csv"],
        common["frame_features_csv"],
        common["video_root"],
        common["raw_root"],
        common["roi_coco_json"],
    ]
    return command, required


def _session_output_dir(profile: dict[str, Any], session_name: str) -> Path:
    if not SESSION_NAME_PATTERN.fullmatch(session_name) or session_name in {".", ".."}:
        raise LauncherError(f"unsafe_session_name={session_name!r}")
    mini_cvat = _require_mapping(profile["mini_cvat"], "mini_cvat")
    root = resolve_profile_path(_require_text(mini_cvat, "output_root"))
    output_dir = (root / session_name).resolve()
    if not output_dir.is_relative_to(root):
        raise LauncherError(f"session_output_outside_root={output_dir}")
    return output_dir


def _mini_cvat_command(
    profile: dict[str, Any],
    *,
    session_name: str,
    reviewer: str,
    review_item_ids: list[str],
    editable_pig_ids: list[str],
    apply_source_csvs: list[Path],
    apply_source_xml: Path | None,
    apply_group_id: str,
) -> tuple[list[str], list[Path]]:
    if bool(apply_source_csvs) != bool(apply_source_xml):
        raise LauncherError("source_apply_requires_both_csv_and_xml")
    common = _common_paths(profile)
    command = [
        sys.executable,
        str(MINI_CVAT_GUI),
        "--review-units-csv",
        str(common["review_units_csv"]),
        "--frame-features-csv",
        str(common["frame_features_csv"]),
        "--output-dir",
        str(_session_output_dir(profile, session_name)),
        "--reviewer",
        reviewer,
        "--video-root",
        str(common["video_root"]),
    ]
    for review_item_id in review_item_ids:
        command.extend(("--review-item-id", review_item_id))
    for pig_id in editable_pig_ids:
        command.extend(("--editable-pig-id", pig_id))
    resolved_csvs = [resolve_profile_path(str(path)) for path in apply_source_csvs]
    for source_csv in resolved_csvs:
        command.extend(("--apply-source-csv", str(source_csv)))
    resolved_xml = (
        resolve_profile_path(str(apply_source_xml)) if apply_source_xml else None
    )
    if resolved_xml is not None:
        command.extend(("--apply-source-xml", str(resolved_xml)))
    if apply_group_id:
        command.extend(("--apply-group-id", apply_group_id))
    required = [
        MINI_CVAT_GUI,
        common["review_units_csv"],
        common["frame_features_csv"],
        common["video_root"],
        *resolved_csvs,
    ]
    if resolved_xml is not None:
        required.append(resolved_xml)
    return command, required


def _check_required_paths(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise LauncherError("required_path_missing=" + "|".join(missing))


def _run(command: list[str], required: list[Path], *, dry_run: bool) -> int:
    print("COMMAND=" + subprocess.list2cmdline(command), flush=True)
    if dry_run:
        print("DRY_RUN=YES", flush=True)
        return 0
    _check_required_paths(required)
    result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    return result.returncode


def _status(profile_path: Path, profile: dict[str, Any]) -> int:
    common = _common_paths(profile)
    print(f"PROFILE={profile_path.resolve()}")
    missing = False
    for name, path in common.items():
        exists = path.exists()
        missing = missing or not exists
        print(f"INPUT {name} EXISTS={'YES' if exists else 'NO'} PATH={path}")
    behavior = _require_mapping(profile["behavior"], "behavior")
    mini_cvat = _require_mapping(profile["mini_cvat"], "mini_cvat")
    print(
        "OUTPUT behavior_output_dir INSPECTED=NO PATH="
        + str(resolve_profile_path(_require_text(behavior, "output_dir")))
    )
    print(
        "OUTPUT mini_cvat_output_root INSPECTED=NO PATH="
        + str(resolve_profile_path(_require_text(mini_cvat, "output_root")))
    )
    return 2 if missing else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("status", help="Check configured inputs without reading ledgers.")

    behavior = subparsers.add_parser("behavior", help="Open the final Behavior GUI.")
    behavior.add_argument("--max-items", type=int, default=0)
    behavior.add_argument("--start-review-unit-id", default="")
    behavior.add_argument("--prepare-frame-cache-only", action="store_true")
    behavior.add_argument("--dry-run", action="store_true")

    mini_cvat = subparsers.add_parser("mini-cvat", help="Open mini-CVAT V2.")
    mini_cvat.add_argument("--session-name", required=True)
    mini_cvat.add_argument("--reviewer", default="")
    mini_cvat.add_argument("--review-item-id", action="append", required=True)
    mini_cvat.add_argument("--editable-pig-id", action="append", required=True)
    mini_cvat.add_argument("--apply-source-csv", action="append", type=Path, default=[])
    mini_cvat.add_argument("--apply-source-xml", type=Path)
    mini_cvat.add_argument("--apply-group-id", default="")
    mini_cvat.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        profile_path = resolve_profile_path(str(args.profile))
        profile = load_profile(profile_path)
        if args.command == "status":
            return _status(profile_path, profile)
        if args.command == "behavior":
            command, required = _behavior_command(
                profile,
                max_items=args.max_items,
                start_review_unit_id=args.start_review_unit_id.strip(),
                prepare_frame_cache_only=args.prepare_frame_cache_only,
            )
            return _run(command, required, dry_run=args.dry_run)
        reviewer = args.reviewer.strip() or _require_text(profile, "default_reviewer")
        command, required = _mini_cvat_command(
            profile,
            session_name=args.session_name,
            reviewer=reviewer,
            review_item_ids=args.review_item_id,
            editable_pig_ids=args.editable_pig_id,
            apply_source_csvs=args.apply_source_csv,
            apply_source_xml=args.apply_source_xml,
            apply_group_id=args.apply_group_id.strip(),
        )
        return _run(command, required, dry_run=args.dry_run)
    except LauncherError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
