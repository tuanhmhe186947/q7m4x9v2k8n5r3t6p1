"""Audit CVAT source annotations and print exact human-correction locations."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from pig_behavior.classification_v2.sources.cvat_annotation_quality import (
    SOURCE_LEGACY_TASK,
    SOURCE_TRACKING_XML,
    audit_legacy_task_export,
    audit_tracking_xml,
    combine_annotation_audits,
)


def main() -> None:
    args = _parse_args()
    reports: list[dict[str, Any]] = []

    if args.task_export_root is not None:
        reports.append(
            _run_source(
                source_kind=SOURCE_LEGACY_TASK,
                source_path=args.task_export_root,
                audit=lambda: audit_legacy_task_export(
                    args.task_export_root
                ),
            )
        )

    xml_paths = set(args.tracking_xml)
    if args.tracking_xml_dir is not None:
        xml_paths.update(args.tracking_xml_dir.glob("*.xml"))
    for path in sorted(xml_paths):
        reports.append(
            _run_source(
                source_kind=SOURCE_TRACKING_XML,
                source_path=path,
                audit=lambda path=path: audit_tracking_xml(path),
            )
        )

    if not reports:
        raise SystemExit(
            "Provide --task-export-root, --tracking-xml-dir, or "
            "--tracking-xml."
        )

    result = combine_annotation_audits(reports)
    if args.excluded_actor_key_csv is not None:
        result = _apply_actor_exclusion_policy(
            result,
            args.excluded_actor_key_csv,
        )
    _write_outputs(
        result,
        output_json=args.output_json,
        issues_csv=args.issues_csv,
    )
    _print_result(result, print_issues=args.print_issues)

    has_error = result["summary"]["error_count"] > 0
    review_required = result["summary"]["review_count"] > 0
    if has_error or (review_required and not args.allow_review_required):
        raise SystemExit(1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-export-root",
        type=Path,
        help=(
            "Root containing task_*/task.json, annotations.xml/json, and "
            "data/manifest.jsonl."
        ),
    )
    parser.add_argument(
        "--tracking-xml-dir",
        type=Path,
        help="Directory containing CVAT interpolation tracking XML files.",
    )
    parser.add_argument(
        "--tracking-xml",
        type=Path,
        action="append",
        default=[],
        help="One CVAT tracking XML; repeat the option for multiple files.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional complete machine-readable audit.",
    )
    parser.add_argument(
        "--issues-csv",
        type=Path,
        help="Optional flat issue table for filtering by task/frame/code.",
    )
    parser.add_argument(
        "--print-issues",
        action="store_true",
        help="Print every issue after the compact summary.",
    )
    parser.add_argument(
        "--allow-review-required",
        action="store_true",
        help=(
            "Exit zero when only review issues remain. Errors always exit "
            "nonzero."
        ),
    )
    parser.add_argument(
        "--excluded-actor-key-csv",
        type=Path,
        help=(
            "Explicit policy CSV with group_id,pig_id,reason. Matching review "
            "issues become declared exclusions."
        ),
    )
    return parser.parse_args()


def _apply_actor_exclusion_policy(
    result: dict[str, Any],
    policy_path: Path,
) -> dict[str, Any]:
    """Convert only matching review issues into explicit exclusions."""
    if not policy_path.is_file():
        raise FileNotFoundError(f"actor_exclusion_policy_not_found={policy_path}")
    with policy_path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    required = {"group_id", "pig_id", "reason"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(
            "actor_exclusion_policy_requires_group_id_pig_id_reason_columns"
        )
    keys: dict[tuple[str, str], str] = {}
    for row in rows:
        key = (row["group_id"].strip(), row["pig_id"].strip())
        reason = row["reason"].strip()
        if not all(key) or not reason:
            raise ValueError("actor_exclusion_policy_contains_blank_values")
        if key in keys:
            raise ValueError("actor_exclusion_policy_contains_duplicate_keys")
        keys[key] = reason

    issue_keys = {
        (str(issue.get("group_id", "")), str(issue.get("pig_id", "")))
        for issue in result["issues"]
    }
    unknown = sorted(set(keys).difference(issue_keys))
    if unknown:
        raise ValueError(f"actor_exclusion_policy_unknown_keys={unknown}")
    excluded = 0
    for issue in result["issues"]:
        key = (str(issue.get("group_id", "")), str(issue.get("pig_id", "")))
        if key not in keys:
            continue
        if issue["severity"] != "review":
            raise ValueError(
                f"actor_exclusion_policy_matches_non_review_issue={key}"
            )
        issue["severity"] = "excluded"
        issue["suggested_action"] = (
            "Excluded by explicit operator policy: " + keys[key]
        )
        issue["evidence"] = {
            **issue.get("evidence", {}),
            "exclusion_policy_path": str(policy_path.resolve()),
            "exclusion_reason": keys[key],
        }
        excluded += 1

    severity_counts = {
        severity: sum(issue["severity"] == severity for issue in result["issues"])
        for severity in ("error", "review", "info", "excluded")
    }
    result["status"] = (
        "FAIL"
        if severity_counts["error"]
        else (
            "PASS_WITH_DECLARED_EXCLUSIONS"
            if excluded
            else ("REVIEW_REQUIRED" if severity_counts["review"] else "PASS")
        )
    )
    result["summary"].update(
        {
            "error_count": severity_counts["error"],
            "review_count": severity_counts["review"],
            "info_count": severity_counts["info"],
            "excluded_count": severity_counts["excluded"],
        }
    )
    result["exclusion_policy"] = {
        "path": str(policy_path.resolve()),
        "keys": [
            {"group_id": group_id, "pig_id": pig_id, "reason": reason}
            for (group_id, pig_id), reason in sorted(keys.items())
        ],
    }
    return result


def _run_source(
    *,
    source_kind: str,
    source_path: Path,
    audit: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    try:
        return audit()
    except (
        FileNotFoundError,
        OSError,
        ET.ParseError,
        ValueError,
    ) as exc:
        return {
            "source_kind": source_kind,
            "source_path": str(source_path),
            "status": "FAIL",
            "summary": {"load_error": str(exc)},
            "issues": [
                {
                    "severity": "error",
                    "code": "source_load_error",
                    "source_kind": source_kind,
                    "annotation_path": str(source_path),
                    "task": "",
                    "video_key": "",
                    "group_id": "",
                    "pig_id": None,
                    "slot": None,
                    "frame_id": None,
                    "frame_position_1based": None,
                    "total_frames": None,
                    "image_name": "",
                    "observed_slots": [],
                    "missing_slots": [],
                    "evidence": {"error": str(exc)},
                    "suggested_action": (
                        "Repair the input path or annotation schema."
                    ),
                }
            ],
        }


def _write_outputs(
    result: dict[str, Any],
    *,
    output_json: Path | None,
    issues_csv: Path | None,
) -> None:
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    if issues_csv is not None:
        issues_csv.parent.mkdir(parents=True, exist_ok=True)
        rows = [_flat_issue(issue) for issue in result["issues"]]
        fieldnames = list(_flat_issue({}).keys())
        with issues_csv.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def _flat_issue(issue: dict[str, Any]) -> dict[str, Any]:
    fields = [
        "severity",
        "code",
        "source_kind",
        "annotation_path",
        "task",
        "video_key",
        "group_id",
        "pig_id",
        "slot",
        "frame_id",
        "frame_position_1based",
        "total_frames",
        "image_name",
        "suggested_action",
    ]
    row = {field: issue.get(field, "") for field in fields}
    row["observed_slots"] = "|".join(
        map(str, issue.get("observed_slots", []))
    )
    row["missing_slots"] = "|".join(
        map(str, issue.get("missing_slots", []))
    )
    row["evidence_json"] = json.dumps(
        issue.get("evidence", {}),
        ensure_ascii=False,
        sort_keys=True,
    )
    return row


def _print_result(
    result: dict[str, Any],
    *,
    print_issues: bool,
) -> None:
    compact = {
        "status": result["status"],
        "source_count": result["source_count"],
        "summary": result["summary"],
        "sources": result["sources"],
    }
    print(json.dumps(compact, indent=2, ensure_ascii=True))
    if print_issues:
        print(json.dumps(result["issues"], indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
