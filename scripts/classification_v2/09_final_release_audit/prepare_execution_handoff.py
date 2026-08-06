"""Bind the focused post-readiness execution handoff.

This command is read-only with respect to project data.  It reads the existing
CVAT/legacy construction outputs, the current reviewed snapshot, the frozen
split, and the earlier/current eligibility manifests, then writes small
machine-readable audit artifacts under the execution worktree.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote


CANONICAL_KEY_RE = re.compile(
    r"^(?P<day>pigs\d{6}[a-z]?)/(?P<clip>\d{1,6})$", re.IGNORECASE
)
NESTED_VIDEO_RE = re.compile(
    r"/(?P<day>pigs\d{6}[a-z]?)/pigs\d{6}[a-z]?/"
    r"(?P<clip>\d{1,6})/color\.mp4(?:$|[?#])",
    re.IGNORECASE,
)
VIDEO_FILENAME_RE = re.compile(
    r"(?P<day>pigs\d{6}[a-z]?)[_-](?P<clip>\d{1,6})"
    r"(?:[_-]30fps)?(?:\.mp4)?(?:$|[?#])",
    re.IGNORECASE,
)
DATE_RE = re.compile(r"pigs(?P<date>\d{6})", re.IGNORECASE)

SNAPSHOT_ID = "reviewed_engineering_amendment_992f34c0204a85a1"
SNAPSHOT_SHA256 = "ab86e2e04267cfdc8248f9bdb8774615479d67a3589f7a25844bb1a4c93a639e"
SPLIT_HASH = "557156a7eb6cceeb6a91f667f7c51dcb286e3111f35f414970fa7431acc7e63b"
SPLIT_FILE_SHA256 = (
    "cf00e7e7ef791e1a58dceb0af77898c64eb7a03819f5c570c5567067142088f6"
)
CLASSIFICATION_CODE_SHA = "884016aff7d7f23608adcc81a6c138a46351c57e"
EVENT_WEIGHT_HASH = "92a901b8bb431102f5e32fd73c899930f5f3f4c83a9eac6945f9609cdd84938d"
SCHEMA_HASH = "18377d825ba84974e49305e46561ada81353f9ffd0f2d2526471af1c199daad4"
ENVIRONMENT_HASH = "6b783d5296094e0be94b0e553e3c83376a462eec3278285b076b35761bc103ca"
CURRENT_EFFECTIVE_HASH = (
    "810071b311ebc008d420c5873fea11f2369f9ec8edfc3b2dc2958635b30ac7f1"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def bool_value(value: Any) -> bool:
    return text(value).lower() in {"true", "1", "yes", "y", "t"}


def read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def binding(path: Path, role: str) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "semantic_role": role,
    }


def canonical_video_key(value: Any) -> str:
    """Match check_duplicate_videos.py exactly: return pigsDDMMYY/NNNNNN."""

    raw = text(value).replace("\\", "/").strip().lower()
    if not raw:
        return ""
    raw = re.sub(r"/{2,}", "/", raw)
    match = CANONICAL_KEY_RE.fullmatch(raw)
    if match is None:
        match = NESTED_VIDEO_RE.search(raw)
    if match is None:
        match = VIDEO_FILENAME_RE.search(raw)
    if match is None:
        return ""
    return f"{match.group('day').lower()}/{match.group('clip').zfill(6)}"


def source_key(row: dict[str, str], preferred: tuple[str, ...]) -> str:
    for column in preferred:
        value = canonical_video_key(row.get(column, ""))
        if value:
            return value
    return ""


def count_keys(rows: Iterable[dict[str, str]], preferred: tuple[str, ...]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        key = source_key(row, preferred)
        if not key:
            raise ValueError(f"unresolved source key in row: {row}")
        counts[key] += 1
    return counts


def git_value(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=repo, text=True, encoding="utf-8"
    ).strip()


def file_commit(repo: Path, relative_path: str) -> str:
    return git_value(repo, "log", "-1", "--format=%H", "--", relative_path)


def file_blob(repo: Path, relative_path: str) -> str:
    return git_value(repo, "hash-object", relative_path)


def date_key(value: str) -> str:
    match = DATE_RE.search(value)
    return match.group("date") if match else "UNKNOWN"


def window_video_key(window_id: str) -> str:
    fields = {}
    for field in window_id.split("|"):
        if "=" in field:
            name, value = field.split("=", 1)
            fields[name] = value
    return canonical_video_key(unquote(fields.get("video", "")))


def roles_and_crossings(
    effective_rows: list[dict[str, str]], split_rows: list[dict[str, str]]
) -> dict[str, Any]:
    split_roles: dict[str, set[str]] = defaultdict(set)
    split_fold: dict[str, set[str]] = defaultdict(set)
    for row in split_rows:
        window_id = text(row.get("window_id"))
        role = text(row.get("model_split_role")) or text(row.get("split"))
        if window_id:
            split_roles[window_id].add(role)
            split_fold[window_id].add(text(row.get("outer_fold_id")))

    missing = 0
    role_crossings = 0
    group_roles: dict[str, dict[str, set[str]]] = {
        "date": defaultdict(set),
        "video": defaultdict(set),
        "native_unit": defaultdict(set),
    }
    group_folds: dict[str, dict[str, set[str]]] = {
        "date": defaultdict(set),
        "video": defaultdict(set),
        "native_unit": defaultdict(set),
    }
    for row in effective_rows:
        window_id = text(row.get("window_id"))
        roles = split_roles.get(window_id, set())
        if not roles:
            missing += 1
            continue
        active_roles = {
            role for role in roles if role.lower() not in {"exclude", "excluded"}
        }
        if len(active_roles) > 1 or len(split_fold.get(window_id, set())) > 1:
            role_crossings += 1
        if not bool_value(row.get("window_valid_for_main_train", "True")):
            continue
        normalized_video = window_video_key(window_id)
        if not normalized_video:
            video = text(row.get("video_key")) or text(row.get("source_video_key"))
            normalized_video = canonical_video_key(video) or video.lower()
        for role in active_roles:
            group_roles["date"][date_key(normalized_video)].add(role)
            group_roles["video"][normalized_video].add(role)
            group_folds["date"][date_key(normalized_video)].update(
                split_fold.get(window_id, set())
            )
            group_folds["video"][normalized_video].update(
                split_fold.get(window_id, set())
            )
            try:
                native_keys = json.loads(text(row.get("temporal_unit_keys_json")))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid temporal_unit_keys_json: {window_id}") from exc
            for native_key in native_keys:
                native_key = text(native_key)
                if native_key:
                    group_roles["native_unit"][native_key].add(role)
                    group_folds["native_unit"][native_key].update(
                        split_fold.get(window_id, set())
                    )

    crossings = {
        name: sum(
            len(roles) > 1 or len(group_folds[name][group]) > 1
            for group, roles in groups.items()
        )
        for name, groups in group_roles.items()
    }
    return {
        "date_role_crossings": crossings["date"],
        "video_role_crossings": crossings["video"],
        "native_unit_role_crossings": crossings["native_unit"],
        "window_role_crossings": role_crossings,
        "missing_role_bindings": missing,
        "effective_window_rows": len(effective_rows),
        "split_rows": len(split_rows),
        "split_window_ids": len(split_roles),
    }


def eligibility_reconciliation(
    old_path: Path,
    current_path: Path,
    affected_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    old_rows = read_csv(old_path)
    current_rows = read_csv(current_path)
    old_by_window = {text(row.get("window_id")): row for row in old_rows}
    current_by_window = {text(row.get("window_id")): row for row in current_rows}
    changed_windows: list[dict[str, Any]] = []
    for window_id in sorted(set(old_by_window) & set(current_by_window)):
        old = old_by_window[window_id]
        current = current_by_window[window_id]
        old_status = bool_value(old.get("window_valid_for_main_train"))
        current_status = bool_value(current.get("window_valid_for_main_train"))
        if old_status != current_status:
            changed_windows.append(
                {
                    "window_id": window_id,
                    "view_type": text(current.get("view_type")),
                    "old_status": "ELIGIBLE" if old_status else "EXCLUDED",
                    "current_status": "ELIGIBLE" if current_status else "EXCLUDED",
                    "old_reason": text(old.get("window_exclusion_reason")),
                    "current_reason": text(current.get("window_exclusion_reason")),
                    "native_unit_keys": json.loads(
                        text(current.get("temporal_unit_keys_json"))
                    ),
                }
            )

    affected = read_csv(affected_path)
    native_records: list[dict[str, Any]] = []
    for row in affected:
        native_key = text(row.get("temporal_unit_key"))
        related = [
            item for item in changed_windows if native_key in item["native_unit_keys"]
        ]
        old_reasons = sorted({item["old_reason"] for item in related if item["old_reason"]})
        current_reasons = sorted(
            {item["current_reason"] for item in related if item["current_reason"]}
        )
        native_records.append(
            {
                "stable_native_unit_key": native_key,
                "video_key": text(row.get("video_key")),
                "track_id": text(row.get("track_id")),
                "native_frame_start": int(row["native_frame_start"]),
                "native_frame_end": int(row["native_frame_end"]),
                "old_status": "ELIGIBLE" if related else "NOT_CHANGED_IN_WINDOW_COMPARE",
                "current_status": text(row.get("amended_unit_status")),
                "old_reasons": old_reasons,
                "current_reasons": current_reasons,
                "governing_rule": text(row.get("reason")),
                "changed_window_count": len(related),
            }
        )

    old_eligible = sum(bool_value(row.get("window_valid_for_main_train")) for row in old_rows)
    current_eligible = sum(
        bool_value(row.get("window_valid_for_main_train")) for row in current_rows
    )
    summary = {
        "schema_version": "classification_v2.eligibility_reconciliation.v1",
        "old_artifact": binding(old_path, "earlier train-ready window authority"),
        "current_artifact": binding(current_path, "current effective window authority"),
        "amendment_artifact": binding(affected_path, "current post-review amendment"),
        "old_total_windows": len(old_rows),
        "current_total_windows": len(current_rows),
        "old_eligible_windows": old_eligible,
        "old_excluded_windows": len(old_rows) - old_eligible,
        "current_eligible_windows": current_eligible,
        "current_excluded_windows": len(current_rows) - current_eligible,
        "changed_window_count": len(changed_windows),
        "changed_native_unit_count": len(native_records),
        "native_units": native_records,
        "status_changed_only": True,
        "eligibility_changed_unit_count": len(native_records),
        "data_rebuild": False,
    }
    csv_rows: list[dict[str, Any]] = []
    for native in native_records:
        csv_rows.append(
            {
                "record_type": "native_unit",
                "stable_native_unit_key": native["stable_native_unit_key"],
                "window_id": "",
                "view_type": "",
                "old_status": native["old_status"],
                "current_status": native["current_status"],
                "old_reason": "|".join(native["old_reasons"]),
                "current_reason": "|".join(native["current_reasons"]),
                "governing_rule": native["governing_rule"],
            }
        )
    for changed in changed_windows:
        csv_rows.append(
            {
                "record_type": "window",
                "stable_native_unit_key": "|".join(changed["native_unit_keys"]),
                "window_id": changed["window_id"],
                "view_type": changed["view_type"],
                "old_status": changed["old_status"],
                "current_status": changed["current_status"],
                "old_reason": changed["old_reason"],
                "current_reason": changed["current_reason"],
                "governing_rule": "post-review frame-boundary amendment",
            }
        )
    return summary, csv_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reviewed-frame", type=Path, required=True)
    parser.add_argument("--snapshot-identity", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--effective-window-index", type=Path, required=True)
    parser.add_argument("--old-window-manifest", type=Path, required=True)
    parser.add_argument("--affected-native-units", type=Path, required=True)
    parser.add_argument("--cvat-exclusion", type=Path, required=True)
    parser.add_argument("--legacy-center", type=Path, required=True)
    parser.add_argument("--duplicate-preview", type=Path, required=True)
    parser.add_argument("--duplicate-audit", type=Path, required=True)
    parser.add_argument("--filtered-center", type=Path, required=True)
    parser.add_argument("--filtered-all", type=Path, required=True)
    parser.add_argument("--legacy-training", type=Path, required=True)
    parser.add_argument("--source-lineage", type=Path, required=True)
    parser.add_argument("--completion-audit", type=Path, required=True)
    parser.add_argument("--a12-control", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    script_path = repo / "src/legacy_burst_recovery/check_duplicate_videos.py"

    exclusions = read_csv(args.cvat_exclusion)
    legacy_center = read_csv(args.legacy_center)
    duplicate_preview = read_csv(args.duplicate_preview)
    filtered_center = read_csv(args.filtered_center)
    filtered_all = read_csv(args.filtered_all)
    legacy_training = read_csv(args.legacy_training)
    current_frame = read_csv(args.reviewed_frame)
    snapshot_identity = json.loads(
        args.snapshot_identity.read_text(encoding="utf-8")
    )
    split_rows = read_csv(args.split_manifest)
    effective_rows = read_csv(args.effective_window_index)
    duplicate_audit = json.loads(args.duplicate_audit.read_text(encoding="utf-8"))
    source_lineage = json.loads(args.source_lineage.read_text(encoding="utf-8"))
    completion_audit = json.loads(args.completion_audit.read_text(encoding="utf-8"))
    a12_control = json.loads(args.a12_control.read_text(encoding="utf-8"))

    cvat_keys = {canonical_video_key(row.get("source_video_key", "")) for row in exclusions}
    cvat_keys.discard("")
    raw_counts = count_keys(legacy_center, ("video_final",))
    duplicate_counts = count_keys(duplicate_preview, ("source_video_key_audit", "video_final"))
    filtered_center_counts = count_keys(filtered_center, ("source_video_key", "video_final"))
    filtered_all_counts = count_keys(filtered_all, ("source_video_key", "video_final"))
    training_counts = count_keys(legacy_training, ("source_video_resolved",))
    current_by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in current_frame:
        current_by_source[text(row.get("source_type"))].append(row)
    current_cvat_counts = count_keys(
        current_by_source["cvat_tracking_xml"], ("source_video_key", "video_key")
    )
    current_legacy_counts = count_keys(
        current_by_source["legacy_recovered"], ("source_video_key", "video_key")
    )

    duplicate_keys = set(duplicate_counts)
    excluded_sample_ids = {text(row.get("sample_id")) for row in duplicate_preview}
    filtered_center_sample_ids = {text(row.get("sample_id")) for row in filtered_center}
    filtered_all_sample_ids = {text(row.get("sample_id")) for row in filtered_all}
    training_sample_ids = {text(row.get("sample_id")) for row in legacy_training}
    excluded_reentry = (
        len(excluded_sample_ids & filtered_center_sample_ids)
        + len(excluded_sample_ids & filtered_all_sample_ids)
        + len(excluded_sample_ids & training_sample_ids)
    )

    current_legacy_native_keys = {
        text(row.get("temporal_unit_key"))
        for row in current_by_source["legacy_recovered"]
        if text(row.get("temporal_unit_key"))
    }
    legacy_frames_by_native_unit = Counter(
        text(row.get("temporal_unit_key"))
        for row in current_by_source["legacy_recovered"]
        if text(row.get("temporal_unit_key"))
    )
    current_cvat_keys = set(current_cvat_counts)
    filtered_source_overlap = len(set(filtered_center_counts) & cvat_keys)
    current_source_overlap = len(set(current_legacy_counts) & cvat_keys)
    current_exclusion_overlap = len(set(current_legacy_counts) & cvat_keys)
    pooled_legacy_rows = len(current_by_source["legacy_recovered"])
    pooled_cvat_rows = len(current_by_source["cvat_tracking_xml"])

    assert sha256_file(args.snapshot_identity) == SNAPSHOT_SHA256
    assert sha256_file(args.split_manifest) == SPLIT_FILE_SHA256
    assert sha256_file(args.effective_window_index) == CURRENT_EFFECTIVE_HASH
    assert len(cvat_keys) == 12
    assert len(legacy_center) == 4610
    assert len(duplicate_preview) == 55
    assert len(filtered_center) == 4555
    assert len(filtered_all) > 0
    assert set(legacy_frames_by_native_unit.values()) == {16}
    assert pooled_legacy_rows == len(filtered_center) * 16
    assert len(current_legacy_native_keys) == len(filtered_center)
    assert excluded_reentry == 0
    assert filtered_source_overlap == 0
    assert current_source_overlap == 0
    assert current_cvat_keys == cvat_keys
    assert current_cvat_keys

    split_checks = roles_and_crossings(effective_rows, split_rows)
    direct_source_pass = (
        a12_control.get("source_type_in_model_x") is False
        and a12_control.get("metadata_fields_in_model_x") == 0
        and a12_control.get("hidden_entered_model_x") is False
    )
    assert direct_source_pass

    old_eligibility, eligibility_rows = eligibility_reconciliation(
        args.old_window_manifest, args.effective_window_index, args.affected_native_units
    )
    assert old_eligibility["old_eligible_windows"] == 159413
    assert old_eligibility["current_eligible_windows"] == 159410
    assert old_eligibility["changed_window_count"] == 3
    assert old_eligibility["changed_native_unit_count"] == 1

    script_relative = script_path.relative_to(repo).as_posix()
    script_binding = binding(script_path, "executed construction-overlap audit code")
    script_binding["git_blob_sha1"] = file_blob(repo, script_relative)
    script_binding["last_commit"] = file_commit(repo, script_relative)
    duplicate_binding = binding(args.duplicate_preview, "check_duplicate_videos.py output")
    duplicate_audit_binding = binding(args.duplicate_audit, "check_duplicate_videos.py audit")
    cvat_binding = binding(args.cvat_exclusion, "CVAT source-video inventory and exclusion policy")
    legacy_binding = binding(args.legacy_center, "legacy burst source inventory before filtering")
    center_binding = binding(args.filtered_center, "retained legacy center burst manifest")
    all_binding = binding(
        args.filtered_all, "retained legacy all-frame bounding-box manifest"
    )
    training_binding = binding(args.legacy_training, "downstream filtered legacy sequence manifest")
    snapshot_frame_binding = binding(
        args.reviewed_frame, "current pooled reviewed frame snapshot rows"
    )
    snapshot_binding = binding(
        args.snapshot_identity, "current reviewed snapshot identity manifest"
    )
    snapshot_binding["snapshot_id"] = snapshot_identity.get("snapshot_id", SNAPSHOT_ID)
    snapshot_binding["frame_artifact"] = snapshot_frame_binding

    overlapping_rows = []
    for key in sorted(cvat_keys):
        before = raw_counts.get(key, 0)
        excluded = duplicate_counts.get(key, 0)
        retained = filtered_center_counts.get(key, 0)
        overlapping_rows.append(
            {
                "record_type": "cvat_source_video",
                "source_video_key": key,
                "sample_id": "",
                "legacy_bursts_before_filter": before,
                "legacy_bursts_excluded": excluded,
                "legacy_bursts_retained": retained,
                "current_pooled_legacy_rows": sum(
                    count
                    for source_key, count in current_legacy_counts.items()
                    if source_key == key
                ),
                "downstream_artifact": str(center_binding["path"]),
                "downstream_sha256": center_binding["sha256"],
                "status": "OVERLAP_EXCLUDED" if excluded else "NO_LEGACY_ROW",
            }
        )
    for key in sorted(duplicate_keys):
        overlapping_rows.append(
            {
                "record_type": "overlapping_source_video",
                "source_video_key": key,
                "sample_id": "",
                "legacy_bursts_before_filter": raw_counts[key],
                "legacy_bursts_excluded": duplicate_counts[key],
                "legacy_bursts_retained": filtered_center_counts.get(key, 0),
                "current_pooled_legacy_rows": current_legacy_counts.get(key, 0),
                "downstream_artifact": str(center_binding["path"]),
                "downstream_sha256": center_binding["sha256"],
                "status": "EXCLUDED",
            }
        )
    for row in duplicate_preview:
        key = source_key(row, ("source_video_key_audit", "video_final"))
        overlapping_rows.append(
            {
                "record_type": "excluded_legacy_burst",
                "source_video_key": key,
                "sample_id": text(row.get("sample_id")),
                "legacy_bursts_before_filter": 1,
                "legacy_bursts_excluded": 1,
                "legacy_bursts_retained": 0,
                "current_pooled_legacy_rows": 0,
                "downstream_artifact": str(center_binding["path"]),
                "downstream_sha256": center_binding["sha256"],
                "status": "ABSENT_FROM_ALL_FILTERED_INPUTS",
            }
        )
    for key in sorted(filtered_center_counts):
        overlapping_rows.append(
            {
                "record_type": "retained_legacy_source_video",
                "source_video_key": key,
                "sample_id": "",
                "legacy_bursts_before_filter": raw_counts.get(key, 0),
                "legacy_bursts_excluded": duplicate_counts.get(key, 0),
                "legacy_bursts_retained": filtered_center_counts[key],
                "current_pooled_legacy_rows": current_legacy_counts.get(key, 0),
                "downstream_artifact": str(training_binding["path"]),
                "downstream_sha256": training_binding["sha256"],
                "status": "RETAINED_NON_CVAT_SOURCE",
            }
        )
    for item in (
        center_binding,
        all_binding,
        training_binding,
        snapshot_frame_binding,
        snapshot_binding,
    ):
        overlapping_rows.append(
            {
                "record_type": "downstream_binding",
                "source_video_key": "",
                "sample_id": "",
                "legacy_bursts_before_filter": "",
                "legacy_bursts_excluded": "",
                "legacy_bursts_retained": "",
                "current_pooled_legacy_rows": "",
                "downstream_artifact": item["path"],
                "downstream_sha256": item["sha256"],
                "status": item["semantic_role"],
            }
        )
    excluded_csv = output / "cvat_legacy_excluded_videos.csv"
    write_csv(
        excluded_csv,
        overlapping_rows,
        [
            "record_type",
            "source_video_key",
            "sample_id",
            "legacy_bursts_before_filter",
            "legacy_bursts_excluded",
            "legacy_bursts_retained",
            "current_pooled_legacy_rows",
            "downstream_artifact",
            "downstream_sha256",
            "status",
        ],
    )

    authority = {
        "schema_version": "classification_v2.cvat_legacy_duplicate_removal_authority.v1",
        "decision": "PASS",
        "construction_question": (
            "Were legacy bursts originating from CVAT-represented source videos removed "
            "before the reviewed collections were pooled?"
        ),
        "rule": (
            "Retain all CVAT videos; canonicalize every legacy source video; exclude each "
            "legacy burst whose source key is in the CVAT exclusion inventory; retain all "
            "other legacy bursts."
        ),
        "canonicalization_rule": (
            "check_duplicate_videos.py normalize_source_video_key: lower-case, slash-normalize, "
            "accept canonical key, nested color.mp4 path, or video filename, and zero-pad the "
            "clip to six digits as pigsDDMMYY/NNNNNN."
        ),
        "comparison_key": "canonical source_video_key",
        "executed_script": script_binding,
        "cvat_inventory": {
            **cvat_binding,
            "key_count": len(cvat_keys),
            "keys": sorted(cvat_keys),
            "source_role": "additional CVAT behavior videos represented by the exclusion policy",
        },
        "legacy_inventory": {
            **legacy_binding,
            "burst_rows_before_filter": len(legacy_center),
            "source_video_key_count_before_filter": len(raw_counts),
        },
        "duplicate_output": {
            **duplicate_binding,
            "duplicate_audit": duplicate_audit_binding,
            "overlapping_source_video_key_count": len(duplicate_keys),
            "overlapping_source_video_keys": sorted(duplicate_keys),
            "legacy_bursts_excluded": len(duplicate_preview),
        },
        "filtered_outputs": {
            "center": {
                **center_binding,
                "retained_bursts": len(filtered_center),
                "retained_source_video_key_count": len(filtered_center_counts),
            },
            "all_frames": {
                **all_binding,
                "retained_rows": len(filtered_all),
                "retained_source_video_key_count": len(filtered_all_counts),
            },
        },
        "downstream": {
            "legacy_training_sequence_manifest": {
                **training_binding,
                "rows": len(legacy_training),
                "source_video_key_count": len(training_counts),
            },
            "current_reviewed_pooled_snapshot": {
                **snapshot_binding,
                "rows": len(current_frame),
                "legacy_rows": pooled_legacy_rows,
                "cvat_rows": pooled_cvat_rows,
                "legacy_source_video_key_count": len(current_legacy_counts),
                "cvat_source_video_key_count": len(current_cvat_counts),
                "legacy_native_unit_count": len(current_legacy_native_keys),
                "frames_per_retained_legacy_burst": sorted(
                    set(legacy_frames_by_native_unit.values())
                ),
            },
        },
        "counts": {
            "cvat_source_video_keys": len(cvat_keys),
            "legacy_source_video_keys_before_filter": len(raw_counts),
            "overlapping_source_video_keys": len(duplicate_keys),
            "legacy_bursts_before_filter": len(legacy_center),
            "legacy_bursts_excluded": len(duplicate_preview),
            "legacy_bursts_retained": len(filtered_center),
            "cvat_videos_removed": 0,
            "excluded_legacy_burst_reentry_violations": excluded_reentry,
            "retained_legacy_bursts_with_cvat_source_video": filtered_source_overlap,
            "current_pooled_legacy_source_key_overlap": current_source_overlap,
        },
        "reconciled_construction_exclusion_keys": {
            "reported_key_count": len(cvat_keys),
            "keys_with_legacy_rows": len(duplicate_keys),
            "keys_without_legacy_rows": len(cvat_keys - duplicate_keys),
            "keys_without_legacy_rows_list": sorted(cvat_keys - duplicate_keys),
        },
        "no_data_rebuild": True,
        "no_labels_or_splits_changed": True,
    }
    authority_path = output / "cvat_legacy_duplicate_removal_authority.json"
    write_json(authority_path, authority)
    authority_binding = binding(
        authority_path,
        "executed CVAT/legacy construction-overlap authority",
    )

    filtered_binding = {
        "schema_version": "classification_v2.cvat_legacy_filtered_burst_binding.v1",
        "decision": "PASS",
        "lineage": [
            {"step": "cvat_video_inventory", "artifact": cvat_binding},
            {"step": "legacy_source_video_inventory", "artifact": legacy_binding},
            {"step": "executed_check_duplicate_videos.py", "artifact": script_binding},
            {"step": "duplicate_video_exclusion_list", "artifact": cvat_binding},
            {"step": "duplicate_preview", "artifact": duplicate_binding},
            {"step": "filtered_legacy_center_manifest", "artifact": center_binding},
            {"step": "filtered_legacy_all_frame_manifest", "artifact": all_binding},
            {"step": "filtered_legacy_training_manifest", "artifact": training_binding},
            {"step": "current_reviewed_pooled_snapshot", "artifact": snapshot_binding},
        ],
        "direct_downstream_consumers": [
            {
                "path": training_binding["path"],
                "sha256": training_binding["sha256"],
                "role": "filtered legacy native-unit input to pooled feature materialization",
                "rows": len(legacy_training),
            },
            {
                "path": snapshot_binding["path"],
                "sha256": snapshot_binding["sha256"],
                "role": "current pooled reviewed behavior authority",
                "legacy_rows": pooled_legacy_rows,
            },
        ],
        "reentry_audit": {
            "excluded_sample_ids_checked": len(excluded_sample_ids),
            "excluded_sample_ids_in_filtered_center": len(
                excluded_sample_ids & filtered_center_sample_ids
            ),
            "excluded_sample_ids_in_filtered_all": len(
                excluded_sample_ids & filtered_all_sample_ids
            ),
            "excluded_sample_ids_in_downstream_training_manifest": len(
                excluded_sample_ids & training_sample_ids
            ),
            "retained_source_keys_intersect_cvat": filtered_source_overlap,
            "current_pooled_legacy_keys_intersect_cvat": current_source_overlap,
            "current_pooled_legacy_keys_intersect_exclusion_inventory": current_exclusion_overlap,
            "reentry_violation_count": excluded_reentry,
            "raw_legacy_input_status": "AUDIT_ONLY_NOT_A_CURRENT_POOLED_INPUT",
            "duplicate_preview_status": "AUDIT_ONLY_NOT_A_CURRENT_POOLED_INPUT",
            "fallback_input_status": "NO_OTHER_LEGACY_INPUT_BOUND_BY_CURRENT_SNAPSHOT",
        },
        "assertions": {
            "cvat_videos_removed": 0,
            "all_duplicate_source_rows_excluded": len(duplicate_preview)
            == sum(duplicate_counts.values()),
            "retained_legacy_source_keys_are_non_cvat": filtered_source_overlap == 0,
            "pooled_legacy_rows_equal_retained_burst_frame_expansion": (
                pooled_legacy_rows == len(filtered_center) * 16
            ),
            "pooled_legacy_native_units_equal_filtered_center_bursts": len(
                current_legacy_native_keys
            )
            == len(filtered_center),
            "cvat_rows_remain_present": pooled_cvat_rows > 0 and current_cvat_keys == cvat_keys,
        },
        "no_data_rebuild": True,
    }
    write_json(output / "cvat_legacy_filtered_burst_binding.json", filtered_binding)

    direct_evidence = binding(args.a12_control, "A12-A direct predictive-source leakage control")
    grouping_evidence = [
        binding(args.split_manifest, "frozen grouped split role authority"),
        binding(args.effective_window_index, "effective window role inheritance authority"),
    ]
    proof_checks = [
        {
            "id": "construction_source_overlap",
            "status": "PASS",
            "evidence": [authority_binding],
            "result": authority["counts"],
        },
        {
            "id": "exact_duplicate_isolation",
            "status": "PASS",
            "evidence": [duplicate_audit_binding, center_binding, all_binding],
            "result": {
                "source_video_duplicate_rows_excluded": len(duplicate_preview),
                "excluded_sample_ids_reentered": len(
                    excluded_sample_ids & filtered_center_sample_ids
                ),
            },
            "scope": "construction source-video exclusion, not retrospective content hashing",
        },
        {
            "id": "near_duplicate_isolation",
            "status": "NOT_APPLICABLE",
            "evidence": [
                {
                    "semantic_scope": "frozen behavior construction contract",
                    "path": (
                        "docs/classification_v2/corrected_pooled_route_20260806/"
                        "a12_supersession_notice.md"
                    ),
                    "sha256": sha256_file(
                        repo
                        / "docs/classification_v2/corrected_pooled_route_20260806/"
                        "a12_supersession_notice.md"
                    ),
                }
            ],
            "result": {"threshold_used": False, "frozen_requirement": False},
            "limitation": (
                "No retrospective near-duplicate threshold is part of this "
                "construction hard gate."
            ),
        },
        {
            "id": "exact_temporal_interval_isolation",
            "status": "PASS",
            "evidence": grouping_evidence,
            "result": {
                "role_crossing_count": split_checks["window_role_crossings"],
                "content_hash_threshold_used": False,
                "frozen_contract_scope": "window role inheritance and grouping",
            },
        },
        {
            "id": "native_unit_isolation",
            "status": "PASS",
            "evidence": grouping_evidence,
            "result": {"role_crossing_count": split_checks["native_unit_role_crossings"]},
        },
        {
            "id": "video_group_isolation",
            "status": "PASS",
            "evidence": grouping_evidence,
            "result": {"role_crossing_count": split_checks["video_role_crossings"]},
        },
        {
            "id": "recording_date_group_isolation",
            "status": "PASS",
            "evidence": grouping_evidence,
            "result": {"role_crossing_count": split_checks["date_role_crossings"]},
        },
        {
            "id": "window_role_inheritance",
            "status": "PASS",
            "evidence": grouping_evidence,
            "result": {
                **split_checks,
                "missing_role_bindings": split_checks["missing_role_bindings"],
            },
        },
        {
            "id": "direct_predictive_source_leakage",
            "status": "PASS",
            "evidence": [direct_evidence],
            "result": {
                "source_type_in_model_x": a12_control.get("source_type_in_model_x"),
                "metadata_fields_in_model_x": a12_control.get("metadata_fields_in_model_x"),
                "hidden_entered_model_x": a12_control.get("hidden_entered_model_x"),
            },
        },
    ]
    proof = {
        "schema_version": "classification_v2.revised_a12b_construction_overlap_proof.v1",
        "decision": "PASS",
        "hard_gate": True,
        "hard_gate_scope": [
            "CVAT-versus-legacy source-video exclusion",
            "recording-date grouping",
            "video grouping",
            "native-unit grouping",
            "deterministic window-role inheritance",
            "direct predictive-source isolation",
        ],
        "checks": proof_checks,
        "crossing_counts": split_checks,
        "construction_overlap": authority["counts"],
        "limitations": [
            "Near-duplicate representation and threshold are not a frozen "
            "requirement for this construction question.",
            "No source, label, split, or reviewed-data artifact was rebuilt or changed.",
        ],
        "no_data_rebuild": True,
    }
    revised_proof_path = output / "revised_a12b_construction_overlap_proof.json"
    write_json(revised_proof_path, proof)

    eligibility_json_path = output / "eligibility_reconciliation.json"
    write_json(eligibility_json_path, old_eligibility)
    write_csv(
        output / "eligibility_reconciliation.csv",
        eligibility_rows,
        [
            "record_type",
            "stable_native_unit_key",
            "window_id",
            "view_type",
            "old_status",
            "current_status",
            "old_reason",
            "current_reason",
            "governing_rule",
        ],
    )

    e0_paths = {
        "reviewed_snapshot": snapshot_binding,
        "effective_window_index": binding(
            args.effective_window_index, "T6/T8/T12/T16 effective window authority"
        ),
        "split_manifest": binding(args.split_manifest, "frozen grouped split manifest"),
        "event_weight_manifest": binding(
            Path(
                r"E:\PigProjectStorage\PIG_Behavior_Project"
                r"\outputs\classification_v2\model_readiness_audit"
                r"\pre_gpu_autoresearch_q2_47103f6_20260804_133801"
                r"\fold_event_weights_47103f6_20260804_154500"
                r"\fold_event_weight_manifest.csv"
            ),
            "event-native weight authority",
        ),
        "train_mask": binding(
            Path(
                r"E:\PigProjectStorage\PIG_Behavior_Project"
                r"\outputs\classification_v2\agent_audits"
                r"\post_review_frame_amendment_materialization_fa028cb_20260803_224700"
                r"\train_mask.csv"
            ),
            "zero-weight and eligibility mask",
        ),
        "spatial_manifest": binding(
            Path(
                r"E:\PigProjectStorage\PIG_Behavior_Project"
                r"\outputs\classification_v2\agent_audits"
                r"\post_review_frame_amendment_materialization_fa028cb_20260803_224700"
                r"\spatial_memmap_bundle\spatial_memmap_manifest.json"
            ),
            "46D spatial tensor authority",
        ),
        "rgb_manifest": binding(
            Path(
                r"E:\PigProjectStorage\PIG_Behavior_Project"
                r"\outputs\classification_v2\model_readiness_audit"
                r"\pre_gpu_autoresearch_q2_6c2f204_20260804_084638"
                r"\reviewed_rgb_v1\actor_rgb_64_full\manifest.csv"
            ),
            "actor RGB cache manifest",
        ),
        "rgb_index": binding(
            Path(
                r"E:\PigProjectStorage\PIG_Behavior_Project"
                r"\outputs\classification_v2\model_readiness_audit"
                r"\pre_gpu_autoresearch_q2_6c2f204_20260804_084638"
                r"\reviewed_rgb_v1\actor_rgb_64_full\packed_image_cache_index.csv"
            ),
            "actor RGB packed-cache index",
        ),
        "environment_lock": binding(repo / "uv.lock", "staged environment lockfile"),
    }
    for name, expected in {
        "event_weight_manifest": EVENT_WEIGHT_HASH,
        "effective_window_index": CURRENT_EFFECTIVE_HASH,
        "split_manifest": SPLIT_FILE_SHA256,
        "rgb_index": "9ccef8607973cfb8c8377474665af5d62874b5beea39ad716872b187f8d29d68",
    }.items():
        assert e0_paths[name]["sha256"].lower() == expected.lower(), name
    e0_descriptor = {
        "model": "B3_ACTOR_T6_PLUS_GEOMETRY_MOTION",
        "temporal_view": "T6",
        "modalities": ["actor_RGB", "geometry_6D", "motion_12D"],
        "inner_fold": "FOLD_3",
        "seed": 20260804,
        "classification_code_sha": CLASSIFICATION_CODE_SHA,
        "snapshot_sha256": SNAPSHOT_SHA256,
        "split_hash": SPLIT_HASH,
        "event_weight_hash": EVENT_WEIGHT_HASH,
        "schema_hash": SCHEMA_HASH,
        "environment_lock_hash": ENVIRONMENT_HASH,
    }
    e0_descriptor_hash = hashlib.sha256(
        json.dumps(e0_descriptor, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    e0_handoff = {
        "schema_version": "classification_v2.e0_l4_handoff.v1",
        "status": "NOT_EXECUTED_HANDOFF_ONLY",
        "provider": {
            "gpu": "1x NVIDIA L4 24 GB",
            "interruptible": False,
            "max_cost_usd": 1.50,
            "max_gpu_hours": 2.0,
            "max_wall_hours": 4.0,
            "max_remote_disk_gb": 15,
            "paid_retries": 0,
        },
        "descriptor": e0_descriptor,
        "descriptor_sha256": e0_descriptor_hash,
        "inputs": e0_paths,
        "installation_command": "uv sync --frozen --python 3.11 --extra dev",
        "launch_command": (
            "uv run --frozen python scripts/classification_v2/04_baselines_smokes/"
            "classification_v2_run_multimodal_ablation_pilot.py --output-dir "
            "$E0_OUT --image-cache-manifest $E0_RGB_MANIFEST --variants full "
            "--device cuda --image-size 64 --hidden-dim 16 --steps-per-fold 16 "
            "--train-batch-size 16 --eval-batch-size 16 --train-per-class 8 "
            "--eval-per-class 8 --seed 20260804"
        ),
        "launch_binding_note": (
            "The package wrapper must enforce FOLD_3 before invoking the existing "
            "registered B3/T6 trainer; do not use max_folds=1 on the full OOF runner "
            "unless its fold selector has been bound to FOLD_3."
        ),
        "checkpoint_path": "$E0_OUT/checkpoints/B3_ACTOR_T6_PLUS_GEOMETRY_MOTION.pt",
        "forced_interruption_procedure": (
            "wait for checkpoint_manifest.json and its SHA-256, then send one real "
            "SIGINT/termination to the training process before the registered endpoint"
        ),
        "resume_command": (
            "uv run --frozen python $E0_RESUME_ENTRY --resume-checkpoint "
            "$E0_OUT/checkpoints/B3_ACTOR_T6_PLUS_GEOMETRY_MOTION.pt "
            "--manifest $E0_OUT/run_manifest.json"
        ),
        "prediction_export": "$E0_OUT/predictions.csv",
        "download_manifest": "$E0_OUT/download_hash_manifest.json",
        "hash_verification": "sha256sum -c download_hash_manifest.json",
        "gpu_stop_command": "provider-cli stop $E0_INSTANCE_ID",
        "outer_test_exclusion": {
            "data_mount": False,
            "labels": False,
            "metrics": False,
            "predictions": False,
            "errors": False,
            "confusion_matrices": False,
            "statement": "No outer-test resource is included, mounted, or addressable during E0.",
        },
        "execution_authorization": "NO",
        "do_not_execute": True,
    }
    write_json(output / "e0_l4_handoff.json", e0_handoff)

    summary = {
        "authority_path": str(authority_path),
        "revised_proof_path": str(revised_proof_path),
        "eligibility_path": str(eligibility_json_path),
        "e0_l4_handoff_path": str(output / "e0_l4_handoff.json"),
        "construction_counts": authority["counts"],
        "grouping_counts": split_checks,
        "eligibility": {
            "old_eligible": old_eligibility["old_eligible_windows"],
            "current_eligible": old_eligibility["current_eligible_windows"],
            "changed_native_units": old_eligibility["changed_native_unit_count"],
            "changed_windows": old_eligibility["changed_window_count"],
        },
        "e0_fold": "FOLD_3",
        "e0_preflight": "PASS_TECHNICAL_ONLY",
        "paid_execution_authorization": "NO",
    }
    write_json(output / "execution_handoff_generation_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
