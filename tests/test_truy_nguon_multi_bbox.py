from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

from legacy_burst_recovery import truy_nguon_multi_bbox as trace


def _paths(tmp_path: Path) -> dict[str, Path]:
    output = tmp_path / "outputs"
    return {
        "center": output / "center.csv",
        "all": output / "all.csv",
        "audit": output / "audit.csv",
        "missing": output / "missing.csv",
        "lineage": output / "lineage.json",
    }


def _fixture(
    tmp_path: Path,
    *,
    include_manifest: bool = True,
    include_candidate: bool = False,
    behavior_video: str | None = None,
) -> tuple[Path, Path, str, str]:
    source = tmp_path / "source"
    source.mkdir()
    video = "/trace/pigs010101/PIGS010101/000001/color.mp4"
    group_id = f"burst_color_{trace.md5_code(video)}_400"
    frame_indices = [0, 3, 6, 9, 12, 15]
    behavior_rows = []
    for order, frame_index in enumerate(frame_indices):
        row = {
            "img_name": (
                f"burst_color_{trace.md5_code(video)}_400_"
                f"f{frame_index}_k{order}.jpg"
            ),
            "group_id": group_id,
            "order": order,
            "frame": order,
            "pig_id": "ID_1",
            "x1": 1,
            "y1": 2,
            "x2": 20,
            "y2": 30,
            "behavior": "stand",
            "hidden": "No",
            "hidden_attribute_present": True,
            "behavior_before_authority": "stand",
            "behavior_authority_task_frame": 0,
            "behavior_authority_slot": 0,
            "behavior_disagrees_with_authority": False,
            "behavior_authority_policy": "first_task_frame_per_group_pig",
            "is_burst_first_task_frame": order == 0,
        }
        if behavior_video is not None:
            row["video"] = behavior_video
        behavior_rows.append(row)
    behavior_csv = tmp_path / "behavior.csv"
    pd.DataFrame(behavior_rows).to_csv(behavior_csv, index=False)

    if include_manifest:
        manifest_rows = []
        frames = "|".join(str(value) for value in frame_indices)
        for row in behavior_rows:
            manifest_rows.append(
                {
                    "img_path": f"/images/{row['img_name']}",
                    "video": video,
                    "day": "pigs010101",
                    "center_frame": 0,
                    "center_ts": 0.4,
                    "group_id": group_id,
                    "frames": frames,
                }
            )
        pd.DataFrame(manifest_rows).to_csv(
            source / "manifest_frame_attribute.csv", index=False
        )

    if include_candidate:
        pd.DataFrame(
            [
                {
                    "group_id": "pigs010101/color/400",
                    "video": video,
                    "day": "pigs010101",
                    "center_frame": 0,
                    "center_ts": 0.4,
                }
            ]
        ).to_csv(source / "bursts_candidates.csv", index=False)
    return behavior_csv, source, group_id, video


def _run_main(monkeypatch: pytest.MonkeyPatch, args: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["truy_nguon_multi_bbox.py", *args])
    trace.main()


def _common_args(behavior: Path, source: Path, paths: dict[str, Path]) -> list[str]:
    return [
        "--behavior-csv",
        str(behavior),
        "--search-roots",
        str(source),
        "--out-center-csv",
        str(paths["center"]),
        "--out-all-bbox-csv",
        str(paths["all"]),
        "--out-audit-csv",
        str(paths["audit"]),
        "--out-missing-csv",
        str(paths["missing"]),
        "--out-lineage-json",
        str(paths["lineage"]),
    ]


def test_clean_six_anchor_lineage_preserves_rows_and_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    behavior, source, group_id, video = _fixture(tmp_path)
    paths = _paths(tmp_path)
    _run_main(monkeypatch, _common_args(behavior, source, paths))

    output = pd.read_csv(paths["all"])
    assert len(output) == 6
    assert not output.duplicated(["group_id", "pig_id", "legacy_order"]).any()
    assert output["video_final"].eq(video).all()
    assert output["group_id"].eq(group_id).all()
    assert len(pd.read_csv(paths["center"])) == 1
    assert paths["lineage"].exists()


def test_manifest_conflict_fails_closed(tmp_path: Path) -> None:
    behavior, source, _, video = _fixture(tmp_path)
    conflict = pd.read_csv(source / "manifest_frame_attribute.csv")
    conflict.loc[0, "video"] = video + ".conflict"
    conflict.to_csv(source / "manifest_conflict.csv", index=False)
    with pytest.raises(ValueError, match="conflicting_manifest_authority_keys"):
        trace.load_manifest_sources([source])


def test_manifest_source_group_key_can_differ_from_canonical_burst_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    behavior, source, _, _ = _fixture(tmp_path)
    manifest_path = source / "manifest_frame_attribute.csv"
    manifest = pd.read_csv(manifest_path)
    manifest["group_id"] = "pigs010101/color/400"
    manifest.to_csv(manifest_path, index=False)
    paths = _paths(tmp_path)

    _run_main(monkeypatch, _common_args(behavior, source, paths))

    output = pd.read_csv(paths["all"])
    assert len(output) == 6
    assert output["group_id"].str.endswith("_400").all()


def test_candidate_conflict_fails_closed(tmp_path: Path) -> None:
    _, source, group_id, video = _fixture(
        tmp_path, include_manifest=False, include_candidate=True
    )
    candidates = pd.read_csv(source / "bursts_candidates.csv")
    candidates.loc[0, "center_frame"] = 99
    candidates.to_csv(source / "bursts_candidates_conflict.csv", index=False)
    with pytest.raises(ValueError, match="conflicting_candidate_authority_keys"):
        trace.load_candidate_sources([source], {group_id})


def test_stale_csv_without_first_frame_authority_evidence_fails(
    tmp_path: Path,
) -> None:
    behavior, _, _, _ = _fixture(tmp_path)
    frame = pd.read_csv(behavior)
    frame = frame.drop(columns=list(trace.BEHAVIOR_AUTHORITY_COLUMNS))

    contract = trace.validate_behavior_contract(
        trace.parse_behavior_group(trace.parse_img_name_fields(frame))
    )

    assert any(
        issue.startswith("missing_behavior_authority_columns")
        for issue in contract["issues"]
    )


def test_duplicate_actor_slot_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    behavior, source, _, _ = _fixture(tmp_path)
    frame = pd.read_csv(behavior)
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    frame.to_csv(behavior, index=False)
    paths = _paths(tmp_path)
    with pytest.raises(ValueError, match="duplicate_actor_slot"):
        _run_main(monkeypatch, _common_args(behavior, source, paths))


def test_behavior_video_cannot_override_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    behavior, source, _, video = _fixture(
        tmp_path, behavior_video="/wrong/behavior/video.mp4"
    )
    paths = _paths(tmp_path)
    _run_main(monkeypatch, _common_args(behavior, source, paths))
    output = pd.read_csv(paths["all"])
    assert output["video_final"].eq(video).all()


def test_candidate_is_explicit_fallback_when_manifest_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    behavior, source, _, video = _fixture(
        tmp_path, include_manifest=False, include_candidate=True
    )
    paths = _paths(tmp_path)
    _run_main(monkeypatch, _common_args(behavior, source, paths))
    output = pd.read_csv(paths["all"])
    assert len(output) == 6
    assert output["match_source"].eq("candidate").all()
    assert output["video_final"].eq(video).all()


def test_dry_run_validates_without_writing_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    behavior, source, _, _ = _fixture(tmp_path)
    paths = _paths(tmp_path)
    args = _common_args(behavior, source, paths) + ["--dry-run"]
    _run_main(monkeypatch, args)
    assert not any(path.exists() for path in paths.values())


def test_unresolved_video_is_not_written_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    behavior, source, _, _ = _fixture(tmp_path, include_manifest=False)
    paths = _paths(tmp_path)
    with pytest.raises(ValueError, match="unresolved_video_groups"):
        _run_main(monkeypatch, _common_args(behavior, source, paths))


def test_output_under_code_directory_is_rejected(tmp_path: Path) -> None:
    behavior, source, _, _ = _fixture(tmp_path)
    paths = _paths(tmp_path)
    paths["all"] = Path("src") / "forbidden.csv"
    args = trace.parse_args(_common_args(behavior, source, paths))
    with pytest.raises(ValueError, match="cannot be written under a code directory"):
        trace.validate_output_paths(args)


def test_default_search_roots_resolve_drive_shortcut_target(
    tmp_path: Path,
) -> None:
    drive_root = tmp_path / "My Drive"
    drive_root.mkdir()
    for index in (1, 2, 3, 5):
        (drive_root / f"pig-selected_frame_attribute_({index})").mkdir()
    shortcut_target = (
        tmp_path
        / ".shortcut-targets-by-id"
        / "source-four-id"
        / "pig-selected_frame_attribute_(4)"
    )
    shortcut_target.mkdir(parents=True)

    roots, records = trace.resolve_default_search_roots(drive_root)

    assert len(roots) == 5
    assert roots[3] == shortcut_target.resolve()
    assert records[3]["resolution"] == "shortcut_target"


def test_ambiguous_drive_shortcut_target_fails_closed(tmp_path: Path) -> None:
    drive_root = tmp_path / "My Drive"
    drive_root.mkdir()
    for index in (1, 2, 3, 5):
        (drive_root / f"pig-selected_frame_attribute_({index})").mkdir()
    for target_id in ("first", "second"):
        (
            tmp_path
            / ".shortcut-targets-by-id"
            / target_id
            / "pig-selected_frame_attribute_(4)"
        ).mkdir(parents=True)

    with pytest.raises(ValueError, match="ambiguous Drive shortcut target"):
        trace.resolve_default_search_roots(drive_root)


def test_video_existence_uses_local_path_but_hash_uses_canonical(
    tmp_path: Path,
) -> None:
    drive_root = tmp_path / "My Drive"
    local_video = (
        drive_root
        / "pig_data_unzipped"
        / "pigs010101"
        / "PIGS010101"
        / "000001"
        / "color.mp4"
    )
    local_video.parent.mkdir(parents=True)
    local_video.touch()
    canonical = (
        "/content/drive/MyDrive/pig_data_unzipped/"
        "pigs010101/PIGS010101/000001/color.mp4"
    )
    group_id = f"burst_color_{trace.md5_code(canonical)}_400"
    combined = pd.DataFrame(
        {
            "group_id": [group_id],
            "video_final": [canonical],
            "video_local_path": [
                trace.resolve_video_local_path(
                    canonical,
                    drive_root=drive_root,
                )
            ],
        }
    )

    audit = trace.validate_resolved_video_contract(
        combined,
        allow_unresolved=False,
        require_exists=True,
    )

    assert audit["resolved_rows"] == 1
    assert Path(combined.loc[0, "video_local_path"]) == local_video


def test_day_must_match_canonical_video_path() -> None:
    combined = pd.DataFrame(
        {
            "group_id": ["burst_color_deadbeef_400"],
            "day_final": ["pigs020202"],
            "video_final": [
                "/content/drive/MyDrive/pig_data_unzipped/"
                "pigs010101/PIGS010101/000001/color.mp4"
            ],
        }
    )

    with pytest.raises(ValueError, match="day_video_path_mismatch"):
        trace.validate_day_contract(combined)


def test_manifest_candidate_day_mismatch_fails_closed() -> None:
    combined = pd.DataFrame(
        {
            "group_id": ["burst_color_deadbeef_400"],
            "img_name": ["burst_color_deadbeef_400_f0_k0.jpg"],
            "manifest__video": ["/same/video.mp4"],
            "candidate__video": ["/same/video.mp4"],
            "manifest__day": ["pigs010101"],
            "candidate__day": ["pigs020202"],
        }
    )

    with pytest.raises(ValueError, match="manifest_candidate_day_mismatch"):
        trace.validate_manifest_candidate_agreement(combined)


def test_existing_outputs_require_explicit_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    behavior, source, _, _ = _fixture(tmp_path)
    paths = _paths(tmp_path)
    args = _common_args(behavior, source, paths)
    _run_main(monkeypatch, args)
    with pytest.raises(FileExistsError, match="output already exists"):
        _run_main(monkeypatch, args)
