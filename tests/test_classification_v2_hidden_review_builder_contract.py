from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd
import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "classification_v2"
    / "01_review_units_gui"
    / "classification_v2_build_hidden_review_units.py"
)


def _load_builder() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "hidden_review_builder_contract_test",
        SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _complete_units(
    *,
    cvat_units: int = 4,
    legacy_units: int = 4,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    source_specs = (
        ("cvat_tracking_xml", cvat_units, 6),
        ("legacy_recovered", legacy_units, 16),
    )
    for source, unit_count, unit_length in source_specs:
        for unit_index in range(unit_count):
            dataset = f"{source}_dataset"
            video = f"{source}_video_{unit_index // 2}"
            actor = f"{source}_actor_{unit_index}"
            unit_key = f"{source}|{video}|{actor}|unit={unit_index}"
            start = unit_index * 1000
            for offset in range(unit_length):
                frame_index = start + offset
                rows.append(
                    {
                        "source_type": source,
                        "dataset_id": dataset,
                        "video_key": video,
                        "frame_uid": f"{source}-{unit_index}-{offset}",
                        "frame_index": frame_index,
                        "pig_id": f"ID_{unit_index}",
                        "track_id": str(unit_index),
                        "object_track_key": actor,
                        "temporal_unit_key": unit_key,
                        "behavior": "stand",
                        "hidden": "Yes" if offset == 0 else "No",
                        "bbox_valid": True,
                        "bbox_was_clipped": offset % 3 == 0,
                        "x1": 10.0,
                        "y1": 10.0,
                        "x2": 110.0,
                        "y2": 70.0,
                        "nearest_pair_iou": 0.4 if offset % 2 else 0.0,
                        "nearest_pair_overlap_ratio": 0.5 if offset % 2 else 0.0,
                        "nearest_dist_n": 0.02 if offset % 2 else 0.5,
                        "pair_contact_with_nearest": offset % 2 == 1,
                        "shape_change_score": 0.5 if offset % 2 else 0.0,
                        "delta_area_n": 0.3 if offset % 2 else 0.0,
                    }
                )
    return pd.DataFrame(rows)


def _write_policy(path: Path, *, minimum: int = 1) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "classification_v2.hidden_scientific_policy.v1",
                "confidence_level": 0.95,
                "bootstrap_iterations": 200,
                "bootstrap_seed": 20260714,
                "random_false_negative_upper_threshold": 0.05,
                "high_risk_yield_upper_threshold": 0.10,
                "min_random_reviewed_items": minimum,
                "min_random_native_clusters": minimum,
                "min_random_recording_clusters": minimum,
                "min_high_risk_reviewed_items": minimum,
                "min_high_risk_native_clusters": minimum,
                "min_high_risk_recording_clusters": minimum,
            }
        ),
        encoding="utf-8",
    )


def _run_builder(
    module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    input_csv: Path,
    output_dir: Path,
    policy_json: Path,
    *,
    design_scope: str,
    extra: list[str] | None = None,
) -> None:
    argv = [
        str(SCRIPT_PATH),
        "--input-csv",
        str(input_csv),
        "--output-dir",
        str(output_dir),
        "--design-scope",
        design_scope,
        "--scientific-policy-json",
        str(policy_json),
        "--random-no-per-stratum",
        "1",
        "--max-high-risk-per-stratum",
        "4",
    ]
    if extra:
        argv.extend(extra)
    monkeypatch.setattr(sys, "argv", argv)
    module.main()


def _canonical_paths(module: ModuleType, output_dir: Path) -> list[Path]:
    return [output_dir / name for name in module._canonical_output_names()]


def test_cli_requires_explicit_design_scope_and_help_is_unambiguous(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_builder()
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT_PATH), "--input-csv", "input.csv", "--output-dir", "out"],
    )
    with pytest.raises(SystemExit) as missing:
        module.parse_args()
    assert missing.value.code == 2

    monkeypatch.setattr(sys, "argv", [str(SCRIPT_PATH), "--help"])
    with pytest.raises(SystemExit) as help_exit:
        module.parse_args()
    assert help_exit.value.code == 0
    help_text = capsys.readouterr().out
    normalized_help = " ".join(help_text.split())
    assert "--design-scope {smoke,full}" in help_text
    assert "smoke" in help_text and "final-support" in help_text
    assert "never changes --design-scope" in normalized_help


def test_row_caps_do_not_select_scope_and_bounded_full_is_rejected() -> None:
    module = _load_builder()
    smoke = module.argparse.Namespace(
        design_scope="smoke",
        max_rows=6,
        max_rows_per_source=None,
    )
    module._validate_args(smoke)
    assert module._input_bounding_mode(smoke) == "max_rows"
    assert smoke.design_scope == "smoke"

    full = module.argparse.Namespace(
        design_scope="full",
        max_rows=6,
        max_rows_per_source=None,
    )
    with pytest.raises(ValueError, match="full cannot be combined"):
        module._validate_args(full)
    assert full.design_scope == "full"


def test_complete_unit_smoke_keeps_704_rows_and_64_units(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_builder()
    frames = _complete_units(cvat_units=32, legacy_units=32)
    assert len(frames) == 704
    input_csv = tmp_path / "complete_units.csv"
    policy_json = tmp_path / "policy.json"
    output_dir = tmp_path / "smoke"
    frames.to_csv(input_csv, index=False)
    _write_policy(policy_json, minimum=1000)

    _run_builder(
        module,
        monkeypatch,
        input_csv,
        output_dir,
        policy_json,
        design_scope="smoke",
    )

    audit = json.loads(
        (output_dir / "hidden_review_template_audit.json").read_text()
    )
    assert audit["input_rows_before_bounding"] == 704
    assert audit["input_rows_after_bounding"] == 704
    assert audit["input_was_bounded"] is False
    assert audit["input_bounding_mode"] == "none"
    assert audit["design_scope"] == "smoke"
    assert audit["require_final_support"] is False
    assert audit["structural_checks_pass"] is True
    assert audit["structural_audit"]["temporal_unit_count"] == 64
    assert audit["final_support_checks_required"] is False
    assert audit["final_support_checks_pass"] is False
    assert audit["outputs_published"] is True


def test_full_mode_enforces_support_and_failure_is_nonpublishing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_builder()
    input_csv = tmp_path / "complete_units.csv"
    policy_json = tmp_path / "policy.json"
    output_dir = tmp_path / "full_short"
    _complete_units(cvat_units=2, legacy_units=2).to_csv(input_csv, index=False)
    _write_policy(policy_json, minimum=100)

    with pytest.raises(ValueError, match="insufficient planned support"):
        _run_builder(
            module,
            monkeypatch,
            input_csv,
            output_dir,
            policy_json,
            design_scope="full",
        )

    assert not any(path.exists() for path in _canonical_paths(module, output_dir))
    failure = json.loads(
        (output_dir / module.FAILURE_AUDIT_FILENAME).read_text()
    )
    assert failure["design_scope"] == "full"
    assert failure["require_final_support"] is True
    assert failure["final_support_checks_required"] is True
    assert failure["final_support_checks_pass"] is False
    assert failure["final_support_policy_version"] == (
        "classification_v2.hidden_scientific_policy.v1"
    )
    assert failure["no_outputs_published"] is True
    assert failure["output_transaction_status"] == "aborted"


def test_smoke_still_fails_structural_corruption_without_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_builder()
    frames = _complete_units(cvat_units=2, legacy_units=2)
    frames = frames.iloc[:-1].copy()
    input_csv = tmp_path / "corrupt.csv"
    policy_json = tmp_path / "policy.json"
    output_dir = tmp_path / "corrupt_out"
    frames.to_csv(input_csv, index=False)
    _write_policy(policy_json)

    with pytest.raises(ValueError, match="structural audit failed"):
        _run_builder(
            module,
            monkeypatch,
            input_csv,
            output_dir,
            policy_json,
            design_scope="smoke",
        )

    assert not any(path.exists() for path in _canonical_paths(module, output_dir))
    failure = json.loads(
        (output_dir / module.FAILURE_AUDIT_FILENAME).read_text()
    )
    assert failure["structural_checks_pass"] is False
    assert failure["no_outputs_published"] is True


def test_full_success_publishes_complete_set_and_matching_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_builder()
    input_csv = tmp_path / "complete_units.csv"
    policy_json = tmp_path / "policy.json"
    output_dir = tmp_path / "full"
    _complete_units().to_csv(input_csv, index=False)
    _write_policy(policy_json)

    _run_builder(
        module,
        monkeypatch,
        input_csv,
        output_dir,
        policy_json,
        design_scope="full",
    )

    canonical = _canonical_paths(module, output_dir)
    assert all(path.exists() for path in canonical)
    audit = json.loads(
        (output_dir / "hidden_review_template_audit.json").read_text()
    )
    assert audit["design_scope"] == "full"
    assert audit["require_final_support"] is True
    assert audit["final_support_checks_required"] is True
    assert audit["final_support_checks_pass"] is True
    assert audit["output_transaction_status"] == "committed"
    assert audit["outputs_published"] is True
    for name, expected_hash in audit["published_file_hashes"].items():
        assert module.sha256_file(output_dir / name) == expected_hash


def test_existing_authority_is_not_overwritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_builder()
    input_csv = tmp_path / "complete_units.csv"
    policy_json = tmp_path / "policy.json"
    output_dir = tmp_path / "authority"
    output_dir.mkdir()
    authority = output_dir / "hidden_review_unit_manifest.csv"
    authority.write_text("authority\n", encoding="utf-8")
    _complete_units().to_csv(input_csv, index=False)
    _write_policy(policy_json)

    with pytest.raises(FileExistsError, match="Output already exists"):
        _run_builder(
            module,
            monkeypatch,
            input_csv,
            output_dir,
            policy_json,
            design_scope="full",
        )

    assert authority.read_text(encoding="utf-8") == "authority\n"
    assert not (output_dir / module.FAILURE_AUDIT_FILENAME).exists()


def test_publish_exception_rolls_back_all_canonical_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_builder()
    input_csv = tmp_path / "complete_units.csv"
    policy_json = tmp_path / "policy.json"
    output_dir = tmp_path / "transaction_failure"
    _complete_units().to_csv(input_csv, index=False)
    _write_policy(policy_json)
    real_replace = module._replace_for_commit
    calls = 0

    def fail_second_replace(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected publish failure")
        real_replace(source, destination)

    monkeypatch.setattr(module, "_replace_for_commit", fail_second_replace)
    with pytest.raises(OSError, match="injected publish failure"):
        _run_builder(
            module,
            monkeypatch,
            input_csv,
            output_dir,
            policy_json,
            design_scope="full",
        )

    assert not any(path.exists() for path in _canonical_paths(module, output_dir))
    failure = json.loads(
        (output_dir / module.FAILURE_AUDIT_FILENAME).read_text()
    )
    assert failure["no_outputs_published"] is True
    assert failure["outputs_published"] is False


def test_repeat_build_is_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_builder()
    input_csv = tmp_path / "complete_units.csv"
    policy_json = tmp_path / "policy.json"
    first = tmp_path / "first"
    second = tmp_path / "second"
    _complete_units().to_csv(input_csv, index=False)
    _write_policy(policy_json)

    for output_dir in (first, second):
        _run_builder(
            module,
            monkeypatch,
            input_csv,
            output_dir,
            policy_json,
            design_scope="full",
        )

    deterministic_names = [
        name
        for name in module._canonical_output_names()
        if name != "hidden_review_template_audit.json"
    ]
    for name in deterministic_names:
        assert (first / name).read_bytes() == (second / name).read_bytes()
