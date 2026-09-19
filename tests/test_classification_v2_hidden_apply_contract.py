from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts/classification_v2/01_review_units_gui"
APPLY_SCRIPT = SCRIPT_DIR / "classification_v2_apply_hidden_review_decisions.py"
CHECK_SCRIPT = SCRIPT_DIR / "check_apply_hidden_review_decisions_output.py"
V6_ROOT = (
    ROOT
    / "human_review_workspace/classification_v2/"
    "c2v2_human_review_20260722_reviewer01_v6"
)
V6_FRAME_LOCAL = V6_ROOT / "data/02_frame_features/frame_local_primitives.csv"
V6_MANIFEST = V6_ROOT / "data/03_hidden_review/hidden_review_unit_manifest.csv"
V6_DECISIONS = (
    V6_ROOT / "human_decisions/hidden/hidden_review_decisions.csv"
)
HIDDEN_EXTERNAL_REASON = (
    "OPTIONAL_EXTERNAL_HIDDEN_V6_FIXTURE_UNAVAILABLE:"
    "supply the versioned v6 human-review bundle"
)


def _all_files_readable(paths: tuple[Path, ...]) -> bool:
    try:
        for path in paths:
            with path.open("rb") as handle:
                handle.read(1)
    except OSError:
        return False
    return True


def _load_apply_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("hidden_apply_script", APPLY_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_apply_bundle_failure_promotes_nothing(
    tmp_path: Path,
) -> None:
    module = _load_apply_script()
    outputs = [
        tmp_path / "reviewed.csv",
        tmp_path / "apply.json",
        tmp_path / "confusion.json",
    ]

    with pytest.raises(RuntimeError, match="contains errors"):
        module._publish_output_transaction(
            pd.DataFrame({"row": [1]}),
            {"errors": ["injected"]},
            {"errors": []},
            outputs,
            overwrite=False,
        )

    assert not any(path.exists() for path in outputs)
    assert not list(tmp_path.glob(".*.tmp"))
    assert not list(tmp_path.glob(".*.backup"))


def test_apply_bundle_publish_failure_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_apply_script()
    outputs = [
        tmp_path / "reviewed.csv",
        tmp_path / "apply.json",
        tmp_path / "confusion.json",
    ]
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
        module._publish_output_transaction(
            pd.DataFrame({"row": [1]}),
            {"errors": []},
            {"errors": []},
            outputs,
            overwrite=False,
        )

    assert not any(path.exists() for path in outputs)


def test_apply_bundle_is_deterministic(tmp_path: Path) -> None:
    module = _load_apply_script()
    bundles: list[list[Path]] = []
    for name in ("first", "second"):
        root = tmp_path / name
        outputs = [
            root / "reviewed.csv",
            root / "apply.json",
            root / "confusion.json",
        ]
        module._publish_output_transaction(
            pd.DataFrame({"row": [1, 2], "value": [0.1, 0.2]}),
            {"errors": [], "rows": 2},
            {"errors": [], "rows": 2},
            outputs,
            overwrite=False,
        )
        bundles.append(outputs)

    assert [path.read_bytes() for path in bundles[0]] == [
        path.read_bytes() for path in bundles[1]
    ]


@pytest.mark.skipif(
    not _all_files_readable((V6_FRAME_LOCAL, V6_MANIFEST, V6_DECISIONS)),
    reason=HIDDEN_EXTERNAL_REASON,
)
def test_actual_v6_apply_and_independent_checker_pass(
    tmp_path: Path,
) -> None:
    input_hashes = {
        path: _sha256(path)
        for path in (V6_FRAME_LOCAL, V6_MANIFEST, V6_DECISIONS)
    }
    output_csv = tmp_path / "hidden_reviewed_frame_features.csv"
    apply_audit = tmp_path / "apply_hidden_review_audit.json"
    confusion_audit = tmp_path / "hidden_confusion_audit.json"
    result = subprocess.run(
        [
            sys.executable,
            str(APPLY_SCRIPT),
            "--input-csv",
            str(V6_FRAME_LOCAL),
            "--manifest-csv",
            str(V6_MANIFEST),
            "--decisions-csv",
            str(V6_DECISIONS),
            "--output-csv",
            str(output_csv),
            "--audit-json",
            str(apply_audit),
            "--confusion-audit-json",
            str(confusion_audit),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    apply_payload = json.loads(apply_audit.read_text(encoding="utf-8"))
    confusion_payload = json.loads(
        confusion_audit.read_text(encoding="utf-8")
    )
    expected_drift = {
        "hidden_false_negative_risk_reasons": 82,
        "hidden_false_negative_risk_score": 16,
    }
    assert apply_payload["input_rows"] == 245680
    assert apply_payload["output_rows"] == 245680
    assert apply_payload["applied_decision_items"] == 5233
    assert apply_payload["approved_metadata_drift_counts"] == expected_drift
    assert apply_payload["approved_metadata_drift_unique_items"] == 82
    assert apply_payload["fatal_metadata_mismatch_counts"] == {}
    assert apply_payload["errors"] == []
    assert confusion_payload["approved_metadata_drift_counts"] == expected_drift
    assert confusion_payload["approved_metadata_drift_unique_items"] == 82
    assert confusion_payload["fatal_metadata_mismatch_counts"] == {}
    assert confusion_payload["errors"] == []

    checker_audit = tmp_path / "apply_output_checker.json"
    checked = subprocess.run(
        [
            sys.executable,
            str(CHECK_SCRIPT),
            "--input-csv",
            str(V6_FRAME_LOCAL),
            "--output-csv",
            str(output_csv),
            "--audit-json",
            str(checker_audit),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert checked.returncode == 0, checked.stderr
    checker_payload = json.loads(checker_audit.read_text(encoding="utf-8"))
    assert checker_payload["input_rows"] == 245680
    assert checker_payload["output_rows"] == 245680
    assert checker_payload["errors"] == []
    assert {path: _sha256(path) for path in input_hashes} == input_hashes
