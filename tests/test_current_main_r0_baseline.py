from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "tracking" / "run_current_main_r0_baseline.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("r0_baseline", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_r0_contract_is_baseline_only() -> None:
    module = _load_module()

    assert module.PROFILE == "realtime_fast"
    assert module.IOU_THRESHOLD == 0.5
    assert module.GAP_TOLERANCE_FRAMES == 15
    source = SCRIPT.read_text(encoding="utf-8")
    assert "group_error_events" not in source
    assert "reconcile_historical" not in source
    assert "rank_mechanisms" not in source


def test_csv_writer_refuses_empty_authority(tmp_path: Path) -> None:
    module = _load_module()

    try:
        module.write_csv(tmp_path / "empty.csv", [])
    except module.R0BaselineError as exc:
        assert "refusing empty CSV" in str(exc)
    else:
        raise AssertionError("empty R0 authority CSV did not fail closed")
