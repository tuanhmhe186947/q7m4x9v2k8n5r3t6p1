from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "classification_v2"
    / "03_image_cache_context"
    / "check_classification_v2_image_loader.py"
)


def test_loader_audit_requires_exact_gui_video_basename() -> None:
    module = _load_script()
    audit = module._audit_mandatory_gui_case(_results("Pigs291119_000231_30fps.mp4"))

    assert audit["ok"] is True
    assert audit["rows"] == 6
    assert audit["resolved_media_basenames"] == ["Pigs291119_000231_30fps.mp4"]


def test_loader_audit_rejects_loadable_wrong_gui_video_basename() -> None:
    module = _load_script()
    audit = module._audit_mandatory_gui_case(_results("Pigs291119_000231.mp4"))

    assert audit["ok"] is False
    assert audit["unloadable_rows"] == 0
    assert any("resolved_media_basename_mismatch" in error for error in audit["errors"])


def _results(basename: str) -> list[dict[str, object]]:
    return [
        {
            "kind": "cvat_video_bbox",
            "identity": {
                "video_key": "Pigs291119_000231",
                "pig_id": "ID_4",
                "frame_index": frame_index,
            },
            "resolved_video_path": f"C:/videos/{basename}",
            "ok": True,
        }
        for frame_index in range(678, 684)
    ]


def _load_script() -> ModuleType:
    module_name = "classification_v2_image_loader_audit_fixture"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
