from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pandas as pd
from PIL import Image

from pig_behavior.classification_v2.datasets.legacy_unreviewed_development import (
    LEGACY_DEVELOPMENT_SCOPE,
)

_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, relative_path: str) -> ModuleType:
    path = _ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_CACHE = _load_script(
    "classification_v2_build_image_cache_test",
    "scripts/classification_v2/03_image_cache_context/"
    "classification_v2_build_image_cache.py",
)
_PACKED = _load_script(
    "classification_v2_build_packed_cache_test",
    "scripts/classification_v2/03_image_cache_context/"
    "classification_v2_build_packed_image_cache.py",
)


def test_cache_and_packed_index_propagate_explicit_lineage_claim(
    tmp_path: Path,
) -> None:
    crop_path = tmp_path / "legacy_crop.png"
    Image.new("RGB", (4, 2), color=(25, 50, 75)).save(crop_path)
    frame_context = tmp_path / "image_frame_context_manifest.csv"
    window_context = tmp_path / "image_window_context_manifest.csv"
    pd.DataFrame(
        [
            {
                "image_context_id": "legacy-context-0",
                "source_type": "legacy_recovered",
                "resolved_media_path": str(crop_path),
                "image_context_loadable": True,
                "image_context_source": "legacy_crop",
                "x1": 0.0,
                "y1": 0.0,
                "x2": 4.0,
                "y2": 2.0,
                "frame_index": 0,
                "lineage_scope": LEGACY_DEVELOPMENT_SCOPE,
                "human_review_complete": False,
            }
        ]
    ).to_csv(frame_context, index=False)
    pd.DataFrame(
        [
            {
                "window_id": "window-0",
                "source_type": "legacy_recovered",
                "video_key": "pigs291119/000100/color.mp4",
                "expected_frame_indices": "0",
                "image_context_id_sequence": "legacy-context-0",
                "window_image_context_complete": True,
                "lineage_scope": LEGACY_DEVELOPMENT_SCOPE,
                "human_review_complete": False,
            }
        ]
    ).to_csv(window_context, index=False)
    cache_root = tmp_path / "cache"

    cache_audit = _CACHE.build_image_cache(
        frame_context_csv=frame_context,
        window_context_csv=window_context,
        output_dir=cache_root,
        image_size=8,
        max_contexts=None,
        source_type=None,
        preview_jpg=False,
        preview_limit=0,
        checkpoint_every=0,
        resume_from_partial=False,
        overwrite=False,
    )
    packed_audit = _PACKED.build_packed_cache(
        cache_manifest=cache_root / "manifest.csv",
        image_size=8,
        output_dir=cache_root,
        max_contexts=None,
        workers=1,
        checkpoint_every=10,
        resume=False,
        overwrite=False,
    )

    manifest = pd.read_csv(cache_root / "manifest.csv")
    packed_index = pd.read_csv(cache_root / "packed_image_cache_index.csv")
    assert manifest.loc[0, "lineage_scope"] == LEGACY_DEVELOPMENT_SCOPE
    assert bool(manifest.loc[0, "human_review_complete"]) is False
    assert packed_index.loc[0, "lineage_scope"] == LEGACY_DEVELOPMENT_SCOPE
    assert bool(packed_index.loc[0, "human_review_complete"]) is False
    assert cache_audit["lineage_scope"] == LEGACY_DEVELOPMENT_SCOPE
    assert cache_audit["human_review_complete"] is False
    assert packed_audit["lineage_scope"] == LEGACY_DEVELOPMENT_SCOPE
    assert packed_audit["human_review_complete"] is False
