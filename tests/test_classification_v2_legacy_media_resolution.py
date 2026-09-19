from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pig_behavior.classification_v2.training.legacy_media_resolution import (
    LegacyMediaResolutionError,
    attach_canonical_legacy_media_paths,
    runtime_media_path,
)

CANONICAL_FRAME = (
    "outputs/legacy_16f_rebuild/legacy_16f_rebuild_20260718_v2/"
    "06_full_recovery/crops/dense_tracklet_0_to_12/pigs031219/000064/"
    "burst_color_35986623_733/ID_1/f000013.jpg"
)


def test_legacy_context_resolves_to_canonical_runtime_jpeg(tmp_path: Path) -> None:
    frames = pd.DataFrame(
        {
            "source_type": ["legacy_recovered"],
            "image_context_id": ["legacy-context"],
            "resolved_media_path": [CANONICAL_FRAME.replace("/", "\\")],
        }
    )

    resolved = attach_canonical_legacy_media_paths(frames)
    runtime = runtime_media_path(
        input_root=tmp_path,
        canonical_relative_path=resolved.loc[0, "resolved_media_path"],
    )

    assert resolved.loc[0, "image_context_id"] == "legacy-context"
    assert resolved.loc[0, "resolved_media_path"] == CANONICAL_FRAME
    assert runtime == tmp_path / Path(CANONICAL_FRAME)


@pytest.mark.parametrize(
    "invalid",
    [
        "reviewed_rgb_v1/legacy-context",
        "../outside/f000013.jpg",
        "C:/outside/f000013.jpg",
    ],
)
def test_legacy_resolution_rejects_noncanonical_paths(invalid: str) -> None:
    frames = pd.DataFrame(
        {
            "source_type": ["legacy_recovered"],
            "image_context_id": ["legacy-context"],
            "resolved_media_path": [invalid],
        }
    )

    with pytest.raises(LegacyMediaResolutionError, match="invalid canonical"):
        attach_canonical_legacy_media_paths(frames)
