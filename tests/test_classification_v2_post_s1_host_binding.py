from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pig_behavior.classification_v2.training import post_s1_host_binding as binding
from pig_behavior.classification_v2.training.remote_input_resolution import (
    RemoteInputAuthority,
)
from pig_behavior.classification_v2.training.stage1_rgb_binding import (
    ResolvedStage1RgbBinding,
)


def _authority(authority_id: str = "input-authority") -> RemoteInputAuthority:
    return RemoteInputAuthority(
        authority_id=authority_id,
        expected_file_count=12,
        expected_total_bytes=345,
        preferred_runtime_locator=Path("/registered-inputs"),
        registered_runtime_locators=(Path("/registered-inputs"),),
        sentinel_sha256={},
        historical_parity_evidence={"relative_path": "report.json", "sha256": "a" * 64},
        parity_report_locator=Path("report.json"),
    )


def _runtime(root: Path, authority_id: str = "input-authority") -> dict[str, object]:
    return {
        "scientific_input_authority_id": authority_id,
        "effective_remote_input_root": str(root),
        "expected_file_count": 12,
        "expected_total_bytes": 345,
        "parity_report_sha256": "b" * 64,
    }


def _roles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "window_id": ["train-1", "train-2", "validation-1"],
            "primary_s1_role": ["train", "train", "validation"],
        }
    )


@pytest.fixture
def fake_stage1(monkeypatch: pytest.MonkeyPatch) -> None:
    def materialize(**kwargs: object) -> dict[str, object]:
        output_dir = Path(str(kwargs["output_dir"]))
        output_dir.mkdir(parents=True)
        scientific = output_dir / "scientific_stage1_rgb_binding.json"
        data_bindings = output_dir / "stage1_temporal_data_bindings.json"
        scientific.write_text(json.dumps({"scientific": "fixed"}), encoding="utf-8")
        data_bindings.write_text(json.dumps({"execution": "derived"}), encoding="utf-8")
        return {
            "scientific_binding_path": str(scientific),
            "data_bindings_path": str(data_bindings),
        }

    def resolve(**kwargs: object) -> ResolvedStage1RgbBinding:
        root = Path(str(kwargs["data_bindings_path"])).parent
        return ResolvedStage1RgbBinding(
            frame_context_path=root / "frames.csv",
            window_context_path=root / "windows.csv",
            packed_index_path=root / "index.csv",
            packed_cache_path=root / "cache.npy",
            hashes={"rgb_scientific_binding": "c" * 64},
            coverage={"train_windows": 2, "validation_windows": 1},
            audit={"valid": True},
        )

    monkeypatch.setattr(binding, "materialize_stage1_rgb_binding", materialize)
    monkeypatch.setattr(binding, "resolve_stage1_execution_rgb_binding", resolve)


def _ensure(
    tmp_path: Path,
    *,
    code_sha: str = "c" * 40,
    input_authority: RemoteInputAuthority | None = None,
    runtime: dict[str, object] | None = None,
    roles: pd.DataFrame | None = None,
    resolution: int = 64,
    name: str = "host-binding.json",
    cvat_source_registration_path: Path | None = None,
) -> binding.ResolvedPostS1HostBinding:
    root = tmp_path / "inputs"
    root.mkdir(exist_ok=True)
    return binding.ensure_post_s1_t6_host_binding(
        binding_path=tmp_path / name,
        canonical_code_sha=code_sha,
        input_authority=input_authority or _authority(),
        runtime_input_binding=runtime or _runtime(root),
        media_root=root,
        rgb_source_root=root / "reviewed_rgb_v1",
        t6_population_authority_sha256="d" * 64,
        t6_population_provenance_hashes={"fold_roles": "e" * 64},
        requested_roles=roles if roles is not None else _roles(),
        input_resolution=resolution,
        cvat_source_registration_path=cvat_source_registration_path,
    )


def test_valid_existing_binding_is_accepted(
    tmp_path: Path,
    fake_stage1: None,
) -> None:
    initial = _ensure(tmp_path)
    existing = _ensure(tmp_path)

    assert initial.regenerated is True
    assert existing.regenerated is False
    assert existing.binding_sha256 == initial.binding_sha256


def test_absent_or_stale_code_binding_is_materialized_again(
    tmp_path: Path,
    fake_stage1: None,
) -> None:
    initial = _ensure(tmp_path, code_sha="c" * 40)
    refreshed = _ensure(tmp_path, code_sha="f" * 40)

    assert initial.regenerated is True
    assert refreshed.regenerated is True
    assert refreshed.binding_sha256 != initial.binding_sha256
    assert refreshed.payload["runtime_realization"]["canonical_code_sha"] == "f" * 40


def test_wrong_input_authority_or_t6_population_fails_closed(
    tmp_path: Path,
    fake_stage1: None,
) -> None:
    _ensure(tmp_path)
    root = tmp_path / "inputs"

    with pytest.raises(binding.PostS1HostBindingError, match="scientific identity"):
        _ensure(
            tmp_path,
            input_authority=_authority("other-authority"),
            runtime=_runtime(root, "other-authority"),
        )
    changed_roles = _roles()
    changed_roles.loc[0, "primary_s1_role"] = "validation"
    with pytest.raises(binding.PostS1HostBindingError, match="scientific identity"):
        _ensure(tmp_path, roles=changed_roles)


def test_outer_roles_are_rejected_before_materialization(
    tmp_path: Path,
    fake_stage1: None,
) -> None:
    roles = _roles()
    roles.loc[0, "primary_s1_role"] = "outer"

    with pytest.raises(binding.PostS1HostBindingError, match="outer/test"):
        _ensure(tmp_path, roles=roles)


def test_equivalent_locator_keeps_scientific_identity_and_resolution_only_changes_runtime(
    tmp_path: Path,
    fake_stage1: None,
) -> None:
    first_root = tmp_path / "first-inputs"
    second_root = tmp_path / "second-inputs"
    first_root.mkdir()
    second_root.mkdir()
    bindings = []
    for resolution in (64, 128, 160):
        root = first_root if resolution == 64 else second_root
        bindings.append(
            binding.ensure_post_s1_t6_host_binding(
                binding_path=tmp_path / f"r{resolution}.json",
                canonical_code_sha="c" * 40,
                input_authority=_authority(),
                runtime_input_binding=_runtime(root),
                media_root=root,
                rgb_source_root=root / "reviewed_rgb_v1",
                t6_population_authority_sha256="d" * 64,
                t6_population_provenance_hashes={"fold_roles": "e" * 64},
                requested_roles=_roles(),
                input_resolution=resolution,
            )
        )

    identities = [item.payload["scientific_identity"] for item in bindings]
    runtimes = [item.payload["runtime_realization"] for item in bindings]
    assert identities[0] == identities[1] == identities[2]
    assert [item["input_resolution"] for item in runtimes] == [64, 128, 160]
    assert runtimes[0]["effective_remote_input_root"] != runtimes[1]["effective_remote_input_root"]


def test_cvat_registration_hash_is_runtime_provenance_and_drift_regenerates(
    tmp_path: Path,
    fake_stage1: None,
) -> None:
    registration = tmp_path / "registration.json"
    registration.write_text(
        json.dumps(
            {
                "schema_version": "classification_v2.cvat_source_registration.v1",
                "status": "ACTIVE_PATH_ONLY_ENRICHMENT",
                "registrations": [
                    {
                        "source_video_key": "key-a",
                        "registered_relative_media_path": "data/videos/a.mp4",
                        "source_provenance": {"tracking_video_key": "a"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    initial = _ensure(
        tmp_path,
        cvat_source_registration_path=registration,
    )
    registration.write_text(
        registration.read_text(encoding="utf-8").replace("a.mp4", "b.mp4"),
        encoding="utf-8",
    )
    refreshed = _ensure(
        tmp_path,
        cvat_source_registration_path=registration,
    )

    assert initial.regenerated is True
    assert refreshed.regenerated is True
    assert refreshed.binding_sha256 != initial.binding_sha256
