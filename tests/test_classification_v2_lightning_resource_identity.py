"""Exact identity tests for active Lightning scientific execution resources."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from pig_behavior.classification_v2.training import lightning_resource_identity as identity

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / (
    "docs/classification_v2/corrected_pooled_route_20260806/"
    "next_phase_20260806_r2/"
    "s1_stage1_lightning_resource_naming_contract_20260810_v2.json"
)
ADDENDUM = CONTRACT.with_name("s1_stage1_confirmation_execution_addendum_20260810_v2.json")


def _observed() -> dict[str, object]:
    return {
        "teamspace": "pig-project",
        "studio": "pig-gpu-l4",
        "ssh_alias": "lightning-pig-gcp",
        "gpu_count": 1,
        "gpu_model": "NVIDIA L4",
    }


def test_exact_active_resource_contract_passes_and_is_bound_to_addendum(
    tmp_path: Path,
) -> None:
    contract = identity.load_lightning_resource_contract(CONTRACT)
    assert contract["expected_resource"] == _observed()
    observed_path = tmp_path / "control_plane_observation.json"
    observed_path.write_text(json.dumps(_observed()), encoding="utf-8")
    report_path = tmp_path / "resource_preflight.json"
    report = identity.write_lightning_resource_preflight(
        contract_path=CONTRACT,
        observed_resource_path=observed_path,
        output_path=report_path,
    )
    assert report["status"] == "PASS"
    assert identity.validate_passing_lightning_resource_preflight(
        contract_path=CONTRACT,
        preflight_path=report_path,
    )["resource"] == _observed()
    addendum = json.loads(ADDENDUM.read_text(encoding="utf-8"))
    assert addendum["resource_naming_contract"]["relative_path"] == CONTRACT.name
    base_packet = ADDENDUM.with_name(
        "s1_stage1_confirmation_l4_launch_packet_20260810.json"
    )
    assert addendum["base_launch_packet"]["sha256"] == sha256(
        base_packet.read_bytes()
    ).hexdigest()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("teamspace", "pig-gpu-l4"),
        ("studio", "pig-project"),
        ("studio", "pig_project"),
        ("ssh_alias", "lightning-pig-gcp-alt"),
        ("gpu_count", 2),
        ("gpu_model", "NVIDIA A10"),
    ],
)
def test_wrong_active_resource_identity_fails_closed(field: str, value: object) -> None:
    observed = _observed()
    observed[field] = value
    with pytest.raises(identity.LightningResourceIdentityError, match="mismatch"):
        identity.build_lightning_resource_preflight(
            contract_path=CONTRACT,
            observed_resource=observed,
        )


def test_historical_old_studio_name_is_not_an_active_resource_observation(
    tmp_path: Path,
) -> None:
    historical = tmp_path / "historical_execution.json"
    historical.write_text(json.dumps({"studio": "pig_project"}), encoding="utf-8")
    assert historical.is_file()
    assert identity.build_lightning_resource_preflight(
        contract_path=CONTRACT,
        observed_resource=_observed(),
    )["status"] == "PASS"
