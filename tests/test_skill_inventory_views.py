from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT
    / ".agents"
    / "skills"
    / "project-state-steward"
    / "scripts"
    / "render_skill_inventory_views.py"
)
VALIDATOR_PATH = SCRIPT_PATH.with_name("validate_governance_contracts.py")
SPEC = importlib.util.spec_from_file_location("render_skill_inventory_views", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
RENDERER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RENDERER)


def _load_validator():
    spec = importlib.util.spec_from_file_location(
        "skill_inventory_view_validator",
        VALIDATOR_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture_inventory() -> dict[str, object]:
    return {
        "schema_version": "pig.skill-inventory.v1",
        "generated_views": [
            ".agents/skills/skill_registry.json",
            ".agents/memory/11_SKILL_PORTFOLIO.json",
            ".agents/skills/README.md",
        ],
        "view_contract": {
            "registry_schema_version": "project_skill_registry_v2",
            "registry_native_root": ".agents/skills",
            "portfolio_schema_version": "pig.skill-portfolio.v1",
            "review_policy": {
                "reasoning_review_days": 60,
                "code_skill_review_days": 30,
                "stale_signals": ["user_correction"],
            },
            "readme": {
                "intro": "Generated fixture inventory.",
                "dependency_order": "Use the reasoning skill before execution.",
                "shared_resources": [
                    {"label": "checks", "path": "checks"},
                ],
                "scope_note": "The fixture grants no effects.",
            },
        },
        "task_routes": {
            "fixture_task": {
                "required_all": ["fixture-reasoning"],
                "reasoning_required": True,
            }
        },
        "skills": [
            {
                "skill_id": "fixture-reasoning",
                "status": "active",
                "implicit": False,
                "category": "reasoning",
                "source_root": "project",
                "relative_path": ".agents/skills/fixture-reasoning/SKILL.md",
                "depends_on": [],
                "registry": {
                    "order": 1,
                    "triggers": ["fixture"],
                    "invoke_for": "fixture reasoning",
                    "do_not_invoke_for": "unrelated tasks",
                },
                "portfolio": {
                    "version_or_commit": "fixture-v1",
                    "file_sha256": "a" * 64,
                    "tool_api_dependencies": ["filesystem"],
                    "selected_date": "2026-08-13",
                    "last_reviewed": "2026-08-13",
                    "last_real_use": "2026-08-13",
                    "proof_task": "fixture parity",
                    "stale_signal": "fixture_change",
                    "next_maintenance_action": "rerun fixture tests",
                },
            }
        ],
    }


def _write_inventory(root: Path, payload: dict[str, object]) -> None:
    path = root / RENDERER.INVENTORY_RELATIVE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def test_checked_in_views_match_canonical_inventory() -> None:
    assert RENDERER.check_views(PROJECT_ROOT) == []


def test_views_preserve_declared_machine_fields() -> None:
    inventory = RENDERER.load_inventory(PROJECT_ROOT)
    views = RENDERER.build_views(inventory)
    registry = json.loads(views[RENDERER.VIEW_RELATIVES[0]])
    portfolio = json.loads(views[RENDERER.VIEW_RELATIVES[1]])
    registry_by_id = {record["name"]: record for record in registry["skills"]}
    portfolio_by_id = {
        record["skill_id"]: record for record in portfolio["skills"]
    }
    canonical_by_id = {
        record["skill_id"]: record for record in inventory["skills"]
    }

    for skill_id, canonical in canonical_by_id.items():
        if "registry" in canonical:
            registry_record = registry_by_id[skill_id]
            assert registry_record["triggers"] == canonical["registry"]["triggers"]
            assert registry_record["depends_on"] == canonical["depends_on"]
        if "portfolio" in canonical:
            portfolio_record = portfolio_by_id[skill_id]
            for field, value in canonical["portfolio"].items():
                assert portfolio_record[field] == value
    assert portfolio["task_routes"]["governance_implementation"] == (
        inventory["task_routes"]["governance_implementation"]
    )
    assert portfolio["task_routes"]["classification_v2_scientific_experiment"] == (
        inventory["task_routes"]["classification_v2_scientific_experiment"]
    )
    readme = views[RENDERER.VIEW_RELATIVES[2]].decode("utf-8")
    assert "agent-governance-steward" in readme
    assert "Generated by render_skill_inventory_views.py" in readme


def test_umbrella_validator_enforces_view_parity(tmp_path: Path) -> None:
    _write_inventory(tmp_path, _fixture_inventory())
    RENDERER.render_views(tmp_path)
    validator = _load_validator()

    assert validator._check_skill_inventory_views(tmp_path) == []

    target = tmp_path / RENDERER.VIEW_RELATIVES[0]
    target.write_bytes(target.read_bytes() + b"tampered\n")
    assert validator._check_skill_inventory_views(tmp_path) == [
        "skill_inventory_view_generated_view_mismatch:"
        ".agents/skills/skill_registry.json"
    ]


def test_modern_inventory_requires_view_declaration(tmp_path: Path) -> None:
    payload = _fixture_inventory()
    payload["generated_views"] = []
    _write_inventory(tmp_path, payload)

    validator = _load_validator()

    assert validator._check_skill_inventory_views(tmp_path) == [
        "skill_inventory_views_declaration_empty"
    ]


def test_legacy_inventory_without_view_contract_is_accepted(tmp_path: Path) -> None:
    payload = _fixture_inventory()
    payload.pop("view_contract")
    payload["generated_views"] = []
    for skill in payload["skills"]:
        skill.pop("registry")
        skill.pop("portfolio")
    _write_inventory(tmp_path, payload)

    validator = _load_validator()

    assert validator._check_skill_inventory_views(tmp_path) == []


def test_render_is_byte_idempotent(tmp_path: Path) -> None:
    _write_inventory(tmp_path, _fixture_inventory())
    RENDERER.render_views(tmp_path)
    first = {
        relative: (tmp_path / relative).read_bytes()
        for relative in RENDERER.VIEW_RELATIVES
    }

    RENDERER.render_views(tmp_path)

    assert {
        relative: (tmp_path / relative).read_bytes()
        for relative in RENDERER.VIEW_RELATIVES
    } == first
    assert RENDERER.check_views(tmp_path) == []


@pytest.mark.parametrize("relative", RENDERER.VIEW_RELATIVES)
def test_check_rejects_tampered_view(tmp_path: Path, relative: Path) -> None:
    _write_inventory(tmp_path, _fixture_inventory())
    RENDERER.render_views(tmp_path)
    path = tmp_path / relative
    path.write_bytes(path.read_bytes() + b"tampered\n")

    assert RENDERER.check_views(tmp_path) == [
        f"generated_view_mismatch:{relative.as_posix()}"
    ]


def test_inventory_change_invalidates_all_views(tmp_path: Path) -> None:
    payload = _fixture_inventory()
    _write_inventory(tmp_path, payload)
    RENDERER.render_views(tmp_path)
    payload["view_contract"]["readme"]["scope_note"] = ["Changed authority."]
    _write_inventory(tmp_path, payload)

    assert RENDERER.check_views(tmp_path) == [
        f"generated_view_mismatch:{relative.as_posix()}"
        for relative in RENDERER.VIEW_RELATIVES
    ]
