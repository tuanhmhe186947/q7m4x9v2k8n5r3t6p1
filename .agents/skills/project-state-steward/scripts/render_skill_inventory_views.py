#!/usr/bin/env python3
"""Render and verify generated views of the canonical skill inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

INVENTORY_RELATIVE = Path(".agents/skills/skill_inventory.json")
VIEW_RELATIVES = (
    Path(".agents/skills/skill_registry.json"),
    Path(".agents/memory/11_SKILL_PORTFOLIO.json"),
    Path(".agents/skills/README.md"),
)
INVENTORY_SCHEMA = "pig.skill-inventory.v1"
ALLOWED_STATUSES = {"active", "future", "disabled", "retired"}
CANONICAL_SKILL_FIELDS = {
    "skill_id",
    "status",
    "implicit",
    "category",
    "source_root",
    "relative_path",
    "depends_on",
}
PORTFOLIO_REQUIRED_FIELDS = {
    "version_or_commit",
    "file_sha256",
    "tool_api_dependencies",
    "selected_date",
    "last_reviewed",
    "last_real_use",
    "proof_task",
    "stale_signal",
    "next_maintenance_action",
}


class InventoryViewError(ValueError):
    """Raised when the canonical inventory cannot produce safe views."""


def _json_bytes(payload: Any) -> bytes:
    text = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    return f"{text}\n".encode()


def _semantic_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_text(value: Any, locator: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InventoryViewError(f"missing_or_empty:{locator}")
    return value


def _require_text_list(value: Any, locator: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise InventoryViewError(f"invalid_text_list:{locator}")
    return value


def _markdown_lines(value: Any, locator: str) -> list[str]:
    if isinstance(value, str) and value.strip():
        return [value]
    if isinstance(value, list) and value and all(
        isinstance(item, str) and item.strip() for item in value
    ):
        return value
    raise InventoryViewError(f"invalid_markdown_lines:{locator}")


def _validate_inventory(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if payload.get("schema_version") != INVENTORY_SCHEMA:
        raise InventoryViewError("inventory_schema_invalid")
    declared = payload.get("generated_views")
    expected = [path.as_posix() for path in VIEW_RELATIVES]
    if declared != expected:
        raise InventoryViewError("generated_views_contract_mismatch")
    view_contract = payload.get("view_contract")
    if not isinstance(view_contract, dict):
        raise InventoryViewError("view_contract_missing")
    for field in (
        "registry_schema_version",
        "registry_native_root",
        "portfolio_schema_version",
    ):
        _require_text(view_contract.get(field), f"view_contract.{field}")
    review_policy = view_contract.get("review_policy")
    if not isinstance(review_policy, dict):
        raise InventoryViewError("view_contract.review_policy_invalid")
    readme = view_contract.get("readme")
    if not isinstance(readme, dict):
        raise InventoryViewError("view_contract.readme_invalid")
    _markdown_lines(readme.get("intro"), "view_contract.readme.intro")
    _markdown_lines(
        readme.get("dependency_order"),
        "view_contract.readme.dependency_order",
    )
    _markdown_lines(
        readme.get("scope_note"),
        "view_contract.readme.scope_note",
    )
    resources = readme.get("shared_resources")
    if not isinstance(resources, list):
        raise InventoryViewError("view_contract.readme.shared_resources_invalid")
    for index, resource in enumerate(resources):
        if not isinstance(resource, dict):
            raise InventoryViewError(f"readme_resource_invalid:{index}")
        _require_text(resource.get("label"), f"readme_resource:{index}:label")
        _require_text(resource.get("path"), f"readme_resource:{index}:path")

    skills = payload.get("skills")
    if not isinstance(skills, list):
        raise InventoryViewError("skills_invalid")
    mapped: dict[str, dict[str, Any]] = {}
    registry_orders: dict[int, str] = {}
    for index, skill in enumerate(skills):
        if not isinstance(skill, dict):
            raise InventoryViewError(f"skill_record_invalid:{index}")
        missing = CANONICAL_SKILL_FIELDS - set(skill)
        if missing:
            joined = ",".join(sorted(missing))
            raise InventoryViewError(f"skill_fields_missing:{index}:{joined}")
        skill_id = _require_text(skill.get("skill_id"), f"skill:{index}:skill_id")
        if skill_id in mapped:
            raise InventoryViewError(f"skill_duplicate:{skill_id}")
        if skill.get("status") not in ALLOWED_STATUSES:
            raise InventoryViewError(f"skill_status_invalid:{skill_id}")
        if not isinstance(skill.get("implicit"), bool):
            raise InventoryViewError(f"skill_implicit_invalid:{skill_id}")
        for field in ("category", "source_root", "relative_path"):
            _require_text(skill.get(field), f"skill:{skill_id}:{field}")
        _require_text_list(skill.get("depends_on"), f"skill:{skill_id}:depends_on")
        registry = skill.get("registry")
        if registry is not None:
            if not isinstance(registry, dict):
                raise InventoryViewError(f"registry_invalid:{skill_id}")
            order = registry.get("order")
            if not isinstance(order, int) or isinstance(order, bool):
                raise InventoryViewError(f"registry_order_invalid:{skill_id}")
            if order in registry_orders:
                other = registry_orders[order]
                raise InventoryViewError(
                    f"registry_order_duplicate:{order}:{other}:{skill_id}"
                )
            registry_orders[order] = skill_id
            _require_text_list(
                registry.get("triggers"),
                f"registry:{skill_id}:triggers",
            )
            include = registry.get("include_in_readme", True)
            if not isinstance(include, bool):
                raise InventoryViewError(f"readme_include_invalid:{skill_id}")
            if include:
                _require_text(
                    registry.get("invoke_for"),
                    f"registry:{skill_id}:invoke_for",
                )
                _require_text(
                    registry.get("do_not_invoke_for"),
                    f"registry:{skill_id}:do_not_invoke_for",
                )
        portfolio = skill.get("portfolio")
        if portfolio is not None:
            if not isinstance(portfolio, dict):
                raise InventoryViewError(f"portfolio_invalid:{skill_id}")
            duplicated = CANONICAL_SKILL_FIELDS & set(portfolio)
            if duplicated:
                joined = ",".join(sorted(duplicated))
                raise InventoryViewError(
                    f"portfolio_duplicates_canonical:{skill_id}:{joined}"
                )
            missing_portfolio = PORTFOLIO_REQUIRED_FIELDS - set(portfolio)
            if missing_portfolio:
                joined = ",".join(sorted(missing_portfolio))
                raise InventoryViewError(
                    f"portfolio_fields_missing:{skill_id}:{joined}"
                )
        mapped[skill_id] = skill

    for skill_id, skill in mapped.items():
        for dependency in skill["depends_on"]:
            if dependency not in mapped:
                raise InventoryViewError(
                    f"skill_dependency_unknown:{skill_id}:{dependency}"
                )
    routes = payload.get("task_routes")
    if not isinstance(routes, dict):
        raise InventoryViewError("task_routes_invalid")
    for task_class, route in routes.items():
        if not isinstance(route, dict):
            raise InventoryViewError(f"task_route_invalid:{task_class}")
        required_all = _require_text_list(
            route.get("required_all", []),
            f"task_route:{task_class}:required_all",
        )
        required_any = _require_text_list(
            route.get("required_any", []),
            f"task_route:{task_class}:required_any",
        )
        required = required_all + required_any
        if not required:
            raise InventoryViewError(f"task_route_empty:{task_class}")
        for skill_id in required:
            if skill_id not in mapped:
                raise InventoryViewError(
                    f"task_route_skill_unknown:{task_class}:{skill_id}"
                )
        if route.get("reasoning_required") is True and not any(
            mapped[skill_id]["category"] == "reasoning"
            for skill_id in required
        ):
            raise InventoryViewError(
                f"task_route_reasoning_missing:{task_class}"
            )
    return mapped


def _registry_view(
    payload: dict[str, Any],
    inventory_sha256: str,
) -> dict[str, Any]:
    contract = payload["view_contract"]
    selected = [skill for skill in payload["skills"] if "registry" in skill]
    selected.sort(key=lambda item: (item["registry"]["order"], item["skill_id"]))
    records = []
    for skill in selected:
        registry = skill["registry"]
        records.append(
            {
                "name": skill["skill_id"],
                "status": skill["status"],
                "implicit": skill["implicit"],
                "order": registry["order"],
                "triggers": registry["triggers"],
                "depends_on": skill["depends_on"],
            }
        )
    return {
        "schema_version": contract["registry_schema_version"],
        "native_root": contract["registry_native_root"],
        "skills": records,
        "generated_view": {
            "source": INVENTORY_RELATIVE.as_posix(),
            "inventory_sha256": inventory_sha256,
        },
    }


def _portfolio_view(
    payload: dict[str, Any],
    inventory_sha256: str,
) -> dict[str, Any]:
    contract = payload["view_contract"]
    records = []
    for skill in sorted(payload["skills"], key=lambda item: item["skill_id"]):
        if "portfolio" not in skill:
            continue
        record = {
            "skill_id": skill["skill_id"],
            "category": skill["category"],
            "source_root": skill["source_root"],
            "relative_path": skill["relative_path"],
        }
        record.update(skill["portfolio"])
        records.append(record)
    mandatory = {}
    for task_class, route in sorted(payload["task_routes"].items()):
        if route.get("required_all"):
            mandatory[task_class] = route["required_all"]
    return {
        "schema_version": contract["portfolio_schema_version"],
        "review_policy": contract["review_policy"],
        "mandatory_reasoning_routes": mandatory,
        "skills": records,
        "task_routes": payload["task_routes"],
        "generated_view": {
            "source": INVENTORY_RELATIVE.as_posix(),
            "inventory_sha256": inventory_sha256,
        },
    }


def _escape_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _readme_view(payload: dict[str, Any], inventory_sha256: str) -> bytes:
    contract = payload["view_contract"]
    readme = contract["readme"]
    selected = [
        skill
        for skill in payload["skills"]
        if "registry" in skill
        and skill["registry"].get("include_in_readme", True)
    ]
    selected.sort(key=lambda item: (item["registry"]["order"], item["skill_id"]))
    lines = [
        "# Project Skills",
        "",
        "<!-- Generated by render_skill_inventory_views.py. -->",
        f"<!-- Source: {INVENTORY_RELATIVE.as_posix()} -->",
        f"<!-- Inventory SHA-256: {inventory_sha256} -->",
        "",
    ]
    lines.extend(_markdown_lines(readme["intro"], "readme.intro"))
    lines.extend(
        [
            "",
            "## Routing Index",
            "",
            "| Order | Skill | Status |",
            "|---:|---|---|",
        ]
    )
    for skill in selected:
        registry = skill["registry"]
        lines.append(
            "| "
            f"{registry['order']} | `{_escape_cell(skill['skill_id'])}` | "
            f"{_escape_cell(skill['status'])} |"
        )
    lines.extend(["", "### Routing details", ""])
    for skill in selected:
        registry = skill["registry"]
        lines.append(
            f"- `{_escape_cell(skill['skill_id'])}`: invoke for "
            f"{_escape_cell(registry['invoke_for'])};"
        )
        lines.append(
            f"  do not invoke for {_escape_cell(registry['do_not_invoke_for'])}."
        )
    lines.extend(
        [
            "",
            "## Dependency Order",
            "",
        ]
    )
    lines.extend(
        _markdown_lines(readme["dependency_order"], "readme.dependency_order")
    )
    lines.extend(["", "Shared deterministic resources:", ""])
    for resource in readme["shared_resources"]:
        lines.append(f"- [{resource['label']}]({resource['path']})")
    lines.append("")
    lines.extend(_markdown_lines(readme["scope_note"], "readme.scope_note"))
    lines.append("")
    return "\n".join(lines).encode("utf-8")


def build_views(payload: dict[str, Any]) -> dict[Path, bytes]:
    """Build all declared view bytes without touching the filesystem."""
    _validate_inventory(payload)
    inventory_sha256 = _semantic_sha256(payload)
    return {
        VIEW_RELATIVES[0]: _json_bytes(_registry_view(payload, inventory_sha256)),
        VIEW_RELATIVES[1]: _json_bytes(_portfolio_view(payload, inventory_sha256)),
        VIEW_RELATIVES[2]: _readme_view(payload, inventory_sha256),
    }


def load_inventory(root: Path) -> dict[str, Any]:
    path = root / INVENTORY_RELATIVE
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InventoryViewError(f"inventory_read_failed:{path}:{exc}") from exc
    if not isinstance(payload, dict):
        raise InventoryViewError("inventory_root_invalid")
    return payload


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def render_views(root: Path) -> list[Path]:
    views = build_views(load_inventory(root))
    for relative, content in views.items():
        _atomic_write(root / relative, content)
    return list(views)


def check_views(root: Path) -> list[str]:
    expected = build_views(load_inventory(root))
    errors = []
    for relative, content in expected.items():
        path = root / relative
        if not path.is_file():
            errors.append(f"generated_view_missing:{relative.as_posix()}")
        elif path.read_bytes() != content:
            errors.append(f"generated_view_mismatch:{relative.as_posix()}")
    return errors


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("render", "check"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.root.resolve()
    try:
        if args.action == "render":
            rendered = render_views(root)
            for relative in rendered:
                print(f"rendered:{relative.as_posix()}")
            return 0
        errors = check_views(root)
    except InventoryViewError as exc:
        print(f"skill_inventory_view_error:{exc}")
        return 1
    if errors:
        for error in errors:
            print(error)
        return 1
    print(f"skill_inventory_views_match:{len(VIEW_RELATIVES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
