"""Dry-run deterministic routing against project skill metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import finish, load_json


def audit(root: Path, scenarios_path: Path) -> dict[str, object]:
    """Select active skills by declared trigger phrases without executing work."""
    registry = load_json(root / "skill_registry.json")
    scenarios = load_json(scenarios_path).get("scenarios", [])
    reports: dict[str, object] = {}
    errors: list[str] = []
    for scenario in scenarios:
        prompt = str(scenario["prompt"]).lower()
        selected: set[str] = set()
        for skill in registry.get("skills", []):
            if not skill.get("implicit", False):
                continue
            if any(str(trigger).lower() in prompt for trigger in skill.get("triggers", [])):
                selected.add(str(skill["name"]))
        changed = True
        while changed:
            changed = False
            for skill in registry.get("skills", []):
                if skill["name"] not in selected:
                    continue
                for dependency in skill.get("depends_on", []):
                    if dependency not in selected:
                        selected.add(dependency)
                        changed = True
        expected = set(scenario.get("expected_skills", []))
        missing = sorted(expected - selected)
        unexpected_future = sorted(
            name
            for name in selected
            if next(item for item in registry["skills"] if item["name"] == name)["status"]
            == "future"
        )
        if missing:
            errors.append(f"{scenario['id']}:missing={missing}")
        if unexpected_future:
            errors.append(f"{scenario['id']}:future_selected={unexpected_future}")
        reports[str(scenario["id"])] = {
            "selected": sorted(selected),
            "expected": sorted(expected),
            "missing": missing,
            "future_selected": unexpected_future,
        }
    return {"check": "dry_run_skill_discovery", "scenarios": reports, "errors": errors}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(".agents/skills"))
    parser.add_argument(
        "--scenarios-json",
        type=Path,
        default=Path(".agents/skills/examples/discovery_scenarios.json"),
    )
    args = parser.parse_args()
    return finish(audit(args.root.resolve(), args.scenarios_json.resolve()))


if __name__ == "__main__":
    raise SystemExit(main())
