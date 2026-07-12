"""Validate project-local skill structure, metadata, sections, and links."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from _common import finish, load_json

SECTIONS = [
    "Purpose",
    "When to use",
    "Project context",
    "Required inputs",
    "Scientific invariants",
    "Ordered procedure",
    "Required outputs",
    "Validation commands",
    "Stop conditions",
    "Forbidden actions",
    "Completion report format",
]
RESOURCE_DIRS = ("templates", "checks", "examples")
SHARED_TEMPLATES = (
    "skill_completion_report.md",
    "run_manifest.example.json",
    "feature_whitelist.example.json",
    "experiment_matrix.example.csv",
    "promotion_decision.example.json",
)
LINK_RE = re.compile(r"\[[^]]+\]\(([^)]+)\)")


def _frontmatter(text: str) -> dict[str, str]:
    if not text.startswith("---\n") or "\n---\n" not in text[4:]:
        return {}
    block = text.split("\n---\n", 1)[0].splitlines()[1:]
    values: dict[str, str] = {}
    for line in block:
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    return values


def _broken_links(path: Path, text: str) -> list[str]:
    broken: list[str] = []
    for raw in LINK_RE.findall(text):
        target = raw.strip().split("#", 1)[0]
        if not target or "://" in target or target.startswith("mailto:"):
            continue
        if not (path.parent / target).resolve().exists():
            broken.append(raw)
    return broken


def audit(root: Path) -> dict[str, object]:
    """Validate every skill declared in the machine routing contract."""
    registry = load_json(root / "skill_registry.json")
    errors: list[str] = []
    skill_reports: dict[str, object] = {}
    markdown_paths = [root / "README.md"]
    markdown_paths.extend((root / "templates").rglob("*.md"))
    markdown_paths.extend((root / "examples").rglob("*.md"))
    for record in registry.get("skills", []):
        name = str(record["name"])
        skill_dir = root / name
        skill_md = skill_dir / "SKILL.md"
        markdown_paths.append(skill_md)
        markdown_paths.extend((skill_dir / "templates").rglob("*.md"))
        markdown_paths.extend((skill_dir / "examples").rglob("*.md"))
        text = skill_md.read_text(encoding="utf-8") if skill_md.is_file() else ""
        metadata = _frontmatter(text)
        sections = re.findall(r"^## (.+)$", text, flags=re.MULTILINE)
        local_errors: list[str] = []
        if metadata.get("name") != name:
            local_errors.append("invalid_frontmatter_name")
        if not metadata.get("description") or "TODO" in metadata.get("description", ""):
            local_errors.append("invalid_frontmatter_description")
        if sections != SECTIONS:
            local_errors.append(f"section_order={sections}")
        if "TODO" in text or not text.strip():
            local_errors.append("placeholder_or_empty_markdown")
        for resource in RESOURCE_DIRS:
            resource_dir = skill_dir / resource
            if not resource_dir.is_dir() or not any(resource_dir.iterdir()):
                local_errors.append(f"empty_resource_dir={resource}")
        check_manifest = skill_dir / "checks" / "check_manifest.json"
        if check_manifest.is_file():
            check_payload = load_json(check_manifest)
            for raw_path in check_payload.get("shared_checks", []):
                resolved = (check_manifest.parent / str(raw_path)).resolve()
                if not resolved.is_file():
                    local_errors.append(f"missing_shared_check={raw_path}")
        broken = _broken_links(skill_md, text)
        if broken:
            local_errors.append(f"broken_links={broken}")
        openai_yaml = skill_dir / "agents" / "openai.yaml"
        policy = openai_yaml.read_text(encoding="utf-8") if openai_yaml.is_file() else ""
        if "default_prompt:" not in policy:
            local_errors.append("missing_default_prompt")
        if record.get("status") == "future" and "allow_implicit_invocation: false" not in policy:
            local_errors.append("future_skill_implicit_policy_not_disabled")
        if local_errors:
            errors.extend(f"{name}:{value}" for value in local_errors)
        skill_reports[name] = {
            "status": record.get("status"),
            "sections": len(sections),
            "errors": local_errors,
        }
    for name in SHARED_TEMPLATES:
        if not (root / "templates" / name).is_file():
            errors.append(f"missing_shared_template={name}")
    for markdown_path in markdown_paths:
        text = markdown_path.read_text(encoding="utf-8") if markdown_path.is_file() else ""
        display = markdown_path.relative_to(root) if markdown_path.exists() else markdown_path
        if not text.strip():
            errors.append(f"empty_markdown={display}")
            continue
        broken = _broken_links(markdown_path, text)
        if broken:
            errors.append(f"broken_markdown_links={display}:{broken}")
    return {
        "check": "skill_pack",
        "native_root": str(root),
        "skill_count": len(skill_reports),
        "skills": skill_reports,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(".agents/skills"))
    return finish(audit(parser.parse_args().root.resolve()))


if __name__ == "__main__":
    raise SystemExit(main())
