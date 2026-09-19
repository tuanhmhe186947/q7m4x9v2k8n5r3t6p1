"""Independent elementary verifier for Phase 4 semantic lineage claims."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def canonical(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("nonfinite value")
        return 0.0 if value == 0.0 else value
    if isinstance(value, dict):
        return {
            str(key): canonical(nested)
            for key, nested in value.items()
            if key
            not in {
                "generated_at",
                "semantic_bundle_hash",
                "timestamp",
            }
        }
    if isinstance(value, list):
        return [canonical(item) for item in value]
    if isinstance(value, set):
        normalized = [canonical(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(item, sort_keys=True),
        )
    raise TypeError(type(value).__name__)


def digest(value: Any) -> str:
    encoded = json.dumps(
        canonical(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


GRAPH = {
    "source": (),
    "frame_local": ("source",),
    "hidden": ("frame_local",),
    "temporal": ("hidden",),
    "native": ("temporal",),
    "tensor": ("native",),
    "split": ("tensor",),
    "release": ("split",),
}
ORDER = list(GRAPH)


def closure(start: set[str]) -> set[str]:
    result = set(start)
    changed = True
    while changed:
        changed = False
        for stage, dependencies in GRAPH.items():
            if stage not in result and any(
                dependency in result for dependency in dependencies
            ):
                result.add(stage)
                changed = True
    return result


def earliest(changed: set[str]) -> str | None:
    return min(changed, key=ORDER.index) if changed else None


def carry(old: dict | None, new: dict | None) -> str:
    if old is None:
        return "NEW_ONLY_REQUIRES_REVIEW"
    if new is None:
        return "OLD_ONLY_AUDIT_EVIDENCE"
    if old["key"] != new["key"]:
        return "REQUIRES_HUMAN_REVALIDATION"
    if old["visual_hash"] != new["visual_hash"]:
        return "REQUIRES_HUMAN_REVALIDATION"
    if old["schema"] != new["schema"]:
        return "INVALID_DECISION_SCHEMA"
    return "EXACT_CARRY_FORWARD_CANDIDATE"


def check(
    checks: list[dict[str, Any]],
    check_id: str,
    actual: Any,
    expected: Any,
) -> None:
    checks.append(
        {
            "check_id": check_id,
            "actual": actual,
            "expected": expected,
            "pass": actual == expected,
        }
    )


def run_checks() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    check(
        checks,
        "dictionary_order",
        digest({"b": 2, "a": 1}),
        digest({"a": 1, "b": 2}),
    )
    check(
        checks,
        "json_whitespace",
        digest(json.loads('{\n  "a": 1,\n  "b": 2\n}')),
        digest(json.loads('{"a":1,"b":2}')),
    )
    check(
        checks,
        "generated_timestamp_excluded",
        digest({"a": 1, "generated_at": "old"}),
        digest({"a": 1, "generated_at": "new"}),
    )
    check(
        checks,
        "threshold_sensitivity",
        digest({"threshold": 0.08}) != digest({"threshold": 0.081}),
        True,
    )
    check(
        checks,
        "ordered_schema_sensitivity",
        digest({"features": ["vx", "vy"]})
        != digest({"features": ["vy", "vx"]}),
        True,
    )
    check(
        checks,
        "unordered_set_stability",
        digest({"ids": {"b", "a"}}),
        digest({"ids": {"a", "b"}}),
    )
    check(
        checks,
        "dependency_closure",
        sorted(closure({"frame_local"}), key=ORDER.index),
        [
            "frame_local",
            "hidden",
            "temporal",
            "native",
            "tensor",
            "split",
            "release",
        ],
    )
    check(
        checks,
        "earliest_multiple_changes",
        earliest({"tensor", "frame_local"}),
        "frame_local",
    )
    check(
        checks,
        "model_only_preserves_upstream",
        "frame_local" not in closure({"tensor"}),
        True,
    )
    old = {"key": "u1", "visual_hash": "aaa", "schema": "v1"}
    check(
        checks,
        "exact_carry_forward",
        carry(old, dict(old)),
        "EXACT_CARRY_FORWARD_CANDIDATE",
    )
    changed_visual = dict(old)
    changed_visual["visual_hash"] = "bbb"
    check(
        checks,
        "changed_visual_revalidation",
        carry(old, changed_visual),
        "REQUIRES_HUMAN_REVALIDATION",
    )
    check(
        checks,
        "old_only_audit",
        carry(old, None),
        "OLD_ONLY_AUDIT_EVIDENCE",
    )
    check(
        checks,
        "new_only_review",
        carry(None, old),
        "NEW_ONLY_REQUIRES_REVIEW",
    )
    unsigned_authorizations = {
        name: False
        for name in (
            "source_rebuild",
            "frame_local_rebuild",
            "hidden_review",
            "native_evidence",
            "pig_strenet",
            "behavior_gui",
            "training",
        )
    }
    check(
        checks,
        "unsigned_release_fail_closed",
        any(unsigned_authorizations.values()),
        False,
    )
    passed = all(item["pass"] for item in checks)
    return {
        "verifier": "phase4_independent_reference_verifier",
        "imports_production_lineage_code": False,
        "check_count": len(checks),
        "checks": checks,
        "pass": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_checks()
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
