from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def main() -> None:
    """Audit whether the saved full-OOF preflight is fresh enough to authorize."""

    parser = argparse.ArgumentParser(
        description="Check classification_v2 full OOF preflight freshness."
    )
    parser.add_argument(
        "--preflight-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_design/full_multimodal_oof_preflight.json"
        ),
    )
    parser.add_argument(
        "--snapshot-check-json",
        type=Path,
        default=Path("outputs/classification_v2/model_design/q2_snapshot_check_audit.json"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_design/"
            "full_oof_preflight_freshness_audit.json"
        ),
    )
    args = parser.parse_args()

    audit = check_preflight_freshness(
        preflight_json=args.preflight_json,
        snapshot_check_json=args.snapshot_check_json,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if audit["errors"]:
        raise SystemExit(1)


def check_preflight_freshness(
    *,
    preflight_json: Path,
    snapshot_check_json: Path,
) -> dict[str, Any]:
    """Return a fail-closed freshness audit without executing full OOF."""

    errors: list[str] = []
    warnings: list[str] = []
    preflight = _load_json(preflight_json, errors)
    snapshot_check = _load_json(snapshot_check_json, errors)
    git_state = _git_state()
    current_snapshot_id = snapshot_check.get("current_snapshot_id")
    preflight_git_commit = preflight.get("git_commit")
    preflight_snapshot_id = preflight.get("snapshot_id")
    git_matches = bool(
        git_state.get("commit") and git_state.get("commit") == preflight_git_commit
    )
    snapshot_matches = bool(
        current_snapshot_id and current_snapshot_id == preflight_snapshot_id
    )
    preflight_valid = preflight.get("valid") is True and not preflight.get("errors")
    preflight_fresh = bool(preflight_valid and git_matches and snapshot_matches)
    if not preflight_valid:
        warnings.append(f"preflight_invalid={preflight.get('errors')}")
    if not git_matches:
        warnings.append(
            "preflight_git_commit_stale="
            f"preflight:{preflight_git_commit},current:{git_state.get('commit')}"
        )
    if not snapshot_matches:
        warnings.append(
            "preflight_snapshot_stale="
            f"preflight:{preflight_snapshot_id},current:{current_snapshot_id}"
        )
    return {
        "schema_version": "classification_v2_full_oof_preflight_freshness_v1",
        "preflight_json": str(preflight_json),
        "snapshot_check_json": str(snapshot_check_json),
        "preflight_valid": preflight_valid,
        "preflight_git_commit": preflight_git_commit,
        "current_git_commit": git_state.get("commit"),
        "git_dirty": git_state.get("dirty"),
        "git_commit_matches": git_matches,
        "preflight_snapshot_id": preflight_snapshot_id,
        "current_snapshot_id": current_snapshot_id,
        "snapshot_matches": snapshot_matches,
        "preflight_fresh": preflight_fresh,
        "full_oof_execution_allowed": preflight_fresh,
        "authorization_must_refresh_preflight": not preflight_fresh,
        "warnings": warnings,
        "errors": errors,
        "valid": not errors,
    }


def _load_json(path: Path, errors: list[str]) -> dict[str, Any]:
    if not path.exists():
        errors.append(f"missing_json={path}")
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _git_state() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--short"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except Exception:
        return {"commit": None, "dirty": None}
    return {"commit": commit or None, "dirty": dirty}


if __name__ == "__main__":
    main()
