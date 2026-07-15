"""Audit byte-identical primary/repeat full-frame ResNet feature caches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = (
    "classification_v2.legacy_development_l6."
    "full_frame_feature_repeat.v1"
)
LINEAGE_SCOPE = "legacy-only-unreviewed-development"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_feature_repeat(config_path: Path) -> tuple[Path, dict[str, Any]]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("lineage_scope") != LINEAGE_SCOPE:
        raise ValueError("full-frame feature lineage scope drift")
    for name, spec in config["inputs"].items():
        path = (config_path.parents[2] / spec["path"]).resolve()
        if _sha256(path) != spec["sha256"]:
            raise ValueError(f"full-frame feature input hash drift={name}")
    for name, spec in config["implementation"].items():
        path = (config_path.parents[2] / spec["path"]).resolve()
        if _sha256(path) != spec["sha256"]:
            raise ValueError(f"full-frame feature implementation drift={name}")
    audits = {}
    for replica in ("primary", "repeat"):
        audit_path = (
            config_path.parents[2]
            / config["outputs"][f"{replica}_audit_relative_path"]
        ).resolve()
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("valid") is not True:
            raise ValueError(f"full-frame feature audit invalid={replica}")
        expected = config["feature_contract"]
        for field, value in expected.items():
            if audit.get(field) != value:
                raise ValueError(f"full-frame feature contract drift={field}")
        audits[replica] = audit
    fields = (
        "source_tensor_sha256",
        "source_index_sha256",
        "feature_tensor_sha256",
        "feature_index_sha256",
        "rows",
        "feature_dim",
        "feature_dtype",
        "backbone_name",
        "pretrained_weight_enum",
        "weights_sha256",
    )
    equality = {
        field: audits["primary"].get(field) == audits["repeat"].get(field)
        for field in fields
    }
    errors = [field for field, equal in equality.items() if not equal]
    output = (
        config_path.parents[2] / config["outputs"]["repeat_gate_relative_path"]
    ).resolve()
    if output.exists():
        raise FileExistsError(f"full-frame feature repeat gate exists={output}")
    payload = {
        "schema_version": SCHEMA,
        "status": (
            "PASS_LEGACY_DEVELOPMENT_L6_FULL_FRAME_FEATURE_REPEAT"
            if not errors
            else "FAIL_LEGACY_DEVELOPMENT_L6_FULL_FRAME_FEATURE_REPEAT"
        ),
        "lineage_scope": LINEAGE_SCOPE,
        "canonical_source_name": "legacy_16f",
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "config_sha256": _sha256(config_path),
        "equality": equality,
        "primary_audit_sha256": _sha256(
            config_path.parents[2]
            / config["outputs"]["primary_audit_relative_path"]
        ),
        "repeat_audit_sha256": _sha256(
            config_path.parents[2]
            / config["outputs"]["repeat_audit_relative_path"]
        ),
        "source_media_reads": 0,
        "outer_holdout_rows": 0,
        "errors": errors,
        "valid": not errors,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output, payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    output, payload = audit_feature_repeat(args.config)
    print(
        json.dumps(
            {
                "output_path": str(output),
                "status": payload["status"],
                "valid": payload["valid"],
                "errors": payload["errors"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
