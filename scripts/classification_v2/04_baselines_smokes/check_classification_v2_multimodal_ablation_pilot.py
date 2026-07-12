from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pig_behavior.classification_v2.training.full_multimodal_oof import ABLATION_VARIANTS, _ablation_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Check matched branch semantics for the bounded ablation pilot.")
    parser.add_argument(
        "--audit-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_smoke/multimodal_ablation_pilot/multimodal_ablation_pilot_audit.json"
        ),
    )
    args = parser.parse_args()
    audit = json.loads(args.audit_json.read_text(encoding="utf-8"))
    result = check_ablation_pilot(audit, args.audit_json)
    print(json.dumps(result, indent=2))
    if not result["valid"]:
        raise SystemExit(2)


def check_ablation_pilot(audit: dict[str, Any], audit_path: Path) -> dict[str, Any]:
    """Fail closed on variant naming, topology, matched rows, or cache fallback."""

    errors: list[str] = []
    records = list(audit.get("records", []))
    by_variant = {str(record.get("variant")): record for record in records}
    missing_variants = sorted(set(ABLATION_VARIANTS).difference(by_variant))
    unexpected_variants = sorted(set(by_variant).difference(ABLATION_VARIANTS))
    if missing_variants:
        errors.append(f"missing_variants={missing_variants}")
    if unexpected_variants:
        errors.append(f"unexpected_variants={unexpected_variants}")
    if len(by_variant) != len(records):
        errors.append("duplicate_variant_records")
    train_hashes = {record.get("train_indices_sha256") for record in records}
    eval_hashes = {record.get("eval_indices_sha256") for record in records}
    if len(train_hashes) != 1 or None in train_hashes:
        errors.append("train_indices_not_matched")
    if len(eval_hashes) != 1 or None in eval_hashes:
        errors.append("eval_indices_not_matched")

    topology_mismatches = []
    instantiated_branch_mismatches = []
    invalid_records = []
    cache_violations = []
    for variant, record in by_variant.items():
        if record.get("ablation_settings") != _ablation_settings(variant):
            topology_mismatches.append(variant)
        expected = _ablation_settings(variant)
        expected_instantiated = {
            "image": bool(expected["enable_image"]),
            "spatial": bool(expected["enable_spatial"]),
            "interaction": bool(expected["enable_interaction"]),
        }
        if record.get("instantiated_branches") != expected_instantiated:
            instantiated_branch_mismatches.append(f"{variant}:branches")
        expected_group_order = sorted(expected["spatial_groups"]) if expected["enable_spatial"] else []
        if record.get("spatial_branch_order") != expected_group_order:
            instantiated_branch_mismatches.append(f"{variant}:spatial_groups")
        if record.get("valid") is not True or record.get("prediction_schema_valid") is not True:
            invalid_records.append(variant)
        load = record.get("image_load_audit", {})
        if int(load.get("disk_image_cache_misses", -1)) != 0 or int(load.get("source_image_loads", -1)) != 0:
            cache_violations.append(variant)
        image_enabled = bool(record.get("ablation_settings", {}).get("enable_image"))
        disk_hits = int(load.get("disk_image_cache_hits", -1))
        if image_enabled and disk_hits <= 0:
            cache_violations.append(f"{variant}:image_enabled_without_cache_hits")
        if not image_enabled and disk_hits != 0:
            cache_violations.append(f"{variant}:image_disabled_with_cache_hits")
    if topology_mismatches:
        errors.append(f"topology_mismatches={sorted(topology_mismatches)}")
    if instantiated_branch_mismatches:
        errors.append(f"instantiated_branch_mismatches={sorted(instantiated_branch_mismatches)}")
    if invalid_records:
        errors.append(f"invalid_records={sorted(invalid_records)}")
    if cache_violations:
        errors.append(f"cache_violations={sorted(cache_violations)}")
    if audit.get("paper_facing_result") is not False:
        errors.append("bounded_ablation_must_not_be_paper_facing")
    return {
        "schema_version": "classification_v2_multimodal_ablation_pilot_check_v1",
        "audit_json": str(audit_path),
        "variant_count": int(len(records)),
        "missing_variants": missing_variants,
        "unexpected_variants": unexpected_variants,
        "matched_train_indices": len(train_hashes) == 1 and None not in train_hashes,
        "matched_eval_indices": len(eval_hashes) == 1 and None not in eval_hashes,
        "topology_mismatches": sorted(topology_mismatches),
        "instantiated_branch_mismatches": sorted(instantiated_branch_mismatches),
        "invalid_records": sorted(invalid_records),
        "cache_violations": sorted(cache_violations),
        "paper_facing_result": audit.get("paper_facing_result"),
        "errors": errors,
        "valid": not errors,
    }


if __name__ == "__main__":
    main()
