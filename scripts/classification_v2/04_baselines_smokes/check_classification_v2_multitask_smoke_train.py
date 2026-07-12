from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Check multitask smoke training audit.")
    parser.add_argument(
        "--audit-json",
        type=Path,
        default=Path("outputs/classification_v2/model_smoke/multitask_visual_v3/multitask_smoke_audit.json"),
    )
    args = parser.parse_args()
    audit = json.loads(args.audit_json.read_text(encoding="utf-8"))
    errors = list(audit.get("errors", []))
    if audit.get("valid") is not True:
        errors.append("smoke_audit_not_valid")
    if audit.get("interpretation") != "bounded_trainability_smoke_not_model_quality_evidence":
        errors.append("smoke_interpretation_missing")
    initial = audit.get("initial_losses", {})
    final = audit.get("final_losses", {})
    for name in ["total", "behavior"]:
        if not float(final.get(name, float("inf"))) < float(initial.get(name, float("-inf"))):
            errors.append(f"loss_not_reduced={name}")
    if audit.get("auxiliary_targets_used_as_model_inputs") is not False:
        errors.append("auxiliary_targets_entered_model_inputs")
    visual = audit.get("visual_context_load_audit", {})
    actor = audit.get("actor_image_load_audit", {})
    if int(visual.get("packed_cache_misses", -1)) != 0 or int(visual.get("individual_cache_loads", -1)) != 0:
        errors.append("visual_context_not_strict_packed")
    if int(actor.get("disk_image_cache_misses", -1)) != 0 or int(actor.get("source_image_loads", -1)) != 0:
        errors.append("actor_image_not_strict_packed")
    result = {
        "audit_json": str(args.audit_json),
        "rows": audit.get("rows"),
        "initial_total_loss": initial.get("total"),
        "final_total_loss": final.get("total"),
        "initial_behavior_loss": initial.get("behavior"),
        "final_behavior_loss": final.get("behavior"),
        "errors": errors,
        "valid": not errors,
    }
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
