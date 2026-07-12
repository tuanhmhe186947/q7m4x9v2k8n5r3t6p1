from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Check B4 validation-only seed variance audit.")
    parser.add_argument(
        "--audit-json",
        type=Path,
        default=Path("outputs/classification_v2/model_smoke/b4_seed_variance_cuda/b4_seed_variance_audit.json"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/classification_v2/model_design/b4_seed_variance_check_audit.json"),
    )
    parser.add_argument("--expected-seed-count", type=int, default=3)
    args = parser.parse_args()

    audit = json.loads(args.audit_json.read_text(encoding="utf-8"))
    errors: list[str] = []
    if audit.get("mode") != "execute":
        errors.append(f"mode={audit.get('mode')}")
    if audit.get("valid") is not True:
        errors.append(f"audit_invalid={audit.get('errors')}")
    if audit.get("metric") != "validation_window_macro_f1":
        errors.append(f"metric={audit.get('metric')}")
    if audit.get("outer_test_used_for_threshold_tuning") is not False:
        errors.append("outer_test_used_for_threshold_tuning")
    if audit.get("full_oof_executed") is not False:
        errors.append("full_oof_executed")
    rows = list(audit.get("rows", []))
    if len(rows) != args.expected_seed_count:
        errors.append(f"seed_count={len(rows)}")
    for row in rows:
        seed = row.get("seed")
        if row.get("device") != "cuda":
            errors.append(f"seed_{seed}_device={row.get('device')}")
        if row.get("git", {}).get("dirty") is not False:
            errors.append(f"seed_{seed}_git_dirty={row.get('git', {}).get('dirty')}")
        if row.get("errors"):
            errors.append(f"seed_{seed}_errors={row.get('errors')}")
        if row.get("validation_window_macro_f1") is None:
            errors.append(f"seed_{seed}_missing_validation_metric")
    summary = audit.get("summary", {})
    if summary.get("count") != args.expected_seed_count:
        errors.append(f"summary_count={summary.get('count')}")

    result = {
        "schema_version": "classification_v2_b4_seed_variance_check_audit_v1",
        "audit_json": str(args.audit_json),
        "seed_count": len(rows),
        "summary": summary,
        "errors": errors,
        "valid": not errors,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
