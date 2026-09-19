from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Write classification_v2 model readiness report.")
    parser.add_argument("--root", type=Path, default=Path("outputs/classification_v2/train_ready_windows"))
    parser.add_argument("--classification-root", type=Path, default=Path("outputs/classification_v2"))
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    root = args.root
    classification_root = args.classification_root
    summary = {
        "claim_boundary": {
            "target": "Q2-strong: improved pig behavior recognition under session/video-safe validation.",
            "not_claimed": "Q1 external farm/camera/cohort generalization.",
            "identity_scope": "pig_id is an annotation/track ID within a video/session, not biological identity.",
        },
        "contract": _load_json(root / "model_input_contract.json"),
        "blueprint": _load_json(root / "model_upgrade_blueprint.json"),
        "image_context": _load_json(root / "image_context_index_audit.json"),
        "image_tensor_loader": _load_json(root / "image_tensor_loader_smoke_audit.json"),
        "multimodal_forward": _load_json(classification_root / "model_smoke" / "multimodal_forward_smoke_audit.json"),
        "multimodal_smoke_train": _load_json(
            classification_root / "model_smoke" / "multimodal_smoke_train" / "multimodal_smoke_train_audit.json"
        ),
        "interaction_context": _load_json(root / "interaction_context_audit.json"),
        "auxiliary_targets": _load_json(root / "auxiliary_targets_audit.json"),
    }
    summary["pass_fail"] = _pass_fail(summary)

    output_md = args.output_md or (root / "model_readiness_report.md")
    output_json = args.output_json or (root / "model_readiness_report.json")
    output_md.write_text(_render_markdown(summary), encoding="utf-8")
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output_md": str(output_md), "output_json": str(output_json), "pass_fail": summary["pass_fail"]}))
    if summary["pass_fail"]["status"] != "PASS":
        raise SystemExit(1)


def _pass_fail(summary: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    for key in [
        "contract",
        "image_context",
        "image_tensor_loader",
        "multimodal_forward",
        "multimodal_smoke_train",
        "interaction_context",
        "auxiliary_targets",
    ]:
        errors = summary.get(key, {}).get("errors", [])
        if errors:
            failures.append(f"{key}_errors={errors}")
    if summary.get("contract", {}).get("missing_artifacts"):
        failures.append("contract_missing_artifacts")
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "caveats": [
            "PASS means smoke/data-contract readiness, not full training completion.",
            "Use session/video-safe validation; do not claim external generalization without external data.",
            "Interaction claims must account for context-ready subset versus crop-only legacy rows.",
        ],
    }


def _render_markdown(summary: dict[str, Any]) -> str:
    pass_fail = summary["pass_fail"]
    interaction = summary.get("interaction_context", {})
    aux = summary.get("auxiliary_targets", {})
    forward = summary.get("multimodal_forward", {})
    smoke = summary.get("multimodal_smoke_train", {})
    blueprint = summary.get("blueprint", {})
    lines = [
        "# classification_v2 Model Readiness Report",
        "",
        f"PASS/FAIL: **{pass_fail['status']}**",
        "",
        "## Claim Boundary",
        f"- Target: {summary['claim_boundary']['target']}",
        f"- Not claimed: {summary['claim_boundary']['not_claimed']}",
        f"- Identity scope: {summary['claim_boundary']['identity_scope']}",
        "",
        "## Implemented Evidence",
        f"- Multimodal forward logits: {forward.get('logit_shape')}; "
        f"mask delta: {forward.get('max_masked_padding_delta')}",
        f"- Multimodal smoke train rows: {smoke.get('train_rows')} train / {smoke.get('eval_rows')} eval",
        f"- Multimodal smoke loss reduction: {smoke.get('loss_reduction')}",
        f"- Interaction windows: {interaction.get('interaction_window_rows')}; "
        f"ready: {interaction.get('interaction_ready_rows')}",
        f"- Auxiliary target rows: {aux.get('rows')}; positives: {aux.get('aux_target_positive_counts')}",
        f"- Blueprint phases: {len(blueprint.get('training_phases', []))}",
        "",
        "## Checklist",
        "- Data contract and leakage forbidden inputs recorded.",
        "- Image tensor loader smoke covers CVAT video+bbox and legacy crops.",
        "- Multimodal image+spatial forward smoke passes.",
        "- Tiny split-safe multimodal smoke train passes; not a benchmark.",
        "- Interaction full-frame/partner context audit is available.",
        "- Auxiliary multi-task targets are y/mask artifacts only.",
        "",
        "## Remaining Gates Before Full Training",
        "- Add interaction full-frame/partner visual branch before strong interaction claims.",
        "- Keep source-balanced and video/session-safe validation as the main reported metric.",
        "- Collapse window predictions to native temporal/review units for confirmatory evaluation.",
        "- Treat source shortcut as a known risk and report controls.",
    ]
    if pass_fail["failures"]:
        lines.extend(["", "## Failures", *[f"- {failure}" for failure in pass_fail["failures"]]])
    return "\n".join(lines) + "\n"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"errors": [f"missing_json={path}"]}
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
