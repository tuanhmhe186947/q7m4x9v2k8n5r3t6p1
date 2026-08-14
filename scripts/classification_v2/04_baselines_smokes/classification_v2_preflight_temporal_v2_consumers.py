"""Load one Temporal-v2 target through each final CPU-only consumer route."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from pig_behavior.classification_v2.training.post_s1_resolution_screening import (
    load_canonical_resolution_temporal_target,
)
from pig_behavior.classification_v2.training.pre_s1_rgb_binding import (
    load_canonical_s1_temporal_target,
)
from pig_behavior.classification_v2.training.temporal_v2_consumer import (
    FULL,
    MATCHED,
    VIEWS,
    TemporalV2ConsumerInput,
    audit_resolution_parity,
    verify_registered_canonical_authority,
)


def parse_args() -> argparse.Namespace:
    """Accept only canonical selection parameters; no historical selector exists."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority-root", required=True, type=Path)
    parser.add_argument("--canonical-mapping", required=True, type=Path)
    parser.add_argument("--frame-offset-index", required=True, type=Path)
    parser.add_argument("--corpus", required=True, choices=(MATCHED, FULL))
    parser.add_argument("--view", required=True, choices=tuple(sorted(VIEWS)))
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _record(value: TemporalV2ConsumerInput) -> dict[str, object]:
    target = value.target
    return {
        "consumer": value.consumer,
        "input_resolution": value.input_resolution,
        "membership_source": value.membership_source,
        "historical_selectors_reachable": value.historical_selectors_reachable,
        "corpus": target.corpus,
        "view": target.view,
        "target_id": target.target_id,
        "frames": list(target.frames),
        "behavior": target.behavior,
        "split": target.split,
        "group": target.group,
        "observed_mask": list(target.observed_mask),
        "boundary_reset": target.boundary_reset,
        "temporal_feature_authority": target.provenance["authority_sha256"],
    }


def main() -> None:
    """Exercise S1 and every resolution route without opening media or a model."""

    args = parse_args()
    verified = verify_registered_canonical_authority(
        args.authority_root,
        mapping_path=args.canonical_mapping,
    )
    common = {
        "authority_root": args.authority_root,
        "corpus": args.corpus,
        "view": args.view,
        "target_id": args.target_id,
        "verify_hashes": False,
        "frame_offset_index": args.frame_offset_index,
    }
    s1 = load_canonical_s1_temporal_target(**common)
    resolutions = [
        load_canonical_resolution_temporal_target(
            input_resolution=resolution,
            **common,
        )
        for resolution in (64, 128, 160)
    ]
    result = {
        "status": "PASS",
        "s1": _record(s1),
        "resolution": [_record(value) for value in resolutions],
        "resolution_parity": audit_resolution_parity(
            [value.target for value in resolutions]
        ),
        "registered_authority": verified,
    }
    if args.output is not None:
        _write_json_atomic(args.output, result)
    print(json.dumps(result, sort_keys=True))


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    """Persist a generated preflight receipt without overwriting an authority."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SystemExit(f"preflight output already exists={path}")
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as handle:
        json.dump(value, handle, sort_keys=True)
        temporary = Path(handle.name)
    os.replace(temporary, path)


if __name__ == "__main__":
    main()
