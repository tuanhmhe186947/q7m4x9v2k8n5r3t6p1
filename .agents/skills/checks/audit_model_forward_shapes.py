"""Compare observed forward tensor shapes with a declared model contract."""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import finish, load_json


def _compare_group(
    expected: dict[str, object],
    observed: dict[str, object],
    symbols: dict[str, int],
) -> list[str]:
    errors: list[str] = []
    for name, raw_spec in expected.items():
        spec = dict(raw_spec)
        required = bool(spec.get("required", True))
        if name not in observed:
            if required:
                errors.append(f"missing_tensor={name}")
            continue
        actual = list(observed[name])
        shape = list(spec.get("shape", []))
        if len(actual) != len(shape):
            errors.append(f"rank_mismatch={name}:{shape}!={actual}")
            continue
        for index, (expected_dim, actual_dim) in enumerate(zip(shape, actual)):
            if isinstance(expected_dim, int) and expected_dim != actual_dim:
                errors.append(f"dim_mismatch={name}[{index}]={actual_dim}")
            elif isinstance(expected_dim, str):
                previous = symbols.setdefault(expected_dim, actual_dim)
                if previous != actual_dim:
                    errors.append(f"symbol_mismatch={expected_dim}:{previous}!={actual_dim}")
    return errors


def audit(spec_path: Path, observed_path: Path) -> dict[str, object]:
    """Validate input and output shapes with shared symbolic dimensions."""
    spec = load_json(spec_path)
    observed = load_json(observed_path)
    symbols: dict[str, int] = {}
    errors = _compare_group(spec.get("inputs", {}), observed.get("inputs", {}), symbols)
    errors.extend(
        _compare_group(spec.get("outputs", {}), observed.get("outputs", {}), symbols)
    )
    return {
        "check": "model_forward_shapes",
        "resolved_symbols": symbols,
        "ten_class_output_required": True,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec-json", type=Path, required=True)
    parser.add_argument("--observed-json", type=Path, required=True)
    args = parser.parse_args()
    return finish(audit(args.spec_json, args.observed_json))


if __name__ == "__main__":
    raise SystemExit(main())
