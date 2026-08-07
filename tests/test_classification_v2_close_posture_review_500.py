"""Focused regression tests for post-review closure artifact semantics."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _closure_module():
    root = Path(__file__).resolve().parents[1]
    path = (
        root
        / "scripts"
        / "classification_v2"
        / "02_train_ready_exports"
        / "close_posture_review_500.py"
    )
    spec = importlib.util.spec_from_file_location("close_posture_review_500", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exclude_is_an_eligibility_status_not_a_second_split_role() -> None:
    """Changed invariant: exclude|validation has one model role.

    Prior output falsely reported role crossings because it counted the
    eligibility token as a second role.  The smallest frozen-format input must
    resolve to validation plus exclude metadata.  Authority produced: the
    posture gold-table split binding used by the completed-authority audit.
    """
    module = _closure_module()

    roles, statuses = module.normalize_role_values(["exclude|validation"])

    assert roles == ["validation"]
    assert statuses == ["exclude"]


def test_registered_modal_mapping_is_strong_without_creating_targets() -> None:
    """Changed invariant: a registered modal mapping may be non-deterministic.

    Prior evidence categorized the completed lying and sitting review evidence
    as mixed.  The smallest diagnostic counts preserve counterexamples while
    recording a unique registered modal mapping.  Expected output is the
    strong-but-not-deterministic category.  Authority produced: the empirical
    mapping audit; it remains diagnostic only and creates no label.
    """
    module = _closure_module()

    category = module.mapping_evidence_category(
        200,
        {
            "lying": 185,
            "sitting": 14,
            "upright": 1,
            "unresolved": 0,
            "exclude": 0,
        },
        "lying",
    )

    assert category == "strong_but_not_deterministic"
