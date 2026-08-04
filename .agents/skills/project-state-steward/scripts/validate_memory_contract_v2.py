"""Project memory validation entrypoint."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_PATH = Path(__file__).with_name("validate_governance_contracts.py")
_SPEC = importlib.util.spec_from_file_location("governance_contracts", _PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

ROOT = _MODULE.ROOT
audit = _MODULE.audit
evaluate_claim = _MODULE.evaluate_claim
validate_observation = _MODULE.validate_observation
_check_short_ttl = _MODULE._check_short_ttl
_check_short_checklist = _MODULE._check_short_checklist
_check_memory_maturity = _MODULE._check_memory_maturity
_valid_method_transition = _MODULE._valid_method_transition


if __name__ == "__main__":
    raise SystemExit(_MODULE.main())
