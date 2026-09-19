from __future__ import annotations

from pathlib import Path

import pytest

from pig_behavior.classification_v2.contracts.runtime_dependencies import (
    audit_all_stage_runtime_dependencies,
    resolve_runtime_dependency_closure,
    stage_runtime_dependency_audit,
)
from pig_behavior.classification_v2.contracts.semantic_lineage import (
    compute_stage_code_hash,
    load_code_contract_mapping,
    load_scientific_contract,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    REPO_ROOT
    / "docs"
    / "classification_v2"
    / "scientific_contract_v1"
    / "00_pipeline_contract.yaml"
)
MAPPING_PATH = CONTRACT_PATH.with_name("10_code_contract_mapping.csv")
PACKAGE_ROOT = Path("src/pig_behavior/classification_v2")


def _write_module(repo_root: Path, relative: str, source: str = "") -> str:
    path = repo_root / PACKAGE_ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path.relative_to(repo_root).as_posix()


def _mapping(stage_id: str, source: str, symbol: str = "") -> dict[str, str]:
    return {
        "contract_item_id": stage_id,
        "contract_item_type": "stage",
        "source_file": source,
        "symbol": symbol,
    }


def test_import_forms_packages_recursion_cycles_and_inactive_imports(
    tmp_path: Path,
) -> None:
    package = _write_module(tmp_path, "__init__.py")
    entry = _write_module(
        tmp_path,
        "entry.py",
        "\n".join(
            [
                "from typing import TYPE_CHECKING",
                "from . import alpha as alpha_alias",
                "from . import beta",
                "import pig_behavior.classification_v2.gamma as gamma_alias",
                "if TYPE_CHECKING:",
                "    from . import missing_type_checking",
                "if False:",
                "    from . import missing_inactive",
            ]
        ),
    )
    alpha = _write_module(tmp_path, "alpha.py", "from . import beta\n")
    beta = _write_module(tmp_path, "beta.py", "from . import alpha\n")
    gamma = _write_module(tmp_path, "gamma.py")

    result = resolve_runtime_dependency_closure(tmp_path, [entry])

    assert result["runtime_dependency_closure"] == sorted(
        [package, entry, alpha, beta, gamma]
    )
    assert result["missing_production_files"] == []
    assert result["unresolved_local_imports"] == []
    assert result["unresolved_dynamic_imports"] == []


def test_unresolved_dynamic_import_and_missing_entry_fail_closed(
    tmp_path: Path,
) -> None:
    entry = _write_module(
        tmp_path,
        "entry.py",
        "import importlib\nimportlib.import_module(module_name)\n",
    )
    rows = [_mapping("stage.test", entry, "run")]
    dynamic = stage_runtime_dependency_audit(
        tmp_path,
        "stage.test",
        rows,
    )
    assert dynamic["status"] == "FAIL"
    assert dynamic["unresolved_dynamic_imports"]

    missing = stage_runtime_dependency_audit(
        tmp_path,
        "stage.test",
        [_mapping("stage.test", f"{PACKAGE_ROOT.as_posix()}/missing.py", "run")],
    )
    assert missing["status"] == "FAIL"
    assert missing["missing_production_files"]


def test_stage_without_code_authority_fails_closed(tmp_path: Path) -> None:
    result = stage_runtime_dependency_audit(tmp_path, "stage.none", [])
    assert result["status"] == "FAIL_NO_CODE_AUTHORITY"


def test_dependency_mapping_mutation_gate_and_stage_hash(
    tmp_path: Path,
) -> None:
    package = _write_module(tmp_path, "__init__.py")
    dependency = _write_module(tmp_path, "dependency.py", "VALUE = 1\n")
    entry = _write_module(
        tmp_path,
        "entry.py",
        "from .dependency import VALUE\n\ndef run():\n    return VALUE\n",
    )
    rows = [
        _mapping("stage.test", package),
        _mapping("stage.test", entry, "run"),
    ]
    failed = stage_runtime_dependency_audit(
        tmp_path,
        "stage.test",
        rows,
    )
    assert failed["status"] == "FAIL"
    assert failed["missing_dependencies"] == [dependency]
    with pytest.raises(ValueError, match="authority incomplete"):
        compute_stage_code_hash(tmp_path, "stage.test", rows)

    rows.append(_mapping("stage.test", dependency))
    passing = stage_runtime_dependency_audit(
        tmp_path,
        "stage.test",
        rows,
    )
    assert passing["status"] == "PASS"
    first_hash = compute_stage_code_hash(tmp_path, "stage.test", rows)
    (tmp_path / dependency).write_text("VALUE = 2\n", encoding="utf-8")
    second_hash = compute_stage_code_hash(tmp_path, "stage.test", rows)
    assert first_hash != second_hash


def test_all_exact_contract_stages_have_complete_runtime_authority() -> None:
    contract = load_scientific_contract(CONTRACT_PATH)
    stage_ids = [str(stage["stage_id"]) for stage in contract["stages"]]
    result = audit_all_stage_runtime_dependencies(
        REPO_ROOT,
        stage_ids,
        load_code_contract_mapping(MAPPING_PATH),
    )
    assert len(stage_ids) == 17
    assert result["stage_count"] == 17
    assert result["unmapped_production_dependencies"] == 0
    assert result["unresolved_dynamic_import_count"] == 0
    assert result["status"] == "PASS"
