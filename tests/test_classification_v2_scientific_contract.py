from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from pig_behavior.classification_v2.contracts.text_canonicalization import (
    TEXT_CANONICALIZATION_ID,
    TEXT_CANONICALIZATION_VERSION,
    canonical_contract_text_sha256,
    canonicalize_contract_text_bytes,
    text_canonicalization_contract,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_ROOT = (
    PROJECT_ROOT
    / "scripts"
    / "classification_v2"
    / "scientific_contract"
)
CONTRACT = (
    PROJECT_ROOT
    / "docs"
    / "classification_v2"
    / "scientific_contract_v1"
    / "00_pipeline_contract.yaml"
)
MAPPING = CONTRACT.parent / "10_code_contract_mapping.csv"


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


VALIDATOR = _load_module(
    "validate_scientific_contract",
    TOOL_ROOT / "validate_scientific_contract.py",
)
RENDERER = _load_module(
    "render_scientific_contract_docs",
    TOOL_ROOT / "render_scientific_contract_docs.py",
)
MAPPING_CHECKER = _load_module(
    "check_code_contract_mapping",
    TOOL_ROOT / "check_code_contract_mapping.py",
)


def test_scientific_contract_passes_all_machine_checks() -> None:
    result = VALIDATOR.validate_contract(CONTRACT)

    assert result["valid"], result["errors"]
    assert result["counts"]["stages"] >= 17
    assert result["counts"]["features"] >= 58
    assert result["counts"]["invariants"] >= 24
    assert result["counts"]["golden_cases"] == 25


def test_scientific_contract_render_is_deterministic_and_in_sync() -> None:
    first = RENDERER.render_payloads(CONTRACT, PROJECT_ROOT)
    second = RENDERER.render_payloads(CONTRACT, PROJECT_ROOT)

    assert first == second
    assert RENDERER.write_payloads(
        CONTRACT,
        PROJECT_ROOT,
        check=True,
    ) == []


def test_motion_schema_hash_is_deterministic() -> None:
    contract = VALIDATOR.load_contract(CONTRACT)
    schema = contract["model_schemas"][0]

    first = VALIDATOR.schema_hash(schema)
    second = VALIDATOR.schema_hash(json.loads(json.dumps(schema)))

    assert first == second == schema["schema_hash"]
    assert schema["dimension"] == len(schema["ordered_feature_names"])
    assert len(schema["ordered_feature_names"]) == len(
        set(schema["ordered_feature_names"])
    )


def test_code_contract_mapping_has_two_way_coverage() -> None:
    result = MAPPING_CHECKER.check_mapping(
        CONTRACT,
        MAPPING,
        PROJECT_ROOT,
    )

    assert result["valid"], result["errors"]
    assert (
        result["contract_item_count"]
        == result["mapped_contract_item_count"]
    )
    assert result["implementation_inventory_count"] >= 10


def test_golden_cases_have_required_schema_and_independent_numbers() -> None:
    contract = VALIDATOR.load_contract(CONTRACT)
    cases = VALIDATOR.expand_entities(
        contract,
        "golden_cases",
        "golden_case_defaults",
    )

    assert len(cases) == 25
    assert VALIDATOR._golden_errors(contract) == []
    assert {
        "case.direction_change_constant_speed",
        "case.non_square_image_distance",
        "case.roi_none_available",
        "case.row_permutation_neighbor_tie",
        "case.missing_required_exporter_feature",
    }.issubset({case["case_id"] for case in cases})


def test_model_schema_references_only_declared_model_features() -> None:
    contract = VALIDATOR.load_contract(CONTRACT)
    features = VALIDATOR.expand_entities(
        contract,
        "features",
        "feature_defaults",
    )

    assert VALIDATOR._schema_errors(contract, features) == []


def test_contract_text_canonicalization_is_explicit_and_versioned() -> None:
    contract = text_canonicalization_contract()
    assert contract["text_canonicalization_id"] == (
        TEXT_CANONICALIZATION_ID
    )
    assert contract["text_canonicalization_version"] == (
        TEXT_CANONICALIZATION_VERSION
    )
    assert contract == VALIDATOR.load_contract(CONTRACT)[
        "contract_metadata"
    ]["text_canonicalization"]


def test_lf_and_crlf_have_identical_canonical_bytes_and_hashes() -> None:
    logical_lf = b"flowchart TD\n  A --> B\n"
    logical_crlf = logical_lf.replace(b"\n", b"\r\n")
    assert canonicalize_contract_text_bytes(logical_lf) == logical_lf
    assert canonicalize_contract_text_bytes(logical_crlf) == logical_lf
    assert canonical_contract_text_sha256(logical_lf) == (
        canonical_contract_text_sha256(logical_crlf)
    )


def test_contract_text_semantic_difference_and_forbidden_bytes() -> None:
    baseline = b"flowchart TD\n  A --> B\n"
    changed = b"flowchart TD\n  A --> C\n"
    assert canonicalize_contract_text_bytes(baseline) != (
        canonicalize_contract_text_bytes(changed)
    )
    assert canonical_contract_text_sha256(baseline) != (
        canonical_contract_text_sha256(changed)
    )
    with pytest.raises(
        ValueError,
        match="TRAILING_WHITESPACE",
    ):
        canonicalize_contract_text_bytes(b"flowchart TD \n")
    with pytest.raises(
        ValueError,
        match="UTF8_BOM_FORBIDDEN",
    ):
        canonicalize_contract_text_bytes(
            b"\xef\xbb\xbfflowchart TD\n"
        )


def test_renderer_manifest_hash_is_checkout_newline_independent(
    tmp_path: Path,
) -> None:
    metadata = VALIDATOR.load_contract(CONTRACT)["contract_metadata"]
    lf_root = tmp_path / "lf"
    crlf_root = tmp_path / "crlf"
    lf_root.mkdir()
    crlf_root.mkdir()
    name = "01_pipeline_dataflow.mmd"
    logical = b"flowchart TD\n  A --> B\n"
    (lf_root / name).write_bytes(logical)
    (crlf_root / name).write_bytes(logical.replace(b"\n", b"\r\n"))
    lf_manifest = RENDERER._manifest_payload(
        lf_root,
        [name],
        metadata,
    )
    crlf_manifest = RENDERER._manifest_payload(
        crlf_root,
        [name],
        metadata,
    )
    assert lf_manifest == crlf_manifest
    assert lf_manifest["text_canonicalization"] == (
        text_canonicalization_contract()
    )
