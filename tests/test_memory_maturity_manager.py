from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANAGER_PATH = (
    ROOT
    / ".agents"
    / "skills"
    / "project-state-steward"
    / "scripts"
    / "manage_memory_maturity.py"
)


def _load_manager():
    spec = importlib.util.spec_from_file_location(
        "project_memory_maturity_manager",
        MANAGER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_workspace(tmp_path: Path) -> Path:
    memory = tmp_path / ".agents" / "memory"
    memory.mkdir(parents=True)
    (memory / "04_PROJECT_MEMORY_MEDIUM.md").write_text(
        "# Medium Memory\n\n## Active cross-day entries\n\n- None.\n",
        encoding="utf-8",
    )
    (memory / "05_PROJECT_MEMORY_LONG.md").write_text(
        "\n".join(
            [
                "# Long Memory",
                "",
                "<!-- memory-maturity:dossier:start -->",
                "placeholder",
                "<!-- memory-maturity:dossier:end -->",
                "",
                "## Historical",
                "preserved",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _write_json(
        memory / "21_MEMORY_MATURITY.json",
        {
            "schema_version": "pig.memory-maturity.v1",
            "registry_revision": 1,
            "policy": {"elapsed_inactivity_is_evidence": False},
            "dossier": {
                "path": ".agents/memory/05_PROJECT_MEMORY_LONG.md",
                "last_synthesized_at": None,
                "sha256": None,
                "sync_status": "NEEDS_SYNC",
            },
            "entries": [],
        },
    )
    _write_json(
        memory / "18_AUTHORITY_INDEX.json",
        {
            "entries": [
                {
                    "scope": "fixture.current",
                    "current_authority": "evidence.txt",
                }
            ]
        },
    )
    _write_json(memory / "13_METHOD_STATE.json", {"entries": []})
    _write_json(memory / "14_CLAIM_REGISTRY.json", {"claims": []})
    evidence = tmp_path / "evidence.txt"
    evidence.write_text("validated evidence\n", encoding="utf-8")
    return evidence


def _packet(tmp_path: Path, evidence: Path, entry_id: str = "fixture.fact") -> dict:
    digest = _sha256(evidence)
    relative = evidence.relative_to(tmp_path).as_posix()
    return {
        "entry_id": entry_id,
        "knowledge_kind": "project_fact",
        "dossier_section": "data_and_system",
        "title": "Validated fixture fact",
        "summary": "A bounded fact accepted for the fixture project.",
        "scope": "fixture only",
        "reuse_value": "Prevents rediscovery of the fixture contract.",
        "source_medium_entry": "fixture.completed_work",
        "source_refs": [{"path": relative, "sha256": digest}],
        "authority_refs": [{"path": relative}],
        "evidence_refs": [
            {
                "path": relative,
                "sha256": digest,
                "evidence_class": "ARTIFACT_VERIFIED",
            }
        ],
        "limitations": ["Fixture scope only."],
        "invalidation_conditions": ["The evidence content changes."],
        "revalidation_triggers": [
            {
                "type": "file_sha256",
                "path": relative,
                "expected_sha256": digest,
            },
            {
                "type": "authority_scope",
                "scope": "fixture.current",
                "expected_current_authority": "evidence.txt",
            },
        ],
        "supersedes": [],
        "kind_payload": {},
    }


def _accept(module, registry, entry: dict, current: datetime | None = None) -> dict:
    return registry.review(
        entry_id=entry["entry_id"],
        decision="accept",
        reviewer="fixture-reviewer",
        authority="fixture-authority",
        basis="Evidence, scope, and invalidation gates passed.",
        medium_disposition="resolved_removed_from_active",
        independent_review=False,
        expected_revision=entry["revision"],
        expected_sha256=entry["entry_sha256"],
        current=current,
    )["entry"]


def test_completed_fact_promotes_without_time_gate(tmp_path: Path) -> None:
    module = _load_manager()
    evidence = _write_workspace(tmp_path)
    registry = module.MaturityRegistry(tmp_path)
    registry.synthesize()
    registered = registry.register(
        _packet(tmp_path, evidence),
        created_by="fixture-agent",
    )["entry"]

    accepted = _accept(module, registry, registered)
    promoted = registry.promote(
        accepted["entry_id"],
        "fixture-authority",
        accepted["revision"],
        accepted["entry_sha256"],
    )["entry"]

    assert promoted["state"] == "PROMOTED"
    dossier = (tmp_path / module.DOSSIER_RELATIVE).read_text(encoding="utf-8")
    assert "#### Validated fixture fact" in dossier
    assert "No maturity-registry entry" not in dossier.split(
        "### Data and System Architecture",
        maxsplit=1,
    )[1].split("###", maxsplit=1)[0]
    assert registry.audit()["status"] == "PASS"


def test_elapsed_time_never_repairs_missing_evidence(tmp_path: Path) -> None:
    module = _load_manager()
    evidence = _write_workspace(tmp_path)
    registry = module.MaturityRegistry(tmp_path)
    packet = _packet(tmp_path, evidence)
    packet["evidence_refs"] = []
    old = datetime(2020, 1, 1, tzinfo=timezone(timedelta(hours=7)))
    registered = registry.register(
        packet,
        created_by="fixture-agent",
        current=old,
    )["entry"]

    with pytest.raises(module.MaturityError, match="cannot be accepted"):
        _accept(
            module,
            registry,
            registered,
            current=datetime(2030, 1, 1, tzinfo=timezone(timedelta(hours=7))),
        )


def test_changed_evidence_requires_revalidation_and_reopen(tmp_path: Path) -> None:
    module = _load_manager()
    evidence = _write_workspace(tmp_path)
    registry = module.MaturityRegistry(tmp_path)
    registry.synthesize()
    registered = registry.register(
        _packet(tmp_path, evidence),
        created_by="fixture-agent",
    )["entry"]
    accepted = _accept(module, registry, registered)
    promoted = registry.promote(
        accepted["entry_id"],
        "fixture-authority",
        accepted["revision"],
        accepted["entry_sha256"],
    )["entry"]

    evidence.write_text("changed evidence\n", encoding="utf-8")
    scan = registry.scan()
    assert scan["revalidation_entry_ids"] == [promoted["entry_id"]]
    assert any(
        error.startswith("maturity_revalidation_required")
        for error in registry.audit()["errors"]
    )

    reopened = registry.reopen(
        promoted["entry_id"],
        "fixture-authority",
        "The bound artifact changed.",
        promoted["revision"],
        promoted["entry_sha256"],
    )["entry"]
    dossier = (tmp_path / module.DOSSIER_RELATIVE).read_text(encoding="utf-8")
    assert reopened["state"] == "REVALIDATION_REQUIRED"
    assert "#### Validated fixture fact" not in dossier
    assert "`fixture.fact`: Validated fixture fact" in dossier

    revised = registry.revise(
        reopened["entry_id"],
        _packet(tmp_path, evidence),
        "fixture-authority",
        "Bind the changed evidence after revalidation.",
        reopened["revision"],
        reopened["entry_sha256"],
    )["entry"]
    assert revised["state"] == "REVALIDATION_REQUIRED"
    assert len(revised["historical_acceptances"]) == 1
    assert len(revised["historical_promotions"]) == 1
    accepted_again = _accept(module, registry, revised)
    registry.promote(
        accepted_again["entry_id"],
        "fixture-authority",
        accepted_again["revision"],
        accepted_again["entry_sha256"],
    )
    assert registry.audit()["status"] == "PASS"


def test_stale_review_cannot_overwrite_newer_review(tmp_path: Path) -> None:
    module = _load_manager()
    evidence = _write_workspace(tmp_path)
    registry = module.MaturityRegistry(tmp_path)
    registered = registry.register(
        _packet(tmp_path, evidence),
        created_by="fixture-agent",
    )["entry"]
    _accept(module, registry, registered)

    with pytest.raises(module.MaturityError, match="changed after it was inspected"):
        _accept(module, registry, registered)


def test_supported_claim_kind_requires_supported_claim_registry(
    tmp_path: Path,
) -> None:
    module = _load_manager()
    evidence = _write_workspace(tmp_path)
    registry = module.MaturityRegistry(tmp_path)
    packet = _packet(tmp_path, evidence, "fixture.claim")
    packet.update(
        knowledge_kind="supported_claim",
        dossier_section="supported_findings",
        kind_payload={"claim_id": "MISSING-CLAIM"},
    )
    registered = registry.register(packet, created_by="fixture-agent")["entry"]

    with pytest.raises(module.MaturityError, match="claim_not_supported"):
        registry.review(
            registered["entry_id"],
            "accept",
            "independent-reviewer",
            "fixture-authority",
            "Claim review.",
            "resolved_removed_from_active",
            True,
            registered["revision"],
            registered["entry_sha256"],
        )


def test_validated_correction_requires_reuse_boundaries(tmp_path: Path) -> None:
    module = _load_manager()
    evidence = _write_workspace(tmp_path)
    registry = module.MaturityRegistry(tmp_path)
    packet = _packet(tmp_path, evidence, "fixture.correction")
    packet.update(
        knowledge_kind="validated_correction",
        dossier_section="corrective_methods",
        kind_payload={
            "root_cause": "A stale source was treated as current.",
            "validated_correction": "Resolve authority before retrieval.",
            "reuse_when": "Authority can drift.",
        },
    )
    registered = registry.register(packet, created_by="fixture-agent")["entry"]

    with pytest.raises(module.MaturityError, match="do_not_reuse_when"):
        _accept(module, registry, registered)


def test_dossier_hash_detects_out_of_band_edit(tmp_path: Path) -> None:
    module = _load_manager()
    _write_workspace(tmp_path)
    registry = module.MaturityRegistry(tmp_path)
    registry.synthesize()
    dossier = tmp_path / module.DOSSIER_RELATIVE
    dossier.write_text(
        dossier.read_text(encoding="utf-8") + "out of band\n",
        encoding="utf-8",
    )

    assert "maturity_dossier_hash_mismatch" in registry.audit()["errors"]


def test_two_process_reviews_cannot_lose_an_update(tmp_path: Path) -> None:
    module = _load_manager()
    evidence = _write_workspace(tmp_path)
    registry = module.MaturityRegistry(tmp_path)
    registered = registry.register(
        _packet(tmp_path, evidence),
        created_by="fixture-agent",
    )["entry"]
    command = [
        sys.executable,
        str(MANAGER_PATH),
        "--root",
        str(tmp_path),
        "review",
        "--entry-id",
        registered["entry_id"],
        "--decision",
        "accept",
        "--reviewer",
        "fixture-reviewer",
        "--authority",
        "fixture-authority",
        "--basis",
        "Concurrent evidence review.",
        "--medium-disposition",
        "resolved_removed_from_active",
        "--expected-revision",
        str(registered["revision"]),
        "--expected-entry-sha256",
        registered["entry_sha256"],
    ]
    processes = [
        subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for _ in range(2)
    ]
    outputs = [process.communicate(timeout=20) for process in processes]

    assert sorted(process.returncode for process in processes) == [0, 1]
    assert any("stale_entry_cas" in output for output, _ in outputs)
    assert registry.inspect(registered["entry_id"])["entry"]["state"] == "ACCEPTED"


def test_promotion_blocks_a_source_still_active_in_medium(tmp_path: Path) -> None:
    module = _load_manager()
    evidence = _write_workspace(tmp_path)
    medium = tmp_path / module.MEDIUM_RELATIVE
    medium.write_text(
        "# Medium Memory\n\n"
        "## Active cross-day entries\n\n"
        "- `fixture.completed_work`\n"
        "  - Status: `DONE`.\n",
        encoding="utf-8",
    )
    registry = module.MaturityRegistry(tmp_path)
    registered = registry.register(
        _packet(tmp_path, evidence),
        created_by="fixture-agent",
    )["entry"]
    accepted = _accept(module, registry, registered)

    with pytest.raises(module.MaturityError, match="still active in file 04"):
        registry.promote(
            accepted["entry_id"],
            "fixture-authority",
            accepted["revision"],
            accepted["entry_sha256"],
        )
