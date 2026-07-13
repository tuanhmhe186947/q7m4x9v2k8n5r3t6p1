from __future__ import annotations

from pathlib import Path

import pytest

from pig_behavior.classification_v2.sources.cvat_tracking_xml import (
    load_cvat_tracking_xml,
)


def _write_hidden_cvat_xml(path: Path) -> None:
    path.write_text(
        """<annotations>
  <meta>
    <task>
      <id>1</id>
      <name>fixture_video</name>
      <size>1</size>
      <original_size><width>1280</width><height>720</height></original_size>
    </task>
  </meta>
  <track id="1" label="ID_1" source="manual">
    <box frame="0" outside="0" xtl="10" ytl="20" xbr="110" ybr="80">
      <attribute name="ID">ID_1</attribute>
      <attribute name="Behavior">stand</attribute>
      <attribute name="Hidden">Yes</attribute>
    </box>
  </track>
</annotations>
""",
        encoding="utf-8",
    )


def test_tracking_hidden_is_kept_without_automatic_downweight(
    tmp_path: Path,
) -> None:
    xml_path = tmp_path / "fixture.xml"
    _write_hidden_cvat_xml(xml_path)

    rows = load_cvat_tracking_xml(xml_path)

    assert len(rows) == 1
    assert rows.iloc[0]["hidden"] == "Yes"
    assert bool(rows.iloc[0]["include_in_training"])
    assert rows.iloc[0]["sample_weight"] == 1.0
    assert rows.iloc[0]["qa_status"] == "ok"


def test_parser_flag_cannot_claim_human_hidden_trust(tmp_path: Path) -> None:
    xml_path = tmp_path / "fixture.xml"
    _write_hidden_cvat_xml(xml_path)

    with pytest.raises(
        ValueError,
        match="cannot be trusted by parser flag",
    ):
        load_cvat_tracking_xml(xml_path, trust_hidden=True)
