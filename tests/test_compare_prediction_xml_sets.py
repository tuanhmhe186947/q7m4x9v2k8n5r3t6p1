from __future__ import annotations

from pathlib import Path

from scripts.tracking.compare_prediction_xml_sets import (
    compare_prediction_sets,
)


def _xml(
    path: Path,
    *,
    created: str,
    identity: str = "1",
    hidden: str = "No",
    xtl: str = "1.0",
) -> None:
    path.write_text(
        "\n".join(
            (
                "<annotations>",
                "  <meta><task>",
                "    <name>video</name>",
                "    <size>1</size>",
                f"    <created>{created}</created>",
                "  </task></meta>",
                '  <track id="0" label="pig">',
                (
                    f'    <box frame="0" xtl="{xtl}" ytl="2.0" '
                    'xbr="11.0" ybr="12.0" outside="0" occluded="0">'
                ),
                f'      <attribute name="ID">{identity}</attribute>',
                f'      <attribute name="Hidden">{hidden}</attribute>',
                "    </box>",
                "  </track>",
                "</annotations>",
                "",
            )
        ),
        encoding="utf-8",
    )


def test_metadata_only_difference_is_not_scientific(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    _xml(first / "video.xml", created="one")
    _xml(second / "video.xml", created="two")

    aggregate = compare_prediction_sets(first, second)["aggregate"]

    assert not aggregate["byte_equal"]
    assert aggregate["canonical_content_equal"]
    assert aggregate["metadata_or_non_scientific_only_difference"]


def test_scientific_differences_are_counted(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    _xml(first / "video.xml", created="same")
    _xml(
        second / "video.xml",
        created="same",
        identity="2",
        hidden="Yes",
        xtl="3.0",
    )

    aggregate = compare_prediction_sets(first, second)["aggregate"]

    assert aggregate["identity_differences"] == 1
    assert aggregate["hidden_state_differences"] == 1
    assert aggregate["bbox_exact_differences"] == 1
    assert aggregate["bbox_tolerance_violations"] == 1
    assert not aggregate["canonical_content_equal"]
