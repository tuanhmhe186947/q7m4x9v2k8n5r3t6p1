"""Normalize archived Jupyter notebooks for GitHub review.

The cleaner intentionally keeps notebook code cells in place while removing
volatile execution state and corrupted non-English notes from comments,
messages, and markdown cells.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

NON_ASCII_RE = re.compile(r"[^\x00-\x7f]")
GENERATED_PLACEHOLDER_RE = re.compile(
    r"Archived experiment note translated to English|"
    r"Notebook processing step completed|"
    r"Notebook assertion failed|"
    r"Notebook validation failed"
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        type=Path,
        nargs="+",
        help="Notebook files or directories to clean.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate notebooks without writing changes.",
    )
    return parser.parse_args()


def iter_notebooks(paths: list[Path]) -> list[Path]:
    """Return sorted notebook paths from files or directories."""
    notebooks: set[Path] = set()
    for path in paths:
        if path.is_dir():
            notebooks.update(path.rglob("*.ipynb"))
        elif path.suffix == ".ipynb":
            notebooks.add(path)
        else:
            raise ValueError(f"Not a notebook or directory: {path}")
    return sorted(notebooks)


def normalize_notebook(notebook: dict[str, Any]) -> dict[str, Any]:
    """Return a normalized notebook object."""
    normalized = {
        "cells": [],
        "metadata": _normalize_top_metadata(notebook.get("metadata", {})),
        "nbformat": int(notebook.get("nbformat", 4)),
        "nbformat_minor": int(notebook.get("nbformat_minor", 5)),
    }

    for cell in notebook.get("cells", []):
        cell_type = str(cell.get("cell_type", "code"))
        normalized_source = _normalize_source(cell_type, cell.get("source", []))
        if not normalized_source:
            continue

        normalized_cell: dict[str, Any] = {
            "cell_type": cell_type,
            "metadata": {},
            "source": normalized_source,
        }
        if cell_type == "code":
            normalized_cell["execution_count"] = None
            normalized_cell["outputs"] = []
        normalized["cells"].append(normalized_cell)

    return normalized


def _normalize_top_metadata(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}

    output: dict[str, Any] = {}
    for key in ("kernelspec", "language_info"):
        value = metadata.get(key)
        if isinstance(value, dict):
            output[key] = value
    return output


def _normalize_source(cell_type: str, source: Any) -> list[str]:
    text = "".join(source) if isinstance(source, list) else str(source or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    if cell_type == "markdown":
        if _has_problem_text(text) or _has_generated_placeholder(text):
            return []
        return text.splitlines(keepends=True)

    return _normalize_code_source(text)


def _normalize_code_source(text: str) -> list[str]:
    lines = text.splitlines(keepends=True)
    output: list[str] = []
    in_triple = False
    triple_token = ""

    for line in lines:
        newline = "\n" if line.endswith("\n") else ""
        body = line[:-1] if newline else line

        if in_triple:
            if not (_has_problem_text(body) or _has_generated_placeholder(body)):
                output.append(line)
            if triple_token in body:
                in_triple = False
                triple_token = ""
            continue

        if not (_has_problem_text(body) or _has_generated_placeholder(body)):
            output.append(line)
            if _starts_unclosed_triple(body):
                in_triple = True
                triple_token = _triple_token(body)
            continue

        normalized_line = _normalize_problem_code_line(body, newline)
        if normalized_line is not None:
            output.append(normalized_line)
        if _starts_unclosed_triple(body):
            in_triple = True
            triple_token = _triple_token(body)

    return output


def _normalize_problem_code_line(body: str, newline: str) -> str | None:
    stripped = body.lstrip()
    indent = body[: len(body) - len(stripped)]

    if not stripped:
        return newline
    if stripped.startswith("#"):
        return None
    if "#" in body:
        before, _comment = body.split("#", 1)
        before = before.rstrip()
        return f"{before}{newline}" if before else None
    if stripped.startswith("assert ") and "," in body:
        condition = body.split(",", 1)[0].rstrip()
        return f"{condition}{newline}"
    if stripped.startswith("print("):
        return f"{indent}pass{newline}"
    if stripped.startswith("raise "):
        match = re.match(r"(\s*raise\s+\w+)\(", body)
        if match:
            return f"{match.group(1)}(){newline}"
    if '"""' in body or "'''" in body:
        return _normalize_triple_quote_line(body, newline)

    ascii_body = NON_ASCII_RE.sub("", body)
    return f"{ascii_body}{newline}"


def _normalize_triple_quote_line(body: str, newline: str) -> str:
    for token in ('"""', "'''"):
        if token not in body:
            continue
        parts = body.split(token)
        if len(parts) >= 3:
            before = parts[0].rstrip()
            after = parts[-1].rstrip()
            return f"{before}{after}{newline}" if before or after else newline
    indent = body.split(body.lstrip(), 1)[0]
    return f"{indent}{newline}"


def _starts_unclosed_triple(body: str) -> bool:
    token = _triple_token(body)
    return bool(token and body.count(token) % 2 == 1)


def _triple_token(body: str) -> str:
    quote3 = body.find('"""')
    apostrophe3 = body.find("'''")
    if quote3 == -1 and apostrophe3 == -1:
        return ""
    if apostrophe3 == -1 or (quote3 != -1 and quote3 < apostrophe3):
        return '"""'
    return "'''"


def _has_problem_text(text: str) -> bool:
    return bool(NON_ASCII_RE.search(text))


def _has_generated_placeholder(text: str) -> bool:
    return bool(GENERATED_PLACEHOLDER_RE.search(text))


def normalize_text(notebook: dict[str, Any]) -> str:
    """Serialize a notebook with stable JSON and LF endings."""
    return json.dumps(
        notebook,
        ensure_ascii=False,
        indent=2,
        sort_keys=False,
    ) + "\n"


def clean_notebook(path: Path, *, check: bool) -> bool:
    """Clean one notebook. Return True when changes were needed."""
    original_text = path.read_text(encoding="utf-8-sig")
    notebook = json.loads(original_text)
    normalized_text = normalize_text(normalize_notebook(notebook))
    changed = original_text.replace("\r\n", "\n").replace("\r", "\n") != normalized_text

    if check:
        if changed:
            print(f"needs cleanup: {path}", file=sys.stderr)
        return changed

    if changed:
        path.write_text(normalized_text, encoding="utf-8", newline="\n")
        print(f"cleaned: {path}")
    return changed


def main() -> int:
    """Run notebook cleanup or validation."""
    args = parse_args()
    try:
        notebooks = iter_notebooks(args.paths)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    changed_count = 0
    for notebook in notebooks:
        changed_count += int(clean_notebook(notebook, check=args.check))

    if args.check and changed_count:
        print(f"{changed_count} notebook(s) need cleanup.", file=sys.stderr)
        return 1
    print(f"Validated {len(notebooks)} notebook(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
