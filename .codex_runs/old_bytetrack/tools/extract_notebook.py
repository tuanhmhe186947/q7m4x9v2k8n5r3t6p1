"""Extract code cells from a Jupyter notebook into a Python file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path, help="Input .ipynb file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output .py file. Defaults to <notebook stem>_source.py.",
    )
    return parser.parse_args()


def extract_code(notebook_path: Path, output_path: Path) -> int:
    """Write all code cells from a notebook to a Python script."""
    with notebook_path.open("r", encoding="utf-8") as file:
        notebook: dict[str, Any] = json.load(file)

    cells = notebook.get("cells", [])
    code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]

    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for index, cell in enumerate(code_cells):
            source = cell.get("source", [])
            output.write(f"# %% Cell {index}\n")
            output.write("".join(source))
            output.write("\n\n")

    return len(code_cells)


def main() -> int:
    args = parse_args()
    output_path = args.output or args.notebook.with_name(
        f"{args.notebook.stem}_source.py"
    )
    count = extract_code(args.notebook, output_path)
    print(f"Extracted {count} code cells to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
