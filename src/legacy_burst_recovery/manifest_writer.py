from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_manifests(output_root: Path, manifests: dict[str, pd.DataFrame]) -> None:
    for filename, df in manifests.items():
        write_csv(df, output_root / filename)

