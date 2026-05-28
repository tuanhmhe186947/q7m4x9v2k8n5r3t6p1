# Reproducibility

## Environment

Recommended baseline:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e .
```

For full research workflows:

```bash
pip install -r requirements-dev.txt
```

## Data

Place raw images at:

```text
data/raw/images_clean/
```

Keep the processed CSV at:

```text
data/processed/behavior_with_feats_rectROI.csv
```

The default split seed is `42`.

## Minimal Verification

```bash
ruff check src main.py tools tests
pytest -q
python main.py --mode train --dry-run
```

The dry run still requires the local image dataset.

## Public Release Checklist

- Add code license.
- Add data license and dataset download instructions.
- Add author metadata and citation information.
- Record final training command, commit hash, random seed, and hardware.
- Publish model artifacts outside the Git history and document checksums.
