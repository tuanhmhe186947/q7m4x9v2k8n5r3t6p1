# Classification V2 scientific contract v1

This package is an audit authority, not approval of current code.
`00_pipeline_contract.yaml` is the single source of truth. It is
JSON-compatible YAML so validation requires only Python's standard
library. Derived Markdown, CSV and JSON files are deterministic.

## Scope

- Pipeline stages: 17
- Feature semantics: 73
- Invariants: 29
- Golden numerical cases: 25
- Model schemas: 1
- Contract version: `classification_v2.scientific_contract.v1.1.0`
- Audited code SHA: `d925c9004e7aff5a3c8469b158d2428432c6031a`

Dataset counts in the contract are instance evidence only; they
are never interpreted as universal mathematical semantics.

## Validation

```text
python scripts/classification_v2/scientific_contract/validate_scientific_contract.py
python scripts/classification_v2/scientific_contract/check_code_contract_mapping.py
python scripts/classification_v2/scientific_contract/render_scientific_contract_docs.py --check
```

Any missing, reordered, duplicated or unexpected required tensor
feature is a fail-closed schema violation. Image-coordinate
distances and rates are not physical measurements.

## Current decision

`CURRENT_CODE_SCIENTIFICALLY_APPROVED=NO`. See
`11_known_gaps_and_risks.csv` and
`13_independent_review_report.md`.
