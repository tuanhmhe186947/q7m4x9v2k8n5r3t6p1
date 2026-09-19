# Guide to Reproducing Corrected Paper Results

All quantitative tables, figures, and statistical tests in the paper can be reproduced
from the committed physical artifacts without re-running long GPU training:

1. **Verify Master Paper Ledger and Claim Numbers**:
   `cmd
   python scripts/paper/check_claim_numbers.py
   `

2. **Inspect Canonical Behavior 5-Fold Cross-Validation Table**:
   `cmd
   type docs\paper\final_behavior_cv_table.md
   `

3. **Inspect Tracking Confirmatory Authority (CONFIRMATORY_B)**:
   `cmd
   type outputs\paper_confirmatory\tracking\FINAL_TRACKING_CONFIRMATORY_AUTHORITY.json
   `

4. **Verify Artifact Hashes**:
   `cmd
   type docs\paper\freeze_corrected_20260917\artifact_hashes.csv
   `
