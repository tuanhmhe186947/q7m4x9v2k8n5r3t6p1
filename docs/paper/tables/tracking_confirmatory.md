# Tracking Confirmatory Evaluation (12 Held-Out Videos)

### Provenance Classification: CONFIRMATORY_B
Re-evaluated, verified, and certified physical August 19–20, 2026 execution artifacts under `TRACKING_EVALUATOR_STANDARD_V2`. Zero post-hoc parameter or threshold tuning.

| Method | Role | Provenance | Videos | Frames | Agg HOTA | Mean HOTA [95% CI] | Agg IDF1 [95% CI] | IDSW | Exposure Matched (%) | Exposure All-GT (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RealTime-Fast** | Proposed Causal Online | CONFIRMATORY_B | 12 | 21,600 | 90.33% | 89.89% [84.13%, 94.43%] | 95.03% [89.49%, 99.16%] | 39 | 23,029 (13.36%) | 23,029 (13.33%) |
| **Hybrid-ByteTrack** | Proposed Retrospective Offline | CONFIRMATORY_B | 12 | 21,600 | **93.55%** | **93.47%** [89.85%, 96.24%] | **98.41%** [96.26%, 99.70%] | **8** | **8,597 (4.99%)** | **8,597 (4.98%)** |
| **Raw ByteTrack** | Frozen Technical Baseline | CONFIRMATORY_B | 12 | 21,600 | 89.41% | 89.11% [83.21%, 94.03%] | 94.23% [89.61%, 97.96%] | 64 | 28,603 (16.65%) | 28,603 (16.55%) |

### Paired Differences & Bootstrap Uncertainty (B = 10,000, seed 240494961)
- **Hybrid-ByteTrack vs. Raw Baseline**:
  - $\Delta\text{HOTA} = +4.36\%$ [95% CI: $+1.73\%, +7.31\%$], Bootstrap Positive Fraction = 0.9999
  - $\Delta\text{IDF1} = +4.18\%$ [95% CI: $+1.43\%, +7.29\%$], Bootstrap Positive Fraction = 0.9997
  - IDSW reduction: $-87.5\%$ ($64 \to 8$).
- **Hybrid-ByteTrack vs. RealTime-Fast**:
  - $\Delta\text{HOTA} = +3.58\%$ [95% CI: $+0.55\%, +7.14\%$], Bootstrap Positive Fraction = 0.9920
  - $\Delta\text{IDF1} = +3.38\%$ [95% CI: $+0.30\%, +7.16\%$], Bootstrap Positive Fraction = 0.9896
  - IDSW reduction: $-79.5\%$ ($39 \to 8$).
- **RealTime-Fast vs. Raw Baseline**:
  - $\Delta\text{HOTA} = +0.78\%$ [95% CI: $-2.93\%, +4.53\%$], Bootstrap Positive Fraction = 0.6602
  - $\Delta\text{IDF1} = +0.80\%$ [95% CI: $-2.90\%, +4.69\%$], Bootstrap Positive Fraction = 0.6611
  - IDSW reduction: $-39.1\%$ ($64 \to 39$) with causal online processing.
