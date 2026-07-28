# B0/B1/R0 Standard-V2 headline update

Date: 20260728

The frozen prediction bytes were re-evaluated under
`TRACKING_EVALUATOR_STANDARD_V2`; no tracker or detector ran.

| Arm | HOTA | DetA | AssA | LocA | IDF1 | IDSW_STANDARD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | 0.849511403 | 0.899665398 | 0.802730886 | 0.920588388 | 0.920646368 | 84 |
| B1 | 0.849873389 | 0.904498739 | 0.799552800 | 0.923907826 | 0.914081197 | 64 |
| R0 | 0.888187232 | 0.899159459 | 0.878060107 | 0.919742139 | 0.971892400 | 29 |

Legacy HOTA/DetA/AssA values remain historical non-standard
diagnostics and must be replaced in current headline reporting.

The old B1 > R0 > B0 headline ordering is not preserved.

B1−B0 is the matched-cadence offline-repair comparison.
R0 comparisons are whole-pipeline effects including detector cadence;
they are not pure association-core estimates.

`000216` remains aggregate-only and is excluded from authoritative
mechanism ranking because its GT authority is unresolved.
