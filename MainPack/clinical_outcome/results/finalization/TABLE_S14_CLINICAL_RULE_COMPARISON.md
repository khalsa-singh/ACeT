# Supplementary Table S14 | Held-out internal performance of ACeT and fixed single-assay and multi-flag clinical-outcome rules

| Rule family | Rule | Threshold or flag requirement | Approvals retained, n/13 | Terminations flagged, n/10 | Approval-retention rate | Termination-detection rate | Balanced accuracy | MCC | Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Published single-assay | BVP | > 4.3 | 13/13 | 5/10 | 1.000 | 0.500 | 0.750 | 0.601 | 0.783 |
| Published single-assay | AS | > 0.08 | 13/13 | 4/10 | 1.000 | 0.400 | 0.700 | 0.523 | 0.739 |
| Published single-assay | ELISA | > 1.9 | 12/13 | 4/10 | 0.923 | 0.400 | 0.662 | 0.388 | 0.696 |
| Published single-assay | AC-SINS | > 11.8 | 12/13 | 3/10 | 0.923 | 0.300 | 0.612 | 0.292 | 0.652 |
| Published single-assay | PSR | > 0.27 | 13/13 | 4/10 | 1.000 | 0.400 | 0.700 | 0.523 | 0.739 |
| Training-optimized single-assay | BVP | > 3.1572 | 11/13 | 5/10 | 0.846 | 0.500 | 0.673 | 0.373 | 0.696 |
| Training-optimized single-assay | AS | > 0.05 | 10/13 | 4/10 | 0.769 | 0.400 | 0.585 | 0.182 | 0.609 |
| Training-optimized single-assay | ELISA | > 1.826 | 12/13 | 4/10 | 0.923 | 0.400 | 0.662 | 0.388 | 0.696 |
| Training-optimized single-assay | AC-SINS | > 11.2 | 12/13 | 3/10 | 0.923 | 0.300 | 0.612 | 0.292 | 0.652 |
| Training-optimized single-assay | PSR | > 0.0 | 10/13 | 7/10 | 0.769 | 0.700 | 0.735 | 0.469 | 0.739 |
| Published multi-flag heuristic | At least 1 of 5 published flags | >= 1 flag | 11/13 | 7/10 | 0.846 | 0.700 | 0.773 | 0.555 | 0.783 |
| Published multi-flag heuristic | At least 2 of 5 published flags | >= 2 flags | 13/13 | 4/10 | 1.000 | 0.400 | 0.700 | 0.523 | 0.739 |
| Published multi-flag heuristic | At least 3 of 5 published flags | >= 3 flags | 13/13 | 4/10 | 1.000 | 0.400 | 0.700 | 0.523 | 0.739 |
| Published multi-flag heuristic | At least 4 of 5 published flags | >= 4 flags | 13/13 | 4/10 | 1.000 | 0.400 | 0.700 | 0.523 | 0.739 |
| Published multi-flag heuristic | All 5 published flags | 5 flags | 13/13 | 1/10 | 1.000 | 0.100 | 0.550 | 0.243 | 0.609 |
| Training-optimized multi-flag heuristic | At least 1 of 5 training-optimized flags | >= 1 flag | 6/13 | 8/10 | 0.462 | 0.800 | 0.631 | 0.272 | 0.609 |
| Training-optimized multi-flag heuristic | At least 2 of 5 training-optimized flags | >= 2 flags | 10/13 | 6/10 | 0.769 | 0.600 | 0.685 | 0.375 | 0.696 |
| Training-optimized multi-flag heuristic | At least 3 of 5 training-optimized flags | >= 3 flags | 13/13 | 4/10 | 1.000 | 0.400 | 0.700 | 0.523 | 0.739 |
| Training-optimized multi-flag heuristic | At least 4 of 5 training-optimized flags | >= 4 flags | 13/13 | 4/10 | 1.000 | 0.400 | 0.700 | 0.523 | 0.739 |
| Training-optimized multi-flag heuristic | All 5 training-optimized flags | 5 flags | 13/13 | 1/10 | 1.000 | 0.100 | 0.550 | 0.243 | 0.609 |
| Multivariate classifier | ACeT | Two-class softmax argmax (Approved probability 0.5 absent ties) | 10/13 | 8/10 | 0.769 | 0.800 | 0.785 | 0.565 | 0.783 |

See `TABLE_S14_FOOTNOTE.md` for the table footnote. This draft is intended to be appended after existing Supplementary Table S13 without renumbering Tables S1-S13.
