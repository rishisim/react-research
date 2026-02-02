# Walkthrough - Summary Tables Generation

## Goal
Generate summary tables for FEVER and HotPotQA datasets, calculating metrics like Accuracy, Tokens, and Gains for various frameworks, excluding "action-pruning". Also includes a combined accuracy summary table.

## Changes
- Created `generate_summary_tables.py` to:
    - Load data from CSV (React) and JSON (other frameworks) files.
    - Calculate metrics: Accuracy, Tokens/Task, Tokens/Success Task, Gains, etc.
    - Generate formatting tables printed to stdout.
    - **[New]** Generate a combined table showing Framework, Accuracy (%) and Fraction (Correct/Total) for both datasets side-by-side.
    - **[New]** Renamed "Self-Reflection" to "Trajectory-Conditioned Answer Revision (TCAR)".

## Verification Results

### Combined Accuracy Summary Table
| Framework | HotPotQA Accuracy | HotPotQA Num | FEVER Accuracy | FEVER Num |
| --- | --- | --- | --- | --- |
| CoT-SC | 37.88 | 189/499 | 65.00 | 325/500 |
| Majority Voting | 40.08 | 200/499 | 64.40 | 322/500 |
| ReAct | 35.80 | 179/500 | 63.40 | 317/500 |
| Reflexion | 44.20 | 221/500 | 80.60 | 403/500 |
| Trajectory-Conditioned Answer Revision (TCAR) | 36.60 | 183/500 | 65.67 | 329/501 |

### HotPotQA Detailed Summary
| Framework | Accuracy | Num | SUM of em | COUNT of em | SUM of total_tokens | Tokens/Task | Tokens/Success Task | Gain | Add. Tokens | Add. Token Per Percent Gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ReAct | 0.3580 | 500 | 179 | 500 | 2669027 | 5338.0540 | 5894.4860 | 0.0000 | 0.0000 | 0.0000 |
| Reflexion | 0.4440 | 500 | 222 | 500 | 8320490 | 16640.9800 | 12903.6261 | 0.0860 | 11302.9260 | 1314.2937 |
| Trajectory-Conditioned Answer Revision (TCAR) | 0.3800 | 500 | 190 | 500 | 2673415 | 5346.8300 | 5869.7579 | 0.0220 | 8.7760 | 3.9891 |
| Majority Voting | 0.3660 | 500 | 183 | 500 | 13159049 | 26318.0980 | 27271.9399 | 0.0080 | 20980.0440 | 26225.0550 |
| CoT-SC | 0.3788 | 499 | 189 | 499 | 17982042 | 36036.1563 | 32036.7619 | 0.0208 | 26698.1083 | 12861.9000 |

### FEVER Detailed Summary
| Framework | Accuracy | Num | SUM of em | COUNT of em | SUM of total_tokens | Tokens/Task | Tokens/Success Task | Gain | Add. Tokens | Add. Token Per Percent Gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ReAct | 0.6340 | 500 | 317 | 500 | 1693611 | 3387.2220 | 2910.7603 | 0.0000 | 0.0000 | 0.0000 |
| Reflexion | 0.8060 | 500 | 403 | 500 | 8328162 | 16656.3240 | 7643.5062 | 0.1720 | 13269.1020 | 771.4594 |
| Trajectory-Conditioned Answer Revision (TCAR) | 0.6567 | 501 | 329 | 501 | 2967499 | 5923.1517 | 5423.1672 | 0.0227 | 2535.9297 | 1117.8082 |
| Majority Voting | 0.6440 | 500 | 322 | 500 | 5268856 | 10537.7120 | 9385.6584 | 0.0100 | 7150.4900 | 7150.4900 |
| CoT-SC | 0.6500 | 500 | 325 | 500 | 7419269 | 14838.5380 | 13814.6154 | 0.0160 | 11451.3160 | 7157.0725 |

### Notes
- **Combined Table**: Shows Accuracy (0-100 scale) and Fraction (Correct/Total) side-by-side.
- **Accuracy**: Calculated as `SUM of em / Num`.
- **Gains**: Relative to ReAct baseline.
- **Excluded**: Action Pruning as requested.
