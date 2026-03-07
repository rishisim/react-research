## Table 1 — Main Results (HOTPOTQA)
| Method | Accuracy (Correct/Eval) | Total Tokens | Tokens/Task | Tokens/Success |
| --- | --- | --- | --- | --- |
| ReAct | 35.80% (179/500) | 4,669,024 | 9338.05 | 26083.93 |
| Reflexion | 44.20% (221/500) | 23,873,741 | 47747.48 | 108025.98 |
| TCAR | 36.60% (183/500) | 6,276,242 | 12552.48 | 34296.40 |
| Majority Voting | 40.08% (200/499) | 14,274,892 | 28607.00 | 71374.46 |
| CoT-SC | 37.88% (189/499) | 17,982,042 | 36036.16 | 95143.08 |

## Table 1 — Main Results (FEVER)
| Method | Accuracy (Correct/Eval) | Total Tokens | Tokens/Task | Tokens/Success |
| --- | --- | --- | --- | --- |
| ReAct | 63.40% (317/500) | 1,693,611 | 3387.22 | 5342.62 |
| Reflexion | 80.60% (403/500) | 8,328,162 | 16656.32 | 20665.41 |
| TCAR | 65.67% (329/501) | 2,967,499 | 5923.15 | 9019.75 |
| Majority Voting | 64.40% (322/500) | 5,268,856 | 10537.71 | 16362.91 |
| CoT-SC | 65.00% (325/500) | 7,419,269 | 14838.54 | 22828.52 |

## Table 2 — HotPotQA Relative Deltas vs ReAct
| Method | Accuracy (Correct/Eval) | Gain vs ReAct (pp) | Tokens/Task | Delta Tokens/Task | Tokens/Success | Delta Tokens/Success | Tokens/Task (x ReAct) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Reflexion | 44.20% (221/500) | 8.40 | 47747.48 | 38409.43 | 108025.98 | 81942.04 | 5.113 |
| TCAR | 36.60% (183/500) | 0.80 | 12552.48 | 3214.44 | 34296.40 | 8212.47 | 1.344 |
| Majority Voting | 40.08% (200/499) | 4.28 | 28607.00 | 19268.95 | 71374.46 | 45290.53 | 3.063 |
| CoT-SC | 37.88% (189/499) | 2.08 | 36036.16 | 26698.11 | 95143.08 | 69059.15 | 3.859 |

## Table 3 — FEVER Relative Deltas vs ReAct
| Method | Accuracy (Correct/Eval) | Gain vs ReAct (pp) | Tokens/Task | Delta Tokens/Task | Tokens/Success | Delta Tokens/Success | Tokens/Task (x ReAct) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Reflexion | 80.60% (403/500) | 17.20 | 16656.32 | 13269.10 | 20665.41 | 15322.79 | 4.917 |
| TCAR | 65.67% (329/501) | 2.27 | 5923.15 | 2535.93 | 9019.75 | 3677.13 | 1.749 |
| Majority Voting | 64.40% (322/500) | 1.00 | 10537.71 | 7150.49 | 16362.91 | 11020.29 | 3.111 |
| CoT-SC | 65.00% (325/500) | 1.60 | 14838.54 | 11451.32 | 22828.52 | 17485.90 | 4.381 |

## Appendix Table A1 — Token Distribution Stats (HOTPOTQA)
| Method | median tokens | p90 | p95 | p99 | max |
| --- | --- | --- | --- | --- | --- |
| ReAct | 8068 | 16300 | 17005 | 18511 | 21222 |
| Reflexion | 13062 | 157041 | 169096 | 183856 | 197386 |
| TCAR | 11035 | 20280 | 21035 | 22943 | 25454 |
| Majority Voting | 25051 | 48270 | 50392 | 55148 | 59723 |
| CoT-SC | 32197 | 56963 | 59662 | 65825 | 70613 |

## Appendix Table A1 — Token Distribution Stats (FEVER)
| Method | median tokens | p90 | p95 | p99 | max |
| --- | --- | --- | --- | --- | --- |
| ReAct | 2112 | 7548 | 9420 | 10489 | 11434 |
| Reflexion | 3172 | 48130 | 82062 | 104287 | 113205 |
| TCAR | 4581 | 11761 | 12872 | 13987 | 14832 |
| Majority Voting | 8038 | 21282 | 24926 | 28637 | 31447 |
| CoT-SC | 12001 | 26901 | 31038 | 35223 | 38340 |
