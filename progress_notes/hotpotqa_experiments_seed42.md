# HotPotQA Experiment Results (Seed 42)

**Date:** December 17, 2025
**Model:** `gemini-2.5-flash`
**Seed:** 42
**Total Examples:** 21 (20 new + 1 initial test)

## Summary Statistics

| Framework | Accuracy (EM) | Accuracy (LLM-Judge) | Accuracy (F1) | LLM Calls (Avg) |
|-----------|---------------|----------------------|---------------|-----------------|
| **Majority Voting** | **52.38%** | **57.14%** | **52.38%** | 14.57 |
| React | 42.86% | 47.62% | 44.76% | 4.67 |
| CoT-SC | 42.86% | 47.62% | 44.76% | 14.9 |
| Reflexion | 38.10% | 42.86% | 40.48% | 7.24 |
| Self-Reflection | 33.33% | 33.33% | 35.71% | 5.48 |

## Key Findings

1.  **Majority Voting Dominance:**
    -   Achieved the highest accuracy across all metrics (EM, F1, LLM-Judge).
    -   Significantly outperformed baseline ReAct (+9.5% EM, +9.5% LLM-Judge).
    -    Demonstrated the value of aggregating semantic answers from multiple traces, successfully identifying correct answers where individual traces failed (e.g., questions about "Saoirse Ronan", "The Lorax", "Vitor Belfort").

2.  **LLM-as-Judge Uplift:**
    -   Across most frameworks, LLM-as-judge evaluation identified ~3-5% more correct answers than exact match (EM).
    -   This highlights the importance of semantic evaluation for free-form QA tasks like HotPotQA, where answer phrasing can vary (e.g., "Armenia" vs "Armenian").

3.  **Single vs. Multi-Trace:**
    -   **ReAct (Baseline)** performed surprisingly well, matching CoT-SC in accuracy but with significantly fewer LLM calls (4.67 vs 14.9).
    -   **CoT-SC** (LLM synthesis) did not provide an advantage over standard ReAct in this run, suggesting that simply asking the LLM to synthesize an answer from multiple traces might not be as effective as semantic voting (Majority Voting) for this dataset/model combination.

4.  **Reflexion & Self-Correction Issues:**
    -   **Reflexion** and **Self-Reflection** underperformed the baseline.
    -   This mirrors findings from FEVER experiments, where "correction" steps often led to over-correction or hallucination, changing correct answers to incorrect ones or failing to salvage incorrect traces efficiently.
    -   Self-Reflection had the lowest performance, indicating that a single trace + verification step is often detrimental if the verification model is not highly robust.

## Next Steps
-   Investigate specific failure cases for Reflexion to understand the over-correction patterns.
-   Consider testing with a stronger model (e.g., `gemini-1.5-pro`) to see if reasoning capabilities improve, particularly for the verification steps.
-   Analyze the "null" answer failure modes to improve answer extraction robustness.
