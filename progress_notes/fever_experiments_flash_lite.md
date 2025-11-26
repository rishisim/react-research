# FEVER Experiment Results: Baseline vs Multi-Trace vs Reflexion

**Date:** November 25, 2025  
**Model:** `gemini-2.5-flash-lite`  
**Dataset:** FEVER (Fact Extraction and VERification)  
**Sample Size:** 15 Examples

## 📊 Performance Summary

| Framework | Accuracy (EM) | Success Count | Total LLM Calls | Avg Calls/Example |
|-----------|---------------|---------------|-----------------|-------------------|
| **Baseline ReAct** | **40.0%** | 6/15 | 56 | 3.73 |
| **Multi-Trace ReAct** | **40.0%** | 6/15 | 142 | 9.47 |
| **Reflexion** | **33.3%** | 5/15 | 148 | 9.87 |

> [!WARNING]
> **Unexpected Result**: Advanced methods (Multi-Trace and Reflexion) failed to outperform the simple Baseline. Reflexion actually performed *worse* despite highest cost.

## 🔍 Analysis

### 1. Efficiency vs Accuracy
- **Baseline** was the clear winner: same accuracy as Multi-Trace but **~2.5x more efficient**.
- **Reflexion** was the most expensive (~2.6x baseline cost) and least accurate.

### 2. Why did advanced methods fail?
- **Model Capability**: `gemini-2.5-flash-lite` is likely too small/fast for these complex workflows.
    - **Multi-Trace**: The model may be "consistently wrong" (hallucinating the same wrong answer multiple times), so voting doesn't help.
    - **Reflexion**: The model may struggle to self-correct. When asked to reflect on a wrong answer, it might just double down or get confused by the additional context.
- **Task Difficulty**: FEVER requires precise fact-checking. If the model can't find the evidence in the first place, reasoning about it more (Reflexion) or trying multiple times (Multi-Trace) won't help if the search queries are fundamentally flawed.

## 📝 Detailed Results

### Baseline ReAct
- **Correct**: 6
- **Incorrect**: 9
- **Avg Calls**: 3.73

### Multi-Trace ReAct (CoT-SC)
- **Correct**: 6
- **Incorrect**: 9
- **Avg Calls**: 9.47
- **Note**: Voting didn't change the outcome for any of the 15 examples compared to baseline aggregate performance.

### Reflexion
- **Correct**: 5
- **Incorrect**: 10
- **Avg Calls**: 9.87
- **Note**: Actually *flipped* one correct answer to incorrect (or failed a case that others got right).

## 🚀 Recommendation

To see the benefits of agentic workflows (CoT-SC and Reflexion), we **must upgrade the model**.
- **Next Step**: Run the same experiment with `gemini-2.5-flash` (standard) or `gemini-1.5-pro`.
- These larger models have better reasoning capabilities to:
    1. Generate diverse traces (for Multi-Trace)
    2. Critically evaluate their own mistakes (for Reflexion)
