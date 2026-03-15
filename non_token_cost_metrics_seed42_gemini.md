# Non-token Cost Metrics Report (Gemini seed42, FEVER + HotPotQA)

## Scope & Path Notes
- Scope used: `results/fever/gemini/seed42_mixed` and `results/hotpotqa/gemini/seed42_mixed`.
- Note: repository path is lowercase **hotpotqa** (not `hotpotQA`).
- TCAR mapping in this repo: **Self-Reflection** is renamed to **Trajectory-Conditioned Answer Revision (TCAR)** in summaries: [summary_tables.md](summary_tables.md#L12).

---

## Available now

Yes — you can add non-token operational metrics **without inventing anything**.

### What is directly logged and usable
- `n_calls` (API/LLM calls per instance)
- `n_badcalls` (failed/bad calls per instance)
- `num_trials` (Reflexion only)
- `num_traces_run` (Majority Voting / CoT-SC)

### Where metric production is implemented
- FEVER runner framework map + status + summary aggregation:
  - [src/agents/fever/run_fever_experiments.py](src/agents/fever/run_fever_experiments.py#L32-L40)
  - [src/agents/fever/run_fever_experiments.py](src/agents/fever/run_fever_experiments.py#L288)
  - [src/agents/fever/run_fever_experiments.py](src/agents/fever/run_fever_experiments.py#L438-L440)
- HotPotQA runner framework map + status + summary aggregation:
  - [src/agents/hotpotqa/run_hotpotqa_experiments.py](src/agents/hotpotqa/run_hotpotqa_experiments.py#L34-L42)
  - [src/agents/hotpotqa/run_hotpotqa_experiments.py](src/agents/hotpotqa/run_hotpotqa_experiments.py#L282)
  - [src/agents/hotpotqa/run_hotpotqa_experiments.py](src/agents/hotpotqa/run_hotpotqa_experiments.py#L436-L438)
- Agent-level metric writes:
  - ReAct: [src/agents/fever/react_agent.py](src/agents/fever/react_agent.py#L70-L73), [src/agents/hotpotqa/react_agent.py](src/agents/hotpotqa/react_agent.py#L72-L73)
  - Reflexion: [src/agents/fever/reflexion_react_agent.py](src/agents/fever/reflexion_react_agent.py#L274-L279), [src/agents/hotpotqa/reflexion_react_agent.py](src/agents/hotpotqa/reflexion_react_agent.py#L269-L274)
  - Majority Voting: [src/agents/fever/majority_voting_agent.py](src/agents/fever/majority_voting_agent.py#L143-L148), [src/agents/hotpotqa/majority_voting_agent.py](src/agents/hotpotqa/majority_voting_agent.py#L142-L147)
  - CoT-SC: [src/agents/fever/cot_sc_agent.py](src/agents/fever/cot_sc_agent.py#L132-L137), [src/agents/hotpotqa/cot_sc_agent.py](src/agents/hotpotqa/cot_sc_agent.py#L153-L158)
  - Self-Reflection/TCAR: [src/agents/fever/self_reflection_agent.py](src/agents/fever/self_reflection_agent.py#L229-L230), [src/agents/hotpotqa/self_reflection_agent.py](src/agents/hotpotqa/self_reflection_agent.py#L200-L201)

---

## Metric inventory (exact name, source, producer, granularity, reliability)

| Metric name | Source of truth | Producing code | Granularity | Reliability |
|---|---|---|---|---|
| `n_calls` | per-method result JSONs in seed42_mixed | agent files + runner summaries | per-instance; aggregatable per-run | High |
| `n_badcalls` | per-method result JSONs in seed42_mixed | agent files + runner summaries | per-instance; aggregatable per-run | High |
| `num_trials` | Reflexion result JSONs | reflexion agent files | per-instance (Reflexion only) | FEVER: High; HotPotQA: Medium (missing on 3 rows) |
| `num_traces_run` | Majority Voting / CoT-SC JSONs | majority/cot_sc agent files | per-instance (MV/CoT-SC only) | High |
| `total_llm_calls` | FEVER summary file | FEVER runner summary generation | aggregate per framework | High (FEVER only present in file) |
| `avg_calls_per_example` | FEVER summary file | FEVER runner summary generation | aggregate per framework | High (FEVER only present in file) |
| `duration_minutes` | FEVER run history file | runner run-history write | run-level only | Low for paper latency claims (not per-instance; limited run history) |

Evidence examples:
- FEVER summary fields: [results/fever/gemini/seed42_mixed/summary.json](results/fever/gemini/seed42_mixed/summary.json#L9-L11)
- FEVER run-level duration: [results/fever/gemini/seed42_mixed/run_history.json](results/fever/gemini/seed42_mixed/run_history.json#L6)
- HotPotQA seed42_mixed consolidation note: [results/hotpotqa/gemini/seed42_mixed/config.json](results/hotpotqa/gemini/seed42_mixed/config.json#L7)

---

## Not reliable / missing

| Item | Status | Why |
|---|---|---|
| Per-task latency/runtime | Not logged | No per-instance `latency`/`duration` fields in seed42_mixed result JSONs |
| Controlled wall-clock latency comparisons | Not reliable | Only FEVER has a short run-level `duration_minutes`; not per-instance and likely environment-noisy |
| Tool-call count as a separate explicit metric | Not logged | No dedicated top-level `tool_calls` field in final result records |
| Monetary API cost (USD) | Not logged | No direct cost field in seed42_mixed results |

---

## Paste-ready appendix table

### HotPotQA (Gemini, seed42)

| Method | File | Rows | API calls/task (`n_calls` mean) | Failed calls/task (`n_badcalls` mean) | Other op metric | Reliability |
|---|---|---:|---:|---:|---|---|
| ReAct | [results/hotpotqa/gemini/seed42_mixed/react.json](results/hotpotqa/gemini/seed42_mixed/react.json) | 500 | 4.290 | 0.010 | — | High |
| Reflexion | [results/hotpotqa/gemini/seed42_mixed/reflexion.json](results/hotpotqa/gemini/seed42_mixed/reflexion.json) | 500 | 18.046 *(497/500 rows present)* | 0.286 *(497/500 rows present)* | `num_trials` mean = 3.179 *(497/500 present)* | Medium |
| TCAR (`self_reflection`) | [results/hotpotqa/gemini/seed42_mixed/self_reflection.json](results/hotpotqa/gemini/seed42_mixed/self_reflection.json) | 500 | 5.256 | 0.006 | — | High |
| Majority Voting | [results/hotpotqa/gemini/seed42_mixed/majority_voting.json](results/hotpotqa/gemini/seed42_mixed/majority_voting.json) | 499 | 12.938 | 0.034 | `num_traces_run` mean = 3.000 | High |
| CoT-SC | [results/hotpotqa/gemini/seed42_mixed/cot_sc.json](results/hotpotqa/gemini/seed42_mixed/cot_sc.json) | 499 | 12.938 | 0.034 | `num_traces_run` mean = 3.000 | High |

### FEVER (Gemini, seed42)

| Method | File | Rows | API calls/task (`n_calls` mean) | Failed calls/task (`n_badcalls` mean) | Other op metric | Reliability |
|---|---|---:|---:|---:|---|---|
| ReAct | [results/fever/gemini/seed42_mixed/react.json](results/fever/gemini/seed42_mixed/react.json) | 505 | 2.980 | 0.008 | — | High |
| Reflexion | [results/fever/gemini/seed42_mixed/reflexion.json](results/fever/gemini/seed42_mixed/reflexion.json) | 500 | 11.268 | 0.100 | `num_trials` mean = 2.528 | High |
| TCAR (`self_reflection`) | [results/fever/gemini/seed42_mixed/self_reflection.json](results/fever/gemini/seed42_mixed/self_reflection.json) | 501 | 4.034 *(500/501 rows present)* | 0.006 *(500/501 rows present)* | — | Medium |
| Majority Voting | [results/fever/gemini/seed42_mixed/majority_voting.json](results/fever/gemini/seed42_mixed/majority_voting.json) | 500 | 9.226 | 0.044 | `num_traces_run` mean = 3.000 | High |
| CoT-SC | [results/fever/gemini/seed42_mixed/cot_sc.json](results/fever/gemini/seed42_mixed/cot_sc.json) | 500 | 9.226 | 0.044 | `num_traces_run` mean = 3.000 | High |

---

## Latency suitability (main table vs appendix)

- **Main table:** Do **not** include latency/runtime from current seed42 artifacts.
- **Appendix:** You may mention that FEVER has one run-level duration artifact only:
  - [results/fever/gemini/seed42_mixed/run_history.json](results/fever/gemini/seed42_mixed/run_history.json#L6)
- HotPotQA seed42_mixed appears consolidated and lacks run history in that directory:
  - [results/hotpotqa/gemini/seed42_mixed/config.json](results/hotpotqa/gemini/seed42_mixed/config.json#L7)

---

## One-sentence recommendation for the paper

Report `n_calls`, `n_badcalls`, and framework-specific `num_trials`/`num_traces_run` as non-token operational costs in the appendix now, and exclude latency/tool-cost claims from main results until you have controlled per-instance timing and explicit tool-call/currency logging.
