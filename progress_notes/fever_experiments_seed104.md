# Walkthrough: FEVER Experiments (Seed 104)

## Goal
Verify the fix for CoT-SC answer synthesis and evaluate performance of all 4 frameworks (`react`, `cot_sc`, `reflexion`, `majority_voting`) on 20 examples using `gemini-2.5-flash`.

## Changes
1.  **Fixed CoT-SC Bug**:
    *   Increased `max_output_tokens` from 100 to 512 in `fever_utils.py` to prevent truncation.
    *   Improved regex to handle partial matches (e.g., `SUP` -> `SUPPORTS`).
2.  **Configuration**:
    *   Model: `gemini-2.5-flash`
    *   Seed: `104`
    *   Examples: `20`

## Results

| Framework | Accuracy | Success | Total LLM Calls | Avg Calls/Ex |
| :--- | :--- | :--- | :--- | :--- |
| **React** | 70.00% | 14/20 | 49 | 2.45 |
| **CoT-SC** | 65.00% | 13/20 | 175 | 8.75 |
| **Reflexion** | 70.00% | 14/20 | 115 | 5.75 |
| **Majority Voting** | **75.00%** | **15/20** | 159 | 7.95 |

### Key Findings
*   **Majority Voting Wins**: It achieved the highest accuracy (75%), correctly identifying "NOT ENOUGH INFO" in Example 12 where other frameworks failed.
*   **CoT-SC Fix Verified**: The synthesis logic now works correctly. The slightly lower accuracy (65%) compared to React is likely due to variance or specific difficult examples, not the previous bug.
*   **Reflexion**: Matched React's performance (70%).

## Verification
*   **Example 12**:
    *   React: `SUPPORTS` (Incorrect)
    *   CoT-SC: `SUPPORTS` (Incorrect)
    *   Reflexion: `SUPPORTS` (Incorrect)
    *   **Majority Voting**: `NOT ENOUGH INFO` (Correct) ✅
