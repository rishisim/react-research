# FEVEROUS-Specific Prompts for the ReWOO Framework
# ReWOO uses Planner -> Worker -> Solver paradigm with #E placeholders

# Phase 1: PLANNER
# Generates a complete plan upfront with #E placeholders for tool outputs
PLANNER_PROMPT_FEVEROUS = """For the following claim, generate a step-by-step plan to gather evidence and verify it.
Use the following format:
Plan: [reasoning about what to do next]
#E[n] = Tool[input]

Available tools:
- Search[entity]: Search for a Wikipedia page about the entity
- Lookup[keyword]: Look up text containing keyword on the current page
- TableLookup[query]: Look up table data on the current page (for statistics, dates, numbers)

You can reference previous evidence using #E[n] placeholders, e.g., #E2 = Lookup[name from #E1]

Examples:

Claim: "Magnus Carlsen became World Chess Champion in 2013."
Plan: First, I need to search for Magnus Carlsen to find information about his chess career.
#E1 = Search[Magnus Carlsen]
Plan: Now I need to look up information about when he became World Chess Champion.
#E2 = Lookup[World Chess Champion]

Claim: "The population of Berlin in 2019 was exactly 3,644,826."
Plan: First, search for Berlin to find its Wikipedia page.
#E1 = Search[Berlin]
Plan: The population figures are usually in a table, so I'll use TableLookup.
#E2 = TableLookup[population 2019]

Claim: "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company."
Plan: Search for Nikolaj Coster-Waldau to find his filmography.
#E1 = Search[Nikolaj Coster-Waldau]
Plan: Look up information about Fox Broadcasting Company in his career.
#E2 = Lookup[Fox]
Plan: Also search for Fox Broadcasting Company directly for more context.
#E3 = Search[Fox Broadcasting Company]

Claim: "{question}"
"""

# Phase 3: SOLVER
# Takes original claim + evidence log, produces final verdict
SOLVER_PROMPT_FEVEROUS = """Based on the evidence collected, determine whether the claim is SUPPORTED, REFUTED, or if there is NOT ENOUGH INFO.

Claim: "{question}"

Evidence Collected:
{evidence_log}

Instructions:
1. Analyze the evidence carefully
2. Compare the claim against the evidence
3. Provide your reasoning step by step
4. Give a final verdict: SUPPORTS, REFUTES, or NOT ENOUGH INFO

Format:
Reasoning: [Step-by-step analysis of evidence]
Answer: [SUPPORTS | REFUTES | NOT ENOUGH INFO]
"""
