# Prompts for the Nexus Framework

# Phase 1: SCOUT (Entity Extraction)
SCOUT_PROMPT = """You are the SCOUT. Your goal is to identify the key entities and claims in a user's question so that we can fetch their "Passports" (core summaries) in parallel.

Input: "Tracey Edmonds produced Soul Food."
Output: ["Tracey Edmonds", "Soul Food"]

Input: "What is the shared country of ancestry between Art Laboe and Scout Tufankjian?"
Output: ["Art Laboe", "Scout Tufankjian"]

Input: "Did the same director who directed 'Jaws' also direct 'E.T.'?"
Output: ["Jaws (film)", "E.T. the Extra-Terrestrial"]

Input: "{question}"
Output:
"""

# Phase 2: ARCHITECT (Link Analysis & Bridging)
ARCHITECT_PROMPT = """You are the ARCHITECT. Your goal is to verify if independent "Passports" (summaries) connected to form a valid answer, or if we need a "Bridge Query" to fill a gap.

Question: "{question}"

[Passport A]: {passport_a}
[Passport B]: {passport_b}

Task 1: Entity Resolution (NLI Check)
Given the contexts above, is the relationship defined in the question supported, refuted, or unknown? 
- If the relationship is EXPLICITLY confirmed or denied in the text, Output: "Status: RESOLVED".
- If the relationship is inferred but not explicit, or if a link is missing (e.g., A mentions B, but B doesn't mention A, or neither mentions the other), Output: "Status: GAP".

Task 2: Bridge Query Generation (Only if GAP)
If there is a GAP, formulate a "Bridge Query" to connect them.
CRITICAL: Do NOT just concatenate the names (e.g., "Tracey Edmonds Soul Food").
Search for the MISSING ATTRIBUTE or LIST. Make the query "Conceptually Orthogonal".

Example: 
Question: "Did Tracey Edmonds produce Soul Food?"
Gap: Tracey's bio doesn't mention Soul Food. Soul Food's bio doesn't mention Tracey.
Bad Bridge: "Tracey Edmonds Soul Food" (Redundant)
Good Bridge: "producers of the film Soul Food" (Targeting the attribute)

Output Format:
Status: [RESOLVED | GAP]
Reasoning: [Brief explanation]
Bridge Actions: ["Search(Query 1)", "Search(Query 2)", "Search(Query 3)"]
(Provide up to 3 queries, prioritized from most specific to broad. If no bridge is needed, return empty list [])
"""

# Phase 3: ADJUDICATOR (Synthesis)
ADJUDICATOR_PROMPT = """You are the ADJUDICATOR. You have a full dossier of evidence. Your job is to give the final answer.

Question: "{question}"

[Dossier]:
{dossier}

Instructions:
1. Answer the question based ONLY on the dossier.
2. If the question is a verification (True/False/Claims), output SUPPORTS, REFUTES, or NOT ENOUGH INFO.
3. If the question is a QA task, output the short answer.

Format:
Reasoning: [Step-by-step deduction]
Answer: [Final Answer]
"""
