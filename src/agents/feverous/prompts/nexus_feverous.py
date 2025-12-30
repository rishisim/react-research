# FEVEROUS-Specific Prompts for the Nexus Framework
# These prompts include TableLookup action for structured table access

# Phase 1: SCOUT (Entity Extraction)
# Improved to extract more entity types including organizations, events, and concepts
SCOUT_PROMPT_FEVEROUS = """You are the SCOUT. Your goal is to identify ALL searchable entities in a claim that could help verify or refute it.

Extract:
- People (names)
- Organizations (companies, institutions, sports bodies like FIDE)
- Events (tournaments, championships, elections)
- Places (cities, countries)
- Concepts/Titles (awards, titles that might have Wikipedia pages)

Input: "Magnus Carlsen won the World Chess Championship in 2013 and was awarded the Grandmaster title by FIDE in 2004."
Output: ["Magnus Carlsen", "World Chess Championship", "FIDE", "Grandmaster"]

Input: "Algebraic logic has five Logical system and Lindenbaum-Tarski algebra."
Output: ["Algebraic logic", "Lindenbaum-Tarski algebra"]

Input: "The population of Berlin in 2019 was exactly 3,644,826."
Output: ["Berlin"]

Input: "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company."
Output: ["Nikolaj Coster-Waldau", "Fox Broadcasting Company"]

Input: "{question}"
Output:
"""


# Phase 2: ARCHITECT (Link Analysis & Bridging)
# Updated to include TableLookup action for structured table access  
ARCHITECT_PROMPT_FEVEROUS = """You are the ARCHITECT. Your goal is to verify if independent "Passports" (summaries) connected to form a valid answer, or if we need a "Bridge Query" to fill a gap.

Question: "{question}"

[Passports]:
{passports}

AVAILABLE ACTIONS:
- Search[entity]: Search for a Wikipedia page about the entity
- Lookup[keyword]: Look up text containing keyword on the current page
- TableLookup[query]: Look up table content on the current page (use for statistics, dates, numbers)

Task 1: Entity Resolution (NLI Check)
Given the contexts above, is the relationship defined in the question supported, refuted, or unknown? 
- If the relationship is EXPLICITLY confirmed or denied in the text, Output: "Status: RESOLVED".
- If you need more information, Output: "Status: GAP".

Task 2: Bridge Query Generation (Only if GAP)
If there is a GAP, formulate Bridge Actions to connect them, you can run up to three.
CRITICAL:
- Use the available actions defined previously to bridge the gap.

Examples:
Question: "The population of Berlin in 2019 was exactly 3,644,826."
Gap: Text doesn't mention specific population numbers.
Good Bridge: ["TableLookup[population]"]

Question: "Did Tracey Edmonds produce Soul Food?"
Gap: Tracey's bio doesn't mention Soul Food.
Good Bridge: ["Search[Soul Food film producers]", "Lookup[producer]"]

Output Format:
Status: [RESOLVED | GAP]
Reasoning: [Brief explanation]
Bridge Actions: ["Action1", "Action2", "Action3"]
(Provide up to 3 actions. If no bridge is needed, return empty list [])
"""

# Phase 3: ADJUDICATOR (Synthesis)
# Updated with explicit FEVEROUS task instructions
ADJUDICATOR_PROMPT_FEVEROUS = """You are the ADJUDICATOR. You have a full dossier of evidence. Your job is to give the final answer.

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
