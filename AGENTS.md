# Agents

This file provides an overview of the different agents implemented for various tasks and datasets within this research project. Each agent is designed to interact with a specific environment and solve tasks using a ReAct-based approach.

## ReAct Framework Overview

The agents in this project are built upon the ReAct (Reasoning and Acting) framework. ReAct enables language models to solve complex tasks by interleaving reasoning (thought) and action steps. This allows the agent to create and update a plan, handle exceptions, and learn from its interactions with the environment.

Two primary variations of the ReAct framework are utilized:

*   **Typical ReAct (num_traces = 1)**: This is the standard implementation where the agent generates a single trajectory of thought-action-observation steps to solve a task. It follows a linear path of reasoning and action.

*   **Synthesized ReAct (num_traces = 3)**: This is a proposed enhancement to the ReAct framework. In this approach, the agent generates multiple (in this case, three) reasoning traces in parallel. These traces are then synthesized to produce a more robust and often more accurate final answer. This method allows for exploration of different reasoning paths and can lead to better decision-making, especially in complex or ambiguous scenarios.

## Alfworld Agent

*   **Dataset**: [ALFWorld](https://alfworld.github.io/)
*   **File**: `src/agents/alfworld/`
*   **Description**: The Alfworld agent is designed to solve complex, multi-step tasks in a text-based simulated household environment. It processes natural language instructions to perform actions like navigating rooms, interacting with objects (e.g., `take`, `put`, `open`), and ultimately achieving a high-level goal.

## FEVER Agent

*   **Dataset**: [FEVER (Fact Extraction and VERification)](https://fever.ai/)
*   **File**: `src/agents/fever/fever_agent.py`
*   **Frameworks**: See [`progress_notes/frameworks.md`](progress_notes/frameworks.md) for detailed documentation
*   **Description**: The FEVER agent is tasked with verifying the truthfulness of a given claim. It interacts with a Wikipedia-like environment to find evidence supporting or refuting the claim. Its primary actions include `search` to find relevant articles and `lookup` to find specific information within an article, concluding with a `finish` action to state whether the claim is "SUPPORTS", "REFUTES", or "NOT ENOUGH INFO".

## HotPotQA Agent

*   **Dataset**: [HotPotQA](https://hotpotqa.github.io/)
*   **File**: `src/agents/hotpotqa/hotpotqa_agent.py`
*   **Description**: This agent is built for multi-hop question answering. Given a question, it must navigate through multiple documents to find different pieces of information and then synthesize them to form a final answer. Like the FEVER agent, it uses `search` and `lookup` to gather information before providing an answer with the `finish` action.

## WebShop Agent

*   **Dataset**: WebShop
*   **File**: `WebShop.ipynb`
*   **Description**: The WebShop agent simulates a user on an e-commerce website. It is given a product request as an instruction and must navigate the site to find and purchase the correct item. The agent can perform actions such as `search[query]`, `click[button]`, and `think[thought]`. The environment provides observations in a simplified HTML-like format, and the agent's goal is to successfully purchase an item that matches the user's request.
