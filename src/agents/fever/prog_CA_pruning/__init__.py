"""
Programmatic Combined Action & Context Pruning module for FEVER.

Combines action pruning (loop detection, success gating, query dedup, cooldowns,
failure patterns, confidence stabilization) with context pruning (evidence state,
dropped thoughts, compact summary).
"""

from .action_pruner import ActionPruner, PrunerState
from .context_pruner import ContextPruner, ContextState, Evidence, build_compact_prompt
from .prog_ca_pruning_agent import run_prog_ca_pruning_react, PROG_CA_PRUNING_PROMPT_TEMPLATE

__all__ = [
    'ActionPruner',
    'PrunerState',
    'ContextPruner',
    'ContextState',
    'Evidence',
    'build_compact_prompt',
    'run_prog_ca_pruning_react',
    'PROG_CA_PRUNING_PROMPT_TEMPLATE',
]
