
import unittest
from unittest.mock import MagicMock
# Adjust import for running as a module from root
from src.agents.nexus.nexus_agent import NexusAgent

class TestNexusAgent(unittest.TestCase):
    def setUp(self):
        # Mock environment and LLM
        self.mock_env = MagicMock()
        self.mock_llm = MagicMock()
        # NexusAgent now takes llm_func and env in __init__
        self.agent = NexusAgent(llm_func=self.mock_llm, env=self.mock_env)

    def test_scout_phase_parsing(self):
        # Mock LLM response for SCOUT
        self.mock_llm.side_effect = ['["Entity A", "Entity B"]']
        
        entities = self.agent.scout_phase("Test Question")
        self.assertEqual(entities, ["Entity A", "Entity B"])
        
    def test_scout_phase_fallback(self):
        # Mock LLM failure to return JSON
        self.mock_llm.side_effect = ["Invalid JSON"]
        
        entities = self.agent.scout_phase("Test Question")
        self.assertEqual(entities, ["Test Question"])

    def test_architect_phase_bridge_detection(self):
        # Mock LLM response for ARCHITECT with a GAP
        # Assuming the prompt returns "Bridge Action: Search[New Query]"
        self.mock_llm.side_effect = [
            "Status: GAP\nReasoning: Missing link.\nBridge Action: Search[New Query]"
        ]
        
        # Mock Env step return
        self.mock_env.step.return_value = ("Bridge content", "reward", "done", "info")
        
        passports = {"A": "Info A", "B": "Info B"}
        bridge_info, trace = self.agent.architect_phase("Question", passports)
        
        self.assertIn("Bridge_New Query", bridge_info)
        self.assertEqual(bridge_info["Bridge_New Query"], "Bridge content")
        self.mock_env.step.assert_called_with("Search[New Query]")

    def test_architect_phase_resolved(self):
        # Mock LLM response for ARCHITECT with RESOLVED
        self.mock_llm.side_effect = [
            "Status: RESOLVED\nReasoning: Link explicitly stated."
        ]
        
        passports = {"A": "Info A", "B": "Info B"}
        bridge_info, trace = self.agent.architect_phase("Question", passports)
        
        self.assertEqual(bridge_info, {})
        self.assertIn("No Bridge Query needed", trace)

if __name__ == '__main__':
    unittest.main()
