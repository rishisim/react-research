#!/usr/bin/env python3
"""
Test script to verify that trajectory formatting works correctly.
"""

import json
import sys
import os

# Add the webshop-agent directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.log_in_traj_reflexion import LogInTrajReflexionAgent
from webshop_env import WebShopEnv

def test_trajectory_format():
    """Test that the trajectory is properly formatted with reflexions and LTM updates."""
    
    # Initialize environment and agent
    env = WebShopEnv()
    agent = LogInTrajReflexionAgent()
    
    # Use a simple test instruction
    session_id = "381"  # Using one from the existing data
    instruction = "i am looking for a queen sized bed that is black, and price lower than 140.00 dollars"
    
    print("Testing trajectory formatting...")
    print(f"Session ID: {session_id}")
    print(f"Instruction: {instruction}")
    print("-" * 60)
    
    try:
        # Run for just a few steps to test the format
        reward, trajectory, llm_calls, st_memory = agent.run_episode(
            env, session_id, instruction, max_steps=5, to_print=True, num_traces=1
        )
        
        print("\n" + "=" * 60)
        print("TRAJECTORY STRUCTURE:")
        print("=" * 60)
        
        # Pretty print the trajectory structure
        for i, step in enumerate(trajectory):
            print(f"\nStep {i}:")
            print(f"  Type: {step.get('type', 'action')}")
            print(f"  Step: {step.get('step', 'N/A')}")
            
            if 'action' in step:
                print(f"  Action: {step['action']}")
                print(f"  Observation: {step['observation'][:100]}...")
                print(f"  Reward: {step.get('reward', 'N/A')}")
                print(f"  Done: {step.get('done', 'N/A')}")
            elif 'reflexion' in step:
                print(f"  Reflexion: {step['reflexion'][:100]}...")
                print(f"  Trigger Step: {step.get('trigger_step', 'N/A')}")
            elif 'ltm_update' in step:
                print(f"  LTM Update: {step['ltm_update'][:100]}...")
                print(f"  Update Step: {step.get('update_step', 'N/A')}")
        
        print("\n" + "=" * 60)
        print("JSON SERIALIZATION TEST:")
        print("=" * 60)
        
        # Test JSON serialization
        test_data = {
            'session_id_index': 381,
            'instruction': instruction,
            'final_reward': reward,
            'trajectory': trajectory,
            'llm_calls': llm_calls,
            'agent_type': 'log_in_traj_reflexion',
            'st_memory': st_memory
        }
        
        json_str = json.dumps(test_data, indent=2)
        print("JSON serialization successful!")
        print(f"JSON length: {len(json_str)} characters")
        
        # Verify it can be loaded back
        loaded_data = json.loads(json_str)
        print("JSON deserialization successful!")
        
        # Save a sample to see the full format
        with open('sample_trajectory_format.json', 'w') as f:
            json.dump([test_data], f, indent=2)
        print("Sample saved to: sample_trajectory_format.json")
        
        return True
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_trajectory_format()
    if success:
        print("\n✅ Test completed successfully!")
    else:
        print("\n❌ Test failed!")
        sys.exit(1)
