#!/usr/bin/env python3
"""
Script to compare old vs new trajectory formats
"""

import json

def analyze_trajectory_formats():
    """Compare the formats of old and new trajectory files."""
    
    print("TRAJECTORY FORMAT COMPARISON")
    print("=" * 60)
    
    # Load the old format (reflexion trajectories)
    try:
        with open('webshop_log_in_traj_reflexion_trajectories.json', 'r') as f:
            old_data = json.load(f)
        
        print("\n📄 OLD FORMAT (webshop_log_in_traj_reflexion_trajectories.json):")
        print("-" * 40)
        if old_data:
            sample_old = old_data[0]
            print(f"Keys: {list(sample_old.keys())}")
            print(f"Trajectory type: {type(sample_old.get('trajectory', 'N/A'))}")
            if 'trajectory' in sample_old:
                traj = sample_old['trajectory']
                if isinstance(traj, str):
                    print(f"Trajectory is a string of length: {len(traj)}")
                    print(f"First 200 chars: {traj[:200]}...")
                else:
                    print(f"Trajectory is a {type(traj)} with {len(traj)} items")
        else:
            print("No data found")
    except FileNotFoundError:
        print("OLD FORMAT FILE NOT FOUND")
    except Exception as e:
        print(f"Error loading old format: {e}")
    
    # Load the new format (regular trajectories)
    try:
        with open('flash-lite-results/webshop_trajectories.json', 'r') as f:
            new_data = json.load(f)
        
        print("\n📄 NEW FORMAT (webshop_trajectories.json):")
        print("-" * 40)
        if new_data:
            sample_new = new_data[0]
            print(f"Keys: {list(sample_new.keys())}")
            print(f"Trajectory type: {type(sample_new.get('trajectory', 'N/A'))}")
            if 'trajectory' in sample_new:
                traj = sample_new['trajectory']
                if isinstance(traj, list):
                    print(f"Trajectory is a list with {len(traj)} items")
                    if traj:
                        print(f"First item keys: {list(traj[0].keys())}")
                        print(f"First item: {traj[0]}")
                else:
                    print(f"Trajectory is a {type(traj)}")
        else:
            print("No data found")
    except FileNotFoundError:
        print("NEW FORMAT FILE NOT FOUND")
    except Exception as e:
        print(f"Error loading new format: {e}")
    
    print("\n🎯 WHAT WE WANT FOR LOG-IN-TRAJ-REFLEXION:")
    print("-" * 40)
    print("✓ Structured list format (like webshop_trajectories.json)")
    print("✓ Each step should include: step, action, observation, reward, done")
    print("✓ Reflexions should be inserted at appropriate step positions")
    print("✓ LTM updates should be inserted at appropriate step positions")
    print("✓ Clear JSON structure for easy analysis")
    print("✓ Backwards compatibility with existing tools")

if __name__ == "__main__":
    analyze_trajectory_formats()
