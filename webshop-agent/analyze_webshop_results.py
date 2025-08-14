import json
import csv

def extract_rewards_from_trajectories(file_path, output_file):
    """
    Extract instructions and final rewards from trajectory data and save to CSV
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = []
    
    for i, item in enumerate(data, 1):
        # Use sequential question number instead of full instruction
        question_number = i
        
        # Get the final reward
        final_reward = item.get('final_reward', 0.0)
        
        results.append({
            'Question': question_number,
            'Reward': final_reward
        })
    
    # Write to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Question', 'Reward']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    
    print(f"Created {output_file} with {len(results)} entries")

def main():
    # CASE 2A: STM Results
    print("Processing CASE 2A STM Results...")
    extract_rewards_from_trajectories(
        'webshop_log_in_traj_reflexion_st_trajectories.json',
        'CASE 2A STM Results.csv'
    )
    
    # CASE 2B: LTM Results  
    print("Processing CASE 2B LTM Results...")
    extract_rewards_from_trajectories(
        'webshop_log_in_traj_reflexion_trajectories.json',
        'CASE 2B LTM Results.csv'
    )
    
    print("All files created successfully!")

if __name__ == "__main__":
    main()
