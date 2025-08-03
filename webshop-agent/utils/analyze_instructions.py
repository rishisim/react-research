#!/usr/bin/env python3
"""
Temporary analysis script to understand unique instructions in items_human_ins.json
"""
import json

def analyze_instructions():
    with open('../webshop/data/items_human_ins.json', 'r') as f:
        data = json.load(f)
    
    instruction_counts = {}
    asin_mapping = {}
    
    for asin, instruction_list in data.items():
        for item in instruction_list:
            instruction = item['instruction']
            if instruction not in instruction_counts:
                instruction_counts[instruction] = 0
                asin_mapping[instruction] = []
            instruction_counts[instruction] += 1
            asin_mapping[instruction].append(asin)
    
    total_instructions = sum(instruction_counts.values())
    unique_instructions = len(instruction_counts)
    duplicates = total_instructions - unique_instructions
    
    print(f"Total instructions: {total_instructions}")
    print(f"Unique instructions: {unique_instructions}")
    print(f"Duplicates: {duplicates}")
    print()
    
    # Show most frequent duplicates
    duplicate_instructions = {k: v for k, v in instruction_counts.items() if v > 1}
    sorted_duplicates = sorted(duplicate_instructions.items(), key=lambda x: x[1], reverse=True)
    
    print(f"Top 10 most duplicated instructions:")
    for i, (instruction, count) in enumerate(sorted_duplicates[:10]):
        print(f"{i+1}. Count: {count} - '{instruction[:80]}{'...' if len(instruction) > 80 else ''}'")
        print(f"   ASINs: {', '.join(asin_mapping[instruction][:5])}{'...' if len(asin_mapping[instruction]) > 5 else ''}")
        print()
    
    # Save duplicates to file for further analysis
    with open('duplicate_instructions.json', 'w') as f:
        json.dump({
            'summary': {
                'total': total_instructions,
                'unique': unique_instructions,
                'duplicates': duplicates
            },
            'duplicate_details': sorted_duplicates
        }, f, indent=2)
    
    print(f"Detailed duplicate analysis saved to 'duplicate_instructions.json'")

def analyze_trajectory_instructions():
    """Analyze instructions already used in trajectory files"""
    files = ['webshop_trajectories.json', 'webshop_synthesized_trajectories.json']
    
    print("\n" + "="*60)
    print("ANALYZING TRAJECTORY FILES")
    print("="*60)
    
    all_used_instructions = set()
    
    for filename in files:
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            used_instructions = []
            for entry in data:
                if 'instruction' in entry:
                    used_instructions.append(entry['instruction'])
                    all_used_instructions.add(entry['instruction'])
            
            # Check for duplicates within this file
            unique_in_file = set(used_instructions)
            duplicates_in_file = len(used_instructions) - len(unique_in_file)
            
            print(f"\n{filename}:")
            print(f"  Total entries: {len(data)}")
            print(f"  Total instructions: {len(used_instructions)}")
            print(f"  Unique instructions: {len(unique_in_file)}")
            print(f"  Duplicates in file: {duplicates_in_file}")
            
        except FileNotFoundError:
            print(f"\n{filename}: File not found")
    
    print(f"\nOverall used instructions across both files: {len(all_used_instructions)}")
    
    return all_used_instructions

if __name__ == "__main__":
    analyze_instructions()
    used_instructions = analyze_trajectory_instructions()
    
    print(f"\nSUMMARY:")
    print(f"- Available unique instructions in dataset: 11,724")
    print(f"- Instructions already used in trajectories: {len(used_instructions)}")
    print(f"- Remaining unique instructions available: {11724 - len(used_instructions)}")
    analyze_trajectory_instructions()
