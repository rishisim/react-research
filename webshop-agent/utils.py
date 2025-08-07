import os
import json

# Define the output files for each mode
OUTPUT_FILES = {
    'standard': 'webshop_react_trajectories.json',
    'synthesized': 'webshop_synthesized_trajectories.json',
    'reflexion': 'webshop_reflexion_trajectories.json'
}
INSTRUCTION_FILE = 'used_instructions.json'

def append_to_json(data, filename):
    """Appends a JSON object to a file containing a list of JSON objects."""
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        with open(filename, 'r+', encoding='utf-8') as f:
            try:
                file_data = json.load(f)
            except json.JSONDecodeError:
                file_data = []

            if isinstance(file_data, list):
                file_data.append(data)
            else:
                # If the file contains a single object, wrap it in a list
                file_data = [file_data, data]

            f.seek(0)
            json.dump(file_data, f, indent=2)
            f.truncate()
    else:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump([data], f, indent=2)

def get_all_processed_indices():
    """
    Gets all processed session indices by checking the output files for all three modes.
    """
    processed_indices = set()
    for mode, filename in OUTPUT_FILES.items():
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    for entry in data:
                        if 'session_id_index' in entry:
                            processed_indices.add(entry['session_id_index'])
                except (json.JSONDecodeError, AttributeError):
                    print(f"Warning: Could not decode JSON from {filename}. It might be empty or corrupted.")
                    pass
    return processed_indices

def get_processed_instructions():
    """
    Get all instructions that have been processed so far from the instruction file
    and all three trajectory logs.
    """
    processed_instructions = set()

    # Load from the persistent instruction tracking file
    if os.path.exists(INSTRUCTION_FILE):
        try:
            with open(INSTRUCTION_FILE, 'r', encoding='utf-8') as f:
                instruction_list = json.load(f)
                processed_instructions = set(instruction_list)
        except (json.JSONDecodeError, AttributeError):
            pass

    # Also load from all existing trajectory files to be robust
    for filename in OUTPUT_FILES.values():
        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for entry in data:
                        if 'instruction' in entry:
                            processed_instructions.add(entry['instruction'])
            except (json.JSONDecodeError, AttributeError):
                pass

    return processed_instructions

def save_processed_instruction(instruction):
    """
    Save a newly processed instruction to the persistent tracking file.
    """
    processed_instructions = get_processed_instructions()
    if instruction not in processed_instructions:
        processed_instructions.add(instruction)
        try:
            with open(INSTRUCTION_FILE, 'w', encoding='utf-8') as f:
                json.dump(list(processed_instructions), f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save instruction to {INSTRUCTION_FILE}: {e}")
