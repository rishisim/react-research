import os
import json

def append_to_json(data, filename):
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        with open(filename, 'r+', encoding='utf-8') as f:
            try: file_data = json.load(f)
            except json.JSONDecodeError: file_data = []
            if isinstance(file_data, list): file_data.append(data)
            else: file_data = [data]
            f.seek(0)
            json.dump(file_data, f, indent=2)
            f.truncate()
    else:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump([data], f, indent=2)

def get_processed_indices(output_file):
    processed_indices = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                for entry in data:
                    if 'session_id_index' in entry:
                        processed_indices.add(entry['session_id_index'])
            except (json.JSONDecodeError, AttributeError): pass
    return processed_indices

def get_processed_instructions(instruction_file='used_instructions.json'):
    """
    Get all instructions that have been processed so far.
    Similar to get_processed_indices but for instructions.
    Loads from a persistent file that tracks used instructions.
    """
    processed_instructions = set()

    # Load from persistent instruction tracking file
    if os.path.exists(instruction_file):
        try:
            with open(instruction_file, 'r', encoding='utf-8') as f:
                instruction_list = json.load(f)
                processed_instructions = set(instruction_list)
        except (json.JSONDecodeError, AttributeError):
            pass

    # Also load from existing trajectory files (in case file doesn't exist yet)
    trajectory_files = ['webshop_trajectories.json', 'webshop_synthesized_trajectories.json']
    for filename in trajectory_files:
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

def save_processed_instruction(instruction, instruction_file='used_instructions.json'):
    """
    Save a newly processed instruction to the persistent tracking file.
    Similar to how session indices are tracked, but for instructions.
    """
    processed_instructions = get_processed_instructions(instruction_file)
    processed_instructions.add(instruction)

    # Save back to file
    try:
        with open(instruction_file, 'w', encoding='utf-8') as f:
            json.dump(list(processed_instructions), f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save instruction to {instruction_file}: {e}")

def get_output_filename(num_traces):
    """Get appropriate filename based on whether using synthesis or not."""
    if num_traces == 1:
        return 'webshop_trajectories.json'  # Standard ReAct
    else:
        return 'webshop_synthesized_trajectories.json'  # Synthesized ReAct
