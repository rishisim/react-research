import requests
from bs4 import BeautifulSoup

def find_unique_instruction_sessions(max_sessions=100):
    """Find session IDs that have unique instructions"""
    WEBSHOP_URL = 'http://localhost:3000'
    instructions_seen = {}
    unique_sessions = []
    
    for session_id in range(max_sessions):
        try:
            url = f'{WEBSHOP_URL}/{session_id}'
            response = requests.get(url)
            if response.status_code == 200:
                html = response.text
                soup = BeautifulSoup(html, 'html.parser')
                text = soup.get_text()
                if 'Instruction:' in text:
                    instruction_start = text.find('Instruction:')
                    instruction_end = text.find('\n\n\n', instruction_start)
                    if instruction_end == -1:
                        instruction_end = instruction_start + 100
                    instruction = text[instruction_start:instruction_end].strip()
                    
                    # Only keep first occurrence of each unique instruction
                    if instruction not in instructions_seen:
                        instructions_seen[instruction] = session_id
                        unique_sessions.append(session_id)
                        print(f'Session {session_id}: {instruction[:80]}...')
        except Exception as e:
            print(f'Session {session_id}: Error - {e}')
    
    print(f'\nFound {len(unique_sessions)} unique instructions in first {max_sessions} sessions')
    print(f'Unique session IDs: {unique_sessions}')
    return unique_sessions

if __name__ == "__main__":
    unique_sessions = find_unique_instruction_sessions(100)
    
    # Save to file for use in run_agent.py
    with open('unique_session_ids.txt', 'w') as f:
        f.write(','.join(map(str, unique_sessions)))
    print(f'Saved unique session IDs to unique_session_ids.txt')
