import requests
from bs4 import BeautifulSoup

# Test fixed_X session IDs like the notebook uses
session_ids = [f'fixed_{i}' for i in range(20)]  # fixed_0, fixed_1, fixed_2, etc.
WEBSHOP_URL = 'http://localhost:3000'

instructions_seen = {}

for session_id in session_ids:
    try:
        url = f'{WEBSHOP_URL}/{session_id}'
        response = requests.get(url)
        if response.status_code == 200:
            html = response.text
            # Extract instruction if present
            soup = BeautifulSoup(html, 'html.parser')
            text = soup.get_text()
            if 'Instruction:' in text:
                instruction_start = text.find('Instruction:')
                instruction_end = text.find('\n\n\n', instruction_start)
                if instruction_end == -1:
                    instruction_end = instruction_start + 100
                instruction = text[instruction_start:instruction_end].strip()
                
                # Check for duplicates
                if instruction in instructions_seen:
                    print(f'DUPLICATE FOUND!')
                    print(f'Session {session_id}: {instruction[:80]}...')
                    print(f'Previously seen in session: {instructions_seen[instruction]}')
                    print('-' * 50)
                else:
                    instructions_seen[instruction] = session_id
                    print(f'Session {session_id}: {instruction[:80]}...')
            else:
                print(f'Session {session_id}: No instruction found')
        else:
            print(f'Session {session_id}: HTTP {response.status_code}')
    except Exception as e:
        print(f'Session {session_id}: Error - {e}')

print(f'\nTotal unique instructions found: {len(instructions_seen)}')
print(f'Total sessions tested: {len(session_ids)}')
