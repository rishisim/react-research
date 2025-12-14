from dotenv import load_dotenv
import os
import time
load_dotenv(override=True)

from google import genai

client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])

print(f"Testing with key: {os.environ['GEMINI_API_KEY'][:15]}...")

# Test 3 calls with 5s delay
for i in range(3):
    try:
        start = time.time()
        resp = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=f'Say hi #{i+1}'
        )
        elapsed = time.time() - start
        print(f"Call {i+1}: Success in {elapsed:.2f}s - {resp.text[:30]}")
    except Exception as e:
        print(f"Call {i+1}: Error - {str(e)[:150]}")
    
    if i < 2:  # Don't sleep after last call
        time.sleep(5)
