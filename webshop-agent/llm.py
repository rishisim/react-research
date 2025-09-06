import sys
import time
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables from .env file in parent directory
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(env_path)

# --- Gemini API Configuration ---
try:
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable not set")
        sys.exit(1)
    client = genai.Client(api_key=api_key)
except Exception as e:
    print(f"ERROR: Failed to initialize Gemini client: {e}")
    sys.exit(1)

def call_llm(prompt, stop=None, num_traces=1):
    if stop is None: stop = ["\n"]
    time.sleep(15)
    temperature_setting = 0.0 if num_traces == 1 else 0.7
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                stop_sequences=stop,
                temperature=temperature_setting,
                max_output_tokens=400,
            )
        )
        return response.text
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return ""
