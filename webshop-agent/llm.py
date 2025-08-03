import sys
import time
from google import genai
from google.genai import types

# --- Gemini API Configuration ---
try:
    client = genai.Client()
except Exception as e:
    print(f"ERROR: Failed to initialize Gemini client: {e}")
    sys.exit(1)

def call_llm(prompt, stop=None, num_traces=1):
    if stop is None: stop = ["\n"]
    time.sleep(10)
    temperature_setting = 0.0 if num_traces == 1 else 0.7
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                stop_sequences=stop,
                temperature=temperature_setting,
                max_output_tokens=300,
            )
        )
        return response.text
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return ""
