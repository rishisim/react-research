import os
import time
from tenacity import retry, stop_after_attempt, wait_random_exponential
from typing import List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load .env from project root
env_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '.env')
load_dotenv(env_path)

api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set")
client = genai.Client(api_key=api_key)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def llm_call(prompt: str, stop: Optional[List[str]] = None, num_traces: int = 1, model: str = "gemini-2.5-flash-lite", max_tokens: int = 400) -> str:
    # Throttle calls slightly to respect service limits
    time.sleep(4.1)
    temperature = 0.0 if num_traces == 1 else 0.7
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            stop_sequences=stop,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
    )
    return response.text
