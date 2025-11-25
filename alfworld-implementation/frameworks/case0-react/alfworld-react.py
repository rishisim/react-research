import os
import sys
import json
import yaml
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
import alfworld
import alfworld.agents.environment

# Resolve paths relative to this file
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load .env from the project root (react-research folder)
env_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '.env')
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

def llm(prompt, stop=None, num_traces=1):
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

# Change to file-relative path:
config_path = os.path.abspath(os.path.join(script_dir, '..', 'base_config.yaml'))
with open(config_path) as reader:
    config = yaml.safe_load(reader)

split = "eval_out_of_distribution"

env = alfworld.agents.environment.get_environment(config["env"]["type"])(config, train_eval=split)
env = env.init_env(batch_size=1)

def process_ob(ob):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]
    return ob

folder = os.path.join(script_dir, 'prompts/')
# Assuming the prompt file from the original code exists relative to this script's location
prompt_file = 'alfworld_3prompts.json'
with open(folder + prompt_file, 'r') as f:
    d = json.load(f)

def alfworld_run(prompt, to_print=True, ob=''):
    init_prompt = prompt + ob + '\n>'
    prompt_str = ''
    if to_print:
        print(ob)
        sys.stdout.flush()
    for i in range(1, 50):
        action = llm(init_prompt + prompt_str, stop=['\n']).strip()
        observation, reward, done, info = env.step([action])
        observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]
        if action.startswith('think:'):
            observation = 'OK.'
        if to_print:
            print(f'Act {i}: {action}\nObs {i}: {observation}')
            sys.stdout.flush()
        prompt_str += f' {action}\n{observation}\n>'
        if done:
            return reward
    return 0

prefixes = {
    'pick_and_place': 'put',
    'pick_clean_then_place': 'clean',
    'pick_heat_then_place': 'heat',
    'pick_cool_then_place': 'cool',
    'look_at_obj': 'examine',
    'pick_two_obj': 'puttwo'
}
cnts = [0] * 6
rs = [0] * 6

# Add a check for the number of episodes, 134 is the total for the eval set.
num_episodes = 1
for i_episode in range(num_episodes):
    ob, info = env.reset()
    ob = '\n'.join(ob[0].split('\n\n')[1:])
    name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])
    print(name)
    r = 0
    for i, (k, v) in enumerate(prefixes.items()):
        if name.startswith(k):
            prompt = 'Interact with a household to solve a task. Here are two examples.\n' + d[f'react_{v}_1'] + d[f'react_{v}_0'] + '\nHere is the task.\n'
            print(k, v)
            r = alfworld_run(prompt, ob=ob)
            rs[i] += r
            cnts[i] += 1
            break
    print(i_episode+1, 'r', r, 'rs', rs, 'cnts', cnts, 'sum(rs)/sum(cnts)', sum(rs) / sum(cnts) if sum(cnts) > 0 else 0)
    print('------------\n')
