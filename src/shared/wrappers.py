import json
import os
import gym
import numpy as np
import re
import string
from decimal import Decimal, InvalidOperation
from collections import Counter

    
DATA_DIR = "data"
HOTPOTQA_SPLIT_FILE = {
  "train": "hotpotqa/hotpot_train_v1.1_simplified.json",
  "dev": "hotpotqa/hotpot_dev_distractor_v1.json",
  "test": "hotpotqa/hotpot_test_v1_simplified.json",
}

FEVER_SPLIT_FILE = {
  "train": "fever/train.jsonl",
  "dev": "fever/paper_dev.jsonl",
}

GSM8K_SPLIT_FILE = {
  "train": "gsm8k/train.jsonl",
  "test": "gsm8k/test.jsonl",
}

GSM8K_ANS_RE = re.compile(r"####\s*([^\n]+)")
GSM8K_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def normalize_gsm8k_number(value):
  """Normalize a numeric string to a canonical form for exact GSM8K matching."""
  if value is None:
    return None

  text = str(value).strip()
  if not text:
    return None

  text = text.replace(",", "")
  if text.startswith("+"):
    text = text[1:]
  # Normalize trailing-dot forms such as "42.".
  if re.fullmatch(r"-?\d+\.", text):
    text = text[:-1]

  if not re.fullmatch(r"-?\d*\.?\d+", text):
    return text.lower()

  try:
    dec = Decimal(text)
  except InvalidOperation:
    return text.lower()

  normalized = format(dec.normalize(), "f")
  if "." in normalized:
    normalized = normalized.rstrip("0").rstrip(".")
  if normalized in {"", "-0"}:
    normalized = "0"
  return normalized


def parse_gsm8k_ground_truth_answer(answer_text):
  """
  Extract the official GSM8K final answer from a solution string using `#### ...`.
  """
  if answer_text is None:
    return None
  match = GSM8K_ANS_RE.search(str(answer_text))
  if not match:
    return None
  return normalize_gsm8k_number(match.group(1))


def parse_gsm8k_prediction_answer(prediction_text):
  """
  Extract a predicted numeric answer using GSM8K-style parsing with robust fallback.
  """
  if prediction_text is None:
    return None

  text = str(prediction_text).strip()
  if not text:
    return None

  tagged_match = GSM8K_ANS_RE.search(text)
  if tagged_match:
    return normalize_gsm8k_number(tagged_match.group(1))

  numeric_matches = GSM8K_NUM_RE.findall(text)
  if numeric_matches:
    return normalize_gsm8k_number(numeric_matches[-1])

  return text.lower()


def gsm8k_answers_match(prediction_text, ground_truth_text):
  """
  Compare predicted and ground-truth GSM8K answers with official-style normalization.
  """
  pred = parse_gsm8k_prediction_answer(prediction_text)
  gt = parse_gsm8k_ground_truth_answer(ground_truth_text)
  if pred is None or gt is None:
    return 0
  return int(pred == gt)


class HistoryWrapper(gym.ObservationWrapper):
  def __init__(self, env, obs_format, prompt=None):
    super().__init__(env)
    assert obs_format in ["obs", "history"]
    if obs_format == "history":
      assert hasattr(self.env, "traj")
    self.obs_format = obs_format
    self.prompt = prompt if prompt is not None else ""

  def observation(self, obs):
    if self.obs_format == "obs":
      return obs
    elif self.obs_format == "history":
      observation = self.env.traj["observations"][0] + "\n"
      for i, (o, a) in enumerate(zip(self.env.traj["observations"][1:], self.env.traj["actions"]), 1):
        observation += f"Action {i}: {a}\nObservation {i}: {o}\n\n"
      return self.prompt + observation
    

def normalize_answer(s):
  def remove_articles(text):
    return re.sub(r"\b(a|an|the)\b", " ", text)
  
  def white_space_fix(text):
      return " ".join(text.split())

  def remove_punc(text):
      exclude = set(string.punctuation)
      return "".join(ch for ch in text if ch not in exclude)

  def lower(text):
      return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
  normalized_prediction = normalize_answer(prediction)
  normalized_ground_truth = normalize_answer(ground_truth)

  ZERO_METRIC = (0, 0, 0)

  if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
    return ZERO_METRIC
  if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
    return ZERO_METRIC
  
  prediction_tokens = normalized_prediction.split()
  ground_truth_tokens = normalized_ground_truth.split()
  common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
  num_same = sum(common.values())
  if num_same == 0:
    return ZERO_METRIC
  precision = 1.0 * num_same / len(prediction_tokens)
  recall = 1.0 * num_same / len(ground_truth_tokens)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1, precision, recall
  
class HotPotQAWrapper(gym.Wrapper):
  def __init__(self, env, split):
    super().__init__(env)
    # Use absolute path relative to this file's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up to project root (shared -> src -> react-research)
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    data_file = os.path.join(project_root, DATA_DIR, HOTPOTQA_SPLIT_FILE[split])
    self.data = json.load(open(data_file))
    self.data = [(d['question'], d['answer']) for d in self.data]
    self.data_idx = 0
    self.split = split

  def reset(self, seed=None, return_info=False, options=None, idx=None):
    self.env.reset(seed=seed, return_info=return_info, options=options)
    try:
      self.env.step('')
    except:
      pass
    self.env.reset(seed=seed, return_info=return_info, options=options)
    self.data_idx = int(np.random.randint(len(self.data))) if idx is None else idx
    observation = f"Question: {self.data[self.data_idx][0]}"
    info = self._get_info()
    return (observation, info) if return_info else observation

  def _get_info(self):
    return {
      "steps": self.steps, 
      "answer": self.answer,
      "question": self.data[self.data_idx][0], 
      "hotpot_split": self.split
    }

  def get_reward(self, info):
    if info['answer'] is not None:
      pred = normalize_answer(self.data[self.data_idx][1])
      gt = normalize_answer(info['answer'])
      score = (pred == gt)
      return int(score)
    return 0
  
  def get_metrics(self, info):
    if info['answer'] is not None:
      pred = normalize_answer(self.data[self.data_idx][1])
      gt = normalize_answer(info['answer'])
      em = (pred == gt)
      f1 = f1_score(pred, gt)[0]
      return {'reward': em, 'em': em, 'f1': f1}
    return {'reward': 0, 'em': 0, 'f1': 0}

  def step(self, action):
    # TODO: first step obs does not have question. 
    obs, _, done, info = self.env.step(action)
    reward = self.get_reward(info)
    if done:
      obs = f"Episode finished, reward = {reward}\n"
      info.update({"gt_answer": self.data[self.data_idx][1], "question_idx": self.data_idx})
      info.update(self.get_metrics(info))
    return obs, reward, done, info
  
  def __len__(self):
    return len(self.data)

class FeverWrapper(gym.Wrapper):
  def __init__(self, env, split):
    super().__init__(env)
    
    # Use absolute path relative to this file's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up to project root (shared -> src -> react-research)
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    data_path = os.path.join(project_root, "data", FEVER_SPLIT_FILE[split])
    
    with open(data_path, "r") as json_file:
      json_list = list(json_file)

    data = []
    for json_str in json_list:
      json_str = json.loads(json_str)
      label = json_str["label"]
      claim = json_str["claim"]
      data.append((claim, label))

    self.data = data
    self.data_idx = 0
    self.split = split

  def reset(self, seed=None, return_info=False, options=None, idx=None):
    self.env.reset(seed=seed, return_info=return_info, options=options)
    try:
      self.env.step('')
    except:
      pass
    self.env.reset(seed=seed, return_info=return_info, options=options)
    self.data_idx = int(np.random.randint(len(self.data))) if idx is None else idx
    observation = f"Claim: {self.data[self.data_idx][0]}"
    info = self._get_info()
    return (observation, info) if return_info else observation

  def _get_info(self):
    return {
      "steps": self.steps, 
      "answer": self.answer,
      "question": self.data[self.data_idx][0], 
      "fever_split": self.split
    }

  def get_reward(self, info):
    if info['answer'] is not None:
      label = normalize_answer(self.data[self.data_idx][1])
      pred = normalize_answer(info['answer'])
      if label == pred:
        return 1
    return 0

  def step(self, action):
    # TODO: first step obs does not have question. 
    obs, _, done, info = self.env.step(action)
    reward = self.get_reward(info)
    if done:
      obs = f"Episode finished, reward = {reward}\n"
      info.update({"gt_answer": self.data[self.data_idx][1], "question_idx": self.data_idx})
      info.update({'em': reward, 'reward': reward, 'f1': reward})
    return obs, reward, done, info
    
  def __len__(self):
    return len(self.data)


class GSM8KWrapper(gym.Wrapper):
  def __init__(self, env, split):
    super().__init__(env)
    if split not in GSM8K_SPLIT_FILE:
      raise ValueError(f"Unknown GSM8K split: {split}. Expected one of {list(GSM8K_SPLIT_FILE)}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    data_path = os.path.join(project_root, DATA_DIR, GSM8K_SPLIT_FILE[split])

    with open(data_path, "r", encoding="utf-8") as json_file:
      json_list = list(json_file)

    data = []
    for json_str in json_list:
      record = json.loads(json_str)
      question = record["question"]
      raw_answer = record["answer"]
      gt_answer = parse_gsm8k_ground_truth_answer(raw_answer)
      data.append((question, raw_answer, gt_answer))

    self.data = data
    self.data_idx = 0
    self.split = split

  def reset(self, seed=None, return_info=False, options=None, idx=None):
    self.env.reset(seed=seed, return_info=return_info, options=options)
    try:
      self.env.step('')
    except:
      pass
    self.env.reset(seed=seed, return_info=return_info, options=options)
    self.data_idx = int(np.random.randint(len(self.data))) if idx is None else idx
    observation = f"Question: {self.data[self.data_idx][0]}"
    info = self._get_info()
    return (observation, info) if return_info else observation

  def _get_info(self):
    return {
      "steps": self.steps,
      "answer": self.answer,
      "question": self.data[self.data_idx][0],
      "gsm8k_split": self.split
    }

  def get_reward(self, info):
    return gsm8k_answers_match(info.get('answer'), self.data[self.data_idx][1])

  def get_metrics(self, info):
    reward = self.get_reward(info)
    return {'reward': reward, 'em': reward, 'f1': reward}

  def step(self, action):
    obs, _, done, info = self.env.step(action)
    reward = self.get_reward(info)
    if done:
      obs = f"Episode finished, reward = {reward}\n"
      info.update({
        "gt_answer": self.data[self.data_idx][2],
        "gt_answer_raw": self.data[self.data_idx][1],
        "question_idx": self.data_idx
      })
      info.update(self.get_metrics(info))
    return obs, reward, done, info

  def __len__(self):
    return len(self.data)
  
  
class LoggingWrapper(gym.Wrapper):
  def __init__(self, env, folder="trajs", file_id=None):
    super().__init__(env)
    self.trajs = []
    self.traj = {"observations": [], "actions": []}
    self.folder = folder
    self.file_id = np.random.randint(0, 10000000) if file_id is None else file_id
    self.file_path = f"{self.folder}/{self.file_id}.json"
    os.makedirs("trajs", exist_ok=True)

  def __len__(self):
    return len(self.env.data)
  

  def reset(self, seed=None, return_info=False, options=None, idx=None):
    output = self.env.reset(seed=seed, return_info=return_info, options=options, idx=idx)
    observation = output[0] if return_info else output
    self.traj = {"observations": [observation], "actions": []}
    return output

  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    self.traj["observations"].append(obs)
    self.traj["actions"].append(action)
    if done:
      self.traj.update(info)
    return obs, reward, done, info

  def update_record(self):
    if len(self.traj) > 0:
      self.trajs.append(self.traj)
      self.traj = {"observations": [], "actions": []}
  
  def write(self):
    self.update_record()
    with open(self.file_path, "w") as f:
      json.dump(self.trajs, f)
      print(f"Saved trajs to trajs/{self.file_id}.json")
    
  def close(self):
    self.write()
