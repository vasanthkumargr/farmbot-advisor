"""
Inference Script for Farmbot Advisor — OpenEnv Environment.

Runs an LLM agent through all three tasks and emits structured
stdout logs in the required [START]/[STEP]/[END] format.

Required environment variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

NOTE: The environment server uses openenv-core's create_app() which wraps
responses in {"observation": {...}, "done": bool, "reward": float} format
and requires actions as {"action": {"recommendation": ..., "task_id": ...}}.
"""

import os
import sys
from typing import List, Optional

import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")
HF_TOKEN = os.getenv("HF_TOKEN")
BASE_URL = os.getenv("BASE_URL", "http://localhost:7860")

BENCHMARK = "farmbot_advisor"
SUCCESS_SCORE_THRESHOLD = 0.5
MAX_STEPS_SAFETY = 4  # hard cap — prevents infinite loops if done never arrives

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

TASKS = [
    "irrigation_decision",
    "fertilizer_recommendation",
    "harvest_timing",
]

# ── Structured stdout log helpers ──────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    action_clean = action.replace("\n", " ").strip()[:100]
    print(f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# ── Environment HTTP helpers ───────────────────────────────────────────────────
# openenv-core create_app wraps responses as:
#   /reset → {"observation": {...}, "done": bool, "reward": float}
#   /step  → {"observation": {...}, "done": bool, "reward": float}
# and requires step payload as:
#   {"action": {"recommendation": "...", "task_id": "..."}}

def _parse_response(data: dict) -> tuple:
    """Extract (obs_dict, done, reward) from openenv-core wrapped response."""
    # Try wrapped format first
    if "observation" in data:
        obs = data["observation"]
        done = bool(data.get("done", False))
        reward = float(data.get("reward", 0.0))
    else:
        # Fallback: flat format (legacy)
        obs = data
        done = bool(data.get("done", False))
        reward = float(data.get("reward", 0.0))
    return obs, done, reward


def reset_env(task_id: str) -> tuple:
    """Returns (obs_dict, done, reward)."""
    try:
        r = requests.post(
            f"{BASE_URL}/reset",
            json={"task_id": task_id},
            timeout=15,
        )
        return _parse_response(r.json())
    except Exception as e:
        fallback_obs = {
            "task_id": task_id, "step": 0, "soil_moisture": 0.5,
            "temperature": 30.0, "crop_stage": "vegetative",
            "days_since_planting": 50, "weather_forecast": [10.0] * 7,
            "market_price": 40.0,
        }
        return fallback_obs, False, 0.0


def step_env(task_id: str, recommendation: str) -> tuple:
    """Returns (obs_dict, done, reward)."""
    try:
        r = requests.post(
            f"{BASE_URL}/step",
            json={"action": {"recommendation": recommendation, "task_id": task_id}},
            timeout=15,
        )
        return _parse_response(r.json())
    except Exception as e:
        return {}, True, 0.0

# ── LLM action selection ───────────────────────────────────────────────────────

FALLBACKS = {
    "irrigation_decision": "Irrigate the field now",
    "fertilizer_recommendation": "Apply nitrogen fertilizer at 50kg/acre",
    "harvest_timing": "Wait 7 more days before harvest",
}

def get_recommendation(task_id: str, obs: dict) -> str:
    try:
        prompt = (
            f"You are a farming advisor for Indian farmers.\n"
            f"Task: {task_id}\n"
            f"Soil moisture: {obs.get('soil_moisture', 0.5):.2f}, "
            f"Temp: {obs.get('temperature', 30.0):.1f}C, "
            f"Stage: {obs.get('crop_stage', 'vegetative')}, "
            f"Days: {obs.get('days_since_planting', 50)}, "
            f"Price: Rs{obs.get('market_price', 40.0):.0f}/kg\n"
            f"For irrigation_decision: say 'Irrigate the field now' or 'No irrigation needed today'\n"
            f"For fertilizer_recommendation: say 'Apply [type] fertilizer at [N]kg/acre'\n"
            f"For harvest_timing: say 'Harvest the crop today' or 'Wait X more days before harvest'\n"
            f"Reply with one short sentence only."
        )
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=40,
            timeout=10,
        )
        text = response.choices[0].message.content.strip()
        return text if text else FALLBACKS.get(task_id, "Irrigate the field now")
    except Exception:
        return FALLBACKS.get(task_id, "Irrigate the field now")

# ── Episode runner ─────────────────────────────────────────────────────────────

def run_episode(task_id: str, episode: int) -> float:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs, done, _ = reset_env(task_id)

        while not done and steps_taken < MAX_STEPS_SAFETY:
            action = get_recommendation(task_id, obs)
            error_msg = None

            try:
                obs, done, reward = step_env(task_id, action)
            except Exception as e:
                error_msg = str(e)
                done = True
                reward = 0.0

            rewards.append(reward)
            steps_taken += 1
            log_step(step=steps_taken, action=action, reward=reward, done=done, error=error_msg)

        # Score = average reward, clamped to [0, 1]
        score = sum(rewards) / max(len(rewards), 1)
        score = round(min(max(score, 0.0), 1.0), 3)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] task={task_id} episode={episode} error={e}", file=sys.stderr, flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score

# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for task_id in TASKS:
        run_episode(task_id=task_id, episode=1)
