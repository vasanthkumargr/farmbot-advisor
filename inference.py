"""
Inference Script for Farmbot Advisor — OpenEnv Environment.

Runs an LLM agent through all three tasks and emits structured
stdout logs in the required [START]/[STEP]/[END] format.

Required environment variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
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
SUCCESS_SCORE_THRESHOLD = 0.5  # score in [0,1] needed to count as success

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
    done_val = str(done).lower()
    # Truncate action to keep line readable; strip newlines
    action_clean = action.replace("\n", " ").strip()[:120]
    print(f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# ── Environment HTTP helpers ───────────────────────────────────────────────────

def reset_env(task_id: str) -> dict:
    try:
        r = requests.post(f"{BASE_URL}/reset", json={"task_id": task_id}, timeout=30)
        return r.json()
    except Exception as e:
        return {
            "task_id": task_id, "step": 0, "soil_moisture": 0.5,
            "temperature": 30.0, "crop_stage": "vegetative",
            "days_since_planting": 50, "weather_forecast": [10.0] * 7,
            "market_price": 40.0, "done": False, "reward": 0.0,
        }


def step_env(task_id: str, recommendation: str) -> dict:
    try:
        r = requests.post(
            f"{BASE_URL}/step",
            json={"recommendation": recommendation, "task_id": task_id},
            timeout=30,
        )
        return r.json()
    except Exception as e:
        return {"task_id": task_id, "step": 99, "done": True, "reward": 0.0}

# ── LLM action selection ───────────────────────────────────────────────────────

def get_recommendation(task_id: str, obs: dict) -> str:
    try:
        prompt = f"""You are an expert agricultural advisor helping Indian farmers.

Task: {task_id}
Current farm conditions:
- Soil moisture: {obs.get('soil_moisture', 0.5)} (0=dry, 1=wet)
- Temperature: {obs.get('temperature', 30.0)}C
- Crop stage: {obs.get('crop_stage', 'vegetative')}
- Days since planting: {obs.get('days_since_planting', 50)}
- 7-day rainfall forecast (mm): {obs.get('weather_forecast', [10]*7)}
- Market price: Rs {obs.get('market_price', 40.0)} per kg

Provide one concise farming recommendation.
For irrigation_decision: say "Irrigate the field now" or "No irrigation needed today"
For fertilizer_recommendation: say e.g. "Apply nitrogen fertilizer at 50kg/acre"
For harvest_timing: say "Harvest the crop today" or "Wait X more days before harvest"
"""
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        fallbacks = {
            "irrigation_decision": "Irrigate the field now",
            "fertilizer_recommendation": "Apply nitrogen fertilizer at 50kg/acre",
            "harvest_timing": "Wait 7 more days before harvest",
        }
        return fallbacks.get(task_id, "Irrigate the field now")

# ── Episode runner ─────────────────────────────────────────────────────────────

def run_episode(task_id: str, episode: int) -> float:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = reset_env(task_id)

        while not obs.get("done", False):
            action = get_recommendation(task_id, obs)
            error_msg = None

            try:
                obs = step_env(task_id, action)
            except Exception as e:
                error_msg = str(e)
                obs = {"done": True, "reward": 0.0}

            reward = float(obs.get("reward", 0.0))
            done = bool(obs.get("done", False))
            rewards.append(reward)
            steps_taken += 1

            log_step(step=steps_taken, action=action, reward=reward, done=done, error=error_msg)

            if done:
                break

        # Score = average reward per step, clamped to [0, 1]
        score = sum(rewards) / max(len(rewards), 1)
        score = round(min(max(score, 0.0), 1.0), 3)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Episode error task={task_id} episode={episode}: {e}", file=sys.stderr, flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score

# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for task_id in TASKS:
        for episode in range(1, 3):
            run_episode(task_id=task_id, episode=episode)
