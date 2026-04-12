"""
Inference Script
===================================
Farmbot Advisor Environment

This script evaluates the model against the Farmbot environment.
"""

import os
import sys
import time
from typing import List, Optional

import requests
from openai import OpenAI

# ─────────────────────────────────────────────────────────────────────────────
# Environment Variables (Required by Competition)
# ─────────────────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")
HF_TOKEN = os.getenv("HF_TOKEN")

# The space URL or localhost if running locally
BASE_URL = os.getenv("BASE_URL", "http://localhost:7860")

# Setup configuration
BENCHMARK = "farmbot_advisor"
TASKS = ["irrigation_decision", "fertilizer_recommendation", "harvest_timing"]
SUCCESS_SCORE_THRESHOLD = 0.5
MAX_STEPS = 4

# Initialize OpenAI client
client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

# ─────────────────────────────────────────────────────────────────────────────
# STDOUT Logging (Strict Competition Format)
# ─────────────────────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = "true" if done else "false"
    # Action must be on one line, stripped
    action_clean = action.replace("\n", " ").strip()[:80]
    # NOTE: Output format MUST exactly match:
    # [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    print(f"[STEP]  step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    success_val = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    # [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
    print(f"[END]   success={success_val} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# HTTP Environment API
# ─────────────────────────────────────────────────────────────────────────────

def _parse_response(data: dict) -> tuple:
    """Parse HTTP openenv-core wrapped response."""
    if "observation" in data:
        obs = data["observation"]
        done = bool(data.get("done", False))
        reward = float(data.get("reward", 0.0))
    else:
        obs = data
        done = bool(data.get("done", False))
        reward = float(data.get("reward", 0.0))
    return obs, done, reward


def reset_env(task_id: str) -> tuple:
    r = requests.post(
        f"{BASE_URL}/reset",
        json={"task_id": task_id},
        timeout=30,
    )
    r.raise_for_status()
    return _parse_response(r.json())


def step_env(task_id: str, action_str: str) -> tuple:
    r = requests.post(
        f"{BASE_URL}/step",
        json={"action": {"recommendation": action_str, "task_id": task_id}},
        timeout=30,
    )
    r.raise_for_status()
    return _parse_response(r.json())


# ─────────────────────────────────────────────────────────────────────────────
# LLM Logic
# ─────────────────────────────────────────────────────────────────────────────

def get_action(task_id: str, obs: dict) -> str:
    prompt = (
        f"You are a farming AI acting in task '{task_id}'.\n"
        f"Soil moisture: {obs.get('soil_moisture', 0.5):.2f}\n"
        f"Temperature: {obs.get('temperature', 30.0):.1f}C\n"
        f"Crop stage: {obs.get('crop_stage', 'sowing')}\n"
        f"Days planted: {obs.get('days_since_planting', 10)}\n"
        f"Provide one action sentence in double quotes. Do not explain. Options:\n"
        f"- 'Irrigate the field now' OR 'No irrigation needed today'\n"
        f"- 'Apply nitrogen/phosphorus/potassium fertilizer at 50kg/acre'\n"
        f"- 'Harvest the crop today' OR 'Wait before harvest'\n"
    )
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0.1,
            timeout=15,
        )
        return response.choices[0].message.content.strip().strip("'\"")
    except Exception as e:
        print(f"[DEBUG] LLM Failed: {e}", file=sys.stderr, flush=True)
        # Fallbacks for graceful degradation
        return "Wait"


# ─────────────────────────────────────────────────────────────────────────────
# Execution
# ─────────────────────────────────────────────────────────────────────────────

def run_episode(task_id: str) -> None:
    steps_taken = 0
    score = 0.0
    success = False
    rewards: List[float] = []

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs, done, last_reward = reset_env(task_id)

        # Environment loop - hard exit if we exceed max steps
        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            action = get_action(task_id, obs)
            error_msg = None

            try:
                obs, done, reward = step_env(task_id, action)
            except Exception as e:
                error_msg = str(e).replace("\n", " ")
                reward = 0.0
                done = True

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            log_step(step=step, action=action, reward=reward, done=done, error=error_msg)

        # Average step rewards
        avg_reward = sum(rewards) / max(len(rewards), 1)
        score = max(0.0, min(1.0, avg_reward))
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Env connection error: {e}", file=sys.stderr, flush=True)
        # If env is unreachable, we must still output an [END] block per rules
        pass

    finally:
        # Mandatory [END] line always emitted
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    # Give the evaluator server a moment to spin up if ran locally in parallel
    time.sleep(2)
    
    for task_id in TASKS:
        run_episode(task_id)


if __name__ == "__main__":
    main()
