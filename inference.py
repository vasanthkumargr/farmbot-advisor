"""
Inference script for the Farmbot Advisor environment.

Runs an LLM agent through all three tasks of the Farmbot environment
and logs structured output in the required [START]/[STEP]/[END] format.

Environment Variables:
    API_BASE_URL: HuggingFace inference API base URL
    MODEL_NAME: Model to use for inference
    HF_TOKEN: HuggingFace API token (required, no default)
"""

import json
import os

import requests
from openai import OpenAI

# Required environment variables — API_BASE_URL and MODEL_NAME have defaults,
# HF_TOKEN must be supplied in the submission environment.
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")
HF_TOKEN = os.getenv("HF_TOKEN")

BASE_URL = "https://Metsuu13-farmbot-advisor.hf.space"

client = OpenAI(
    api_key=HF_TOKEN,
    base_url=API_BASE_URL,
)

TASKS = [
    "irrigation_decision",
    "fertilizer_recommendation",
    "harvest_timing",
]


def reset_env(task_id: str) -> dict:
    """Reset the environment for a given task."""
    try:
        response = requests.post(
            f"{BASE_URL}/reset",
            json={"task_id": task_id},
            timeout=30,
        )
        return response.json()
    except Exception as e:
        print(json.dumps({"type": "[ERROR]", "stage": "reset", "task_id": task_id, "error": str(e)}))
        return {
            "task_id": task_id, "step": 0, "soil_moisture": 0.5,
            "temperature": 30.0, "crop_stage": "vegetative",
            "days_since_planting": 50, "weather_forecast": [10.0] * 7,
            "market_price": 40.0, "done": False, "reward": 0.0, "message": "",
        }


def step_env(task_id: str, recommendation: str) -> dict:
    """Execute one step in the environment."""
    try:
        response = requests.post(
            f"{BASE_URL}/step",
            json={"recommendation": recommendation, "task_id": task_id},
            timeout=30,
        )
        return response.json()
    except Exception as e:
        print(json.dumps({"type": "[ERROR]", "stage": "step", "task_id": task_id, "error": str(e)}))
        return {
            "task_id": task_id, "step": 99, "soil_moisture": 0.5,
            "temperature": 30.0, "crop_stage": "vegetative",
            "days_since_planting": 50, "weather_forecast": [10.0] * 7,
            "market_price": 40.0, "done": True, "reward": 0.0, "message": "",
        }


def get_recommendation(task_id: str, obs: dict) -> str:
    """Query the LLM for a farming recommendation based on current observations."""
    try:
        prompt = f"""You are an expert agricultural advisor helping Indian farmers.

Task: {task_id}
Current farm conditions:
- Soil moisture: {obs.get('soil_moisture', 0.5)} (0=completely dry, 1=waterlogged)
- Temperature: {obs.get('temperature', 30.0)}C
- Crop stage: {obs.get('crop_stage', 'vegetative')}
- Days since planting: {obs.get('days_since_planting', 50)}
- 7-day rainfall forecast (mm): {obs.get('weather_forecast', [10]*7)}
- Market price: Rs {obs.get('market_price', 40.0)} per kg

Provide a specific, actionable farming recommendation.
For irrigation_decision: respond with "Irrigate the field now" or "No irrigation needed today"
For fertilizer_recommendation: specify fertilizer type and quantity, e.g. "Apply nitrogen fertilizer at 50kg/acre"
For harvest_timing: respond with "Harvest the crop today" or "Wait X more days before harvest"
Keep your response concise and include specific quantities where relevant.
"""
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(json.dumps({"type": "[ERROR]", "stage": "llm", "task_id": task_id, "error": str(e)}))
        # Deterministic fallback so the episode still completes and scores
        fallbacks = {
            "irrigation_decision": "Irrigate the field now",
            "fertilizer_recommendation": "Apply nitrogen fertilizer at 50kg/acre",
            "harvest_timing": "Wait 7 more days before harvest",
        }
        return fallbacks.get(task_id, "Irrigate the field now")


def run_episode(task_id: str, episode: int) -> float:
    """Run one full episode for a given task and return total reward."""
    try:
        obs = reset_env(task_id)

        print(json.dumps({
            "type": "[START]",
            "task_id": task_id,
            "episode": episode,
            "initial_state": {
                "soil_moisture": obs.get("soil_moisture"),
                "temperature": obs.get("temperature"),
                "crop_stage": obs.get("crop_stage"),
                "days_since_planting": obs.get("days_since_planting"),
                "market_price": obs.get("market_price"),
            },
        }))

        total_reward = 0.0
        step_num = 0

        while not obs.get("done", False):
            recommendation = get_recommendation(task_id, obs)
            obs = step_env(task_id, recommendation)
            reward = float(obs.get("reward", 0.0))
            total_reward += reward
            step_num += 1

            print(json.dumps({
                "type": "[STEP]",
                "task_id": task_id,
                "episode": episode,
                "step": step_num,
                "action": recommendation[:150],
                "reward": reward,
                "done": obs.get("done", False),
                "message": obs.get("message", ""),
            }))

        print(json.dumps({
            "type": "[END]",
            "task_id": task_id,
            "episode": episode,
            "total_reward": round(total_reward, 3),
            "steps": step_num,
        }))

        return total_reward

    except Exception as e:
        print(json.dumps({"type": "[ERROR]", "task_id": task_id, "episode": episode, "error": str(e)}))
        return 0.0


if __name__ == "__main__":
    all_results = []

    for task_id in TASKS:
        for episode in range(1, 3):
            total_reward = run_episode(task_id, episode)
            all_results.append({
                "task_id": task_id,
                "episode": episode,
                "total_reward": round(total_reward, 3),
            })

    print(json.dumps({
        "type": "[SUMMARY]",
        "results": all_results,
        "total_tasks": len(TASKS),
        "total_episodes": len(all_results),
    }))
