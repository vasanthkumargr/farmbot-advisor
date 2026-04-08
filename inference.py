import os
import json
import requests
from openai import OpenAI

API_BASE_URL = os.environ["API_BASE_URL"]
MODEL_NAME = os.environ["MODEL_NAME"]
HF_TOKEN = os.environ["HF_TOKEN"]

BASE_URL = "http://localhost:7860"

client = OpenAI(
    api_key=HF_TOKEN,
    base_url=API_BASE_URL
)

TASKS = [
    "irrigation_decision",
    "fertilizer_recommendation",
    "harvest_timing"
]

def reset_env(task_id: str) -> dict:
    response = requests.post(f"{BASE_URL}/reset", json={"task_id": task_id})
    return response.json()

def step_env(task_id: str, action: str) -> dict:
    response = requests.post(f"{BASE_URL}/step", json={
        "task_id": task_id,
        "action": {"action": action}
    })
    return response.json()

def get_action(task_id: str, state: dict) -> str:
    prompt = f"""You are an expert agricultural advisor helping Indian farmers.

Task: {task_id}
Current farm conditions:
- Soil moisture: {state['soil_moisture']:.2f} (0=dry, 1=wet)
- Temperature: {state['temperature']:.1f}C
- Crop stage: {state['crop_stage']}
- Days since planting: {state['days_since_planting']}
- 7-day rainfall forecast (mm): {state['weather_forecast']}
- Market price: Rs {state['market_price']:.1f} per kg

Provide a specific farming recommendation with quantities where relevant. Be concise.
"""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    )
    return response.choices[0].message.content.strip()

def run_episode(task_id: str, episode: int):
    state = reset_env(task_id)

    print(json.dumps({
        "type": "[START]",
        "task_id": task_id,
        "episode": episode,
        "initial_state": {
            "soil_moisture": state["soil_moisture"],
            "temperature": state["temperature"],
            "crop_stage": state["crop_stage"]
        }
    }))

    total_reward = 0.0
    step_num = 0

    while not state.get("done", False):
        action = get_action(task_id, state)
        state = step_env(task_id, action)
        total_reward += state.get("reward", 0.0)
        step_num += 1

        print(json.dumps({
            "type": "[STEP]",
            "task_id": task_id,
            "episode": episode,
            "step": step_num,
            "action": action[:100],
            "reward": state.get("reward", 0.0),
            "done": state.get("done", False)
        }))

    print(json.dumps({
        "type": "[END]",
        "task_id": task_id,
        "episode": episode,
        "total_reward": round(total_reward, 3),
        "steps": step_num
    }))

    return total_reward

if __name__ == "__main__":
    for task_id in TASKS:
        for episode in range(1, 3):
            run_episode(task_id, episode)