import os
import json
import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")
HF_TOKEN = os.getenv("HF_TOKEN")

BASE_URL = "https://Metsuu13-farmbot-advisor.hf.space"

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
    response = requests.post(
        f"{BASE_URL}/reset",
        json={"task_id": task_id}
    )
    return response.json()

def step_env(task_id: str, action: str) -> dict:
    response = requests.post(
        f"{BASE_URL}/step",
        json={
            "task_id": task_id,
            "action": {"action": action}
        }
    )
    return response.json()

def get_action(task_id: str, state: dict) -> str:
    prompt = f"""You are an expert agricultural advisor helping Indian farmers.

Task: {task_id}
Current farm conditions:
- Soil moisture: {state['soil_moisture']} (0=dry, 1=wet)
- Temperature: {state['temperature']}C
- Crop stage: {state['crop_stage']}
- Days since planting: {state['days_since_planting']}
- 7-day rainfall forecast (mm): {state['weather_forecast']}
- Market price: Rs {state['market_price']} per kg

Based on these conditions, provide a specific farming recommendation.
Be concise and mention specific quantities where relevant.
For irrigation_decision: say either "Irrigate the field now" or "No irrigation needed today"
For fertilizer_recommendation: say which fertilizer and how much per acre
For harvest_timing: say either "Harvest the crop today" or "Wait X more days before harvest"
"""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    )
    return response.choices[0].message.content.strip()

def run_episode(task_id: str, episode: int) -> float:
    state = reset_env(task_id)

    print(json.dumps({
        "type": "[START]",
        "task_id": task_id,
        "episode": episode,
        "initial_state": {
            "soil_moisture": state["soil_moisture"],
            "temperature": state["temperature"],
            "crop_stage": state["crop_stage"],
            "days_since_planting": state["days_since_planting"],
            "market_price": state["market_price"]
        }
    }))

    total_reward = 0.0
    step_num = 0

    while not state.get("done", False):
        action = get_action(task_id, state)
        state = step_env(task_id, action)
        reward = state.get("reward", 0.0)
        total_reward += reward
        step_num += 1

        print(json.dumps({
            "type": "[STEP]",
            "task_id": task_id,
            "episode": episode,
            "step": step_num,
            "action": action[:150],
            "reward": reward,
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
    all_results = []

    for task_id in TASKS:
        for episode in range(1, 3):
            total_reward = run_episode(task_id, episode)
            all_results.append({
                "task_id": task_id,
                "episode": episode,
                "total_reward": round(total_reward, 3)
            })

    print(json.dumps({
        "type": "[SUMMARY]",
        "results": all_results,
        "total_tasks": len(TASKS),
        "total_episodes": len(all_results)
    }))