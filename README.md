---
title: Farmbot Advisor
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: docker
pinned: false
license: mit
---

# 🌾 Farmbot Advisor — OpenEnv RL Environment

> An AI agent learns to make real-world agricultural decisions for Indian smallholder farmers — advising on **irrigation**, **fertilization**, and **harvest timing** based on live soil, weather, and market conditions.

**Team:** BinaryBrains | **Event:** Meta PyTorch Hackathon × Scaler School of Technology, Round 1

🔗 **Live Space:** [https://huggingface.co/spaces/Metsuu13/farmbot-advisor](https://huggingface.co/spaces/Metsuu13/farmbot-advisor)

---

## 🌍 Why This Environment?

Indian smallholder farmers manage crops with limited access to expert advice, real-time data, or market intelligence. Poor irrigation and fertilization decisions cause significant yield and income loss every season. This environment simulates the decision-making loop a farm advisory AI agent must master — turning sensor readings into actionable, crop-stage-aware recommendations.

This is a **genuine real-world task**, not a toy or game. The agent must reason about:
- Soil hydration vs. evapotranspiration stress thresholds
- Crop nutritional requirements at each growth stage
- Market timing to maximise harvest revenue

---

## 📋 Tasks

| Task ID | Difficulty | Description |
|---|---|---|
| `irrigation_decision` | 🟢 Easy | Decide whether to irrigate based on soil moisture (< 0.3 = dry) and temperature (> 38 °C = stress) |
| `fertilizer_recommendation` | 🟡 Medium | Recommend the correct fertilizer type and quantity for the current crop growth stage |
| `harvest_timing` | 🔴 Hard | Decide whether to harvest now or wait, balancing crop maturity (≥ 85 days) against market price (> Rs 40/kg) |

Each task runs for **7 steps per episode**. The environment state evolves at every step — soil moisture decreases, temperature drifts, the weather forecast shifts, and days advance.

---

## 🔭 Observation Space

Each call to `/reset` or `/step` returns a `FarmState` object:

| Field | Type | Range / Values | Description |
|---|---|---|---|
| `task_id` | `str` | one of 3 task IDs | Active task identifier |
| `step` | `int` | 0 – 7 | Current step within the episode |
| `soil_moisture` | `float` | [0.0, 1.0] | 0 = bone dry, 1 = fully saturated |
| `temperature` | `float` | Celsius | Ambient air temperature |
| `crop_stage` | `str` | sowing / vegetative / flowering / harvest | Current crop growth stage (randomised each episode) |
| `days_since_planting` | `int` | 10 – 90+ | Running crop age in days (increments each step) |
| `weather_forecast` | `list[float]` | 7 values in mm/day | Rolling 7-day rainfall forecast |
| `market_price` | `float` | Rs per kg | Current crop market price |
| `reward` | `float` | (0.05, 0.95) | Reward received for the previous action |
| `done` | `bool` | — | True when episode ends (step >= 7) |

---

## 🎮 Action Space

Actions are **natural language strings** — the agent expresses its recommendation in plain English.

**Grader-recognised keywords by task:**

| Task | Effective Action Examples |
|---|---|
| `irrigation_decision` | `"Irrigate the field now"` · `"No irrigation needed today"` |
| `fertilizer_recommendation` | `"Apply nitrogen fertilizer at 50 kg/acre"` · `"Apply phosphorus at 30 kg/acre"` |
| `harvest_timing` | `"Harvest the crop today"` · `"Wait 10 more days before harvest"` |

---

## 🏆 Reward Design

All rewards are **strictly in (0.05, 0.95)** — never exactly 0 or 1 — with meaningful partial credit at every decision level.

### `irrigation_decision`

| Condition | Action Contains | Reward |
|---|---|---|
| Should irrigate (moisture < 0.3 or temp > 38 °C) | "irrigate" | 0.92 |
| Should irrigate | "no irrigation" / "don't irrigate" | 0.08 |
| Should irrigate | Other | 0.45 |
| Should NOT irrigate | "no irrigation" / "wait" | 0.92 |
| Should NOT irrigate | "irrigate" | 0.15 |
| Should NOT irrigate | Other | 0.50 |

### `fertilizer_recommendation`

Scoring is additive:

| Match | Score |
|---|---|
| Correct fertilizer for stage (nitrogen/phosphorus/potassium) | +0.55 |
| Any fertilizer mentioned (wrong stage) | +0.20 |
| No fertilizer mentioned | +0.05 |
| Quantity mentioned (kg / grams / per acre) | +0.35 |
| No quantity | +0.05 |

**Stage → correct fertilizer:** `sowing` → phosphorus · `vegetative` → nitrogen · `flowering` → phosphorus · `harvest` → potassium

### `harvest_timing`

| Maturity | Market Price | Best Action | Reward |
|---|---|---|---|
| ≥ 85 days | > Rs 40/kg | harvest | 0.92 |
| ≥ 85 days | ≤ Rs 40/kg | wait | 0.78 |
| < 85 days | any | wait | 0.92 |
| Early harvest attempt | — | harvest | 0.12 |

**Episode score** = average per-step reward across all steps (always strictly within (0, 1)).

---

## 🔌 API Reference

Base URL (deployed): `https://Metsuu13-farmbot-advisor.hf.space`
Base URL (local): `http://localhost:7860`

Interactive API docs: `/docs`

---

### `POST /reset`

Start a new episode for a task. Initialises a fresh randomised `FarmState`.

**Request body:**
```json
{ "task_id": "irrigation_decision" }
```

**Response:** Full `FarmState` JSON object.

---

### `POST /step`

Submit an action and receive the next state with reward.

**Request body:**
```json
{
  "task_id": "irrigation_decision",
  "action": { "action": "Irrigate the field now" }
}
```

**Response:** Updated `FarmState` with new `reward`, evolved observations, incremented `step`, and `done` flag.

---

### `GET /state?task_id=irrigation_decision`

Return the current state of an active session without consuming a step.

---

### `GET /tasks`

List all available tasks.

**Response:**
```json
{
  "tasks": [
    {"id": "irrigation_decision", "difficulty": "easy"},
    {"id": "fertilizer_recommendation", "difficulty": "medium"},
    {"id": "harvest_timing", "difficulty": "hard"}
  ]
}
```

---

## ⚙️ Setup & Running Locally

### Install dependencies

```bash
pip install -r requirements.txt
```

### Start the API server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

Interactive docs: [http://localhost:7860/docs](http://localhost:7860/docs)

---

## 🐳 Docker

```bash
# Build image
docker build -t farmbot-advisor .

# Run container
docker run -p 7860:7860 farmbot-advisor
```

The container starts the FastAPI server on port **7860** automatically.

---

## 🤖 Running Inference

### Required Environment Variables

| Variable | Description | Example |
|---|---|---|
| `API_BASE_URL` | LLM API endpoint | `https://api-inference.huggingface.co/v1` |
| `MODEL_NAME` | Model identifier | `mistralai/Mistral-7B-Instruct-v0.3` |
| `HF_TOKEN` | Hugging Face API token | `hf_...` |

### Run

```bash
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3"
export HF_TOKEN="your_hf_token_here"

python inference.py
```

### Expected stdout format

```
[START] task=irrigation_decision episode=1
[STEP] step=1 reward=0.92 done=False
[STEP] step=2 reward=0.5 done=False
[STEP] step=3 reward=0.92 done=False
[STEP] step=4 reward=0.92 done=False
[STEP] step=5 reward=0.45 done=False
[STEP] step=6 reward=0.92 done=False
[STEP] step=7 reward=0.92 done=True
[END] task=irrigation_decision score=0.793 steps=7
[START] task=fertilizer_recommendation episode=1
...
[SUMMARY] total_tasks=3 total_episodes=6
```

Inference covers **3 tasks × 2 episodes** and completes in under **20 minutes**.

---

## 📁 Project Structure

```
farmbot-advisor/
├── app/
│   ├── main.py          # FastAPI routes: /reset, /step, /state, /tasks
│   ├── environment.py   # State init, step evolution, task graders
│   └── models.py        # Pydantic typed models (FarmState, StepRequest, etc.)
├── inference.py         # Baseline LLM agent inference script
├── openenv.yaml         # OpenEnv spec: tasks, obs/action space, API
├── Dockerfile           # Container definition for HF Spaces
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## 📊 Baseline Results

Running with `mistralai/Mistral-7B-Instruct-v0.3` via Hugging Face Inference API:

| Task | Avg Episode Score |
|---|---|
| `irrigation_decision` | ~0.70 |
| `fertilizer_recommendation` | ~0.55 |
| `harvest_timing` | ~0.65 |

Scores vary per episode due to randomised initial state. More capable models (Llama-3, Qwen-2.5) are expected to score higher on all tasks.

---

## 🔧 Runtime Constraints

| Constraint | Value |
|---|---|
| Max steps per episode | 7 |
| Tasks | 3 |
| Episodes per task | 2 |
| Total inference time | < 20 minutes |
| Min hardware | 2 vCPU, 8 GB RAM |