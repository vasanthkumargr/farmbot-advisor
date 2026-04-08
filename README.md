---
title: Farmbot Advisor
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: docker
pinned: false
license: mit
---

# Farmbot Advisor — OpenEnv Environment

An RL environment simulating real-world agricultural decision-making for Indian farmers.
An AI agent learns to advise on irrigation, fertilization, and harvest timing
based on live farm conditions including soil, weather, and market data.

## Tasks

| Task | Difficulty | Description |
|------|-----------|-------------|
| irrigation_decision | Easy | Decide whether to irrigate based on soil moisture and temperature |
| fertilizer_recommendation | Medium | Recommend fertilizer type and quantity based on crop stage |
| harvest_timing | Hard | Choose optimal harvest window using crop maturity and market price |

## Observation Space

- `soil_moisture` — float [0.0, 1.0]
- `temperature` — float in Celsius
- `crop_stage` — one of: sowing, vegetative, flowering, harvest
- `days_since_planting` — integer
- `weather_forecast` — list of 7 floats (rainfall in mm)
- `market_price` — float in Rs per kg

## Action Space

Natural language string. Examples:
- "Irrigate the field now"
- "Apply nitrogen fertilizer at 50kg/acre"
- "Wait 7 more days before harvest"

## Reward

All rewards in range [0.0, 1.0] with partial credit for partially correct decisions.

## Setup

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

## Run Inference

```bash
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3"
export HF_TOKEN="your_token_here"
python inference.py
```

## Docker

```bash
docker build -t farmbot-advisor .
docker run -p 7860:7860 farmbot-advisor
```