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
An AI agent advises on irrigation, fertilization, and harvest timing based on live
farm conditions including soil moisture, temperature, weather forecasts, and market prices.

## Tasks

| Task | Difficulty | Description |
|------|-----------|-------------|
| `irrigation_decision` | Easy | Decide whether to irrigate based on soil moisture and temperature |
| `fertilizer_recommendation` | Medium | Recommend fertilizer type and quantity based on crop stage |
| `harvest_timing` | Hard | Choose optimal harvest window using crop maturity and market price |

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `soil_moisture` | float [0.0, 1.0] | 0 = completely dry, 1 = waterlogged |
| `temperature` | float (°C) | Ambient temperature |
| `crop_stage` | string | One of: sowing, vegetative, flowering, harvest |
| `days_since_planting` | int | Days since crop was planted |
| `weather_forecast` | list[float] | 7-day rainfall forecast in mm |
| `market_price` | float (Rs/kg) | Current market price in Indian Rupees |

## Action Space

Natural language recommendation string. Examples:
- `"Irrigate the field now"`
- `"No irrigation needed today"`
- `"Apply nitrogen fertilizer at 50kg/acre"`
- `"Apply phosphorus fertilizer at 30kg/acre"`
- `"Harvest the crop today"`
- `"Wait 7 more days before harvest"`

## Reward

All rewards in range [0.0, 1.0] with partial credit for partially correct decisions:
- **irrigation_decision**: 1.0 for correct decision, 0.2-0.5 for partial
- **fertilizer_recommendation**: up to 0.6 for correct type + 0.4 for specifying quantity
- **harvest_timing**: 1.0 for optimal decision, 0.3-0.8 for suboptimal but reasonable

## Quick Start

```bash
# Install dependencies
pip install openenv-core uvicorn fastapi

# Run the server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Test reset
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "irrigation_decision"}'
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

## Environment Details

- **Max steps per episode**: 7
- **Concurrent sessions**: 4
- **Farm state evolution**: Soil moisture decreases, temperature fluctuates, forecast shifts each step
- **Context**: Tamil Nadu crop seasons, Indian market prices (Rs/kg)
