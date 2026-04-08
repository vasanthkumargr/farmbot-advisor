import random
from app.models import FarmState

def clamp(score: float) -> float:
    """Ensure score is strictly between 0 and 1, never exactly 0.0 or 1.0"""
    return round(max(0.05, min(0.95, score)), 3)

def get_initial_state(task_id: str) -> FarmState:
    return FarmState(
        task_id=task_id,
        step=0,
        soil_moisture=round(random.uniform(0.1, 0.9), 3),
        temperature=round(random.uniform(22.0, 42.0), 2),
        crop_stage=random.choice(["sowing", "vegetative", "flowering", "harvest"]),
        days_since_planting=random.randint(10, 90),
        weather_forecast=[round(random.uniform(0, 30), 2) for _ in range(7)],
        market_price=round(random.uniform(10.0, 80.0), 2),
        done=False,
        reward=0.5
    )

def evolve_state(state: FarmState) -> FarmState:
    state.soil_moisture = round(max(0.0, min(1.0, state.soil_moisture - random.uniform(0.02, 0.08))), 3)
    state.temperature = round(state.temperature + random.uniform(-1.5, 1.5), 2)
    state.days_since_planting += 1
    state.weather_forecast = state.weather_forecast[1:] + [round(random.uniform(0, 30), 2)]
    return state

def grade_irrigation(action: str, state: FarmState) -> float:
    should_irrigate = state.soil_moisture < 0.3 or state.temperature > 38.0
    action_lower = action.lower()
    if should_irrigate:
        if "irrigate" in action_lower:
            return clamp(0.92)
        elif "no irrigation" in action_lower or "don't irrigate" in action_lower:
            return clamp(0.08)
        else:
            return clamp(0.45)
    else:
        if "no irrigation" in action_lower or "don't irrigate" in action_lower or "wait" in action_lower:
            return clamp(0.92)
        elif "irrigate" in action_lower:
            return clamp(0.15)
        else:
            return clamp(0.5)

def grade_fertilizer(action: str, state: FarmState) -> float:
    score = 0.0
    action_lower = action.lower()
    stage_map = {
        "sowing": "phosphorus",
        "vegetative": "nitrogen",
        "flowering": "phosphorus",
        "harvest": "potassium"
    }
    correct = stage_map.get(state.crop_stage, "nitrogen")
    if correct in action_lower:
        score += 0.55
    elif any(f in action_lower for f in ["nitrogen", "phosphorus", "potassium"]):
        score += 0.2
    else:
        score += 0.05
    if any(q in action_lower for q in ["kg", "grams", "kg/acre", "per acre"]):
        score += 0.35
    else:
        score += 0.05
    return clamp(score)

def grade_harvest(action: str, state: FarmState) -> float:
    action_lower = action.lower()
    optimal_maturity = state.days_since_planting >= 85
    good_price = state.market_price > 40.0
    if optimal_maturity and good_price:
        if "harvest" in action_lower:
            return clamp(0.92)
        elif "wait" in action_lower:
            return clamp(0.3)
        return clamp(0.5)
    elif optimal_maturity and not good_price:
        if "wait" in action_lower:
            return clamp(0.78)
        elif "harvest" in action_lower:
            return clamp(0.45)
        return clamp(0.4)
    else:
        if "wait" in action_lower:
            return clamp(0.92)
        elif "harvest" in action_lower:
            return clamp(0.12)
        return clamp(0.4)

def compute_reward(task_id: str, action: str, state: FarmState) -> float:
    if task_id == "irrigation_decision":
        return grade_irrigation(action, state)
    elif task_id == "fertilizer_recommendation":
        return grade_fertilizer(action, state)
    elif task_id == "harvest_timing":
        return grade_harvest(action, state)
    return clamp(0.5)