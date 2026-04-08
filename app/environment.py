import random
from app.models import FarmState

def get_initial_state(task_id: str) -> FarmState:
    return FarmState(
        task_id=task_id,
        step=0,
        soil_moisture=random.uniform(0.1, 0.9),
        temperature=random.uniform(22.0, 42.0),
        crop_stage="vegetative",
        days_since_planting=random.randint(10, 90),
        weather_forecast=[random.uniform(0, 30) for _ in range(7)],
        market_price=random.uniform(10.0, 80.0),
        done=False,
        reward=0.0
    )

def grade_irrigation(action: str, state: FarmState) -> float:
    should_irrigate = state.soil_moisture < 0.3 or state.temperature > 38.0
    action_lower = action.lower()
    if should_irrigate and "irrigate" in action_lower:
        return 1.0
    elif not should_irrigate and "no irrigation" in action_lower:
        return 1.0
    elif should_irrigate and "no irrigation" in action_lower:
        return 0.0
    elif not should_irrigate and "irrigate" in action_lower:
        return 0.2
    return 0.3

def grade_fertilizer(action: str, state: FarmState) -> float:
    score = 0.0
    action_lower = action.lower()
    if state.crop_stage == "vegetative" and "nitrogen" in action_lower:
        score += 0.5
    elif state.crop_stage == "flowering" and "phosphorus" in action_lower:
        score += 0.5
    elif state.crop_stage == "harvest" and "potassium" in action_lower:
        score += 0.5
    if "kg" in action_lower or "grams" in action_lower:
        score += 0.5
    return min(score, 1.0)

def grade_harvest(action: str, state: FarmState) -> float:
    score = 0.0
    action_lower = action.lower()
    optimal = state.days_since_planting >= 85 and state.market_price > 40.0
    if optimal and "harvest" in action_lower:
        score = 1.0
    elif not optimal and "wait" in action_lower:
        score = 0.8
    elif "harvest" in action_lower and state.market_price > 30.0:
        score = 0.5
    return score

def compute_reward(task_id: str, action: str, state: FarmState) -> float:
    if task_id == "irrigation_decision":
        return grade_irrigation(action, state)
    elif task_id == "fertilizer_recommendation":
        return grade_fertilizer(action, state)
    elif task_id == "harvest_timing":
        return grade_harvest(action, state)
    return 0.0