from pydantic import BaseModel
from typing import Optional, Dict, Any

class FarmState(BaseModel):
    task_id: str
    step: int
    soil_moisture: float      # 0.0 to 1.0
    temperature: float        # Celsius
    crop_stage: str           # sowing / vegetative / flowering / harvest
    days_since_planting: int
    weather_forecast: list    # next 7 days rainfall mm
    market_price: float       # Rs per kg
    done: bool
    reward: float

class FarmAction(BaseModel):
    action: str               # the agent's decision as text

class ResetRequest(BaseModel):
    task_id: str

class StepRequest(BaseModel):
    action: FarmAction