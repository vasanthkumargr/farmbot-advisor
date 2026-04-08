from pydantic import BaseModel
from typing import Optional
import random

class FarmState(BaseModel):
    task_id: str
    step: int
    soil_moisture: float
    temperature: float
    crop_stage: str
    days_since_planting: int
    weather_forecast: list
    market_price: float
    done: bool
    reward: float

class FarmAction(BaseModel):
    action: str

class ResetRequest(BaseModel):
    task_id: str = "irrigation_decision"

class StepRequest(BaseModel):
    task_id: str
    action: FarmAction