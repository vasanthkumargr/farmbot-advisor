"""
Data models for the Farmbot Advisor Environment.

The farmbot environment simulates real-world agricultural decision-making
for Indian farmers. An AI agent advises on irrigation, fertilization,
and harvest timing based on live farm conditions.
"""

from typing import List, Optional
from pydantic import Field

try:
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    from openenv.core.env_server.types import Action, Observation, State


class FarmbotAction(Action):
    """Action for the Farmbot Advisor environment - the agent's farming recommendation."""

    recommendation: str = Field(
        ...,
        description="The agent's farming recommendation as natural language. "
                    "Examples: 'Irrigate the field now', "
                    "'Apply nitrogen fertilizer at 50kg/acre', "
                    "'Wait 7 more days before harvest'"
    )
    task_id: Optional[str] = Field(
        default="irrigation_decision",
        description="Task to perform: irrigation_decision, fertilizer_recommendation, or harvest_timing"
    )


class FarmbotObservation(Observation):
    """Observation from the Farmbot Advisor environment."""

    task_id: str = Field(
        default="irrigation_decision",
        description="Current task: irrigation_decision, fertilizer_recommendation, harvest_timing"
    )
    step: int = Field(
        default=0,
        description="Current step number in the episode"
    )
    soil_moisture: float = Field(
        default=0.5,
        description="Soil moisture level from 0.0 (completely dry) to 1.0 (waterlogged)"
    )
    temperature: float = Field(
        default=30.0,
        description="Ambient temperature in Celsius"
    )
    crop_stage: str = Field(
        default="vegetative",
        description="Current crop growth stage: sowing, vegetative, flowering, or harvest"
    )
    days_since_planting: int = Field(
        default=50,
        description="Number of days since the crop was planted"
    )
    weather_forecast: List[float] = Field(
        default_factory=lambda: [10.0] * 7,
        description="7-day rainfall forecast in millimetres per day"
    )
    market_price: float = Field(
        default=40.0,
        description="Current market price in Indian Rupees per kg"
    )
    reward: float = Field(
        default=0.0,
        description="Reward for the last action, in range [0.0, 1.0]"
    )
    message: str = Field(
        default="",
        description="Human-readable feedback on the last action"
    )


class FarmbotState(State):
    """Server-side state for the Farmbot environment."""

    task_id: str = Field(default="irrigation_decision")
    step: int = Field(default=0)
    soil_moisture: float = Field(default=0.5)
    temperature: float = Field(default=30.0)
    crop_stage: str = Field(default="vegetative")
    days_since_planting: int = Field(default=50)
    weather_forecast: List[float] = Field(default_factory=lambda: [10.0] * 7)
    market_price: float = Field(default=40.0)
    last_reward: float = Field(default=0.0)
