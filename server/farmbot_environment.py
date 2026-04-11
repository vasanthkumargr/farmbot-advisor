"""
Farmbot Advisor Environment Implementation.

Simulates agricultural decision-making for Indian farmers.
An AI agent learns to advise on irrigation, fertilization,
and harvest timing based on real farm conditions.
"""

import random
from uuid import uuid4
from typing import Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from models import FarmbotAction, FarmbotObservation, FarmbotState
except ImportError:
    from ..models import FarmbotAction, FarmbotObservation, FarmbotState


TASKS = ["irrigation_decision", "fertilizer_recommendation", "harvest_timing"]
MAX_STEPS = 3


class FarmbotEnvironment(Environment):
    """
    Farmbot Advisor RL environment for agricultural decision-making.

    Each episode presents a farm scenario with soil, weather, crop stage,
    and market data. The agent must make farming decisions and receives
    rewards based on agronomic correctness.

    Tasks:
    - irrigation_decision (easy): decide whether to irrigate
    - fertilizer_recommendation (medium): choose fertilizer type and quantity
    - harvest_timing (hard): decide optimal harvest window

    Example:
        >>> env = FarmbotEnvironment()
        >>> obs = env.reset(task_id="irrigation_decision")
        >>> print(obs.soil_moisture)
        >>> obs = env.step(FarmbotAction(recommendation="Irrigate the field now"))
        >>> print(obs.reward)
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the Farmbot environment."""
        self._state = FarmbotState(episode_id=str(uuid4()), step_count=0)
        self._task_id = "irrigation_decision"
        self._step = 0
        self._done = False
        self._farm_data = self._random_farm_data()

    def _random_farm_data(self) -> dict:
        """Generate randomized farm conditions for a new episode."""
        return {
            "soil_moisture": round(random.uniform(0.1, 0.9), 3),
            "temperature": round(random.uniform(22.0, 42.0), 2),
            "crop_stage": random.choice(["sowing", "vegetative", "flowering", "harvest"]),
            "days_since_planting": random.randint(10, 90),
            "weather_forecast": [round(random.uniform(0, 30), 2) for _ in range(7)],
            "market_price": round(random.uniform(10.0, 80.0), 2),
        }

    def _evolve_farm(self):
        """Simulate farm conditions changing over time after each step."""
        d = self._farm_data
        d["soil_moisture"] = round(
            max(0.0, min(1.0, d["soil_moisture"] - random.uniform(0.02, 0.08))), 3
        )
        d["temperature"] = round(d["temperature"] + random.uniform(-1.5, 1.5), 2)
        d["days_since_planting"] += 1
        d["weather_forecast"] = d["weather_forecast"][1:] + [round(random.uniform(0, 30), 2)]

    def _compute_reward(self, recommendation: str) -> tuple:
        """
        Compute reward and feedback message for a given recommendation.
        Returns (reward, message) where reward is in [0.0, 1.0].
        """
        r = recommendation.lower()
        d = self._farm_data

        if self._task_id == "irrigation_decision":
            need_water = d["soil_moisture"] < 0.3 or d["temperature"] > 38.0
            if need_water:
                if "irrigate" in r and "no irrigation" not in r:
                    return 1.0, "Correct - field needs water given low moisture/high temperature."
                elif "no irrigation" in r or "don't irrigate" in r:
                    return 0.0, "Incorrect - field is dry and needs irrigation."
                else:
                    return 0.4, "Partial - unclear recommendation, irrigation is needed."
            else:
                if "no irrigation" in r or "don't irrigate" in r or "wait" in r:
                    return 1.0, "Correct - soil is sufficiently moist, no irrigation needed."
                elif "irrigate" in r:
                    return 0.2, "Incorrect - soil is moist enough, over-irrigation wastes water."
                else:
                    return 0.5, "Partial - unclear recommendation, no irrigation needed."

        elif self._task_id == "fertilizer_recommendation":
            stage_map = {
                "sowing": "phosphorus",
                "vegetative": "nitrogen",
                "flowering": "phosphorus",
                "harvest": "potassium",
            }
            correct_fert = stage_map.get(d["crop_stage"], "nitrogen")
            score = 0.0
            if correct_fert in r:
                score += 0.6
                msg_fert = f"Correct fertilizer ({correct_fert}) for {d['crop_stage']} stage."
            elif any(f in r for f in ["nitrogen", "phosphorus", "potassium"]):
                score += 0.2
                msg_fert = f"Wrong fertilizer type. {d['crop_stage']} stage needs {correct_fert}."
            else:
                msg_fert = f"No fertilizer type mentioned. {d['crop_stage']} stage needs {correct_fert}."
            if any(q in r for q in ["kg", "grams", "per acre", "kg/acre"]):
                score += 0.4
                msg_qty = " Good - specific quantity provided."
            else:
                msg_qty = " Partial - no quantity specified."
            return round(min(score, 1.0), 2), msg_fert + msg_qty

        elif self._task_id == "harvest_timing":
            mature = d["days_since_planting"] >= 85
            good_price = d["market_price"] > 40.0
            if mature and good_price:
                if "harvest" in r and "wait" not in r:
                    return 1.0, "Correct - crop is mature and price is good. Harvest now."
                elif "wait" in r:
                    return 0.3, "Suboptimal - crop is ready and price is good, should harvest."
                return 0.5, "Partial - crop ready and price good, recommend harvesting."
            elif mature and not good_price:
                if "wait" in r:
                    return 0.8, "Good - crop is mature but price is low. Waiting for better price."
                elif "harvest" in r:
                    return 0.5, "Acceptable - crop is mature but market price is currently low."
                return 0.4, "Partial - crop mature but price is poor, consider waiting."
            else:
                if "wait" in r:
                    return 1.0, "Correct - crop is not yet mature. Waiting is the right choice."
                elif "harvest" in r:
                    return 0.1, "Incorrect - crop is not mature yet. Harvesting early reduces yield."
                return 0.4, "Partial - crop is not mature, should wait."

        return 0.0, "Unknown task."

    def reset(self, task_id: Optional[str] = None, **kwargs) -> FarmbotObservation:
        """
        Reset the environment for a new episode.

        Args:
            task_id: Which task to run. One of: irrigation_decision,
                     fertilizer_recommendation, harvest_timing.
                     Defaults to irrigation_decision.

        Returns:
            FarmbotObservation with initial farm conditions.
        """
        self._task_id = task_id if task_id in TASKS else "irrigation_decision"
        self._step = 0
        self._done = False
        self._farm_data = self._random_farm_data()
        self._state = FarmbotState(
            episode_id=str(uuid4()),
            step_count=0,
            task_id=self._task_id,
            step=0,
            last_reward=0.0,
            **self._farm_data,
        )
        return FarmbotObservation(
            task_id=self._task_id,
            step=0,
            done=False,
            reward=0.0,
            message=f"Episode started. Task: {self._task_id}",
            **self._farm_data,
        )

    def step(self, action: FarmbotAction) -> FarmbotObservation:
        """
        Execute one step: evaluate the agent's recommendation and evolve the farm.

        Args:
            action: FarmbotAction with the agent's natural language recommendation.

        Returns:
            FarmbotObservation with updated farm state and reward.
        """
        reward, message = self._compute_reward(action.recommendation)
        self._evolve_farm()
        self._step += 1
        self._done = self._step >= MAX_STEPS
        self._state.step_count = self._step
        self._state.last_reward = reward

        return FarmbotObservation(
            task_id=self._task_id,
            step=self._step,
            done=self._done,
            reward=reward,
            message=message,
            **self._farm_data,
        )

    @property
    def state(self) -> FarmbotState:
        """Get the current environment state."""
        return self._state
