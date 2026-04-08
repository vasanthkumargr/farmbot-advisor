from fastapi import FastAPI
from app.models import ResetRequest, StepRequest
from app.environment import get_initial_state, compute_reward, evolve_state
from typing import Optional

app = FastAPI(title="Farmbot Advisor OpenEnv")

sessions: dict = {}

@app.get("/")
def root():
    return {"status": "ok", "env": "farmbot-advisor"}

@app.post("/reset")
def reset(req: Optional[ResetRequest] = None):
    task_id = req.task_id if req else "irrigation_decision"
    state = get_initial_state(task_id)
    sessions[task_id] = state
    return state

@app.post("/step")
def step(req: StepRequest):
    task_id = req.task_id
    if task_id not in sessions:
        return {"error": f"No active session for task_id '{task_id}'. Call /reset first."}
    
    state = sessions[task_id]
    reward = compute_reward(task_id, req.action.action, state)
    state = evolve_state(state)
    state.reward = reward
    state.step += 1
    state.done = state.step >= 7

    sessions[task_id] = state
    return state

@app.get("/state")
def get_state(task_id: str = "irrigation_decision"):
    if task_id not in sessions:
        return {"error": f"No active session for task_id '{task_id}'"}
    return sessions[task_id]

@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {"id": "irrigation_decision", "difficulty": "easy"},
            {"id": "fertilizer_recommendation", "difficulty": "medium"},
            {"id": "harvest_timing", "difficulty": "hard"}
        ]
    }