from fastapi import FastAPI
from app.models import ResetRequest, StepRequest, FarmState
from app.environment import get_initial_state, compute_reward
import copy

app = FastAPI(title="Farmbot Advisor OpenEnv")

sessions: dict = {}

@app.get("/")
def root():
    return {"status": "ok", "env": "farmbot-advisor"}

@app.post("/reset")
def reset(req: ResetRequest):
    state = get_initial_state(req.task_id)
    sessions[req.task_id] = state
    return state

@app.post("/step")
def step(req: StepRequest):
    task_id = req.action.action
    # find active session — use first available
    if not sessions:
        return {"error": "Call /reset first"}
    task_id = list(sessions.keys())[0]
    state = sessions[task_id]

    reward = compute_reward(task_id, req.action.action, state)
    state.reward = reward
    state.step += 1
    state.done = state.step >= 3

    sessions[task_id] = state
    return state

@app.get("/state")
def get_state():
    if not sessions:
        return {"error": "No active session"}
    task_id = list(sessions.keys())[0]
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