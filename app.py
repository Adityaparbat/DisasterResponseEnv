# app.py
"""
FastAPI server for RL/MDP Disaster Response Evaluation.
Standardized RL API: observation, reward, terminated, truncated, info.
"""

import os
import json
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

from env import DisasterResponseEnv
from models import Action, Dispatch
from tasks import TASKS

app = FastAPI(
    title="Disaster Response RL Environment",
    description="Gymnasium-standard LLM evaluation environment for emergency dispatch coordination.",
    version="2.0.0"
)

# Global store for active environments (task_id -> env instance)
_envs: Dict[str, DisasterResponseEnv] = {}
LATEST_ACTIVE_TASK = "single_incident_response"

@app.get("/")
async def root():
    return FileResponse("index.html")

@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)

@app.get("/latest")
async def get_latest():
    """Returns the ID of the task currently being evaluated."""
    return {"task_id": LATEST_ACTIVE_TASK}

@app.get("/tasks")
async def get_tasks():
    """Returns the list of all available tasks."""
    return list(TASKS.keys())

# ------------------------------------------------------------------
# Request schemas
# ------------------------------------------------------------------
class ResetRequest(BaseModel):
    task_id: Optional[str] = "single_incident_response"

class StepRequest(BaseModel):
    task_id: Optional[str] = "single_incident_response"
    dispatches: List[Dict[str, str]] = [] # [{"unit": "id", "incident_id": "id"}]
    reasoning: str = ""
    priority_order: Optional[List[str]] = None
    reserved_units: Optional[List[str]] = None

# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------
@app.post("/reset")
async def reset(request: ResetRequest):
    """Resets the environment to its initial state."""
    task_id = request.task_id or "single_incident_response"
    if task_id not in TASKS:
        raise HTTPException(400, f"Unknown task_id: {task_id}.")
    
    global LATEST_ACTIVE_TASK
    LATEST_ACTIVE_TASK = task_id
    
    env = DisasterResponseEnv(task_id=task_id)
    _envs[task_id] = env
    
    result = await env.reset()
    return JSONResponse(status_code=200, content={
        "observation": result.observation.model_dump(),
        "reward": result.reward,
        "terminated": result.terminated,
        "truncated": result.truncated,
        "info": result.info
    })

@app.get("/state")
async def get_state(task_id: str):
    """Polls the current state for the UI/Dashboard."""
    env = _envs.get(task_id)
    if not env:
        return {"not_started": True}
    return {
        "observation": env._make_obs().model_dump(),
        "reward": sum(json.loads(h).get("reward", 0) for h in env._history) if env._history else 0.0,
        "terminated": env._done and len(env.active_incidents) == 0,
        "truncated": env._done and len(env.active_incidents) > 0,
        "history": [json.loads(h) for h in env._history]
    }

@app.post("/step")
async def step(request: StepRequest):
    """Processes an agent action and returns the next state and reward."""
    task_id = request.task_id or "single_incident_response"
    
    global LATEST_ACTIVE_TASK
    LATEST_ACTIVE_TASK = task_id
    
    env = _envs.get(task_id)
    if env is None:
        raise HTTPException(400, "Call /reset first.")
    
    # Transform raw dict dispatches into Dispatch objects
    dispatches = [Dispatch(**d) for d in request.dispatches]
    
    action = Action(
        dispatches=dispatches,
        reasoning=request.reasoning,
        priority_order=request.priority_order,
        reserved_units=request.reserved_units
    )
    
    try:
        result = await env.step(action)
    except RuntimeError as e:
        raise HTTPException(400, str(e))
        
    return JSONResponse(status_code=200, content={
        "observation": result.observation.model_dump(),
        "reward": result.reward,
        "terminated": result.terminated,
        "truncated": result.truncated,
        "info": result.info
    })

# Local dev server entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
