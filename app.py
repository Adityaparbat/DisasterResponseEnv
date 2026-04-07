# app.py
"""
FastAPI server for RL/MDP Disaster Response Evaluation.
Hardened for UI Dashboard: Handles JSON serialization and State Tracking.
Bulletproof /reset: Handles empty/null POST bodies for automated checkers.
"""

import os
import json
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Response, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

from env import DisasterResponseEnv
from models import Action, Dispatch
from tasks import TASKS

app = FastAPI(
    title="Disaster Response RL Environment",
    description="Gymnasium-standard LLM evaluation environment for emergency dispatch coordination.",
    version="2.0.1"
)

# Global store for active environments (task_id -> env instance)
_envs: Dict[str, DisasterResponseEnv] = {}
LATEST_ACTIVE_TASK = "single_incident_response"

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serves the dashboard UI or a basic health check."""
    index_path = "index.html"
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    return HTMLResponse(content="<h1>Disaster Response Env</h1><p>Dashboard (index.html) not found.</p>", status_code=200)

@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)

@app.get("/latest")
async def get_latest():
    """Returns the ID of the task currently being evaluated."""
    return JSONResponse(content={"task_id": LATEST_ACTIVE_TASK})

@app.get("/tasks")
async def get_tasks():
    """Returns the list of all available tasks."""
    return JSONResponse(content=jsonable_encoder(list(TASKS.keys())))

# ------------------------------------------------------------------
# Request schemas
# ------------------------------------------------------------------
class StepRequest(BaseModel):
    task_id: Optional[str] = "single_incident_response"
    dispatches: List[Dict[str, str]] = [] 
    reasoning: str = ""

# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------
@app.post("/reset")
async def reset(request: Request):
    """
    Resets the environment to its initial state.
    Hardened to accept empty or malformed POST bodies.
    """
    try:
        # Checkbox: Bulletproof JSON parsing
        data = await request.json()
    except Exception:
        data = {}
    
    if data is None: data = {}

    task_id = data.get("task_id", "single_incident_response")
    
    if task_id not in TASKS:
        raise HTTPException(400, f"Unknown task_id: {task_id}.")
    
    global LATEST_ACTIVE_TASK
    LATEST_ACTIVE_TASK = task_id
    
    env = DisasterResponseEnv(task_id=task_id)
    _envs[task_id] = env
    
    result = await env.reset()
    return JSONResponse(status_code=200, content=jsonable_encoder({
        "observation": result.observation.model_dump(),
        "reward": result.reward,
        "terminated": result.terminated,
        "truncated": result.truncated,
        "info": result.info
    }))

@app.get("/state")
async def get_state(task_id: str):
    """Polls the current state for the UI/Dashboard."""
    try:
        env = _envs.get(task_id)
        if not env:
            return JSONResponse(content={"not_started": True})
        
        obs = env._make_obs()
        return JSONResponse(content=jsonable_encoder({
            "observation": obs.model_dump(),
            "reward": sum(json.loads(h) if isinstance(h, str) else h for h in env._history) if env._history else 0.0,
            "terminated": env._done and len(env.active_incidents) == 0,
            "truncated": env._done and len(env.active_incidents) > 0,
            "history": [json.loads(h) if isinstance(h, str) else h for h in env._history]
        }))
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/step")
async def step(request: StepRequest):
    """Processes an agent action and returns the next state and reward."""
    try:
        task_id = request.task_id or "single_incident_response"
        
        global LATEST_ACTIVE_TASK
        LATEST_ACTIVE_TASK = task_id
        
        env = _envs.get(task_id)
        if env is None:
            raise HTTPException(400, "Call /reset first.")
        
        # Transform raw dict dispatches into Dispatch objects
        dispatches = [Dispatch(**d) for d in request.dispatches]
        action = Action(dispatches=dispatches, reasoning=request.reasoning)
        
        result = await env.step(action)
        return JSONResponse(status_code=200, content=jsonable_encoder({
            "observation": result.observation.model_dump(),
            "reward": result.reward,
            "terminated": result.terminated,
            "truncated": result.truncated,
            "info": result.info
        }))
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

# Local dev server entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
