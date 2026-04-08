# server/app.py
"""
FastAPI server for RL/MDP Disaster Response Evaluation.
Standardized OpenEnv Layout (v0.2.0 compliant).
"""

import os
import sys
import json
import math
from typing import Optional, List, Dict, Any

# Ensure project root is in path for imports from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
    version="2.0.2"
)

# Global store for active environments (task_id -> env instance)
_envs: Dict[str, DisasterResponseEnv] = {}
LATEST_ACTIVE_TASK = "single_incident_response"

def sigmoid_normalize(x):
    """Unified Normalizer: Maps raw MDP return (-inf, inf) to grader-friendly (0.01, 0.99)"""
    return 0.01 + (0.98 * (1 / (1 + math.exp(-x))))

@app.get("/", response_class=HTMLResponse)
async def root():
    """
    Serves the dashboard UI. 
    Location-Aware: Checks absolute, package, and relative paths.
    """
    possible_paths = [
        "/app/index.html", # Hugging Face standard
        os.path.join(os.path.dirname(__file__), "..", "index.html"), # Package standard
        os.path.join(os.path.dirname(__file__), "index.html"), # Internal standard
        "index.html", # Local CWD
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read(), status_code=200)
                
    return HTMLResponse(content="<h1>Disaster Response Env</h1><p>Dashboard (index.html) not found. Status: Running.</p>", status_code=200)

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
async def reset(request: Request, task_id: Optional[str] = None):
    """
    Resets the environment to its initial state.
    Hyper-Resilient: Supports task_id via Query Param OR JSON Body.
    """
    body_data = {}
    try:
        if await request.body():
            body_data = await request.json()
    except Exception:
        body_data = {}
    
    if body_data is None: body_data = {}
    final_task_id = body_data.get("task_id") or task_id or "single_incident_response"
    
    if final_task_id not in TASKS:
        raise HTTPException(400, f"Unknown task_id: {final_task_id}.")
    
    global LATEST_ACTIVE_TASK
    LATEST_ACTIVE_TASK = final_task_id
    
    env = DisasterResponseEnv(task_id=final_task_id)
    _envs[final_task_id] = env
    
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
        
        # Calculate scores from the new rich history object list
        raw_rewards = [h["reward"] for h in env._history]
        total_raw = sum(raw_rewards) if raw_rewards else 0.0
        
        # Normalize for the dashboard's (0.01, 0.99) chart limits
        normalized_history = []
        for h in env._history:
            h_copy = h.copy()
            h_copy["reward"] = sigmoid_normalize(h["reward"])
            normalized_history.append(h_copy)

        return JSONResponse(content=jsonable_encoder({
            "observation": obs.model_dump(),
            "reward": sigmoid_normalize(total_raw),
            "terminated": env._done and len(env.active_incidents) == 0,
            "truncated": env._done and len(env.active_incidents) > 0,
            "history": normalized_history
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

# ------------------------------------------------------------------
# Entry point for the openenv-server CLI
# ------------------------------------------------------------------
def main():
    import uvicorn
    # Important: Run from the package perspective so openenv-server script works
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()
