import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Any
import json

from env import DisasterResponseEnv
from models import Observation, Action, Dispatch, StepResult
from tasks import TASKS

class DisasterResponseGymEnv(gym.Env):
    """
    Gymnasium wrapper for the Disaster Response MDP.
    Pads observations and actions to fixed sizes for Neural Network compatibility.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, task_id: str = "citywide_crisis_management"):
        super().__init__()
        self.underlying_env = DisasterResponseEnv(task_id)
        self.task = TASKS[task_id]
        
        # 1. Define Constants for Space Sizes
        self.MAX_INCIDENTS = 10
        self.MAX_UNITS = len(self.task["resources_manifest"])
        self.UNIT_IDS = sorted(list(self.task["resources_manifest"].keys()))
        
        # Map unit types to indices for one-hot encoding
        all_types = sorted(list(set(self.task["resources_manifest"].values())))
        self.TYPE_TO_IDX = {t: i for i, t in enumerate(all_types)}
        self.NUM_TYPES = len(all_types)
        
        # 2. Observation Space (spaces.Dict as requested)
        self.observation_space = spaces.Dict({
            "unit_statuses": spaces.Box(low=0, high=10, shape=(self.MAX_UNITS,), dtype=np.int32),
            "incident_requirements": spaces.Box(low=0, high=1, shape=(self.MAX_INCIDENTS, self.NUM_TYPES), dtype=np.float32),
            "incident_metadata": spaces.Box(low=0, high=5, shape=(self.MAX_INCIDENTS, 2), dtype=np.float32), # [severity_score, progress]
            "time_steps": spaces.Box(low=0, high=10, shape=(1,), dtype=np.int32)
        })
        
        # 3. Action Space (MultiDiscrete)
        # Each unit can be assigned to incident 0 (None) or 1..MAX_INCIDENTS
        self.action_space = spaces.MultiDiscrete([self.MAX_INCIDENTS + 1] * self.MAX_UNITS)

    def _get_severity_score(self, severity: str) -> float:
        scores = {"critical": 3.0, "moderate": 2.0, "low": 1.0, "none": 0.0}
        return scores.get(severity.lower(), 0.0)

    def _flatten_obs(self, obs: Observation) -> Dict[str, np.ndarray]:
        # A. Unit Statuses (Steps until free)
        unit_statuses = np.zeros(self.MAX_UNITS, dtype=np.int32)
        for i, uid in enumerate(self.UNIT_IDS):
            unit_statuses[i] = obs.busy_units.get(uid, 0)
            
        # B. Incident Requirements (Padded)
        inc_reqs = np.zeros((self.MAX_INCIDENTS, self.NUM_TYPES), dtype=np.float32)
        inc_meta = np.zeros((self.MAX_INCIDENTS, 2), dtype=np.float32)
        
        for i, inc in enumerate(obs.active_incidents):
            if i >= self.MAX_INCIDENTS: break
            # Meta
            inc_meta[i, 0] = self._get_severity_score(inc.severity)
            inc_meta[i, 1] = 1.0 # Active flag
            
            # Requirements
            for req_type in inc.requires:
                if req_type in self.TYPE_TO_IDX:
                    inc_reqs[i, self.TYPE_TO_IDX[req_type]] = 1.0
                    
        return {
            "unit_statuses": unit_statuses,
            "incident_requirements": inc_reqs,
            "incident_metadata": inc_meta,
            "time_steps": np.array([obs.step], dtype=np.int32)
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset internal env
        # Note: DisasterResponseEnv.reset() is async, but we can call it synchronously 
        # because it doesn't actually perform I/O in its base version
        import asyncio
        step_result = asyncio.run(self.underlying_env.reset())
        
        obs = self._flatten_obs(step_result.observation)
        return obs, {}

    def step(self, action_vector):
        # Translate MultiDiscrete vector to List[Dispatch]
        dispatches = []
        active_incidents = self.underlying_env.active_incidents
        
        for unit_idx, target_inc_idx in enumerate(action_vector):
            # 0 means "Do nothing"
            if target_inc_idx == 0: continue
            
            # target_inc_idx is 1-indexed (1 to MAX_INCIDENTS)
            actual_inc_idx = target_inc_idx - 1
            if actual_inc_idx < len(active_incidents):
                unit_id = self.UNIT_IDS[unit_idx]
                inc_id = active_incidents[actual_inc_idx].id
                dispatches.append(Dispatch(unit=unit_id, incident_id=inc_id))
        
        # Create Action object
        action = Action(dispatches=dispatches, reasoning="RL Agent Action")
        
        # Step the environment
        import asyncio
        step_result = asyncio.run(self.underlying_env.step(action))
        
        obs = self._flatten_obs(step_result.observation)
        reward = step_result.reward
        terminated = step_result.terminated
        truncated = step_result.truncated
        
        return obs, reward, terminated, truncated, {"info": step_result.info}

    def render(self):
        pass
