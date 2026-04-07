import copy
import json as _json
from typing import Optional, List, Dict, Any
from models import Observation, Action, StepResult, Incident, Dispatch
from tasks import TASKS

class DisasterResponseEnv:
    """
    OpenEnv-compliant async environment for emergency dispatch coordination.
    Handles State Mutation, Time Penalties, and dynamic Rewards.
    Architecture: Markov Decision Process (MDP).
    """

    def __init__(self, task_id: Optional[str] = None):
        self.task_id = task_id or "single_incident_response"
        if self.task_id not in TASKS:
            raise ValueError(f"Task {self.task_id} not found.")
        self.task = TASKS[self.task_id]
        self._done = False
        self._history = []
        
    async def reset(self) -> StepResult:
        self._step = 0
        self._max = self.task.get("max_steps", 5)
        self._done = False
        self._history = []
        
        # Deep copy to allow state mutation without altering the base task config
        self.active_incidents: List[Incident] = [
            Incident(**inc) for inc in self.task["initial_incidents"]
        ]
        self.available_units: List[str] = copy.deepcopy(self.task["initial_resources"])
        
        # Track units currently deployed (busy) and when they return
        # Format: {"unit_id": "id", "free_at_step": int}
        self.busy_units: List[Dict[str, Any]] = []

        # Remove unavailable units (maintenance, etc) from pool permanently
        for unavail in self.task.get("unavailable_units", []):
            if unavail in self.available_units:
                self.available_units.remove(unavail)

        obs = self._make_obs()
        return StepResult(
            observation=obs,
            reward=0.0,
            terminated=False,
            truncated=False,
            info={"reason": "Environment reset."}
        )

    async def step(self, action: Action) -> StepResult:
        if self._done:
            raise RuntimeError("Episode finished — call reset() first.")

        self._step += 1
        reward = 0.0
        info = {"violations": [], "reason": ""}
        
        manifest = self.task["resources_manifest"]
        dispatches = action.dispatches or []
        
        # 1. State Transition: Release units that finished their work
        still_busy = []
        for bu in self.busy_units:
            if bu["free_at_step"] <= self._step:
                self.available_units.append(bu["unit_id"])
            else:
                still_busy.append(bu)
        self.busy_units = still_busy

        # 2. State Transition: Process New Dispatches
        turn_assignments: Dict[str, List[str]] = {} # inc_id -> list of unit TYPES
        identity_locks = self.task.get("identity_locked_units", {})
        
        coverage_raw = 0.0
        constraint_penalty = 0.0
        efficiency_penalty = 0.0
        step_violations = 0

        # Saturation tracker for over-dispatching checks
        saturation_tracker = {inc.id: {rtype: 0 for rtype in set(inc.requires)} for inc in self.active_incidents}

        for d in dispatches:
            unit = d.unit
            inc_id = d.incident_id

            if unit in self.available_units:
                unit_type = manifest.get(unit, "unknown")
                
                # Check for Identity locks
                if inc_id in identity_locks:
                    locked_unit = identity_locks[inc_id]
                    locked_type = manifest.get(locked_unit)
                    if unit_type == locked_type and unit != locked_unit:
                        constraint_penalty += 1.0
                        step_violations += 1
                        info["violations"].append(f"Security Breach: {unit} not authorized for {inc_id}.")
                        continue

                # Check for forbidden combinations
                forbidden = self.task.get("forbidden_dispatches", {}).get(inc_id, [])
                if unit_type in forbidden:
                    constraint_penalty += 1.0
                    step_violations += 1
                    info["violations"].append(f"Safety: {unit_type} forbidden from {inc_id} zone.")
                    continue

                # Check for Saturation (Efficiency)
                target_inc = next((i for i in self.active_incidents if i.id == inc_id), None)
                if target_inc:
                    if unit_type in saturation_tracker[inc_id]:
                        req_count = target_inc.requires.count(unit_type)
                        if saturation_tracker[inc_id][unit_type] >= req_count:
                            efficiency_penalty += 1.0 # Significant penalty for redundancy
                            info["violations"].append(f"Inefficiency: {unit_type} is redundant for {inc_id}.")
                        saturation_tracker[inc_id][unit_type] += 1
                    else:
                        efficiency_penalty += 0.5 # Wrong type
                        info["violations"].append(f"Wrong Type: {unit_type} not needed for {inc_id}.")
                else:
                    efficiency_penalty += 0.5 # Incident doesn't exist

                # Mark unit as busy
                self.available_units.remove(unit)
                self.busy_units.append({"unit_id": unit, "free_at_step": self._step + 2})
            else:
                constraint_penalty += 1.0
                step_violations += 1
                info["violations"].append(f"Invalid: {unit} is unavailable.")

        # 3. Reward Calculation: Progress Tracking
        resolved_indices = []
        priority_bonus = 0.0
        
        for i, inc in enumerate(self.active_incidents):
            needed = inc.requires
            inc_coverage_sum = 0
            for rtype in set(needed):
                req_count = needed.count(rtype)
                sat_count = saturation_tracker.get(inc.id, {}).get(rtype, 0)
                inc_coverage_sum += min(sat_count, req_count)
            
            local_coverage = inc_coverage_sum / len(needed) if needed else 1.0
            coverage_raw += local_coverage

            if local_coverage >= 1.0:
                resolved_indices.append(i)
                if inc.severity == "critical":
                    priority_bonus += 1.0
                elif inc.severity == "moderate":
                    priority_bonus += 0.5
                info["reason"] += f"Resolved {inc.id}. "

        # Remove resolved
        for idx in sorted(resolved_indices, reverse=True):
            self.active_incidents.pop(idx)

        # 4. Final Reward Synthesis
        active_count = max(1, len(saturation_tracker))
        coverage_norm = coverage_raw / active_count
        
        # INCREASED Weights to suppress reward hacking
        reward = (coverage_norm * 0.4) + (priority_bonus * 0.4) - (constraint_penalty * 0.4) - (efficiency_penalty * 0.3)
        
        # Speed Factor
        speed_factor = max(0.5, 1.1 - (self._step * 0.1))
        reward = reward * speed_factor

        # Clip BASE signal to standard RL range [-1, 1]
        reward = max(-1.0, min(1.0, reward))

        # 5. Global Modifiers (Outside the clip to ensure impact)
        terminated = len(self.active_incidents) == 0
        truncated = (self._step >= self._max) or (step_violations >= 4)
        self._done = terminated or truncated

        if terminated:
            reward += 2.0 # BOARD CLEARED
            info["reason"] += "Board cleared."
        elif step_violations >= 4:
            reward -= 5.0 # FATAL SPAM PENALTY
            info["reason"] += "Mission aborted: Excessive violations."
        elif truncated:
            info["reason"] += "Max steps reached."

        self._history.append(reward)
        return StepResult(
            observation=self._make_obs(),
            reward=round(float(reward), 4),
            terminated=terminated,
            truncated=truncated,
            info=info
        )

    def _make_obs(self) -> Observation:
        manifest = self.task["resources_manifest"]
        busy_status = {bu["unit_id"]: bu["free_at_step"] - self._step for bu in self.busy_units}
        
        return Observation(
            step=self._step,
            max_steps=self._max,
            active_incidents=self.active_incidents,
            available_units=self.available_units,
            busy_units=busy_status,
            resources_manifest=manifest,
            constraints=self.task.get("constraints", []),
            previous_actions=[round(r, 4) for r in self._history if isinstance(r, (int, float))]
        )