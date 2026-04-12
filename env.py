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
        recalls = action.recalls or []
        
        identity_locks = self.task.get("identity_locked_units", {})
        
        # 1. State Transition: Process Recalls First
        for r in recalls:
            unit = r.unit
            inc_id = r.incident_id
            target_inc = next((i for i in self.active_incidents if i.id == inc_id), None)
            if target_inc and unit in target_inc.assigned_units:
                target_inc.assigned_units.remove(unit)
                
            # Remove from busy units and return to available
            self.busy_units = [bu for bu in self.busy_units if bu.get("unit_id") != unit]
            if unit not in self.available_units:
                self.available_units.append(unit)
                info["reason"] += f"Recalled {unit} from {inc_id}. "

        # Saturation tracker for over-dispatching checks
        saturation_tracker = {inc.id: {rtype: 0 for rtype in set(inc.requires)} for inc in self.active_incidents}
        # Prepopulate with existing assignments
        for inc in self.active_incidents:
            for u in inc.assigned_units:
                u_type = manifest.get(u, "unknown")
                if u_type in saturation_tracker[inc.id]:
                    saturation_tracker[inc.id][u_type] += 1

        # 2. State Transition: Process New Dispatches
        constraint_penalty = 0.0
        efficiency_penalty = 0.0
        step_violations = 0

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
                            efficiency_penalty += 1.0
                            info["violations"].append(f"Inefficiency: {unit_type} is redundant for {inc_id}.")
                        saturation_tracker[inc_id][unit_type] += 1
                        target_inc.assigned_units.append(unit)
                    else:
                        efficiency_penalty += 0.5
                        info["violations"].append(f"Wrong Type: {unit_type} not needed for {inc_id}.")
                        target_inc.assigned_units.append(unit)
                else:
                    efficiency_penalty += 0.5

                # Mark unit as busy indefinitely locked
                self.available_units.remove(unit)
                self.busy_units.append({"unit_id": unit, "free_at_step": -1})
            else:
                constraint_penalty += 1.0
                step_violations += 1
                info["violations"].append(f"Invalid: {unit} is unavailable.")

        # 3. Reward Calculation: Progress Tracking
        resolved_indices = []
        priority_bonus = 0.0
        coverage_raw = 0.0
        
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
                inc.turns_worked += 1
                if inc.turns_worked >= inc.time_to_resolve:
                    resolved_indices.append(i)
                    if inc.severity == "critical":
                        priority_bonus += 1.0
                    elif inc.severity == "moderate":
                        priority_bonus += 0.5
                    info["reason"] += f"Resolved {inc.id}. "
                else:
                    reward -= 0.1 # Still active time penalty while working on it
            else:
                reward -= 0.1 # Time penalty

        # Remove resolved and FREE THEIR UNITS
        for idx in sorted(resolved_indices, reverse=True):
            resolved_inc = self.active_incidents.pop(idx)
            for u in resolved_inc.assigned_units:
                self.busy_units = [bu for bu in self.busy_units if bu.get("unit_id") != u]
                if u not in self.available_units:
                    self.available_units.append(u)

        # 4. Final Reward Synthesis
        active_count = max(1, len(saturation_tracker))
        coverage_norm = coverage_raw / active_count
        reward += (coverage_norm * 0.4) + (priority_bonus * 0.4) - (constraint_penalty * 0.4) - (efficiency_penalty * 0.3)
        
        speed_factor = max(0.5, 1.1 - (self._step * 0.1))
        reward = reward * speed_factor
        reward = max(-1.0, min(1.0, reward))

        terminated = len(self.active_incidents) == 0
        truncated = (self._step >= self._max) or (step_violations >= 4)
        self._done = terminated or truncated

        if terminated:
            reward += 2.0
            info["reason"] += "Board cleared."
        elif step_violations >= 4:
            reward -= 5.0
            info["reason"] += "Mission aborted: Excessive violations."
        elif truncated:
            info["reason"] += "Max steps reached."

        # 5. State Management for UI/Logs
        history_entry = {
            "step": self._step,
            "reward": float(reward),
            "dispatches": [d.model_dump() for d in dispatches],
            "recalls": [r.model_dump() for r in recalls],
            "reasoning": action.reasoning,
            "violations": info["violations"]
        }
        self._history.append(history_entry)
        
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
        
        # UI expects JSON strings in previous_actions
        previous_actions_json = [_json.dumps(h) for h in self._history]

        return Observation(
            step=self._step,
            max_steps=self._max,
            active_incidents=self.active_incidents,
            available_units=self.available_units,
            busy_units=busy_status,
            resources_manifest=manifest,
            constraints=self.task.get("constraints", []),
            previous_actions=previous_actions_json
        )