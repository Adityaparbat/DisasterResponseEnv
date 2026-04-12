from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class Incident(BaseModel):
    id: str
    type: str
    severity: str
    requires: List[str]
    location: str
    notes: Optional[str] = None
    time_to_resolve: int = 1
    turns_worked: int = 0
    assigned_units: List[str] = []

class Observation(BaseModel):
    step: int
    max_steps: int
    active_incidents: List[Incident]
    available_units: List[str]
    busy_units: Dict[str, int]
    resources_manifest: Dict[str, str]
    constraints: List[str]
    previous_actions: List[Any] = []

class Dispatch(BaseModel):
    unit: str
    incident_id: str

class Action(BaseModel):
    dispatches: List[Dispatch] = []
    recalls: List[Dispatch] = []
    reasoning: str = ""
    priority_order: Optional[List[str]] = None
    reserved_units: Optional[List[str]] = None

class StepResult(BaseModel):
    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any] = {}