---
title: Disaster Response Env
emoji: 🚨
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
---

# 🚨 Disaster Response RL Environment (MDP-v2)

**An embodied, multi-turn dispatch simulator that benchmarks LLM resource management under strict real-world time pressure.**

The agent plays the role of a City Emergency Coordinator. It must read live incident feeds, interpret complex resource constraints, and dispatch specialized units across simultaneous crises — all while a ticking clock bleeds its score for every second of hesitation. This is not a static text benchmark. This is a living MDP.

[![Space Status](https://img.shields.io/badge/HF%20Space-Running-brightgreen)](https://huggingface.co/spaces/AdityParbat/disaster-response-env)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-v0.2.0-blue)](https://github.com/huggingface/openenv)
[![Framework](https://img.shields.io/badge/Framework-FastAPI%20%2B%20MDP-orange)]()
[![Model](https://img.shields.io/badge/Evaluated%20On-Qwen%2F2.5--72B-purple)]()

---

## 🔥 What Makes This Different

Most LLM evaluation benchmarks test *knowledge retrieval*. This environment tests **active, ongoing decision-making under compounding pressure**. Three mechanics make this uniquely difficult:

### ⏱️ The Ticking Clock (Multi-Turn Dynamics)
Incidents have a `time_to_resolve` that ranges from 1 to 4 turns. The LLM cannot dispatch a unit and walk away. It must **sustain unit presence** across multiple turns while the environment applies a flat **-0.1 reward penalty per active incident per turn**. A hesitant or distracted agent bleeds out.

### 🚫 Anti-Spam Penalties (Inefficiency Enforcement)
If the agent panics and floods an incident with more units than it requires, it receives **severe inefficiency penalties**. This directly combats the "just send everything" reward-hacking strategy that defeats most static graders. The optimal policy requires precision, not volume.

### 🔄 Dynamic Recalls (Active Mission Abort)
The agent can issue `recalls` — pulling units away from ongoing incidents and immediately returning them to the available pool. This enables real tactical re-routing: abort a low-priority mission to redeploy critical assets to an emerging crisis, all within the same turn.

---

## 🖥️ Command Center UI

The environment ships with a fully custom **Emergency Command Center** dashboard (`index.html`) served directly from the FastAPI backend.

- **Live Reward Trajectory**: A real-time chart tracks normalized mission score across every turn.
- **MDP State Visualization**: Active incidents, their severity, resource requirements, and current turn progress are rendered dynamically.
- **Action Log**: Every dispatch, recall, and constraint violation is logged in real time, allowing human observers to trace agent reasoning.

The UI is accessible at the root of the deployed Hugging Face Space at `/`.

---

## ⚡ Action Space

At each step, the agent submits a structured JSON action. The environment accepts both `dispatches` and `recalls` simultaneously, enabling complex re-routing within a single turn.

```json
{
  "dispatches": [
    {"unit": "fire_truck_1", "incident_id": "INC-001"},
    {"unit": "ambulance_2",  "incident_id": "INC-003"}
  ],
  "recalls": [
    {"unit": "police_unit_1", "incident_id": "INC-002"}
  ],
  "reasoning": "Re-routing police_unit_1 from the low-priority INC-002 to cover the new critical sector. Dispatching fire and medical to INC-001 and INC-003."
}
```

### Reward Structure

| Component | Value | Trigger |
| :--- | :--- | :--- |
| **Board Clearance Bonus** | **+2.0** | All incidents resolved |
| **Incident Resolution** | **+0.4–1.0** | Per incident cleared (severity-weighted) |
| **Priority Bonus** | **+1.0 / +0.5** | Critical / Moderate incidents resolved |
| **Time Penalty** | **-0.1 / turn** | Per active incident per step |
| **Constraint Violation** | **-0.4** | Unavailable, locked, or forbidden unit dispatched |
| **Inefficiency Penalty** | **-0.3** | Redundant unit sent to a saturated incident |
| **Fatal Abort** | **-5.0** | ≥4 critical violations in a single turn |

All raw rewards are normalized through a sigmoid to the **(0.01, 0.99)** range for grader compliance.

---

## 🎯 Tasks & Constraints

### 1. `single_incident_response` — Easy
Basic Unit Mapping. A building fire and a crowd control situation. Resolve with correct unit types.
`time_to_resolve`: **1 turn**

### 2. `multi_incident_triage` — Medium
Three simultaneous incidents with anonymized unit codes (`unit_alpha`, `unit_bravo`). The agent must read the `resources_manifest` to decode types before dispatching.
`time_to_resolve`: **1–2 turns**

### 3. `dynamic_escalation` — Hard
Complex operational constraints: faulty equipment bans specific units from hazmat zones, mental health units are required alongside police for de-escalation incidents. Requires forward planning across resource types.
`time_to_resolve`: **2–3 turns**

### 4. `citywide_crisis_management` — Master
High-dimensional four-incident scenario with 28 total units. Features:
- **Identity-Locked Units**: `INC-003` (Critical Server Farm Fire) **must** be handled by `unit_delta_4`. Sending any other fire truck triggers a Security Breach violation.
- **Saturation Management**: Heavy penalty for over-dispatching redundant types.
- **Cascading Priority**: Critical infrastructure and mass casualty incidents must be triaged simultaneously.

`time_to_resolve`: **3–4 turns**

---

## 🧪 Deep RL Validation (PPO Baseline)

To mathematically validate our MDP reward shaping and prove the environment is learnable, we trained a **PPO (Proximal Policy Optimization)** agent using Stable-Baselines3 for 50,000 timesteps.

**Key Finding:** The PPO agent learned to avoid constraint violations (Rule of Law) but struggled with temporal management, setting a measurable floor for LLM evaluation.

| Agent | Strategy | Final Score |
| :--- | :--- | :--- |
| Qwen 7B | Spam dispatcher (old env) | **FAILED** — -6.00 |
| PPO Agent (Deep RL) | Conservative matcher | -0.54 |
| **Qwen 72B (Reasoning)** | **Precision dispatch + recall** | **+3.64 ✅** |

---

## 🚀 Setup & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Environment Server
```bash
python server/app.py
# Server starts on http://localhost:7860
```

### 3. Configure & Run Inference
```bash
# Required: Your Hugging Face token for the inference LLM
export HF_TOKEN=your_hf_token_here

# Optional: Override the default model
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"

# Optional: Point at your local server instead of the HF Space
export API_BASE_URL="https://router.huggingface.co/v1"

# Run all tasks
python inference.py

# Run a single task
python inference.py --task citywide_crisis_management
```

---

## 📂 File Structure

```text
.
├── server/
│   └── app.py              ← FastAPI server (OpenEnv v0.2.0 compliant)
├── Dockerfile              ← Container setup for HF Spaces deployment
├── README.md               ← This file (HF Space config + documentation)
├── env.py                  ← Core MDP logic, reward shaping, and state transitions
├── evaluate_rl.py          ← Evaluation script for trained RL baselines
├── gym_wrapper.py          ← Gymnasium interface for RL training (PPO/SAC)
├── index.html              ← Emergency Command Center real-time UI dashboard
├── inference.py            ← Submission-ready LLM evaluator (OpenEnv compliant)
├── models.py               ← Pydantic schemas: Incident, Action, Dispatch, Observation
├── openenv.yaml            ← OpenEnv hackathon metadata and entry point config
├── pyproject.toml          ← Project build system configuration
├── requirements.txt        ← Python package dependencies
├── tasks.py                ← Scenario definitions: incidents, units, constraints
├── train.py                ← PPO training pipeline (Stable-Baselines3)
└── uv.lock                 ← Dependency lock file for strict reproducibility
```

---

## 🔗 Links

- **Live Space**: [huggingface.co/spaces/AdityParbat/disaster-response-env](https://huggingface.co/spaces/AdityParbat/disaster-response-env)
- **OpenEnv Spec**: [huggingface.co/docs/openenv](https://huggingface.co/docs/hub/spaces-config-reference)
