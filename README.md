---
title: Disaster Response Env
emoji: 🚨
colorFrom: red
colorTo: blue
sdk: docker
python_version: "3.11"
app_file: app.py
pinned: false
---

# Disaster Response RL Environment (MDP-v2)

An OpenEnv-compliant, deterministic **Markov Decision Process (MDP)** where agents act as emergency dispatch controllers. This platform is designed to benchmark the coordination capability of Large Language Models (LLMs) against traditional Reinforcement Learning (RL) baselines.

## 🏗️ Architecture

The environment is built as a high-fidelity state machine that tracks unit availability, temporal "busy" states, and complex mission requirements across escalating disaster scenarios.

### Density-Shaped Reward System

Our reward architecture uses a hybrid **Sparse + Dense** signal to ensure stable learning while prioritizing mission success:

| Component | Value | Description |
| :--- | :--- | :--- |
| **Sparse Completion** | **+5.0** | Massive bonus for clearing the entire mission board. |
| **Dense Resolution** | **+2.0** | Reward for successfully resolving an individual incident. |
| **Time Penalty** | **-0.1** | Small penalty per turn for every incident remaining active. |
| **Constraint Penalty** | **-0.5** | Penalty for dispatching unavailable, busy, or incorrect units. |
| **Fatal Failure** | **-5.0** | Instant mission termination if >3 critical violations occur in one turn. |

## 🎯 Tasks

### 1. `single_incident_response` (Easy)
Basic coordination. Resolve a building fire with a fire truck and ambulance.

### 2. `multi_incident_triage` (Medium)
Three simultaneous incidents with coded unit names. Requires reading the `resources_manifest`.

### 3. `dynamic_escalation` (Hard)
Complex constraints (faulty equipment, specific unit blocks) requiring forward planning.

### 4. `citywide_crisis_management` (Master)
High-dimensional coordination involving 28 units and 4 incidents simultaneously. Requires:
- **Identity-Locked Units**: Incident `INC-003` (Server Farm) *must* be handled by `unit_delta_4`.
- **Saturation Management**: Punishes over-dispatching redundant units to the same zone.

## 🧪 Deep RL Validation (PPO)

To mathematically prove the validity of our MDP state transitions and reward shaping, we implemented a `gym_wrapper.py` and trained a **PPO (Proximal Policy Optimization)** neural network using Stable-Baselines3.

- **Training**: 50,000 timesteps.
- **Goal**: Establish a "Neural Baseline" that proves the environment is learnable and the penalties are sufficiently strict to prevent "Reward Hacking."
- **Finding**: The PPO agent successfully learned to avoid "spamming" (Rule of Law), setting a floor for LLM evaluation.

## 📊 The Scientific Win: Final Benchmarks

| Agent | Environment | Status | Final Reward |
| :--- | :--- | :--- | :--- |
| **Qwen 7B (Spammer)** | Static Grader (Old) | FAILED (Spamming) | -6.00 |
| **PPO Agent (Deep RL)** | **MDP (New)** | FAILED (Conservative) | -0.54 |
| **Qwen 72B (Reasoning)**| **MDP (New)** | **PASSED (Scientific Win)**| **+3.64** |

## 🚀 Setup & Execution

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the Environment Server
python app.py --port 7860

# 3. Run Inference (Submission-Compliant Format)
export HF_TOKEN=your_token
python inference.py --model Qwen/Qwen2.5-72B-Instruct
```

## 📂 File Structure

```text
├── env.py            <-- Core MDP Logic & Reward Shaping
├── app.py            <-- FastAPI Server (OpenEnv Standard)
├── models.py         <-- Pydantic Data Schemas
├── tasks.py          <-- Scenario Definitions & Graders
├── gym_wrapper.py    <-- Gymnasium Interface for RL training
├── train.py          <-- PPO Training Pipeline (SB3)
├── inference.py      <-- Submission-Ready LLM Evaluator
└── walkthrough.md    <-- Detailed Architectural Analysis
```
