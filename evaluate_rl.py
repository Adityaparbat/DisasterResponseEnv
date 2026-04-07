import gymnasium as gym
from stable_baselines3 import PPO
from gym_wrapper import DisasterResponseGymEnv
import json
import numpy as np

def evaluate(task_id: str = "citywide_crisis_management"):
    print(f"\n=== EVALUATING RL AGENT ON {task_id} ===")
    
    # 1. Load Environment
    env = DisasterResponseGymEnv(task_id=task_id)
    
    # 2. Load Trained Model
    model_path = f"models/ppo_disaster_final_{task_id}.zip"
    try:
        model = PPO.load(model_path)
        print(f"Loaded model from {model_path}")
    except Exception as e:
        print(f"FAILED to load model: {e}")
        return

    # 3. Run Episode
    obs, _ = env.reset()
    total_reward = 0.0
    done = False
    step = 0
    
    while not done:
        step += 1
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Log Step
        print(f"[STEP {step}] Reward: {reward:+.3f} | Total: {total_reward:+.3f}")
        
        # Check violations in info
        v = info.get("info", {}).get("violations", [])
        if v: print(f"       Violations: {', '.join(v)}")
        
        total_reward += reward
        done = terminated or truncated
        
    print(f"\n[EPISODE END] Total Reward: {total_reward:.2f}")
    if terminated:
        print("STATUS: SUCCESS (Board Cleared)")
    else:
        print("STATUS: FAILED (Timeout)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="citywide_crisis_management")
    args = parser.parse_args()
    evaluate(args.task)
