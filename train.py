import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from gym_wrapper import DisasterResponseGymEnv

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=20000)
    parser.add_argument("--task", type=str, default="citywide_crisis_management")
    args = parser.parse_args()

    print(f"Starting PPO training on {args.task} for {args.timesteps} steps...")

    # 1. Create and Wrap Environment
    # We use a lambda to ensure the env is created correctly in the vec_env
    env = DisasterResponseGymEnv(task_id=args.task)
    
    # 2. Initialize PPO Model
    # MultiInputPolicy is required for spaces.Dict observations
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log="./logs/ppo_disaster_tensorboard/"
    )

    # 3. Setup Callpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path="./models/",
        name_prefix="ppo_disaster_model"
    )

    # 4. Train
    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint_callback,
        progress_bar=False
    )

    # 5. Save Final Model
    model_path = os.path.join("models", f"ppo_disaster_final_{args.task}")
    model.save(model_path)
    print(f"Training complete. Model saved to {model_path}")

if __name__ == "__main__":
    # Ensure models and logs directories exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    train()
