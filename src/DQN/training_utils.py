# src/DQN/training_utils.py

import os
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback

def setup_eval_callback(env, log_dir='data/performance_logs/logs/', eval_freq=10000):

    os.makedirs(log_dir, exist_ok=True)
    return EvalCallback(env, best_model_save_path=log_dir,
                        log_path=log_dir, eval_freq=eval_freq,
                        deterministic=True, render=False)

def load_model(env, model_path):

    if os.path.exists(model_path):
        print("Model found at:", model_path)
        model = DQN.load(model_path, env=env)
        model.set_env(env)
        return model
    else:
        raise FileNotFoundError(f"No model found at the specified path: {model_path}")

def create_model(env, policy_kwargs={}, log_dir=None):

    print("Creating a new model.")
    model = DQN('CnnPolicy', env, policy_kwargs=policy_kwargs, verbose=1, buffer_size=10000, tensorboard_log=log_dir)
    return model

def train_model(model, total_timesteps=500, eval_callback=None):

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    return model

def save_model(model, save_path):

    os.makedirs(save_path, exist_ok=True)
    model.save(save_path)

