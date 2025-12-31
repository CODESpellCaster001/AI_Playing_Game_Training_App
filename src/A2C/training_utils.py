# src/A2C/training_utils.py

import os
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import EvalCallback

def setup_eval_callback(env, log_dir='data/performance_logs/logs/', eval_freq=10000):
    """
    Sets up the evaluation callback for the model training.
    
    Parameters:
    env (VecEnv): The environment to evaluate the model on.
    log_dir (str): Directory to save logs and best models.
    eval_freq (int): Frequency of evaluation.
    
    Returns:
    EvalCallback: The evaluation callback object.
    """
    os.makedirs(log_dir, exist_ok=True)
    return EvalCallback(env, best_model_save_path=log_dir,
                        log_path=log_dir, eval_freq=eval_freq,
                        deterministic=True, render=False)

def load_model(env, model_path):
    """
    Loads an existing model. Raises an error if the model path does not exist.
    
    Parameters:
    env (VecEnv): The environment for the model.
    model_path (str): Path to the saved model.
    
    Returns:
    A2C: The loaded A2C model.
    """
    if os.path.exists(model_path):
        print("Model found at:", model_path)
        model = A2C.load(model_path, env=env)
        model.set_env(env)
        return model
    else:
        raise FileNotFoundError(f"No model found at the specified path: {model_path}")

def create_model(env, policy_kwargs={}, log_dir=None):
    """
    Creates a new model.
    
    Parameters:
    env (VecEnv): The environment for the model.
    policy_kwargs (dict): Policy keyword arguments for the model.
    
    Returns:
    A2C: The newly created A2C model.
    """
    print("Creating a new model.")
    model = A2C('CnnPolicy', env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=log_dir)
    return model

def train_model(model, total_timesteps=500, eval_callback=None):
    """
    Trains the A2C model.
    
    Parameters:
    model (A2C): The A2C model to train.
    total_timesteps (int): Total number of timesteps to train the model.
    eval_callback (EvalCallback): Callback for evaluation during training.
    
    Returns:
    A2C: The trained A2C model.
    """
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    return model

def save_model(model, save_path):
    """
    Saves the trained model to a specified path.
    
    Parameters:
    model (A2C): The trained A2C model to save.
    save_path (str): Path to save the model.
    """
    os.makedirs(save_path, exist_ok=True)
    model.save(save_path)

