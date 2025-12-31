# src/PPO/ppo.py

import os
import datetime
from PPO.cnn_architecture import CustomCNN
from Common.game_env import create_env, create_eval_env
from PPO.training_utils import setup_eval_callback, create_model, train_model, save_model


def main(env_name, save_dir, log_dir, timesteps, vectorstacks, env_count):
    # Policy with custom CNN feature extractor
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=128),
    )
    # Create environment
    env = create_env(environment_name=env_name, n_envs=env_count, n_stack=vectorstacks)

    # Create evaluation environment with the same setup
    eval_env = create_eval_env(environment_name=env_name, n_envs=env_count, n_stack=vectorstacks)

    # 处理环境名称以适配文件名（与模型保存路径一致）
    base_env_name = env_name.split("-")[0].replace("NoFrameskip", "")
    sanitized_env_name = base_env_name.replace(" ", "_").lower()

    # 在日志路径中添加游戏名称和算法名（例如：./Training/Logs/DQN_Breakout_20231025_153000）
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_subdir = f"PPO_{sanitized_env_name}_{current_time}"
    log_dir = os.path.join(log_dir, log_subdir)  # 更新log_dir路径
    # Load or create model
    model = create_model(env, policy_kwargs=policy_kwargs, log_dir=log_dir)

    # Setup evaluation callback
    eval_callback = setup_eval_callback(eval_env, log_dir=log_dir)

    # Train the model
    model = train_model(model, total_timesteps=timesteps, eval_callback=eval_callback)

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # 处理环境名称以适配文件名
    base_env_name = env_name.split("-")[0].replace("NoFrameskip", "")  # 移除版本号和冗余字段
    sanitized_env_name = base_env_name.replace(" ", "_").lower()  # 处理空格和大小写
    updated_model_path = os.path.join(
        save_dir,
        f"CustomCNN_PPO_{sanitized_env_name}_{current_time}"
    )
    save_model(model, updated_model_path)


if __name__ == '__main__':
    import sys

    args = sys.argv[1:]
    env_name = args[0] if len(args) > 0 else 'Breakout-v4'
    save_dir = args[1] if len(args) > 1 else './Training/Saved_Models/'
    log_dir = args[2] if len(args) > 2 else './Training/Logs/'
    timesteps = int(args[3]) if len(args) > 3 else 50000
    vectorstacks = int(args[4]) if len(args) > 4 else 4
    env_count = int(args[5]) if len(args) > 5 else 4
    main(env_name, save_dir, log_dir, timesteps, vectorstacks, env_count)
