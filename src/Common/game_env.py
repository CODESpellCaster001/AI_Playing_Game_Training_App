# src/Common/game_env.py
from collections import Counter
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import cv2
import numpy as np
from scipy.stats import skew, kurtosis

def create_env(environment_name='Breakout-v4', n_envs=4, n_stack=4, seed=0):
    env = make_atari_env(environment_name, n_envs=n_envs, seed=seed)
    env.metadata['render_fps'] = 30
    env = VecFrameStack(env, n_stack=n_stack)
    return env

def create_eval_env(environment_name='Breakout-v4', n_envs=1, n_stack=4, seed=0):
    eval_env = make_atari_env(environment_name, n_envs=n_envs, seed=seed)
    eval_env.metadata['render_fps'] = 30
    eval_env = VecFrameStack(eval_env, n_stack=n_stack)
    return eval_env

# def record_env(env, model, video_path, video_fps=30, recording_time=60):
#     env.reset()
#     obs = env.reset()
#     done = np.array([False] * env.num_envs)
#     step = 0
#     max_steps = recording_time * video_fps
#
#     frame = env.render(mode='rgb_array')
#     height, width, _ = frame.shape
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out = cv2.VideoWriter(video_path, fourcc, video_fps, (width, height))
#
#     while step < max_steps:
#         action, _ = model.predict(obs, deterministic=True)
#         obs, reward, done, info = env.step(action)
#         frame = env.render(mode='rgb_array')
#         out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
#
#         if done.any():
#             obs = env.reset()
#
#         step += 1
#
#     out.release()
#     env.close()

def record_env(env, model, video_path, video_fps=30, recording_time=60, pass_threshold=5):
    """
    通用版：跑任何 Gym 环境（只要装了 Monitor wrapper／返回 episode info），
    遇到一局结束就自动记录一局分数，不用写死命数。
    """
    # 确保 env 被 Monitor 包了一层，这样 info[i] 会有 'episode'
    # 如果你还没包： env = Monitor(env, './videos/', force=True)

    obs = env.reset()
    num_envs = getattr(env, 'num_envs', 1)
    step = 0
    max_steps = recording_time * video_fps

    # 准备视频写入
    frame = env.render(mode='rgb_array')
    h, w, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path, fourcc, video_fps, (w, h))

    episode_scores = []

    while step < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)

        # 渲染并写帧
        frame = env.render(mode='rgb_array')
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # 检查 info 里有没有 episode 结束的数据
        # 多 env 时 infos 是列表；单 env 时直接是 dict
        if isinstance(infos, dict):
            infos = [infos]

        for info in infos:
            if 'episode' in info:
                # info['episode']['r'] 是这一局的总 reward
                episode_scores.append(info['episode']['r'])

        # reset done 的 env
        if dones.any() if num_envs > 1 else dones:
            new_obs = env.reset()
            obs = new_obs

        step += 1

    out.release()
    env.close()

    if not episode_scores:
        print("⚠️ 没有完整跑完一局，试试延长 recording_time 或降低 fps。")
        return

    scores = np.array(episode_scores)
    total_eps = len(scores)

    # 基础统计
    mean_score   = scores.mean()
    median_score = np.median(scores)
    max_score    = scores.max()
    min_score    = scores.min()
    var_score    = scores.var()
    std_score    = scores.std()

    # CV
    cv = std_score / mean_score if mean_score != 0 else float('nan')
    # 偏度 & 峰度
    sk = skew(scores)
    kt = kurtosis(scores)
    # 排除最高/最低
    trimmed = scores[(scores != max_score) & (scores != min_score)]
    var_trimmed, std_trimmed = (trimmed.var(), trimmed.std()) if len(trimmed) >= 2 else (float('nan'), float('nan'))
    # 众数
    mode_score, mode_count = Counter(scores).most_common(1)[0]
    # 命中率 & 达标率
    nonzero_eps = np.count_nonzero(scores)
    accuracy = nonzero_eps / total_eps
    pass_count = np.sum(scores >= pass_threshold)
    pass_rate  = pass_count / total_eps

    # 打印
    print("📊 推理统计结果：")
    print(f"  • 完成局数：{total_eps}")
    print(f"  • 有分局数：{nonzero_eps}，命中率：{accuracy:.2%}")
    print(f"  • 阈值：{pass_threshold}，达标局数：{pass_count}，达标率：{pass_rate:.2%}")
    print(f"  • 平均分：{mean_score:.2f}，中位数：{median_score:.2f}")
    print(f"  • 范围：[{min_score:.2f}, {max_score:.2f}]")
    print(f"  • 原方差/标准差：{var_score:.2f}/{std_score:.2f}")
    print(f"  • 去极端后方差/标准差：{var_trimmed:.2f}/{std_trimmed:.2f}")
    print(f"  • 变异系数 CV：{cv:.2f}")
    print(f"  • 偏度/峰度：{sk:.2f}/{kt:.2f}")
    print(f"  • 众数分：{mode_score:.2f}（{mode_count} 次）")

def create_single_env(environment_name='Breakout-v4', n_envs=1, n_stack=1, seed=0):
    env = make_atari_env(environment_name, n_envs=n_envs, seed=seed)
    env.metadata['render_fps'] = 30
    if n_stack > 0:
        env = VecFrameStack(env, n_stack=n_stack)
    return env

def record_fsm_env(env, fsm_agent, video_path, video_fps=30, recording_time=60):
    env.reset()
    state = env.reset()
    done = False
    step = 0
    max_steps = recording_time * video_fps

    frame = env.render(mode='rgb_array')
    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path, fourcc, video_fps, (width, height))

    while step < max_steps:
        action = fsm_agent.act(frame)
        state, reward, done, info = env.step(action)
        frame = env.render(mode='rgb_array')
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        if done:
            state = env.reset()

        step += 1

    out.release()
    env.close()
