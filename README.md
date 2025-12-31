AI 游戏训练界面
项目概述
本文档提供了“AI 游戏训练界面”应用程序的详细技术概述。该应用程序旨在使用不同的机器学习模型（包括自定义的基于CNN的深度Q网络（DQN）、近端策略优化（PPO）和优势演员评论家（A2C）），训练、监控和评估在Atari游戏中操作的AI智能体。

目录
项目概述
安装说明
使用方法
功能特性
文件结构
使用技术

本地运行本项目的步骤如下：

下载软件压缩包并解压

创建虚拟环境：
python -m venv venv
source venv/bin/activate  # Windows系统使用 

安装依赖库：
pip install -r requirements.txt

使用方法
运行主脚本启动应用程序：
python src/main.py
图形用户界面（GUI）将打开，支持以下操作：

训练智能体

监控智能体性能

对比不同AI算法的结果

选择游戏环境。

设置训练参数。

选择模型保存目录和日志目录。

点击“开始训练”。

监控智能体性能

选择算法后，加载训练好的模型并设置监控参数。

功能特性
训练模块：支持自定义参数训练智能体。

TensorBoard可视化模块：实时可视化训练进度。

监控模块：观察已训练智能体的行为。

对比功能：对比不同智能体的性能指标。

文件结构
项目结构如下：

AI_Playing_Game_Training_App/
├── data/
│   ├── models/                 # 保存模型
│   ├── performance_logs/       # 模型性能指标日志
├── docs/
│   ├── user_guide.md           # 用户指南
├── src/
│   ├── Common/
│   │   ├── __init__.py         
│   │   ├── game_env.py         # 游戏环境封装模块
│   ├── DQN/
│   │   ├── __init__.py 
│   │   ├── cnn_architecture.py # DQN自定义CNN架构
│   │   ├── dqn.py              # DQN实现
│   │   ├── training_utils.py   # DQN训练工具
│   ├── PPO/
│   │   ├── __init__.py 
│   │   ├── cnn_architecture.py # PPO自定义CNN架构
│   │   ├── ppo.py              # PPO实现
│   │   ├── training_utils.py   # PPO训练工具
│   ├── A2C/
│   │   ├── __init__.py 
│   │   ├── cnn_architecture.py # A2C自定义CNN架构
│   │   ├── a2c.py              # A2C实现
│   │   ├── training_utils.py   # A2C训练工具
│   ├── __init__.py 
│   ├── gui.py                  # 图形用户界面实现
│   ├── main.py                 # 应用程序主入口
├── tests/
│   ├── __init__.py 
│   ├── test_dqn.py             # DQN单元测试
│   ├── test_render.py          # 游戏环境中DQN智能体监控测试
├── requirements.txt            # 依赖库列表
└── README.md                   # 项目概述与安装说明
使用技术
Python：主要编程语言。

PyTorch：用于构建和训练模型的深度学习框架。

stable-baselines3：强化学习算法库。

OpenAI Gym：强化学习算法开发与测试工具包。

tkinter：Python GUI开发库。

NumPy：数值计算库。

OpenCV：计算机视觉任务库。
