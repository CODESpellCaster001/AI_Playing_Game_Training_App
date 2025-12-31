# src/gui.py

import customtkinter as ctk
from tkinter import filedialog, messagebox
import threading
import subprocess
from DQN.dqn import main as train_dqn_model_main
from DQN.training_utils import load_model as load_dqn_model
from QRDQN.qrdqn import main as train_qrdqn_model_main
from QRDQN.training_utils import load_model as load_qrdqn_model
from PPO.ppo import main as train_ppo_model_main
from PPO.training_utils import load_model as load_ppo_model
from A2C.a2c import main as train_a2c_model_main
from A2C.training_utils import load_model as load_a2c_model
from Common.game_env import create_env, record_env, create_single_env, record_fsm_env
import tkinter as tk
from tkinter import ttk, filedialog
import subprocess
from stable_baselines3 import DQN
import numpy as np
import webbrowser
import os
import psutil

# ================================
# 主题配置
# ================================
ctk.set_appearance_mode("dark")  # 可选: "dark", "light", "system"
ctk.set_default_color_theme("blue")  # 可选: "blue", "green", "dark-blue"
class BreakoutAIApp:
    def __init__(self, root):
        """
        Initialize the AI training interface application.

        Parameters:
        root (Tk): The root window of the Tkinter application.
        """
        self.root = root
        self.root.title("强化学习与神经网络协同优化在Atari游戏中的智能决策系统")
        # List of Atari games available in OpenAI Gym
        self.env_options = ["Breakout-v4", "AirRaid-v4", "Alien-v4", "Amidar-v4", "Assault-v4", "Asterix-v4",
                            "Asteroids-v4",
                            "Atlantis-v4", "BankHeist-v4", "BattleZone-v4", "BeamRider-v4", "Berzerk-v4",
                            "Bowling-v4", "Boxing-v4", "Carnival-v4", "Centipede-v4",
                            "ChopperCommand-v4", "CrazyClimber-v4", "Defender-v4", "DemonAttack-v4",
                            "DoubleDunk-v4", "ElevatorAction-v4", "Enduro-v4", "FishingDerby-v4",
                            "Freeway-v4", "Frostbite-v4", "Gopher-v4", "Gravitar-v4", "Hero-v4",
                            "IceHockey-v4", "Jamesbond-v4", "JourneyEscape-v4", "Kangaroo-v4",
                            "Krull-v4", "KungFuMaster-v4", "MontezumaRevenge-v4", "MsPacman-v4",
                            "NameThisGame-v4", "Phoenix-v4", "Pitfall-v4", "Pong-v4", "PrivateEye-v4",
                            "Qbert-v4", "Riverraid-v4", "RoadRunner-v4", "Robotank-v4", "Seaquest-v4",
                            "Skiing-v4", "Solaris-v4", "SpaceInvaders-v4", "StarGunner-v4",
                            "Tennis-v4", "TimePilot-v4", "Tutankham-v4", "UpNDown-v4",
                            "Venture-v4", "VideoPinball-v4", "WizardOfWor-v4", "YarsRevenge-v4",
                            "Zaxxon-v4"]
        # Create and initialize the GUI widgets
        self.create_widgets()

        self.env = None
        self.fsm_agent = None
        self.monitoring = False
        self.tensorboard_port = 6012

    def create_widgets(self):
        # ================================
        # Training Section
        # ================================
        self.training_frame = tk.LabelFrame(self.root, text="Train Model", padx=10, pady=10)
        self.training_frame.pack(fill="both", expand="yes", padx=10, pady=5)

        # --- 第一行：算法选择 ---
        self.algo_label = tk.Label(self.training_frame, text="Select Algorithm:")
        self.algo_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.algorithm_options = ["DQN", "PPO", "A2C", "QRDQN"]
        self.algo_var = tk.StringVar(self.training_frame)
        self.algo_var.set(self.algorithm_options[0])  # 默认选择 DQN
        self.algo_menu = ttk.Combobox(self.training_frame, textvariable=self.algo_var,
                                      values=self.algorithm_options, state="readonly")
        self.algo_menu.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # --- 第二行：游戏环境选择 ---
        self.env_label = tk.Label(self.training_frame, text="Choose Game Environment:")
        self.env_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.env_var = tk.StringVar(self.training_frame)
        self.env_var.set(self.env_options[0])
        self.env_menu = ttk.Combobox(self.training_frame, textvariable=self.env_var,
                                     values=self.env_options, state="readonly")
        self.env_menu.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        # --- 第三行：训练参数 ---
        self.timesteps_label = tk.Label(self.training_frame, text="Timesteps:")
        self.timesteps_label.grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.timesteps_entry = tk.Entry(self.training_frame)
        self.timesteps_entry.insert(0, "50000")
        self.timesteps_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")

        self.vectorstacks_label = tk.Label(self.training_frame, text="Vector Stacks:")
        self.vectorstacks_label.grid(row=2, column=2, padx=5, pady=5, sticky="w")
        self.vectorstacks_entry = tk.Entry(self.training_frame)
        self.vectorstacks_entry.insert(0, "4")
        self.vectorstacks_entry.grid(row=2, column=3, padx=5, pady=5, sticky="w")

        self.env_count_label = tk.Label(self.training_frame, text="Number of Environments:")
        self.env_count_label.grid(row=2, column=4, padx=5, pady=5, sticky="w")
        self.env_count_entry = tk.Entry(self.training_frame)
        self.env_count_entry.insert(0, "4")
        self.env_count_entry.grid(row=2, column=5, padx=5, pady=5, sticky="w")

        # --- 第四行：保存和日志目录，以及启动训练 ---
        self.save_button = tk.Button(self.training_frame, text="Choose Save Directory",
                                     command=self.choose_save_dir)
        self.save_button.grid(row=3, column=0, padx=5, pady=5)
        self.log_button = tk.Button(self.training_frame, text="Choose Log Directory",
                                    command=self.choose_log_dir)
        self.log_button.grid(row=3, column=1, padx=5, pady=5)
        self.train_button = tk.Button(self.training_frame, text="Train Model", command=self.train_model)
        self.train_button.grid(row=3, column=2, padx=5, pady=5)

        # ================================
        # TensorBoard Section
        # ================================
        self.tensorboard_frame = tk.LabelFrame(self.root, text="View Training Results", padx=10, pady=10)
        self.tensorboard_frame.pack(fill="both", expand="yes", padx=10, pady=5)
        self.log_button_tb = tk.Button(self.tensorboard_frame, text="Choose Log Directory", command=self.choose_log_dir)
        self.log_button_tb.grid(row=0, column=0, padx=5, pady=5)
        self.tensorboard_button = tk.Button(self.tensorboard_frame, text="Show TensorBoard",
                                            command=self.run_tensorboard)
        self.tensorboard_button.grid(row=0, column=1, padx=5, pady=5)
        self.tensorboard_output = tk.Label(self.tensorboard_frame, text="", wraplength=400)
        self.tensorboard_output.grid(row=1, column=0, columnspan=2, padx=5, pady=5)
        self.tensorboard_hint = tk.Label(self.tensorboard_frame,
                                         text="Note: TensorBoard may cache previous runs. If the log does not update, try again or use a new port.",
                                         wraplength=400, fg="gray")
        self.tensorboard_hint.grid(row=2, column=0, columnspan=2, padx=5, pady=5)

        # ================================
        # Monitor Section
        # ================================
        self.monitor_frame = tk.LabelFrame(self.root, text="Monitor Agent", padx=10, pady=10)
        self.monitor_frame.pack(fill="both", expand="yes", padx=10, pady=5)
        # 在监控区域 monitor_frame 中，添加监控算法下拉菜单
        self.monitor_algo_label = tk.Label(self.monitor_frame, text="Select Model Algorithm:")
        self.monitor_algo_label.grid(row=0, column=3, padx=5, pady=5, sticky="w")
        self.monitor_algorithm_options = ["DQN", "PPO", "A2C", "QRDQN"]
        self.monitor_algo_var = tk.StringVar(self.monitor_frame)
        self.monitor_algo_var.set(self.monitor_algorithm_options[0])  # 默认选择 DQN
        self.monitor_algo_menu = ttk.Combobox(self.monitor_frame, textvariable=self.monitor_algo_var,
                                              values=self.monitor_algorithm_options, state="readonly")
        self.monitor_algo_menu.grid(row=0, column=4, padx=5, pady=5, sticky="w")
        self.model_button = tk.Button(self.monitor_frame, text="Choose Model File", command=self.choose_model_file)
        self.model_button.grid(row=0, column=0, padx=5, pady=5)
        self.monitor_env_label = tk.Label(self.monitor_frame, text="Choose Game Environment:")
        self.monitor_env_label.grid(row=0, column=1, padx=5, pady=5)
        self.monitor_env_var = tk.StringVar(self.monitor_frame)
        self.monitor_env_var.set(self.env_options[0])
        self.monitor_env_menu = ttk.Combobox(self.monitor_frame, textvariable=self.monitor_env_var,
                                             values=self.env_options, state="readonly")
        self.monitor_env_menu.grid(row=0, column=2, padx=5, pady=5)

        # 为监控部分的环境个数和 vector stacks 使用独立控件名称
        self.monitor_env_count_label = tk.Label(self.monitor_frame, text="Number of Environments:")
        self.monitor_env_count_label.grid(row=1, column=0, padx=5, pady=5)
        self.monitor_env_count_entry = tk.Entry(self.monitor_frame)
        self.monitor_env_count_entry.insert(0, "4")
        self.monitor_env_count_entry.grid(row=1, column=1, padx=5, pady=5)

        self.monitor_vectorstacks_label = tk.Label(self.monitor_frame, text="Vector Stacks:")
        self.monitor_vectorstacks_label.grid(row=1, column=2, padx=5, pady=5)
        self.monitor_vectorstacks_entry = tk.Entry(self.monitor_frame)
        self.monitor_vectorstacks_entry.insert(0, "4")
        self.monitor_vectorstacks_entry.grid(row=1, column=3, padx=5, pady=5)

        self.recording_time_label = tk.Label(self.monitor_frame, text="Recording Time (seconds):")
        self.recording_time_label.grid(row=1, column=4, padx=5, pady=5)
        self.recording_time_entry = tk.Entry(self.monitor_frame)
        self.recording_time_entry.insert(0, "60")
        self.recording_time_entry.grid(row=1, column=5, padx=5, pady=5)

        self.monitor_button = tk.Button(self.monitor_frame, text="Monitor Agent", command=self.start_monitoring_thread)
        self.monitor_button.grid(row=2, column=0, columnspan=6, padx=5, pady=5)


    # -------------------------------
    # Training Section Functions
    # -------------------------------
    def choose_save_dir(self):
        """Allow the user to select a directory for saving the trained model."""
        self.save_dir = filedialog.askdirectory()
        if self.save_dir:
            messagebox.showinfo("Save Directory Selected", f"Save directory: {self.save_dir}")

    def choose_log_dir(self):
        """Allow the user to select a directory for storing training logs."""
        self.log_dir = filedialog.askdirectory()
        if self.log_dir:
            messagebox.showinfo("Log Directory Selected", f"Log directory: {self.log_dir}")

    def train_model(self):
        """
        Start the training process for the model in a separate thread.
        根据算法下拉菜单的选择，调用对应的训练入口。
        """

        def train_thread():
            env_name = self.env_var.get()
            save_dir = getattr(self, 'save_dir', './Training/Saved_Models/')
            log_dir = getattr(self, 'log_dir', './Training/Logs/')
            timesteps = int(self.timesteps_entry.get())
            vectorstacks = int(self.vectorstacks_entry.get())
            env_count = int(self.env_count_entry.get())
            algorithm = self.algo_var.get()

            # 根据所选算法调用不同训练函数
            if algorithm == "DQN":
                train_dqn_model_main(env_name, save_dir, log_dir, timesteps, vectorstacks, env_count)
            elif algorithm == "PPO":
                train_ppo_model_main(env_name, save_dir, log_dir, timesteps, vectorstacks, env_count)
            elif algorithm == "A2C":
                train_a2c_model_main(env_name, save_dir, log_dir, timesteps, vectorstacks, env_count)
            elif algorithm == "QRDQN":
                train_qrdqn_model_main(env_name, save_dir, log_dir, timesteps, vectorstacks, env_count)
            else:
                messagebox.showerror("Algorithm Error", f"Unsupported algorithm: {algorithm}")
                return

            # 训练完成后，在主线程显示提示
            self.root.after(0, lambda: messagebox.showinfo("Training Complete", f"{algorithm} training has finished."))

        threading.Thread(target=train_thread, daemon=True).start()

    # -------------------------------
    # TensorBoard Section Functions
    # -------------------------------
    def run_tensorboard(self):
        """
        Start TensorBoard in a separate thread to visualize training logs.
        """
        if hasattr(self, 'log_dir'):
            tb_thread = threading.Thread(target=self._run_tensorboard)
            tb_thread.start()
        else:
            messagebox.showwarning("Log Directory Not Selected", "Please select a log directory first.")

    def _run_tensorboard(self):
        """
        Run the TensorBoard process in the background and display the output URL.
        """
        try:
            print(f"[DEBUG] TensorBoard logdir: {self.log_dir}")
            if not hasattr(self, 'tensorboard_port'):
                self.tensorboard_port = 6012
            else:
                self.tensorboard_port += 1
            port = str(self.tensorboard_port)
            subprocess.Popen([
                "tensorboard", "--logdir", self.log_dir,
                "--port", port, "--reload_interval", "5"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.tensorboard_output.config(text=f"TensorBoard is running at: http://localhost:{port}")
            webbrowser.open(f"http://localhost:{port}")
        except Exception as e:
            self.tensorboard_output.config(text=f"Error running TensorBoard: {e}")

    # -------------------------------
    # Monitor Section Functions
    # -------------------------------
    def choose_model_file(self):
        """Allow the user to select the trained DQN model file for monitoring."""
        self.model_file = filedialog.askopenfilename()
        if self.model_file:
            messagebox.showinfo("Model File Selected", f"Model file: {self.model_file}")

    def start_monitoring_thread(self):
        """Start a separate thread to monitor the DQN agent."""
        self.monitoring_thread = threading.Thread(target=self.monitor_agent, daemon=True)
        self.monitoring_thread.start()


    def monitor_agent(self):
        """
        Monitor the performance of the trained agent and record a video of its gameplay.
        使用监控区的环境个数和 vector stacks 输入，并根据算法选择加载模型。
        """
        env_name = self.monitor_env_var.get()
        model_file = self.model_file
        env_count = int(self.monitor_env_count_entry.get())
        vectorstacks = int(self.monitor_vectorstacks_entry.get())
        recording_time = int(self.recording_time_entry.get())

        if not model_file:
            messagebox.showwarning("Model File Not Selected", "Please select a model file first.")
            return

        self.env = create_env(environment_name=env_name, n_envs=env_count, n_stack=vectorstacks)

        # 根据监控算法下拉菜单的选择加载不同的模型
        algorithm = self.monitor_algo_var.get()
        if algorithm == "DQN":
            self.model = load_dqn_model(env=self.env, model_path=model_file)
        elif algorithm == "PPO":
            self.model = load_ppo_model(env=self.env, model_path=model_file)
        elif algorithm == "A2C":
            self.model = load_a2c_model(env=self.env, model_path=model_file)
        elif algorithm == "QRDQN":
            self.model = load_qrdqn_model(env=self.env, model_path=model_file)
        else:
            messagebox.showerror("Algorithm Error", f"Unsupported algorithm: {algorithm}")
            return

        video_path = filedialog.asksaveasfilename(defaultextension=".avi", filetypes=[("AVI files", "*.avi")])
        if video_path:
            self.record_agent(video_path, recording_time)
    def record_agent(self, video_path, recording_time):
        """
        Record the gameplay of the DQN agent and save it as a video file.
        """
        record_env(self.env, self.model, video_path, recording_time=recording_time)
        messagebox.showinfo("Recording Complete", f"Video saved to: {video_path}")



if __name__ == "__main__":
    root = tk.Tk()
    app = BreakoutAIApp(root)
    root.mainloop()
