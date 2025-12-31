import tkinter as tk
import webbrowser
import subprocess
import threading
import customtkinter as ctk
from tkinter import filedialog, messagebox
from DQN.dqn import main as train_dqn_model_main
from DQN.training_utils import load_model as load_dqn_model
from PPO.ppo import main as train_ppo_model_main
from PPO.training_utils import load_model as load_ppo_model
from A2C.a2c import main as train_a2c_model_main
from A2C.training_utils import load_model as load_a2c_model
from Common.game_env import create_env, record_env
from PIL import Image, ImageSequence
# 主题配置
ctk.set_appearance_mode("system")
ctk.set_default_color_theme("dark-blue")

# 字体设置
# 字体配置（全系统默认字体保障兼容性）
FONT_TITLE = ("Segoe UI", 28, "bold")          # Windows现代字体
FONT_LABEL = ("Arial", 22, "bold")            # 通用粗体
FONT_TAB_TITLE = ("Franklin Gothic Medium", 22, "bold")  # 紧凑科技感
FONT_DECOR = ("Calibri", 18, "italic")  # 优雅斜体
FONT_WELCOME = ("Impact", 48, "normal")       # 厚重标题

# 科技感配色方案
CYBER_BLUE = "#00FFFF"       # 赛博青
NEON_PURPLE = "#9D00FF"       # 霓虹紫
DARK_SPACE = "#0A0A12"        # 深空黑
HUD_GREEN = "#39FF14"         # HUD绿

class BreakoutAIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Playing Games Training Interface")
        self.root.geometry("1920x1080")

        # 环境选项列表
        self.env_options = [
            "Breakout-v4", "AirRaid-v4", "Alien-v4", "Amidar-v4", "Assault-v4", "Asterix-v4",
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
            "Zaxxon-v4"
        ]

        self.env = None
        self.model = None
        self.tensorboard_port = 6012
        # GIF 路径与尺寸
        self.gif_paths = {
            'gif1': ('assets/train1.gif', (200, 200)),
            'gif2': ('assets/train2.gif', (200, 200)),
            'gif3': ('assets/tb1.gif', (400, 200)),
            'gif4': ('assets/tb2.gif', (400, 200)),
            'gif5': ('assets/monitor1.gif', (200, 200)),
            'gif6': ('assets/monitor2.gif', (200, 200)),
        }
        self.preloaded_gifs = {}
        threading.Thread(target=self._preload_gifs, daemon=True).start()
        # 先显示欢迎界面
        self.create_welcome_screen()

    def _preload_gifs(self):
        """后台线程中预加载所有 GIF 帧"""
        for key, (path, size) in self.gif_paths.items():
            try:
                pil = Image.open(path)
                frames = [
                    ctk.CTkImage(light_image=frame.copy().resize(size),
                                 dark_image=frame.copy().resize(size),
                                 size=size)
                    for frame in ImageSequence.Iterator(pil)
                ]
                self.preloaded_gifs[key] = frames
            except Exception as e:
                print(f"Preload {key} error: {e}")

    def _create_gif_label(self, parent, key, row, column, rowspan=2, colspan=1):
        """创建 GIF 标签并后台动画"""
        # 如果已经预加载，直接用缓存，否则临时加载第一帧
        if key in self.preloaded_gifs:
            frames = self.preloaded_gifs[key]
        else:
            path, size = self.gif_paths[key]
            try:
                pil = Image.open(path)
                frame = pil.copy().resize(size)
                img = ctk.CTkImage(light_image=frame, dark_image=frame, size=size)
                frames = [img]
            except:
                frames = [None]

        label = ctk.CTkLabel(parent, image=frames[0], text="")
        label.grid(row=row, column=column, rowspan=rowspan, columnspan=colspan, padx=10, pady=10)

        def animate():
            if key in self.preloaded_gifs:
                frames = self.preloaded_gifs[key]
                idx = getattr(self, f"{key}_idx", 0)
                idx = (idx + 1) % len(frames)
                setattr(self, f"{key}_idx", idx)
                label.configure(image=frames[idx])
            label.after(100, animate)

        animate()
        setattr(self, f"{key}_label", label)

    def create_welcome_screen(self):
        """创建欢迎界面，用户点击开始后进入主界面"""
        self.welcome_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        self.welcome_frame.pack(fill="both", expand=True)

        # 背景图像
        try:
            bg_image = ctk.CTkImage(
                light_image=Image.open("assets/bg_light.png"),
                dark_image=Image.open("assets/bg_dark.png"),
                size=(1920, 1080)
            )
            self.welcome_bg_label = ctk.CTkLabel(self.welcome_frame, image=bg_image, text="")
            self.welcome_bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        except Exception as e:
            print(f"Welcome bg load error: {e}")



        # 空白文本框（用于后续动态展示或其他用途）
        empty_textbox = ctk.CTkTextbox(
            self.welcome_frame,
            width=1920,
            height=500,  # 设置不同高度
            corner_radius=12,
            font=FONT_DECOR,
            border_width=1,  # 添加细边框
            border_color="#E2E8F0",  # 浅灰色边框
            fg_color="#FFFFFF"  # 白色背景
        )
        empty_textbox.configure(state="disabled")  # 保持禁用状态
        empty_textbox.place(relx=0.5, rely=0.9, anchor="center")  # 调整位置到下方


        # 说明文本框
        info_box = ctk.CTkTextbox(
            self.welcome_frame,
            width=720,
            height=440,
            corner_radius=12,
            font=FONT_DECOR,
            border_width=0,
            border_color="#4A5568"

        )
        info_box.insert("0.0", """平台功能指南：
1. 训练模块 - 选择算法并开始训练
2. TensorBoard - 可视化训练过程
3. 监控模块 - 回放训练结果

操作提示：
• 点击上方按钮开始使用
• 确保已选择正确的游戏环境
• 训练前请指定保存目录""")
        info_box.configure(state="disabled")
        info_box.place(relx=0.65, rely=0.95, anchor="center")

        start_btn = ctk.CTkButton(
            self.welcome_frame,
            text="开始体验",
            font=FONT_LABEL,
            width=200,
            height=60,
            fg_color="white",  # 设置按钮主体颜色为白色
            text_color="#7FFF00",  # 设置文字颜色为黑色
            hover_color="#DDDDDD",  # 可选：设置悬停时的浅灰色
            command=self.start_app
        )
        start_btn.place(relx=0.5, rely=0.66, anchor="center")

    def start_app(self):
        """销毁欢迎界面，初始化主界面"""
        self.welcome_frame.destroy()
        self.create_widgets()
        self.set_background_image()

    def set_background_image(self):
        try:
            bg_image = ctk.CTkImage(
                light_image=Image.open("assets/bg_light.png"),
                dark_image=Image.open("assets/bg_dark.png"),
                size=(1920, 1080)
            )
            self.bg_label = ctk.CTkLabel(self.root, image=bg_image, text="")
            self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)
            self.tabview.lift()
        except Exception as e:
            print(f"Background image error: {e}")



    def create_widgets(self):
        title = ctk.CTkLabel(self.root, text="🎮 AI Playing Games Training Interface", font=FONT_TITLE, fg_color="transparent")
        title.pack(pady=30)



        self.tabview = ctk.CTkTabview(
            self.root,

            segmented_button_selected_color="#38B2AC",
            segmented_button_unselected_color="#2D3748",
            segmented_button_selected_hover_color="#4FD1C5",
            segmented_button_unselected_hover_color="#4A5568",
            corner_radius=10,
            border_width=0,
            border_color="#4A5568",
            fg_color="transparent",

        )
        self.tabview.pack(fill="both", expand=True, padx=100, pady=50)
        self.tabview.add("Train")
        self.tabview.add("TensorBoard")
        self.tabview.add("Monitor")

        # Train Tab
        train_tab = self.tabview.tab("Train")
        train_tab.grid_rowconfigure([0,1,2,3,4], weight=1)
        train_tab.grid_columnconfigure([0,1,2,3,4,5], weight=1)

        # —— 1. 第一个 GIF ——
        self._create_gif_label(train_tab, 'gif1', 0, 2)
        self._create_gif_label(train_tab, 'gif2', 0, 3)


        desc = ctk.CTkLabel(
            train_tab,
            text="图1，训练后台记录",
            font=FONT_DECOR,
            fg_color="white",  # 深色底，圆角文本框
            corner_radius=8,
            wraplength=300,
            justify="left"
        )
        # 放在 GIF 下面
        desc.grid(row=2, column=2, padx=10, pady=(0, 20))

        desc2 = ctk.CTkLabel(
            train_tab,
            text="图2，gui界面代码",
            font=FONT_DECOR,
            fg_color="white",  # 深色底，圆角文本框
            corner_radius=8,
            wraplength=300,
            justify="left"
        )
        # 放在 GIF 下面
        desc2.grid(row=2, column=3, padx=10, pady=(0, 20))

        desc_combined = ctk.CTkLabel(
            train_tab,
            text=(
                "1. Select Algorithm: 请先选择您想要训练的算法\n"
                "2. Game Environment: 选择您想要的游戏环境\n"
                "3. Timesteps: 输入您的训练时间步数\n"
                "4. Vector Stacks: 帧数堆叠数量，不建议改变\n"
                "5. Environments: 输入您想要并行运行的环境数量"
            ),
            font=FONT_DECOR,
            fg_color="white",  # 深色底，圆角文本框
            corner_radius=0,
            wraplength=400,
            justify="left"
        )
        desc_combined.grid(row=3, column=2, rowspan=2, columnspan=2, padx=10, pady=(0, 20))

        ctk.CTkLabel(train_tab, text="✨Select Algorithm:", font=FONT_LABEL).grid(row=0, column=0, sticky="w", padx=10, pady=10)
        self.algo_var = tk.StringVar(value="DQN")
        ctk.CTkComboBox(train_tab, values=["DQN","PPO","A2C"], variable=self.algo_var, width=300).grid(row=0, column=1, padx=10, pady=10)

        ctk.CTkLabel(train_tab, text="🕹Game Environment:", font=FONT_LABEL).grid(row=1, column=0, sticky="w", padx=10, pady=10)
        self.env_var = tk.StringVar(value=self.env_options[0])
        ctk.CTkComboBox(train_tab, values=self.env_options, variable=self.env_var, width=300).grid(row=1, column=1, padx=20, pady=20)

        self.timesteps_entry = self._add_labeled_entry(train_tab, "⏱️Timesteps:", "50000", 2, 0)
        self.vectorstacks_entry = self._add_labeled_entry(train_tab, "🔢Vector Stacks:", "4", 3, 0)
        self.env_count_entry = self._add_labeled_entry(train_tab, "🌐Environments:", "4", 4, 0)

        button_width = 260
        ctk.CTkButton(train_tab, text="💾Choose Save Dir", width=button_width, command=self.choose_save_dir).grid(row=5,column=0,padx=20,pady=30,sticky="ew")
        ctk.CTkButton(train_tab, text="💾Choose Log Dir", width=button_width, command=self.choose_log_dir).grid(row=5, column=1,padx=20,pady=30,sticky="ew")
        ctk.CTkButton(train_tab, text="🚀Train Model", width=button_width, command=self.train_model).grid(row=5,column=2,padx=20,pady=30,sticky="ew")
        # 配置第 5 列的权重
        train_tab.grid_columnconfigure(5, weight=1)

        # 添加“退出”按钮
        exit_button = ctk.CTkButton(
            train_tab,
            text="❌ 退出",
            width=100,
            fg_color="#E53E3E",  # 红色背景
            text_color="white",  # 白色文字
            corner_radius=10,
            command=self.root.quit  # 点击按钮时退出应用
        )
        exit_button.grid(row=5, column=4, padx=20, pady=20, sticky="se")
        # ctk.CTkButton(train_tab, text="💾Choose Save Dir", width=260, corner_radius=10, command=self.choose_save_dir).grid(row=5, column=0, padx=20, pady=30)
        # ctk.CTkButton(train_tab, text="💾Choose Log Dir", width=260, corner_radius=10, command=self.choose_log_dir).grid(row=5, column=1, padx=20, pady=30)
        # ctk.CTkButton(train_tab, text="🚀Train Model", width=360, corner_radius=10, command=self.train_model).grid(row=5, column=2, padx=20, pady=30)

        ctk.CTkLabel(train_tab, text="✨ Ready to Rock! ✨", font=FONT_DECOR).grid(row=6, column=0, columnspan=6)

        # TensorBoard Tab###############
        tb_tab = self.tabview.tab("TensorBoard")
        tb_tab.grid_rowconfigure([0, 1, 2, 3, 4, 5], weight=1)
        # 2. 配置列：0～5 共 6 列，统一 weight 和 uniform 保证等宽
        for col in range(6):
            tb_tab.grid_columnconfigure(col, weight=1, uniform="col")

        # Title Label
        title_label = ctk.CTkLabel(tb_tab, text="📊View Training Results", font=FONT_TAB_TITLE)
        title_label.grid(row=0, column=0, columnspan=6, pady=30)



        # Choose Log Directory Button
        choose_log_btn = ctk.CTkButton(tb_tab, text="💾Choose Log Directory", width=100, corner_radius=10,command=self.choose_log_dir)
        choose_log_btn.grid(row=3, column=1, columnspan=2,pady=20, sticky="ew")

        self._create_gif_label(tb_tab, 'gif3', 1, 1, colspan=2)
        self._create_gif_label(tb_tab, 'gif4', 1, 3, colspan=2)


        # Show TensorBoard Button
        show_tb_btn = ctk.CTkButton(tb_tab, text="📈Show TensorBoard", width=100, corner_radius=10,command=self.run_tensorboard)
        show_tb_btn.grid(row=3, column=3, columnspan=2, pady=20, sticky="ew")

        # Info Label
        self.tb_info = ctk.CTkLabel(tb_tab, text="", wraplength=1000, font=FONT_LABEL)
        self.tb_info.grid(row=5, column=0, columnspan=6, pady=20)

        # Decorative Text
        decor_label = ctk.CTkLabel(tb_tab, text="🪄 Data Magic Loading! 🪄", font=FONT_DECOR)
        decor_label.grid(row=6, column=0, columnspan=6)

        # 添加“退出”按钮
        exit_button = ctk.CTkButton(
            tb_tab,
            text="❌ 退出",
            width=100,
            fg_color="#E53E3E",  # 红色背景
            text_color="white",  # 白色文字
            corner_radius=10,
            command=self.root.quit  # 点击按钮时退出应用
        )
        exit_button.grid(row=5, column=5, padx=20, pady=20, sticky="se")

        # Monitor Tab##################
        mon_tab = self.tabview.tab("Monitor")
        mon_tab.grid_rowconfigure([0,1,2,3,4], weight=1)
        mon_tab.grid_columnconfigure([0,1,2,3,4,5], weight=1)

        self._create_gif_label(mon_tab, 'gif5', 0, 2)
        self._create_gif_label(mon_tab, 'gif6', 0, 3)

        desc = ctk.CTkLabel(
            mon_tab,
            text="图1，dqn打breakout游戏",
            font=FONT_DECOR,
            fg_color="white",  # 深色底，圆角文本框
            corner_radius=8,
            wraplength=300,
            justify="left"
        )
        # 放在 GIF 下面
        desc.grid(row=2, column=2, padx=10, pady=(0, 20))

        desc2 = ctk.CTkLabel(
            mon_tab,
            text="图2，dqn打pong游戏",
            font=FONT_DECOR,
            fg_color="white",  # 深色底，圆角文本框
            corner_radius=8,
            wraplength=300,
            justify="left"
        )
        # 放在 GIF 下面
        desc2.grid(row=2, column=3, padx=10, pady=(0, 20))

        desc_combined2 = ctk.CTkLabel(
            mon_tab,
            text=(
                "1. Model Algorithm: 请先选择您想要推理的算法\n"
                "2. Game Environment: 选择您想要的游戏环境\n"
                "3. Environments: 输入您想要并行运行的环境数量\n"
                "4. Vector Stacks: 帧数堆叠数量，不建议改变\n"
                "5. Rec Time (s):录制时间"
            ),
            font=FONT_DECOR,
            fg_color="white",  # 深色底，圆角文本框
            corner_radius=0,
            wraplength=400,
            justify="left"
        )
        desc_combined2.grid(row=3, column=2, rowspan=2, columnspan=2, padx=10, pady=(0, 20))

        ctk.CTkLabel(mon_tab, text="🎮Model Algorithm:", font=FONT_LABEL).grid(row=0, column=0, sticky="w", padx=20, pady=20)
        self.monitor_algo_var = tk.StringVar(value="DQN")
        ctk.CTkComboBox(mon_tab, values=["DQN","PPO","A2C"], variable=self.monitor_algo_var, width=300).grid(row=0, column=1, padx=20, pady=20)

        ctk.CTkLabel(mon_tab, text="🐣Game Environment:", font=FONT_LABEL).grid(row=1, column=0, sticky="w", padx=20, pady=20)
        self.monitor_env_var = tk.StringVar(value=self.env_options[0])
        ctk.CTkComboBox(mon_tab, values=self.env_options, variable=self.monitor_env_var, width=300).grid(row=1, column=1, padx=20, pady=20)

        self.monitor_env_count_entry = self._add_labeled_entry(mon_tab, "🌐Environments:", "4", 2, 0)
        self.monitor_vectorstacks_entry = self._add_labeled_entry(mon_tab, "🔢Vector Stacks:", "4", 3, 0)
        self.recording_time_entry = self._add_labeled_entry(mon_tab, "⏲️Rec Time (s):", "60", 4, 0)

        button_width = 260

        ctk.CTkButton(mon_tab, text="📁Choose Model File", width=button_width,  command=self.choose_model_file).grid(row=5, column=0, padx=20, pady=30,sticky="ew")
        ctk.CTkButton(mon_tab, text="🎥Monitor Agent", width=button_width,  command=self.start_monitoring_thread).grid(row=5, column=1, padx=20, pady=30,sticky="ew")
        # 添加“退出”按钮
        exit_button = ctk.CTkButton(
            mon_tab,
            text="❌ 退出",
            width=100,
            fg_color="#E53E3E",  # 红色背景
            text_color="white",  # 白色文字
            corner_radius=10,
            command=self.root.quit  # 点击按钮时退出应用
        )
        exit_button.grid(row=5, column=5, padx=20, pady=20, sticky="se")

        ctk.CTkLabel(mon_tab, text="🍿 Enjoy the Show! 🍿", font=FONT_DECOR).grid(row=6, column=0, columnspan=6)


    def _add_labeled_entry(self, parent, label_text, default, row, col):
        ctk.CTkLabel(parent, text=label_text, font=FONT_LABEL).grid(row=row, column=col, sticky="w", padx=20, pady=20)
        entry = ctk.CTkEntry(parent, width=300, corner_radius=8)
        entry.insert(0, default)
        entry.grid(row=row, column=col + 1, padx=20, pady=20)
        return entry

    def choose_save_dir(self):
        self.save_dir = filedialog.askdirectory(title="Select Save Directory")
        if self.save_dir:
            messagebox.showinfo("Directory Selected", f"Save Dir: {self.save_dir}")

    def choose_log_dir(self):
        self.log_dir = filedialog.askdirectory(title="Select Log Directory")
        if self.log_dir:
            messagebox.showinfo("Directory Selected", f"Log Dir: {self.log_dir}")

    def train_model(self):
        algo = self.algo_var.get()
        env_name = self.env_var.get()
        timesteps = int(self.timesteps_entry.get())
        stacks = int(self.vectorstacks_entry.get())
        n_envs = int(self.env_count_entry.get())

        def _train():
            if algo == "DQN":
                train_dqn_model_main(env_name, self.save_dir, self.log_dir, timesteps, stacks, n_envs)
            elif algo == "PPO":
                train_ppo_model_main(env_name, self.save_dir, self.log_dir, timesteps, stacks, n_envs)
            elif algo == "A2C":
                train_a2c_model_main(env_name, self.save_dir, self.log_dir, timesteps, stacks, n_envs)
            self.root.after(0, lambda: messagebox.showinfo("Training Complete", f"{algo} finished."))
        threading.Thread(target=_train, daemon=True).start()

    def run_tensorboard(self):
        if hasattr(self, 'log_dir'):
            threading.Thread(target=self._run_tensorboard, daemon=True).start()
        else:
            messagebox.showwarning("No Log Dir", "Please select log directory first.")

    def _run_tensorboard(self):
        try:
            self.tensorboard_port += 1
            subprocess.Popen(["tensorboard", "--logdir", self.log_dir, "--port", str(self.tensorboard_port)])
            url = f"http://localhost:{self.tensorboard_port}"
            self.tb_info.configure(text=f"TensorBoard Running at: {url}")
            webbrowser.open(url)
        except Exception as e:
            self.tb_info.configure(text=f"TB Error: {e}")

    def choose_model_file(self):
        self.model_file = filedialog.askopenfilename(title="Select Model File")
        if self.model_file:
            messagebox.showinfo("Model File", f"{self.model_file}")

    def start_monitoring_thread(self):
        threading.Thread(target=self.monitor_agent, daemon=True).start()

    def monitor_agent(self):
        env_name = self.monitor_env_var.get()
        stacks = int(self.monitor_vectorstacks_entry.get())
        n_envs = int(self.monitor_env_count_entry.get())
        rec_time = int(self.recording_time_entry.get())
        algo = self.monitor_algo_var.get()
        if not hasattr(self, 'model_file'):
            messagebox.showwarning("No Model File", "Select model file first.")
            return
        self.env = create_env(environment_name=env_name, n_envs=n_envs, n_stack=stacks)
        if algo == "DQN":
            self.model = load_dqn_model(env=self.env, model_path=self.model_file)
        elif algo == "PPO":
            self.model = load_ppo_model(env=self.env, model_path=self.model_file)
        elif algo == "A2C":
            self.model = load_a2c_model(env=self.env, model_path=self.model_file)
        video_path = filedialog.asksaveasfilename(defaultextension=".avi", filetypes=[("AVI files","*.avi")])
        if video_path:
            record_env(self.env, self.model, video_path, recording_time=rec_time)
            messagebox.showinfo("Recording Saved", f"Video: {video_path}")

if __name__ == "__main__":
    root = ctk.CTk()
    app = BreakoutAIApp(root)
    root.mainloop()
