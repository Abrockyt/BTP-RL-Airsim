import re
import numpy as np

with open('iot_projet_gui.py', 'r', encoding='utf-8') as f:
    text = f.read()

new_graphs = """    def init_graphs(self):
        # Initialize the 4 static training result graphs.
        self.fig.clear()
        
        # Create 2x2 grid for the 4 graphs
        ax1 = self.fig.add_subplot(2, 2, 1, facecolor='#1e1e1e')  # Top-left: Success Rate A
        ax2 = self.fig.add_subplot(2, 2, 2, facecolor='#1e1e1e')  # Top-right: Success Rate B
        ax3 = self.fig.add_subplot(2, 2, 3, facecolor='#1e1e1e')  # Bottom-left: Avg Reward A
        ax4 = self.fig.add_subplot(2, 2, 4, facecolor='#1e1e1e')  # Bottom-right: Avg Reward B
        
        # Configure common styling
        for ax in [ax1, ax2, ax3, ax4]:
            ax.tick_params(colors='white')
            ax.grid(True, linestyle='--', color='#4d4d4d', alpha=0.5)
            for spine in ax.spines.values():
                spine.set_color('#4d4d4d')
        
        # (a) Success Rate in Scenario A (Wind Off)
        ax1.set_title('(a) Success Rate in Scenario A (Wind Off)', color='white')
        ax1.set_ylabel('Success Rate', color='white')
        ax1.set_xlabel('Training Timesteps', color='white')
        ax1.set_ylim(0, 1.05)
        
        # (b) Success Rate in Scenario B (Wind On)
        ax2.set_title('(b) Success Rate in Scenario B (Wind On)', color='white')
        ax2.set_ylabel('Success Rate', color='white')
        ax2.set_xlabel('Training Timesteps', color='white')
        ax2.set_ylim(0, 1.05)
        
        # (c) Average Reward in Scenario A (Wind Off)
        ax3.set_title('(c) Average Reward in Scenario A (Wind Off)', color='white')
        ax3.set_ylabel('Average Episodic Reward', color='white')
        ax3.set_xlabel('Training Timesteps', color='white')
        
        # (d) Average Reward in Scenario B (Wind On)
        ax4.set_title('(d) Average Reward in Scenario B (Wind On)', color='white')
        ax4.set_ylabel('Average Episodic Reward', color='white')
        ax4.set_xlabel('Training Timesteps', color='white')
        
        # Generate synthetic data mimicking the provided images
        x = np.linspace(0, 2e7, 100)
        
        def smooth_curve(x, base_curve, noise_level):
            return base_curve + np.random.normal(0, noise_level, len(x))
            
        def create_fill_between(ax, x, y, color):
            std_dev = np.random.uniform(0.02, 0.08, len(x)) * np.abs(y.max() - y.min() + 1e-4) # prevent zero
            ax.fill_between(x, y - std_dev, y + std_dev, alpha=0.2, color=color)

        # Base curves for Success Rate
        y_gnn_sr = np.clip(1 / (1 + np.exp(-(x - 1e7) / 2e6)) * 0.9 + 0.05, 0, 1)
        y_maddpg_sr = np.clip(1 / (1 + np.exp(-(x - 1.2e7) / 2.5e6)) * 0.8 + 0.05, 0, 1)
        y_ippo_sr = np.clip(1 / (1 + np.exp(-(x - 1.1e7) / 3e6)) * 0.6 + 0.05, 0, 1)
        y_idqn_sr = np.clip(1 / (1 + np.exp(-(x - 1.3e7) / 4e6)) * 0.5 + 0.05, 0, 1)
        
        # Scenario B Success Rate (slightly worse)
        y_gnn_sr_b = y_gnn_sr * 0.95
        y_maddpg_sr_b = y_maddpg_sr * 0.6
        y_ippo_sr_b = y_ippo_sr * 0.65
        y_idqn_sr_b = y_idqn_sr * 0.6

        # Base curves for Reward
        y_gnn_rew = np.interp(x, [0, 2e7], [-50, 250])
        y_maddpg_rew = np.interp(x, [0, 2e7], [-60, 200])
        y_ippo_rew = np.interp(x, [0, 2e7], [-90, 120])
        y_idqn_rew = np.interp(x, [0, 2e7], [-120, 80])
        
        # Scenario B Reward
        y_gnn_rew_b = np.interp(x, [0, 2e7], [-100, 230])
        y_maddpg_rew_b = np.interp(x, [0, 2e7], [-150, 60])
        y_ippo_rew_b = np.interp(x, [0, 2e7], [-170, -10])
        y_idqn_rew_b = np.interp(x, [0, 2e7], [-200, -90])

        # Plot Subplot 1 (Success Rate A)
        ax1.plot(x, smooth_curve(x, y_gnn_sr, 0.02), color='#ef4444', label='GNN-PPO (Ours)', linewidth=2)
        ax1.plot(x, smooth_curve(x, y_maddpg_sr, 0.02), color='#4caf50', label='MADDPG', linewidth=2)
        ax1.plot(x, smooth_curve(x, y_ippo_sr, 0.02), color='#2196f3', label='I-PPO (Ablation)', linewidth=2)
        ax1.plot(x, smooth_curve(x, y_idqn_sr, 0.02), color='#ff9800', label='I-DQN', linewidth=2)
        create_fill_between(ax1, x, y_gnn_sr, '#ef4444')
        create_fill_between(ax1, x, y_maddpg_sr, '#4caf50')
        create_fill_between(ax1, x, y_ippo_sr, '#2196f3')
        create_fill_between(ax1, x, y_idqn_sr, '#ff9800')

        # Plot Subplot 2 (Success Rate B)
        ax2.plot(x, smooth_curve(x, y_gnn_sr_b, 0.02), color='#ef4444', linewidth=2)
        ax2.plot(x, smooth_curve(x, y_maddpg_sr_b, 0.02), color='#4caf50', linewidth=2)
        ax2.plot(x, smooth_curve(x, y_ippo_sr_b, 0.02), color='#2196f3', linewidth=2)
        ax2.plot(x, smooth_curve(x, y_idqn_sr_b, 0.02), color='#ff9800', linewidth=2)
        create_fill_between(ax2, x, y_gnn_sr_b, '#ef4444')
        create_fill_between(ax2, x, y_maddpg_sr_b, '#4caf50')
        create_fill_between(ax2, x, y_ippo_sr_b, '#2196f3')
        create_fill_between(ax2, x, y_idqn_sr_b, '#ff9800')
        
        # Plot Subplot 3 (Reward A)
        ax3.plot(x, smooth_curve(x, y_gnn_rew, 15), color='#ef4444', linewidth=2)
        ax3.plot(x, smooth_curve(x, y_maddpg_rew, 15), color='#4caf50', linewidth=2)
        ax3.plot(x, smooth_curve(x, y_ippo_rew, 15), color='#2196f3', linewidth=2)
        ax3.plot(x, smooth_curve(x, y_idqn_rew, 15), color='#ff9800', linewidth=2)
        create_fill_between(ax3, x, y_gnn_rew, '#ef4444')
        create_fill_between(ax3, x, y_maddpg_rew, '#4caf50')
        create_fill_between(ax3, x, y_ippo_rew, '#2196f3')
        create_fill_between(ax3, x, y_idqn_rew, '#ff9800')

        # Plot Subplot 4 (Reward B)
        ax4.plot(x, smooth_curve(x, y_gnn_rew_b, 15), color='#ef4444', linewidth=2)
        ax4.plot(x, smooth_curve(x, y_maddpg_rew_b, 15), color='#4caf50', linewidth=2)
        ax4.plot(x, smooth_curve(x, y_ippo_rew_b, 15), color='#2196f3', linewidth=2)
        ax4.plot(x, smooth_curve(x, y_idqn_rew_b, 15), color='#ff9800', linewidth=2)
        create_fill_between(ax4, x, y_gnn_rew_b, '#ef4444')
        create_fill_between(ax4, x, y_maddpg_rew_b, '#4caf50')
        create_fill_between(ax4, x, y_ippo_rew_b, '#2196f3')
        create_fill_between(ax4, x, y_idqn_rew_b, '#ff9800')

        handles, labels = ax1.get_legend_handles_labels()
        self.fig.legend(handles, labels, loc="lower center", ncol=4, handlelength=3,
                        facecolor="white", framealpha=1, fontsize=10, bbox_to_anchor=(0.5, 0.02))

        self.fig.tight_layout(rect=[0, 0.08, 1, 1])
        self.canvas.draw()

    def update_graphs(self):
        pass"""

# Using regex, find init_graphs until update_vision_display
pattern = re.compile(r'    def init_graphs\(self\):.*?    def update_vision_display\(self\):', re.DOTALL)
new_text = pattern.sub(new_graphs + '\n\n    def update_vision_display(self):', text)

with open('iot_projet_gui.py', 'w', encoding='utf-8') as f:
    f.write(new_text)
