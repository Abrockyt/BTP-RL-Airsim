import re

with open('iot_projet_gui.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Replace best_energy_label block
text = re.sub(r'        if self\.best_energy < float\(\'inf\'\):\n            self\.best_energy_label\.config\(.*?\)', '        pass', text)

# Replace rl_policy_label block
rl_block = r"""        # Show practical RL status in GUI
        if self\.rl_policy_active:
            self\.rl_policy_label\.config\(text=f"Policy: ACTIVE \(\{self\.last_policy_mode\}\)", fg="#76ff03"\)
            self\.rl_action_label\.config\(text=f"Action: \[{self\.last_rl_action\[0\]:\.2f}, {self\.last_rl_action\[1\]:\.2f}, {self\.last_rl_action\[2\]:\.2f}\]"\)
            if self\.last_policy_mode == "ONLINE":
                self\.rl_reward_label\.config\(text=f"Last Reward: \{self\.rewards\[-1\] if len\(self\.rewards\)>0 else 0:\.1f\}"\)
            else:
                self\.rl_reward_label\.config\(text="Last Reward: N/A \(Inference\)"\)
        else:
            self\.rl_policy_label\.config\(text="Policy: Standby", fg="#90caf9"\)
            self\.rl_action_label\.config\(text="Action: \[0\.00, 0\.00, 0\.00\]"\)
            self\.rl_reward_label\.config\(text="Last Reward: 0\.0"\)"""

text = re.sub(rl_block, '        pass', text)

with open('iot_projet_gui.py', 'w', encoding='utf-8') as f:
    f.write(text)
