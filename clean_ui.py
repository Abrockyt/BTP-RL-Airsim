import re

with open('iot_projet_gui.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Remove RL stats frame
text = re.sub(r'        # Reinforcement Learning Statistics.*?        # Control buttons', '        # Control buttons', text, flags=re.DOTALL)

# Remove Comparison Button
text = re.sub(r'        # Comparison Mode Button.*?\n        self\.comparison_btn\.pack\(pady=8\)\n', '', text, flags=re.DOTALL)

# Remove Multi-Drone Button
text = re.sub(r'        # Multi-Drone Environment Button.*?\n        self\.multidrone_btn\.pack\(pady=8\)\n', '', text, flags=re.DOTALL)

# Change title of graphs
text = text.replace('"📈 PERFORMANCE ANALYTICS"', '"📈 TRAINING RESULTS"')

with open('iot_projet_gui.py', 'w', encoding='utf-8') as f:
    f.write(text)
