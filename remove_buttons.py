import re

file_path = "iot_projet_gui.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Remove the entire reinforcement learning frame block
# We match from the `# Reinforcement Learning Statistics` comment
# all the way down to just before the `# Control buttons` section.

# Wait, the comment might be different in this file, let's just make it broad
rl_pattern = re.compile(
    r"[ \t]*# Reinforcement Learning Statistics.*?# Control buttons.*?\n",
    re.DOTALL
)
# Let's write a targeted function to remove those elements based on my knowledge of the file.
