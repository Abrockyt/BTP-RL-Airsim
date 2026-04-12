with open('iot_projet_gui.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Fix the empty if blocks
text = text.replace("        if self.best_energy < float('inf'):\n\n        if self.best_time < float('inf'):", "        if self.best_energy < float('inf'):\n            pass\n\n        if self.best_time < float('inf'):")
text = text.replace("        if self.best_time < float('inf'):\n\n", "        if self.best_time < float('inf'):\n            pass\n\n")

with open('iot_projet_gui.py', 'w', encoding='utf-8') as f:
    f.write(text)
