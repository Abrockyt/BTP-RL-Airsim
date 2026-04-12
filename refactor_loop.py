import re

with open('iot_projet_gui.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. We replace the async hack with a permanent while loop inside _flight_loop
pattern_loop = r'(def _flight_loop\(self\):\n\s+"""Main flight loop.*?try:\n)(.*?)(?=\s+except Exception as e:\n\s+print\(f"Fatal Error)'
match = re.search(pattern_loop, content, flags=re.DOTALL)
if match:
    body = match.group(2)
    # Indent the body
    indented_body = "\n".join(["    " + line if line else "" for line in body.split('\n')])
    
    new_loop = match.group(1) + "            while getattr(self, 'running', True):\n" + indented_body
    
    content = content[:match.start()] + new_loop + content[match.end():]
else:
    print("Match failed")

with open('iot_projet_gui.py', 'w', encoding='utf-8') as f:
    f.write(content)
