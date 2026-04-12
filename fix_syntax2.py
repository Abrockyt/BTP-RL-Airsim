import re

with open('iot_projet_gui.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
expecting_block = False
indent = ""

for i, line in enumerate(lines):
    if expecting_block:
        if line.strip() == "" or (len(line) - len(line.lstrip()) <= len(indent)):
            # We expected a block but got a blank line or a de-dented line, insert pass
            new_lines.append(indent + "    pass\n")
        expecting_block = False
        
    new_lines.append(line)
    
    if line.strip().endswith(':') and not line.strip().startswith('#'):
        expecting_block = True
        indent = line[:len(line) - len(line.lstrip())]

with open('iot_projet_gui.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
