with open("c:/Users/abroc/Desktop/UAV_Energy_Sim/iot_projet_gui.py", "r", encoding="utf-8") as f:
    text = f.read()

text = text.replace("marker='o', markersize=3, marker='o', markersize=3", "marker='o', markersize=3")
text = text.replace("marker='s', markersize=3, marker='s', markersize=3", "marker='s', markersize=3")

with open("c:/Users/abroc/Desktop/UAV_Energy_Sim/iot_projet_gui.py", "w", encoding="utf-8") as f:
    f.write(text)
