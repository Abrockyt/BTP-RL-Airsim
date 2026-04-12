import re
with open('iot_projet_gui.py', 'r', encoding='utf-8') as f: content = f.read()
pattern = r'(# Fix IOLoop conflicts by ensuring every generated thread has its own fresh loop AND client\s+import asyncio\s+asyncio\.set_event_loop\(asyncio\.new_event_loop\(\)\)\s+print\("?? Connecting to AirSim in a new event loop\.\.\."\)\s+self\.client = airsim\.MultirotorClient\(\)\s+self\.client\.confirmConnection\(\))'
replacement = '''if getattr(self, "client", None) is None:
                print("?? Connecting to AirSim...")
                self.client = airsim.MultirotorClient()
                self.client.confirmConnection()
            else:
                try:
                    self.client.ping()
                except Exception:
                    self.client = airsim.MultirotorClient()
                    self.client.confirmConnection()'''
content = re.sub(pattern, replacement, content)
with open('iot_projet_gui.py', 'w', encoding='utf-8') as f: f.write(content)
