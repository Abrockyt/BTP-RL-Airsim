import re

with open('iot_projet_gui.py', 'r', encoding='utf-8') as f:
    text = f.read()

brain_code = """
# =============================================================================
# BRAIN ARCHITECTURE
# =============================================================================
import torch
import torch.nn as nn

class MHA_Actor(nn.Module):
    def __init__(self, state_dim, action_dim, num_heads=4, embed_dim=64):
        super(MHA_Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.input_projection = nn.Linear(state_dim, embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.fc1 = nn.Linear(embed_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.mean_layer = nn.Linear(64, action_dim)
        self.log_std_layer = nn.Linear(64, action_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def forward(self, state):
        x = self.input_projection(state)
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 2:
            x = x.unsqueeze(1)
        attn_output, _ = self.mha(x, x, x)
        x = attn_output.squeeze(1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        mean = self.tanh(self.mean_layer(x))
        return mean

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
    
    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        value = self.fc3(x)
        return value

class PPO_Agent:
    def __init__(self, state_dim, action_dim, lr=0.0003, gamma=0.99, K_epochs=4, eps_clip=0.2):
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.lr = lr
        
        self.actor = MHA_Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.old_actor = MHA_Actor(state_dim, action_dim)
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.dones = []
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.old_actor.to(self.device)
        self.MseLoss = nn.MSELoss()
    
    def select_action(self, state, training=False):
        state_tensor = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            action = self.actor(state_tensor)
        action_np = action.cpu().numpy()
        if len(action_np.shape) > 1:
            action_np = action_np.flatten()
        return action_np

"""

if 'class PPO_Agent' not in text:
    pattern = re.compile(r'# =============================================================================\s*# CONFIGURATION', re.MULTILINE)
    text = pattern.sub(brain_code + '\n# =============================================================================\n# CONFIGURATION', text)
    with open('iot_projet_gui.py', 'w', encoding='utf-8') as f:
        f.write(text)
    print('Restored Brain Architecture via regex.')
else:
    print('Already present.')
