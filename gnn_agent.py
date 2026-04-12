import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

class GNN_Layer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GNN_Layer, self).__init__()
        # Message Function: Processes neighbor data
        self.message_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # Update Function: Combines own data with neighbor messages
        self.update_net = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, node_features, adj_matrix):
        # node_features: [Batch, Num_Drones, State_Dim]
        # adj_matrix: [Batch, Num_Drones, Num_Drones] (1 if connected, 0 if not)
        
        # 1. Calculate Messages for every node
        messages = self.message_net(node_features) # [B, N, Hidden]
        
        # 2. Aggregate Messages (Sum pooling) based on Adjacency
        # We multiply Adjacency * Messages to sum up neighbor info
        aggregated_messages = torch.matmul(adj_matrix, messages) # [B, N, Hidden]
        
        # 3. Concatenate Own Features + Aggregated Neighbor Messages
        combined = torch.cat([node_features, aggregated_messages], dim=2)
        
        # 4. Update Node State
        new_features = self.update_net(combined)
        return new_features

class GNNActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(GNNActor, self).__init__()
        self.embedding = nn.Linear(state_dim, hidden_dim)
        
        # The GNN Layer (Graph Convolution)
        self.gnn = GNN_Layer(hidden_dim, hidden_dim)
        
        # Output Heads
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        self.tanh = nn.Tanh()
        
    def forward(self, state, adj):
        x = torch.relu(self.embedding(state))
        x = self.gnn(x, adj)
        mean = self.tanh(self.mean_layer(x))
        log_std = torch.clamp(self.log_std_layer(x), min=-20, max=2)
        return mean, log_std

class PPO_GNN_Agent:
    def __init__(self, state_dim, action_dim, lr=0.0003, gamma=0.99, K_epochs=4):
        self.gamma = gamma
        self.K_epochs = K_epochs
        
        self.actor = GNNActor(state_dim, action_dim)
        self.optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.buffer = []

    def select_action(self, state, adj):
        # Expects state as [Num_Drones, State_Dim]
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0) # Add batch dim
            adj = torch.FloatTensor(adj).unsqueeze(0)
            
            mean, log_std = self.actor(state, adj)
            dist = Normal(mean, log_std.exp())
            action = dist.sample()
            
        return action.squeeze(0).cpu().numpy()

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename)
        
    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename))