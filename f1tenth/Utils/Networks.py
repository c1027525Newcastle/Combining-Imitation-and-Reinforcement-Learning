
import torch
import torch.nn as nn
import torch.nn.functional as F

NN_LAYER_1 = 100
NN_LAYER_2 = 100


class DoublePolicyNet(nn.Module):
    def __init__(self, state_dim, act_dim):
        super(DoublePolicyNet, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, NN_LAYER_1)
        self.fc2 = nn.Linear(NN_LAYER_1, NN_LAYER_2)
        self.fc_mu = nn.Linear(NN_LAYER_2, act_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x)) 
        return mu
   

class DoubleQNet(nn.Module):
    def __init__(self, state_dim, act_dim):
        super(DoubleQNet, self).__init__()
        
        self.fc1 = nn.Linear(state_dim + act_dim, NN_LAYER_1)
        self.fc2 = nn.Linear(NN_LAYER_1, NN_LAYER_2)
        self.fc_out = nn.Linear(NN_LAYER_2, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x2 = F.relu(self.fc1(x))
        x3 = F.relu(self.fc2(x2))
        q = self.fc_out(x3)
        return q
    