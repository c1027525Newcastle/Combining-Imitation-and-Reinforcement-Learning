import numpy as np
import torch
import torch.nn.functional as F

from f1tenth.Utils.Networks import DoublePolicyNet, DoubleQNet
from f1tenth.Utils.ReplayBuffers import OffPolicyBuffer
from f1tenth.Utils.utils import soft_update

# hyper parameters
BATCH_SIZE = 100
GAMMA = 0.99
tau = 0.005
# NOISE = 0.2
NOISE_CLIP = 0.5
EXPLORE_NOISE = 0.1
POLICY_FREQUENCY = 2
POLICY_NOISE = 0.2


class TrainBCTD3:
    def __init__(self, state_dim, action_dim, exp_n):
        self.act_dim = action_dim
        
        self.actor = DoublePolicyNet(state_dim, action_dim)
        self.actor_target = DoublePolicyNet(state_dim, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)

        self.critic_1 = DoubleQNet(state_dim, action_dim)
        self.critic_target_1 = DoubleQNet(state_dim, action_dim)
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_2 = DoubleQNet(state_dim, action_dim)
        self.critic_target_2 = DoubleQNet(state_dim, action_dim)
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())
        self.critic_optimizer = torch.optim.Adam(list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=1e-3)

        self.replay_buffer = OffPolicyBuffer(state_dim, action_dim)

        # Initialize the BC model
        self.imitation_model = torch.load(f'Data/Experiment_{exp_n}/AgentOff_BC_Game_gbr_Cth_8_{exp_n}_1/AgentOff_BC_Game_gbr_Cth_8_{exp_n}_1_actor.pth')
        
        # Choose the percentage of times it affects TD3 during training
        self.imitation_probability = 0.05

    def act(self, state, noise=EXPLORE_NOISE):
        state = torch.FloatTensor(state.reshape(1, -1))
        
        # Choose between acting with BC or TD3 model
        if np.random.rand() < self.imitation_probability:
            with torch.no_grad():
                action = self.imitation_model(state).data.numpy().flatten()
        else:
            action = self.actor(state).data.numpy().flatten()
        
        if noise != 0: 
            action = (action + np.random.normal(0, noise, size=self.act_dim))
            
        return action.clip(-1, 1)

    def train(self, iterations=2):
        if self.replay_buffer.size() < BATCH_SIZE:
            return
        for it in range(iterations):
            state, action, next_state, reward, done = self.replay_buffer.sample(BATCH_SIZE)
            self.update_critic(state, action, next_state, reward, done)
        
            if it % POLICY_FREQUENCY == 0:
                self.update_policy(state)
                
                soft_update(self.critic_1, self.critic_target_1, tau)
                soft_update(self.critic_2, self.critic_target_2, tau)
                soft_update(self.actor, self.actor_target, tau)
    
    def update_critic(self, state, action, next_state, reward, done):
        noise = torch.normal(torch.zeros(action.size()), POLICY_NOISE)
        noise = noise.clamp(-NOISE_CLIP, NOISE_CLIP)
        next_action = (self.actor_target(next_state) + noise).clamp(-1, 1)

        target_Q1 = self.critic_target_1(next_state, next_action)
        target_Q2 = self.critic_target_2(next_state, next_action)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward + (done * GAMMA * target_Q).detach()

        current_Q1 = self.critic_1(state, action)
        current_Q2 = self.critic_2(state, action)

        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) 

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_policy(self, state):
        actor_loss = -self.critic_1(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
                

    def save(self, filename, directory):
        torch.save(self.actor, directory + f'{filename}_actor.pth')

    


class TestBCTD3:
    def __init__(self, filename, directory):
        self.actor = torch.load(directory + f'{filename}_actor.pth')

    def act(self, state):
        state = torch.FloatTensor(state.reshape(1, -1))
        action = self.actor(state).data.numpy().flatten()
        
        return action

        