import torch
import numpy as np
from pathlib import Path
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
from f1tenth_drl.Utils.Networks import DoublePolicyNet

#from f1tenth_drl.f1tenth_gym import F110Env ##
#from f1tenth_drl.Utils.utils import * ##
#from f1tenth_drl.run_experiments import * ##

EXPLORE_NOISE = 0.1


class BCNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(BCNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),  # input_size should be 60
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.fc(x)
    

class TrainBC:
    def __init__(self, scans_file, actions_file):
        self.scans = np.load(scans_file)
        self.actions = np.load(actions_file)
        self.model = BCNetwork(self.scans.shape[1], self.actions.shape[1]) ###TODO: changed to/from BCNetwork/DoublePolicyNet
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        

    def train(self, epochs=100000, batch_size=64):
        scans_tensor = torch.tensor(self.scans, dtype=torch.float32)
        actions_tensor = torch.tensor(self.actions, dtype=torch.float32)

        # Store best reuslts and moment they happened
        best_loss = float('inf')
        best_epoch = -1

        # Loop for training
        for epoch in range(epochs):
            permutation = torch.randperm(scans_tensor.size()[0])
            for i in range(0, scans_tensor.size()[0], batch_size):
                indices = permutation[i:i+batch_size]
                batch_scans, batch_actions = scans_tensor[indices], actions_tensor[indices]

                self.optimizer.zero_grad()
                outputs = self.model(batch_scans)
                loss = self.criterion(outputs, batch_actions)
                loss.backward()
                self.optimizer.step()
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_epoch = epoch
            
            if epoch % 1000 == 0:
                print(f'Epoch {epoch+1}, Loss: {loss.item()}')

        # Print best results and moment it happend
        print(f"\nBest loss was {best_loss:.2f} at epoch {best_epoch}")
        return best_loss, best_epoch

    def act(self, state, noise=EXPLORE_NOISE):
        state = torch.FloatTensor(state.reshape(1, -1))  # Ensure state is reshaped to [1, num_features]
        action = self.model(state).data.numpy().flatten()
        
        if noise != 0: 
            action = (action + np.random.normal(0, noise, size=self.act_dim))
            
        return action.clip(-1, 1)

    # Function to save the model not dictionary
    def save(self, filename):
        save_path = Path(filename)
        torch.save(self.model, save_path)
        # Print where the model was saved
        print(f"\nModel saved to {save_path}")

class TestBC:
    #TODO: Will correctly implement this later
    def __init__(self):#, filename, directory):
        #self.actor = torch.load(directory + f'{filename}_actor.pth') # TODO: changed this
        self.actor = torch.load('Data/Experiment_1/AgentOff_BC_Game_mco_Cth_8_1_1/AgentOff_BC_Game_mco_Cth_8_1_1_actor.pth')
        #self.actor = torch.load(directory + f'{filename}_actor.pth')

    def act(self, state):
        state = torch.FloatTensor(state.reshape(1, -1))
        action = self.actor(state).data.numpy().flatten()
        
        return action


if __name__ == "__main__":
    scans_file = 'Data/GenerateDataSet_1/RawData/PurePursuit_mco_DataGen_1_Lap_0_history.npy'
    actions_file = 'Data/GenerateDataSet_1/RawData/PurePursuit_mco_DataGen_1_Lap_0_actions.npy'
    agent = TrainBC(scans_file, actions_file)
    agent.train()
    agent.save('Data/Experiment_1/AgentOff_BC_Game_mco_Cth_8_1_1/AgentOff_BC_Game_mco_Cth_8_1_1_actor.pth')

    # actions = np.load('Data/GenerateDataSet_1/RawData/PurePursuit_gbr_DataGen_1_Lap_0_actions.npy')
    # history = np.load('Data/GenerateDataSet_1/RawData/PurePursuit_gbr_DataGen_1_Lap_0_history.npy')
    # print(f'History shape: {history.shape}')
    # print(f'Actions shape: {actions.shape}')
    
