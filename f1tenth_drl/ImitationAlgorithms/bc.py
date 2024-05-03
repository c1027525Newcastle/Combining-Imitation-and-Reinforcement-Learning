import torch
import numpy as np
from pathlib import Path
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
from f1tenth_drl.Utils.Networks import DoublePolicyNet
import time
import os


class TrainBC:
    def __init__(self, history_files, actions_files):
        self.scans = np.concatenate([np.load(file) for file in history_files], axis=0)
        self.actions = np.concatenate([np.load(file) for file in actions_files], axis=0)
        self.reintialize()

    def reintialize(self):
        self.model = DoublePolicyNet(self.scans.shape[1], self.actions.shape[1])
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0012) #0.005 #0.001
        

    def train(self):
        epochs=3000 #6000
        batch_size=64 #64

        history_tensor = torch.tensor(self.scans, dtype=torch.float32)
        actions_tensor = torch.tensor(self.actions, dtype=torch.float32)

        print('Training model')

        train_algorithm = True
        while train_algorithm:
            # Store best reuslts and moment they happened
            best_loss = float('inf')
            best_epoch = -1

            # Loop for training
            for epoch in range(epochs):
                permutation = torch.randperm(history_tensor.size()[0])
                for i in range(0, history_tensor.size()[0], batch_size):
                    indices = permutation[i:i+batch_size]
                    batch_scans, batch_actions = history_tensor[indices], actions_tensor[indices]

                    self.optimizer.zero_grad()
                    outputs = self.model(batch_scans)
                    loss = self.criterion(outputs, batch_actions)
                    loss.backward()
                    self.optimizer.step()
                
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_epoch = epoch
                    
                
                if (epoch % 500 == 0 and epoch != 0) or epoch == 1 or (epoch == epochs -1):
                    if epoch == epochs -1:
                        print(f'Last Epoch {epoch}, Loss: {loss.item()}\n')
                    else:
                        print(f'Epoch {epoch}, Loss: {loss.item()}')

            # Comment out the following line to enable the check for overfitting
            train_algorithm = False
            # Check if the model is overfitting and if so, restart training
            if best_epoch <= 0.75 * epochs and train_algorithm == True:
                print(f"Best loss was {best_loss} at epoch {best_epoch} out of {epochs} epochs")
                epochs = best_epoch
                self.reintialize()
                print(f"Training will restart for {epochs} epochs\n")
            else:
                train_algorithm = False

        # Print best loss and epoch it happend for tuning purposes
        print(f"Best loss was {best_loss} at epoch {best_epoch}")
        return self.model


    # Function to save the model not just dictionary
    def save(self, filename):
        save_path = Path(filename)
        torch.save(self.model, save_path)
        # Print where the model was saved
        print(f"\nModel saved to {save_path}")


class TestBC:
    def __init__(self):
        #self.actor = torch.load('Data/Experiment_1/AgentOff_BC_Game_gbr_Cth_8_1_1/AgentOff_BC_Game_gbr_Cth_8_1_1_actor.pth') # Experiment 1
        #self.actor = torch.load('Data/Experiment_2/AgentOff_BC_Game_gbr_Cth_8_2_1/AgentOff_BC_Game_gbr_Cth_8_2_1_actor.pth') # Experiment 2
        self.actor = torch.load('Data/Experiment_3/AgentOff_BC_Game_gbr_Cth_8_3_1/AgentOff_BC_Game_gbr_Cth_8_3_1_actor.pth') # Experiment 3


    def act(self, state):
        state = torch.FloatTensor(state.reshape(1, -1))
        action = self.actor(state).data.numpy().flatten()
        
        return action


if __name__ == "__main__":
    history_file = []
    actions_file = []

    # Choose number of laps to train on
    num_laps = 3#3

    # Choose which maps to train on # All maps: ['mco', 'aut', 'gbr', 'esp']
    map_list = ['gbr']
    try:
        for map_name in map_list:
            for lap_num in range(0, num_laps):
                if os.path.exists(f'Data/GenerateDataSet_1/RawData/PurePursuit_{map_name}_DataGen_1_Lap_{lap_num}_history.npy'):
                    # Code to reduce the size of history and actions files so that we can train faster on the beggining of the laps
                    # history = np.load(f'Data/GenerateDataSet_1/RawData/PurePursuit_{map_name}_DataGen_1_Lap_{lap_num}_history.npy')
                    # actions = np.load(f'Data/GenerateDataSet_1/RawData/PurePursuit_{map_name}_DataGen_1_Lap_{lap_num}_actions.npy')
                    # cutt_off = int(len(history) * 0.8)
                    # reduced_history = history[:cutt_off]
                    # reduced_actions = actions[:cutt_off]
                    # np.save(f'Data/GenerateDataSet_1/RawData/PurePursuit_{map_name}_DataGen_1_Lap_{lap_num}_history.npy', reduced_history)
                    # np.save(f'Data/GenerateDataSet_1/RawData/PurePursuit_{map_name}_DataGen_1_Lap_{lap_num}_actions.npy' ,reduced_actions)

                    history_file.append(f'Data/GenerateDataSet_1/RawData/PurePursuit_{map_name}_DataGen_1_Lap_{lap_num}_history.npy')
                    actions_file.append(f'Data/GenerateDataSet_1/RawData/PurePursuit_{map_name}_DataGen_1_Lap_{lap_num}_actions.npy')

                    # Code to check shapes of history and actions
                    # actions = np.load(f'Data/GenerateDataSet_1/RawData/PurePursuit_{map_name}_DataGen_1_Lap_{lap_num}_actions.npy')
                    # history = np.load(f'Data/GenerateDataSet_1/RawData/PurePursuit_{map_name}_DataGen_1_Lap_{lap_num}_history.npy')
                    # print(f'History shape: {history.shape}')
                    # print(f'Actions shape: {actions.shape}')
                else:
                    break
    except FileNotFoundError:
        print('Files for database not found')

    time_start = time.time()

    agent = TrainBC(history_file, actions_file)
    agent.train()
    # Experiment 1
    #agent.save(f'Data/Experiment_1/AgentOff_BC_Game_{map_list[0]}_Cth_8_1_1/AgentOff_BC_Game_{map_list[0]}_Cth_8_1_1_actor.pth')

    # Experiment 2
    agent.save(f'Data/Experiment_2/AgentOff_BC_Game_gbr_Cth_8_2_1/AgentOff_BC_Game_gbr_Cth_8_2_1_actor.pth')

    # Experiment 3
    #agent.save(f'Data/Experiment_3/AgentOff_BC_Game_gbr_Cth_8_3_1/AgentOff_BC_Game_gbr_Cth_8_3_1_actor.pth')

    time_end = time.time()
    print(f"\nTraining took {((time_end - time_start)/60):.2f} minutes")

    # Code to check shapes of history and actions files
    # actions = np.load('Data/GenerateDataSet_1/RawData/PurePursuit_gbr_DataGen_1_Lap_0_actions.npy')
    # history = np.load('Data/GenerateDataSet_1/RawData/PurePursuit_gbr_DataGen_1_Lap_0_history.npy')
    # print(f'History shape: {history.shape}')
    # print(f'Actions shape: {actions.shape}')

    # print("\nActions Statistics:")
    # print(f"Mean: {np.mean(actions, axis=0)}")
    # print(f"Std Deviation: {np.std(actions, axis=0)}")
    # print(f"Min: {np.min(actions, axis=0)}")
    # print(f"Max: {np.max(actions, axis=0)}")

    # # Descriptive statistics for history (state features)
    # print("\nHistory (State Features) Statistics:")
    # print(f"Mean: {np.mean(history, axis=0)}")
    # print(f"Std Deviation: {np.std(history, axis=0)}")
    # print(f"Min: {np.min(history, axis=0)}")
    # print(f"Max: {np.max(history, axis=0)}")
    
