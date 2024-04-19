import torch
import numpy as np
from pathlib import Path
import torch.nn as nn
from pathlib import Path
import torch.optim as optim

def TrainBC():
    pass

class BCNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(BCNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.fc(x)

def train_behavioral_cloning():
    # Load data
    scans = np.load('Data/GenerateDataSet_1/RawData/PurePursuit_gbr_DataGen_1_Lap_0_scans.npy')
    actions = np.load('Data/GenerateDataSet_1/RawData/PurePursuit_gbr_DataGen_1_Lap_0_history.npy')

    # Convert data to PyTorch tensors
    scans_tensor = torch.tensor(scans, dtype=torch.float32)
    actions_tensor = torch.tensor(actions, dtype=torch.float32)

    # Define model, loss, and optimizer
    input_size = scans.shape[1]  # Assuming scans are already flattened if necessary
    output_size = actions.shape[1]
    model = BCNetwork(input_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training settings
    epochs = 30000
    batch_size = 64

    # Training loop
    for epoch in range(epochs):
        permutation = torch.randperm(scans_tensor.size()[0])
        for i in range(0, scans_tensor.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_scans, batch_actions = scans_tensor[indices], actions_tensor[indices]

            optimizer.zero_grad()
            outputs = model(batch_scans)
            loss = criterion(outputs, batch_actions)
            loss.backward()
            optimizer.step()
        
        if epoch % 1000 == 0:
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    # Save the trained model
    save_path = Path('trained_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    train_behavioral_cloning()