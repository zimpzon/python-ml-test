import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
from data_reader import read_board_states


class BoardModel(nn.Module):
    def __init__(self):
        super(BoardModel, self).__init__()

        layer_size = 8

        self.relu = nn.ReLU()

        self.layer0 = nn.Linear(5 * 5 * 9, layer_size)
        self.bn0 = nn.BatchNorm1d(layer_size)

        self.layer1 = nn.Linear(layer_size, layer_size)
        self.bn1 = nn.BatchNorm1d(layer_size)

        self.out = nn.Linear(layer_size, 5 * 5 * 8)

    def forward(self, x):
        x = self.relu(self.bn0(self.layer0(x)))
        x = self.relu(self.bn1(self.layer1(x)))
        x = self.out(x)
        return x


states = read_board_states('c:/temp/ml/gen-0.json')

all_states = list(map(lambda s: s.get('State'), states))
all_states = [np.array(array_data).reshape((9 * 5 * 5))
              for array_data in all_states]

expected_move = list(map(lambda s: s.get('SelectedMove'), states))
expected_move = [np.array(array_data).reshape((8 * 5 * 5))
                 for array_data in expected_move]

expected_value = list(map(lambda s: s.get('Value'), states))
expected_value = [np.array(array_data).reshape((1))
                  for array_data in expected_value]

# Split data into training and test sets
x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(
    all_states, expected_move, expected_value, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
all_states = torch.tensor(all_states).float()
expected_move = torch.tensor(expected_move).float()
expected_value = torch.tensor(expected_value).float()

# Create TensorDataset and DataLoader
train_data, test_data, train_move, test_move, train_value, test_value = train_test_split(
    all_states, expected_move, expected_value, test_size=0.2, random_state=42)

train_dataset = TensorDataset(train_data, train_move, train_value)
test_dataset = TensorDataset(test_data, test_move, test_value)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize network, criterion, and optimizer
model = BoardModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 1000
learning_rate = 0.001
step_size = 100  # Decay the learning rate every x steps (or epochs)
gamma = 0.8  # Decay factor
batch_size = 1000

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for state, move, value in train_loader:
        optimizer.zero_grad()
        predicted_move = model(state)
        loss = criterion(predicted_move, move)
        # loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

# Testing loop
model.eval()
with torch.no_grad():
    total_loss = 0.0
    for state, move, value in test_loader:
        predicted_move = model(state)
        loss = criterion(predicted_move, move)
        total_loss += loss.item()
    print(f"Test Loss: {total_loss / len(test_loader)}")
