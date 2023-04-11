import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
from data_reader import read_board_states

# Network architecture


class BoardModel(nn.Module):
    def __init__(self):
        super(BoardModel, self).__init__()
        self.conv1 = nn.Conv2d(9, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(128, 8, kernel_size=1)
        self.conv4 = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        move = self.conv3(x)
        value = self.conv4(x)
        return move, value.view(-1, 1)


states = read_board_states('c:/temp/ml/gen-0.json')

all_states = list(map(lambda s: s.get('State'), states))
all_states = [np.array(array_data).reshape((9, 5, 5))
              for array_data in all_states]

expected_move = list(map(lambda s: s.get('SelectedMove'), states))
expected_move = [np.array(array_data).reshape((8, 5, 5))
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
criterion1 = nn.MSELoss()
criterion2 = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for state, move, value in train_loader:
        optimizer.zero_grad()
        predicted_move, predicted_value = model(state)
        loss1 = criterion1(predicted_move, move)
        loss2 = criterion2(predicted_value, value)
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

# Testing loop
model.eval()
with torch.no_grad():
    total_loss = 0.0
    for state, move, value in test_loader:
        predicted_move, predicted_value = model(state)
        loss1 = criterion1(predicted_move, move)
        loss2 = criterion2(predicted_value, value)
        loss = loss1 + loss2
        total_loss += loss.item()
    print(f"Test Loss: {total_loss / len(test_loader)}")
