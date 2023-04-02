import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from torch.nn import functional as F
from data_reader import read_board_states


class BoardGameModel(nn.Module):
    def __init__(self):
        super(BoardGameModel, self).__init__()

        self.board_size = 5
        self.channels = 9  # 8 pieces + 1 player turn
        self.num_actions = 8 * self.board_size * self.board_size

        # Input size: [batch_size, 9, 5, 5] (8 planes for different pieces + 1 plane for player turn)
        # Output size is 8 channels/possible moves per cell on the board (8 * 5 * 5)
        self.conv1 = nn.Conv2d(self.channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * self.board_size * self.board_size, 256)

        # Policy head
        self.policy_fc = nn.Linear(256, self.num_actions)

        # Value head
        self.value_fc1 = nn.Linear(256, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))

        # Policy
        policy = torch.softmax(self.policy_fc(x), dim=1)

        # Value
        value = torch.relu(self.value_fc1(x))
        value = torch.tanh(self.value_fc2(value))

        return policy, value

def visualize_results(losses, accuracies, test_losses, loss_ax):
    loss_ax.clear()
    loss_ax.set_ylim(0, 3)
    loss_ax.plot(losses, label='Loss', color='blue')
    loss_ax.plot(accuracies, label='Test Loss', color='yellow')
    loss_ax.plot(test_losses, label='Loss', color='red')
    loss_ax.set_xlabel('Iteration')
    loss_ax.set_ylabel('Value')
    loss_ax.legend()


if __name__ == "__main__":
    states = read_board_states('c:/temp/ml/gen1.json')

    all_states = list(map(lambda s: s.get('State'), states))
    all_states = [np.array(array_data).reshape((9, 5, 5))
                  for array_data in all_states]
    all_states = np.stack(all_states, axis=0)

    # Get class labels instead of one-hot vectors
    expected_policies = list(map(lambda s: np.argmax(
        s.get('SelectedDirection')), states))

    expected_values = list(map(lambda s: np.argmax(
        s.get('ValueSelectedDirection')), states))
    expected_values = np.array(expected_values).reshape((-1, 1))

    # Split data into training and test sets
    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(
        all_states, expected_policies, expected_values, test_size=0.2, random_state=42)

    # Convert training and test data to PyTorch tensors
    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)
    z_train = torch.tensor(z_train, dtype=torch.float32)
    z_test = torch.tensor(z_test, dtype=torch.float32)

    epochs = 1000
    learning_rate = 0.001
    step_size = 100  # Decay the learning rate every x steps (or epochs)
    gamma = 0.8  # Decay factor
    batch_size = 1000  # MUST BE DIVISOR OF SAMPLES

    model = BoardGameModel()

    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    losses = []
    test_losses = []
    accuracies = []
    fig, (loss_ax, accuracy_ax) = plt.subplots(1, 2, figsize=(10, 5))
    plt.ion()
    plt.show()

    print(f'Training samples: {len(x_train)}')
    print(f'Test samples: {len(x_test)}')

total_batches = 0
for epoch in range(epochs):
    epoch_loss = 0
    batch_count = 0
    model.train()
    for i in range(0, len(x_train), batch_size):
        optimizer.zero_grad()

        # batch_size * 9 * 5 * 5
        x_batch = x_train[i:i+batch_size]

        # batch_size * 200 for policy, batch_size * 1 for value
        y_pred, z_pred = model(x_batch)

        y_batch = y_train[i:i+batch_size]
        z_batch = z_train[i:i+batch_size]

        policy_loss = policy_loss_fn(y_pred, y_batch)
        value_loss = value_loss_fn(z_pred, z_batch)

        alpha = 0.5  # You can tune this weight based on the importance of each head during training
        combined_loss = alpha * policy_loss + (1 - alpha) * value_loss
        policy_loss.backward()

        optimizer.step()
        epoch_loss += policy_loss.item()
        batch_count += 1
        total_batches += 1

    epoch_loss /= batch_count
    losses.append(epoch_loss)

    with torch.no_grad():
        model.eval()
        y_pred, z_pred = model(x_test)

        policy_loss = policy_loss_fn(y_pred, y_test)
        value_loss = value_loss_fn(z_pred, z_test)

        alpha = 0.5  # You can tune this weight based on the importance of each head during training
        combined_loss = alpha * policy_loss + (1 - alpha) * value_loss

    test_losses.append(policy_loss.item())

    _, max_indices = torch.max(y_pred, dim=1)

    # max_indices now contains the indices of the maximum values for each element in y_pred
    correct_predictions = (max_indices == y_test).sum().item()
    accuracy = correct_predictions / len(y_test)
    accuracies.append(accuracy)

    scheduler.step()

    visualize_results(losses, accuracies, test_losses, loss_ax)
    fig.canvas.draw()
    fig.canvas.flush_events()

    print(
        f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.8f}, TestLoss: {policy_loss:.8f}, Learning rate: {scheduler.get_last_lr()[0]:.6f}, total_batches: {total_batches}")

    # print(
    #     f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.8f}, TestLoss: {combined_loss:.8f}, Learning rate: {scheduler.get_last_lr()[0]:.6f}, Accuracy: {accuracy:.6f} ({correct_predictions}/{len(y_test)}), total_batches: {total_batches}")
# Save the model if desired
# torch.save(model.state_dict(), 'model.pth')

plt.ioff()
plt.show()
