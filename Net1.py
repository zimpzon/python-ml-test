import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR

from data_reader import read_board_states


class Net(nn.Module):
    def __init__(self, layer_size):
        super(Net, self).__init__()
        self.relu = nn.ReLU()

        self.layer1 = nn.Linear(5 * 5 * 8, layer_size)
        self.bn1 = nn.BatchNorm1d(layer_size)
        self.layer2 = nn.Linear(layer_size, layer_size)
        self.bn2 = nn.BatchNorm1d(layer_size)
        # self.layer3 = nn.Linear(layer_size, layer_size)
        # self.bn3 = nn.BatchNorm1d(layer_size)
        # self.layer4 = nn.Linear(layer_size, layer_size)
        self.bn4 = nn.BatchNorm1d(layer_size)
        self.layer5 = nn.Linear(layer_size, 8)  # estimated best direction

    def forward(self, x):
        x = self.relu(self.bn1(self.layer1(x)))
        x = self.relu(self.bn2(self.layer2(x)))
        # x = self.relu(self.bn3(self.layer3(x)))
        # x = self.relu(self.bn4(self.layer4(x)))
        x = self.layer5(x)
        return x


def visualize_results(losses, accuracies, test_losses, loss_ax):
    loss_ax.clear()
    loss_ax.set_ylim(0, 3)
    loss_ax.plot(losses, label='Loss', )
    loss_ax.plot(accuracies, label='Test Loss', color='yellow')
    loss_ax.plot(test_losses, label='Loss', color='red')
    loss_ax.set_xlabel('Iteration')
    loss_ax.set_ylabel('Value')
    loss_ax.legend()

    accuracy_ax.clear()
    accuracy_ax.set_ylim(0, 1)
    accuracy_ax.plot(accuracies, label='Accuracy')
    accuracy_ax.set_xlabel('Iteration')
    accuracy_ax.set_ylabel('Accuracy')
    accuracy_ax.legend()


if __name__ == "__main__":
    states = read_board_states('c:/temp/ml/gen1.json')

    all_states = list(map(lambda s: s.get('State'), states))
    # Get class labels instead of one-hot vectors
    all_expected = list(map(lambda s: np.argmax(
        s.get('DesiredDirections')), states))

    # Split data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        all_states, all_expected, test_size=0.2, random_state=42)

    # Convert training and test data to PyTorch tensors
    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    # Change the data type to long
    y_train = torch.tensor(y_train, dtype=torch.long)
    # Change the data type to long
    y_test = torch.tensor(y_test, dtype=torch.long)

    # TODO: it converges almost instantly. At the same value every time. Why? It means.. data error, mismatch? Try simplified data.
    layer_size = 1000
    epochs = 1000
    learning_rate = 0.001
    step_size = 100  # Decay the learning rate every x steps (or epochs)
    gamma = 0.8  # Decay factor
    batch_size = 100

    model = Net(layer_size)
    # Replace the loss function with CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()
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
    for i in range(0, len(x_train), batch_size):
        optimizer.zero_grad()
        x_batch = x_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        batch_count += 1
        total_batches += 1

    epoch_loss /= batch_count
    losses.append(epoch_loss)

    with torch.no_grad():
        model.eval()
        y_pred = model(x_test)
        test_loss = criterion(y_pred, y_test).item()

    test_losses.append(test_loss)

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
        f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.6f}, TestLoss: {test_loss:.6f},  Learning rate: {scheduler.get_last_lr()[0]:.6f}, Accuracy: {accuracy} ({correct_predictions}/{len(y_test)}), total_batches: {total_batches}")

with torch.no_grad():
    model.eval()
    y_pred = model(x_test)
    test_loss = criterion(y_pred, y_test).item()

print(f"Test loss: {test_loss:.6f}")

# Save the model if desired
# torch.save(model.state_dict(), 'model.pth')

plt.ioff()
plt.show()
