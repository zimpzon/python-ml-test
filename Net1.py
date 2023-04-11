import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.onnx
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from torch.nn import functional as F
from data_reader import read_board_states


class BoardModel(nn.Module):
    def __init__(self):
        super(BoardModel, self).__init__()
        self.conv1 = nn.Conv2d(9, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.move = nn.Linear(128, 200)
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        move = self.move(x)
        value = self.value(x).view(-1, 1)
        return move, value


def visualize_results(losses, accuracies, test_losses, loss_ax):
    loss_ax.clear()
    loss_ax.set_ylim(0, 5)
    loss_ax.plot(losses, label='Loss', color='blue')
    loss_ax.plot(accuracies, label='Test Loss', color='yellow')
    loss_ax.plot(test_losses, label='Loss', color='red')
    loss_ax.set_xlabel('Iteration')
    loss_ax.set_ylabel('Value')
    loss_ax.legend()


if __name__ == "__main__":
    states = read_board_states('c:/temp/ml/gen-0.json')

    all_states = list(map(lambda s: s.get('State'), states))
    all_states = [np.array(array_data).reshape((9, 5, 5))
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
    batch_size = 1000

    model = BoardModel()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    loss_fn = nn.MSELoss()

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
    rang = range(0, len(x_train), batch_size)
    for i in rang:
        step = min(len(x_train) - i, rang.step)
        optimizer.zero_grad()

        x_batch = x_train[i:i+step]
        y_batch = y_train[i:i+step]
        z_batch = z_train[i:i+step]

        random_indices = np.random.choice(
            len(x_batch), size=step, replace=False)

        x_batch = x_batch[random_indices]
        y_batch = y_batch[random_indices]
        z_batch = z_batch[random_indices]

        outputs = model(x_batch)

        loss = loss_fn(outputs, y_batch)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        batch_count += 1
        total_batches += 1

    epoch_loss /= batch_count
    losses.append(epoch_loss)

    with torch.no_grad():
        model.eval()

    outputs = model(x_test)

    loss = loss_fn(outputs, y_test)

    test_losses.append(loss.item())

    # predicted_classes = torch.argmax(outputs, dim=1)
    # # get the index of the highest scoring move along dim=1
    # pred_indices = torch.argmax(output_tensor, dim=1)

    # compare = predicted_classes == y_test
    # sum = compare.sum()
    # correct_predictions = sum.item()

    # accuracy = correct_predictions / len(y_test)
    # accuracies.append(accuracy)

    scheduler.step()

    visualize_results(losses, accuracies, test_losses, loss_ax)
    fig.canvas.draw()
    fig.canvas.flush_events()

    correct_predictions = 0
    accuracy = 0

    print(
        f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.6f}, TestLoss: {loss:.6f}, Learning rate: {scheduler.get_last_lr()[0]:.6f}, Accuracy: {accuracy:.4f} ({correct_predictions}/{len(y_test)}), total_batches: {total_batches}")


# dummy input in correct format is required to save model, that's just how it works.
dummy_input = x_train[0:1]
path = 'c:/temp/ml/tixy.onnx'

torch.onnx.export(model, dummy_input, path,
                  input_names=["input"], output_names=["output"], export_params=True)

plt.ioff()
plt.show()
