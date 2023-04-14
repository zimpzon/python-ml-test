import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.onnx
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import keyboard
from torch.optim.lr_scheduler import StepLR
from torch.nn import functional as F
from data_reader import read_board_states


class BoardModel(nn.Module):
    def __init__(self):
        super(BoardModel, self).__init__()

        layer_size = 256

        self.relu = nn.ReLU()

        self.layer0 = nn.Linear(5 * 5 * 9, layer_size)
        self.bn0 = nn.BatchNorm1d(layer_size)

        self.layer1 = nn.Linear(layer_size, layer_size)
        self.bn1 = nn.BatchNorm1d(layer_size)

        self.layer2 = nn.Linear(layer_size, layer_size)
        self.bn2 = nn.BatchNorm1d(layer_size)

        self.layer3 = nn.Linear(layer_size, layer_size)
        self.bn3 = nn.BatchNorm1d(layer_size)

        self.layer4 = nn.Linear(layer_size, layer_size)
        self.bn4 = nn.BatchNorm1d(layer_size)

        self.layer5 = nn.Linear(layer_size, layer_size)
        self.bn5 = nn.BatchNorm1d(layer_size)

        self.out = nn.Linear(layer_size, 5 * 5 * 8)
        self.value = nn.Linear(layer_size, 1)

    def forward(self, x):
        x = self.relu(self.bn0(self.layer0(x)))
        x = self.relu(self.bn1(self.layer1(x)))
        x = self.relu(self.bn2(self.layer2(x)))
        # x = self.relu(self.bn3(self.layer3(x)))
        # x = self.relu(self.bn4(self.layer4(x)))
        # x = self.relu(self.bn5(self.layer5(x)))
        out = self.out(x)
        value = self.value(x)
        return out, value


ylim = 0.1


def visualize_results(losses, accuracies, test_losses, loss_ax):
    global ylim
    if len(losses) > 5:
        ylim = losses[5]

    loss_ax.clear()
    loss_ax.set_ylim(0, ylim)
    loss_ax.plot(losses, label='Loss', color='blue')
    loss_ax.plot(accuracies, label='Test Loss', color='yellow')
    loss_ax.plot(test_losses, label='Loss', color='red')
    loss_ax.set_xlabel('Iteration')
    loss_ax.set_ylabel('Value')
    loss_ax.legend()


if __name__ == "__main__":
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

    # Convert training and test data to PyTorch tensors
    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    z_train = torch.tensor(z_train, dtype=torch.float32)
    z_test = torch.tensor(z_test, dtype=torch.float32)

    epochs = 1000
    learning_rate = 0.01
    step_size = 100  # decay the learning rate every x steps (or epochs)
    gamma = 0.9  # lr decay factor
    batch_size = 1000

    model = BoardModel()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()

    losses = []
    test_losses = []
    accuracies = []
    fig, (loss_ax, accuracy_ax) = plt.subplots(1, 2, figsize=(10, 5))
    plt.ion()
    plt.show()

    print(f'Training samples: {len(x_train)}')
    print(f'Test samples: {len(x_test)}')


stop = False

total_batches = 0
for epoch in range(epochs):
    if stop == True:
        print('Stopping training')
        break

    epoch_loss = 0
    batch_count = 0
    model.train()
    rang = range(0, len(x_train), batch_size)
    for i in rang:
        if keyboard.is_pressed('b'):
            stop = True
            break

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

        outputs, value = model(x_batch)
        loss_p = criterion_policy(outputs, y_batch)
        loss_v = criterion_value(outputs, z_batch)

        loss = loss_p + loss_v

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        batch_count += 1
        total_batches += 1

    epoch_loss /= batch_count
    losses.append(epoch_loss)

    with torch.no_grad():
        model.eval()

    outputs, value = model(x_test)

    loss = criterion_value(outputs, z_test)

    test_losses.append(loss.item())

    predicted_classes = torch.argmax(outputs, dim=1)
    predicted_layers = predicted_classes // (5 * 5)

    desired_predictions = torch.argmax(y_test, dim=1)
    desired_layers = desired_predictions // (5 * 5)

    correct_predictions = (predicted_classes ==
                           desired_predictions).sum().item()
    correct_layers = (predicted_layers == desired_layers).sum().item()

    accuracy = correct_predictions / len(y_test)
    accuracies.append(accuracy)

    layer_accuracy = correct_layers / len(y_test)

    scheduler.step()

    visualize_results(losses, accuracies, test_losses, loss_ax)
    fig.canvas.draw()
    fig.canvas.flush_events()

    print(
        f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.6f}, TestLoss: {loss:.6f}, Learning rate: {scheduler.get_last_lr()[0]:.6f}, Accuracy: {accuracy:.4f} ({correct_predictions}/{len(y_test)}), correct_layers: {layer_accuracy:.4f} ({correct_layers}/{len(y_test)})")


# dummy input in correct format is required to save model, that's just how it works.
dummy_input = x_train[0:1]
path = 'c:/temp/ml/tixy.onnx'

torch.onnx.export(model, dummy_input, path,
                  input_names=["input"], output_names=["output"], export_params=True)
