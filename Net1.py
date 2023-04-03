import numpy as np
import onnx
from sklearn.model_selection import train_test_split
import torch
import torch.onnx
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from torch.nn import functional as F
from data_reader import read_board_states


class BoardGameModel(nn.Module):
    def __init__(self):
        super(BoardGameModel, self).__init__()

        self.input_size = 225
        self.output_size = 200

        self.fc1 = nn.Linear(self.input_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, self.output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return torch.softmax(x, dim=1)


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
    states = read_board_states('c:/temp/ml/gen1.json')

    all_states = list(map(lambda s: s.get('State'), states))
    # all_states = [np.array(array_data).reshape((9 * 5 * 5))
    #               for array_data in all_states]
    all_states = np.stack(all_states, axis=0)

    # Get class labels instead of one-hot vectors
    expected_move = list(map(lambda s: s.get('SelectedMove'), states))

    # Split data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        all_states, expected_move, test_size=0.2, random_state=42)

    # Convert training and test data to PyTorch tensors
    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    epochs = 10
    learning_rate = 0.001
    step_size = 100  # Decay the learning rate every x steps (or epochs)
    gamma = 0.8  # Decay factor
    batch_size = 1000  # MUST BE DIVISOR OF SAMPLES

    model = BoardGameModel()

    loss_fn = nn.CrossEntropyLoss()

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

        x_batch = x_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]

        random_indices = np.random.choice(
            len(x_batch), size=batch_size, replace=True)

        x_batch = x_batch[random_indices]
        y_batch = y_batch[random_indices]

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

    # outputs is a len(y_test) sized list of 200 elements.
    # get a list of the indexes of all max values in the arrays.
    _, max_output_indices = torch.max(outputs, dim=1)
    _, max_expected_indices = torch.max(y_test, dim=1)

    correct_predictions = (max_output_indices ==
                           max_expected_indices).sum().item()
    accuracy = correct_predictions / len(y_test)
    accuracies.append(accuracy)

    scheduler.step()

    visualize_results(losses, accuracies, test_losses, loss_ax)
    fig.canvas.draw()
    fig.canvas.flush_events()

    print(
        f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.6f}, TestLoss: {loss:.6f}, Learning rate: {scheduler.get_last_lr()[0]:.6f}, Accuracy: {accuracy:.4f} ({correct_predictions}/{len(y_test)}), total_batches: {total_batches}")

    # print(
    #     f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.8f}, TestLoss: {combined_loss:.8f}, Learning rate: {scheduler.get_last_lr()[0]:.6f}, Accuracy: {accuracy:.6f} ({correct_predictions}/{len(y_test)}), total_batches: {total_batches}")
# Save the model if desired

dummy_input = x_train[0:1]
path = 'c:/temp/ml/tixy.onnx'

torch.onnx.export(model, dummy_input, path,
                  input_names=["input"], output_names=["output"], export_params=True)

model = onnx.load('c:\\temp\\ml\\tixy.onnx')

# Check the model's validity
onnx.checker.check_model(model)
print("The model is valid.")


torch.save(model.state_dict(), path)

plt.ioff()
plt.show()
