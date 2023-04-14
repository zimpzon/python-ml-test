import time
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from torch.optim.lr_scheduler import StepLR

# NEXT: more categories. Just 2 may not be enough to tell the diff between cross entropy and MSE (indeed, MSE seems to do better right now, if overfitting is considered good)


class Net(nn.Module):
    def __init__(self, layer_size):
        super(Net, self).__init__()
        self.relu = nn.ReLU()
        self.input = nn.Linear(2, layer_size)
        self.fc1 = nn.Linear(layer_size, layer_size)
        self.bn1 = nn.BatchNorm1d(layer_size)
        self.fc2 = nn.Linear(layer_size, layer_size)
        self.bn2 = nn.BatchNorm1d(layer_size)
        self.fc3 = nn.Linear(layer_size, layer_size)
        self.bn3 = nn.BatchNorm1d(layer_size)
        self.fc4 = nn.Linear(layer_size, layer_size)
        self.bn4 = nn.BatchNorm1d(layer_size)
        self.fc5 = nn.Linear(layer_size, layer_size)
        self.bn5 = nn.BatchNorm1d(layer_size)
        self.fc6 = nn.Linear(layer_size, layer_size)
        self.bn6 = nn.BatchNorm1d(layer_size)
        self.fc7 = nn.Linear(layer_size, layer_size)
        self.bn7 = nn.BatchNorm1d(layer_size)
        self.fc8 = nn.Linear(layer_size, layer_size)
        self.bn8 = nn.BatchNorm1d(layer_size)
        self.fc_out = nn.Linear(layer_size, 4)

    def forward(self, x):
        x = self.relu(self.input(x))
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.relu(self.bn4(self.fc4(x)))
        x = self.relu(self.bn5(self.fc5(x)))
        # x = self.relu(self.bn6(self.fc6(x)))
        # x = self.relu(self.bn7(self.fc7(x)))
        # x = self.relu(self.bn8(self.fc8(x)))
        x = self.fc_out(x)
        return x


def moving_average(data, window_size):
    moving_averages = []
    for i in range(len(data) - window_size + 1):
        window = data[i:i + window_size]
        average = sum(window) / window_size

        if not moving_averages:
            moving_averages = [average] * (window_size - 1)

        moving_averages.append(average)
    return moving_averages


class MinValueStepLR(StepLR):
    def __init__(self, optimizer, step_size, gamma, min_lr):
        super().__init__(optimizer, step_size, gamma)
        self.min_lr = min_lr

    def get_lr(self):
        # Get the learning rates from the parent class
        lr_list = super().get_lr()

        # Apply the minimum learning rate constraint
        constrained_lr_list = [max(lr, self.min_lr) for lr in lr_list]

        return constrained_lr_list


layer_size = 128
epochs = 500000
batch_size = 50
learning_rate = 0.01
step_size = 100  # Decay the learning rate every x steps (or epochs)
gamma = 0.8  # Decay factor


def create_dataset(image):
    width, height = image.size
    x, y = np.meshgrid(np.arange(width), np.arange(height), indexing='ij')
    coords = np.stack(
        [x.ravel() / width - 0.5, y.ravel() / height - 0.5], axis=1)
    pixels = np.array(image).reshape(-1, 3) / 255

    labels = np.zeros((pixels.shape[0], 4))

    r_component = pixels[:, 0]
    g_component = pixels[:, 1]
    b_component = pixels[:, 2]

    labels[True] = [1, 0, 0, 0]
    labels[r_component > 0.5] = [0, 1, 0, 0]
    labels[g_component > 0.5] = [0, 0, 1, 0]
    labels[b_component > 0.5] = [0, 0, 0, 1]

    return coords, labels


def train_model(device, model, epochs, batch_size, coords_tensor, labels_tensor, fig, loss_ax, image_ax):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    losses = []

    last_visualization_time = time.time()

    t_forward = 0
    t_backwards = 0
    t_viz = 0

    for epoch in range(epochs):
        epoch_loss = 0
        for _ in range(coords_tensor.shape[0] // batch_size):
            random_indices = np.random.choice(
                coords_tensor.shape[0], size=batch_size, replace=False)

            x_batch = coords_tensor[random_indices].to(device)
            y_batch = labels_tensor[random_indices].to(device)

            optimizer.zero_grad()
            start = time.time()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

            t_forward += time.time() - start
            start = time.time()

            loss.backward()
            optimizer.step()
            t_backwards += time.time() - start

            current_time = time.time()
            if current_time - last_visualization_time >= 2:
                start = time.time()
                last_visualization_time = current_time
                visualize_results(
                    losses, model, coords_tensor, labels_tensor, loss_ax, image_ax)
                fig.canvas.draw()
                fig.canvas.flush_events()
                t_viz = time.time() - start

            epoch_loss += loss.item()

        epoch_loss /= (coords_tensor.shape[0] // batch_size)
        losses.append(epoch_loss)

        print(
            f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.6f}, Learning rate: {scheduler.get_last_lr()[0]:.6f}, forwardSec: {t_forward:.3f}, backSec: {t_backwards:.3f}, vizSec: {t_viz:.3f}")
        t_forward = 0
        t_backwards = 0

        scheduler.step()

    return losses


ylim = 0.1


def visualize_results(losses, model, coords_tensor, labels_tensor, loss_ax, image_ax):
    global ylim
    if len(losses) > 10:
        ylim = losses[10]

    loss_ax.clear()
    loss_ax.set_ylim(0.0, ylim)
    loss_ax.plot(losses, label='Loss')
    loss_ax.plot(moving_average(losses, window_size=10),
                 label='Smoothed loss', color='y')
    loss_ax.set_xlabel('Iteration')
    loss_ax.set_ylabel('Loss')
    loss_ax.legend()

    with torch.no_grad():
        model.eval()
        labels_pred_tensor = model(coords_tensor)

        labels_pred = labels_pred_tensor.cpu().numpy()

        indices = np.argmax(labels_pred, axis=1)

        pix = np.zeros(width * height * 3)
        for i in range(len(labels_pred)):
            if indices[i] != 0:
                pix[i * 3 + (indices[i] - 1)] = 200

        image_pred = Image.fromarray(
            np.uint8(np.clip(pix.reshape(width, height, 3), 0, 255)))

        labels_pred_tensor.detach()

    image_ax.clear()
    image_ax.imshow(image_pred)
    image_ax.set_title('Predicted Image')

    plt.draw()


if __name__ == "__main__":
    image_path = 'c:\\temp\\img\\classes.png'
    image = Image.open(image_path)
    width, height = image.size
    image = image.resize((128, 128))
    width, height = image.size

    fig, (loss_ax, image_ax) = plt.subplots(1, 2, figsize=(10, 5))
    plt.ion()
    plt.show()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    coords, labels = create_dataset(image)
    coords_tensor = torch.tensor(coords, dtype=torch.float32).to(device)
    labels_tensor = torch.tensor(labels, dtype=torch.float32).to(device)

    model = Net(layer_size).to(device)

    if next(model.parameters()).is_cuda:
        print("Model is running on GPU.")
    else:
        print("Model is running on CPU.")

    print(
        f"Training model with layer size: {layer_size}, batch size: {batch_size}, learning rate: {learning_rate}")
    losses = train_model(device, model, epochs, batch_size,
                         coords_tensor, labels_tensor, fig, loss_ax, image_ax)
