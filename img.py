import time
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR

# Load a single image
image_path = 'c:\\temp\\img\\dino.jpg'

image = Image.open(image_path)
image = image.resize((128, 64))
# image.save('c:\\temp\\img\\spodoman_64.jpg')
width, height = image.size

# Define a simple feedforward neural network

# Hyperparameters
epochs = 100000000
batch_size = 10
learning_rate = 0.001

ls = 100

print(f'layer-size: {ls}')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, ls)
        self.fc2 = nn.Linear(ls, ls)
        self.fc3 = nn.Linear(ls, ls)
        self.fc4 = nn.Linear(ls, ls)
        self.fc5 = nn.Linear(ls, ls)
        self.fc6 = nn.Linear(ls, ls)
        # self.fc7 = nn.Linear(ls, ls)
        # self.fc8 = nn.Linear(ls, ls)
        # self.fc9 = nn.Linear(ls, ls)
        # self.fc10 = nn.Linear(ls, ls)
        self.fc11 = nn.Linear(ls, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        # x = torch.relu(self.fc7(x))
        # x = torch.relu(self.fc8(x))
        # x = torch.relu(self.fc9(x))
        # x = torch.relu(self.fc10(x))
        x = self.fc11(x)
        return x

# Prepare the dataset


def create_dataset(image):
    height, width = image.size
    x, y = np.meshgrid(np.arange(width), np.arange(height), indexing='ij')
    # Normalize the coordinates
    coords = np.stack([x.ravel()/(width-1) - 0.5,
                      y.ravel()/(height-1) - 0.5], axis=1)
    pixels = np.array(image).reshape(-1, 3) / 255
    return coords, pixels


coords, pixels = create_dataset(image)
coords_tensor = torch.tensor(coords, dtype=torch.float32)
pixels_tensor = torch.tensor(pixels, dtype=torch.float32)

print(f'batch-size: {batch_size}')

img_counter = 0

# Model, loss, and optimizer
model = Net()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
if next(model.parameters()).is_cuda:
    print("Model is running on GPU.")
else:
    print("Model is running on CPU.")

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=1000, gamma=0.9)

plt.ion()

# Create a figure with two subplots
fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(10, 5))

# Plot the runtime loss graph
ax1.set_title('Runtime Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')

next_view = 0

# Training loop
losses = []  # Initialize a list to store the losses at each epoch


def weighted_moving_average(data, window_size):
    weights = np.arange(1, window_size + 1)
    wma = np.convolve(data, weights[::-1], mode='valid') / np.sum(weights)
    return wma


def update_view():
    global img_counter

    max_value = np.amax(losses[-40:])

    ax1.set_xlim([0, len(losses)])
    ax1.set_ylim([0, max_value])
    ax1.plot(losses, 'b')

    losses_wma = weighted_moving_average(losses, 100)
    ax1.plot(losses_wma, 'y', label='WMA')

    fig.canvas.draw()
    fig.canvas.flush_events()

    img_counter += 1

    with torch.no_grad():
        model.eval()
        pixels_pred_tensor = model(coords_tensor)
        pixels_pred = pixels_pred_tensor.numpy() * 255

    # Reshape the predicted pixel values into an image
    pixels_pred = pixels_pred.reshape(height, width, 3)
    pixels_pred_clipped = np.clip(pixels_pred, 0, 255)
    image_rgb = np.uint8(pixels_pred_clipped)
    image_pred = Image.fromarray(image_rgb)

    # image_pred = Image.fromarray(
    #     np.uint8(np.clip(pixels_pred.reshape(height, width, 1), 0, 255)))

    # Show the predicted image in the second subplot
    ax3.imshow(image_pred)
    ax3.set_title('Predicted Image')


pix_sampled = 0

# This is not officially epochs, since an epoch requires seeing all training data. Ths is purely random.
for epoch in range(epochs):
    start_time = time.time()

    fig.canvas.flush_events()

    epoch_loss = 0
    count = 0
    scheduler.step()

    repeat = 10
    for i in range(0, repeat):
        random_indices = np.random.choice(
            coords_tensor.shape[0], size=batch_size, replace=True)

        x_batch = coords_tensor[random_indices]
        y_batch = pixels_tensor[random_indices]

        pix_sampled += batch_size

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        count += 1

        # Calculate the time for the epoch
        epoch_time = time.time() - start_time

        if time.time() > next_view:
            loss = epoch_loss / count
            losses.append(loss)
            print(
                f"Epoch [{epoch+1}/{epochs}], Loss: {loss:.5}, Pix sampled: {pix_sampled}, Batch size: {batch_size}, lr: {optimizer.param_groups[0]['lr']:.5f}")
            update_view()

            epoch_loss = 0
            count = 0
            # batch_size -= 20
            # if batch_size < 20:
            #     batch_size = 20

            next_view = time.time() + 2

print("Training completed.")

plt.ioff()
update_view(epochs)
plt.show()
