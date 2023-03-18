import time
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# Load a single image
image_path = 'c:\\temp\\img\\red.jpg'

image = Image.open(image_path)
image = image.resize((256, 256))
# image.save('c:\\temp\\img\\spodoman_64.jpg')
width, height = image.size

# Define a simple feedforward neural network

ls = 10

print(f'layer-size: {ls}')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, ls)
        self.fc2 = nn.Linear(ls, ls)
        self.fc3 = nn.Linear(ls, ls)
        self.fc4 = nn.Linear(ls, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Prepare the dataset


def create_dataset(image):
    x, y = np.meshgrid(np.arange(width), np.arange(height), indexing='ij')
    coords = np.stack([x.ravel(), y.ravel()], axis=1)
    pixels = np.array(image).reshape(-1, 3)
    return coords, pixels


coords, pixels = create_dataset(image)
coords_tensor = torch.tensor(coords, dtype=torch.float32)
pixels_tensor = torch.tensor(pixels, dtype=torch.float32)

# Hyperparameters
epochs = 10000
batch_size = 1000
learning_rate = 0.001

print(f'batch-size: {batch_size}')

# Model, loss, and optimizer
model = Net()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
if next(model.parameters()).is_cuda:
    print("Model is running on GPU.")
else:
    print("Model is running on CPU.")

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

plt.ion()

# Create a figure with two subplots
fig, (ax1,  ax3) = plt.subplots(1, 2, figsize=(10, 5))

# Plot the runtime loss graph
ax1.set_title('Runtime Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')

next_view = 0

# Training loop
losses = []  # Initialize a list to store the losses at each epoch


def update_view(current_epoch):
    display_epoch = round(current_epoch / 10) * 10 + 10

    ax1.plot(losses, 'b')
    ax1.set_xlim([0, display_epoch])
    ax1.set_ylim([0, min(max(losses), 5000)])

    fig.canvas.draw()
    fig.canvas.flush_events()

    with torch.no_grad():
        model.eval()
        pixels_pred_tensor = model(coords_tensor)
        pixels_pred = pixels_pred_tensor.numpy()

    # Reshape the predicted pixel values into an image
    image_pred = Image.fromarray(
        np.uint8(np.clip(pixels_pred.reshape(height, width, 3), 0, 255)))

    # Show the predicted image in the second subplot
    ax3.imshow(image_pred)
    ax3.set_title('Predicted Image')


# Yhis is not officially epochs, since an epoch requires seeing all training data. Ths is purely random.
for epoch in range(epochs):
    start_time = time.time()

    repeat = round(1000 / batch_size)
    for i in range(0, repeat):
        random_indices = np.random.choice(
            coords_tensor.shape[0], size=batch_size, replace=False)

        x_batch = coords_tensor[random_indices]
        y_batch = pixels_tensor[random_indices]

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        losses.append(loss.item() / repeat)

        # Calculate the time for the epoch
        epoch_time = time.time() - start_time

        if time.time() > next_view:
            batch_size = max(10, round(batch_size * 0.8 - 1))

            print(
                f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}, Epoch time: {epoch_time:.2f}, Batch size: {batch_size}")
            update_view(epoch)
            next_view = time.time() + 2

print("Training completed.")

plt.ioff()
update_view(epochs)
plt.show()
