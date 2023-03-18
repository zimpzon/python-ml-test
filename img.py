import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# Load a single image
image_path = 'c:\\temp\\img\\spodoman.jpg'

image = Image.open(image_path)
image = image.resize((64, 64))
# image.save('c:\\temp\\img\\spodoman_64.jpg')
width, height = image.size

# Define a simple feedforward neural network

ls = 256


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 256)
        # self.fc2 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
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
epochs = 50
batch_size = 4
learning_rate = 0.001

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
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Plot the runtime loss graph
ax1.set_title('Runtime Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')

# Training loop
losses = []  # Initialize a list to store the losses at each epoch
for epoch in range(epochs):
    for i in range(0, len(coords), batch_size):
        x_batch = coords_tensor[i:i + batch_size]
        y_batch = pixels_tensor[i:i + batch_size]

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

    losses.append(loss.item())

    # Update the plot at each epoch
    ax1.plot(losses, 'b')
    ax1.set_xlim([0, epochs])
    ax1.set_ylim([0, max(losses)])
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
    ax2.imshow(image_pred)
    ax2.set_title('Predicted Image')

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}")

print("Training completed.")

# plt.ioff()

# # Show the figure
# plt.show()
