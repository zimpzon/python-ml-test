import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural network model


class Adder(nn.Module):
    def __init__(self):
        super(Adder, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Generate training data
num_samples = 100
inputs = torch.randint(0, 100, (num_samples, 2)).float()
outputs = (inputs[:, 0] * inputs[:, 1]).view(-1, 1)

# Initialize the model, loss function, and optimizer
model = Adder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    predictions = model(inputs)
    loss = criterion(predictions, outputs)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Test the trained model
test_inputs = torch.tensor([[3, 5], [10, 20], [50, 50]], dtype=torch.float32)
test_outputs = model(test_inputs)
print("Test results:")
for i, result in enumerate(test_outputs):
    print(f"{test_inputs[i, 0]} * {test_inputs[i, 1]} = {result.item():.2f}")
