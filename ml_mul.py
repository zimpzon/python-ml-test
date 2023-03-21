import torch
import matplotlib.pyplot as plt
import time

# Define the neural network


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(2, 100)
        self.fc2 = torch.nn.Linear(100, 100)
        self.fc3 = torch.nn.Linear(100, 1)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Define the loss function
loss_fn = torch.nn.loss()

# Define the training data
x_train = torch.randn(10000, 2)
y_train = x_train[:, 0] * x_train[:, 1]

# Define the test data
x_test2 = torch.Tensor([[2, 3], [4, 5], [6, 7]])
y_test2 = x_test2[:, 0] * x_test2[:, 1]

x_test = torch.Tensor([[i, j] for i in range(1, 101, 10)
                      for j in range(1, 101, 10)])
y_test = x_test[:, 0] * x_test[:, 1]


# Initialize the model and optimizer
net = Net()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

# Train the model
train_losses = []
test_losses = []
fig, ax = plt.subplots()
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.plot(train_losses, 'g', label='Training Loss')
ax.plot(test_losses, 'r', label='Test Loss', )
ax.legend()

plt.ion()


def update_graph(train_losses, test_losses):
    ax.plot(train_losses, 'g', label='Training Loss')
    ax.plot(test_losses, 'r', label='Test Loss', )
    fig.canvas.draw()
    plt.pause(0.001)


start_time = time.time()
for epoch in range(10000):
    optimizer.zero_grad()
    outputs = net(x_train)
    loss = loss_fn(outputs.view(-1), y_train)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    # Test the model on the test data
    with torch.no_grad():
        net.eval()
        y_pred = net(x_test).view(-1)
        test_loss = loss_fn(y_pred, y_test)
        test_losses.append(test_loss.item())
    net.train()

    # Update the graph every 0.5 seconds
    if time.time() - start_time > 0.5:
        update_graph(train_losses, test_losses)
        start_time = time.time()

    print(
        f'Epoch {epoch+1}/{100}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')

update_graph(train_losses, test_losses)
plt.show(block=True)

# Test the model
with torch.no_grad():
    net.eval()
    y_pred = net(x_test).view(-1)
    test_loss = loss_fn(y_pred, y_test)
print(f'Test loss: {test_loss.item():.4f}, Predictions: {y_pred.tolist()}')
