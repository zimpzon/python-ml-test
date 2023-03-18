import torch
import torch.nn as nn
import torch.optim as optim


class Multiplier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Multiplier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train(model, criterion, optimizer, x_train, y_train, epochs):
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 1000 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')


if __name__ == '__main__':
    input_size = 2
    hidden_size = 200
    output_size = 1
    learning_rate = 0.001
    epochs = 10000

    model = Multiplier(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Generate training data
    x_train = torch.tensor([[i, j] for i in range(1, 10)
                           for j in range(1, 10)], dtype=torch.float32)
    y_train = torch.tensor([[i * j] for i in range(1, 10)
                           for j in range(1, 10)], dtype=torch.float32)

    # Train the model
    train(model, criterion, optimizer, x_train, y_train, epochs)

    # Test the model
    with torch.no_grad():
        x_test = torch.tensor(
            [[3.0, 5.0], [6.0, 7.0], [80.0, 10.0]], dtype=torch.float32)
        y_test = model(x_test)
        print('Test results:')
        for i in range(len(x_test)):
            print(f'{x_test[i][0]} * {x_test[i][1]} = {y_test[i][0]:.2f}')
