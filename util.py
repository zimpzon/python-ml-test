import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

class Net(nn.Module):
    def __init__(self, ls):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, ls)
        self.fc2 = nn.Linear(ls, ls)
        self.fc3 = nn.Linear(ls, ls)
        self.fc4 = nn.Linear(ls, ls)
        self.fc5 = nn.Linear(ls, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

def moving_average(data, window_size):
    moving_averages = []
    for i in range(len(data) - window_size + 1):
        window = data[i:i + window_size]
        average = sum(window) / window_size
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
