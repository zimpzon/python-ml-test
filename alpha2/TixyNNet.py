import sys
sys.path.append('..')
from utils import *

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class TixyNNet(nn.Module):
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        super(TixyNNet, self).__init__()
        self.conv1 = nn.Conv2d(8, args.num_channels, 3, stride=1, padding=1) # Updated number of input channels
        self.conv2 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1)

        self.bn1 = nn.BatchNorm2d(args.num_channels)
        self.bn2 = nn.BatchNorm2d(args.num_channels)
        self.bn3 = nn.BatchNorm2d(args.num_channels)
        self.bn4 = nn.BatchNorm2d(args.num_channels)

        self.fc1 = nn.Linear(args.num_channels*(self.board_x-2)*(self.board_y-2), 1024) # Modified input size
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.action_size)

        self.fc4 = nn.Linear(512, 1)

    def one_hot_encode(self, boards):
        batch_count = boards.size(0)
        one_hot_planes = torch.zeros((batch_count, 8, boards.size(1), boards.size(2)), device=boards.device, dtype=torch.float32)

        for batch_idx in range(batch_count):
            for i in range(1, 5):
                one_hot_planes[batch_idx, i-1] = (boards[batch_idx] == i)
                one_hot_planes[batch_idx, i+3] = (boards[batch_idx] == -i)

        return one_hot_planes

    def forward(self, s):
        s = self.one_hot_encode(s)

        s = F.relu(self.bn1(self.conv1(s)))
        s = F.relu(self.bn2(self.conv2(s)))
        s = F.relu(self.bn3(self.conv3(s)))
        s = F.relu(self.bn4(self.conv4(s)))
        s = s.view(-1, self.args.num_channels*(self.board_x-2)*(self.board_y-2))

        s = F.relu(self.fc_bn1(self.fc1(s)))

        s = F.dropout(s, p=self.args.dropout, training=self.training)  # batch_size x 1024
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training)  # batch_size x 512

        pi = self.fc3(s)                                                                         # batch_size x action_size
        v = self.fc4(s)                                                                          # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)
