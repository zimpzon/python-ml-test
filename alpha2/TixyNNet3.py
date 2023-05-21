import sys
sys.path.append('..')
from utils import *

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class TixyNet(nn.Module):
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        super(TixyNet, self).__init__()

        self.fc1 = nn.Linear(8 * 5 * 5, 512)
        self.fc_bn1 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.action_size)

        self.fc4 = nn.Linear(512, 1)

    def forward(self, s):
        s = self.one_hot_encode(s)
        s = s.flatten(start_dim=1)

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout, training=self.training)
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training)

        pi = self.fc3(s)
        v = self.fc4(s)

        return F.log_softmax(pi, dim=1), torch.tanh(v)
    
    def one_hot_encode(self, boards):
        batch_count = boards.size(0)
        total_piece_types = 8 # 4 for each player

        # should this be 1, 2 or 2, 1? can't tell when board is square 5x5
        # does it matter? as long as it is consistent?
        one_hot_planes = torch.zeros((batch_count, total_piece_types, boards.size(2), boards.size(1)), device=boards.device, dtype=torch.float32)

        # one_hot_planes should now be batch_count * 8 * board_x * board_y
        for batch_idx in range(batch_count):
            for i in range(1, 5): # these are the piece types. Its sets a 1 in the correct plane for each piece type, for each batch.
                one_hot_planes[batch_idx, i-1] = (boards[batch_idx] == i) # planes 0-3
                one_hot_planes[batch_idx, i+3] = (boards[batch_idx] == -i) # planes 4-7

        return one_hot_planes
