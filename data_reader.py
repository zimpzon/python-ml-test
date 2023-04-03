import json


class BoardState:
    def __init__(self):
        self.PlayerIdx = 0
        self.State
        self.SelectedMove
        self.Value = 0


def read_board_states(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
        return data
