import json


class BoardState:
    def __init__(self):
        self.State = [0.0] * (5 * 5 * 8)
        self.Selectedmove = [0.0] * 8 * 5 * 5
        self.Value = 0


def read_board_states(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
        return data
