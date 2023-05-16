import numpy as np


class MCTS2():
    def __init__(self, game, nnet, args):
        self.game = game

    def getActionProb(self, board, temp=1):
        valids = self.game.getValidMoves(board, 1)
        return valids/np.sum(valids)
