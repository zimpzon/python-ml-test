import numpy as np


class TixyRandomPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a

class TixyGreedyPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = self.game.getValidMoves(board, 1)

        for i in range(0, self.game.getActionSize()):
            if valids[i] == 0:
                 continue
            
            action = i
            row, col, piece, dx, dy = self.game.decodeAction(board, action)
            piece = board[row + dy, col + dx]
            if (piece != 0):
                print(piece)

        a = np.random.randint(self.game.getActionSize())
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())


        return a
