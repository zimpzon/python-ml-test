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
                return action

        a = np.random.randint(self.game.getActionSize())
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())

        return a

class TixyHumanPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        # display(board)
        valid = self.game.getValidMoves(board, 1)
        for i in range(len(valid)):
            if valid[i]:
                x = i % 5
                y = i // 5
                move = i // 25
                piece = board[y, x]

                print("[", int(i/self.game.n), int(i%self.game.n), end="] ")
        while True:
            input_move = input()
            input_a = input_move.split(" ")
            if len(input_a) == 2:
                try:
                    x,y = [int(i) for i in input_a]
                    if ((0 <= x) and (x < self.game.n) and (0 <= y) and (y < self.game.n)) or \
                            ((x == self.game.n) and (y == 0)):
                        a = self.game.n * x + y if x != -1 else self.game.n ** 2
                        if valid[a]:
                            break
                except ValueError:
                    # Input needs to be an integer
                    'Invalid integer'
            print('Invalid move')