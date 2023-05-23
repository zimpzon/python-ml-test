import numpy as np

from TixyGame import TixyGame
from TixyLogic import TixyBoard


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

            is_winning_move = row + dy == 0
            if is_winning_move:
                return action
            
            piece = board[row + dy, col + dx]
            can_capture = piece != 0
            if can_capture:
                return action

        a = np.random.randint(self.game.getActionSize())
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())

        return a

class TixyHumanPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        TixyGame.display(board)

        self.moves = []
        valid = self.game.getValidMoves(board, 1)

        for i in range(len(valid)):
            if valid[i]:
                action = i
                row, col, piece, dx, dy = self.game.decodeAction(board, action)
                print(f'[{len(self.moves)}]: {i}, row: {row}, col: {col}, dx: {dx}, dy: {dy}, piece: {piece}')
                self.moves.append((i, row, col, dx, dy, piece))

        print('your move, punk:')
        input_move = input()
        id = int(input_move)
        move = self.moves[id]
        return move[0]
