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
        # display(board)
        valid = self.game.getValidMoves(board, 1)
        for i in range(len(valid)):
            if valid[i]:
                x = i % 5
                y = i // 5
                move = i // 25
                piece = board[y, x]
                print(f'valid move: {i}, piece: {piece}, move: {move}, x: {x}, y: {y}')


        while True:
            input_move = input()
            input_a = input_move.split(" ")
            print('your move:' + input_a)
