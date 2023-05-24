import re
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

            is_winning_move = row + dy == 0 and piece == 2 # I
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

    def parse_input(self, input_move):
        input_move = input_move.upper()

        # Define the mapping from letters to column indexes
        col_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6 }
        
        # Split the input string into the 'from' and 'to' parts
        from_str, to_str = input_move.split()
        
        # Parse the 'from' part
        from_col = col_mapping[from_str[0]]
        from_row = int(from_str[1]) - 1  # Subtract 1 to convert to 0-based indexing
        
        # Parse the 'to' part
        to_col = col_mapping[to_str[0]]
        to_row = int(to_str[1]) - 1  # Subtract 1 to convert to 0-based indexing
        
        return (from_row, from_col), (to_row, to_col)

    def find_move(self, from_pos, to_pos):
        # Calculate the differences in the row and column positions
        dy = to_pos[0] - from_pos[0]
        dx = to_pos[1] - from_pos[1]

        # Iterate over the list of moves
        for move in self.moves:
            # If the move matches the given positions and differences, return it
            if move[1:5] == (from_pos[0], from_pos[1], dx, dy):
                return move

        # If no matching move was found, return None
        return None

    def is_valid_input(self, input_move):
        # Define the regex pattern for the input format
        pattern = r'^[A-Ga-g][1-7]\s[A-Ga-g][1-7]$'
        
        # Check if the input string matches the pattern
        match = re.match(pattern, input_move)
        
        # If the input string matches the pattern, return True, otherwise return False
        return match is not None

    def play(self, board):
        TixyGame.display(board)

        self.moves = []
        valid = self.game.getValidMoves(board, 1)

        for i in range(len(valid)):
            if valid[i]:
                action = i
                row, col, piece, dx, dy = self.game.decodeAction(board, action)
                # print(f'[{len(self.moves)}]: {i}, row: {row}, col: {col}, dx: {dx}, dy: {dy}, piece: {piece}')
                self.moves.append((i, row, col, dx, dy, piece))

        while True:
            print('your move, punk:')
            input_move = input()
            if not self.is_valid_input(input_move):
                print('invalid input, punk')
                continue

            from_pos, to_pos = self.parse_input(input_move)
            valid_move = self.find_move(from_pos, to_pos)
            if valid_move is  None:
                print('invalid move, punk')
                continue

            return valid_move[0]
