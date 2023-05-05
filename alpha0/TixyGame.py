from ast import Tuple
import numpy as np
from Game import Game
from TixyLogic import TixyBoard

class TixyGame(Game):
    def __init__(self, w, h):
        self.W = w
        self.H = h

    def getInitBoard(self):
        # starting board, same format as returned by getNextState
        b = TixyBoard(self.W, self.H)
        return b.cells

    def getBoardSize(self) -> Tuple(int, int):
        # inits nn, not used while playing
        return (self.W, self.H)

    def getActionSize(self) -> int:
        # all possible actions in our chosen encoding = 6 * W * H = pos + move dir
        return 6 * self.W * self.H # no E, W

    def getNextState(self, board, player, action) -> np.ndarray:
        # not canonical (does it matter?)
        # action is pos + move, already filtered by valid actions (I think)
        # execute move for player and return board
        # which player doesn't matter, I think?
        return board

    def getValidMoves(self, board: np.ndarray, player: int) -> np.ndarray:
        # board is canonical, so our pieces are > 0, indeed it is always called with hardcoded player = 1
        # move[getActionSize] of 1 or one for valid/not valid, for current player.

        # 6 planes for move direction, 0 = N, 1 = NE, 3 = SE, 4 = S, 5 = SW

        valid_moves = np.zeros(self.W * self.H * 6, dtype=int) # size == action size
        assert valid_moves.size == self.getActionSize()

        flat = board.flatten()
        assert flat.size == (self.W * self.H)

        for i in range(flat.size):
            piece = flat[i] # idx -4 .. 0 .. 4
            # get valid directions for piece, then set a 1 in the according planes
            # lets hack in up/down only for now
            if piece != 0:
                # TODO: planes = getValidDirections(piece)
                planes = [0, 4] # N, S
                # TODO: then filter out invalid moves, like moving off the board, moving to own piece
                # probably get x, y after move and check if it is on the board and empty?
                for plane in planes:
                    valid_moves[i + plane * self.W * self.H] = 1

        return valid_moves

    def getGameEnded(self, board, player) -> int:
        # not canonical (does it matter?)
        # win condition. a piece reached opponents line (for now).
        # 1 if player won, -1 if player lost, small non-zero value for draw.
        return 0

    def getCanonicalForm(self, board, player) -> np.ndarray:
        # canonical: your pieces are positive and opponents pieces are negative, no matter if you are p1 or p2.
        return player * board # mul 1 or -1

    def getSymmetries(self, board, pi):
        # probably canonical?
        # definately mirror. rotation if square, or padded.
        return board

    def stringRepresentation(self, board):
        return np.array2string(board)
