from ast import Tuple
import numpy as np
from Game import Game
from TixyLogic import TixyBoard

class TixyGame(Game):
    def __init__(self, w, h):
        self.W = w
        self.H = h

    def getInitBoard(self):
        self.Turns = 0
        # starting board, same format as returned by getNextState
        return TixyBoard.getStartingBoard(self.W, self.H)

    def getBoardSize(self) -> Tuple(int, int):
        # inits nn, not used while playing
        return (self.W, self.H)

    def getActionSize(self) -> int:
        # all possible actions in our chosen encoding = 6 * W * H = pos + move dir
        return 6 * self.W * self.H # no E, W

    def debug(self):
        for i in range(self.getActionSize()):
            index = i
            rows = self.H
            cols = self.W

            # Calculate x, y, and plane id
            plane_id = index // (rows * cols)
            row = (index % (rows * cols)) // cols
            col = (index % (rows * cols)) % cols
            dx, dy = TixyBoard._action_idx[plane_id]
            print(f'actionNo: {i}, plane_id: {plane_id} row: {row}, col: {col}, dx: {dx}, dy: {dy}')

    def decodeAction(self, board, action):
        index = action
        rows = self.H
        cols = self.W

        # Calculate x, y, and plane id
        plane_id = index // (rows * cols)
        row = (index % (rows * cols)) // cols
        col = (index % (rows * cols)) % cols
        # print(f'nextState: actionNo: {plane_id}, row: {row}, col: {col}')

        piece = board[row, col]
        dx, dy = TixyBoard._action_idx[plane_id]

        return row, col, piece, dx, dy

    def getNextState(self, board, player, action) -> np.ndarray:
        row, col, piece, dx, dy = self.decodeAction(board, action)
        assert piece != 0
        assert dx != 0 or dy != 0

        board[row, col] = 0
        # dstPiece = board[row + dy, col + dx]
        # if dstPiece != 0:
        #     print("haps")

        board[row + dy, col + dx] = piece

        return board, -player

    def getValidMoves(self, board: np.ndarray, player: int) -> np.ndarray:
        # board is canonical, so our pieces are > 0, indeed it is always called with hardcoded player = 1

        valid_moves = np.zeros(self.W * self.H * 6, dtype=int) # size == action size
        assert valid_moves.size == self.getActionSize()

        flat = board.flatten()
        assert flat.size == (self.W * self.H)

        for i in range(flat.size):
            piece = flat[i]
            piece_found = piece > 0 if player == 1 else piece < 0
            if piece_found:
                x = i % self.W
                y = i // self.W

                # valid_directions is a list of validated tuples (dx, dy)
                valid_directions = TixyBoard.getValidDirections(board, x, y, piece, player)
                for _, _, plane_idx in valid_directions:
                    valid_moves[i + plane_idx * self.W * self.H] = 1

        # print(f'getValid moves for player {player}, valid count: {np.sum(valid_moves == 1)}')
        return valid_moves

    def getGameEnded(self, board, player) -> float:
        # ignore player, always seen from player 1.

        # return draw if too many turns
        self.Turns += 1
        if (self.Turns > 50):
            return 1e-4

        pl1NoPieces = np.all(board <= 0)
        pl2NoPieces = np.all(board >= 0)

        row0_has_positive_value = np.any(board[0] > 0)
        row4_has_negative_value = np.any(board[4] < 0)

        if (row0_has_positive_value or pl2NoPieces):
            return 1
        elif (row4_has_negative_value or pl1NoPieces):
            return -1
        else:
            return 0

    def getCanonicalForm(self, board, player) -> np.ndarray:
        # canonical: your pieces are positive at the bottom, and opponents pieces are negative at the top, no matter if you are p1 or p2.
        c_board = board if player == 1 else np.rot90(np.rot90(board))
        return c_board * player

    def getSymmetries(self, board, pi):
        # returns array of tuple (board, pi)
        # probably canonical?
        # definately mirror. rotation if square, or padded.
        return [(board, pi)]

    def stringRepresentation(self, board):
        return np.array2string(board)

    @staticmethod
    def display(board):
        print(np.array2string(board))
