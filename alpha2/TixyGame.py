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

    def getNextState(self, immutable_board, player, action) -> np.ndarray:
        board = np.copy(immutable_board)

        row, col, piece, dx, dy = self.decodeAction(board, action)
        assert piece != 0
        assert dx != 0 or dy != 0
        assert row + dy >= 0 and row + dy < self.H
        assert col + dx >= 0 and col + dx < self.W

        board[row, col] = 0
        # dstPiece = board[row + dy, col + dx]
        # if dstPiece != 0:
        #     print("haps")

        board[row + dy, col + dx] = piece

        return board, -player

    def getValidMoves(self, board: np.ndarray, player: int = 0) -> np.ndarray:
        # always seen from player1
        valid_moves = np.zeros(self.W * self.H * 6, dtype=int) # size == action size
        assert valid_moves.size == self.getActionSize()

        flat = board.flatten()
        assert flat.size == (self.W * self.H)

        for i in range(flat.size):
            piece = flat[i]
            piece_found = piece > 0 # always seen from player1
            if piece_found:
                x = i % self.W
                y = i // self.W

                # valid_directions are validated for own pieces and board size
                valid_directions = TixyBoard.getValidDirections(board, x, y, piece, 1)
                for _, _, plane_idx in valid_directions:
                    valid_moves[i + plane_idx * self.W * self.H] = 1

        #print(f'getValid moves for player {player}, valid count: {np.sum(valid_moves == 1)}')
        return valid_moves

    def getGameEnded(self, board, player: int = 0) -> float:
        pl1_no_pieces = np.all(board <= 0)
        pl2_mo_pieces = np.all(board >= 0)

        row0_has_positive_value = np.any(board[0] > 0)
        row4_has_negative_value = np.any(board[4] < 0)

        if (row0_has_positive_value or pl2_mo_pieces):
            return 1
        elif (row4_has_negative_value or pl1_no_pieces):
            return -1
        else:
            return 0

    def turnBoard(self, board):
        return np.rot90(np.rot90(board.copy())) * -1
    
    def getSymmetries2(self, board, pi):
        return [(board, pi)]

    def getSymmetries(self, board, pi):
        # convert pi into 5x5 so both have the same shape
        # do the stuff and flatten pi, then return.
        # Verify by rotating/flipping again and compare with input.
        return [(board, pi)]
    
        # flipping ALMOST works, but it also needs to flip the directiom of the move idx (0..5)
        # since it introduces more uncertainty save it for later
        assert len(board.shape) == 2, "Board must be 2D"
        
        # Original board and pi
        original = (board, pi)

        # Flip board along the appropriate axis
        board_flipped = np.flip(board, axis=1)

        # Reshape pi to 3D, flip it and then flatten it back to 1D
        pi = np.array(pi).reshape(6, 5, 5)
        pi_flipped = np.flip(pi, axis=2).flatten().tolist()

        return [original, (board_flipped, pi_flipped)]

    def stringRepresentation(self, board):
        return np.array2string(board)

    @staticmethod
    def display(board):
        print(np.array2string(board))
