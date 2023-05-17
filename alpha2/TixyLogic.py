import numpy as np

# axis 0 is down. axis 1 is right. axis 2 is back (think z)
class TixyBoard():
    _piece_directions = {
        +1: [(-1, -1), (0, -1), (1, -1), (0, 1)], #Tu
        +2: [(0, -1), (0, 1)], #Iu
        +3: [(-1, -1), (1, -1), (-1, 1), (1, 1)], #Xu
        +4: [(-1, -1), (1, -1), (0, 1)], #Yu
        
        -1: [(0, -1), (0, 1), (-1, 1), (1, 1)], #Td
        -2: [(0, -1), (0, 1)], #Id
        -3: [(-1, -1), (1, -1), (-1, 1), (1, 1)], #Xd
        -4: [(-1, 1), (1, 1), (0, -1)], #Yd
    }

    _plane_idx = {
        (0, -1): 0,
        (1, -1): 1,
        (1, 1): 2,
        (0, 1): 3,
        (-1, 1): 4,
        (-1, -1): 5,
    }

    _action_idx = {v: k for k, v in _plane_idx.items()}

    @staticmethod
    def getStartingBoard(w: int, h: int) -> np.ndarray:
        board = np.zeros((h, w), dtype=int)

        board[4, 0] = 0
        board[4, 1] = 2
        board[4, 2] = 0
        board[4, 3] = 0
        board[4, 4] = 0

        board[0, 0] = 0
        board[0, 1] = 0
        board[0, 2] = 0
        board[0, 3] = -3
        board[0, 4] = 0

        return board

    @staticmethod
    def getValidDirections(board: np.ndarray, x: int, y: int, piece: int, player: int) -> np.ndarray:
        result = []
        for direction in TixyBoard._piece_directions[piece]:
            dx, dy = direction
            if x + dx >= 0 and x + dx < board.shape[1] and y + dy >= 0 and y + dy < board.shape[0]:
                piece = board[y + dy, x + dx]
                is_legal_target = piece <= 0 if player == 1 else piece >= 0
                if is_legal_target:
                    plane_idx = TixyBoard._plane_idx[direction]
                    result.append((dx, dy, plane_idx))

        return result