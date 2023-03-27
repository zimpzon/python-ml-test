class Board:
    class Piece:
        def __init__(self):
            self.type = None
            self.valid_moves = [[0 for _ in range(3)] for _ in range(3)]

    class BoardMove:
        def __init__(self):
            self.cmd = None
            self.x = 0
            self.y = 0
            self.dx = 0
            self.dy = 0

    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.cells = [['.' for _ in range(h)] for _ in range(w)]

        self.pieces = {}
        self._init_pieces()
        self._init_board()

    def parse_move_command(self, cmd):
        move = self.BoardMove()

        if cmd is None or len(cmd) != 5 or cmd[2] != ' ':
            return False, move

        x0 = ord(cmd[0]) - ord('A')
        y0 = self.h - int(cmd[1]) - 1

        x1 = ord(cmd[3]) - ord('A')
        y1 = self.h - int(cmd[4]) - 1

        dx = x1 - x0
        dy = y1 - y0

        move.cmd = cmd
        move.x = x0
        move.y = y0
        move.dx = dx
        move.dy = dy

        return True, move

    def print(self):
        print("  ", end='')
        for x in range(self.w):
            print(chr(ord('A') + x), end='')

        print()
        for y in range(self.h):
            print(chr(ord('0') + self.h - y - 1), end=' ')
            for x in range(self.w):
                print(self.cells[x][y], end='')
            print()

    def is_valid_move(self, move):
        if move is None or move.x < 0 or move.y < 0 or move.x >= self.w or move.y >= self.h:
            return False

        new_x = move.x + move.dx
        new_y = move.y + move.dy

        if new_x < 0 or new_y < 0 or new_x >= self.w or new_y >= self.h:
            return False

        if self.cells[move.x][move.y] not in self.pieces:
            return False

        return True

    def move(self, move):
        if move is None:
            raise ValueError("move cannot be None")

        new_x = move.x + move.dx
        new_y = move.y + move.dy

        piece = self.pieces.get(self.cells[move.x][move.y], None)

        self.cells[move.x][move.y] = '.'
        self.cells[new_x][new_y] = piece.type if piece is not None else '?'

    def _init_board(self):
        for y in range(self.h):
            for x in range(self.w):
                self.cells[x][y] = '.'

        self.cells[0][0] = 'X'
        self.cells[0][self.h - 1] = 'X'

    def _init_pieces(self):
        piece_T = self.Piece()
        piece_T.type = 'T'
        piece_T.valid_moves = [
            [1, 1, 1],
            [0, 0, 0],
            [0, 1, 0]
        ]

        piece_I = self.Piece()
        piece_I.type = 'I'
        piece_I.valid_moves = [
            [1, 1, 1],
            [0, 0, 0],
            [0, 1, 0]
    ]

        piece_X = self.Piece()
        piece_X.type = 'X'
        piece_X.valid_moves = [
            [1, 0, 1],
            [0, 0, 0],
            [1, 1, 1]
        ]

        piece_Y = self.Piece()
        piece_Y.type = 'Y'
        piece_Y.valid_moves = [
            [1, 0, 1],
            [0, 0, 0],
            [0, 1, 0]
        ]

        self.pieces[piece_T.type] = piece_T
        self.pieces[piece_I.type] = piece_I
        self.pieces[piece_X.type] = piece_X
        self.pieces[piece_Y.type] = piece_Y

