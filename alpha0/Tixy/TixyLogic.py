class TixyBoard():
    def __init__(self, w, h):
        self.W = w
        self.H = h
        self.size = w * h

        # Create the empty board array.
        self.pieces = [None] * self.size
        for i in range(self.n):
            self.pieces[i] = [0] * self.size

    def __str__(self):
        return str(self.np_pieces)
