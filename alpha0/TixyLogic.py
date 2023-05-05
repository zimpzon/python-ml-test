import numpy as np

# 1 equals T
# 2 equals I
# 3 equals X
# 4 equals Y

class TixyBoard():
    def __init__(self, w, h):
        self.W = w
        self.H = h
        self.size = w * h

        # axis 0 is down. axis 1 is right. axis 2 is back (think z)
        self.cells = np.zeros((h, w), dtype=int)
        self.cells[4, 0] = 1
        self.cells[0, 2] = -1
