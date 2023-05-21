import unittest

import numpy as np
import TixyGame
from TixyNNetWrapper import TixyNetWrapper

class MyTest(unittest.TestCase):
   
        def test(self):
            game = TixyGame.TixyGame(5, 5)
            board = game.getInitBoard()
            wrapper = TixyNetWrapper(game)

            # network output: board_size * 6 planes where 0-5 are the 6 possible actions
            #board[:] = 0
            pi = [0.0] * 150
            pi[10] = 1
            tup = (board, pi, -1)
            examples =  [tup] * 10000

            # list of tuple (board, 150 probs, v)
            # can predict probs + 5 from board
            for i in range(1):
                wrapper.train(examples)

            # LOOKING pretty good! hits pi[100] = 1 exactly and v = -1 exactly!
            b = examples[0][0]
            pi, v = wrapper.predict(b)
            print(pi)


if __name__ == '__main__':
    unittest.main()