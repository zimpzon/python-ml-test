import unittest
import TixyGame

class MyTest(unittest.TestCase):
   
        def test(self):
            game = TixyGame.TixyGame(3, 5)
            board = game.getInitBoard()

            print('initial board:')
            print(game.stringRepresentation(board))

            print('canonical form, p1:')
            print(game.stringRepresentation(game.getCanonicalForm(board, 1)))

            print('canonical form, p2:')
            print(game.stringRepresentation(game.getCanonicalForm(board, -1)))

            print('board size: '+ str(game.getBoardSize()))
            print('action size: '+ str(game.getActionSize()))

            valid_moves = game.getValidMoves(board, 1)
            print('valid moves: '+ str(valid_moves))

            # expected values for moves: plane repeated w * h times, so plane 0 is the first 15, plane 3 if 60..75
            
            plane1 = valid_moves[:15]
            print('plane 1: '+ str(plane1))
            plane3 = valid_moves[60:75]
            print('plane 3: '+ str(plane3))

if __name__ == '__main__':
    unittest.main()