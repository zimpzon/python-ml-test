import sys
from board import Board

board = Board(6, 5)

while True:
    board.print()
    cmd = input("Command: ").upper()

    success, move = board.parse_move_command(cmd)
    if not success:
        print("Bad command, format must be: A1 B2")
        continue

    if not board.is_valid_move(move):
        print("Not a valid move")
        continue

    board.move(move)
    print()
