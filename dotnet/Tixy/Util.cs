namespace Tixy
{
    public static class Util
    {
        // A plane is size 5 x 5. This is repeated 1 + 8 times.
        public static BoardState ExportBoard(IBoard board, int playerIdHasTurn)
        {
            var result = new BoardState();
            result.PlayerIdx = playerIdHasTurn - 1;
            var state = result.State;

            int planeSize = Board.W * Board.H;
            int idx = 0;

            // Fill player plane
            for (int i = 0; i < planeSize; ++i)
            {
                state[idx] = result.PlayerIdx == 0 ? 0 : 1;
                idx++;
            }

            // Fill plane 1 - 8 for pieces
            for (int pieceIdx = 0; pieceIdx < 8; ++pieceIdx)
            {
                for (int y = 0; y < Board.H; ++y)
                {
                    for (int x = 0; x < Board.W; ++x)
                    {
                        var piece = board.GetPieceAt(x, y);
                        if (piece?.Piece?.TypeToIdx() == pieceIdx)
                        {
                            state[idx] = 1;
                        }
                        
                        idx++;
                    }
                }
            }

            // Now we have filled 9 planes, each the size of the board.
            return result;
        }
    }
}
