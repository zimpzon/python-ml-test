namespace Tixy
{
    public static class Util
    {
        public static BoardState ExportBoard(IBoard board)
        {
            var result = new BoardState();

            var pieces = board.GetPlayerPieces();
            foreach(var piece in pieces)
            {
                int layerIdx = "TIXYtixy".IndexOf(piece.Piece.Type);
                int stateIdx = (piece.Y * Board.W) + piece.X + (layerIdx * Board.W * Board.H);
                result.State[stateIdx] = 1;
            }
            
            return result;
        }
        
        public static void StateToBoard(IBoard board, BoardState state)
        {
            for (int y = 0; y < Board.H; ++y)
            {
                for (int x = 0; x < Board.W; ++x)
                {
                    for (int layer = 0; layer < 8; ++layer)
                    {
                        int stateIdx = (y * Board.W) + x + (layer * Board.W * Board.H);
                        if (state.State[stateIdx] > 0)
                        {
                            char type = "TIXYtixy"[layer];
                            board.AddPiece(1, type, x, y);
                        }
                    }
                }
            }
        }
    }
}
