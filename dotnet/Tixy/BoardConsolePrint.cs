namespace Tixy
{
    public static class BoardConsolePrint
    {
        public static void Print(IBoard board, float[] moveScores = null)
        {
            var allPieces = board.GetPlayerPieces();
            
            Console.Write("  ");
            for (int x = 0; x < Board.W; x++)
                Console.Write((char)('A' + x));

            Console.WriteLine();
            for (int y = 0; y < Board.H; y++)
            {
                Console.Write((char)(Board.H - y + '0'));
                Console.Write(' ');
                for (int x = 0; x < Board.W; x++)
                {
                    char c = '.';
                    var piece = allPieces.FirstOrDefault(p => p.X == x && p.Y == y);
                    if (piece == null)
                    {
                        if (moveScores != null)
                        {
                            var idx = y * Board.W + x;
                            var score = moveScores[idx];
                            c = score >= 0 ? '+' : '-';
                        }
                    }
                    else
                    {
                        c = piece?.Piece?.Type ?? '?';
                    }
                    Console.Write(c);
                }
                Console.WriteLine();
            }
        }
    }
}
