namespace Tixy
{
    public static class BoardConsolePrint
    {
        public static void Print(Board board)
        {
            var allPieces = board.GetActivePieces();
            
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
                    var piece = allPieces.FirstOrDefault(p => p.X == x && p.Y == y);
                    Console.Write(piece?.Piece?.Type ?? '.');
                }
                Console.WriteLine();
            }
        }
    }
}
