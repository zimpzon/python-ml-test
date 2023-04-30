namespace Tixy
{
    public static class BoardConsolePrintLarge
    {
        public static void Print(IBoard board, string waitMsg = null, bool clear = true)
        {
            if (clear)
                Console.Clear();

            var allPieces = board.GetPlayerPieces();

            int scale = 3; // Adjust this value to change font size

            Console.Write("  ");
            for (int x = 0; x < Board.W; x++)
                Console.Write(new string(' ', scale - 1) + (char)('A' + x) + " ");

            Console.WriteLine();
            for (int y = 0; y < Board.H; y++)
            {
                Console.Write(new string(' ', scale - 1) + (char)(y + '0' + 1) + " ");
                for (int x = 0; x < Board.W; x++)
                {
                    var piece = allPieces.FirstOrDefault(p => p.X == x && p.Y == y);
                    Console.Write(piece == null ? new string('.', scale) : new string(piece.Piece.Type, scale));
                    Console.Write(' ');
                }
                Console.WriteLine();
                for (int i = 0; i < scale - 1; i++)
                {
                    Console.Write(new string(' ', scale - 1) + " ");
                    for (int x = 0; x < Board.W; x++)
                    {
                        Console.Write(new string(' ', scale) + ' ');
                    }
                    Console.WriteLine();
                }
            }
            if (!string.IsNullOrEmpty(waitMsg))
            {
                Console.WriteLine(waitMsg);
                Console.ReadKey();
            }
        }
    }
}
