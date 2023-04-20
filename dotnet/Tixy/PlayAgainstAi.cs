namespace Tixy
{
    internal static class PlayAgainstAi
    {
        public static void Play()
        {
            while (true)
            {
                var player1 = new PlayerAgentOnnx();
                var player2 = new PlayerAgentRandom();

                var board = new Board();
                board.Reset();

                player1.Reset(board, playerId: 1);
                player2.Reset(board, playerId: 2);

                while (true)
                {
                    BoardConsolePrint.Print(board, "\nAI move.", clear: false);

                    var movePlayer1 = player1.GetMove();
                    board.Move(movePlayer1, 1);

                    if (board.IsGameOver)
                        break;

                    BoardConsolePrint.Print(board, "\nYour move.", clear: false);

                    var movePlayer2 = player2.GetMove();
                    board.Move(movePlayer2, 2);

                    if (board.IsGameOver)
                        break;
                }

                Console.WriteLine($"\nGame over, winner: {(board.WinnerId == 1 ? player1.Name : player2.Name)}");
                Console.ReadLine();
            }
        }
    }
}
