namespace Tixy
{
    internal static class Explore
    {
        public static double Play(IPlayerAgent player1, IPlayerAgent player2)
        {
            int[] wins = new int[2];

            for (int i = 0; i < 100; ++i)
            {
                var board = new Board();
                board.Reset();

                player1.Reset(board, playerId: 1);
                player2.Reset(board, playerId: 2);

                while (true)
                {
                    var movePlayer1 = player1.GetMove();
                    board.Move(movePlayer1, 1);

                    if (board.IsGameOver)
                        break;

                    var movePlayer2 = player2.GetMove();
                    board.Move(movePlayer2, 2);

                    if (board.IsGameOver)
                        break;
                }

                wins[board.WinnerId - 1]++;
            }

            double player2Win = (double)wins[1] / (wins[0] + wins[1]);
            return player2Win;
        }
    }
}
