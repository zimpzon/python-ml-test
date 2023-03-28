namespace Tixy
{
    public class PlayerAgentConsole : IPlayerAgent
    {
        private Board _board;
        private int _playerId;

        public void Reset(Board board, int playerId)
        {
            _board = board;
            _playerId = playerId;
        }

        public Move GetMove()
        {
            while (true)
            {
                Console.WriteLine();
                Console.Write("Enter move: ");
                string cmd = Console.ReadLine()?.ToUpper();

                if (!ParseMoveCommand(cmd, out var move))
                {
                    Console.WriteLine($"Bad command, format must be: A1 B2");
                    continue;
                }

                var movePiece = _board.GetPieceAt(move.X0, move.Y0, _playerId);
                if (movePiece == null)
                {
                    Console.WriteLine($"You don't have a piece there");
                    continue;
                }

                var destinationPiece = _board.GetPieceAt(move.X1, move.Y1, _playerId);
                if (destinationPiece != null)
                {
                    Console.WriteLine($"You already have a piece there");
                    continue;
                }

                if (!_board.IsValidMove(move, _playerId))
                {
                    Console.WriteLine($"Not a valid move");
                    continue;
                }
                
                return move;
            }
        }

        private bool ParseMoveCommand(string cmd, out Move move)
        {
            move = null;

            if (cmd == null || cmd?.Length < 5 || cmd?.Length > 5 || cmd?[2] != ' ')
                return false;

            var from = _board.FromStrPos(cmd[..2]);
            var to = _board.FromStrPos(cmd[3..5]);

            move = new Move
            {
                X0 = from.x,
                Y0 = from.y,
                Dx = to.x - from.x,
                Dy = to.y - from.y,
            };

            return true;
        }
    }
}
