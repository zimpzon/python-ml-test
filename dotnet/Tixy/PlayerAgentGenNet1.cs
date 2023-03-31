using System.Text.Json;

namespace Tixy
{
    public class PlayerAgentGen1 : IPlayerAgent
    {
        private IBoard _board;
        private int _playerId;
        private readonly Random _rnd = new (42);

        public void Reset(IBoard board, int playerId)
        {
            _board = board;
            _playerId = playerId;
        }

        Dictionary<(int, int), int> directionLookup = new Dictionary<(int, int), int>
        {
            { (-1, -1), 5 },
            { (0, -1), 4 },
            { (1, -1), 3 },
            { (1, 0), 2 },
            { (1, 1), 1 },
            { (0, 1), 0 },
            { (-1, 1), 7 },
            { (-1, 0), 6 }
            
        }; public Move GetMove()
        {
            List<BoardState> states = new ();
            for (int i = 0; i < 100000; ++i)
            {
                _board.PlayerPieces.Clear();

                char randomType = "tixy"[_rnd.Next(0, 4)];
                int x = _rnd.Next(0, Board.W);
                int y = _rnd.Next(0, Board.H);
                var p = _board.AddPiece(_playerId, randomType, x, y);
                var moves = _board.GetPieceValidMoves(p);
                var moveToOpponent = moves[_rnd.Next(0, moves.Count)];

                // Start with 'i', expand to random opponent.
                _board.AddPiece(Board.IdOtherPlayer(_playerId), 'i', moveToOpponent.X1, moveToOpponent.Y1);
                
                var state = Util.ExportBoard(_board);
                int dx = moveToOpponent.Dx;
                int dy = moveToOpponent.Dy;
                state.BestDirection = directionLookup[(dx, dy)];
                state.DesiredDirections[state.BestDirection] = 1;
                states.Add(state);
                    
                _board.PlayerPieces.Clear();

                Util.StateToBoard(_board, state);
            }

            File.WriteAllText("c:\\temp\\ml\\gen1.json", JsonSerializer.Serialize(states, new JsonSerializerOptions { WriteIndented = true }));

            return null;
        }
    }
}
