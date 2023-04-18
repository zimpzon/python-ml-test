using System.Text;

namespace Tixy
{
    public class Board : IBoard
    {
        public static int W { get; } = 5;
        public static int H { get; } = 5;
        public static int CellCount { get; } = W * H;

        public bool IsGameOver { get; private set; }
        public int WinnerId { get; private set; }
        
        public List<ActivePiece> PlayerPieces { get; } = new();

        public List<BoardState> Moves { get; private set; } = new();
        public List<(int x, int y)> DebugPos { get; } = new();
        public List<long> DebugDiretionIdx { get; } = new();

        private readonly StringBuilder _sb = new (1000);

        public Board()
        {
            Reset();
        }

        public void Reset()
        {
            IsGameOver = false;
            WinnerId = 0;
            Moves.Clear();
            _sb.Clear();
            Init();
        }

        // Need starting state, game score, move list with discounted score per move, pair-wise moves for training
        public string GetStoredMoves()
            => _sb.ToString();

        public static int IdOtherPlayer(int myId)
            => myId == 1 ? 2 : 1;

        public List<Move> GetPieceDirections(ActivePiece activePiece)
            => activePiece.Piece.ValidDirections.Select(vm => new Move { X0 = activePiece.X, Y0 = activePiece.Y, Dx = vm.dx, Dy = vm.dy }).ToList();

        public List<Move> GetPieceValidMoves(ActivePiece activePiece)
            => GetPieceDirections(activePiece).Where(m => IsValidMove(m, activePiece.OwnerId)).ToList();

        public List<ActivePiece> GetPlayerPieces(int? playerId = null)
            => PlayerPieces.Where(pp => playerId == null || pp.OwnerId == playerId).ToList();

        public ActivePiece GetPieceAt(int x, int y, int? playerId = null)
            => PlayerPieces.FirstOrDefault(p => p.X == x && p.Y == y && (playerId == null || p.OwnerId == playerId));

        public ActivePiece AddPiece(int playerId, char type, string pos)
        {
            (int x, int y) = FromStrPos(pos);
            return AddPiece(playerId, type, x, y);
        }

        public ActivePiece AddPiece(int playerId, char type, int x, int y)
        {
            var newPiece = new ActivePiece { OwnerId = playerId, X = x, Y = y, Piece = Pieces.PiecesTypes[type] };
            PlayerPieces.Add(newPiece);
            return newPiece;
        }

        public void Init()
        {
            PlayerPieces.Clear();

            //AddPiece(1, 'T', "A1");
            AddPiece(1, 'i', "B1");
            //AddPiece(1, 'X', "C1");
            //AddPiece(1, 'Y', "D1");

            //AddPiece(2, 't', "A5");
            AddPiece(2, 'I', "D5");
            //AddPiece(2, 'x', "C5");
            //AddPiece(2, 'y', "D5");
        }

        public bool IsValidMove(int x, int y, int dir, int playerId)
        {
            (int dx, int dy) = directionLookup.First(pair => pair.Value == dir).Key;

            int x1 = x + dx;
            int y1 = y + dy;

            if (x < 0 || y < 0 || x >= W || y >= H || x1 < 0 || y1 < 0 || x1 >= W || y1 >= H)
                return false;

            var playerPiece = GetPieceAt(x, y, playerId);
            if (playerPiece == null)
                return false;

            var directionIsValid = playerPiece.Piece.ValidDirections.Any(vm => vm.dx == dx && vm.dy == dy);
            if (!directionIsValid)
                return false;

            var destinationPiece = GetPieceAt(x1, y1);
            if (destinationPiece?.OwnerId == playerId)
                return false;

            return true;
        }

        public bool IsValidMove(Move move, int playerId)
        {
            if (move == null || move.X0 < 0 || move.Y0 < 0 || move.X0 >= W || move.Y0 >= H || move.X1 < 0 || move.Y1 < 0 || move.X1 >= W || move.Y1 >= H)
                return false;

            var playerPiece = GetPieceAt(move.X0, move.Y0, playerId);
            if (playerPiece == null)
                return false;

            var directionIsValid = playerPiece.Piece.ValidDirections.Any(vm => vm.dx == move.Dx && vm.dy == move.Dy);
            if (!directionIsValid)
                return false;

            var destinationPiece = GetPieceAt(move.X1, move.Y1);
            if (destinationPiece?.OwnerId == playerId)
                return false;
            
            return true;
        }

        public (int dx, int dy) DeltasFromDirection(int direction)
        {
            return directionLookup.First(pair => pair.Value == direction).Key;
        }

        public void Move(Move move, int playerId)
        {
            if (!IsValidMove(move, playerId))
                throw new InvalidOperationException(nameof(move));

            var movedPiece = GetPieceAt(move.X0, move.Y0, playerId);
            var takenPiece = GetPieceAt(move.X1, move.Y1);

            StoreState(move, playerId);

            movedPiece.X += move.Dx;
            movedPiece.Y += move.Dy;

            if (takenPiece != null)
                PlayerPieces.Remove(takenPiece);

            int winLineIdx = playerId == 1 ? H - 1 : 0;
            bool winByLine = movedPiece.Y == winLineIdx;

            var otherPlayersPieces = PlayerPieces.Where(pp => pp.OwnerId != playerId).ToList();
            bool winByElimination = !otherPlayersPieces.Any();

            if (Moves.Count > 100)
            {
                var rnd = new Random();
                WinnerId = rnd.NextDouble() > 0.5 ? 1 : 2;
                IsGameOver = true;
                //Console.WriteLine($"Too many turns ({Moves.Count}), winner determined randomly: player {WinnerId}");

                // Skip rounds that didn't conclude.
                Moves.Clear();
            }

            if (winByLine || winByElimination)
            {
                //Console.WriteLine($"elim: {winByElimination}, line: {winByLine}");
                WinnerId = playerId;
                IsGameOver = true;

                PostProcessMoves();
            }
        }

        private void PostProcessMoves()
        {
            int loserId = IdOtherPlayer(WinnerId);

            // Only export playe 1 moves, to simplify.
            Moves = Moves.Where(m => m.PlayerIdx == 0).ToList();
            // Discount earlier moves, then normalize (per episode, could also have been per batch).
            var winnerMoves = Moves.Where(m => m.PlayerIdx == WinnerId - 1).ToList();
            var loserMoves = Moves.Where(m => m.PlayerIdx == loserId - 1).ToList();
            DiscountAndNormalize(winnerMoves, 1);
            DiscountAndNormalize(loserMoves, -1);
            //Moves = new List<BoardState>(winnerMoves);
        }

        private static double StdDev(List<BoardState> moves, double mean)
        {
            double variance = moves.Select(x => Math.Pow(x.Value - mean, 2)).Average();
            return Math.Sqrt(variance);
        }

        private void DiscountAndNormalize(List<BoardState> moves, double reward)
        {
            const double DiscountFactor = 1.0;
            double v = reward;
            for (int i = moves.Count - 1; i >= 0; --i)
            {
                var m = moves[i];
                m.Value = v;
                v *= DiscountFactor;
            }

            // NB: skip stdDev for now, it results in 0 when all moves are the same (no discounting). We are already in the nice -1..1 range.
            //double mean = moves.Average(m => m.Value);
            //double stdDev = StdDev(moves, mean) + double.MinValue;

            //for (int i = 0; i < moves.Count; ++i)
            //{
            //    var m = moves[i];
            //    m.Value -= mean;
            //    m.Value /= stdDev;
            //}
        }

        public static (int x, int y) FromStrPos(string p)
        {
            int x = p[0] - 'A';
            int y = p[1] - '1';
            return (x, y);
        }

        Dictionary<(int dx, int dy), int> directionLookup = new Dictionary<(int dx, int dy), int>
        {
            { (-1, -1), 0 },
            { (0, -1), 1 },
            { (1, -1), 2 },
            { (1, 0), 3 },
            { (1, 1), 4 },
            { (0, 1), 5 },
            { (-1, 1), 6 },
            { (-1, 0), 7 }
        };

        private void StoreState(Move m, int playerId)
        {
            var state = Util.ExportBoard(this, playerId);
            int dx = m.Dx;
            int dy = m.Dy;
            state.SelectedDirection = directionLookup[(dx, dy)];

            // planeidx equals moveDir
            long moveDstIdx = (state.SelectedDirection * W * H) + m.Y0 * W + m.X0; // plane + pos
            state.SelectedMove[moveDstIdx] = 1;
            state.SelectedMoveIdx = moveDstIdx;

            Moves.Add(state);
            DebugDiretionIdx.Add(state.SelectedDirection);
            DebugPos.Add((m.X0, m.Y0));
            
            _sb.Append((char)(m.X0 + 'A'));
            _sb.Append((char)(H - m.Y0 + '0'));
            _sb.Append((char)(m.X1 + 'A'));
            _sb.Append((char)(H - m.Y1 + '0'));
        }
    }
}
