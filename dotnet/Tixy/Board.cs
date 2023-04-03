using System.Text;

namespace Tixy
{
    public class Board : IBoard
    {
        public static int W { get; } = 5;
        public static int H { get; } = 5;
        public static int Cells { get; } = W * H;

        public bool IsGameOver { get; private set; }
        public int WinnerId { get; private set; }
        
        public List<ActivePiece> PlayerPieces { get; } = new();

        public List<BoardState> Moves { get; } = new();
        public List<(int x, int y)> DebugPos { get; } = new();
        public List<int> DebugDiretionIdx { get; } = new();

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

            AddPiece(1, 'T', "A1");
            AddPiece(1, 'I', "B1");
            AddPiece(1, 'X', "C1");
            AddPiece(1, 'Y', "D1");

            AddPiece(2, 't', "A5");
            AddPiece(2, 'i', "B5");
            AddPiece(2, 'x', "C5");
            AddPiece(2, 'y', "D5");
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

            int winLineIdx = playerId == 1 ? 0 : H - 1;
            bool winByLine = movedPiece.Y == winLineIdx;

            var otherPlayersPieces = PlayerPieces.Where(pp => pp.OwnerId != playerId).ToList();
            bool winByElimination = !otherPlayersPieces.Any();
            if (winByLine || winByElimination)
            {
                //Console.WriteLine($"elim: {winByElimination}, line: {winByLine}");
                WinnerId = playerId;
                int loserId = IdOtherPlayer(WinnerId);
                IsGameOver = true;

                var winnerMoves = Moves.Where(m => m.PlayerIdx == WinnerId - 1).ToList();
                foreach (var winnerMove in winnerMoves)
                    winnerMove.Value = 1;

                var loserMoves = Moves.Where(m => m.PlayerIdx == loserId - 1).ToList();
                foreach (var loserMove in loserMoves)
                    loserMove.Value = -1;
            }
        }

        public static (int x, int y) FromStrPos(string p)
        {
            int x = p[0] - 'A';
            int y = H - (p[1] - '1') - 1;
            return (x, y);
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
        };

        private void StoreState(Move m, int playerId)
        {
            var state = Util.ExportBoard(this, playerId);
            int dx = m.Dx;
            int dy = m.Dy;
            int selectedDirectionPlaneIdx = directionLookup[(dx, dy)];

            // Set a single 1 on the direction plane of the cell we move from.
            int moveDstIdx = (selectedDirectionPlaneIdx * W * H) + m.Y1 * W + m.X1;
            state.SelectedMove[moveDstIdx] = 1;

            Moves.Add(state);
            DebugDiretionIdx.Add(selectedDirectionPlaneIdx);
            DebugPos.Add((m.X0, m.Y0));
            
            _sb.Append((char)(m.X0 + 'A'));
            _sb.Append((char)(H - m.Y0 + '0'));
            _sb.Append((char)(m.X1 + 'A'));
            _sb.Append((char)(H - m.Y1 + '0'));
        }
    }
}
