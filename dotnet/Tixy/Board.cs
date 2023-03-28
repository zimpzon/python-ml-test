namespace Tixy
{
    public class Board : IBoard
    {
        public int W { get; }
        public int H { get; }
        public bool IsGameOver { get; private set; }
        public int WinnerId { get; private set; }

        private readonly List<ActivePiece> _playerPieces = new();

        public Board(int w, int h)
        {
            W = w;
            H = h;
            Init();
        }

        public static int IdOtherPlayer(int myId)
            => myId == 1 ? 2 : 1;
        
        public static List<Move> GetValidMoves(ActivePiece activePiece)
            => activePiece.Piece.ValidDirections.Select(vm => new Move { X0 = activePiece.X, Y0 = activePiece.Y, Dx = vm.dx, Dy = vm.dy }).ToList();

        public List<ActivePiece> GetActivePieces(int? playerId = null)
            => _playerPieces.Where(pp => playerId == null || pp.OwnerId == playerId).ToList();

        public ActivePiece GetPieceAt(int x, int y, int? playerId = null)
            => _playerPieces.FirstOrDefault(p => p.X == x && p.Y == y && (playerId == null || p.OwnerId == playerId));

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

        public void Move(Move move, int playerId)
        {
            if (!IsValidMove(move, playerId))
                throw new InvalidOperationException(nameof(move));

            var movedPiece = GetPieceAt(move.X0, move.Y0, playerId);
            var takenPiece = GetPieceAt(move.X1, move.Y1);
            if (takenPiece != null)
            {
                _playerPieces.Remove(takenPiece);

                var otherPlayersPieces = _playerPieces.Where(pp => pp.OwnerId != playerId).ToList();
                if (!otherPlayersPieces.Any())
                {
                    WinnerId = playerId;
                    IsGameOver = true;
                }
            }
            
            movedPiece.X += move.Dx;
            movedPiece.Y += move.Dy;
        }

        public (int x, int y) FromStrPos(string p)
        {
            int x = p[0] - 'A';
            int y = H - (p[1] - '1') - 1;
            return (x, y);
        }

        public void Init()
        {
            _playerPieces.Clear();

            void Add(int playerId, char type, string pos)
            {
                (int x, int y) = FromStrPos(pos);
                _playerPieces.Add(new ActivePiece { OwnerId = playerId, X = x, Y = y, Piece = Pieces.PiecesTypes[type] });
            }

            Add(1, 'T', "A1");
            Add(1, 'I', "B1");
            Add(1, 'X', "C1");
            Add(1, 'Y', "D1");

            Add(2, 't', "A5");
            Add(2, 'i', "B5");
            Add(2, 'x', "C5");
            Add(2, 'y', "D5");
        }
    }
}
