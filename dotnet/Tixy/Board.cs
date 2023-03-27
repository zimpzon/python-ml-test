namespace Tixy
{
    // Minimum information required:
    // w, h
    // piece types with valid moves
    // pieces per player

    // who's turn is it
    // game rules
    public class Board : IBoard
    {
        static Board()
        {
            CreatePieces();
        }

        public class Piece
        {
            public char Type { get; set; }
            public int[,] ValidMoves { get; set; } = new int[3, 3];
        }

        public class PlayerPiece
        {
            public int OwnerId { get; set; }
            public Piece? Piece { get; set; }
            public int X { get; set; }
            public int Y { get; set; }
        }

        public class BoardMove
        {
            public int X { get; set; }
            public int Y { get; set; }
            public int Dx { get; set; }
            public int Dy { get; set; }
        };

        public static readonly Dictionary<char, Piece> PiecesTypes = new ();

        public List<PlayerPiece> PlayerPieces = new ();
        private readonly int _w;
        private readonly int _h;

        public Board(int w, int h)
        {
            _w = w;
            _h = h;

            CreatePieces();
            Init();
        }

        public List<BoardMove> GetValidMoves(PlayerPiece piece)
        {
            return new List<BoardMove>();
        }

        public bool ParseMoveCommand(string? cmd, out BoardMove? move)
        {
            move = null;

            if (cmd == null || cmd?.Length < 5 || cmd?.Length > 5 || cmd?[2] != ' ')
                return false;

            int x0 = cmd[0] - 'A';
            int y0 = _h - (cmd[1] - '1') - 1;

            int x1 = cmd[3] - 'A';
            int y1 = _h - (cmd[4] - '1') - 1;

            int dx = x1 - x0;
            int dy = y1 - y0;

            move = new BoardMove
            {
                X = x0,
                Y = y0,
                Dx = dx,
                Dy = dy
            };

            return true;           
        }

        public void Print()
        {
            Console.Write("  ");
            for (int x = 0; x < _w; x++)
                Console.Write((char)('A' + x));

            Console.WriteLine();
            for (int y = 0; y < _h; y++)
            {
                Console.Write((char)((_h - y) + '0'));
                Console.Write(' ');
                for (int x = 0; x < _w; x++)
                {
                    var piece = PlayerPieces.Where(p => p.X == x && p.Y == y).FirstOrDefault();
                    Console.Write(piece?.Piece?.Type ?? '.');
                }
                Console.WriteLine();
            }
        }

        public bool IsValidMove(BoardMove? move)
        {
            if (move == null ||move.X < 0 || move.Y < 0 || move.X >= _w || move.Y >= _h)
                return false;

            int newX = move.X + move.Dx;
            int newY = move.Y + move.Dy;

            if (newX < 0 || newY < 0 || newX >= _w || newY >= _h)
                return false;

            return true;
        }

        public void Move(BoardMove? move)
        {
            if (move == null || !IsValidMove(move))
                throw new InvalidOperationException(nameof(move));

            int newX = move.X + move.Dx;
            int newY = move.Y + move.Dy;

            var playerPiece = PlayerPieces.Where(p => p.X == move.X && p.Y == move.Y).FirstOrDefault();
            if (playerPiece == null)
                throw new InvalidOperationException(nameof(move));

            playerPiece.X += move.Dx;
            playerPiece.Y += move.Dy;
        }

        public void Init()
        {
            PlayerPieces.Clear();

            PlayerPieces.Add(new PlayerPiece { OwnerId = 1, X = 0, Y = _h - 1, Piece = PiecesTypes['Y'] });
            PlayerPieces.Add(new PlayerPiece { OwnerId = 1, X = 1, Y = _h - 1, Piece = PiecesTypes['I'] });
            PlayerPieces.Add(new PlayerPiece { OwnerId = 1, X = 2, Y = _h - 1, Piece = PiecesTypes['X'] });
            PlayerPieces.Add(new PlayerPiece { OwnerId = 1, X = 3, Y = _h - 1, Piece = PiecesTypes['I'] });
            PlayerPieces.Add(new PlayerPiece { OwnerId = 1, X = 4, Y = _h - 1, Piece = PiecesTypes['Y'] });

            PlayerPieces.Add(new PlayerPiece { OwnerId = 2, X = 0, Y = 0, Piece = PiecesTypes['Y'] });
            PlayerPieces.Add(new PlayerPiece { OwnerId = 2, X = 1, Y = 0, Piece = PiecesTypes['I'] });
            PlayerPieces.Add(new PlayerPiece { OwnerId = 2, X = 2, Y = 0, Piece = PiecesTypes['X'] });
            PlayerPieces.Add(new PlayerPiece { OwnerId = 2, X = 3, Y = 0, Piece = PiecesTypes['I'] });
            PlayerPieces.Add(new PlayerPiece { OwnerId = 2, X = 4, Y = 0, Piece = PiecesTypes['Y'] });
        }

        private static void CreatePieces()
        {
            var pieceT1 = new Piece
            {
                Type = 'T',
                ValidMoves = new int[,]
                {
                    { 1, 1, 1 },
                    { 0, 0, 0 },
                    { 0, 1, 0 }
                }
            };
            var pieceI1 = new Piece
            {
                Type = 'I',
                ValidMoves = new int[,]
                {
                    { 0, 1, 0 },
                    { 0, 0, 0 },
                    { 0, 1, 0 }
                }
            };
            var pieceX1 = new Piece
            {
                Type = 'X',
                ValidMoves = new int[,]
                {
                    { 1, 0, 1 },
                    { 0, 0, 0 },
                    { 1, 0, 1 }
                }
            };
            var pieceY1 = new Piece
            {
                Type = 'Y',
                ValidMoves = new int[,]
                {
                    { 1, 0, 1 },
                    { 0, 0, 0 },
                    { 0, 1, 0 }
                }
            };

            var pieceT2 = new Piece
            {
                Type = 't',
                ValidMoves = new int[,]
    {
                    { 0, 1, 0 },
                    { 0, 0, 0 },
                    { 1, 1, 1 }
    }
            };
            var pieceI2 = new Piece
            {
                Type = 'i',
                ValidMoves = new int[,]
                {
                    { 0, 1, 0 },
                    { 0, 0, 0 },
                    { 0, 1, 0 }
                }
            };
            var pieceX2 = new Piece
            {
                Type = 'x',
                ValidMoves = new int[,]
                {
                    { 1, 0, 1 },
                    { 0, 0, 0 },
                    { 1, 0, 1 }
                }
            };
            var pieceY2 = new Piece
            {
                Type = 'y',
                ValidMoves = new int[,]
                {
                    { 0, 1, 0 },
                    { 0, 0, 0 },
                    { 1, 0, 1 }
                }
            };


            PiecesTypes[pieceT1.Type] = pieceT1;
            PiecesTypes[pieceI1.Type] = pieceI1;
            PiecesTypes[pieceX1.Type] = pieceX1;
            PiecesTypes[pieceY1.Type] = pieceY1;

            PiecesTypes[pieceT2.Type] = pieceT2;
            PiecesTypes[pieceI2.Type] = pieceI2;
            PiecesTypes[pieceX2.Type] = pieceX2;
            PiecesTypes[pieceY2.Type] = pieceY2;
        }
    }
}
