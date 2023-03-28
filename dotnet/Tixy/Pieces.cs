namespace Tixy
{
    public static class Pieces
    {
        static Pieces()
        {
            CreatePieces();
        }

        public static Dictionary<char, Piece> PiecesTypes { get; } = new();

        private static void CreatePieces()
        {
            var pieceT1 = new Piece
            {
                Type = 'T',
                MoveFlags = new int[]
                {
                    1, 1, 1,
                    0, 0, 0,
                    0, 1, 0
                }
            };
            var pieceI1 = new Piece
            {
                Type = 'I',
                MoveFlags = new int[]
                {
                    0, 1, 0,
                    0, 0, 0,
                    0, 1, 0
                }
            };
            var pieceX1 = new Piece
            {
                Type = 'X',
                MoveFlags = new int[]
                {
                    1, 0, 1,
                    0, 0, 0,
                    1, 0, 1
                }
            };
            var pieceY1 = new Piece
            {
                Type = 'Y',
                MoveFlags = new int[]
                {
                    1, 0, 1,
                    0, 0, 0,
                    0, 1, 0
                }
            };

            var pieceT2 = new Piece
            {
                Type = 't',
                MoveFlags = new int[]
                {
                    0, 1, 0,
                    0, 0, 0,
                    1, 1, 1
    }
            };
            var pieceI2 = new Piece
            {
                Type = 'i',
                MoveFlags = new int[]
                {
                    0, 1, 0,
                    0, 0, 0,
                    0, 1, 0
                }
            };
            var pieceX2 = new Piece
            {
                Type = 'x',
                MoveFlags = new int[]
                {
                    1, 0, 1,
                    0, 0, 0,
                    1, 0, 1
                }
            };
            var pieceY2 = new Piece
            {
                Type = 'y',
                MoveFlags = new int[]
                {
                    0, 1, 0,
                    0, 0, 0,
                    1, 0, 1
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

            // Use the 3x3 allowed movement patterns to create a list of valid move deltas for each piece.
            var indices = new List<(int dx, int dy)>
            {
                (-1, -1),
                (0, -1),
                (1, -1),
                (-1, 0),
                (0, 0),
                (1, 0),
                (-1, 1),
                (0, 1),
                (1, 1),
            };

            foreach (var piece in PiecesTypes.Values)
            {
                for (int i = 0; i < piece.MoveFlags.Length; ++i)
                {
                    if (piece.MoveFlags[i] != 0)
                        piece.ValidDirections.Add(indices[i]);
                }
            }
        }
    }
}
