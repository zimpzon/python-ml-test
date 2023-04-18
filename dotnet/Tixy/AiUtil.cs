namespace Tixy
{
    public class PieceMoveScores
    {
        public ActivePiece Piece { get; set; }
        public List<float> MoveScores { get; set; }
    }

    public static class AiUtil
    {
        public static float[] Softmax(float[] values)
        {
            var maxVal = values.Max();
            var exp = values.Select(v => Math.Exp(v - maxVal));
            var sumExp = exp.Sum();

            return exp.Select(v => (float)(v / sumExp)).ToArray();
        }
        
        public static Move SelectMoveByProbability(IBoard board, List<PieceMoveScores> pieceMoveScores, int playerId, double epsilon)
        {
            var moveSelection = new List<(PieceMoveScores piece, int moveIdx, float score)>();

            // Find all valid moves
            for (int i = 0; i < pieceMoveScores.Count; ++i)
            {
                var piece = pieceMoveScores[i];

                for (int j = 0; j < piece.MoveScores.Count; ++j)
                {
                    int direction = j;
                    if (!board.IsValidMove(piece.Piece.X, piece.Piece.Y, direction, playerId))
                        continue;

                    moveSelection.Add((piece, j, piece.MoveScores[j]));
                }
            }

            if (!moveSelection.Any())
                throw new InvalidOperationException("No valid moves found");

            // Sort valid moves by score
            moveSelection.Sort((a, b) => a.score.CompareTo(b.score));

            // Select a move randomly or randomly weighted by probability (depending on random against epsilon)
            var rnd = new Random();
            var selected = moveSelection[0];
            
            bool useRandomMove = rnd.NextDouble() > epsilon;
            if (useRandomMove)
            {
                selected = moveSelection[rnd.Next(moveSelection.Count)];
            }
            else
            {
                double roll = rnd.NextDouble();

                for (int i = moveSelection.Count - 1; i >= 0; --i)
                {
                    var move = moveSelection[i];
                    if (roll <= move.score || i == 0)
                    {
                        selected = move;
                        break;
                    }
                }
            }

            selected = moveSelection.Last();
            
            (int dx, int dy) = board.DeltasFromDirection(selected.moveIdx);
            return new Move
            {
                X0 = selected.piece.Piece.X,
                Y0 = selected.piece.Piece.Y,
                Dx = dx,
                Dy = dy,
            };
        }

        public static Move GetHighestScoringMove(IBoard board, List<PieceMoveScores> pieceMoveScores, int playerId)
        {
            PieceMoveScores bestPiece = null;
            float bestScore = float.MinValue;
            int bestIdx = -1;

            for (int i = 0; i < pieceMoveScores.Count; ++i)
            {
                var piece = pieceMoveScores[i];

                float pieceBestScore = float.MinValue;
                int idx = -1;
                for (int j = 0; j < piece.MoveScores.Count; ++j)
                {
                    int direction = j;
                    if (!board.IsValidMove(piece.Piece.X, piece.Piece.Y, direction, playerId))
                        continue;

                    if (piece.MoveScores[j] > pieceBestScore)
                    {
                        idx = j;
                        pieceBestScore = piece.MoveScores[j];
                    }
                }

                if (pieceBestScore > bestScore)
                {
                    bestScore = pieceBestScore;
                    bestIdx = idx;
                    bestPiece = piece;
                }
            }

            if (bestPiece == null)
                throw new InvalidOperationException("No best move found");

            (int dx, int dy) = board.DeltasFromDirection(bestIdx);
            return new Move
            {
                X0 = bestPiece.Piece.X,
                Y0 = bestPiece.Piece.Y,
                Dx = dx,
                Dy = dy,
            };
        }

        public static List<PieceMoveScores> GetPiecesRawScores(IBoard board, float[] aiOutput, int playerId)
        {
            if (aiOutput.Length != Board.CellCount * 8)
                throw new InvalidOperationException($"AI output array must have length Board.W * Board.H * 8 ({Board.CellCount * 8}) but has length {aiOutput.Length}");

            var result = new List<PieceMoveScores>();
            
            for (int y = 0; y < Board.H; ++y)
            {
                for (int x = 0; x < Board.W; ++x)
                {
                    var myPiece = board.GetPieceAt(x, y, playerId);
                    if (myPiece == null)
                        continue;

                    List<float> moveScores = new List<float>();

                    // planes are move directions. Imagine drilling down from the top. B2 = (1, 2) = idx 6 (1 + 5).
                    // Move scores are then 6 + (layerIdx) * 25
                    // ex. 136 = B2, dir 5 = down. 
                    int pieceBoardIdx = myPiece.Y * 5 + myPiece.X;

                    for (int layerIdx = 0; layerIdx < 8; ++layerIdx)
                    {
                        int idx = layerIdx * 25 + pieceBoardIdx;
                        moveScores.Add(aiOutput[idx]);
                    }

                    var pieceMoveScores = new PieceMoveScores
                    {
                        Piece = myPiece,
                        MoveScores = moveScores,
                    };
                    result.Add(pieceMoveScores);
                }
            }

            return result;
        }
    }
}
