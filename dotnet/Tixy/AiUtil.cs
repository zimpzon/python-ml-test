namespace Tixy
{
    public class PieceMoveScores
    {
        public ActivePiece Piece { get; set; }
        public List<float> MoveScores { get; set; }
    }

    public static class AiUtil
    {
        public static Move GetHighestScoringMove(IBoard board, List<PieceMoveScores> pieceMoveScores, int playerId)
        {
            PieceMoveScores bestPiece = null;
            float bestScore = float.MinValue;
            int bestIdx = -1;

            // Algo:
            // foreach piece
            //  foreach move
            //    if valid move and best so far:
            //      store which piece + which more
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

                    for (int layerIdx = 0; layerIdx < 8; ++layerIdx)
                    {
                        int idx = layerIdx * Board.CellCount + (y * Board.W) + x;
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
