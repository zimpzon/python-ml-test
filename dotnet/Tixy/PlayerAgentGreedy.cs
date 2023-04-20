namespace Tixy
{
    public class PlayerAgentGreedy : IPlayerAgent
    {
        private IBoard _board;
        private int _playerId;
        private readonly Random _rnd = new ();

        public string Name => "Greedy";

        public void Reset(IBoard board, int playerId)
        {
            _board = board;
            _playerId = playerId;
        }

        public Move GetMove()
        {
            var myPieces = _board.GetPlayerPieces(_playerId);
            if (myPieces.Count == 0)
                throw new InvalidOperationException("I have no pieces");

            // If we have a piece that can capture, then do it.
            foreach (var piece in myPieces)
            {
                var validMoves = _board.GetPieceValidMoves(piece);
                foreach (var move in validMoves)
                {
                    var dstPiece = _board.GetPieceAt(move.X1, move.Y1);
                    if (dstPiece != null && dstPiece.OwnerId != _playerId)
                    {
                        return move;
                    }
                }
            }

            // Else pick random move                .
            int cnt = 0;
            while (true)
            {
                int rndPieceIdx = _rnd.Next(myPieces.Count);
                var selectedPiece = myPieces[rndPieceIdx];

                int rndDirectionIdx = _rnd.Next(selectedPiece.Piece.ValidDirections.Count);
                var (dx, dy) = selectedPiece.Piece.ValidDirections[rndDirectionIdx];

                var move = new Move
                {
                    X0 = selectedPiece.X,
                    Y0 = selectedPiece.Y,
                    Dx = dx,
                    Dy = dy,
                };

                if (!_board.IsValidMove(move, _playerId))
                {
                    if (cnt++ > 1000)
                        throw new InvalidOperationException("Failed to find a valid move after x attempts.");

                    continue;
                }
                
                return move;
            }
        }
    }
}
