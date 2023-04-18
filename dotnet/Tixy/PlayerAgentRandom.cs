namespace Tixy
{
    public class PlayerAgentRandom : IPlayerAgent
    {
        private IBoard _board;
        private int _playerId;
        private readonly Random _rnd = new ();

        public string Name => "Random";

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

            //if (_playerId == 1)
            //    return new Move
            //    {
            //        X0 = myPieces[0].X,
            //        Y0 = myPieces[0].Y,
            //        Dx = 0,
            //        Dy = 1,
            //    };
            
            int cnt = 0;
            while (true)
            {
                // Really want options ordered randomly, or removing the option once it's been tested.
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
                        throw new InvalidOperationException("Failed to find a valid move after 100 attempts.");

                    continue;
                }
                
                return move;
            }
        }
    }
}
