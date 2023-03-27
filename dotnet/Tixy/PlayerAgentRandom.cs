namespace Tixy
{
    public class PlayerAgentRandom : IPlayerAgent
    {
        private Board _board;
        private int _playerId;
        private Random _rnd = new Random(42);

        public void Init(Board board, int playerId)
        {
            _board = board;
            _playerId = playerId;
        }

        public void TakeTurn()
        {
            var myPieces = _board.PlayerPieces.Where(pp => pp.OwnerId == _playerId).ToList();
            if (myPieces.Count == 0)
                return;

            var piece = myPieces[_rnd.Next(myPieces.Count)];
            var moves = _board.GetValidMoves(piece);
        }
    }
}
