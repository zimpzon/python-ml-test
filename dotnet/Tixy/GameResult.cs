namespace Tixy
{
    // An episode/game returns this.
    // -> who will use this?
    public class GameResult
    {
        public class ScoredMove
        {
            public double DiscountedReward { get; set; }
            public float[] State { get; set; } = new float[Board.W * Board.H * 9]; // 8 possible moves + 1 plane for player 0 | 1
        }

        public string Moves { get; set; }
        public int WinnerId { get; set; }
        public double Reward { get; set; }
    }
}
