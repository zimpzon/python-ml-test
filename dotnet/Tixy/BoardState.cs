namespace Tixy
{
    public class BoardState
    {
        public int PlayerIdx { get; set; }
        public float[] State { get; } = new float[9 * Board.W * Board.H]; // player [0 | 1] + 8 pieces = 9
        public long[] SelectedMove { get; } = new long[Board.W * Board.H]; // 8 possible moves per cell
        public long SelectedMoveIdx { get; set; }
        public float Value { get; set; } // -1, 0 or 1 for lose, draw, win
    };
}
