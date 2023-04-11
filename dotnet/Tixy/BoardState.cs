namespace Tixy
{
    public class BoardState
    {
        public int PlayerIdx { get; set; }
        public float[] State { get; } = new float[9 * Board.W * Board.H]; // player [0 | 1] + 8 pieces = 9
        public long[] SelectedMove { get; } = new long[Board.W * Board.H * 8]; // 8 possible moves per cell
        public long SelectedMoveIdx { get; set; }
        public double Value { get; set; }

        public override string ToString()
        {
            return $"Player: {PlayerIdx}, value: {Value}, selectedIdx: {SelectedMoveIdx}";
        }
    }
}
