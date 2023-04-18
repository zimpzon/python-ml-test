namespace Tixy
{
    public class BoardState
    {
        public int PlayerIdx { get; set; }
        public float[] State { get; } = new float[8 * Board.W * Board.H]; // 8 pieces
        public long[] SelectedMove { get; } = new long[Board.W * Board.H * 8]; // 8 possible moves per cell
        public long SelectedMoveIdx { get; set; }
        public long SelectedDirection { get; set; }
        public double Value { get; set; }

        public override string ToString()
        {
            return $"Player: {PlayerIdx}, value: {Value}, selectedIdx: {SelectedMoveIdx}";
        }
    }
}
