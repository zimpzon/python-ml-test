namespace Tixy
{
    public class BoardState
    {
        public float[] State { get; } = new float[9 * Board.W * Board.H]; // player [0 | 1] + 8 pieces = 9
        public float[] SelectedMove { get; } = new float[8 * Board.W * Board.H]; // 8 possible moves per cell
        public float Value { get; set; } // -1, 0 or 1 for lose, draw, win
    };
}
