namespace Tixy
{
    public class BoardState
    {
        public float[] State { get; } = new float[Board.W * Board.H * 8];
        public float[] DesiredDirections { get; } = new float[8];
        public int BestDirection { get; set; }
    };
}
