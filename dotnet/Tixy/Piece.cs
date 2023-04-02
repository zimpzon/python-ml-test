namespace Tixy
{
    public class Piece
    {
        public int TypeToIdx()
            => "TIXYtixy".IndexOf(Type);
        
        public char Type { get; set; }
        public int[] MoveFlags { get; set; } = new int[3 * 3];
        public List<(int dx, int dy)> ValidDirections { get; set; } = new ();
    }
}
