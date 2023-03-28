namespace Tixy
{
    public class Move
    {
        public int X0 { get; set; }
        public int Y0 { get; set; }
        public int Dx { get; set; }
        public int Dy { get; set; }

        public int X1 => X0 + Dx;
        public int Y1 => Y0 + Dy;
    };
}
