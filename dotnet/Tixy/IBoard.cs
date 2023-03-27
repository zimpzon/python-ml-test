using static Tixy.Board;

namespace Tixy
{
    // To simulate games:
    //    Init
    //    while (!done)
    //      record state in state memory
    //      take turns making a move (move decided by PlayerAgent)

    //    distribute reward over saved states
    
    public interface IBoard
    {
        void Init();
        bool IsValidMove(BoardMove? move);
        void Move(BoardMove? move);
    }
}
