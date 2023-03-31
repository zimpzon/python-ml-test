namespace Tixy
{
    public interface IPlayerAgent
    {
        void Reset(IBoard board, int playerId);
        Move GetMove();
    }
}
