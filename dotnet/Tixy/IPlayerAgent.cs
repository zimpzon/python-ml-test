namespace Tixy
{
    public interface IPlayerAgent
    {
        void Reset(Board board, int playerId);
        Move GetMove();
    }
}
