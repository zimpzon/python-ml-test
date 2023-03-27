namespace Tixy
{
    public interface IPlayerAgent
    {
        void Init(Board board, int playerId);
        void TakeTurn();
    }
}
