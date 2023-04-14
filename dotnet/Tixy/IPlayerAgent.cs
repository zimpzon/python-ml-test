namespace Tixy
{
    public interface IPlayerAgent
    {
        string Name { get; }
        void Reset(IBoard board, int playerId);
        Move GetMove();
    }
}
