namespace Tixy
{
    public interface IBoard
    {
        void Init();
        List<ActivePiece> GetActivePieces(int? playerId = null);
        ActivePiece GetPieceAt(int x, int y, int? playerId = null);
        bool IsValidMove(Move move, int playerId);
        void Move(Move move, int playerId);
    }
}



public interface IPriorityQueue<T>
{
    bool TryDequeue(out T priority, out long value);
    void Enqueue(T priority, long value);
    void UpdatePriority(T priority, long value);
    void Clear();
    int Count { get; }
}