namespace Tixy
{
    public interface IBoard
    {
        void Init();
        List<ActivePiece> PlayerPieces { get; }
        ActivePiece AddPiece(int playerId, char type, string pos);
        ActivePiece AddPiece(int playerId, char type, int x, int y);

        List<ActivePiece> GetPlayerPieces(int? playerId = null);
        ActivePiece GetPieceAt(int x, int y, int? playerId = null);
        List<Move> GetPieceDirections(ActivePiece activePiece);
        List<Move> GetPieceValidMoves(ActivePiece activePiece);
        (int dx, int dy) DeltasFromDirection(int direction);
        bool IsValidMove(Move move, int playerId);
        bool IsValidMove(int x, int y, int dir, int playerId);
        void Move(Move move, int playerId);
    }
}
