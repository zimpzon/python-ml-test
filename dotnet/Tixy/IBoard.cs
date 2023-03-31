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
        bool IsValidMove(Move move, int playerId);
        void Move(Move move, int playerId);
    }
}
