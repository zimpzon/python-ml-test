using System.Diagnostics;
using System.Text;
using System.Text.Json;
using Tixy;

var player1 = new PlayerAgentRandom();
var player2 = new PlayerAgentRandom();

int win1 = 0;
int win2 = 0;

var sw = Stopwatch.StartNew();
long nextPrint = 0;
var board = new Board();

List<BoardState> moves = new ();

int steps = 100;
for (int i = 0; i < steps; ++i)
{
    int turnsThisGame = 0;
    
    board.Reset();
    player1.Reset(board, playerId: 1);
    player2.Reset(board, playerId: 2);

    while (true)
    {
        if (player1.GetType() == typeof(PlayerAgentConsole))
        {
            Console.Clear();
            BoardConsolePrint.Print(board);
        }

        var movePlayer1 = player1.GetMove();
        board.Move(movePlayer1, 1);

        if (board.IsGameOver)
            break;

        var movePlayer2 = player2.GetMove();
        board.Move(movePlayer2, 2);
        turnsThisGame++;
        
        if (board.IsGameOver)
            break;
    }

    if (board.WinnerId == 1)
        win1++;
    else
        win2++;
    
    moves.AddRange(board.Moves);

    long ms = sw.ElapsedMilliseconds;
    if (ms > nextPrint || i == steps - 1)
    {
        int totalGames = win1 + win2;
        double perSec = totalGames / (ms / 1000.0);
        double win1Percentage = (double)win1 / totalGames * 100;
        double win2Percentage = (double)win2 / totalGames * 100;

        Console.WriteLine($"Total: {totalGames}, Player 1: {win1Percentage:0.00}%, Player 2: {win2Percentage:0.00}%, {perSec:0.00} games/s, avgTurns: {(double)moves.Count / (i + 1):0.0}");

        nextPrint = ms + 1000;
    }
}

string json = JsonSerializer.Serialize(moves, new JsonSerializerOptions { WriteIndented = false });
File.WriteAllText("c:\\temp\\ml\\gen1.json", json);
