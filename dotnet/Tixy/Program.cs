using System.Diagnostics;
using Tixy;

var player1 = new PlayerAgentConsole();
var player2 = new PlayerAgentRandom();

int win1 = 0;
int win2 = 0;

var sw = Stopwatch.StartNew();
long nextPrint = 0;
var board = new Board(5, 5);

while (true)
{
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

        Console.ReadLine();
        var movePlayer2 = player2.GetMove();
        board.Move(movePlayer2, 2);
        if (board.IsGameOver)
            break;
    }

    if (board.WinnerId == 1)
        win1++;
    else
        win2++;

    long ms = sw.ElapsedMilliseconds;
    if (ms > nextPrint)
    {
        int totalGames = win1 + win2;
        double perSec = totalGames / (ms / 1000.0);
        double win1Percentage = (double)win1 / totalGames * 100;
        double win2Percentage = (double)win2 / totalGames * 100;

        Console.WriteLine($"Total: {totalGames}, Player 1: {win1Percentage:0.00}%, Player 2: {win2Percentage:0.00}%, {perSec:0.00} games/s");

        nextPrint = ms + 1000;
    }
}
