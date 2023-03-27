using Tixy;

var board = new Board(5, 5);
while (true)
{
    board.Print();
    Console.Write("Command: ");

    string? cmd = Console.ReadLine()?.ToUpper();

    if (!board.ParseMoveCommand(cmd, out var move))
    {
        Console.WriteLine($"Bad command, format must be: A1 B2");
        continue;
    }

    if (!board.IsValidMove(move))
    {
        Console.WriteLine($"Not a valid move");
        continue;
    }

    board.Move(move);
    Console.WriteLine();
}
