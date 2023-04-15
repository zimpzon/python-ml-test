using System.Diagnostics;
using System.Text.Json;
using Tixy;

int iteration = 1;

while (true)
{
    IPlayerAgent player1;
    IPlayerAgent player2 = new PlayerAgentRandom();

    if (!File.Exists("c:\\temp\\ml\\tixy.onnx"))
    {
        Console.WriteLine("No model exists yet, using random player instead of AI.");
        player1 = new PlayerAgentRandom();
    }
    else
    {
        player1 = new PlayerAgentOnnx();
    }

    //player2 = new PlayerAgentOnnx();

    var sw = Stopwatch.StartNew();
    long nextPrint = 100000;
    var board = new Board();

    bool isReplay = false;
    bool allDone = false;

    // PARAM
    bool watch = false;
    int steps = 1000;

    Console.WriteLine($"Starting iteration {iteration++}\n");

    while (!allDone)
    {
        for (int k = 0; k < 1; ++k)
        {
            int win1 = 0;
            int win2 = 0;
            List<BoardState> moves = new();

            for (int i = 0; i < steps; ++i)
            {
                int turnsThisGame = 0;

                board.Reset();
                player1.Reset(board, playerId: 1);
                player2.Reset(board, playerId: 2);

                while (true)
                {
                    if (watch)
                        BoardConsolePrint.Print(board, "\n(...)", clear: false);

                    var movePlayer1 = player1.GetMove();
                    board.Move(movePlayer1, 1);

                    if (board.IsGameOver)
                        break;

                    if (watch)
                        BoardConsolePrint.Print(board, "\n(...)", clear: false);

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
                if (i == steps - 1)
                {
                    int totalGames = win1 + win2;
                    double perSec = totalGames / (ms / 1000.0);
                    double win1Percentage = (double)win1 / totalGames * 100;
                    double win2Percentage = (double)win2 / totalGames * 100;

                    Console.WriteLine($"k: {k},Total: {totalGames}, Player 1 ({player1.Name}): {win1Percentage:0.00}%, Player 2 ({player2.Name}): {win2Percentage:0.00}%, {perSec:0.00} games/s, avgTurns: {(double)moves.Count / (i + 1):0.0}");

                    nextPrint = ms + 1000;
                }
            }
            double lastWinPct = -1;
            if (File.Exists($"c:\\temp\\ml\\last-ai-win-pct.txt"))
            {
                string tmp = File.ReadAllText($"c:\\temp\\ml\\last-ai-win-pct.txt");
                lastWinPct = double.Parse(tmp);
            }

            string onnxModelPath = "c:\\temp\\ml\\tixy.onnx";
            string onnxModelBackupPath = "c:\\temp\\ml\\tixy-backup.onnx";
            string torchModelPath = "c:\\temp\\ml\\tixy.pt";
            string torchModelBackupPath = "c:\\temp\\ml\\tixy-backup.pt";

            int totalGames2 = win1 + win2;
            double win1Percentage2 = (double)win1 / totalGames2 * 100;
            if (win1Percentage2 <= lastWinPct && !isReplay)
            {
                Console.WriteLine($"\nModel did not improve, previous wins: {lastWinPct}%, new wins: {win1Percentage2}%");
                if (File.Exists(onnxModelBackupPath))
                {
                    Console.WriteLine("Restoring model backup and running again");
                    File.Copy(onnxModelBackupPath, onnxModelPath, true);
                    File.Copy(torchModelBackupPath, torchModelPath, true);
                }
                isReplay = true;
            }
            else
            {
                File.WriteAllText($"c:\\temp\\ml\\last-ai-win-pct.txt", $"{win1Percentage2}");

                string json = JsonSerializer.Serialize(moves, new JsonSerializerOptions { WriteIndented = false });
                File.WriteAllText($"c:\\temp\\ml\\gen-{k}.json", json);

                if (!isReplay)
                {
                    if (lastWinPct > 0)
                    {
                        Console.WriteLine($"\nModel IMPROVED, previous wins: {lastWinPct}%, new wins: {win1Percentage2}%");
                        if (File.Exists(onnxModelPath))
                        {
                            Console.WriteLine("Backing up new model");
                            File.Copy(onnxModelPath, onnxModelBackupPath, true);
                            File.Copy(torchModelPath, torchModelBackupPath, true);
                        }
                    }
                }

                allDone = true;
            }
        }
    }

    Console.WriteLine("------------------------------------------------------------------------------------");
    Console.WriteLine("Training...");

    string trainingCommand = "C:\\Users\\peter\\miniconda3\\python.exe";
    string trainingParam = "c:\\Repos\\python-ml-test\\Net1.py";
    var procInfo = new ProcessStartInfo(trainingCommand)
    {
        WorkingDirectory = "c:\\Repos\\python-ml-test",
        Arguments = trainingParam,
        UseShellExecute = false,
        RedirectStandardOutput = true,
        RedirectStandardError = true,
    };
    var proc = Process.Start(procInfo);
    proc.WaitForExit();
    string err = proc.StandardError.ReadToEnd();
    string inf = proc.StandardOutput.ReadToEnd();
    if (!string.IsNullOrEmpty(err))
    {
        //Console.WriteLine(err);
    }
    Console.WriteLine("------------------------------------------------------------------------------------");
}
