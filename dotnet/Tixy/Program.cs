using System.Diagnostics;
using System.Text.Json;
using Tixy;

int iteration = 1;

//PlayAgainstAi.Play();

//ExportAnalyzer.Run();

while (true)
{
    IPlayerAgent player1;
    IPlayerAgent player2 = new PlayerAgentGreedy();

    if (!File.Exists("c:\\temp\\ml\\tixy.onnx"))
    {
        Console.WriteLine("No model exists yet, using random player instead of AI.");
        player1 = new PlayerAgentRandom();
    }
    else
    {
        player1 = new PlayerAgentOnnx();
    }

    var sw = Stopwatch.StartNew();
    long nextPrint = 1000;
    var board = new Board();

    bool isReplay = false;
    bool allDone = false;

    // PARAM
    bool watch = false;
    int steps = 2000;

    var onnxPlayer = player1 as PlayerAgentOnnx;
    if (onnxPlayer != null)
        onnxPlayer.Epsilon = Math.Min(0.9, (iteration - 1) * 0.1 + 0.2);

    Console.WriteLine($"Starting iteration {iteration++}, epsilon = {onnxPlayer?.Epsilon}\n");

    while (!allDone)
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
            if (ms > nextPrint || i == steps - 1)
            {
                int totalGames = win1 + win2;
                double perSec = totalGames / (ms / 1000.0);
                double win1Percentage = (double)win1 / totalGames * 100;
                double win2Percentage = (double)win2 / totalGames * 100;

                Console.WriteLine($"Total: {totalGames}, Player 1 ({player1.Name}): {win1Percentage:0.00}%, Player 2 ({player2.Name}): {win2Percentage:0.00}%, {perSec:0.00} games/s, avgTurns: {(double)moves.Count / (i + 1):0.0}");

                nextPrint = ms + 1000;
            }
        }

        //return;
            
        double lastWinPct = -1;
        if (File.Exists($"c:\\temp\\ml\\last-ai-win-pct.txt"))
        {
            string tmp = File.ReadAllText($"c:\\temp\\ml\\last-ai-win-pct.txt");
            lastWinPct = double.Parse(tmp);
        }

        string onnxModelPath = "c:\\temp\\ml\\tixy.onnx";
        string onnxModelBackupPath = "c:\\temp\\ml\\tixy-backup.onnx";
        string torchModelPath = "c:\\temp\\ml\\tixy.pth";
        string torchModelBackupPath = "c:\\temp\\ml\\tixy-backup.pth";

        int totalGames2 = win1 + win2;
        bool storeMoves = false;
        double win1Percentage2 = (double)win1 / totalGames2 * 100;
        if (false && win1Percentage2 <= lastWinPct && !isReplay)
        {
            Console.WriteLine($"\nModel did not improve, previous wins: {lastWinPct}%, new wins: {win1Percentage2}%");
            if (File.Exists(onnxModelBackupPath))
            {
                Console.WriteLine("Restoring model backup and running again");
                File.Copy(onnxModelBackupPath, onnxModelPath, true);
                File.Copy(torchModelBackupPath, torchModelPath, true);

                isReplay = true;
            }
            else
            {
                Console.WriteLine("...but since this is the first run we allow it anyway");
                storeMoves = true;
            }
        }
        else
        {
            storeMoves = true;
            if (!isReplay)
            {
                if (lastWinPct > 0)
                {
                    Console.WriteLine($"\nChanging model, previous wins: {lastWinPct}%, new wins: {win1Percentage2}%");
                    //Console.WriteLine($"\nModel IMPROVED, previous wins: {lastWinPct}%, new wins: {win1Percentage2}%");
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
        
        if (storeMoves)
        {
            File.WriteAllText($"c:\\temp\\ml\\last-ai-win-pct.txt", $"{win1Percentage2}");
            File.AppendAllText($"c:\\temp\\ml\\win-pct-list.txt", $"{win1Percentage2:0.00}\n");

            string json = JsonSerializer.Serialize(moves, new JsonSerializerOptions { WriteIndented = false });
            File.WriteAllText($"c:\\temp\\ml\\gen-0.json", json);
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
        //RedirectStandardOutput = true,
        //RedirectStandardError = true,
    };
    var proc = Process.Start(procInfo);
    proc.WaitForExit();
    //string err = proc.StandardError.ReadToEnd();
    //string inf = proc.StandardOutput.ReadToEnd();
    //if (!string.IsNullOrEmpty(err))
    //{
    //    //Console.WriteLine(err);
    //}

    Console.WriteLine("------------------------------------------------------------------------------------");
}
