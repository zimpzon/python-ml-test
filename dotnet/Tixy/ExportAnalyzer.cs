using System.Text.Json;

namespace Tixy
{
    internal static class ExportAnalyzer
    {
        public static void Run()
        {
            string json = File.ReadAllText("c:\\temp\\ml\\gen-1.json");
            var states = JsonSerializer.Deserialize<List<BoardState>>(json);

            int winnerForward = 0;
            int winnerBackward = 0;
            int loserForward = 0;
            int loserBackward = 0;

            foreach (var state in states)
            {
                int forwardDir = state.PlayerIdx == 0 ? 5 : 1;
                
                if (state.Value > 0)
                {
                    // Winner
                    if (state.SelectedDirection == forwardDir)
                        winnerForward++;
                    else
                        winnerBackward++;
                }
                else
                {
                    // Loser
                    if (state.SelectedDirection == forwardDir)
                        loserForward++;
                    else
                        loserBackward++;
                }
            }

            int totalWinner = winnerForward + winnerBackward;
            int totalLoser = loserForward + loserBackward;
            double winnerPctForward = winnerForward / (double)totalWinner;
            double loserPctForward = loserForward / (double)totalLoser;

            Console.WriteLine($"winnerForward = {winnerForward}");
        }
    }
}
