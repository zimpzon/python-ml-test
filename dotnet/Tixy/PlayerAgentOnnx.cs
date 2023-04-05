using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;

namespace Tixy
{
    public class PlayerAgentOnnx : IPlayerAgent
    {
        private IBoard _board;
        private int _playerId;
        private readonly InferenceSession _onnxSession;

        public PlayerAgentOnnx()
        {
            // Load the ONNX model and perform inference
            string onnxModelPath = "c:\\temp\\ml\\tixy.onnx";
            _onnxSession = new InferenceSession(onnxModelPath);
        }

        public void Reset(IBoard board, int playerId)
        {
            _board = board;
            _playerId = playerId;
        }

        public Move GetMove()
        {
            var boardState = Util.ExportBoard(_board, _playerId);

            var inputArray = new float[1, 225];
            for (int i = 0; i < inputArray.Length; ++i)
                inputArray[0, i] = boardState.State[i];

            var inputTensor = inputArray.ToTensor();
            var newShape = new int[] { 1, 9, 5, 5 };
            var reshapedTensor = inputTensor.Reshape(newShape);

            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input", reshapedTensor) };
            using var results = _onnxSession.Run(inputs);
            var output = results.First(item => item.Name == "output").AsEnumerable<float>().ToArray();

            // Algo alternative:
            //   repeat
            //     find max value and convert to move
            //       if invalid move set to float.min and start over
            //       if valie move do move and stop

            ActivePiece bestPiece = null;
            int bestDirection = -1;
            float maxScore = float.MinValue;

            for (int y = 0; y < Board.H; ++y)
            {
                for (int x = 0; x < Board.W; ++x)
                {
                    for (int layerIdx = 0; layerIdx < 8; ++layerIdx)
                    {
                        int idx = layerIdx * Board.Cells + (y * Board.W) + x;

                        int planeId = idx / 25;
                        int posXY = idx - (planeId * 25);
                        int yp = posXY / 5;
                        int xp = posXY - (y * 5);
                        var myPiece = _board.GetPieceAt(xp, yp, _playerId);
                        if (myPiece == null)
                        {
                            output[idx] = float.MinValue;
                            continue;
                        }

                        if (!_board.IsValidMove(xp, yp, planeId, _playerId))
                        {
                            output[idx] = float.MinValue;
                            continue;
                        }

                        if (output[idx] > maxScore)
                        {
                            bestPiece = myPiece;
                            bestDirection = planeId;
                            maxScore = output[idx];
                        }
                    }
                }
            }

            int validCount = output.Count(o => o > float.MinValue);
            if (validCount == 0)
            {
                BoardConsolePrint.Print(_board, "no moves");
                throw new ArgumentException($"I give up, no valid moves");
            }

            if (!_board.IsValidMove(bestPiece.X, bestPiece.Y, bestDirection, _playerId))
                throw new ArgumentException($"Not a valid move, should not be possible, should have been caught in check above");

            var deltas = _board.DeltasFromDirection(bestDirection);
            var move = new Move { X0 = bestPiece.X, Y0 = bestPiece.Y, Dx = deltas.dx, Dy = deltas.dy};
            return move;
        }
    }
}
