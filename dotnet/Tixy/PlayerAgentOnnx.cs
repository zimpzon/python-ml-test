using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;

namespace Tixy
{
    public class PlayerAgentOnnx : IPlayerAgent
    {
        public double Epsilon { get; set; } = 1.0;
        private IBoard _board;
        private int _playerId;
        private InferenceSession _onnxSession;

        public string Name => "AI";
        
        public void Reset(IBoard board, int playerId)
        {
            _board = board;
            _playerId = playerId;

            string onnxModelPath = "c:\\temp\\ml\\tixy.onnx";
            _onnxSession = new InferenceSession(onnxModelPath);
        }

        public Move GetMove()
        {
            // Mirroring will take a lot more work. Maybe just use a plane like AlphaZero did.
            var boardState = Util.ExportBoard(_board, _playerId);

            var inputArray = new float[1, 200];
            for (int i = 0; i < inputArray.Length; ++i)
                inputArray[0, i] = boardState.State[i];

            var inputTensor = inputArray.ToTensor();
            //var newShape = new int[] { 1, 9, 5, 5 };
            //var reshapedTensor = inputTensor.Reshape(newShape);
            
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input", inputTensor) };
            using var results = _onnxSession.Run(inputs);
            var output = results.First(item => item.Name == "output").AsEnumerable<float>().ToArray();
            // Convert scores to probabilities using softmax.
            output = AiUtil.Softmax(output);

            var value = results.First(item => item.Name == "value").AsEnumerable<float>().ToArray();

            var rawScores = AiUtil.GetPiecesRawScores(_board, output, _playerId);

            var move = AiUtil.SelectMoveByProbability(_board, rawScores, _playerId, Epsilon);
            return move;
        }
    }
}
