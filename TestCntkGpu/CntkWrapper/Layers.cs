using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CNTK;

namespace CntkWrapper
{
    public static class Layers
    {
        public static uint Seed = 1;
        public static double NormalMean = 0.0;
        public static double StandartDev = 0.25;

        public static Function Dense<ElementType>(int dim, Variable previousLayer, Func<Variable, Function> activationFunction, DeviceDescriptor device)
        {
            int inputDimension = previousLayer.Shape.Dimensions[0];
            var weights = new Parameter(NDArrayView.RandomNormal<ElementType>(new[] { dim, inputDimension }, NormalMean, StandartDev, Seed++, device));
            var bias = new Parameter(NDArrayView.RandomNormal<ElementType>(new[] { dim }, NormalMean, StandartDev, Seed++, device));
            var layer = activationFunction(CNTKLib.Plus(bias, CNTKLib.Times(weights, previousLayer)));
            return layer;
        }
        public static Function LSTM<ElementType>(int cellDim, Variable previousLayer, DeviceDescriptor device)
        {
            Func<Variable, Function> pastValueRecurrenceHook = (x) => CNTKLib.PastValue(x);
            /*Function LSTMFunction = LSTMPComponentWithSelfStabilization<float>(
                new int[] { cellDim },
                new int[] { cellDim },
                pastValueRecurrenceHook,
                pastValueRecurrenceHook,
                device).Item1;*/

            // /\ LSTMPComponentWithSelfStabilization
            var dh = Variable.PlaceholderVariable(new int[] { cellDim }, previousLayer.DynamicAxes);
            var dc = Variable.PlaceholderVariable(new int[] { cellDim }, previousLayer.DynamicAxes);

            //var LSTMCell = LSTMPCellWithSelfStabilization<ElementType>(previousLayer, dh, dc, device);

            // /\ LSTMPCellWithSelfStabilization
            var prevOutput = dh;
            var prevCellState = dc;
            int outputDim = prevOutput.Shape[0];
            //int cellDim = dc.Shape[0];

            bool isFloatType = typeof(ElementType).Equals(typeof(float));
            DataType dataType = isFloatType ? DataType.Float : DataType.Double;

            Func<int, Parameter> createBiasParam;
            if (isFloatType)
                createBiasParam = (dim) => new Parameter(new int[] { dim }, 0.01f, device, "");
            else
                createBiasParam = (dim) => new Parameter(new int[] { dim }, 0.01, device, "");

            uint seed2 = 1;
            Func<int, Parameter> createProjectionParam = (oDim) => new Parameter(new int[] { oDim, NDShape.InferredDimension }, dataType, CNTKLib.GlorotUniformInitializer(1.0, 1, 0, seed2++), device);

            Func<int, Parameter> createDiagWeightParam = (dim) => new Parameter(new int[] { dim }, dataType, CNTKLib.GlorotUniformInitializer(1.0, 1, 0, seed2++), device);

            Func<Variable> projectInput = () =>
                createBiasParam(cellDim) + (createProjectionParam(cellDim) * previousLayer);

            // Input gate
            Function it =
                CNTKLib.Sigmoid(
                    (Variable)(projectInput() + (createProjectionParam(cellDim) * prevOutput)) +
                    CNTKLib.ElementTimes(createDiagWeightParam(cellDim), prevCellState));
            Function bit = CNTKLib.ElementTimes(
                it,
                CNTKLib.Tanh(projectInput() + (createProjectionParam(cellDim) * prevOutput)));

            // Forget-me-not gate
            Function ft = CNTKLib.Sigmoid(
                (Variable)(
                        projectInput() + (createProjectionParam(cellDim) * prevOutput)) +
                        CNTKLib.ElementTimes(createDiagWeightParam(cellDim), prevCellState));
            Function bft = CNTKLib.ElementTimes(ft, prevCellState);

            Function ct = (Variable)bft + bit;

            // Output gate
            Function ot = CNTKLib.Sigmoid(
                (Variable)(projectInput() + (createProjectionParam(cellDim) * prevOutput)) +
                CNTKLib.ElementTimes(createDiagWeightParam(cellDim), ct));
            Function ht = CNTKLib.ElementTimes(ot, CNTKLib.Tanh(ct));

            Function c = ct;
            Function h = (outputDim != cellDim) ? (createProjectionParam(outputDim) * ht) : ht;

            var LSTMCell = new Tuple<Function, Function>(h, c);
            // \/ LSTMPCellWithSelfStabilization

            var actualDh = pastValueRecurrenceHook(LSTMCell.Item1);
            var actualDc = pastValueRecurrenceHook(LSTMCell.Item2);

            // Form the recurrence loop by replacing the dh and dc placeholders with the actualDh and actualDc
            (LSTMCell.Item1).ReplacePlaceholders(new Dictionary<Variable, Variable> { { dh, actualDh }, { dc, actualDc } });

            Function LSTMFunction = LSTMCell.Item1;
            // \/ LSTMPComponentWithSelfStabilization

            Function layer = CNTKLib.SequenceLast(LSTMFunction);
            return layer;
        }
    }
}
