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
        public static Function Dense(int dimension, Variable previousLayer, Func<Variable, Function> activationFunction, DeviceDescriptor device)
        {
            int inputDimension = previousLayer.Shape.Dimensions[0];
            var weights = new Parameter(NDArrayView.RandomNormal<float>(new[] { dimension, inputDimension }, NormalMean, StandartDev, Seed++, device));
            var bias = new Parameter(NDArrayView.RandomNormal<float>(new[] { dimension }, NormalMean, StandartDev, Seed++, device));
            var layer = activationFunction(CNTKLib.Plus(bias, CNTKLib.Times(weights, previousLayer)));
            return layer;
        }
    }
}
