﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Diagnostics;
using CNTK;
using System.Globalization;

namespace TestCntkGpu
{
    /// <summary>
    /// Логика взаимодействия для MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }

        private void TestFunc()
        {
            var inputs = new[] { new[] { 0.0f, 0.0f }, new[] { 1.0f, 0.0f }, new[] { 0.0f, 1.0f }, new[] { 1.0f, 1.0f } };
            var expected = new[] { new[] { 0.0f }, new[] { 1.0f }, new[] { 1.0f }, new[] { 0.0f } };

            var device = DeviceDescriptor.CPUDevice;

            const int inputDimensions = 2;
            const int hiddenDimensions = 2;
            const int outputDimensions = 1;

            double initMin = -0.15;
            double initMax = 0.15;
            double normalMean = 0.0;
            double standartDev = 0.25;
            Random random = new Random();
            uint seed = (uint)random.Next(1, 10000);

            var inputVariable = Variable.InputVariable(new[] { inputDimensions }, DataType.Float);
            var outputVariable = Variable.InputVariable(new[] { outputDimensions }, DataType.Float);
            
            //var hiddenWeights = new Parameter(NDArrayView.RandomUniform<float>(new[] { hiddenDimensions, inputDimensions }, initMin, initMax, 1, device));
            //var hiddenBias = new Parameter(NDArrayView.RandomUniform<float>(new[] { hiddenDimensions }, initMin, initMax, 1, device));
            var hiddenWeights = new Parameter(NDArrayView.RandomNormal<float>(new[] { hiddenDimensions, inputDimensions }, normalMean, standartDev, seed++, device));
            var hiddenBias = new Parameter(NDArrayView.RandomNormal<float>(new[] { hiddenDimensions }, normalMean, standartDev, seed++, device));
            var hidden = CNTKLib.Sigmoid(CNTKLib.Plus(hiddenBias, CNTKLib.Times(hiddenWeights, inputVariable)));

            //var outWeights = new Parameter(NDArrayView.RandomUniform<float>(new[] { outputDimensions, hiddenDimensions }, initMin, initMax, 1, device)); 
            //var outBias = new Parameter(NDArrayView.RandomUniform<float>(new[] { outputDimensions }, initMin, initMax, 1, device));
            var outWeights = new Parameter(NDArrayView.RandomNormal<float>(new[] { outputDimensions, hiddenDimensions }, normalMean, standartDev, seed++, device));
            var outBias = new Parameter(NDArrayView.RandomNormal<float>(new[] { outputDimensions }, normalMean, standartDev, seed++, device));
            var prediction = CNTKLib.Sigmoid(CNTKLib.Plus(outBias, CNTKLib.Times(outWeights, hidden)));



            NDArrayView hiddenWeightsArrayViewBefore = hiddenWeights.Value();
            Value hiddenWeightValueBefore = new Value(hiddenWeightsArrayViewBefore);
            IList<IList<float>> hiddenWeightDataBefore = hiddenWeightValueBefore.GetDenseData<float>(hiddenWeights);

            NDArrayView hiddenBiasArrayViewBefore = hiddenBias.Value();
            Value hiddenBiasValueBefore = new Value(hiddenBiasArrayViewBefore);
            IList<IList<float>> hiddenBiasDataBefore = hiddenBiasValueBefore.GetDenseData<float>(hiddenBias);

            NDArrayView outWeightsArrayViewBefore = outWeights.Value();
            Value outWeightsValueBefore = new Value(outWeightsArrayViewBefore);
            IList<IList<float>> outWeightsDataBefore = outWeightsValueBefore.GetDenseData<float>(outWeights);

            NDArrayView outBiasArrayViewBefore = outBias.Value();
            Value outBiasValueBefore = new Value(outBiasArrayViewBefore);
            IList<IList<float>> outBiasDataBefore = outBiasValueBefore.GetDenseData<float>(outBias);


            var trainingLoss = CNTKLib.BinaryCrossEntropy(prediction, outputVariable);

            /*var schedule = new TrainingParameterScheduleDouble(0.05);
            var momentum = CNTKLib.MomentumAsTimeConstantSchedule(new DoubleVector(new[] { 1.0, 10.0, 100.0 }));
            var learner = CNTKLib.AdamLearner(new ParameterVector(prediction.Parameters().Select(o => o).ToList()), schedule, momentum);
            var trainer = CNTKLib.CreateTrainer(prediction, trainingLoss, trainingLoss, new LearnerVector(new[] { learner }));*/

            TrainingParameterScheduleDouble learningRatePerSample = new TrainingParameterScheduleDouble(0.01, 1);
            IList<Learner> parameterLearners = new List<Learner>() { Learner.SGDLearner(prediction.Parameters(), learningRatePerSample) };
            var trainer = Trainer.CreateTrainer(prediction, trainingLoss, trainingLoss, parameterLearners);

            const int minibatchSize = 4;

            var features = new float[minibatchSize][];
            var labels = new float[minibatchSize][];

            const int numMinibatchesToTrain = 100;
            var trainIndex = 0;

            for (var i = 0; i < numMinibatchesToTrain; ++i)
            {
                for (var j = 0; j < minibatchSize; ++j)
                {
                    features[j] = inputs[trainIndex];
                    labels[j] = expected[trainIndex];
                    trainIndex = (trainIndex + 1) % inputs.Length;
                }
                var batchInput = Value.Create(new[] { inputDimensions }, features.Select(o => new NDArrayView(new[] { inputDimensions }, o, device)), device);
                var batchLabels = Value.Create(new[] { outputDimensions }, labels.Select(o => new NDArrayView(new[] { outputDimensions }, o, device)), device);
                var minibatchBindings = new Dictionary<Variable, Value> { { inputVariable, batchInput }, { outputVariable, batchLabels } };

                trainer.TrainMinibatch(minibatchBindings, true, device);
            }

            float[] inputValuesArr = new float[inputs.Length * inputs[0].Length];
            for (int i = 0; i < inputs.Length; i++)
            {
                for (int k = 0; k < inputs[i].Length; k++)
                {
                    inputValuesArr[i * inputs[i].Length + k] = inputs[i][k];
                }
            }
            Value inputValues = Value.CreateBatch<float>(new int[] { inputDimensions }, inputValuesArr, device);
            var inputDataMap = new Dictionary<Variable, Value>() { { inputVariable, inputValues } };
            var outputDataMap = new Dictionary<Variable, Value>() { { prediction.Output, null } };
            prediction.Evaluate(inputDataMap, outputDataMap, device);
            var outputValue = outputDataMap[prediction.Output];
            IList<IList<float>> actualLabelSoftMax = outputValue.GetDenseData<float>(prediction.Output);


            NDArrayView hiddenWeightsArrayViewAfter = hiddenWeights.Value();
            Value hiddenWeightValueAfter = new Value(hiddenWeightsArrayViewAfter);
            IList<IList<float>> hiddenWeightDataAfter = hiddenWeightValueAfter.GetDenseData<float>(hiddenWeights);

            NDArrayView hiddenBiasArrayViewAfter = hiddenBias.Value();
            Value hiddenBiasValueAfter = new Value(hiddenBiasArrayViewAfter);
            IList<IList<float>> hiddenBiasDataAfter = hiddenBiasValueAfter.GetDenseData<float>(hiddenBias);

            NDArrayView outWeightsArrayViewAfter = outWeights.Value();
            Value outWeightsValueAfter = new Value(outWeightsArrayViewAfter);
            IList<IList<float>> outWeightsDataAfter = outWeightsValueAfter.GetDenseData<float>(outWeights);

            NDArrayView outBiasArrayViewAfter = outBias.Value();
            Value outBiasValueAfter = new Value(outBiasArrayViewAfter);
            IList<IList<float>> outBiasDataAfter = outBiasValueAfter.GetDenseData<float>(outBias);
            int y = 0;
        }

        private void Button_Click(object sender, RoutedEventArgs e)
        {
            TestFunc();



            int iterations = 100;
            // initialize input and output values
            double[][] input = new double[4][] {
                        new double[] {0, 0}, new double[] {0, 1},
                        new double[] {1, 0}, new double[] {1, 1}
                    };
            double[][] output = new double[4][] {
                        new double[] {0}, new double[] {1},
                        new double[] {1}, new double[] {0}
                    };
            // create neural network
            /*AForge.Neuro.ActivationNetwork activationNetwork = AForgeExtensions.Neuro.ActivationNetworkFeatures.BuildRandom(-1f, 1f, new AForge.Neuro.SigmoidFunction(), 2, 2, 1);
            AForge.Neuro.Learning.BackPropagationLearning teacher = new AForge.Neuro.Learning.BackPropagationLearning(activationNetwork);
            double[] aforgeLosses = new double[iterations];
            for (int i = 0; i < iterations; i++)
            {
                aforgeLosses[i] = teacher.RunEpoch(input, output);
            }
            double[] o0 = activationNetwork.Compute(input[0]);
            double[] o1 = activationNetwork.Compute(input[1]);
            double[] o2 = activationNetwork.Compute(input[2]);
            double[] o3 = activationNetwork.Compute(input[3]);*/


            int inputDim = 2;
            int hiddenDim = 8;
            int numOutputClasses = 1;
            DeviceDescriptor device = DeviceDescriptor.CPUDevice;
            Variable inputVariable1 = Variable.InputVariable(new int[] { inputDim }, DataType.Float);
            Variable outputVariable2 = Variable.InputVariable(new int[] { numOutputClasses }, DataType.Float);

            var weightParam1 = new Parameter(new int[] { hiddenDim, inputDim }, DataType.Float, 1, device, "w");
            var biasParam1 = new Parameter(new int[] { hiddenDim }, DataType.Float, 0, device, "b");
            var classifierOutput0 = CNTKLib.Sigmoid(CNTKLib.Plus(CNTKLib.Times(weightParam1, inputVariable1), biasParam1));

            var weightParam2 = new Parameter(new int[] { numOutputClasses, hiddenDim }, DataType.Float, 1, device, "ww");
            var biasParam2 = new Parameter(new int[] { numOutputClasses }, DataType.Float, 0, device, "bb");
            //var classifierOutput1 = CNTKLib.Sigmoid(CNTKLib.Times(weightParam2, classifierOutput0) + biasParam2);
            var classifierOutput1 = CNTKLib.Sigmoid(CNTKLib.Plus(CNTKLib.Times(weightParam2, classifierOutput0), biasParam2));



            //var loss = CNTKLib.CrossEntropyWithSoftmax(classifierOutput1, outputVariable2);
            var loss = CNTKLib.BinaryCrossEntropy(classifierOutput1, outputVariable2);
            var evalError = CNTKLib.ClassificationError(classifierOutput1, outputVariable2);

            // prepare for training
            CNTK.TrainingParameterScheduleDouble learningRatePerSample = new CNTK.TrainingParameterScheduleDouble(0.01, 1);
            IList<Learner> parameterLearners = new List<Learner>() { Learner.SGDLearner(classifierOutput1.Parameters(), learningRatePerSample) };
            var trainer = Trainer.CreateTrainer(classifierOutput1, loss, loss, parameterLearners);

            float[] inputValuesArr = new float[input.Length * input[0].Length];
            for (int i = 0; i < input.Length; i++)
            {
                for (int k = 0; k < input[i].Length; k++)
                {
                    inputValuesArr[i * input[i].Length + k] = (float)input[i][k];
                }
            }
            float[] outputValuesArr = new float[output.Length * output[0].Length];
            for (int i = 0; i < output.Length; i++)
            {
                for (int k = 0; k < output[i].Length; k++)
                {
                    outputValuesArr[i * output[i].Length + k] = (float)output[i][k];
                }
            }
            Value inputValues = Value.CreateBatch<float>(new int[] { inputDim }, inputValuesArr, device);
            Value outputValues = Value.CreateBatch<float>(new int[] { numOutputClasses }, outputValuesArr, device);

            // train the model
            for (int minibatchCount = 0; minibatchCount < iterations; minibatchCount++)
            {
                //TODO: sweepEnd should be set properly instead of false.
#pragma warning disable 618
                trainer.TrainMinibatch(new Dictionary<Variable, Value>() { { inputVariable1, inputValues }, { outputVariable2, outputValues } }, device);
#pragma warning restore 618
                //TestHelper.PrintTrainingProgress(trainer, minibatchCount, updatePerMinibatches);
            }

            var inputDataMap = new Dictionary<Variable, Value>() { { inputVariable1, inputValues } };
            var outputDataMap = new Dictionary<Variable, Value>() { { classifierOutput1.Output, null } };
            classifierOutput1.Evaluate(inputDataMap, outputDataMap, device);
            var outputValue = outputDataMap[classifierOutput1.Output];
            IList<IList<float>> actualLabelSoftMax = outputValue.GetDenseData<float>(classifierOutput1.Output);





            



            NDShape nDShape = new NDShape(1, 4);
            /*Function logisticModel = CreateLogisticModel(Variable.InputVariable(nDShape, DataType.Float), 4);
            LogisticRegression.TrainAndEvaluate(DeviceDescriptor.CPUDevice);*/
        }

        private void Window_Loaded(object sender, RoutedEventArgs e)
        {
            var nfi = (NumberFormatInfo)CultureInfo.InvariantCulture.NumberFormat.Clone();
            nfi.NumberGroupSeparator = " "; //string formatted = 1234897.11m.ToString("#,0.00", nfi); // "1 234 897.11"

            //var devices = DeviceDescriptor.AllDevices();
            var cpuDevice = DeviceDescriptor.CPUDevice;
            //var gpuDevice = DeviceDescriptor.GPUDevice(0);
            var modelDevice = cpuDevice;
            //DeviceDescriptor.TrySetDefaultDevice(modelDevice);

            int inputDim = 61;
            int cellDim = 61;
            int outputDim = 6;
            int sequenceLength = 744;
            int sequencesCount = 100;

            /*NDShape inputShape = NDShape.CreateNDShape(new int[] { inputDim });
            NDShape outputShape = NDShape.CreateNDShape(new int[] { outputDim });

            var axis = new Axis("inputAxis");
            var inputVariable = Variable.InputVariable(inputShape, DataType.Float, "inputVariable", new List<Axis> { axis, Axis.DefaultBatchAxis() });
            var outputVariable = Variable.InputVariable(outputShape, DataType.Float, "outputVariable", new List<Axis> { axis, Axis.DefaultBatchAxis() });

            var lstmLayer = CntkWrapper.Layers.LSTM<float>(cellDim, inputVariable, modelDevice);
            var model = CntkWrapper.Layers.Dense<float>(outputDim, lstmLayer, CNTKLib.Sigmoid, modelDevice);*/

            /*for(int i = 0; i < 3; i++)
            {
                int modelsCount = 300;
                Stopwatch stopwatch2 = new Stopwatch();
                stopwatch2.Start();
                List<Function> models = new List<Function>();
                for(int k = 0; k <  modelsCount; k++)
                {
                    var axis2 = new Axis("inputAxis");
                    var inputVariable2 = Variable.InputVariable(inputShape, DataType.Float, "inputVariable2", new List<Axis> { axis2, Axis.DefaultBatchAxis() });
                    var outputVariable2 = Variable.InputVariable(outputShape, DataType.Float, "outputVariable2", new List<Axis> { axis2, Axis.DefaultBatchAxis() });
                    var lstmLayer2 = CntkWrapper.Layers.LSTM<float>(cellDim, inputVariable2, modelDevice);
                    var model2 = CntkWrapper.Layers.Dense<float>(outputDim, lstmLayer2, CNTKLib.Sigmoid, modelDevice);
                    models.Append(model2);
                }
                stopwatch2.Stop();
                Trace.WriteLine($"create {modelsCount} models {i} ElapsedMilliseconds={stopwatch2.ElapsedMilliseconds}");
            }*/

            Random random = new Random();
            List<List<float>> inputSequences = new List<List<float>>();
            for (int i = 0; i < sequencesCount; i++)
            {
                List<float> sequence = new List<float>();
                for (int k = 0; k < sequenceLength; k++)
                {
                    float[] input_vector = new float[inputDim];
                    for (int n = 0; n < input_vector.Length; n++)
                    {
                        input_vector[n] = (float)random.NextDouble();
                    }
                    sequence.AddRange(input_vector);
                }
                inputSequences.Add(sequence);
            }

            var modelTest = CreateModel(modelDevice, inputDim, cellDim, outputDim);

            var modelDeviceInputSequences = Value.CreateBatchOfSequences(modelTest.Arguments[0].Shape, inputSequences, modelDevice);

            /*var inputDataMapTest = new Dictionary<Variable, Value>() { { modelTest.Arguments[0], modelDeviceInputSequences } };
            var outputDataMapTest = new Dictionary<Variable, Value>() { { modelTest.Output, null } };
            modelTest.Evaluate(inputDataMapTest, outputDataMapTest, modelDevice);*/

            int numEvaluate = 3;

            bool isParallelCreateModels = false;
            Stopwatch stopwatchCreateModels = new Stopwatch();
            stopwatchCreateModels.Start();

            Function[] models = new Function[numEvaluate];
            if (isParallelCreateModels)
            {
                int threadsNum = 25;
                Parall<Function> parall = new Parall<Function>(threadsNum);
                parall.SleepMilliseconds = 2;
                List<ParallTask<Function>> parallTasks = new List<ParallTask<Function>>();
                for (int i = 0; i < numEvaluate; i++)
                {
                    parallTasks.Add(new ParallTask<Function>(() => CreateModel(modelDevice, inputDim, cellDim, outputDim)));
                }
                parall.AddParallTasks(parallTasks);
                parall.Run();
                for(int i = 0; i < numEvaluate; i++)
                {
                    models[i] = parall.CompletedParallTasks[i].Res;
                }
            }
            else
            {
                for (int i = 0; i < numEvaluate; i++)
                {
                    models[i] = CreateModel(modelDevice, inputDim, cellDim, outputDim);
                }
            }
            stopwatchCreateModels.Stop();
            Trace.WriteLine($"stopwatchCreateModels: numEvaluate={numEvaluate}, isParallel={isParallelCreateModels}, ElapsedMilliseconds=({stopwatchCreateModels.ElapsedMilliseconds.ToString("#,0", nfi)})");

            bool isParallelEvaluate = false;
            Stopwatch stopwatchEvaluate = new Stopwatch();
            stopwatchEvaluate.Start();

            if (isParallelEvaluate)
            {
                int threadsNum = 3;
                Parall<int> parall = new Parall<int>(threadsNum);
                List<ParallTask<int>> parallTasks = new List<ParallTask<int>>();
                for(int i = 0; i < numEvaluate; i++)
                {
                    var model = models[i];
                    var inputDataMap = new Dictionary<Variable, Value>() { { model.Arguments[0], modelDeviceInputSequences } };
                    var outputDataMap = new Dictionary<Variable, Value>() { { model.Output, null } };
                    parallTasks.Add(new ParallTask<int>(() => model.Evaluate(inputDataMap, outputDataMap, modelDevice)));
                }
                parall.AddParallTasks(parallTasks);
                parall.Run();
            }
            else
            {
                for (int i = 0; i < numEvaluate; i++)
                {
                    Stopwatch stopwatchSingleEvaluation = new Stopwatch();
                    stopwatchSingleEvaluation.Start();
                    var model = models[i];
                    var inputDataMap = new Dictionary<Variable, Value>() { { model.Arguments[0], modelDeviceInputSequences } };
                    var outputDataMap = new Dictionary<Variable, Value>() { { model.Output, null } };
                    model.Evaluate(inputDataMap, outputDataMap, modelDevice);
                    stopwatchSingleEvaluation.Stop();
                    Trace.WriteLine($"SingleEvaluation i={i}, ElapsedMilliseconds=({stopwatchSingleEvaluation.ElapsedMilliseconds.ToString("#,0", nfi)})");
                }
            }

            stopwatchEvaluate.Stop();
            Trace.WriteLine($"stopwatchEvaluate: numEvaluate={numEvaluate}, isParallel={isParallelEvaluate}, ElapsedMilliseconds=({stopwatchEvaluate.ElapsedMilliseconds.ToString("#,0", nfi)})");
            
            /*for(int i = 0; i < 7; i++)
            {
                Stopwatch stopwatch = new Stopwatch();
                stopwatch.Start();
                var inputDataMap = new Dictionary<Variable, Value>() { { inputVariable, modelDeviceInputSequences } };
                var outputDataMap = new Dictionary<Variable, Value>() { { model.Output, null } };
                model.Evaluate(inputDataMap, outputDataMap, modelDevice);
                var outputValue = outputDataMap[model.Output];
                IList<IList<float>> actualLabelSoftMax = outputValue.GetDenseData<float>(model.Output);
                stopwatch.Stop();
                Trace.WriteLine($"{i} ElapsedMilliseconds={stopwatch.ElapsedMilliseconds}");
            }*/

            int u = 0;
            System.Windows.Application.Current.Shutdown();
        }
        private Function CreateModel(DeviceDescriptor device, int inputDim, int cellDim, int outputDim)
        {
            NDShape inputShape = NDShape.CreateNDShape(new int[] { inputDim });
            NDShape outputShape = NDShape.CreateNDShape(new int[] { outputDim });

            var axis = new Axis("inputAxis");
            var inputVariable = Variable.InputVariable(inputShape, DataType.Float, "inputVariable", new List<Axis> { axis, Axis.DefaultBatchAxis() });
            var outputVariable = Variable.InputVariable(outputShape, DataType.Float, "outputVariable", new List<Axis> { axis, Axis.DefaultBatchAxis() });

            var lstmLayer = CntkWrapper.Layers.LSTM<float>(cellDim, inputVariable, device);
            var model = CntkWrapper.Layers.Dense<float>(outputDim, lstmLayer, CNTKLib.Sigmoid, device);
            return model;
        }
    }
}
