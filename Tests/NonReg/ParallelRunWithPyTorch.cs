using System;
using System.Collections.Generic;
using NUnit.Framework;
using SharpNet;
using SharpNet.Data;
using SharpNet.Datasets;
using SharpNet.GPU;
using SharpNet.Layers;
using SharpNet.Models;
using SharpNet.Networks;
using log4net;
using System.IO;
using System.Linq;
using System.Reflection;
using SharpNet.CPU;

namespace SharpNetTests.NonReg
{


    /// <summary>
    /// sandbox to make // run with PyTorch on several kind of networks
    /// </summary>
    //[TestFixture]
    public class ParallelRunWithPyTorch
    {
        private static readonly ILog Log = LogManager.GetLogger(typeof(ParallelRunWithPyTorch));


        private static string GetDefaultWorkingDirectory()
        {
            // ReSharper disable once AssignNullToNotNullAttribute
            return Path.Combine(Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location), "../../../NonReg");
        }
        static ParallelRunWithPyTorch()
        {
            var log_directory = GetDefaultWorkingDirectory();
            Utils.TryDelete(Path.Join(log_directory, "ParallelRunWithPyTorch.log"));
            // ReSharper disable once PossibleNullReferenceException
            Console.WriteLine(Path.GetDirectoryName(Assembly.GetEntryAssembly().Location));
            Utils.ConfigureGlobalLog4netProperties(log_directory, "ParallelRunWithPyTorch");
            Utils.ConfigureThreadLog4netProperties(log_directory, "ParallelRunWithPyTorch");
        }
        private static (float loss_before, float loss_after) Train(Network model,
            CpuTensor<float> X,
            CpuTensor<float> Y,
            double lr,
            int numEpochs,
            int batch_size,
            double momentum  = 0.0,
            double lambdaL2Regularization = 0.0
        )
        {
            Log.Info(model.Summary() + Environment.NewLine);

            Log.Info(model.ToPytorchModule() + Environment.NewLine);


            var predict_before = model.Predict(X, false).ToNumpy();

            using var trainingDataSet = new InMemoryDataSet(X, Y);

            var lossAccuracyBefore = model.ComputeMetricsForValidationDataSet(batch_size, trainingDataSet);
            var loss_before = (float)lossAccuracyBefore.First(t => t.Key == model.NetworkSample.LossFunction).Value;

            Log.Info("-");
            Log.Info("--------------------------------------------------------------------");
            Log.Info("-");

            TestNetwork.Fit(model, X, Y, lr, numEpochs, batch_size);

            var predict_after = model.Predict(X, false).ToNumpy();
            List<KeyValuePair<EvaluationMetricEnum, double>> lossAccuracyAfter = model.ComputeMetricsForValidationDataSet(batch_size, trainingDataSet);
            var loss_after = (float)lossAccuracyAfter.First(t => t.Key == model.NetworkSample.LossFunction).Value;

            Log.Info("C# numEpochs= " + numEpochs);
            Log.Info("C# learningRate= " + lr);
            Log.Info("C# l2regularizer= " + lambdaL2Regularization);
            Log.Info("C# momentum= " + momentum);
            Log.Info("C# batch_size= " + batch_size);
            Log.Info(predict_before);
            Log.Info("C# metrics_before= " + Model.MetricsToString(lossAccuracyBefore, ""));
            Log.Info(predict_after);
            Log.Info("C# metrics_after= " + Model.MetricsToString(lossAccuracyAfter, ""));
            return (loss_before, loss_after);
        }


        [Test, Explicit]
        public void TestParallelRunWithPyTorch_Mse()
        {
            using var X = TestNetworkPropagation.FromNumpyArray(TestNetworkPropagation.X_2_3_4_5);
            using var Y = TestNetworkPropagation.FromNumpyArray(TestNetworkPropagation.Y_2_3);
            const double momentum = 0.9;
            const int deviceId = -1;

            var sample = new NetworkSample
                {
                    LossFunction = EvaluationMetricEnum.Mse,
                    ShuffleDatasetBeforeEachEpoch = false,
                    AutoSaveIntervalInMinutes = -1,
                    CompatibilityMode = NetworkSample.CompatibilityModeEnum.PyTorch,
                    LogNetworkPropagation = true,
                    ResourceIds = new List<int> { deviceId }
                }
                .WithSGD(momentum, false);

            var model = new Network(sample, null, GetDefaultWorkingDirectory(), nameof(TestParallelRunWithPyTorch_Mse), false);
            model
                .Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Dense(3, 0.0, true)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)
                .Dense(3, 0.0, false)
                ;

            TestNetworkPropagation.FromNumpyArray_and_Transpose(
                    "[[-0.0033482015132904053, 0.23990488052368164, -0.368076980113