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
                    "[[-0.0033482015132904053, 0.23990488052368164, -0.36807698011398315, -0.3291219472885132, -0.1722462773323059], [0.11992359161376953, -0.008860737085342407, 0.3545909523963928, -0.039687544107437134, 0.11833834648132324], [-0.13515380024909973, -0.08790671825408936, -0.4272448420524597, -0.2961815595626831, -0.18435177206993103]]")
                .CopyTo(((DenseLayer)model.Layers[1]).Weights);
            TestNetworkPropagation.FromNumpyArray_and_Transpose(
                    "[[-0.11299018561840057, -0.07257714122533798, 0.06053619086742401, 0.13839800655841827, -0.034300029277801514, 0.12471862137317657, -0.026863887906074524, 0.017635688185691833, 0.15091271698474884, -0.154611736536026, -0.10492299497127533, -0.04219420999288559, -0.06496666371822357, 0.1440001279115677, -0.10802994668483734, -0.0767221450805664, -0.11644008010625839, -0.1560935080051422, -0.09729008376598358, 0.14326633512973785, 0.07436972856521606, 0.08077876269817352, 0.008765265345573425, -0.08544725179672241, 0.028197452425956726, -0.15561579167842865, -0.12042771279811859, -0.08592166751623154, 0.1051563173532486, 0.09772022068500519, -0.07391583919525146, -0.006013736128807068, 0.10659344494342804, 0.16568885743618011, 0.06614702939987183, 0.022515475749969482], [0.11174772679805756, -0.09813372790813446, 0.031057342886924744, -0.12921759486198425, -0.1155143603682518, -0.08609726279973984, 0.07541216909885406, 0.06702673435211182, -0.09872542321681976, 0.05035118758678436, 0.09149535000324249, -0.021036222577095032, 0.006363585591316223, 0.03861744701862335, 0.10339610278606415, 0.16003234684467316, -0.12843726575374603, -0.06107829511165619, 0.06550164520740509, 0.138091579079628, 0.1450345665216446, 0.14705945551395416, 0.03316909074783325, -0.14493045210838318, 0.015332087874412537, -0.10426755994558334, -0.15532569587230682, 0.14808209240436554, 0.1267266422510147, -0.1662546694278717, 0.031195342540740967, -0.028076663613319397, -0.02742685377597809, -0.07629281282424927, 0.06409269571304321, -0.09871725738048553], [0.06109856069087982, 0.08428467810153961, 0.11931194365024567, 0.06231851875782013, -0.16495588421821594, -0.10811614990234375, 0.08321917057037354, 0.03488355875015259, -0.13001400232315063, -0.09596991539001465, 0.156791552901268, 0.11230297386646271, -0.07267086207866669, -0.04194746911525726, -0.15876635909080505, -0.0029956847429275513, -0.1255098283290863, -0.12855945527553558, -0.00918327271938324, 0.0250241756439209, -0.06825505197048187, 0.09889627993106842, -0.10142318904399872, 0.15122835338115692