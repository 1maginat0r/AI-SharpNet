using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using log4net;
using Newtonsoft.Json;
using NUnit.Framework;
using SharpNet;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.Datasets;
using SharpNet.GPU;
using SharpNet.Layers;
using SharpNet.Models;
using SharpNet.Networks;
using SharpNet.TextPreprocessing;

// ReSharper disable AccessToDisposedClosure

namespace SharpNetTests.NonReg
{
    /// <summary>
    /// Sand Box to make // run with TensorFlow on several kind of networks
    /// </summary>
    [TestFixture]
    public class ParallelRunWithTensorFlow
    {
        private static readonly ILog Log = LogManager.GetLogger(typeof(ParallelRunWithTensorFlow));

        [Test, Explicit]
        public void TestParallelRunWithTensorFlow_Efficientnet_Inference()
        {
            var xFileName = Path.Combine(NetworkSample.DefaultDataDirectory, "NonReg", "X_1_224_224_3.txt");
            var yExpectedFileName = Path.Combine(NetworkSample.DefaultDataDirectory, "NonReg", "YExpected_1_224_224_3.txt");
            if (!File.Exists(xFileName) || !File.Exists(yExpectedFileName))
            {
                Console.WriteLine("ignoring test "+nameof(TestParallelRunWithTensorFlow_Efficientnet_Inference)+" because some files are missing");
                return;
            }

            var X = TestNetworkPropagation.FromNumpyArray(File.ReadAllText(xFileName));
            X = (CpuTensor<float>)X.ChangeAxis(new[] { 0, 3, 1, 2 });
            var yExpectedFromKeras = TestNetworkPropagation.FromNumpyArray(File.ReadAllText(yExpectedFileName));

            //we ensure that the network prediction is the same as in Keras
            var networkBuilder = EfficientNetNetworkSample.CIFAR10();
            networkBuilder.SetResourceId(0);
            var network = networkBuilder.EfficientNetB0(NetworkSample.DefaultWorkingDirectory, true, "imagenet", new[] {3, 224, 224});
            var yPredicted = network.Predict(X, false);
            Assert.IsTrue(TensorExtensions.SameFloatContent(yExpectedFromKeras, yPredicted, 1e-5));

            //we save the network
            network.Save(network.WorkingDirectory, network.ModelName);
            network.Dispose();

            //we ensure that the saved version of the network behave the same as the original one
            var networkFromSavedFile = Network.LoadTrainedNetworkModel(network.WorkingDirectory, network.ModelName);
            var yPredictedFromSavedFile = networkFromSavedFile.Predict(X, false);
            Assert.IsTrue(TensorExtensions.SameFloatContent(yExpectedFromKeras, yPredictedFromSavedFile, 1e-5));

            var savedModelFile = Network.ToModelFilePath(network.WorkingDirectory, network.ModelName);
            File.Delete(savedModelFile);
            var saveParametersFile = Network.ToParameterFilePath(network.WorkingDirectory, network.ModelName);
            File.Delete(saveParametersFile);
        }


        /// <summary>
        /// the width and height of the processed image must be a multiple of '32' in YOLO V3
        /// </summary>
        /// <param name="originalHeight"></param>
        /// <param name="originalWidth"></param>
        /// <param name="resizedHeight"></param>
        /// <param name="resizedWidth"></param>
        // ReSharper disable once UnusedMember.Local
        private static void PreferredResizedSizeForYoloV3(int originalHeight, int originalWidth, out int resizedHeight, out int resizedWidth)
        {
            const double capacity = 608 * 608;
            double originalCount = originalHeight * originalWidth;

            resizedHeight = originalHeight;
            resizedWidth = originalWidth;

            if (originalCount > capacity)
            {
                double coeff = Math.Sqrt(originalCount / capacity);
                resizedHeight = (int)(resizedHeight / coeff);
                resizedWidth = (int)(resizedWidth / coeff);
            }

            const int forcedSizeMultiple = 32;
            resizedHeight = forcedSizeMultiple * ((resizedHeight + forcedSizeMultiple - 1) / forcedSizeMultiple);
            resizedWidth = forcedSizeMultiple * ((resizedWidth + forcedSizeMultiple - 1) / forcedSizeMultiple);
        }


        [Test, Explicit]
        public void TestParallelRunWithTensorFlow_Efficientnet()
        {
            const int numEpochs = 1;
            const double learningRate = 0.01;
            const double lambdaL2Regularization = 0.00;
            const double momentum = 0.9;

            var hp = EfficientNetNetworkSample.CIFAR10();
            hp.SetResourceId(0);

            //int defaultHeight = 32;
            const int defaultHeight = 224;

            var network = hp.EfficientNetB0(DenseNetNetworkSample.Cifar10WorkingDirectory, true, "imagenet", new[] { 3, defaultHeight, defaultHeight });
            //network.Save();
            //var logFileName = Utils.ConcatenatePathWithFileName(NetworkSample.DefaultLogDirectory, "Efficientnet_" + System.Diagnostics.Process.GetCurrentProcess().Id + "_" + System.Threading.Thread.CurrentThread.ManagedThreadId + ".log");
            //var logger = new Logger(logFileName, true);

            //var xShape = new[] { 1, 3, defaultHeight, defaultHeight };
            var X = TestNetworkPropagation.FromNumpyArray(Path.Combine(NetworkSample.DefaultDataDirectory, "NonReg", "X_1_224_224_3.txt"));
            X = (CpuTensor<float>)X.ChangeAxis(new[] { 0, 3, 1, 2 });
            //for (int i = 0; i < X.Count; ++i)
            //{
            //    X.Content[i] = 0;
            //}


            //var X = new CpuTensor<float>(xShape, null, "input_1");
            //X.Content[0] = 1;
            //Utils.RandomizeNormalDistribution(X.Content, new Random(), 0, 1);
            int batchSize = X.Shape[0];
            var Y = new CpuTensor<float>(new[] { batchSize, 1000 }, null);
            Y.SpanContent[388] = 1; //panda

            Log.Info("x_train" + Environment.NewLine + X.ToNumpy());
            Log.Info("y_train" + Environment.NewLine + Y.ToNumpy());


            Log.Info(network.Summary() + Environment.NewLine);

            var predict_before_tensor = network.Predict(X, false);
            var predict_before = PredictionToString(predict_before_tensor, "C# prediction_before");

            //network.LogContent();

            using var trainingDataSet = new InMemoryDataSet(X, Y, "", Objective_enum.Classification, null);
            var lossAccuracyBefore = network.ComputeMetricsForValidationDataSet(batchSize, trainingDataSet);

            Log.Info("-");
            Log.Info("--------------------------------------------------------------------");
            Log.Info("-");

            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, batchSize);
            //network.LogContent();

            var predict_after_tensor = network.Predict(X, false);
            var predict_after = PredictionToString(predict_after_tensor, "C# prediction_after");

            //network.LogContent();

            var lossAccuracyAfter = network.ComputeMetricsForValidationDataSet(batchSize, trainingDataSet);

            Log.Info("C# numEpochs= " + numEpochs);
            Log.Info("C# learningRate= " + learningRate);
            Log.Info("C# l2regularizer= " + lambdaL2Regularization);
            Log.Info("C# momentum= " + momentum);
            Log.Info(predict_before);
            Log.Info("C# metrics_before= " + Model.MetricsToString(lossAccuracyBefore, ""));
            Log.Info(predict_after);
            Log.Info("C# metrics_after= " + Model.MetricsToString(lossAccuracyAfter, ""));
        }

        private static string PredictionToString(Tensor prediction, string description)
        {
            var tmp = prediction.ToCpuFloat().ReadonlyContent;

            string result = description + " " + Tensor.ShapeToString(prediction.Shape) + Environment.NewLine;
            int idxMax = tmp.IndexOf(tmp.Max());
            result += description + "[" + idxMax + "]=" + tmp[idxMax] + Environment.NewLine;
            result += prediction.ToNumpy();
            return result;
        }

        [Test, Explicit]
        public void TestParallelRunWithTensorFlow_Convolution()
        {
            const int numEpochs = 10;
            const double learningRate = 0.01;
            const double lambdaL2Regularization = 0.00;
            const double momentum = 0.9;
            var X = TestNetworkPropagation.FromNumpyArray(TestNetworkPropagation.X_2_3_4_5);
            var Y = TestNetworkPropagation.FromNumpyArray(TestNetworkPropagation.Y_2_2);

            int batchSize = X.Shape[0];
            //var gpuDeviceId = -1;
            const int gpuDeviceId = 0;

            var sample = new NetworkSample
                    {
                        LossFunction = EvaluationMetricEnum.CategoricalCrossentropy,
                        ShuffleDatasetBeforeEachEpoch = false,
                        CompatibilityMode = NetworkSample.CompatibilityModeEnum.TensorFlow,
                        ResourceIds = new List<int> { gpuDeviceId }
                    }
                    .WithSGD(momentum, false);

            var network = new Network(sample, null, NetworkSample.DefaultWorkingDirectory, "TestParallelRunWithTensorFlow_Convolution", false);

            network.Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Convolution(2, 1, 1, ConvolutionLayer.PADDING_TYPE.SAME, lambdaL2Regularization, true)
                .Convolution(2, 1, 1, ConvolutionLayer.PADDING_TYPE.SAME, lambdaL2Regularization, true)
                .GlobalAvgPooling()
                .MultiplyLayer(1, 3)
                .Flatten().Dense(Y.Shape[1], lambdaL2Regularization, false)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);


            Log.Info(network.Summary() + Environment.NewLine);

            TestNetworkPropagation.FromConvNumpyArray("[[[[-0.4878799319267273, -0.6471760272979736], [-0.11215460300445557, 0.24113142490386963], [-0.5400518774986267, -0.8205036520957947]]]]").CopyTo(((ConvolutionLayer)network.Layers[1]).Weights);
            TestNetworkPropagation.FromConvNumpyArray("[[[[-0.7247111797332764, -0.3986714482307434], [-0.4940018653869629, 0.04389345645904541]]]]").CopyTo(((ConvolutionLayer)network.Layers[2]).Weights);
            TestNetworkPropagation.FromNumpyArray("[[-0.029460519552230835, 0.1628669798374176], [-0.28001704812049866, -0.23855498433113098], [0.07715305685997009, 0.11627233028411865], [0.32925912737846375, 0.011087954044342041], [0.12424156069755554, -0.05900973081588745], [-0.2703372836112976, 0.12233385443687439], [-0.08240920305252075, 0.006095200777053833], [-0.023135006427764893, 0.08786126971244812], [-0.2075882852077484, -0.3384675085544586], [0.10181871056556702, -0.08105111122131348], [0.04287368059158325, -0.014433145523071289], [-0.050517499446868896, 0.19285127520561218], [0.16756221652030945, -0.06256869435310364], [-0.1878374218940735, -0.17477598786354065], [0.3118181526660919, 0.36103251576423645], [0.16790542006492615, 0.27620890736579895], [0.21295377612113953, -0.15440134704113007], [0.03934970498085022, -0.35186851024627686], [-0.19449061155319214, -0.2855254113674164], [-0.08950188755989075, 0.2891680896282196], [-0.37375181913375854, 0.18617329001426697], [0.07124421000480652, 0.28268447518348694], [0.041756272315979004, 0.13584479689598083], [0.12497344613075256, 0.151188462972641], [0.3146173655986786, -0.22298070788383484], [-0.22048203647136688, -0.30460700392723083], [0.12072917819023132, -0.2646358907222748], [-0.15740737318992615, 0.17554828524589539], [0.13976749777793884, -0.357845664024353], [-0.365357369184494, -0.15716126561164856], [0.14519938826560974, 0.22951403260231018], [0.03488221764564514, 0.1870688498020172], [0.28289076685905457, 0.14199396967887878], [0.31583401560783386, 0.08595579862594604], [0.005727171897888184, 0.2800586521625519], [0.013508498668670654, 0.3192369043827057], [-0.14768590033054352, -0.05077126622200012], [-0.28260645270347595, -0.3034713864326477], [-0.05905658006668091, -0.3151003122329712], [-0.12471392750740051, -0.2689373791217804]]").CopyTo(((DenseLayer)network.Layers[6]).Weights);

            network.Sample.LogNetworkPropagation = true;
            var predict_before = network.Predict(X, false).ToNumpy();

            using var trainingDataSet = new InMemoryDataSet(X, Y);
            var lossAccuracyBefore = network.ComputeMetricsForValidationDataSet(batchSize, trainingDataSet);

            Log.Info("-");
            Log.Info("--------------------------------------------------------------------");
            Log.Info("-");

            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, batchSize);

            var predict_after = network.Predict(X, false).ToNumpy();
            var lossAccuracyAfter = network.ComputeMetricsForValidationDataSet(batchSize, trainingDataSet);

            Log.Info("C# numEpochs= " + numEpochs);
            Log.Info("C# learningRate= " + learningRate);
            Log.Info("C# l2regularizer= " + lambdaL2Regularization);
            Log.Info("C# momentum= " + momentum);
            Log.Info(predict_before);
            Log.Info("C# metrics_before= " + Model.MetricsToString(lossAccuracyBefore, ""));
            Log.Info(predict_after);
            Log.Info("C# metrics_after= " + Model.MetricsToString(lossAccuracyAfter, ""));
        }


        [Test, Explicit]
        public void TestParallelRunWithTensorFlow_DotProductAttention()
        {
            const int numEpochs = 1;
            const double learningRate = 1;
            const double lambdaL2Regularization = 0.00;
            const double momentum = 0.9;
            const bool use_scale = false;
            const bool use_causal_mask = false;

            var X = TestNetworkPropagation.FromNumpyArray(TestNetworkPropagation.X_3_4_5);
            //var X = TestNetworkPropagation.FromNumpyArray(TestNetworkPropagation.X_1_1_2);
            //var X = TestNetworkPropagation.FromNumpyArray(@"numpy.array([[[0.25,1.25]]], numpy.float)");
            //var X = TestNetworkPropagation.FromNumpyArray(@"numpy.array([[[0.25,1.25,2.5]]], numpy.float)");
            //var X = TestNetworkPropagation.FromNumpyArray(@"numpy.array([[[0.25,1.25,2.5]]], numpy.float)");
            //var X = TestNetworkPropagation.FromNumpyArray(@"numpy.array([[[0.25],[1.25]]], numpy.float)");
            var Y = TestNetworkPropagation.FromNumpyArray(TestNetworkPropagation.Y_3_3);
            //var Y = TestNetworkPropagation.FromNumpyArray(@"numpy.array([[1,0]], numpy.float)");

            int batchSize = X.Shape[0];
            //const int  gpuDeviceId = -1;
            const int gpuDeviceId = 0;
            var network = TestNetwork.NewForTests(
                        new NetworkSample
                        {
                            LossFunction = EvaluationMetricEnum.CategoricalCrossentropy,
                            ShuffleDatasetBeforeEachEpoch = false,
                            CompatibilityMode = NetworkSample.CompatibilityModeEnum.TensorFlow,
                            ConvolutionAlgoPreference = GPUWrapper.ConvolutionAlgoPreference.FASTEST_DETERMINIST_NO_TRANSFORM,
                            ResourceIds = new List<int> { gpuDeviceId }
                        }
                       .WithSGD(momentum, false),
                        NetworkSample.DefaultWorkingDirectory,
                        "TestParallelRunWithTensorFlow_DotProductAttention"
                );

            network.Input(X.Shape[1], X.Shape[2], -1);
            var lastLayerIndex = network.LastLayerIndex;

            var conv1D_Q = network.Conv1D(2, 1, 1, ConvolutionLayer.PADDING_TYPE.SAME, lambdaL2Regularization, true, lastLayerIndex, "conv1D_Q").Layers.Last();
            var conv1D_V = network.Conv1D(2, 1, 1, ConvolutionLayer.PADDING_TYPE.SAME, lambdaL2Regularization, true, lastLayerIndex, "conv1D_V").Layers.Last();
            var conv1D_K = network.Conv1D(2, 1, 1, ConvolutionLayer.PADDING_TYPE.SAME, lambdaL2Regularization, true, lastLayerIndex, "conv1D_K").Layers.Last();

            network.ScaledDotProductAttention(use_scale, use_causal_mask,conv1D_Q.LayerIndex, conv1D_V.LayerIndex, conv1D_K.LayerIndex);
            network.Flatten()
                .Dense(Y.Shape[1], lambdaL2Regularization, false)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);

            Log.Info(network.Summary() + Environment.NewLine);

            // Weights initialization
            TestNetworkPropagation.FromConvNumpyArray("[[[0.7163782119750977, 0.3252570629119873], [-0.2825784683227539, 0.8999791145324707], [-0.8438777923583984, 0.30466461181640625], [0.9409997463226318, -0.6641757488250732]]]").CopyTo(network.Layers[1].Weights);
            TestNetworkPropagation.FromConvNumpyArray("[[[0.9583117961883545, -0.035964250564575195], [0.5321958065032959, 0.4857454299926758], [0.6319985389709473, -0.7626423835754395], [0.40629100799560547, 0.058797597885131836]]]").CopyTo(network.Layers[2].Weights);
            TestNetworkPropagation.FromConvNumpyArray("[[[0.6856975555419922, -0.04618430137634277], [-0.23545622825622559, -0.6273543834686279], [-0.21123051643371582, 0.7190308570861816], [-0.8074820041656494, -0.12452530860900879]]]").CopyTo(network.Layers[3].Weights);
            TestNetworkPropagation.FromNumpyArray("[[0.429288387298584, 0.18538278341293335, 0.0015860199928283691], [0.24896937608718872, 0.48841190338134766, 0.25818514823913574], [0.4999527931213379, 0.5358568429946899, -0.41722971200942993], [0.24402379989624023, -0.3547265827655792, 0.3244849443435669], [-0.149910569190979, -0.19321483373641968, -0.03135275840759277], [-0.5615183115005493, -0.34524819254875183, -0.028048932552337646], [0.5381102561950684, -0.18948909640312195, 0.07540792226791382], [-0.11106348037719727, 0.6106008291244507, 0.4515535831451416], [-0.5627211332321167, -0.09199190139770508, 0.6016560792922974], [-0.6248162984848022, 0.653769850730896, 0.04975825548171997]]").CopyTo(network.Layers[6].Weights);

            network.Sample.LogNetworkPropagation = true;
            var predict_before = network.Predict(X, false).ToNumpy();

            using var trainingDataSet = new InMemoryDataSet(X, Y);
            var lossAccuracyBefore = network.ComputeMetricsForValidationDataSet(batchSize, trainingDataSet);

            Log.Info("-");
            Log.Info("--------------------------------------------------------------------");
            Log.Info("-");

            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, batchSize);

            var predict_after = network.Predict(X, false).ToNumpy();
            var lossAccuracyAfter = network.ComputeMetricsForValidationDataSet(batchSize, trainingDataSet);

            Log.Info("C# numEpochs= " + numEpochs);
            Log.Info("C# learningRate= " + learningRate);
            Log.Info("C# l2regularizer= " + lambdaL2Regularization);
            Log.Info("C# momentum= " + momentum);
            Log.Info(predict_before);
            Log.Info("C# metrics_before= " + Model.MetricsToString(lossAccuracyBefore, ""));
            Log.Info(predict_after);
            Log.Info("C# metrics_after= " + Model.MetricsToString(lossAccuracyAfter, ""));
        }


        [Test, Explicit]
        public void TestParallelRunWithTensorFlow_MultiHeadAttention()
        {
            const int numEpochs = 10;
            const double learningRate = 0.01;
            const double lambdaL2Regularization = 0.00;
            const double momentum = 0.9;
            const bool use_causal_mask = true;
            const int num_heads = 3;
            const int key_dim = 5;
            const int value_dim = 5;
            const bool use_bias_Q_V_K = true;
            const bool use_bias_O = true;
            const int embedding_dim = 15;

            var X = TestNetworkPropagation.numpy_array_for_tests(3, 7, embedding_dim);
            var Y = TestNetworkPropagation.y_numpy_array_for_tests(X.Shape[0], 3);

            int batchSize = X.Shape[0];
            //const int  gpuDeviceId = -1;
            const int gpuDeviceId = 0;
            var network = TestNetwork.NewForTests(
                        new NetworkSample
                        {
                            LossFunction = EvaluationMetricEnum.CategoricalCrossentropy,
                            ShuffleDatasetBeforeEachEpoch = false,
                            CompatibilityMode = NetworkSample.CompatibilityModeEnum.TensorFlow,
                            ConvolutionAlgoPreference = GPUWrapper.ConvolutionAlgoPreference.FASTEST_DETERMINIST_NO_TRANSFORM,
                            ResourceIds = new List<int> { gpuDeviceId }
                        }
                       .WithSGD(momentum, false),
                        NetworkSample.DefaultWorkingDirectory,
                        "TestParallelRunWithTensorFlow_DotProductAttention"
                );

            network.Input(X.Shape[1], X.Shape[2], -1);
            var lastLayerIndex = network.LastLayerIndex;

            var conv1D_Q = network.Conv1D(2, 1, 1, ConvolutionLayer.PADDING_TYPE.SAME, lambdaL2Regularization, true, lastLayerIndex, "conv1D_Q").Layers.Last();
            var conv1D_K = network.Conv1D(2, 1, 1, ConvolutionLayer.PADDING_TYPE.SAME, lambdaL2Regularization, true, lastLayerIndex, "conv1D_K").Layers.Last();
            var conv1D_V = network.Conv1D(2, 1, 1, ConvolutionLayer.PADDING_TYPE.SAME, lambdaL2Regularization, true, lastLayerIndex, "conv1D_V").Layers.Last();

            network.MultiHeadAttention(num_heads, key_dim, value_dim, use_bias_Q_V_K, use_bias_O,use_causal_mask, conv1D_Q.LayerIndex, conv1D_V.LayerIndex, conv1D_K.LayerIndex);
            network.Flatten()
                .Dense(Y.Shape[1], lambdaL2Regularization, false)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);

            Log.Info(network.Summary() + Environment.NewLine);

            var multiHead = (MultiHeadAttentionLayer)network.Layers[4];

            //3_4_5
            TestNetworkPropagation.FromConvNumpyArray("[[[0.5849204063415527, 0.2655712366104126], [-0.23072433471679688, 0.7348299026489258], [-0.6890233755111694, 0.24875760078430176], [0.7683230638504028, -0.5422972440719604], [0.03884774446487427, 0.2135295867919922], [-0.6753761768341064, -0.08759784698486328], [0.761029839515686, -0.04843282699584961]]]").CopyTo(conv1D_Q.Weights);
            TestNetworkPropagation.FromConvNumpyArray("[[[0.5598697662353516, -0.03770929574966431], [-0.19224923849105835, -0.5122327208518982], [-0.17246901988983154, 0.5870862007141113], [-0.6593062877655029, -0.10167449712753296], [-0.7024011611938477, -0.6345865726470947], [-0.26348328590393066, 0.22904479503631592], [-0.27368175983428955, 0.8041517734527588]]]").CopyTo(conv1D_K.Weights);
            TestNetworkPropagation.FromConvNumpyArray("[[[0.7824583053588867, -0.029364705085754395], [0.4345360994338989, 0.3966095447540283], [0.5160247087478638, -0.6226949095726013], [0.3317352533340454, 0.0480080246925354], [-0.6992490887641907, -0.31082260608673096], [-0.4510287344455719, 0.028438270092010498], [0.1867746114730835, 0.2283872365951538]]]").CopyTo(conv1D_V.Weights);
            TestNetworkPropagation.FromNumpyArray("[[[0.14129608869552612, 0.06101694703102112, 0.0005220323801040649, 0.08194583654403687, 0.16075602173805237], [0.08497914671897888, 0.164554625749588, 0.17637208104133606, -0.13732710480690002, 0.08031806349754333], [-0.11675480753183365, 0.1068010926246643, -0.049341604113578796, -0.06359478831291199, -0.010319441556930542]], [[-0.184818297624588, -0.1136350929737091, -0.009232044219970703, 0.17711377143859863, -0.062368497252464294], [0.02481977641582489, -0.036555469036102295, 0.20097333192825317, 0.148624449968338, -0.18521420657634735], [-0.03027823567390442, 0.19802924990653992, -0.20565220713615417, 0.21518200635910034, 0.01637744903564453]], [[0.13429030776023865, -0.016637876629829407, -0.1585068553686142, -0.15336453914642334, 0.1742730736732483], [0.19442406296730042, -0.1476115882396698, -0.03624379634857178, 0.19438642263412476, 0.03112637996673584], [0.0476759672164917, -0.10107632726430893, 0.0803544819355011, 0.10934033989906311, 0.028994977474212646]], [[0.17840039730072021, -0.1958729773759842, -0.20492978394031525, 0.11065426468849182, -0.11021459102630615], [-0.09652343392372131, -0.161655455827713, 0.22003498673439026, -0.03075048327445984, 0.026907742023468018], [0.03736254572868347, 0.03590479493141174, 0.14915212988853455, 0.21704867482185364, 0.11135169863700867]], [[0.1951104998588562, 0.12457019090652466, 0.06174325942993164, -0.15019002556800842, -0.05161893367767334], [0.062190234661102295, -0.06326925754547119, 0.16584354639053345, -0.07279686629772186, 0.023707956075668335], [-0.17666231095790863, 0.06419098377227783, -0.21906882524490356, 0.16570910811424255, -0.20316238701343536]], [[0.19276532530784607, 0.07132449746131897, 0.04683753848075867, -0.19788172841072083, 0.15723547339439392], [0.09295877814292908, -0.06603087484836578, -0.14335766434669495, 0.19258880615234375, -0.2105855494737625], [0.14123263955116272, -0.06240512430667877, -0.024723708629608154, -0.1908400058746338, -0.00527527928352356]], [[-0.04641781747341156, 0.1885678470134735, -0.0006330758333206177, -0.05687315762042999, 0.03296127915382385], [0.20773836970329285, -0.006775528192520142, 0.060211390256881714, -0.13010354340076447, 0.13610312342643738], [-0.0803108662366867, -0.028895169496536255, 0.03979167342185974, -0.21285882592201233, -0.1136946976184845]], [[-0.13533392548561096, 0.13910260796546936, 0.03203651309013367, 0.1674482524394989, 0.18679416179656982], [-0.07577316462993622, -0.2152363806962967, -0.20256875455379486, -0.09354333579540253, -0.12266460061073303], [0.19916856288909912, -0.1675351858139038, 0.11004975438117981, 0.21152633428573608, 0.17826193571090698]], [[0.1231391429901123, -0.137430801987648, -0.030454382300376892, -0.058268651366233826, 0.07569533586502075], [-0.07693451642990112, -0.08258238434791565, -0.19373852014541626, 0.19636237621307373, 0.1281619668006897], [0.19887647032737732, -0.09771405160427094, -0.048017486929893494, -0.06773446500301361, -0.2051570564508438]], [[0.10087379813194275, -0.08970290422439575, 0.12239265441894531, -0.07138404250144958, -0.10118183493614197], [-0.06366051733493805, 0.20006000995635986, -0.07048913836479187, 0.06967902183532715, 0.1851520836353302], [-0.013733446598052979, -0.048352718353271484, -0.1034972220659256, -0.06287981569766998, -0.021217644214630127]], [[0.09138256311416626, -0.0012300163507461548, -0.04318492114543915, -0.18708091974258423, 0.003314465284347534], [-0.13249865174293518, -0.11865697801113129, 0.1156783401966095, 0.06178063154220581, -0.08980275690555573], [-0.18052594363689423, -0.1012938916683197, -0.21195289492607117, 0.0409967303276062, 0.09625864028930664]], [[0.141539067029953, 0.04887726902961731, -0.1331932544708252, -0.041054949164390564, -0.21199938654899597], [0.10720860958099365, -0.1828010380268097, -0.13069866597652435, -0.11996226757764816, -0.07573755085468292], [0.10687410831451416, 0.005361795425415039, 0.21887439489364624, -0.17264023423194885, -0.15378202497959137]], [[0.18625420331954956, -0.2179793417453766, -0.09786683320999146, 0.1533128321170807, -0.2081925868988037], [0.11204153299331665, -0.01466570794582367, -0.1503351330757141, -0.05215167999267578, 0.1703748106956482], [-0.0066549330949783325, -0.0006037652492523193, 0.00012965500354766846, -0.1913059502840042, -0.221401646733284]], [[0.07095193862915039, -0.01626528799533844, 0.05066266655921936, 0.09122154116630554, 0.0008416324853897095], [-0.20448404550552368, 0.1749488115310669, -0.1536933183670044, 0.17421528697013855, -0.09935083240270615], [0.21947842836380005, -0.20133745670318604, 0.18136203289031982, -0.1197647973895073, 0.20824244618415833]], [[-0.2102494239807129, -0.01064331829547882, -0.22024120390415192, -0.12148506939411163, 0.09896516799926758], [-0.1519101858139038, -0.19189974665641785, -0.12863415479660034, -0.009472638368606567, -0.07009276747703552], [-0.014478370547294617, -0.22000625729560852, -0.09644857048988342, -0.0980745404958725, 0.2194276750087738]]]").CopyTo(multiHead.w_Q);
            TestNetworkPropagation.FromNumpyArray("[[[0.22165775299072266, -0.13717442750930786, 0.145480215549469, -0.20221027731895447, 0.08572754263877869], [-0.1926579475402832, -0.10609965026378632, -0.21877357363700867, 0.13545867800712585, -0.19719330966472626], [0.08979624509811401, 0.09754595160484314, -0.07750697433948517, 0.08985671401023865, -0.047381266951560974]], [[-0.06491740047931671, 0.0688682496547699, -0.1725246012210846, -0.043703436851501465, 0.13100340962409973], [-0.045523613691329956, 0.07025450468063354, 0.011367395520210266, -0.17512735724449158, -0.10255935788154602], [-0.025475144386291504, -0.024232283234596252, 0.0047693997621536255, 0.01170012354850769, 0.2219349443912506]], [[-0.015504896640777588, -0.04346106946468353, 0.00345514714717865, 0.023637697100639343, 0.1937078833580017], [-0.022654250264167786, 0.08266991376876831, 0.0726071298122406, -0.18153034150600433, 0.17230117321014404], [-0.15771682560443878, -0.014823779463768005, -0.11794073134660721, 0.15371805429458618, -0.1722114533185959]], [[0.2025093138217926, 0.0007508993148803711, -0.18187010288238525, -0.08609715104103088, 0.12632164359092712], [-0.04476194083690643, 0.21636557579040527, 0.0465521514415741, 0.08441680669784546, -0.20685525238513947], [-0.08384561538696289, -0.052330002188682556, -0.1517188996076584, -0.1566345989704132, 0.05165603756904602]], [[-0.09660792350769043, 0.049300432205200195, -0.1694292575120926, -0.07399728894233704, 0.12031605839729309], [0.10646218061447144, -0.10674179345369339, 0.03866139054298401, 0.1979830265045166, -0.009777799248695374], [-0.11459390819072723, -0.18785347044467926, 0.06357249617576599, 0.08139374852180481, -0.030373886227607727]], [[0.0643276572227478, -0.15723296999931335, -0.2086613029241562, -0.13947126269340515, -0.2164202779531479], [-0.012708574533462524, 0.1253879964351654, -0.18731874227523804, -0.2220376580953598, 0.0402812659740448], [0.02814638614654541, 0.019313395023345947, 0.16389989852905273, -0.13113999366760254, 0.18701592087745667]], [[0.15522703528404236, -0.18916073441505432, -0.0946551114320755, 0.09302514791488647, -0.16030502319335938], [0.0026899129152297974, 0.07534128427505493, -0.040110886096954346, -0.1513911932706833, -0.13091787695884705], [0.03603893518447876, -0.05386362969875336, -0.10420782119035721, 0.11023479700088501, -0.08642107248306274]], [[-0.13248217105865479, 0.09453019499778748, -0.07274563610553741, -0.14518797397613525, -0.07940275967121124], [0.045930176973342896, 0.1208493709564209, -0.11141326278448105, -0.21623539924621582, -0.12483178824186325], [0.07820525765419006, 0.02550072968006134, 0.06287497282028198, -0.09647880494594574, -0.04357297718524933]], [[0.06735461950302124, -0.08932235836982727, -0.07495535910129547, 0.1100597083568573, -0.18063022196292877], [-0.18526601791381836, -0.09791278839111328, 0.06713148951530457, -0.057783618569374084, 0.049710988998413086], [-0.2078278809785843, 0.20759937167167664, -0.06706587970256805, 0.02048535645008087, -0.2155580073595047]], [[-0.05354557931423187, -0.04738670587539673, -0.08226561546325684, 0.029787302017211914, -0.1437055766582489], [0.10576122999191284, -0.11909259110689163, -0.0027126818895339966, 0.2223045527935028, -0.1511131227016449], [-0.05552515387535095, -0.07811148464679718, 0.10184839367866516, 0.07856622338294983, 0.1894845962524414]], [[-0.05963540077209473, -0.1778710037469864, 0.17548412084579468, 0.14361035823822021, -0.11725231260061264], [-0.11915411055088043, -0.09617748856544495, -0.0872572660446167, 0.1613156497478485, -0.19319890439510345], [-0.09903367608785629, 0.009705498814582825, 0.1998763382434845, 0.06829455494880676, -0.05861181020736694]], [[0.10541832447052002, -0.0030817538499832153, -0.18386349081993103, 0.016329795122146606, 0.10751402378082275], [-0.18489891290664673, 0.10827377438545227, 0.04326152801513672, -0.10764708369970322, 0.09282907843589783], [0.0853399932384491, -0.06369015574455261, 0.15600785613059998, -0.14769592881202698, -0.10395368188619614]], [[-0.15155497193336487, 0.02631261944770813, -0.009926751255989075, 0.1755029857158661, -0.021882236003875732], [-0.0741024762392044, -0.015443846583366394, -0.060976848006248474, 0.17465686798095703, -0.10299241542816162], [-0.19839420914649963, -0.013941839337348938, 0.022107481956481934, -0.013399392366409302, 0.11455899477005005]], [[-0.005638539791107178, -0.009691104292869568, -0.20390842854976654, -0.20642513036727905, -0.14798855781555176], [0.15279090404510498, 0.15521585941314697, 0.006782084703445435, -0.03192339837551117, 0.10671243071556091], [-0.20793946087360382, 0.14368292689323425, 0.08629092574119568, -0.20751525461673737, -0.007837831974029541]], [[-0.1708669811487198, 0.047981828451156616, -0.16512782871723175, -0.08766995370388031, -0.13225671648979187], [0.0180598646402359, -0.08269935846328735, 0.1976669728755951, -0.21748076379299164, -0.13776922225952148], [-0.1490776687860489, 0.06421053409576416, 0.16731053590774536, -0.11375509947538376, 0.14364123344421387]]]").CopyTo(multiHead.w_K);
            TestNetworkPropagation.FromNumpyArray("[[[0.1554521918296814, -0.10820861905813217, -0.03322117030620575, 0.04522201418876648, 0.20756658911705017], [-0.07004424929618835, 0.21449828147888184, 0.20512714982032776, -0.016719713807106018, 0.11969932913780212], [-0.029643133282661438, 0.1971154808998108, -0.21458543837070465, -0.20338240265846252, 0.03531813621520996]], [[-0.09817631542682648, 0.20038428902626038, -0.02066454291343689, 0.2158324122428894, 0.016411036252975464], [-0.00013850629329681396, 0.04106646776199341, -0.1721615493297577, 0.17453432083129883, 0.015325173735618591], [0.2129266858100891, 0.06103515625, 0.21724140644073486, 0.13021516799926758, 0.0783892273902893]], [[-0.11988283693790436, -0.10003354400396347, 0.13907968997955322, 0.17965257167816162, 0.1315813660621643], [-0.019589826464653015, -0.053790390491485596, -0.2008848488330841, -0.08832025527954102, -0.02662886679172516], [-0.05420558154582977, -0.07586880028247833, 0.20590108633041382, 0.0038230568170547485, 0.046146929264068604]], [[0.07858601212501526, 0.1310487687587738, 0.21739652752876282, 0.191181480884552, -0.08531393110752106], [0.05241537094116211, -0.09538435935974121, -0.10498100519180298, -0.04141618311405182, 0.11744600534439087], [-0.04420727491378784, 0.18612796068191528, 0.13731533288955688, 0.126118004322052, -0.06757132709026337]], [[0.16862446069717407, 0.22255867719650269, -0.1451176404953003, 0.17635202407836914, 0.15641522407531738], [-0.17844650149345398, -0.03235650062561035, -0.20060794055461884, -0.05774778127670288, 0.15317457914352417], [-0.04994994401931763, 0.1910828948020935, 0.003880739212036133, 0.09135425090789795, -0.19467559456825256]], [[-0.051625534892082214, 0.18276360630989075, -0.007838413119316101, 0.16586464643478394, -0.12149664014577866], [-0.08461757004261017, 0.05451211333274841, 0.10606509447097778, -0.04040783643722534, 0.16905570030212402], [0.004343599081039429, -0.12476819008588791, 0.09750041365623474, 0.02917996048927307, 0.15605375170707703]], [[0.09300583600997925, -0.1716543436050415, 0.08207389712333679, 0.012580141425132751, -0.11522293835878372], [0.2043442726135254, -0.06721669435501099, -0.0356346070766449, -0.010127529501914978, 0.0341050922870636], [0.1993359625339508, 0.06587553024291992, -0.18645909428596497, 0.024419888854026794, -0.015844598412513733]], [[0.025245800614356995, -0.12486894428730011, -0.18634596467018127, -0.20803596079349518, -0.16400113701820374], [-0.15234771370887756, 0.11437126994132996, 0.18892920017242432, 0.22117242217063904, -0.18268567323684692], [0.16469714045524597, -0.10191012918949127, -0.1045779138803482, -0.14148229360580444, 0.18138709664344788]], [[-0.21043761074543, -0.17088249325752258, -0.16813510656356812, 0.19768008589744568, 0.10788092017173767], [0.04390457272529602, -0.172441765666008, -0.21733452379703522, 0.2205691933631897, -0.205651193857193], [-0.21321718394756317, -0.116811104118824, -0.017902284860610962, 0.18617260456085205, 0.0833391547203064]], [[-0.09193822741508484, -0.15085962414741516, 0.02270345389842987, -0.086122527718544, 0.13172075152397156], [0.02536943554878235, -0.09451505541801453, -0.18745538592338562, 0.0912102460861206, 0.105618417263031], [0.1865694522857666, -0.05070824921131134, 0.06738793849945068, -0.16406287252902985, 0.15485280752182007]], [[-0.22350864112377167, 0.05583608150482178, 0.030397862195968628, 0.030637353658676147, -0.20120108127593994], [-0.11392238736152649, -0.182485431432724, 0.08139964938163757, 0.011268407106399536, -0.11563130468130112], [-0.08572587370872498, 0.2066521942615509, 0.14991620182991028, -0.15597549080848694, -0.03149642050266266]], [[0.07908079028129578, 0.14496949315071106, 0.20769327878952026, 0.14112675189971924, 0.1210641860961914], [-0.15839862823486328, -0.12694038450717926, 0.22341999411582947, 0.09240305423736572, -0.07230862975120544], [0.0737571120262146, -0.05935834348201752, 0.21741130948066711, -0.09267020225524902, -0.15007326006889343]], [[-0.1403738409280777, 0.06355094909667969, 0.21900463104248047, -0.2057722806930542, 0.08042687177658081], [0.1114797294139862, -0.04917585849761963, 0.09080401062965393, -0.03161023557186127, -0.09623260796070099], [0.0674653947353363, 0.10248696804046631, 0.1498968005180359, 0.07014405727386475, 0.18755075335502625]], [[0.16377323865890503, -0.06884051859378815, 0.14213553071022034, 0.2062801718711853, 0.19333046674728394], [0.1752675175666809, 0.08705189824104309, 0.12010839581489563, 0.0664997398853302, -0.12086084485054016], [-0.13989999890327454, -0.16159787774085999, -0.10826011747121811, -0.03301265835762024, 0.19985449314117432]], [[-0.16318050026893616, 0.2036837637424469, 0.20256251096725464, -0.17527897655963898, 0.03682836890220642], [-0.01942785084247589, 0.13524198532104492, 0.026087597012519836, -0.14883354306221008, -0.16815344989299774], [0.07262930274009705, -0.12294614315032959, 0.023978352546691895, -0.2059035748243332, -0.16005274653434753]]]").CopyTo(multiHead.w_V);
            TestNetworkPropagation.FromNumpyArray("[[[0.25226590037345886, -0.06620186567306519, 0.27780595421791077, -0.2058069109916687, 0.06229478120803833, 0.3095511496067047, 0.3063715994358063, 0.1941225826740265, -0.11876794695854187, -0.0451924204826355, 0.172054260969162, 0.18268513679504395, 0.04308399558067322, -0.17067740857601166, 0.07575270533561707], [-0.2174902856349945, 0.03936713933944702, -0.16195440292358398, 0.28038862347602844, 0.07020238041877747, -0.10429452359676361, 0.028418779373168945, 0.19040915369987488, 0.006456553936004639, 0.19566640257835388, -0.0034545063972473145, -0.2764945924282074, 0.0971822440624237, -0.280710369348526, 0.1745675504207611], [-0.05749174952507019, -0.2946738004684448, 0.17463132739067078, 0.13916194438934326, -0.1739022582769394, 0.30906942486763, -0.03304716944694519, 0.2734052240848541, -0.013490110635757446, 0.05475643277168274, 0.12336483597755432, 0.2898010313510895, -0.2343399077653885, -0.19622892141342163, -0.10000871121883392], [0.01572588086128235, 0.19403597712516785, 0.23648908734321594, 0.02909800410270691, -0.13199253380298615, -0.20034825801849365, -0.24272650480270386, 0.26005759835243225, -0.22111719846725464, -0.3050805926322937, -0.03484615683555603, -0.05329039692878723, 0.19618389010429382, 0.11220794916152954, -0.04755061864852905], [-0.25686824321746826, 0.28159037232398987, 0.02097424864768982, -0.0782022625207901, 0.2709979712963104, -0.0030314624309539795, 0.04784396290779114, 0.08569890260696411, 0.20086750388145447, -0.27342256903648376, -0.11570903658866882, -0.18889650702476501, 0.3069729506969452, 0.20271137356758118, -0.26201364398002625]], [[-0.22174176573753357, -0.009068995714187622, 0.1385636031627655, -0.11785988509654999, -0.3082945942878723, 0.2634727656841278, 0.31161239743232727, -0.24749377369880676, 0.12172892689704895, -0.2211729884147644, 0.1254969835281372, 0.30133911967277527, 0.22052696347236633, -0.009212851524353027, -0.109503835439682], [-0.1633254587650299, -0.16653583943843842, -0.1266888976097107, -0.00199735164642334, 0.08238756656646729, 0.08940914273262024, -0.2522202730178833, 0.20541611313819885, 0.052709996700286865, 0.2561238706111908, 0.01781436800956726, 0.2611256539821625, -0.24531525373458862, 0.17461761832237244, 0.14896151423454285], [-0.1323377639055252, 0.19217458367347717, 0.07917892932891846, 0.31033143401145935, 0.01425197720527649, -0.1485365778207779, 0.3134109675884247, 0.2753503620624542, -0.10601012408733368, 0.24949303269386292, 0.031903207302093506, -0.276213675737381, 0.26704803109169006, 0.2959626615047455, -0.21378952264785767], [0.20256510376930237, -0.15764470398426056, 0.06065720319747925, 0.12086278200149536, -0.22762496769428253, -0.2980252504348755, -0.10440558195114136, -0.04366680979728699, -0.015794724225997925, 0.25475093722343445, -0.0757995992898941, 0.15546995401382446, 0.11269897222518921, -0.013426721096038818, 0.22158196568489075], [0.04439043998718262, -0.2915978729724884, -0.1614222675561905, -0.12113186717033386, -0.24576881527900696, -0.11869481205940247, 0.2828429043292999, -0.2768861949443817, 0.006989985704421997, -0.1086135059595108, 0.28116998076438904, -0.22682751715183258, -0.10127156972885132, 0.1389853060245514, -0.07160629332065582]], [[-0.12208446860313416, -0.1382014900445938, 0.30382272601127625, 0.28509023785591125, 0.12687504291534424, 0.08103236556053162, 0.2581331431865692, -0.07364119589328766, 0.18912509083747864, -0.2014368176460266, 0.22105494141578674, -0.12012737989425659, 0.25139889121055603, -0.009560853242874146, -0.19701552391052246], [-0.2103867530822754, -0.24533598124980927, -0.08097761869430542, 0.004114270210266113, -0.06745348870754242, 0.25003448128700256, -0.012592852115631104, 0.2171938717365265, 0.16868352890014648, -0.23029589653015137, 0.16276052594184875, 0.31622037291526794, 0.17517930269241333, 0.04421424865722656, -0.1763058453798294], [-0.19234611093997955, -0.20608851313591003, -0.06101787090301514, -0.1632440984249115, 0.018168270587921143, -0.20071113109588623, 0.2628267705440521, 0.19960948824882507, 0.1034928560256958, -0.25251829624176025, 0.1836385726928711, 0.18507930636405945, -0.26981931924819946, 0.3020854890346527, 0.1894231140613556], [-0.30384165048599243, -0.15589690208435059, 0.30229446291923523, -0.23486262559890747, 0.2869913876056671, 0.029342442750930786, -0.2758048176765442, 0.04614576697349548, -0.25216537714004517, -0.316059947013855, -0.15455706417560577, -0.1664208620786667, -0.23605695366859436, 0.22785916924476624, -0.21244728565216064], [0.2691017687320709, -0.19351540505886078, 0.16510158777236938, -0.10463771224021912, -0.06294867396354675, 0.175502210855484, -0.16798129677772522, -0.2886081635951996, 0.06472474336624146, -0.3057664632797241, 0.006179124116897583, 0.23513951897621155, 0.011693775653839111, -0.1188805103302002, -0.26392653584480286]]]").CopyTo(multiHead.w_O);
            TestNetworkPropagation.FromNumpyArray("[[-0.007786303758621216, -0.02810618281364441, 0.40045446157455444], [0.32519781589508057, 0.35607701539993286, 0.33807969093322754], [-0.28860679268836975, 0.19483697414398193, 0.05582690238952637], [0.04753980040550232, 0.16049432754516602, 0.28472840785980225], [0.05951428413391113, 0.3297981023788452, 0.1207876205444336], [-0.0946618914604187, 0.0776529312133789, -0.3335267901420593], [0.3699030876159668, -0.37217846512794495, -0.08660373091697693], [0.21457314491271973, 0.1802307367324829, -0.3466717600822449], [0.1851181983947754, -0.1656588912010193, -0.15638414025306702], [0.31537723541259766, -0.09058442711830139, 0.2061343789100647], [-0.0956345796585083, -0.2636573314666748, 0.02025464177131653], [0.17130160331726074, -0.12299245595932007, 0.3903971314430237], [0.2945582866668701, 0.06853127479553223, 0.4187239408493042], [0.029855698347091675, -0.11359450221061707, -0.23052382469177246], [0.09309834241867065, 0.37347692251205444, -0.18440993130207062], [0.2694540023803711, 0.3417733907699585, 0.3635774850845337], [-0.2550943195819855, 0.31904083490371704, -0.1524713933467865], [-0.16129493713378906, -0.39779677987098694, -0.11349740624427795], [0.2410150170326233, 0.359241783618927, 0.3464030623435974], [-0.0584568977355957, 0.11081117391586304, -0.39038553833961487], [-0.14634248614311218, -0.2199200689792633, -0.09241780638694763], [-0.21515537798404694, 0.1237913966178894, -0.42633190751075745], [0.1617395281791687, 0.11569511890411377, -0.23396721482276917], [0.12553459405899048, -0.3596823811531067, -0.07172012329101562], [-0.017882049083709717, 0.19433408975601196, 0.2697983980178833], [-0.3990972638130188, -0.02810811996459961, -0.2289561927318573], [0.18411916494369507, -0.16928741335868835, 0.358676016330719], [0.08056449890136719, 0.24910050630569458, -0.2432558685541153], [-0.18793028593063354, -0.4143126904964447, 0.16822582483291626], [0.08330917358398438, -0.24693197011947632, 0.002164602279663086]]").CopyTo(network.Layers[6].Weights);

            network.Sample.LogNetworkPropagation = true;
            var predict_before = network.Predict(X, false).ToNumpy();

            using var trainingDataSet = new InMemoryDataSet(X, Y);
            var lossAccuracyBefore = network.ComputeMetricsForValidationDataSet(batchSize, trainingDataSet);

            Log.Info("-");
            Log.Info("--------------------------------------------------------------------");
            Log.Info("-");

            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, batchSize);

            var predict_after = network.Predict(X, false).ToNumpy();
            var lossAccuracyAfter = network.ComputeMetricsForValidationDataSet(batchSize, trainingDataSet);

            Log.Info("C# numEpochs= " + numEpochs);
            Log.Info("C# learningRate= " + learningRate);
            Log.Info("C# l2regularizer= " + lambdaL2Regularization);
            Log.Info("C# momentum= " + momentum);
            Log.Info(predict_before);
            Log.Info("C# metrics_before= " + Model.MetricsToString(lossAccuracyBefore, ""));
            Log.Info(predict_after);
            Log.Info("C# metrics_after= " + Model.MetricsToString(lossAccuracyAfter, ""));
        }




        [Test, Explicit]
        public void Test_Speed_MultiHeadAttention()
        {
            const int numEpochs = 50;
            const double learningRate = 0.01;
            const double lambdaL2Regularization = 0.00;
            const double momentum = 0.9;
            const bool use_causal_mask = true;
            const int num_heads = 8;
            const int key_dim = 64;
            const int value_dim = 64;
            const bool use_bias_Q_V_K = true;
            const bool use_bias_O = true;
            const int embedding_dim = 512;

            var X = TestNetworkPropagation.numpy_array_for_tests(1024, 128, embedding_dim);
            var Y = TestNetworkPropagation.y_numpy_array_for_tests(X.Shape[0], 3);

            const int batchSize = 32;
            //const int  gpuDeviceId = -1;
            const int gpuDeviceId = 0;
            var network = TestNetwork.NewForTests(
                        new NetworkSample
                        {
                            LossFunction = EvaluationMetricEnum.CategoricalCrossentropy,
                            ShuffleDatasetBeforeEachEpoch = false,
                            CompatibilityMode = NetworkSample.CompatibilityModeEnum.TensorFlow,
                            ConvolutionAlgoPreference = GPUWrapper.ConvolutionAlgoPreference.FASTEST_DETERMINIST_NO_TRANSFORM,
                            ResourceIds = new List<int> { gpuDeviceId }
                        }
                       .WithSGD(momentum, false),
                        NetworkSample.DefaultWorkingDirectory,
                        "TestParallelRunWithTensorFlow_MultiHeadAttention"
                );

            network.Input(X.Shape[1], X.Shape[2], -1);
            var lastLayerIndex = network.LastLayerIndex;

            var conv1D_Q = network.Conv1D(2, 1, 1, ConvolutionLayer.PADDING_TYPE.SAME, lambdaL2Regularization, true, lastLayerIndex, "conv1D_Q").Layers.Last();
            var conv1D_K = network.Conv1D(2, 1, 1, ConvolutionLayer.PADDING_TYPE.SAME, lambdaL2Regularization, true, lastLayerIndex, "conv1D_K").Layers.Last();
            var conv1D_V = network.Conv1D(2, 1, 1, ConvolutionLayer.PADDING_TYPE.SAME, lambdaL2Regularization, true, lastLayerIndex, "conv1D_V").Layers.Last();

            network.MultiHeadAttention(num_heads, key_dim, value_dim, use_bias_Q_V_K, use_bias_O, use_causal_mask, conv1D_Q.LayerIndex, conv1D_V.LayerIndex, conv1D_K.LayerIndex);
            network.Flatten()
                .Dense(Y.Shape[1], lambdaL2Regularization, false)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);

            Log.Info(network.Summary() + Environment.NewLine);


            network.Sample.LogNetworkPropagation = false;
            var predict_before = network.Predict(X, false).ToNumpy();

            using var trainingDataSet = new InMemoryDataSet(X, Y);
            var lossAccuracyBefore = network.ComputeMetricsForValidationDataSet(batchSize, trainingDataSet);

            Log.Info("-");
            Log.Info("--------------------------------------------------------------------");
            Log.Info("-");

            var sp = Stopwatch.StartNew();

            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, batchSize);

            var predict_after = network.Predict(X, false).ToNumpy();
            var lossAccuracyAfter = network.ComputeMetricsForValidationDataSet(batchSize, trainingDataSet);

            Log.Info("C# numEpochs= " + numEpochs);
            Log.Info("C# learningRate= " + learningRate);
            Log.Info("C# l2regularizer= " + lambdaL2Regularization);
            Log.Info("C# momentum= " + momentum);
            Log.Info(predict_before);
            Log.Info("C# metrics_before= " + Model.MetricsToString(lossAccuracyBefore, ""));
            Log.Info(predict_after);
            Log.Info("C# metrics_after= " + Model.MetricsToString(lossAccuracyAfter, ""));
            Log.Info("Training took " + sp.ElapsedMilliseconds + " ms");

        }
        [Test, Explicit]
        public void TestParallelRunWithTensorFlow_Conv1D()
        {
            const int numEpochs = 10;
            const double learningRate = 0.01;
            const double lambdaL2Regularization = 0.00;
            const double momentum = 0.9;
            var X = TestNetworkPropagation.FromNumpyArray(TestNetworkPropagation.X_3_4_5);
            var Y = TestNetworkPropagation.FromNumpyArray(TestNetworkPropagation.Y_3_3);

            int batchSize = X.Shape[0];
            //const int  gpuDeviceId = -1;
            const int gpuDeviceId = 0;
            var network = TestNetwork.NewForTests(
                        new NetworkSample
                        {
                            LossFunction = EvaluationMetricEnum.CategoricalCrossentropy,
                            ShuffleDatasetBeforeEachEpoch = false,
                            CompatibilityMode = NetworkSample.CompatibilityModeEnum.TensorFlow,
                            ResourceIds = new List<int> { gpuDeviceId }
                        }
                       .WithSGD(momentum, false),
                        NetworkSample.DefaultWorkingDirectory,
                        "TestParallelRunWithTensorFlow_Convolution"
                );

            network.Input(X.Shape[1], X.Shape[2], -1)
                .Conv1D(2, 3, 1, ConvolutionLayer.PADDING_TYPE.VALID, lambdaL2Regularization, true)
                .Conv1D(2, 3, 2, ConvolutionLayer.PADDING_TYPE.CAUSAL, lambdaL2Regularization, true)
                .Conv1D(2, 1, 1, ConvolutionLayer.PADDING_TYPE.SAME, lambdaL2Regularization, true)
                .Flatten().Dense(Y.Shape[1], lambdaL2Regularization, false)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);

            Log.Info(network.Summary() + Environment.NewLine);

            //network.Layers[1].Weights.ZeroMemory();
            TestNetworkPropagation.FromConvNumpyArray("[[[0.09934896230697632, -0.11215364933013916], [0.3982505798339844, 0.342079758644104], [-0.06867659091949463, -0.46536481380462646], [0.2547714114189148, -0.08702009916305542]], [[-0.5021747350692749, -0.1221388578414917], [-0.3608691096305847, 0.3861338496208191], [0.10946327447891235, -0.052802085876464844], [-0.016413629055023193, 0.3857215642929077]], [[0.4184006452560425, -0.2657143771648407], [0.296006977558136, -0.28657031059265137], [-0.016508877277374268, -0.2890245020389557], [0.1388271450996399, 0.02789127826690674]]]").CopyTo(network.Layers[1].Weights);
            //network.Layers[2].Weights.ZeroMemory();
            TestNetworkPropagation.FromConvNumpyArray("[[[0.39741700887680054, 0.5679424405097961], [0.103904128074646, 0.46203213930130005]], [[0.5664966702461243, -0.5104600191116333], [-0.4302336871623993, 0.2359222173690796]], [[0.1441558599472046, -0.3472554683685303], [0.3229832053184509, -0.13790547847747803]]]").CopyTo(network.Layers[2].Weights);
            TestNetworkPropagation.FromConvNumpyArray("[[[-1.0770841836929321, 0.557166576385498], [0.405431866645813, -0.2015085220336914]]]").CopyTo(network.Layers[3].Weights);
            TestNetworkPropagation.FromNumpyArray("[[0.38363194465637207, 0.2582963705062866, 0.15701913833618164], [0.5796942710876465, -0.42992860078811646, 0.28377270698547363], [-0.34947991371154785, 0.8033483028411865, -0.22690773010253906], [0.8054455518722534, 0.22870910167694092, -0.36302077770233154]]").CopyTo(network.Layers[5].Weights);
            //TestNetworkPropagation.FromConvNumpyArray("[[[0.4353485107421875, 0.6221498250961304, 0.11382126808166504], [0.5061308145523071, 0.6205660104751587, -0.5591809749603271]], [[-0.47129738330841064, 0.25843989849090576, 0.1579148769378662], [-0.38039931654930115, 0.3538104295730591, -0.15106791257858276]]]").CopyTo(network.Layers[2].Weights);
            //TestNetworkPropagation.FromNumpyArray("[[0.39741700887680054, 0.5679424405097961, 0.103904128074646], [0.46203213930130005, 0.5664966702461243, -0.5104600191116333], [-0.4302336871623993, 0.2359222173690796, 0.1441558599472046], [-0.3472554683685303, 0.3229832053184509, -0.13790547847747803], [0.31644493341445923, -0.011439502239227295, -0.4673982560634613], [-0.6368072032928467, 0.23920577764511108, 0.265876829624176], [0.4810141921043396, 0.5506053566932678, -0.6353087425231934], [-0.13411635160446167, 0.4802754521369934, 0.136569082736969], [-0.1479487419128418, 0.23149830102920532, 0.14344310760498047]]").CopyTo(network.Layers[4].Weights);

            network.Sample.LogNetworkPropagation = true;
            var predict_before = network.Predict(X, false).ToNumpy();

            using var trainingDataSet = new InMemoryDataSet(X, Y);
            var lossAccuracyBefore = network.ComputeMetricsForValidationDataSet(batchSize, trainingDataSet);

            Log.Info("-");
            Log.Info("--------------------------------------------------------------------");
            Log.Info("-");

            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, batchSize);

            var predict_after = network.Predict(X, false).ToNumpy();
            var lossAccuracyAfter = network.ComputeMetricsForValidationDataSet(batchSize, trainingDataSet);

            Log.Info("C# numEpochs= " + numEpochs);
            Log.Info("C# learningRate= " + learningRate);
            Log.Info("C# l2regularizer= " + lambdaL2Regularization);
            Log.Info("C# momentum= " + momentum);
            Log.Info("C# use_causal_mask= " + momentum);
            Log.Info(predict_before);
            Log.Info("C# metrics_before= " + Model.MetricsToString(lossAccuracyBefore, ""));
            Log.Info(predict_after);
            Log.Info("C# metrics_after= " + Model.MetricsToString(lossAccuracyAfter, ""));
        }
        [Test, Explicit]
        public void TestParallelRunWithTensorFlow_Embedding()
        {
            const int numEpochs = 10;
            const double learningRate = 0.1;
            const double lambdaL2Regularization = 0.00;
            const double momentum = 0.9;
            const int vocabularySize = 3;
            const int embeddingDim = 5;
            const int maxWordsBySentence = 4;

            var X = TestNetworkPropagation.FromNumpyArray(@"numpy.array([[1, 2, 1, 1], [2, 2, 1, 1]], numpy.float)");
            var Y = TestNetworkPropagation.FromNumpyArray(@"numpy.array([[1], [0]], numpy.float)");

            int batchSize = X.Shape[0];
            
            const int deviceId = -1;
            //var deviceId = 0;
            var network = TestNetwork.NewForTests(
                        new NetworkSample
                        {
                            LossFunction = EvaluationMetricEnum.BinaryCrossentropy,
                            ShuffleDatasetBeforeEachEpoch = false,
                            CompatibilityMode = NetworkSample.CompatibilityModeEnum.TensorFlow,
                            ResourceIds = new List<int> { deviceId }
                        }
                       .WithSGD(momentum, false),
                        NetworkSample.DefaultWorkingDirectory,
                        "Embedding"
                );

            Debug.Assert(network.Layers.Count == 0);
            network.Input(maxWordsBySentence, -1, -1)
                .Embedding(new [] { vocabularySize }, new[] { embeddingDim }, new[] { -1 }, new[] { 0 }, 0.0)
                .Flatten()
                .Dense(1, 0.0, false)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID);


            Log.Info(network.Summary() + Environment.NewLine);

            TestNetworkPropagation.FromNumpyArray("[[0,0,0,0,0],[-0.020802486687898636, -0.02934335544705391, 0.0035390742123126984, 0.006125748157501221, -0.008332550525665283], [0.0307827927172184, -0.0006774887442588806, 0.0498129241168499, 0.019673515111207962, -0.037462640553712845],[0.020981673151254654, 0.016241561621427536, 0.007225655019283295, -0.013524651527404785, -0.007948171347379684]]")
                .CopyTo(((EmbeddingLayer)network.Layers[1]).Weights);
            TestNetworkPropagation.FromNumpyArray("[[0.05924016237258911], [-0.2979503273963928], [0.39012110233306885], [0.2964285612106323], [0.15513628721237183], [0.032458603382110596], [-0.5190843939781189], [0.3992980718612671], [-0.03236877918243408], [-0.12109190225601196], [0.4128159284591675], [0.14623379707336426], [-0.5325161814689636], [0.38246530294418335], [-0.4191945493221283], [0.4918263554573059], [-0.30854684114456177], [0.1737397313117981], [-0.40517792105674744], [-0.3750319480895996]]")
                .CopyTo(((DenseLayer)network.Layers[3]).Weights);

            //network.Sample.LogNetworkPropagation = true;
            var predict_before = network.Predict(X, false).ToNumpy();

            using var trainingDataSet = new InMemoryDataSet(X, Y);

            var lossAccuracyBefore = network.ComputeMetricsForValidationDataSet(batchSize, trainingDataSet);

            Log.Info("-");
            Log.Info("--------------------------------------------------------------------");
            Log.Info("-");

            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, batchSize);

            var predict_after = network.Predict(X, false).ToNumpy();
            var lossAccuracyAfter = network.ComputeMetricsForValidationDataSet(batchSize, trainingDataSet);

            Log.Info("C# numEpochs= " + numEpochs);
            Log.Info("C# learningRate= " + learningRate);
            Log.Info("C# l2regularizer= " + lambdaL2Regularization);
            Log.Info("C# momentum= " + momentum);
            Log.Info(predict_before);
            Log.Info("C# metrics_before= " + Model.MetricsToString(lossAccuracyBefore, ""));
            Log.Info(predict_after);
            Log.Info("C# metrics_after= " + Model.MetricsToString(lossAccuracyAfter, ""));
        }

        [Test, Explicit]
        public void TestParallelRunWithTensorFlow_Embedding_3D()
        {
            const int numEpochs = 5;
            const double learningRate = 0.01;
            const double lambdaL2Regularization = 0.00;
            const double momentum = 0.9;
            const int batchSize = 2;
            const int deviceId = -1;
            //var deviceId = 0;

            //'X' shape (4,2,3)
            var X = TestNetworkPropagation.FromNumpyArray("[ [[3000,2,3001,0],[1000,0,1001,5]], [[2000,1,2001,7],[1002,0,1003,9]], [[2002,1,2003,2],[3002,2,3003,8]], [[1003,0,1004,6],[2004,1,2005,4]] ]");
            //'Y' shape (4,1)
            var Y = TestNetworkPropagation.FromNumpyArray(@"numpy.array([[1], [0], [1], [0]], numpy.float)");

            var networkSample = new NetworkSample
            {
                LossFunction = EvaluationMetricEnum.BinaryCrossentropy,
                ShuffleDatasetBeforeEachEpoch = false,
                CompatibilityMode = NetworkSample.CompatibilityModeEnum.TensorFlow,
                ResourceIds = new List<int> { deviceId }
            };

            var network = TestNetwork.NewForTests(
                        networkSample
                       .WithAdam(0.9, 0.999, 1e-7),
                        //.WithSGD(momentum, false),
                        NetworkSample.DefaultWorkingDirectory,
                        "Embedding_3D"
                );

            network
                .Input(X.Shape[1], X.Shape[2], -1)
                .Embedding(new []{3,10}, new[] { 5,5 }, new[] { 1, 3 }, new[] { 0, 1 }, 0.0)
                //.Flatten()
                .Dense(1, 0.0, false)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID);

            Log.Info(network.Summary() + Environment.NewLine);
            
            var predict_before = network.Predict(X, false).ToNumpy();

            using var trainingDataSet = new InMemoryDataSet(X, Y);

            var lossAccuracyBefore = network.ComputeMetricsForValidationDataSet(batchSize, trainingDataSet);

            Log.Info("-");
            Log.Info("- Fit -------------------------------------------------------------------");
            Log.Info("-");

            TestNetwork.Fit(network, trainingDataSet, learningRate, numEpochs, batchSize, null);

            Log.Info("-");
            Log.Info("- Using Trained Network -------------------------------------------------------------------");
            Log.Info("-");

            var predict_after = network.Predict(X, false).ToNumpy();
            var lossAccuracyAfter = network.ComputeMetricsForValidationDataSet(batchSize, trainingDataSet);

            Log.Info("C# numEpochs= " + numEpochs);
            Log.Info("C# learningRate= " + learningRate);
            Log.Info("C# l2regularizer= " + lambdaL2Regularization);
            Log.Info("C# momentum= " + momentum);
            Log.Info("C# batchSize= " + batchSize);
            Log.Info(predict_before);
            Log.Info("C# metrics_before= " + Model.MetricsToString(lossAccuracyBefore, ""));
            Log.Info(predict_after);
            Log.Info("C# metrics_after= " + Model.MetricsToString(lossAccuracyAfter, ""));
        }

        [Test, Explicit]
        public void TestParallelRunWithTensorFlow_Embedding_GlobalPooling()
        {
            const int numEpochs = 5;
            const double learningRate = 0.01;
            const double lambdaL2Regularization = 0.00;
            const double momentum = 0.9;
            const int batchSize = 2;
            const int deviceId = -1;
            //var deviceId = 0;
            const int vocabularySize = 3;
            const int embeddingDim = 5;
            const int maxWordsBySentence = 4;

            var X = TestNetworkPropagation.FromNumpyArray(@"numpy.array([[1, 1, 1, 2], [2, 2, 2, 2], [1, 2, 2, 2],[1, 1, 1, 1]], numpy.float)");
            var Y = TestNetworkPropagation.FromNumpyArray(@"numpy.array([[1], [0], [0], [1]], numpy.float)");


            var networkSample = new NetworkSample
            {
                LossFunction = EvaluationMetricEnum.BinaryCrossentropy,
                ShuffleDatasetBeforeEachEpoch = false,
                CompatibilityMode = NetworkSample.CompatibilityModeEnum.TensorFlow,
                ResourceIds = new List<int> { deviceId }
            };

            var network = TestNetwork.NewForTests(
                        networkSample
                       .WithAdam(0.9, 0.999, 1e-7),
                        //.WithSGD(momentum, false),
                        NetworkSample.DefaultWorkingDirectory,
                        "Embedding_GlobalPooling"
                );
            network.Sample.LogNetworkPropagation = true;

            Debug.Assert(network.Layers.Count == 0);
            network.Input(maxWordsBySentence, -1, -1)
                .Embedding(new [] { vocabularySize }, new[] { embeddingDim }, new[] { -1 }, new[] { 0 }, 0.0)
                .GlobalAvgPooling()
                .Dense(4, 0.0, false).Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)
                .Dense(1, 0.0, false).Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID);


            Log.Info(network.Summary() + Environment.NewLine);

            TestNetworkPropagation.FromNumpyArray("[[-0.020802486687898636, -0.02934335544705391, 0.0035390742123126984, 0.006125748157501221, -0.008332550525665283], [0.0307827927172184, -0.0006774887442588806, 0.0498129241168499, 0.019673515111207962, -0.037462640553712845], [0.020981673151254654, 0.016241561621427536, 0.007225655019283295, -0.013524651527404785, -0.007948171347379684]]")
                .CopyTo(((EmbeddingLayer)network.Layers[1]).Weights);
            TestNetworkPropagation.FromNumpyArray("[[0.09049081802368164, -0.45512667298316956, 0.5959198474884033, 0.4528021812438965], [0.2369745969772339, 0.04958134889602661, -0.7929145097732544, 0.6099379062652588], [-0.04944407939910889, -0.18497097492218018, 0.6305867433547974, 0.22337579727172852], [-0.813431978225708, 0.5842254161834717, -0.6403303146362305, 0.7512772083282471], [-0.47131311893463135, 0.26539182662963867, -0.6189195513725281, -0.5728708505630493]]")
                .CopyTo(((DenseLayer)network.Layers[3]).Weights);
            TestNetworkPropagation.FromNumpyArray("[[-0.6677531003952026], [0.5261931419372559], [-0.026724934577941895], [0.8222856521606445]]")
                .CopyTo(((DenseLayer)network.Layers[5]).Weights);


            network.Sample.LogNetworkPropagation = true;
            var predict_before = network.Predict(X, false).ToNumpy();

            using var trainingDataSet = new InMemoryDataSet(X, Y);


            var lossAccuracyBefore = network.ComputeMetricsForValidationDataSet(batchSize, trainingDataSet);

            Log.Info("-");
            Log.Info("- Fit -------------------------------------------------------------------");
            Log.Info("-");

            TestNetwork.Fit(network, trainingDataSet, learningRate, numEpochs, batchSize, null);

            Log.Info("-");
            Log.Info("- Using Trained Network -------------------------------------------------------------------");
            Log.Info("-");

            var predict_after = network.Predict(X, false).ToNumpy();
            var lossAccuracyAfter = network.ComputeMetricsForValidationDataSet(batchSize, trainingDataSet);

            Log.Info("C# numEpochs= " + numEpochs);
            Log.Info("C# learningRate= " + learningRate);
            Log.Info("C# l2regularizer= " + lambdaL2Regularization);
            Log.Info("C# momentum= " + momentum);
            Log.Info("C# batchSize= " + batchSize);
            Log.Info(predict_before);
            Log.Info("C# metrics_before= " + Model.MetricsToString(lossAccuracyBefore, ""));
            Log.Info(predict_after);
            Log.Info("C# metrics_after= " + Model.MetricsToString(lossAccuracyAfter, ""));
        }


        private class SarcasmEntry
        {
            [JsonProperty("article_link")]
            public string ArticleLink { get; set; }
            [JsonProperty("headline")]
            public string Headline { get; set; }
            [JsonProperty("is_sarcastic")]
            public bool IsSarcastic { get; set; }
        }

        [Test, Explicit]
        public void TestParallelRunWithTensorFlow_Sarcasm()
        {
            const int numEpochs = 30;
            const double learningRate = 0.001;
            const double lambdaL2Regularization = 0.00;
            const double momentum = 0.9;
            const int batchSize = 128;
            //var deviceId = -1;
            const int deviceId = 0;
            const int vocab_size = 10000;
            const int embedding_dim = 16;
            const int max_length = 100;
            const string oov_tok = "<OOV>";
            const int training_size = 20000;

            var jsonText = File.ReadAllText(Path.Combine(NetworkSample.DefaultDataDirectory, "Sarcasm", "sarcasm.json"));
            var allEntries = JsonConvert.DeserializeObject<List< SarcasmEntry>>(jsonText);

            // ReSharper disable once AssignNullToNotNullAttribute
            var trainingEntries = allEntries.Take(training_size).ToList();
            var trainingHeadlines = trainingEntries.Select(e => e.Headline).ToList();
            var tokenizer = new Tokenizer(vocab_size, oov_tok);

            tokenizer.FitOnTexts(trainingHeadlines);
            var training_sequences = tokenizer.TextsToSequences(trainingHeadlines);
            //var training_sequences = tokenizer.FitOnTextsAndTextsToSequences(trainingHeadlines);

            var X  = PadSequenceTools.PadSequence(training_sequences, max_length, false, false).Select(x=>(float)x);
            var Y  = new CpuTensor<float>(new[]{X.Shape[0],1}, trainingEntries.Select(e => e.IsSarcastic?1f:0f).ToArray());

            var networkSample = new NetworkSample
            {
                LossFunction = EvaluationMetricEnum.BinaryCrossentropy,
                ShuffleDatasetBeforeEachEpoch = true,
                CompatibilityMode = NetworkSample.CompatibilityModeEnum.TensorFlow,
                ResourceIds = new List<int> { deviceId }
            };

            var network = TestNetwork.NewForTests(
                        networkSample
                        .WithAdam(0.9, 0.999, 1e-7),
                        NetworkSample.DefaultWorkingDirectory,
                        "TestParallelRunWithTensorFlow_Sarcasm"
                );

            Debug.Assert(network.Layers.Count == 0);
            network.Input(max_length, -1, -1)
                .Embedding(new [] { vocab_size }, new[] { embedding_dim }, new[] { -1 }, new[] { 0 }, 0.0)
                .GlobalAvgPoolingOnHeight()
                .Dense(24, 0.0, false).Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)
                .Dense(1, 0.0, false).Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID);
        
            //Log.Info(network.Summary() + Environment.NewLine);
            //network.Sample.LogNetworkPropagation = true;
            var predict_before = network.Predict(X, false).ToNumpy();

            using var trainingDataSet = new InMemoryDataSet(X, Y);
            var validationEntries = allEntries.Skip(training_size).ToList();
            var validationHeadlines = validationEntries.Select(e => e.Headline).ToList();
            var validation_sequences = tokenizer.TextsToSequences(validationHeadlines);
            var X_val = PadSequenceTools.PadSequence(validation_sequences, max_length, false, false).Select(x => (float)x);
            var Y_val = new CpuTensor<float>(new[] { X_val.Shape[0], 1 }, validationEntries.Select(e => e.IsSarcastic ? 1f : 0f).ToArray());
            using var validationDataSet = new InMemoryDataSet(X_val, Y_val);

            var lossAccuracyBefore = network.ComputeMetricsForValidationDataSet(batchSize, trainingDataSet);

            //Log.Info("-");
            //Log.Info("--------------------------------------------------------------------");
            //Log.Info("-");

            TestNetwork.Fit(network, trainingDataSet, learningRate, numEpochs, batchSize, validationDataSet);

            var predict_after = network.Predict(X, false).ToNumpy();
            var lossAccuracyAfter = network.ComputeMetricsForValidationDataSet(batchSize, trainingDataSet);

            Log.Info("C# numEpochs= " + numEpochs);
            Log.Info("C# learningRate= " + learningRate);
            Log.Info("C# l2regularizer= " + lambdaL2Regularization);
            Log.Info("C# momentum= " + momentum);
            Log.Info(predict_before);
            Log.Info("C# metrics_before= " + Model.MetricsToString(lossAccuracyBefore, ""));
            Log.Info(predict_after);
            Log.Info("C# metrics_after= " + Model.MetricsToString(lossAccuracyAfter, ""));
        }


        [Test, Explicit]
        public void TestParallelRunWithTensorFlow_DownSampling2D()
        {
            const int numEpochs = 10;
            const double learningRate = 0.01;
            const double lambdaL2Regularization = 0.00;
            const double momentum = 0.9;
            var X = TestNetworkPropagation.FromNumpyArray(TestNetworkPropagation.X_2_3_4_5);
            var Y = TestNetworkPropagation.FromNumpyArray(TestNetworkPropagation.Y_2_3);

            int batchSize = X.Shape[0];
            const int gpuDeviceId = -1;
            //var gpuDeviceId = 0;
            var network = TestNetwork.NewForTests(
                        new NetworkSample
                        {
                            LossFunction = EvaluationMetricEnum.CategoricalCrossentropy,
                            ShuffleDatasetBeforeEachEpoch = false,
                            CompatibilityMode = NetworkSample.CompatibilityModeEnum.TensorFlow,
                            ConvolutionAlgoPreference = GPUWrapper.ConvolutionAlgoPreference.FASTEST_DETERMINIST_NO_TRANSFORM,
                            ResourceIds = new List<int> { gpuDeviceId }
                        }
                       .WithSGD(momentum, false),
                        NetworkSample.DefaultWorkingDirectory,
                        "TestParallelRunWithTensorFlow_DownSampling2D"
                );


            network.Sample.LogNetworkPropagation = true;

            network
                .Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Convolution(4,1,1,ConvolutionLayer.PADDING_TYPE.SAME, 0.0, true)
                .UpSampling2D(3,2,UpSampling2DLayer.InterpolationEnum.Nearest)
                .Convolution(1, 3, 2, ConvolutionLayer.PADDING_TYPE.SAME, 0.0, true).Dense(Y.Shape[1], 0.0, false)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);

            network.Sample.LogNetworkPropagation = true;

            Log.Info(network.Summary() + Environment.NewLine);


            TestNetworkPropagation.FromConvNumpyArray("[[[[-0.41233378648757935, -0.5469635725021362, -0.09478795528411865, 0.20379328727722168], [-0.45642712712287903, -0.6934521198272705, 0.7060458660125732, 0.6550993919372559], [-0.40876543521881104, 0.5751461982727051, 0.0005752444267272949, 0.8542157411575317]]]]").CopyTo(((ConvolutionLayer)network.Layers[1]).Weights);
            TestNetworkPropagation.FromConvNumpyArray("[[[[-0.1615283042192459], [-0.0656551718711853], [0.1326923966407776], [0.21013426780700684]], [[0.23147475719451904], [0.15308880805969238], [0.0008010268211364746], [0.2704615592956543]], [[-0.2763732671737671], [-0.11263367533683777], [-0.3622085750102997], [0.03678843379020691]]], [[[-0.1616799682378769], [0.029316306114196777], [-0.15289030969142914], [-0.21387864649295807]], [[0.032195329666137695], [-0.013419240713119507], [0.10481679439544678], [-0.18447379767894745]], [[-0.15118040144443512], [0.052129119634628296], [0.07085898518562317], [-0.08211708068847656]]], [[[-0.02411407232284546], [0.17931300401687622], [-0.2963199317455292], [-0.019487440586090088]], [[-0.2584547698497772], [0.23713970184326172], [-0.351848304271698], [0.3424469232559204]], [[0.22793227434158325], [0.13822901248931885], [-0.12481275200843811], [-0.32772859930992126]]]]").CopyTo(((ConvolutionLayer)network.Layers[3]).Weights);
            TestNetworkPropagation.FromNumpyArray("[[0.07366013526916504, 0.3170207142829895, -0.1550242304801941], [0.420951247215271, -0.4191424548625946, 0.3381590247154236], [0.11008310317993164, 0.0986890196800232, 0.31357908248901367], [0.41440945863723755, 0.30317842960357666, 0.3536931872367859], [-0.010290741920471191, -0.21904385089874268, -0.020769357681274414], [-0.2869524359703064, -0.3439455032348633, 0.2285328507423401], [-0.022606879472732544, -0.1754196584224701, -0.12093043327331543], [-0.19505150616168976, 0.32367968559265137, 0.27787232398986816], [0.1375676393508911, -0.1417226493358612, 0.33683180809020996], [-0.36117273569107056, 0.001855224370956421, 0.24049299955368042], [-0.02008679509162903, 0.22243833541870117, -0.27483871579170227], [-0.20811842381954193, -0.17607355117797852, -0.1847764253616333], [-0.41185829043388367, 0.14473176002502441, 0.10743755102157593], [0.3232056498527527, -0.2687329947948456, 0.041926443576812744], [-0.07551324367523193, 0.23673099279403687, -0.4212562143802643], [-0.32285287976264954, -0.20976179838180542, 0.35986894369125366], [-0.42236655950546265, 0.06221747398376465, 0.19280701875686646], [-0.1036037802696228, 0.22280341386795044, 0.2663360834121704], [-0.278300404548645, 0.3701552152633667, -0.3987610638141632], [-0.2845539450645447, 0.08112376928329468, -0.06442150473594666], [0.13321810960769653, 0.39671868085861206, -0.34261322021484375], [-0.23947212100028992, -0.10445082187652588, -0.36301395297050476], [0.20646917819976807, 0.11567127704620361, 0.15597444772720337], [-0.3057088851928711, 0.39422833919525146, -0.23814217746257782], [0.1633470058441162, 0.12872058153152466, 0.2478216290473938], [-0.3868710696697235, -0.335817813873291, 0.42601829767227173], [-0.3151834011077881, 0.30162113904953003, -0.06157597899436951], [-0.19710223376750946, 0.0573333203792572, 0.2074006199836731], [-0.28093406558036804, 0.2030026912689209, 0.4050601124763489], [0.29869991540908813, -0.31979823112487793, 0.41144388914108276]]").CopyTo(((DenseLayer)network.Layers[4]).Weights);


            var predict_before = network.Predict(X, false).ToNumpy();

            using var trainingDataSet = new InMemoryDataSet(X, Y);
            var lossAccuracyBefore = network.ComputeMetricsForValidationDataSet(batchSize, trainingDataSet);

            Log.Info("-");
            Log.Info("--------------------------------------------------------------------");
            Log.Info("-");

            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, batchSize);

            var predict_after = network.Predict(X, false).ToNumpy();
            var lossAccuracyAfter = network.ComputeMetricsForValidationDataSet(batchSize, trainingDataSet);

            Log.Info("C# numEpochs= " + numEpochs);
            Log.Info("C# learningRate= " + learningRate);
            Log.Info("C# l2regularizer= " + lambdaL2Regularization);
            Log.Info("C# momentum= " + momentum);
            Log.Info(predict_before);
            Log.Info("C# metrics_before= " + Model.MetricsToString(lossAccuracyBefore, ""));
            Log.Info(predict_after);
            Log.Info("C# metrics_after= " + Model.MetricsToString(lossAccuracyAfter, ""));
        }

        [Test, Explicit]
        public void TestParallelRunWithTensorFlow_Huber()
        {
            const int numEpochs = 10;
            const double learningRate = 0.1;
            const double lambdaL2Regularization = 0.00;
            const double momentum = 0.9;
            const int batchSize = 2;

            var X = TestNetworkPropagation.FromNumpyArray(TestNetworkPropagation.X_2_3_4_5);
            var Y = TestNetworkPropagation.FromNumpyArray(TestNetworkPropagation.Y_2_3);

            const int deviceId = -1;
            //var deviceId = 0;
            var network = TestNetwork.NewForTests(
                        new NetworkSample
                        {
                            LossFunction = EvaluationMetricEnum.Huber,
                            //LossFunction = LossFunctionEnum.BinaryCrossentropy,
                            ShuffleDatasetBeforeEachEpoch = false,
                            CompatibilityMode = NetworkSample.CompatibilityModeEnum.TensorFlow,
                            ResourceIds = new List<int> { deviceId }
                        }
                       .WithSGD(momentum, false),
                        NetworkSample.DefaultWorkingDirectory,
                        "Huber"
                );

            network
                .Input(X.Shape[1], 1, -1)
                .Dense(3, 0.0, false)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)
                .Dense(1, 0.0, false)
                ;


            Log.Info(network.Summary() + Environment.NewLine);

            TestNetworkPropagation.FromNumpyArray("[[0.17207741737365723, -0.19425582885742188, 0.6897902488708496], [0.5924994945526123, -0.11895132064819336, -0.8060355186462402], [0.44127702713012695, -0.15072321891784668, -0.8697922229766846]]")
                .CopyTo(((DenseLayer)network.Layers[1]).Weights);
            TestNetworkPropagation.FromNumpyArray("[[0.6883463859558105], [0.9837051630020142], [0.17996716499328613]]")
                .CopyTo(((DenseLayer)network.Layers[3]).Weights);

            network.Sample.LogNetworkPropagation = true;
            var predict_before = network.Predict(X, false).ToNumpy();

            using var trainingDataSet = new InMemoryDataSet(X, Y);

            var lossAccuracyBefore = network.ComputeMetricsForValidationDataSet(batchSize, trainingDataSet);

            Log.Info("-");
            Log.Info("--------------------------------------------------------------------");
            Log.Info("-");

            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, batchSize);

            var predict_after = network.Predict(X, false).ToNumpy();
            var lossAccuracyAfter = network.ComputeMetricsForValidationDataSet(batchSize, trainingDataSet);

            Log.Info("C# numEpochs= " + numEpochs);
            Log.Info("C# learningRate= " + learningRate);
            Log.Info("C# 