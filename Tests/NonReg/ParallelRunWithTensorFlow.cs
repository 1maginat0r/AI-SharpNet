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
            TestNetworkPropagation.FromNumpyArray("[[[0.1554521918296814, -0.10820861905813217, -0