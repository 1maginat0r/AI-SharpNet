
using System;
using System.Linq;
using NUnit.Framework;
using SharpNet.Data;
using SharpNet.GPU;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using SharpNet;
using SharpNet.CPU;
using SharpNet.Layers;
using SharpNet.Networks;
using SharpNetTests.CPU;
using SharpNetTests.Datasets;
using SharpNetTests.GPU;
using SharpNetTests.NonReg;

namespace SharpNetTests
{
    [TestFixture]
    public class TestParallelRunCpuVersusGpu
    {
        private const int BatchSize = 9;
        private const int FiltersCount = 8;
        private const int ChannelsCount = 3;
        private const int Height = 17;
        private const int Width = 32;
        private const int Nx = Height * Width * ChannelsCount;
	    private readonly Random _rand = new (0);
        private const GPUWrapper.ConvolutionAlgoPreference ConvolutionAlgoPreference = GPUWrapper.ConvolutionAlgoPreference.FASTEST_DETERMINIST_NO_TRANSFORM;
        // ReSharper disable once MemberCanBeMadeStatic.Local
        private GPUWrapper GpuWrapper => TestGPUTensor.GpuWrapper;

        [Test]
        public void TestConvolution()
        {
            foreach(ConvolutionLayer.PADDING_TYPE paddingType in Enum.GetValues(typeof(ConvolutionLayer.PADDING_TYPE)))
            foreach(NetworkSample.CompatibilityModeEnum compatibilityMode in Enum.GetValues(typeof(NetworkSample.CompatibilityModeEnum)))
            foreach(int stride in new[]{1,2})
            foreach (var isDepthwiseConvolution in new[] { true,false})
            {
                const int channelsCount = 3;
                const int height = 17;
                const int width = 32;
                const int kernelSize = 3;
                const int filterCount = 128;
                var x = RandomTensor(new[] { BatchSize, channelsCount, height, width });
                var convolutionShape = isDepthwiseConvolution
                        ? new[] { 1, channelsCount, kernelSize, kernelSize }
                        : new[] { filterCount, channelsCount, kernelSize, kernelSize };
                var convolution = RandomTensor(convolutionShape);
	            var y = RandomTensor(ConvolutionLayer.OutputShape(x.Shape, convolution.Shape, paddingType, stride, isDepthwiseConvolution));
                ConvolutionLayer.Padding(x.Shape[2], kernelSize, stride, paddingType, compatibilityMode, out int paddingTop, out int paddingBottom);
                ConvolutionLayer.Padding(x.Shape[3], kernelSize, stride, paddingType, compatibilityMode, out int paddingLeft, out int paddingRight);
                var memoryPool =  new TensorMemoryPool(GpuWrapper);
                if (ConvolutionLayer.IsAsymmetricPadding(paddingTop, paddingBottom, paddingLeft, paddingRight))
                {
                    continue; //asymmetric padding is not supported by cuDNN
                }
                TestAll(new[] { x, convolution, y }, tensors => tensors[0].Convolution(tensors[1], paddingTop, paddingBottom, paddingLeft, paddingRight, stride, tensors[2], isDepthwiseConvolution, ConvolutionAlgoPreference, memoryPool));
            }
        }

        [Test]
        public void TestConvolutionBackwardBias()
        {
            var dxShape = new[] { BatchSize, FiltersCount, Height, Width };
            var biasShape = new[] { 1, FiltersCount, 1, 1 };
            var dx = RandomTensor(dxShape);
            var convolutionBackwardBias = new CpuTensor<float>(biasShape);
            TestAll(new[] { dx, convolutionBackwardBias }, tensors => tensors[0].ConvolutionBackwardBias(tensors[1]));
        }




        [Test]
        public void Test_Compute_Row_Mean_Variance()
        {
            foreach (var unbiasedVariance in new[] { true, false })
            {
                var xShape = new[] { BatchSize, FiltersCount, Height, Width };
                var x = RandomTensor(xShape);
                foreach (var rows in new[] { BatchSize, BatchSize * FiltersCount, BatchSize * FiltersCount * Height })
                {
                    var mean = RandomTensor(new[] { rows, 1});
                    var variance = RandomTensor(mean.Shape);
                    TestAll(new[] { x, mean, variance }, tensors => tensors[0].Compute_Row_Mean_Variance(tensors[1], tensors[2], unbiasedVariance));
                }
            }
        }

        [Test]
        public void Test_numpy_sum()
        {
            foreach (var axis in new[] { 0, 1})
            {
                var xShape = new[] { BatchSize, FiltersCount, Height, Width };
                foreach (var rows in new[] { BatchSize, BatchSize * FiltersCount, BatchSize * FiltersCount * Height })
                {
                    var a = RandomTensor(xShape);
                    var cols = Utils.Product(xShape) / rows;
                    var sum_shape = (axis == 1) ? new[] { rows, 1 } : new[] { 1, cols };
                    var sum_buffer = RandomTensor(sum_shape);
                    TestAll(new[] { a, sum_buffer }, tensors => tensors[0].numpy_sum(tensors[1], axis));
                }
            }
        }

        [Test]
        public void Test_StandardizeInPlace()
        {
            foreach (var unbiasedVariance in new[] { true, false })
            foreach (var epsilon in new[] {0.1f, 0.001f, 1e-8f})
            {
                var xShape = new[] { BatchSize, FiltersCount, Height, Width };
                var x = RandomTensor(xShape);
                foreach (var rows in new[] { BatchSize, BatchSize * FiltersCount, BatchSize * FiltersCount * Height })
                {
                    var row_mean = RandomTensor(new[] { rows, 1 });
                    var row_variance = RandomTensor(row_mean.Shape);
                    x.Compute_Row_Mean_Variance(row_mean, row_variance, unbiasedVariance);
                    TestAll(new[] { x, row_mean, row_variance }, tensors => tensors[0].StandardizeInPlace(tensors[1], tensors[2], 1, epsilon));
                }
            }
        }


        [Test]
        public void TestStandardizeRowsInPlaceBroadcastGammasBetas()
        {
            foreach (var unbiasedVariance in new[] { true, false })
            foreach (var epsilon in new[] { 0.1f, 0.001f, 1e-8f })
            {
                var xShape = new[] { BatchSize, FiltersCount, Height, Width };
                var x = RandomTensor(xShape);
                foreach (var rows in new[] { BatchSize, BatchSize * FiltersCount, BatchSize * FiltersCount * Height })
                {
                    var cols = x.Count / rows;

                    var row_mean = RandomTensor(new[] { rows, 1 });
                    var row_variance = RandomTensor(row_mean.Shape);

                    var col_gammas = RandomTensor(new[] { 1, cols});
                    var col_betas = RandomTensor(col_gammas.Shape);

                    x.Compute_Row_Mean_Variance(row_mean, row_variance, unbiasedVariance);
                    TestAll(new[] { x, row_mean, row_variance, col_gammas, col_betas}, tensors => tensors[0].StandardizeRowsInPlaceBroadcastGammasBetas(tensors[1], tensors[2], epsilon, tensors[3], tensors[4]));
                }
            }
        }

        [Test]
        public void TestConvolutionGradient()
        {
            var memoryPool = new TensorMemoryPool(GpuWrapper);
            foreach (ConvolutionLayer.PADDING_TYPE paddingType in Enum.GetValues(typeof(ConvolutionLayer.PADDING_TYPE)))
            foreach (NetworkSample.CompatibilityModeEnum compatibilityMode in Enum.GetValues(typeof(NetworkSample.CompatibilityModeEnum)))
            foreach (int stride in new[] { 1, 2 })
            foreach (int kernelSize in new[] { 3, 5 })
            foreach (var isDepthwiseConvolution in new[] { true, false })
            {
                const int channelsCount = 3;
                const int height = 17;
                const int width = 32;
                var x = RandomTensor(new[] { BatchSize, channelsCount, height, width });
                x = new CpuTensor<float>(x.Shape);
                var convolutionShape = isDepthwiseConvolution
                    ? new[] { 1, channelsCount, kernelSize, kernelSize }
                    : new[] { 9, channelsCount, kernelSize, kernelSize };
                var convolution = RandomTensor(convolutionShape);
                var dy = RandomTensor(ConvolutionLayer.OutputShape(x.Shape, convolution.Shape, paddingType, stride, isDepthwiseConvolution));
                //this will compute 'dx' && 'convolutionGradient'
                var dx = RandomTensor(x.Shape);
                var convolutionGradient = RandomTensor(convolution.Shape);
                ConvolutionLayer.Padding(x.Shape[2], kernelSize, stride, paddingType, compatibilityMode, out int paddingTop, out int paddingBottom);
                ConvolutionLayer.Padding(x.Shape[3], kernelSize, stride, paddingType, compatibilityMode, out int paddingLeft, out int paddingRight);

                if (ConvolutionLayer.IsAsymmetricPadding(paddingTop, paddingBottom, paddingLeft, paddingRight))
                {
                    var paddedXShape = new[] { x.Shape[0], x.Shape[1], paddingTop + x.Shape[2] + paddingBottom, paddingLeft + x.Shape[3] + paddingRight };
                    var padded_X = RandomTensor(paddedXShape);
                    padded_X.ZeroPadding(x, paddingTop, paddingBottom, paddingLeft, paddingRight);
                    var padded_dX = RandomTensor(paddedXShape);
                    TestAll(new[] { padded_X, convolution, dy, dx, padded_dX, convolutionGradient }, tensors => ConvolutionGradientAsymmetricPadding(
                        tensors[0], tensors[1], tensors[2], paddingTop, paddingBottom, paddingLeft, paddingRight, stride, tensors[3], tensors[4], tensors[5], isDepthwiseConvolution, memoryPool));
                }
                else
                {
                    TestAll(new[] { x, convolution, dy, dx, convolutionGradient }, tensors => tensors[0].ConvolutionGradient(tensors[1], tensors[2], paddingTop, paddingBottom, paddingLeft, paddingRight, stride, tensors[3], tensors[4], isDepthwiseConvolution, ConvolutionAlgoPreference, memoryPool));
                }
            }
        }

        private static void ConvolutionGradientAsymmetricPadding(Tensor padded_X, Tensor convolution, Tensor dy, int paddingTop,
            int paddingBottom, int paddingLeft, int paddingRight, int stride, Tensor dx, Tensor padded_dX, Tensor convGradient, bool isDepthwiseConvolution, TensorMemoryPool memoryPool)
        {
            padded_X.ConvolutionGradient(convolution, dy, 0, 0, 0, 0, stride, padded_dX, convGradient, isDepthwiseConvolution, ConvolutionAlgoPreference, memoryPool);
            dx.ZeroUnpadding(padded_dX, paddingTop, paddingBottom, paddingLeft, paddingRight);
        }

        [Test]
        public void TestBatchNormalization()
        {
            foreach (var mode in new []{ cudnnBatchNormMode_t.CUDNN_BATCHNORM_SPATIAL, cudnnBatchNormMode_t.CUDNN_BATCHNORM_PER_ACTIVATION })
            foreach (var batchSize in new[] {/* 1,*/ 4, 9 })                //TODO: enable batchSize = 1
            foreach (var momentum in new []{0.0, 0.5, 0.99 /*, 1.0*/ })     //TODO: enable momentum = 1.0
            foreach (var epsilon in new []{1e-3, 1e-5})
            foreach (var ignoreHW in new []{false, true})
            foreach (var isTraining in new[] { false, true })
            {
                var xShape = new[] { batchSize, ChannelsCount, Height, Width };
                var scaleAndBiasShape = (mode == cudnnBatchNormMode_t.CUDNN_BATCHNORM_PER_ACTIVATION)
                    ? new[] { 1, ChannelsCount, Height, Width }
                    : new[] { 1, ChannelsCount, 1, 1 };
                if (ignoreHW)
                {
                    xShape = xShape.Take(2).ToArray();
                    scaleAndBiasShape = scaleAndBiasShape.Take(2).ToArray();
                }

                var x = RandomTensor(xShape);
                var y = RandomTensor(xShape);
                var scale = RandomTensor(scaleAndBiasShape);
                var bias = RandomTensor(scaleAndBiasShape);
                var runningInputMean = RandomTensor(scaleAndBiasShape);
                var runningInputVariance = TestCpuTensor.RandomFloatTensor(scaleAndBiasShape, _rand, 0.1, 2);
                var meanBuffer = RandomTensor(scaleAndBiasShape);
                var varianceBuffer = TestCpuTensor.RandomFloatTensor(scaleAndBiasShape, _rand, 0.1, 50);
                double exponentialAverageSmoothingFactor = 1 - momentum;
                TestAll(new[] { x, y, scale, bias, runningInputMean, runningInputVariance, meanBuffer, varianceBuffer }, tensors => tensors[0].BatchNormalization(tensors[1], tensors[2], tensors[3], exponentialAverageSmoothingFactor, tensors[4], tensors[5], mode, epsilon, tensors[6], tensors[7], isTraining));
            }
        }

        [Test]
        public void TestBatchNormalizationV2()
        {
            var tensorIdsToIgnore = new List<int> { 6, 7 };
            const cudnnBatchNormMode_t mode = cudnnBatchNormMode_t.CUDNN_BATCHNORM_SPATIAL;
            var xShape = new[] { 1, 32, 112, 112};
            var x = new CpuTensor<float>(xShape);
            x.ZeroMemory();
            var scale = TestNetworkPropagation.FromNumpyArray("[[[[2.82185745]],[[4.20555544]],[[4.44391775]],[[2.95071363]],[[0.901465356]],[[3.83799005]],[[2.20374274]],[[3.30325413]],[[3.38044739]],[[0.202515125]],[[2.14543128]],[[0.645111859]],[[3.29296565]],[[11.9912415]],[[0.810986161]],[[3.39099979]],[[2.6564517]],[[8.52717972]],[[2.52371788]],[[3.94317198]],[[2.74237108]],[[11.1155062]],[[4.08373785]],[[5.75315952]],[[0.335611582]],[[1.24477983]],[[3.90086651]],[[1.98501635]],[[0.818592787]],[[0.626930952]],[[6.75085163]],[[3.4190371]]]]");
            var bias = TestNetworkPropagation.FromNumpyArray("[[[[-3.74896479]],[[2.43146777]],[[2.31554103]],[[7.13698292]],[[-1.38208234]],[[8.66540337]],[[-2.95346022]],[[1.81856453]],[[0.995381236]],[[0.00296683772]],[[-2.85715914]],[[1.74939632]],[[0.599703848]],[[0.165816754]],[[1.90356266]],[[8.97630692]],[[2.26754451]],[[3.72180033]],[[2.572788]],[[1.96836185]],[[-3.36665225]],[[2.64624929]],[[10.5395947]],[[-10.4322577]],[[-1.63009882]],[[1.37903798]],[[9.95489788]],[[1.99438405]],[[0.159816369]],[[2.50823808]],[[-10.8555698]],[[2.08439994]]]]");
            var runningInputMean = TestNetworkPropagation.FromNumpyArray("[[[[-0.0474244691]],[[-0.00338064576]],[[0.00407501776]],[[0.0787407607]],[[0.0313696824]],[[0.0837314799]],[[-0.0393488146]],[[0.0694158077]],[[0.639113843]],[[-0.171755388]],[[-0.382961541]],[[0.0100561073]],[[0.606002986]],[[1.39727235]],[[0.420819908]],[[-0.0792663917]],[[0.00732345507]],[[-0.770392716]],[[0.00307485089]],[[-0.00288994168]],[[-0.0452340655]],[[-0.719747245]],[[-0.0934633166]],[[0.163005278]],[[0.121294215]],[[-0.00648898305]],[[-0.0706383437]],[[0.00286416081]],[[2.91242941E-09]],[[0.0120399296]],[[-0.063189812]],[[-0.00128901063]]]]");
            var runningInputVariance = TestNetworkPropagation.FromNumpyArray("[[[[7.25111055]],[[5.37058496]],[[6.66747379]],[[18.2757835]],[[5.69575691]],[[17.0573292]],[[6.76594353]],[[1.52835393]],[[18.0554256]],[[27.2328396]],[[10.9577389]],[[3.57627463]],[[12.896986]],[[39.5671387]],[[3.67913604]],[[13.6923494]],[[6.86120129]],[[19.7278404]],[[3.81912017]],[[9.09753227]],[[6.9455328]],[[23.5766983]],[[18.0286465]],[[18.6031551]],[[1.11303592]],[[6.78300667]],[[11.5361662]],[[6.32360983]],[[0]],[[1.08625805]],[[19.3687859]],[[5.1940341]]]]");
            var y = RandomTensor(xShape);
            var meanBuffer = TestNetworkPropagation.FromNumpyArray("[[[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]]]]");
            var varianceBuffer = TestNetworkPropagation.FromNumpyArray("[[[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]]]]");
            const double exponentialAverageSmoothingFactor = 1 - 0.99;
            TestAll(new[] { x, y, scale, bias, runningInputMean, runningInputVariance, meanBuffer, varianceBuffer }, tensors => tensors[0].BatchNormalization(tensors[1], tensors[2], tensors[3], exponentialAverageSmoothingFactor, tensors[4], tensors[5], mode, 0.001, tensors[6], tensors[7], false), tensorIdsToIgnore);
        }
        [Test]
        public void TestBatchNormalizationBackward()
        {
            foreach (var mode in new[] { cudnnBatchNormMode_t.CUDNN_BATCHNORM_SPATIAL, cudnnBatchNormMode_t.CUDNN_BATCHNORM_PER_ACTIVATION })
            foreach (var batchSize in new[] { 1, 4, 9 })
            foreach (var exponentialAverageFactor in new[] { 0.5, 0.99, 1.0 })
            foreach (var epsilon in new[] { 1e-3, 1e-5 })
            foreach (var ignoreHW in new[] { false, true })
            {
                var xShape = new[] { batchSize, ChannelsCount, Height, Width };
                var scaleAndBiasShape = (mode == cudnnBatchNormMode_t.CUDNN_BATCHNORM_PER_ACTIVATION)
                    ? new[] { 1, ChannelsCount, Height, Width }
                    : new[] { 1, ChannelsCount, 1, 1 };
                if (ignoreHW)
                {
                    xShape = xShape.Take(2).ToArray();
                    scaleAndBiasShape = scaleAndBiasShape.Take(2).ToArray();
                }
                var x = RandomTensor(xShape);
                var y = RandomTensor(xShape);
                var scale = RandomTensor(scaleAndBiasShape);
                var bias = RandomTensor(scaleAndBiasShape);
                var runningInputMean = RandomTensor(scaleAndBiasShape);
                var runningInputVariance = TestCpuTensor.RandomFloatTensor(scaleAndBiasShape, _rand, 0.1, 2);
                var meanBuffer = RandomTensor(scaleAndBiasShape);
                var invertOfUnbiasedVolatilityBuffer = RandomTensor(scaleAndBiasShape);
                x.BatchNormalization(y, scale, bias, exponentialAverageFactor, runningInputMean, runningInputVariance, mode, epsilon, meanBuffer, invertOfUnbiasedVolatilityBuffer, false);
                var dx = RandomTensor(xShape);
                var dy = RandomTensor(xShape);
                var scaleGradient = RandomTensor(scaleAndBiasShape);
                var biasGradient = RandomTensor(scaleAndBiasShape);
                TestAll(new[] { x, dy, dx, scale, scaleGradient, biasGradient, meanBuffer, invertOfUnbiasedVolatilityBuffer }, tensors => tensors[0].BatchNormalizationBackward(tensors[1], tensors[2], tensors[3], tensors[4], tensors[5], mode, epsilon, tensors[6], tensors[7]));
            } 
        }



        [Test]
        public void TestLayerNormalization()
        {
            foreach (var unbiasedVariance in new[] { true, false })
            foreach (var epsilon in new[] { 1e-3f, 1e-5f })
            foreach (var batchSize in new[] { 1, 4, 9 })
            {
                var xShape = new[] { batchSize, ChannelsCount, Height, Width };
                foreach (var cols in new[] { Width, Width * Height, Width * Height * ChannelsCount })
                {
                    var rows = Utils.Product(xShape) / cols;
                    var x = RandomTensor(xShape);
                    var y = RandomTensor(xShape);
                    var gammas = RandomTensor(new[] { 1, cols });
                    var betas = RandomTensor(gammas.Shape);
                    var mean = RandomTensor(new[] { rows, 1 });
                    var variance = RandomTensor(mean.Shape);
                    x.Compute_Row_Mean_Variance(mean, variance, unbiasedVariance);
                    TestAll(new[] { x, y, gammas, betas, mean, variance }, tensors => tensors[0].LayerNormalization(tensors[1], tensors[2], tensors[3], tensors[4], tensors[5], epsilon));
                }
            }
        }


        [Test]
        public void Test_LayerNormalizationBackward()
        {
            foreach (var unbiasedVariance in new[] { true, false })
            foreach (var epsilon in new[] { 1e-3f, 1e-5f })
            foreach (var batchSize in new[] { 1, 4, 9 })
            {
                var xShape = new[] { batchSize, ChannelsCount, Height, Width };
                foreach (var cols in new[] { Width, Width * Height, Width * Height * ChannelsCount })
                {
                    var rows = Utils.Product(xShape) / cols;
                    var x = RandomTensor(xShape);
                    var dy = RandomTensor(xShape);
                    var dx = RandomTensor(xShape);
                    var gammas = RandomTensor(new[] { 1, cols });
                    var mean = RandomTensor(new[] { rows, 1 });
                    var variance = RandomTensor(mean.Shape);

                    var dmean = RandomTensor(mean.Shape);
                    var dvariance = RandomTensor(mean.Shape);

                    x.Compute_Row_Mean_Variance(mean, variance, unbiasedVariance);
                    TestAll(new[] { x, dy, dx, gammas, mean, variance, dmean, dvariance }, tensors => tensors[0].LayerNormalizationBackward(tensors[1], tensors[2], tensors[3], tensors[4], tensors[5], epsilon, tensors[6], tensors[7]), tensorIdsToIgnore: new List<int> { 6, 7 });
                }
            }
        }

        [Test]
	    public void TestZeroMemory()
	    {
	        var a = RandomTensor(new[] { BatchSize, ChannelsCount, Height, Width });
	        TestAll(new[] { a }, tensors => tensors[0].ZeroMemory());
	    }

        [Test]
	    public void TestDot()
	    {
	        var a = RandomTensor(new[] { 8, 10});
	        var b = RandomTensor(new[] { a.Shape[1], 12});
	        var result = new CpuTensor<float>(new[] { a.Shape[0], b.Shape[1] });
	        TestAll(new[] { a, b, result}, tensors => tensors[2].Dot(tensors[0], false, tensors[1], false, 1, 0));
        }

        [Test]
        public void TestBatchMatrixMultiplication()
        {
            const int batchSize = 3;
            const int maxLength = 12;
            const int dim = 7;
            
            var weights_gradients_buffer = RandomTensor(new[] { batchSize, maxLength, maxLength });
            var V = RandomTensor(new[] { batchSize, maxLength, dim });
            var dy = RandomTensor(V.Shape);
            const float scaling = 0.25f;
            //weights_gradients_buffer.BatchMatrixMultiplication(dy, false, V, true, scaling, 0.0f);
            TestAll(new[] { weights_gradients_buffer, dy, V}, tensors => tensors[0].BatchMatrixMultiplication(tensors[1], false, tensors[2], true, scaling, 0));

            var dQ = RandomTensor(V.Shape);
            var scores_gradients_buffer = RandomTensor(weights_gradients_buffer.Shape);
            var K = RandomTensor(V.Shape);
            //dQ.BatchMatrixMultiplication(scores_gradients_buffer, false, K, false, 1, 0.0f);
            TestAll(new[] { dQ, scores_gradients_buffer, K }, tensors => tensors[0].BatchMatrixMultiplication(tensors[1], false, tensors[2], false, 1, 0));

            var dK = RandomTensor(V.Shape);
            var Q = RandomTensor(V.Shape);
            //dK.BatchMatrixMultiplication(scores_gradients_buffer, true, Q, false, 1, 0.0f);
            TestAll(new[] { dK, scores_gradients_buffer, Q }, tensors => tensors[0].BatchMatrixMultiplication(tensors[1], true, tensors[2], false, 1, 0));
        }


        [Test]
        public void TestUpSampling()