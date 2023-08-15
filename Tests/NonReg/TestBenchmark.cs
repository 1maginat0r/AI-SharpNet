
﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.Linq;
using System.Threading.Tasks;
using log4net;
using NUnit.Framework;
using SharpNet;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.DataAugmentation;
using SharpNet.Datasets;
using SharpNet.GPU;
using SharpNet.Layers;
using SharpNet.Networks;
using SharpNetTests.GPU;
// ReSharper disable AccessToDisposedClosure
// ReSharper disable AccessToModifiedClosure
#pragma warning disable 162

namespace SharpNetTests.NonReg
{
    [TestFixture]
    [SuppressMessage("ReSharper", "UnreachableCode")]
    [SuppressMessage("ReSharper", "ConditionIsAlwaysTrueOrFalse")]
    [SuppressMessage("ReSharper", "HeuristicUnreachableCode")]
    public class TestBenchmark
    {
        private static readonly ILog Log = LogManager.GetLogger(typeof(Network));


        static TestBenchmark()
        {
            Utils.ConfigureThreadLog4netProperties(NetworkSample.DefaultWorkingDirectory, "SharpNet_Benchmark");
        }


        [Test, Explicit]
        public void TestGPUBenchmark_Memory()
        {
            //check RAM => GPU Copy perf
            var tmp_2GB = new float[500 * 1000000];

            var gpu = TestGPUTensor.GpuWrapper;
            Console.WriteLine(gpu.ToString());
            double maxSpeed = 0;
            for (int i = 1; i <= 3; ++i)
            {
                Console.WriteLine(Environment.NewLine + "Loop#" + i);
                var sw = Stopwatch.StartNew();
                var tensors = new GPUTensor<float>[1];
                for(int t=0;t<tensors.Length;++t)
                {
                    tensors[t] = new GPUTensor<float>(new[] { tmp_2GB.Length}, tmp_2GB, gpu);
                }
                Console.WriteLine(gpu.ToString());
                foreach (var t in tensors)
                {
                    t.Dispose();
                }
                var speed = (tensors.Length*((double)tensors[0].CapacityInBytes) / sw.Elapsed.TotalSeconds)/1e9;
                maxSpeed = Math.Max(speed, maxSpeed);
                Console.WriteLine("speed: " + speed + " GB/s");
            }

            System.IO.File.AppendAllText(Utils.ConcatenatePathWithFileName(NetworkSample.DefaultWorkingDirectory, "GPUBenchmark_Memory.csv"),
                DateTime.Now.ToString("F", CultureInfo.InvariantCulture) + ";"
                + "2GB Copy CPU=>GPU;"
                + gpu.DeviceName()+";"
#if DEBUG
                +"DEBUG;"
#else
                + "RELEASE;"
#endif
                + maxSpeed + ";"
                + Environment.NewLine
                );
        }


        [Test, Explicit]
        public void BenchmarkDataAugmentation()
        {
            const bool useMultiThreading = true;
            const bool useMultiGpu = true;
            const int channels = 3;
            const int targetHeight = 118;
            const int targetWidth = 100;
            var miniBatchSize = 300;
            // ReSharper disable once ConditionIsAlwaysTrueOrFalse
            if (useMultiGpu) { miniBatchSize *= GPUWrapper.GetDeviceCount(); }
            var p = EfficientNetNetworkSample.Cancel();
            p.DataAugmentationType = ImageDataGenerator.DataAugmentationEnum.AUTO_AUGMENT_IMAGENET;
            p.BatchSize = miniBatchSize;
            var database = new CancelDatabase();
            //TODO Test with selection of only matching size input in the training set
            using var dataset = database.ExtractDataSet(e => CancelDatabase.IsValidNonEmptyCancel(e.Cancel), ResizeStrategyEnum.BiggestCropInOriginalImageToKeepSameProportion);
            
            //dataAugmentationConfig.DataAugmentationType = ImageDataGenerator.DataAugmentationEnum.AUTO_AUGMENT_CIFAR10;
            var xMiniBatchShape = new []{miniBatchSize, channels, targetHeight, targetWidth};
            var yMiniBatchShape = new[] { miniBatchSize, dataset.LoadFullY().Shape[1] };
            var rand = new Random(0);
            var shuffledElementId = Enumerable.Range(0, dataset.Count).ToArray();
            Utils.Shuffle(shuffledElementId, rand);

            var xOriginalNotAugmentedMiniBatchCpu = new CpuTensor<float>(xMiniBatchShape);
            var xBufferMiniBatchCpu = new CpuTensor<float>(xMiniBatchShape);
            var xBufferForDataAugmentedMiniBatch = new CpuTensor<float>(xMiniBatchShape);
            var yOriginalBufferMiniBatchCpu = new CpuTensor<float>(yMiniBatchShape);
            yOriginalBufferMiniBatchCpu.ZeroMemory();
            var yDataAugmentedBufferMiniBatchCpu = new CpuTensor<float>(yOriginalBufferMiniBatchCpu.Shape);
            yDataAugmentedBufferMiniBatchCpu.ZeroMemory();

            var imageDataGenerator = new ImageDataGenerator(p);

            var swLoad = new Stopwatch();
            var swDA = new Stopwatch();

            int count = 0;
            for (int firstElementId = 0; firstElementId <= (dataset.Count - miniBatchSize); firstElementId += miniBatchSize)
            {
                count += miniBatchSize;
                int MiniBatchIdxToElementId(int miniBatchIdx) => shuffledElementId[firstElementId + miniBatchIdx];
                swLoad.Start();

                if (useMultiThreading)
                {
                    Parallel.For(0, miniBatchSize, indexInBuffer => dataset.LoadAt(MiniBatchIdxToElementId(indexInBuffer), indexInBuffer, xOriginalNotAugmentedMiniBatchCpu, yOriginalBufferMiniBatchCpu, false, false));
                }
                else
                {
                    for (int indexInMiniBatch = 0; indexInMiniBatch < miniBatchSize; ++indexInMiniBatch)
                    {
                        dataset.LoadAt(MiniBatchIdxToElementId(indexInMiniBatch), indexInMiniBatch, xOriginalNotAugmentedMiniBatchCpu, yOriginalBufferMiniBatchCpu, false, false);
                    }
                }
                yOriginalBufferMiniBatchCpu.CopyTo(yDataAugmentedBufferMiniBatchCpu);
                swLoad.Stop();
                swDA.Start();
                int MiniBatchIdxToCategoryIndex(int miniBatchIdx) => dataset.ElementIdToCategoryIndex(MiniBatchIdxToElementId(miniBatchIdx));
                Lazy<ImageStatistic> MiniBatchIdxToImageStatistic(int miniBatchIdx) => new(() => dataset.ElementIdToImageStatistic(MiniBatchIdxToElementId(miniBatchIdx), channels, targetHeight, targetWidth));
                if (useMultiThreading)
                {
                    Parallel.For(0, miniBatchSize, indexInMiniBatch => imageDataGenerator.DataAugmentationForMiniBatch(
                                                       indexInMiniBatch,
                                                       xOriginalNotAugmentedMiniBatchCpu,
                                                       xBufferMiniBatchCpu,
                                                       yOriginalBufferMiniBatchCpu,
                                                       yOriginalBufferMiniBatchCpu,
                                                       MiniBatchIdxToCategoryIndex,
                                                       MiniBatchIdxToImageStatistic,
                                                       dataset.MeanAndVolatilityForEachChannel,
                                                       dataset.GetRandomForIndexInMiniBatch(indexInMiniBatch),
                                                       xBufferForDataAugmentedMiniBatch));
                }
                else
                {
                    for (int indexInMiniBatch = 0; indexInMiniBatch < miniBatchSize; ++indexInMiniBatch)
                    {
                        imageDataGenerator.DataAugmentationForMiniBatch(indexInMiniBatch, xOriginalNotAugmentedMiniBatchCpu, xBufferMiniBatchCpu,  yOriginalBufferMiniBatchCpu, yDataAugmentedBufferMiniBatchCpu, MiniBatchIdxToCategoryIndex, MiniBatchIdxToImageStatistic, dataset.MeanAndVolatilityForEachChannel, dataset.GetRandomForIndexInMiniBatch(indexInMiniBatch), xBufferForDataAugmentedMiniBatch);
                    }
                }

                //var meanAndVolatilityOfEachChannel = new List<Tuple<float, float>> { Tuple.Create(147.02734f, 60.003986f), Tuple.Create(141.81636f, 51.15815f), Tuple.Create(130.15608f, 48.55502f) };
                //var xCpuChunkBytes = xOriginalNotAugmentedMiniBatchCpu /*xBufferMiniBatchCpu*/.Select((n, c, val) => (byte)((val * meanAndVolatilityOfEachChannel[c].Item2 + meanAndVolatilityOfEachChannel[c].Item1)));
                //for (int i = 0; i < 10; ++i)
                //{
                //    SharpNet.Pictures.PictureTools.SaveBitmap(xCpuChunkBytes, i, System.IO.Path.Combine(NetworkSample.DefaultLogDirectory, "Train"), shuffledElementId[i].ToString("D5"), "");
                //}

                swDA.Stop();
            }
            var comment = "count=" + count.ToString("D4") + ",miniBatchSize=" + miniBatchSize.ToString("D4") + ", useMultiThreading=" + (useMultiThreading ? 1 : 0);
            comment += " ; load into memory took " + swLoad.ElapsedMilliseconds.ToString("D4") + " ms";
            comment += " ; data augmentation took " + swDA.ElapsedMilliseconds.ToString("D4") + " ms";
            Log.Info(comment);
        }

        //[Test, Explicit]
        //public void BenchmarkLoadAt()
        //{
        //    const bool useMultiThreading = true;
        //    const int miniBatchSize = 1024;
        //    var p = CFM60NetworkSample.Default();
        //    p.Config.BatchSize = miniBatchSize;

        //    using var cfm60TrainingAndTestDataSet = new Cfm60TrainingAndTestDataset(p, s => Model.Log.Info(s));
        //    var dataset = (CFM60DataSet)cfm60TrainingAndTestDataSet.Training;

        //    var xMiniBatchShape = new[] { miniBatchSize, 3, dataset.Sample.Encoder_TimeSteps, p.CFM60Hyperparameters.Encoder_InputSize };

        //    var rand = new Random(0);