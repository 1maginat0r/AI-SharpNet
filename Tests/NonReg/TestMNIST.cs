using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using NUnit.Framework;
using SharpNet.Datasets;
using SharpNet.GPU;
using SharpNet.Layers;
using SharpNet.Networks;

namespace SharpNetTests.NonReg
{
    [TestFixture]
    public class TestMNIST
    {
        [Test, Explicit]
        [SuppressMessage("ReSharper", "ConditionIsAlwaysTrueOrFalse")]
        public void Test()
        {
            const bool useGpu = true;

            var network = TestNetwork.NewForTests(
                new NetworkSample
                    {
                        BatchSize = 32,
                        NumEpochs = 1000,
                        DisableReduceLROnPlateau = true,
                        ResourceIds = new List<int> { useGpu ? 0 : -1 },
                        InitialLearningRate = 0.01
                }
                    //.WithAdam()
                    .WithSGD(0.99, true)
                    .WithCyclicCosineAnnealingLearningRateScheduler(10,2),
                NetworkSample.DefaultWorkingDirectory,
                "MNIST"
                );

            //Data Augmentation
            network.Sample.WidthShiftRangeInPercentage = 0.1;
            network.Sample.HeightShiftRangeInPercentage = 0.1;

            var mnist 