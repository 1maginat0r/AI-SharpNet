using System;
using System.Diagnostics;
using System.Linq;
using NUnit.Framework;
using SharpNet;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.GPU;
using SharpNetTests.CPU;
using SharpNetTests.NonReg;

namespace SharpNetTests.GPU
{
    [TestFixture]
    public class KernelManagerTests
    {
        private static GPUWrapper GpuWrapper => TestGPUTensor.GpuWrapper;
        private readonly Random _rand = new (0);

        [Test]
        public void KernelManagerTest()
        {
            var km = new KernelManager(GpuWrapper);
            const int size = 1 << 20;
            var shape = new[] { 1, size, 1, 1 };
            var aCpu = RandomTensor(shape);
            var bCpu = RandomTensor(shape);
            var resultCpu = new CpuTensor<float>(shape);
            for (int i = 0; i < aCpu.Count; ++i)
            {
                resultCpu[i] = aCpu[i] + bCpu[i];
            }

            //GPU single precision test
            Tensor a = aCpu.ToGPU<float>(GpuWrapper);
            Tensor b = bCpu.ToGPU<float>(GpuWrapper);
            Tensor resultGpu = new GPUTensor<float>(shape, null, GpuWrapper);
            km.RunKernel("Sum", resultGpu.Count, new object[] { a, b, resultGpu }, 0);
            Assert.IsTrue(TensorExtensions.SameFloatContent(resultCpu, resultGpu, 1e-2));
        }

        [Test]
        public void TestCudaParsingKernelManagerTest()
        {
            var cudaSrcCode =
                " //  y = (x-mean)/volatility" + Environment.NewLine +
                " __global__ void StandardizeInPlaceByRow(int N, int col