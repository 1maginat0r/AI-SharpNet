
﻿using System;
using System.Linq;
using System.Threading;
using NUnit.Framework;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.GPU;

namespace SharpNetTests.GPU
{
    [TestFixture]
    public class TestGPUTensor
    {
        public static GPUWrapper GpuWrapper => GPUWrapper.FromDeviceId(0);



        [Test]
        public void TestQRFactorization()
        {
            //This test is coming from: https://rosettacode.org/wiki/QR_decomposition#C.23
            var A = new GPUTensor<float>(new[] { 5, 3}, new[] { 12.0f, -51, 4, 6, 167, -68, -4, 24, -41,-1, 1, 0, 2, 0, 3 }, GpuWrapper);
            //var A = new GPUTensor<float>(new[] { 250, 10 }, null, GpuWrapper); CPU.TestCpuTensor.RandomFloatTensor(A.Shape, new System.Random(0), -1, 1).CopyTo(A);
            int m = A.Shape[0];
            int n = A.Shape[1];
            // the orthogonal 'Q' matrix of shape (m, m)
            var Q = new GPUTensor<float>(new[] { m, n }, null, GpuWrapper);
            // the upper triangular matrix 'R' of shape (m, n)
            var R = new GPUTensor<float>(new[] { n, n }, null, GpuWrapper);
            var floatBuffer = new GPUTensor<float>(new [] { A.QRFactorization_FloatBufferLength() }, null, GpuWrapper);
            A.QRFactorization(Q, R, floatBuffer);

            //var sw = System.Diagnostics.Stopwatch.StartNew();
            //int count = 1000;
            //for (int i = 0; i < count; ++i)
            //{
            //    A.QRFactorization(Q, R, floatBuffer);
            //}
            //System.Console.WriteLine("took " + (sw.Elapsed.TotalMilliseconds / count) + "ms"); return;

            var a_clone = A.Clone();
            a_clone.Dot(Q, R);
            Assert.IsTrue(TensorExtensions.SameFloatContent(a_clone, A, 1e-4));

            var maxError = Q.MaxErrorIfOrthogonalMatrix();
            Assert.IsTrue(Math.Abs(maxError) < 1e-6);
        }


        [Test]
        public void TestDot()
        {
            // [1x1] x [1x1] matrix
            float[] data = new[] { 2f};
            var a = new GPUTensor<float>(new[] { 1, 1 }, data, GpuWrapper);
            float[] data1 = new[] { 5f };
            var b = new GPUTensor<float>(new[] { 1, 1 }, data1, GpuWrapper);
            var result = new GPUTensor<float>(new[] { 1, 1 }, null, GpuWrapper);
            result.Dot(a, false, b, false, 1, 0);
            var expected = new CpuTensor<float>(new[] { 1, 1 }, new[] { 10f });
            Assert.IsTrue(TensorExtensions.SameFloatContent(expected, result, 1e-9));

            // [2x2] x [2x2] matrix
            float[] data2 = new []{1f,2, 3, 4 };
            a = new GPUTensor<float>(new []{2,2}, data2, GpuWrapper);
            float[] data3 = new []{1f,2, 3, 4 };
            b = new GPUTensor<float>(new []{2,2}, data3, GpuWrapper);
            result = new GPUTensor<float>(new[] { 2, 2}, null, GpuWrapper);
            result.Dot(a, false, b, false, 1, 0);
            expected = new CpuTensor<float>(new[] { 2, 2 }, new[] { 7f, 10f, 15f, 22f });
            Assert.IsTrue(TensorExtensions.SameFloatContent(expected, result, 1e-9));

            // [1x2] x [2x1] matrix
            float[] data4 = new[] { 1f, 2};
            a = new GPUTensor<float>(new[] { 1, 2 }, data4, GpuWrapper);
            float[] data5 = new[] { 3f, 4 };
            b = new GPUTensor<float>(new[] { 2, 1 }, data5, GpuWrapper);
            result = new GPUTensor<float>(new[] { 1, 1 }, null, GpuWrapper);
            result.Dot(a, false, b, false, 1, 0);
            expected = new CpuTensor<float>(new[] { 1, 1 }, new[] { 11f });
            Assert.IsTrue(TensorExtensions.SameFloatContent(expected, result, 1e-9));

            // [2x1] x [1x2] matrix
            float[] data6 = new[] { 1f, 2 };
            a = new GPUTensor<float>(new[] { 2, 1 }, data6, GpuWrapper);
            float[] data7 = new[] { 3f, 4 };
            b = new GPUTensor<float>(new[] { 1, 2 }, data7, GpuWrapper);
            result = new GPUTensor<float>(new[] { 2, 2 }, null, GpuWrapper);
            result.Dot(a, false, b, false, 1, 0);
            expected = new CpuTensor<float>(new[] { 2, 2 }, new[] { 3f, 4, 6, 8 });
            Assert.IsTrue(TensorExtensions.SameFloatContent(expected,result, 1e-9));

            // [2x1x1] x [1x1x2] matrix
            float[] data8 = new[] { 1f, 2 };
            a = new GPUTensor<float>(new[] { 2, 1,1 }, data8, GpuWrapper);
            float[] data9 = new[] { 3f, 4 };
            b = new GPUTensor<float>(new[] { 1, 1,2 }, data9, GpuWrapper);
            result = new GPUTensor<float>(new[] { 2, 2 }, null, GpuWrapper);
            result.Dot(a, false, b, false, 1, 0);
            expected = new CpuTensor<float>(new[] { 2, 2 }, new[] { 3f, 4, 6, 8 });
            Assert.IsTrue(TensorExtensions.SameFloatContent(expected, result, 1e-9));

            // [10x1x28x28] x [784x100] matrix
            a = new GPUTensor<float>(new[] { 10, 1, 28,28 }, null, GpuWrapper);
            b = new GPUTensor<float>(new[] { 784, 100 }, null, GpuWrapper);
            result = new GPUTensor<float>(new[] { 10, 1,1,100 }, null, GpuWrapper);
            result.Dot(a, false, b, false, 1, 0);
            expected = new CpuTensor<float>(new[] { 10, 1,1,100 });
            Assert.IsTrue(TensorExtensions.SameFloatContent(expected, result, 1e-9));

            // [35x124] x [124x21] matrix
            var data10 = new float[35*124];
            a = new GPUTensor<float>(new[] { 35, 124 }, data10, GpuWrapper);
            var data11 = new float[124*21];
            b = new GPUTensor<float>(new[] { 124, 21}, data11, GpuWrapper);
            result = new GPUTensor<float>(new[] { 35, 21 }, null, GpuWrapper);
            result.Dot(a, false, b, false, 1, 0);
            expected = new CpuTensor<float>(new[] { 35, 21}, new float[35*21]);
            Assert.IsTrue(TensorExtensions.SameFloatContent(expected, result, 1e-9));
        }

        [Test]
        public void TestInitializeDeviceMemoryFromDeviceMemory()
        {
            var src = new GPUTensor<float>(new [] {5}, new float[] {0, 1, 2, 3, 4}, GpuWrapper);
            var dest = new GPUTensor<float>(new [] {5}, null, GpuWrapper);
            dest.InitializeFromDeviceMemory(src);
            src.ZeroMemory();
            var expectedDest = new GPUTensor<float>(new[] { 5 }, new float[] { 0, 1, 2, 3, 4 }, GpuWrapper);
            Assert.IsTrue(TensorExtensions.SameFloatContent(dest, expectedDest, 1e-6));
            var expectedSrc = new GPUTensor<float>(new[] { 5 }, new float[] { 0, 0, 0, 0, 0 }, GpuWrapper);
            Assert.IsTrue(TensorExtensions.SameFloatContent(src, expectedSrc, 1e-6));
        }


        #region test copy from device (GPU) to device (GPU)
        [Test]
        public void TestCopyBetweenDevices()
        {
            if (GPUWrapper.GetDeviceCount() < 2)
            {
                return; //this test needs at least 2 GPU To work
            }

            const int firstDeviceId = 0;
            int secondDeviceId = (GPUWrapper.GetDeviceCount() < 2)?0:1;


            var shape = new[] { 3 };
            var fistDeviceWrapper = GPUWrapper.FromDeviceId(firstDeviceId);
            var resultInFirstGpuDevice = new GPUTensor<float>(shape, null, fistDeviceWrapper);

            //we wait for the second GPU device to initialize temporary tensors
            ComputationInProgressInFirstDevice = false;
            var t = new Thread(() => RunInSecondDevice(secondDeviceId, resultInFirstGpuDevice));
            t.Start();
            while (!ComputationInProgressInFirstDevice)
            {
                Thread.Sleep(1);
            }

            //we copy the data from first to second GPU device
            var data1InFirstDevice = new GPUTensor<float>(shape, new float[] { 0, 2, 3 }, fistDeviceWrapper);
            Data1InSecondDevice.InitializeFromDeviceMemory(data1InFirstDevice);
            data1InFirstDevice.Dispose();
            var data2InFirstDevice = new GPUTensor<float>(shape, new float[] { 5, 7, 11 }, fistDeviceWrapper);
            Data2InSecondDevice.InitializeFromDeviceMemory(data2InFirstDevice);
            data2InFirstDevice.Dispose();

            //we wait for the computation to be finished in 2nd GPU device (and copied to 1st device)
            ComputationInProgressInFirstDevice = false;
            while (!ComputationInProgressInFirstDevice)
            {
                Thread.Sleep(1);
            }

            //we ensure that the data has been successfully computed in 2nd device copied to 1st device
            var expectedResult = new GPUTensor<float>(shape, new float[] { 0 * 5, 2 * 7, 3 * 11 }, fistDeviceWrapper);
            Assert.IsTrue(TensorExtensions.SameFloatContent(expectedResult, resultInFirstGpuDevice, 1e-6));
        }
        //volatile to avoid any optimizations made by the compiler/jitter
        private volatile bool ComputationInProgressInFirstDevice;
        private GPUTensor<float> Data1InSecondDevice;
        private GPUTensor<float> Data2InSecondDevice;
        private void RunInSecondDevice(int secondDeviceId, GPUTensor<float> resultInFirstGpuDevice)
        {
            var deviceWrapper = GPUWrapper.FromDeviceId(secondDeviceId);
            Data1InSecondDevice = new GPUTensor<float>(resultInFirstGpuDevice.Shape, null, deviceWrapper);
            Data2InSecondDevice = new GPUTensor<float>(resultInFirstGpuDevice.Shape, null, deviceWrapper);

            //we wait for the 1st device to initialize tensors 'Data1InSecondDevice' and 'Data2InSecondDevice'
            ComputationInProgressInFirstDevice = true;
            while (ComputationInProgressInFirstDevice)
            {
                Thread.Sleep(1);
            }

            //we compute the result in 2nd GPU device (secondDeviceId=1), and store it in 1st GPU device (secondDeviceId=0 )
            var resultInSecondGpuDevice = new GPUTensor<float>(resultInFirstGpuDevice.Shape, null, deviceWrapper);
            resultInSecondGpuDevice.MultiplyTensor(Data1InSecondDevice, Data2InSecondDevice);
            Data1InSecondDevice.Dispose();
            Data1InSecondDevice = null;
            Data2InSecondDevice.Dispose();
            Data2InSecondDevice = null;
            resultInFirstGpuDevice.InitializeFromDeviceMemory(resultInSecondGpuDevice);
            resultInSecondGpuDevice.Dispose();
            //we have finished the computation in 2nd GPU device, and copied the results in 1st GPU device
            ComputationInProgressInFirstDevice = true;
        }
        #endregion

        [Test]
        public void TestOwner()
        {
            var data = new float[]{0,1,2,3,4,5,6,7,8,9};
            var owner = new GPUTensor<float>(new []{5,2}, data, GpuWrapper);
            var tensorTop2Rows = owner.Slice(0, new[] { 2, 2});
            var tensorBottom3Rows = owner.Slice(tensorTop2Rows.Count, new[] { 3, 2 });
            var contentTop = tensorTop2Rows.ContentAsFloatArray();
            Assert.AreEqual(new float[]{0,1,2,3}, contentTop.ToArray());
            var contentBottom = tensorBottom3Rows.ContentAsFloatArray();
            Assert.AreEqual(new float[] { 4, 5, 6, 7, 8, 9 }, contentBottom.ToArray());
        }
    }
}