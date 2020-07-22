using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using SharpNet.Data;

namespace SharpNet.CPU
{
    public static unsafe class CpuTensorActivationFunctions
    {
        #region Softmax

        public static void SoftmaxLastDimension<T>(CpuTensor<T> X, Tensor Y)
        {
            Softmax(X.Reshape(-1, X.Shape[^1]).AsFloatCpu, Y.Reshape(-1, Y.Shape[^1]));
        }

        public static void Softmax<T>(CpuTensor<T> X, Tensor Y)
        {
            Debug.Assert(Tensor.AreCompatible(new List<Tensor> {X, Y}));
            var bat