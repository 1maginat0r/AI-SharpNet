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
            var batchSize = X.Shape[0];
            var xContent = X.AsReadonlyFloatCpuSpan;
            var yContent = Y.AsFloatCpuSpan;
            for (int row = 0; row < batchSize; ++row)
            {
                int start = row * X.MultDim0;
                int end = start + X.MultDim0;
                var rowMax = float.MinValue;
                for (int i = start; i < end; ++i)
                {
                    rowMax = Math.Max(rowMax, xContent[i]);
                }
                var rowExpSum = 0f;
                for (int i = start; i < end; ++i)
                {
                    var tmp = (float)Math.Exp(xContent[i] - rowMax);
                    rowExpSum += tmp;
                    yContent[i] = tmp;
                }
                for (int i = start; i < end; ++i)
                {
                    yContent[i] /= rowExpSum;
                }
            }
        }

        public static void SoftmaxWithHierarchy<T>(CpuTensor<T> X, Tensor Y, Tensor activationParameter)
        {
            X.CopyTo(Y.AsFloatCpu);
            var activationParameterPointer = (float*)activationParameter.Pointer;
            var yPointer = (float*) Y.Pointer;
            int colSize = Y.MultDim0;
            Parallel.For(0, Y.Shape[0], m =>{int idx = 0;SoftmaxWithHierarchy(activationParameterPointer, yPointer+m*colSize, colSize, &idx);});
        }

        private static void SoftmaxWithHierarchy(float* activationParameter, float* y, int endIndexExcluded, int *pNexIndexToCheck)
        {
            float param = activationParameter[*pNexIndexToCheck];
            y[*pNexIndexToCheck] = param;
            int subCategoriesCount = ExtractCount(param);
            *pNexIndexToCheck += 1;
            int[] indexesProba = new int[subCategoriesCount];
            float maxProba = -1e9f;
            bool probaFound = false;

            for(int subCategoriesFound = 0;subCategoriesFound < subCategoriesCount; ++subCategoriesFound)
            {
                float expectedProba = activationParameter[*pNexIndexToCheck];
                if (IsProba(expectedProba))
                {
                    maxProba = fmaxf(maxProba, y[*pNexIndexToCheck]);
                    indexesProba[subCategoriesFound] = *pNexIndexToCheck;
                    probaFound = true;
                    * pNexIndexToCheck += 1;
                    if (*pNexIndexToCheck < endIndexExcluded && IsCountAssociateWithAboveProba(activationParameter[*pNexIndexToCheck]))
                    {
                        SoftmaxWithHierarchy(activationParameter, y, endIndexExcluded, pNexIndexToCheck);
                    }
                }
                else
                {
                   SoftmaxWithHierarchy(activationParameter, y, endIndexExcluded,  pNexIndexToCheck);
                }
            }

            if (probaFound)
            {
                float sumExp = 0.0f;
                for (int i = 0; i < subCategoriesCount; ++i)
                {
                    int idx = indexesProba[i];
                    float tmp = expf(y[idx] - maxProba);
                    sumExp += tmp;
                    y[idx] = tmp;
                }
                for (int i = 0; i < subCategoriesCount; ++i)
                {
                    y[indexesProba[i]] /= sumExp;
                }
            }
        }

        public static void SoftmaxGradientLastDimension(Tensor y, Tensor dy, Tensor dx)
        {
            SoftmaxGradient(y.Reshape(-1, y.Shape[^1]), dy.Reshape(-1, dy.Shape[^1]), dx.Reshape(-1, dx.Shape[^1]));
        }

        public static void SoftmaxGradient(Tensor y, Tensor dy, Tensor dx)
        {
            Debug.Assert(Tensor.AreCompatible(new List<Tensor> { y, dy, dx }));
            var yContent = y.AsFloatCpuSpan;
            var dyContent = dy.AsFloatCpuSpan;
            var dxContent = dx.AsFloatCpuSpan;
            for (int i = 0; i < dx.Count; ++i)
            {
                var yi = yContent[i];
                var dyi = dyContent[i];
                dxContent[i] = (MathF.Abs(dyi - 1.0f) < 1e-6) ? (yi * (1 - yi)) : (-yi * dyi);
            }
        }
        public static void SoftmaxGradientWitHierarchy(Tensor y, Tensor dy, Tensor dx, Tensor activationParameter)
        {
            var activationParameterPointer = (float*)activationParameter.Pointer;
            var yPointer = (float*)y.Pointer;
            var dyPointer = (float*)dy.Pointer;
            var dxPointer = (float*)dx.Pointer;
            int colSize = dx.MultDim0;
            Parallel.For(0, dx.Shape[0], m => SoftmaxGradientWitHierarchy(activationParameterPointer, yPointer+m*colSize, dyPointer + m * colSize, dxPointer + m * colSize, colSize));
        }
        private static void SoftmaxGradientWitHierarchy(float* activationParameter, float* y, float* dy, float* dx, int endIndexExcluded)
        {
            fo