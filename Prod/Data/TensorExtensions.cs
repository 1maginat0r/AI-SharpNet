using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using System.Text;
using SharpNet.CPU;

namespace SharpNet.Data
{
    public static class TensorExtensions
    {
        public static string ToShapeAndNumpy(this Tensor t, int maxLength = 2000)
        {
            return Tensor.ShapeToString(t.Shape) + Environment.NewLine + t.ToNumpy(maxLength);
        }


        public static CpuTensor<float> DeduceRowMean(this CpuTensor<float> tensor)
        {
            Debug.Assert(tensor.Shape.Length == 2);
            var rows = tensor.Shape[0];
            var columns = tensor.Shape[1];
            var resultTensor = (CpuTensor<float>)tensor.Clone();
            var resultContent = resultTensor.Content.ToArray();

            for (int row = 0; row < rows; row++)
            {
                float sumForRow = 0.0f;
                for (int column = 0; column < columns; column++)
                {
                    int idx = column + row * columns;
                    sumForRow +