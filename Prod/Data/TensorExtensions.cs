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
                    sumForRow += resultContent[idx];
                }
                float meanForRow = sumForRow/rows;
                for (int column = 0; column < columns; column++)
                {
                    int idx = column + row * columns;
                    resultContent[idx] -= meanForRow;
                }
            }

            return resultTensor;
        }
        
        public static (CpuTensor<float> normalizedTensor, CpuTensor<float> mean, CpuTensor<float> variance) Normalize(this CpuTensor<float> toNormalize)
        {
            Debug.Assert(toNormalize.Shape.Length == 2);
            var rows = toNormalize.Shape[0];
            var columns = toNormalize.Shape[1];
            var mean = CpuTensor<float>.New(new float[columns], columns);
            var variance = CpuTensor<float>.New(new float[columns], columns);
            toNormalize.Compute_Column_Mean_Variance(mean, variance);
            var normalizedContent = toNormalize.Content.ToArray();
            var meanContent = mean.ReadonlyContent;
            var varianceContent = variance.ReadonlyContent;
            for (int row = 0; row < rows; row++)
            for (int column = 0; column < columns; column++)
            {
                var meanValue = meanContent[column];
                var varianceValue = varianceContent[column];
                int idx = column + row * columns;
                normalizedContent[idx] = varianceValue < 1e-5
                    ? 0
                    : (normalizedContent[idx] - meanValue) / MathF.Sqrt(varianceValue);
            }
            return (new CpuTensor<float>(toNormalize.Shape, normalizedContent), mean, variance);
        }



        // ReSharper disable once UnusedMember.Global
        public static string ToCsv(this CpuTensor<float> t, char separator, bool prefixWithRowIndex = false)
        {
            var sb = new StringBuilder();
            var tSpan = t.AsReadonlyFloatCpuSpan;
            int index = 0;
            if (t.Shape.Length != 2)
            {
                throw new ArgumentException($"can only create csv from matrix Tensor, not {t.Shape.Length} dimension tensor");
            }
            for (int row = 0; row < t.Shape[0]; ++row)
            {
                if (prefixWithRowIndex)
                {
                    sb.Append(row.ToString()+separator);
                }
                for (int col = 0; col < t.Shape[1]; ++col)
                {
                    if (col != 