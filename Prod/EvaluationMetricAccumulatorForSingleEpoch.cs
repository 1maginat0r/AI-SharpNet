using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using JetBrains.Annotations;
using SharpNet.Data;
using SharpNet.Hyperparameters;
using SharpNet.MathTools;

namespace SharpNet;

public class EvaluationMetricAccumulatorForSingleEpoch : IDisposable
{
    private readonly TensorMemoryPool _memoryPool;
    private readonly int _count;
    private int _currentElementCount;
    private readonly IMetricConfig _metricData;
    private Tensor _full_y_true;
    private Tensor _full_y_pred;
    private readonly Dictionary<EvaluationMetricEnum, DoubleAccumulator> _currentAccumulatedMetrics = new();

    /// <summary>
    /// 
    /// </summary>
    /// <param name="memoryPool"></param>
    /// <param name="count">number of elements (rows) in the dataset</param>
    /// <param name="metricConfig"></param>
    public EvaluationMetricAccumulatorForSingleEpoch(TensorMemoryPool memoryPool, int count, IMetricConfig metricConfig)
    {
        _memoryPool = memoryPool;
        _count = count;
        _metricData = metricConfig;
    }

    public void UpdateMetrics([NotNull] Tensor yExpected, [NotNull] Tensor yPredicted)
    {
        var remainingCount = _count - _currentElementCount;
        if (remainingCount <= 0) 
        {
            throw new ArgumentException($"remainingCount = {remainingCount} and received a tensor of {yExpected.Shape[0]} elements");
        }
        if (yExpected.Shape[0] > remainingCount)
        {
            var newShape = Utils.CloneShapeWithNewCount(yExpected.Shape, remainingCount);
            yExpected = yExpected.Reshape(newShape);
            yPredicted = yPredicted.Reshape(newShape);
        }

        if 