using System;
using System.Collections.Generic;
using JetBrains.Annotations;
using SharpNet.CPU;

namespace SharpNet.Datasets;

public abstract class WrappedDataSet : DataSet
{
    protected readonly DataSet _original;

    protected WrappedDataSet(DataSet original, [CanBeNull] string[] Y_IDs)
        : base(original.Name,
            original.DatasetSample,
            original.MeanAndVolatilityForEachChannel,
            original.ResizeStrategy,
            orig