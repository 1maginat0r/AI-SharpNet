using System;

namespace SharpNet.Datasets;

public class ConstDatasetSample : AbstractDatasetSample
{
    private readonly Objective_enum Objective;
    private readonly int[] x_shape_for_1_batchSize;
    private readonly int[] y_shape_for_1_batchSize;
    private readonly Func<string, bool> isCategoricalColumn;

    public ConstDatasetSample(stri