using System.Collections.Generic;

namespace SharpNet.Datasets;

public abstract class WrappedDatasetSample : AbstractDatasetSample
{
    protected AbstractDatasetSample Original { get; }

    protected WrappedDatasetSample(AbstractDatasetSample original)
    {
        Original = original;
    }

    public override bool IsCategoricalColumn(string columnName) => Original.IsCategoricalColumn(columnName);
    public override string IdColumn => Original.IdColumn;
    public override string[] TargetLabels => Original.TargetLabels;
    public override Objective_enum GetObjective() => Original.GetObjective();
    public override DataFrame Predictions_InModelFormat_2_Predictions_InTargetFormat(DataFrame predictions_InModelFormat, Objective_enum objective) => Original.Predictions_InModelFormat_2_Predictions_InTargetFor