using System;
using System.Diagnostics;
using System.Globalization;

namespace SharpNet.HPO;

public class FloatRangeHyperparameterSearchSpace : RangeHyperparameterSearchSpace
{
    #region private fields
    private readonly float _min;
    private readonly float _max;
    #endregion

    public FloatRangeHyperparameterSearchSpace(float min, float max, range_type rangeType) : base(rangeType)
    {
        Debug.Assert(max>=min);
        _min = min;
        _max = max;
    }

    public override float Next_BayesianSearchFloatValue(Random rand, RANDOM_SEARCH_OPTION randomSearchOption)
    {
        return Next_BayesianSearchFloatValue(_min, _max, rand, _rangeType, randomSearchOption, StatsByBucket);
    }
    public override void RegisterScore(object sampleValue, IScore score, double elapsedTimeInSeconds)
    {
        int bucketIndex = 