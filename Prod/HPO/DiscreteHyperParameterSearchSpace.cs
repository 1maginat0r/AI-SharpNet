using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Globalization;
using System.Linq;

namespace SharpNet.HPO;

public class DiscreteHyperparameterSearchSpace : HyperparameterSearchSpace
{
    #region private fields
    private readonly string[] _allHyperparameterValuesAsString;
    private readonly IDictionary<string, SingleHyperparameterValueStatistics> _statistics = new Dictionary<string, SingleHyperparameterValueStatistics>();
    #endregion

    public DiscreteHyperparameterSearchSpace(object HyperparameterSearchSpace, bool isCategoricalHyperparameter) : base(isCategoricalHyperparameter)
    {
        _allHyperparameterValuesAsString = ToObjectArray(HyperparameterSearchSpace);
        foreach (var e in _allHyperparameterValuesAsString)
        {
            _statistics[e] = new SingleHyperparameterValueStatistics();
        }
    }

    public override bool IsConstant => _allHyperparameterValuesAsString.Length <= 1;

    public override string ToString()
    {
        var res = "";
        var targetInvestmentTime = TargetCpuInvestmentTime();
        foreach (var e in _statistics.OrderBy(e=>(e.Value.CostToDecrease.Count==0)?double.MaxValue:e.Value.CostToDecrease.Average))
        {
            res += " "+e.Key + ":" + e.Value;

            int index = Array.IndexOf(_allHyperparameterValuesAsString, e.Key);
            Debug.Assert(index >= 0);
            res += " (target Time: " + Math.Round(100 * targetInvestmentTime[index], 1) + "%)";

           