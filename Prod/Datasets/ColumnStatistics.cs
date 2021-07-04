using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using SharpNet.MathTools;

namespace SharpNet.Datasets;

public class ColumnStatistics
{
    #region private fields
    private readonly bool _standardizeDoubleValues;
    private readonly bool _allDataFrameAreAlreadyNormalized;
    private readonly Dictionary<string, int> _distinctCategoricalValueToCount = new();
    private readonly List<string> _distinctCategoricalValues = new();
    private readonly Dictionary<string, int> _distinctCategoricalValueToIndex = new();
    private readonly DoubleAccumulator _numericalValues = new();
    #endregion

    #region public properties
    /// <summary>
    /// true if the column is categorical (non numerical)
    /// </summary>
    public bool IsCategorical { get; }
    /// <summary>
    /// true if the column contains the target of the dataset
    /// </summary>
    public bool IsTargetLabel { get; }
    /// <summary>
    /// true if the column is (among) the ids needed to identify a unique row
    /// </summary>
    public bool IsId { get; }
    /// <summary>
    /// number of elements of the given features
    /// </summary>
    public int Count { get; private set; }
    /// <summary>
    /// number of empty elements in the DataSet for this column
    /// </summary>
    public int CountEmptyElements { get; private set; }
    public IList<string> GetDistinctCategoricalValues() => _distinctCategoricalValues;
    #endregion

    #region constructors
    /// <summary>
    /// 
    /// </summary>
    /// <param name="isCategorical"></param>
    /// <param name="isTargetLabel"></param>
    /// <param name="isId"></param>
    /// <param name="standardizeDoubleValues">
    /// true if we should standardize double values (mean 0 and volatility 1)
    /// false if we should not transform double values
    /// </param>
    /// <param name="allDataFrameAreAlreadyNormalized"></param>
    public ColumnStatistics(bool isCategorical, bool isTargetLabel, bool isId, bool standardizeDoubleValues, bool allDataFrameAreAlreadyNormalized)
    {
        _standardizeDoubleValues = standardizeDoubleValues;
        _allDataFrameAreAlreadyNormalized = allDataFrameAreAlreadyNormalized;
        IsCategorical = isCategorical;
        IsTargetLabel = isTargetLabel;
        IsId = isId;
    }
    #endregion

    public void Fit(string val_before_encoding)
    {
        ++Count;
        if (IsCategorical)
        {
            lock(this)
            {
                if (!_allDataFrameAreAlreadyNormalized)
                {
                    val_before_encoding = Utils.NormalizeCategoricalFeatureValue(val_before_encoding);
                }
                if (val_before_encoding.Length == 0)
                {
                    ++CountEmptyElements;
                    return;
                }
                //it is a categorical column, we add it to the dictionary
                if (!_distinctCategoricalValueToCount.ContainsKey(val_before_encoding))
                {
                    _distinctCategoricalValueToCount[val_before_encoding] = 1;
                    _distinctCategoricalValues.Add(val_before_encoding);
                    _distinctCategoricalValueToIndex[val_before_encoding] = _distinctCategoricalValues.Count - 1;
                }
                else
                {
                    ++_distinctCategoricalValueToCount[val_before_encoding];
                }
            }
            return;
        }

        //it is a numerical field
        var numeric_val_before_encoding = ExtractDouble(val_before_encoding);
        if (double.IsNaN(numeric_val_before_encoding))
        {
            ++CountEmptyElements;
            return; //missing numerical value
        }
        _numericalValues.Add(numeric_val_before_encoding);
    }
    public void Fit(float val_before_encoding)
    {
        ++Count;

        if (IsCategorical)
        {
            if (float.IsNaN(val_before_encoding))
            {
                val_before_encoding = -1; //!D missing category
            }
            var str_val_before_encoding = val_before_encoding.ToString(CultureInfo.InvariantCulture);
            //it is a categorical column, we add it to the dictionary
            if (!_distinctCategoricalValueToCount.ContainsKey(str_val_before_encoding))
            {
                _distinctCategoricalValueToCount[str_val_before_encoding] = 1;
                _distinctCategoricalValues.Add(str_val_before_encoding);
                _distinctCategoricalValueToIndex[str_val_before_encoding] = Utils.NearestInt(val_before_encoding);
                
                Debug.Assert(MathF.Abs(Utils.NearestInt(val_before_encoding) - val_before_encoding) < 1e-5f);
            }
            else
            {
                ++_distinctCategoricalValueToCount[str_val_before_encoding];
            }
            return;
        }

        //it is a numerical field
        if (float.IsNaN(val_before_encoding))
        {
            ++CountEmptyElements;
            return; //missing numerical value
        }
        _numericalValues.Add(val_before_encoding);
    }
    public void Fit(int val_before_encoding)
    {
        Fit((float)val_before_encoding);
    }
    public double Transform(string val_before_encoding)
    {
        if (IsCategorical)
        {
            if (!_allData