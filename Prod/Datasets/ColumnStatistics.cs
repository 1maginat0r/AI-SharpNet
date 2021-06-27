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
    private readonly List<string> _distinctCategoricalV