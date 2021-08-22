using System;
using System.Collections.Generic;
using System.Diagnostics;
using SharpNet.CPU;

namespace SharpNet.Datasets;

public sealed class RandomizeColumnDataSet : WrappedDataSet
{
    private readonly List<string> _columnNameToRandomize;
    private readonly Random _r;
    private CpuTensor<float> tmp = null;


    private CpuTensor<float> GetBuffer(int[] expectedShape)
    {
        if (tmp == null)
        {
            tmp = new CpuTensor<float>(expectedShape);
        }
        else
        {
            tmp.ReshapeInPlace(expectedShape);
        }
        return tmp;
    }
    

    public RandomizeColumnDataSet(DataSet original, List<string> columnNameToRandomize, Random r)
        : base(original, original.Y_IDs)
    {
        _columnNameToRandomize = columnNameToRandomize;
        _r = r;

    }

    public override void LoadAt(int subElementId, int indexInBuffer, CpuTensor<float> xBuffer,
        CpuTensor<float> yBuffer, bool withDataAugmentation, bool isTraining)
    {
        _original.LoadAt(subElementId, indexInBuffer, xBuffer, yBuffer, withDataAugmentation, isTraining);
        if (xBuffer != 