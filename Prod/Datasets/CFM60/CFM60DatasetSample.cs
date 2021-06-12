using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using SharpNet.HPO;
using SharpNet.Hyperparameters;
using SharpNet.Networks;
// ReSharper disable UnusedMember.Global
// ReSharper disable MemberCanBePrivate.Global
// ReSharper disable FieldCanBeMadeReadOnly.Global

namespace SharpNet.Datasets.CFM60;

public class CFM60DatasetSample : DatasetSampleForTimeSeries
{
    private static readonly TimeSeriesSinglePoint[] EntriesTrain;
    private static readonly TimeSeriesSinglePoint[] EntriesTest;

    static CFM60DatasetSample()
    {
        EntriesTrain = CFM60Entry.Load(
            Path.Combine(NetworkSample.DefaultDataDirectory, "CFM60", "input_training.csv"),
            Path.Combine(NetworkSample.DefaultDataDirectory, "CFM60", "output_training_IxKGwDV.csv"),
            _ => { });
        EntriesTest = CFM60Entry.Load(
            Path.Combine(NetworkSample.DefaultDataDirectory, "CFM60", "input_test.csv"),
            null,
            _ => { });

    }


    public override bool PredictionsMustBeOrderedByIdColumn => true;

    protected override int CountOfDistinctCategoricalValues(string columnName)
    {
        switch (columnName)
        {
            case "pid": return CFM60Entry.DISTINCT_PID_COUNT;
            default:
                throw new ArgumentException($"unknown categorical feature {columnName}");
        }
    }
    public override int LoadEntry(TimeSeriesSinglePoint entry0, float prev_Y, Span<float> xElementId, int idx, EncoderDecoder_NetworkSample networkSample, bool isEncoder)
    {
        var entry = (CFM60Entry)entry0;
        int start_idx = idx;

        if (isEncoder)
        {
            xElementId[idx++] = entry.ID;
        }

        //pid
        if (networkSample.Pid_EmbeddingDim >= 1)
        {
            //pids are in range  [0, 899]
            //EmbeddingLayer is expecting them in range [1,900] that's why we add +1
            xElementId[idx++] = entry.pid + 1;
        }

        //day/year
        if (Use_day)
        {
            xElementId[idx++] = entry.day / Use_day_Divider;
        }

        if (Use_fraction_of_year)
        {
            xElementId[idx++] = DayToFractionOfYear(entry.day);
        }

        if (Use_year_Cyclical_Encoding)
        {
            xElementId[idx++] = (float)Math.Sin(2 * Math.PI * DayToFractionOfYear(entry.day));
            xElementId[idx++] = (float)Math.Cos(2 * Math.PI * DayToFractionOfYear(entry.day));
        }

        if (Use_EndOfYear_flag)
        {
            xElementId[idx++] = EndOfYear.Contains(entry.day) ? 1 : 0;
        }

        if (Use_Christmas_flag)
        {
            xElementId[idx++] = Christmas.Contains(entry.day) ? 1 : 0;
        }

        if (Use_EndOfTrimester_flag)
        {
            xElementId[idx++] = EndOfTrimester.Contains(entry.day) ? 1 : 0;
        }

        //abs_ret
        if (Use_abs_ret)
        {
            //entry.abs_ret.AsSpan().CopyTo(xDest.Slice(idx, entry.abs_ret.Length));
            //idx += entry.abs_ret.Length;
            for (int i = 0; i < entry.abs_ret.Length; ++i)
            {
                xElementId[idx++] = entry.abs_ret[i];
            }
      