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
            