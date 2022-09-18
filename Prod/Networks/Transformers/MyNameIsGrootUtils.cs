using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using log4net;
using SharpNet.CPU;
using SharpNet.Datasets;
using SharpNet.HPO;
using SharpNet.Hyperparameters;
// ReSharper disable UnusedMember.Global

namespace SharpNet.Networks.Transformers;

public static class MyNameIsGrootUtils
{
    public const string NAME = "MyNameIsGroot";


    #region public fields & properties
    private static readonly ILog Log = LogManager.GetLogger(typeof(MyNameIsGrootUtils));
    #endregion

    public static string WorkingDirectory => Path.Combine(Utils.ChallengesPath, NAME);


    private static string GenerateText(Network nn, int textLength, double maxAllowedError)
    {
        var outputShape = nn.YPredicted_MiniBatch_Shape(1);
        var max_length = outputShape[1];
        var vocab_size = outputShape[2];
        var datasetSample = new MyNameIsGrootDatasetSample() { max_length = max_length, vocab_size = vocab_size };
        var tokenizer = datasetSample.GetTokenizer();

        var xInputSingleRow = new CpuTensor<float>