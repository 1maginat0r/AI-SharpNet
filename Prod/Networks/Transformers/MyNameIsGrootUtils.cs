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

        var xInputSingleRow = new CpuTensor<float>(new[] { 1, max_length});
        var xInputSingleRowSpan = xInputSingleRow.SpanContent;

        var fulltext = datasetSample.GetText();
        var r = new Random();
        int randomStartIdx = r.Next(fulltext.Length/2 - max_length-1);

        List<int> tmpSequence = tokenizer.TextsToSequences(new[] { fulltext.Substring(randomStartIdx, 10*max_length) })[0].Take(max_length).ToList();

        int[] newSequence = new int[max_length+textLength];
        for (int j = 0; j < max_length; j++)
        {
            newSequence[j] = tmpSequence[j];
        }

        int nextIndexToGenerate = max_length;
        while (nextIndexToGenerate < newSequence.Length)
        {
            //we know the sequence newSequence[nextIndexToGenerate-max_length] to newSequence[nextIndexToGenerate-1]
            //we want to compute next sequence item at position newSequence[nextIndexToGenerate]

            for (int i = 0; i < max_length; i++)
            {
                xInputSingleRowSpan[i] = newSequence[nextIndexToGenerate - max_length+i];
            }

            var prediction = nn.Predict(xInputSingleRow, false);
            var proba = prediction.As2DTensor(true).RowSlice(max_length-1, 1).ContentAsFloatArray();
            var indexNextToken = GetIndexPrediction(proba, r, maxAllowedError);
            newSequence[nextIndexToGenerate] = indexNextToken;
            ++nextIndexToGenerate;
        }
        var generateText  =tokenizer.SequenceToText(newSequence.Skip(max_length));
        return generateText;
    }


    private static int GetIndexPrediction(float[] proba, Random r, double maxAllowedError)
    {
        List<Tuple<float, int>> probaWithIndex = new List<Tuple<float, int>>();
        for (int i = 0; i < proba.Length; i++)
        {
            probaWithIndex.Add(new Tuple<float, int>(proba[i], i));
        }
        probaWithIndex = probaWithIndex.OrderByDescending(x => x.Item1).ToList();
        int selectionChoice = 1;
        for (int i = 1; i < probaWithIndex.Count; i++)
        {
            if (probaWithIndex[i].Item1 > (1.0*probaWithIndex[0].Item1- maxAllowedError))
            {
                ++selectionChoice;
            }
            else
            {
                break;
            }
        }

        return probaWithIndex[r.Next(selectionChoice)].Item2;
    }

    public static void Run()
    {
        Utils.ConfigureGlobalLog4netProperties(WorkingDirectory, "log");
        Utils.ConfigureThreadLog4n