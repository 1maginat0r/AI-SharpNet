using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using SharpNet.Datasets;
using SharpNet.HPO;
using SharpNet.Hyperparameters;
using SharpNet.LightGBM;
using SharpNet.Models;
using SharpNet.TextPreprocessing;
// ReSharper disable UnusedMember.Global

namespace SharpNet;

public static class ChallengeTools
{
    /// <summary>
    ///  compute feature importance of a Model
    /// </summary>
    public static void ComputeAndSaveFeatureImportance(string workingDirectory, string modelName, bool computeFeatureImportanceForAllDatasetTypes = false)
    {
        Utils.ConfigureGlobalLog4netProperties(workingDirectory, $"{nameof(ComputeAndSaveFeatureImportance)}");
        Utils.ConfigureThreadLog4netProperties(workingDirectory, $"{nameof(ComputeAndSaveFeatureImportance)}");
        using var m = ModelAndDatasetPredictions.Load(workingDirectory, modelName, true);
        m.ComputeAndSaveFeatureImportance(computeFeatureImportanceForAllDatasetTypes);
    }

    public static void EstimateLossContribution(string workingDirectory, string modelName)
    {
        Utils.ConfigureGlobalLog4netProperties(workingDirectory, $"{nameof(EstimateLossContribution)}");
        Utils.ConfigureThreadLog4netProperties(workingDirectory, $"{nameof(EstimateLossContribution)}");
        using var m = ModelAndDatasetPredictions.Load(workingDirectory, modelName, true);
        m.EstimateLossContribution(computeAlsoRankingScore: true, maxGroupSize: 5000);
    }
    


    /// <summary>
    /// normalize all CSV files in directory 'directory' and put the normalized files in sub directory 'subDirectory'
    /// </summary>
    /// <param name="directory"></param>
    /// <param name="hasHeader"></param>
    /// <param name="removeAccentedCharacters"></param>
    public static void NormalizeAllCsvInDirectory(string directory, bool hasHeader, bool removeAccentedCharacters)
    {
        Utils.ConfigureGlobalLog4netProperties(Path.Combine(directory), $"{nameof(NormalizeAllCsvInDirectory)}");
        Utils.ConfigureThreadLog4netProperties(Path.Combine(directory), $"{nameof(NormalizeAllCsvInDirectory)}");
        DataFrame.NormalizeAllCsvInDirectory(directory, hasHeader, removeAccentedCharacters);
    }


    public static void TfIdfEncode()
    {
        string[] csvFiles = { @"C:\Projects\Challenges\KaggleDays\Data\search_train.csv", @"C:\Projects\Challenges\KaggleDays\Data\search_test.csv" };
        const string columnToEncode = "keyword";


        //string[] csvFiles = { @"C:\Projects\Challenges\KaggleDays\Data\item_info.csv" };
        //string columnToEncode = "name";

        const int embeddingDim = 300;
        const bool hasHeader = true