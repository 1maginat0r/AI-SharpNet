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
        const bool hasHeader = true;
        const bool isNormalized = true;
        const bool keepEncodedColumnName = false;
        const bool reduceEmbeddingDimIfNeeded = false;
        const TfIdfEncoding.TfIdfEncoding_norm norm = TfIdfEncoding.TfIdfEncoding_norm.L2;
        const bool scikitLearnCompatibilityMode = false;

        string directory = Path.GetDirectoryName(csvFiles[0]) ?? "";
        Utils.ConfigureGlobalLog4netProperties(directory, $"{nameof(TfIdfEncode)}");
        Utils.ConfigureThreadLog4netProperties(directory, $"{nameof(TfIdfEncode)}");
        DataFrame.TfIdfEncode(csvFiles, hasHeader, isNormalized, columnToEncode, embeddingDim, 
            keepEncodedColumnName, reduceEmbeddingDimIfNeeded, norm, scikitLearnCompatibilityMode);
    }

    ///// <summary>
    ///// encode the string column 'columnToEncode' using Tf*Idf with 'embeddingDim' words and return a new DataFrame with this encoding
    ///// </summary>
    ///// <param name="columnToEncode"></param>
    ///// <param name="embeddingDim">the number of dimension for the encoding.
    ///// Only the top frequent 'embeddingDim' words will be considered for the encoding.
    ///// The other will be discarded</param>
    ///// <param name="keepEncodedColumnName">
    ///// Each new feature will have in its name the associated word for the TfIdf encoding</param>
    ///// <param name="reduceEmbeddingDimIfNeeded"></param>
    ///// <param name="norm"></param>
    ///// <param name="scikitLearnCompatibilityMode"></param>
    ///// <returns></returns>
    //public DataFrame TfIdfEncode(string columnToEncode, int embeddingDim, bool keepEncodedColumnName = false, bool reduceEmbeddingDimIfNeeded = false, TfIdfEncoding.TfIdfEncoding_norm norm = TfIdfEncoding.TfIdfEncoding_norm.L2, bool scikitLearnCompatibilityMode = false)
    //{
    //    return TfIdfEncoding.Encode(new[] { this }, columnToEncode, embeddingDim, keepEncodedColumnName, reduceEmbeddingDimIfNeeded, norm, scikitLearnCompatibilityMode)[0];
    //}



    /// <summary>
    /// Stack several trained models together to compute new predictions
    /// (through a new LightGBM model that will be trained to do the stacking)
    /// </summary>
//    [TestCase(100,0), Explicit]
    public static void StackedEnsemble(int num_iterations = 100, int maxAllowedSecondsForAllComputation = 0)
    {
        //const string workingDirectory = @"C:/Projects/Challenges/WasYouStayWorthItsPrice/submission";
        co