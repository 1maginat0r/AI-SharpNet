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
        const string workingDirectory = @"C:\Projects\Challenges\KaggleDays\aaa7\";
        var modelName = new[]
        {
            "9F587BDFA9_KFOLD",
            "DEBB5D22D9_KFOLD",
           
        };
        const bool use_features_in_secondary = true;
        const int cv = 2;

        Console.WriteLine($"Performing Stacked Ensemble Training with {modelName.Length} models in directory {workingDirectory}");

        var workingDirectoryAndModelNames = modelName.Select(m => Tuple.Create(workingDirectory, m, m + "_FULL")).ToList();
        var datasetSample = StackingCVClassifierDatasetSample.New(workingDirectoryAndModelNames, workingDirectory, use_features_in_secondary, cv);

        var searchSpace = new Dictionary<string, object>
        {
            //related to Dataset 
            { "KFold", cv },

            //high priority
            //TODO
            //{"objective", nameof(LightGBMSample.objective.?)},
            //{"metric", ?}, 
            { "bagging_fraction", new[]{/*0.8f,*/ 0.9f /*, 1.0f*/} },
            { "bagging_freq", new[]{0, 1} },
            { "boosting", new []{/*"gbdt",*/ "dart"}},
            { "colsample_bytree",HyperparameterSearchSpace.Range(0.3f, 1.0f)},
            { "early_stopping_round", num_iterations/10 },
            { "lambda_l1",HyperparameterSearchSpace.Range(0f, 2f)},
            { "learning_rate",HyperparameterSearchSpace.Range(0.005f, 0.2f)},
            { "max_depth", new[]{10, 20, 50, 100 /*, 255*/} },
            { "min_data_in_leaf", new[]{20, 50 /*,100*/} },
            { "num_iterations", num_iterations },
            { "num_leaves", HyperparameterSearchSpace.Range(3, 50) },
            { "num_threads", 1},
            { "verbosity", "0" },

            ////medium priority
            { "drop_rate", new[]{0.05, 0.1, 0.2}},                               //specific to dart mode
            { "lambda_l2",HyperparameterSearchSpace.Range(0f, 2f)},
            { "min_data_in_bin", new[]{3, 10, 100, 150}  },
            { "max_bin", HyperparameterSearchSpace.Range(10, 255) },
            { "max_drop", new[]{40, 50, 60}},                                   //specific to dart mode
            { "skip_drop",HyperparameterSearchSpace.Range(0.1f, 0.6f)},  //specific to dart mode

            ////low priority
            //{ "extra_trees", new[] { true , false } }, //low priority 
            ////{ "colsample_bynode",AbstractHyperparameterSearchSpace.Range(0.5f, 1.0f)}, //very low priority
            //{ "path_smooth", AbstractHyperparameterSearchSpace.Range(0f, 1f) }, //low priority
            //{ "min_sum_hessian_in_leaf", AbstractHyperparameterSearchSpace.Range(1e-3f, 1.0f) },
        };

        var hpoWorkingDirectory = Path.Combine(workingDirectory, "hpo");
        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(new LightGBMSample(), datasetSample), hpoWorkingDirectory); IScore bestScoreSoFar = null;
        hpo.Process(t => SampleUtils.TrainWithHyperparameters((ModelAndDatasetPredictionsSample)t, hpoWorkingDirectory, true, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
    }



    public static void RetrainWithContentUpdate(string workingDirectory, string modelName, Action<IDictionary<string, string>> contentUpdater, bool useAllAvailableCores = true)
    {
        var sw = Stopwatch.StartNew();
        ISample.Log.Info($"Retraining model '{modelName}' with update on parameters ");
        using var modelAndDataset = ModelAndDatasetPredictions.Load(workingDirectory, modelName, useAllAvailableCores, contentUpdater);
        Model.Log.Info($"Training Model '{modelAndDataset.Model.ModelName}' (= Model '{modelName}' with contentUpdater");
        modelAndDataset.Fit(true, true, true);
        ISample.Log.Info($"Model {modelName} retrained in {sw.Elapsed.TotalSeconds}");
    }
    /// <summary>
    /// retrain some models 
    /// </summary>
    /// <param name="workingDirectory"></param>
    /// <param name="modelName"></param>
    /// <param name="n_splits"></param>
    /// <param name="percentageInTraining"></param>
    /// <param name="retrainOnFullDataset"></param>
    /// <param name="useAllAvailableCores"></param>
    /// <param name="computeAndSavePredictions"></param>
    /// <param name="computeValidationRankingScore"></param>
    /// <param name="saveTrainedModel"></param>
    /// <returns>the name of the retrained model</returns>
    /// <exception cref="ArgumentException"></exception>
    public static string Retrain(string workingDirectory, string modelName, int? n_splits = 3, double?percentageInTraining = null, bool retrainOnFullDataset = true, bool useAllAvailableCores = true, bool computeAndSavePredictions = true, bool computeValidationRankingScore = true, bool saveTrainedModel = true)
    {

        string newModelName 