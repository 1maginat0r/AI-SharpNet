using JetBrains.Annotations;
using SharpNet.Models;

namespace SharpNet.Hyperparameters;

public static class SampleUtils
{
    /// <summary>
    /// Train the model (in modelAndDatasetSample) using the dataset (in modelAndDatasetSample)
    /// If validation score of the trained model is better then 'bestScoreSoFar'
    ///     update 'bestScoreSoFar'
    ///     save the model (and associated predictions) in disk
    /// Else
    ///     remove all files associated with the model
    /// </summary>
    /// <param name="modelAndDatasetPredictionsSample">the 'model sample' and dataset to use for training the model</param>
    /// <param name="workingDirectory">the directory where the 'model sample' and 'dataset description' is located</param>
    /// <param name="retrainOnFullDatasetIfBetterModelFound"></param>
    /// <param name="bestScoreSoFar">the best score associated with the best sample found so far for the model</param>
    /// <returns>the score of the ranking evaluation metric for the validation dataset</returns>
    public static IScore TrainWithHyperparameters(
        [NotNull] ModelAndDatasetPredictionsSample modelAndDatasetPredictionsSample, string workingDirectory, bool retrainOnFullDatasetIfBetterModelFound, ref IScore bestScoreSoFar)
    {
        using var modelAndDataset = new ModelAndDatasetPredictions(modelAndDatasetPredictionsSample, workingDirectory, modelAndDatasetPredictionsSample.ComputeHash(), false);
        var model = modelAndDataset.Model;
        var validationRankingScore = modelAndDataset.Fit(false, true, false);

        if (validationRankingScore!=null && validationRankingScore.IsBetterThan(bestScoreSoFar))
        {
            Model.Log.Info($"Model '{model.ModelName}' has new best score: {validationRankingScore} (was: {bestScoreSoFar})");
            bestScoreSoFar = validationRankingScore;
            if (bestScoreSoFar.IsBetterThan(modelAndDataset.Model.ModelSample.GetMinimumRankingScoreToSaveModel()))
            {
                var trainAndValidation = modelAndDataset.DatasetSample.SplitIntoTrainingAndValidation();
                modelAndDataset.ComputeAndSavePredictions(trainAndValidation.Training, trainAndValidation.Test, model.ModelName);
                modelAndDataset.Save(workingDirectory, model.ModelName);
                modelAndDataset.Dispose();
                if (retrainOnFullDatasetIfBetterModelFound)
          