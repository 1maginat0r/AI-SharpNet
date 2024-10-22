
﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using SharpNet.CPU;
using SharpNet.Datasets;
using SharpNet.Hyperparameters;

namespace SharpNet.Models;

/// <summary>
/// Train 'kfold' distinct models on a the training dataset.
/// Use Stacked Ensemble Learning to make predictions for the models 
/// </summary>
public class KFoldModel : Model
{
    #region private fields
    private readonly List<Model> _embeddedModels;
    #endregion
    private const string SuffixKfoldModel = "_KFOLD";

    #region constructors
    public KFoldModel(KFoldSample modelSample, string kfoldWorkingDirectory, string kfoldModelName, AbstractDatasetSample datasetSample) : base(modelSample, kfoldWorkingDirectory, kfoldModelName)
    {
        _embeddedModels = new();
        for (int i = 0; i < modelSample.n_splits; ++i)
        {
            _embeddedModels.Add(LoadEmbeddedModel(i, datasetSample));
        }
    }

    public KFoldModel(KFoldSample modelSample, string kfoldWorkingDirectory, string kfoldModelName, AbstractDatasetSample datasetSample, AbstractModelSample embeddedModelSample) : base(modelSample, kfoldWorkingDirectory, kfoldModelName)
    {
        _embeddedModels = new();
        for (int i = 0; i < modelSample.n_splits; ++i)
        {
            _embeddedModels.Add(embeddedModelSample.NewModel(datasetSample, kfoldWorkingDirectory, KFoldModelNameEmbeddedModelName(ModelName, i)));
        }
    }

    public Model EmbeddedModel(int embeddedModelIndex)
    {
        return _embeddedModels[embeddedModelIndex];
    }


    // ReSharper disable once UnusedParameter.Global
    public static string EmbeddedModelNameToModelNameWithKfold(string embeddedModelName, int n_splits)
    {
        return embeddedModelName + SuffixKfoldModel;
    }


    public static string KFoldModelNameEmbeddedModelName(string kfoldModelName, int index)
    {
        if (index < 0)
        {
            if (kfoldModelName.EndsWith(SuffixKfoldModel))
            {
                kfoldModelName = kfoldModelName.Substring(0, kfoldModelName.Length - SuffixKfoldModel.Length);
            }
            return kfoldModelName;
        }
        return kfoldModelName + "_kfold_" + index;
    }

    private Model LoadEmbeddedModel(int embeddedModelIndex, AbstractDatasetSample datasetSample)
    {
        var e = new Exception();
        //The initial directory of the embedded model may have changed, we'll check also in the KFold Model directory
        foreach (var directory in new[] { KFoldSample.EmbeddedModelWorkingDirectory, WorkingDirectory })
        {
            try { return GetEmbeddedModel(directory, embeddedModelIndex, datasetSample); }
            catch (Exception ex) { e = ex; }
        }
        throw e;
    }

    private Model GetEmbeddedModel(string directory, int embeddedModelIndex, AbstractDatasetSample datasetSample)
    {
        var embeddedModelName = KFoldModelNameEmbeddedModelName(ModelName, embeddedModelIndex);
        AbstractModelSample embeddedModelSample;
        try
        {
            embeddedModelSample = AbstractModelSample.LoadModelSample(directory, embeddedModelName, KFoldSample.Should_Use_All_Available_Cores);
        }
        catch
        {
            //we try to load the embedded model from its original name
            embeddedModelSample = AbstractModelSample.LoadModelSample(directory, KFoldModelNameEmbeddedModelName(ModelName, -1), KFoldSample.Should_Use_All_Available_Cores);
        }
        return embeddedModelSample.NewModel(datasetSample, WorkingDirectory, embeddedModelName);
    }

    #endregion

    public override (string train_XDatasetPath_InModelFormat, string train_YDatasetPath_InModelFormat, string train_XYDatasetPath_InModelFormat, string validation_XDatasetPath_InModelFormat, string validation_YDatasetPath_InModelFormat, string validation_XYDatasetPath_InModelFormat, IScore trainLossIfAvailable, IScore validationLossIfAvailable, IScore trainRankingMetricIfAvailable, IScore validationRankingMetricIfAvailable)
        Fit(DataSet trainDataset, DataSet nullValidationDataset, Func<bool, bool, DataSet, DataSet, string, List<string>> save = null)
    {
        if (ModelSample.GetLoss() == EvaluationMetricEnum.DEFAULT_VALUE)
        {
            throw new ArgumentException("Loss Function not set");
        }
        if (nullValidationDataset != null)
        {
            throw new ArgumentException($"Validation Dataset must be null for KFold");
        }

        const bool includeIdColumns = true;
        const bool overwriteIfExists = false;
        int n_splits = KFoldSample.n_splits;
        var foldedTrainAndValidationDataset = trainDataset.KFoldSplit(n_splits, KFoldSample.CountMustBeMultipleOf);
        var train_XYDatasetPath_InTargetFormat = trainDataset.to_csv_in_directory(DatasetPath, true, includeIdColumns, overwriteIfExists);

        List<IScore> allFoldTrainLossIfAvailable = new();
        List<IScore> allFoldValidationLossIfAvailable = new();
        List<IScore> allFoldTrainRankingMetricIfAvailable = new();
        List<IScore> allFoldValidationRankingMetricIfAvailable = new();

        for (int fold = 0; fold < n_splits; ++fold)
        {
            var embeddedModel = _embeddedModels[fold];
            LogDebug($"Training embedded model '{embeddedModel.ModelName}' on fold[{fold}/{n_splits}]");
            var trainAndValidation = foldedTrainAndValidationDataset[fold];
            var (_, _, _, _, _, _, foldTrainLossIfAvailable, foldValidationLossIfAvailable, foldTrainRankingMetricIfAvailable, foldValidationRankingMetricIfAvailable) = embeddedModel.Fit(trainAndValidation.Training, trainAndValidation.Test, save);
            if (foldTrainLossIfAvailable != null)
            {
                allFoldTrainLossIfAvailable.Add(foldTrainLossIfAvailable);
            }

            if (foldTrainRankingMetricIfAvailable != null)
            {
                allFoldTrainRankingMetricIfAvailable.Add(foldTrainRankingMetricIfAvailable);
            }
            if (foldValidationRankingMetricIfAvailable != null)
            {
                allFoldValidationRankingMetricIfAvailable.Add(foldValidationRankingMetricIfAvailable);
            }

            // we retrieve (or recompute if required) the validation score 
            if (foldValidationLossIfAvailable != null) //the validation score is already available
            {
                //LogDebug($"No need to recompute Validation Loss for fold[{fold}/{n_splits}] : it is already available");
                allFoldValidationLossIfAvailable.Add(foldValidationLossIfAvailable);
            }
            else
            {
                LogDebug($"Computing Validation Loss for fold[{fold}/{n_splits}]");
                var validationDataset = trainAndValidation.Test;
                var fold_y_pred = embeddedModel.Predict(validationDataset, false);
                foldValidationLossIfAvailable = embeddedModel.ComputeLoss(validationDataset.Y_InModelFormat().FloatCpuTensor(), fold_y_pred.FloatCpuTensor());
            }
            LogDebug($"Validation Loss for fold[{fold}/{n_splits}] : {foldValidationLossIfAvailable}");
        }
        var trainLossIfAvailable = IScore.Average(allFoldTrainLossIfAvailable);
        var validationLossIfAvailable = IScore.Average(allFoldValidationLossIfAvailable);
        var trainRankingMetricIfAvailable = IScore.Average(allFoldTrainRankingMetricIfAvailable);
        var validationRankingMetricIfAvailable = IScore.Average(allFoldValidationRankingMetricIfAvailable);
        return (null, null, train_XYDatasetPath_InTargetFormat, null, null, train_XYDatasetPath_InTargetFormat, trainLossIfAvailable, validationLossIfAvailable, trainRankingMetricIfAvailable, validationRankingMetricIfAvailable);
    }

    public override (DataFrame trainPredictions_InTargetFormat, IScore trainRankingScore_InTargetFormat,
        DataFrame trainPredictions_InModelFormat, IScore trainLoss_InModelFormat,
        DataFrame validationPredictions_InTargetFormat, IScore validationRankingScore_InTargetFormat,
        DataFrame validationPredictions_InModelFormat, IScore validationLoss_InModelFormat)
        ComputePredictionsAndRankingScore(DataSet trainDataset, DataSet validationDatasetMustBeNull, AbstractDatasetSample datasetSample, bool computeTrainMetrics)
    {
        Debug.Assert(validationDatasetMustBeNull == null);
        int n_splits = KFoldSample.n_splits;
        var foldedTrainAndValidationDataset = trainDataset.KFoldSplit(n_splits, KFoldSample.CountMustBeMultipleOf);
        var validationIntervalForKfold = trainDataset.IdToValidationKFold(n_splits, KFoldSample.CountMustBeMultipleOf);

        DataFrame trainPredictions_InModelFormat = null;
        DataFrame validationPredictions_InModelFormat = null;

        for (int fold = 0; fold < n_splits; ++fold)
        {
            var embeddedModel = _embeddedModels[fold];
            LogDebug($"Computing embedded model '{embeddedModel.ModelName}' predictions on fold[{fold}/{n_splits}]");
            var trainAndValidation = foldedTrainAndValidationDataset[fold];
            var fold_trainPredictions_InModelFormat = computeTrainMetrics
                ?embeddedModel.Predict(trainAndValidation.Training, false)
                : null;
            var fold_validationPredictions_InModelFormat = embeddedModel.Predict(trainAndValidation.Test, false);

            if (validationPredictions_InModelFormat == null)
            {
                validationPredictions_InModelFormat = fold_validationPredictions_InModelFormat.ResizeWithNewNumberOfRows(trainDataset.Count);
                trainPredictions_InModelFormat = computeTrainMetrics
                    ?validationPredictions_InModelFormat.Clone()
                    :null;
                validationPredictions_InModelFormat.FloatTensor?.SetValue(0);
                trainPredictions_InModelFormat?.FloatTensor?.SetValue(0);
            }
            int fold_validation_row = 0;
            for(int row=0;row < trainDataset.Count; row++)
            {
                if (validationIntervalForKfold[row] == fold)
                {
                    validationPredictions_InModelFormat.RowSlice(row, 1, true).Add(fold_validationPredictions_InModelFormat.RowSlice(fold_validation_row++, 1, true));
                }
            }
            if (computeTrainMetrics)
            {
                int fold_training_row = 0;
                for (int row = 0; row < trainDataset.Count; row++)
                {
                    if (validationIntervalForKfold[row] != fold)
                    {
                        // ReSharper disable once PossibleNullReferenceException
                        trainPredictions_InModelFormat.RowSlice(row, 1, true).Add(fold_trainPredictions_InModelFormat.RowSlice(fold_training_row++, 1, true));
                    }
                }
                if (fold_training_row+ fold_validation_row != trainDataset.Count)
                {
                    throw new ArgumentException($"{fold_training_row}+{fold_validation_row} != {trainDataset.Count}");
                }
            }
        }
        Debug.Assert(validationPredictions_InModelFormat != null);


        var y_true_InModelFormat = trainDataset.Y_InModelFormat().FloatCpuTensor();
        var y_true_InTargetFormat = datasetSample.Predictions_InModelFormat_2_Predictions_InTargetFormat(DataFrame.New(y_true_InModelFormat), ModelSample.GetObjective());

        if (n_splits >= 2)
        {
            trainPredictions_InModelFormat?.Mult(1f / (n_splits-1));
        }

        IScore trainLoss_InModelFormat = null;
        DataFrame trainPredictions_InTargetFormat = null;
        IScore trainRankingScore_InTargetFormat = null;
        if (computeTrainMetrics)
        {
            // ReSharper disable once PossibleNullReferenceException
            trainLoss_InModelFormat = ComputeLoss(y_true_InModelFormat, trainPredictions_InModelFormat.FloatCpuTensor());
            trainPredictions_InTargetFormat = datasetSample.Predictions_InModelFormat_2_Predictions_InTargetFormat(trainPredictions_InModelFormat, ModelSample.GetObjective());
            trainRankingScore_InTargetFormat = datasetSample.ComputeRankingEvaluationMetric(y_true_InTargetFormat, trainPredictions_InTargetFormat, ModelSample);
        }

        //validationPredictions_InModelFormat.Mult(1f / n_splits);
        // ReSharper disable once PossibleNullReferenceException
        var validationLoss_InModelFormat = ComputeLoss(y_true_InModelFormat, validationPredictions_InModelFormat.FloatCpuTensor());
        var validationPredictions_InTargetFormat = datasetSample.Predictions_InModelFormat_2_Predictions_InTargetFormat(validationPredictions_InModelFormat, ModelSample.GetObjective());
        var validationRankingScore_InTargetFormat = datasetSample.ComputeRankingEvaluationMetric(y_true_InTargetFormat, validationPredictions_InTargetFormat, ModelSample);

        return 
            (trainPredictions_InTargetFormat, trainRankingScore_InTargetFormat,
            trainPredictions_InModelFormat, trainLoss_InModelFormat,
            validationPredictions_InTargetFormat, validationRankingScore_InTargetFormat,
            validationPredictions_InModelFormat, validationLoss_InModelFormat);
    }

    public override (DataFrame, string) PredictWithPath(DataSet dataset, bool removeAllTemporaryFilesAtEnd)
    {
        CpuTensor<float> res = null;
        Debug.Assert(KFoldSample.n_splits == _embeddedModels.Count);
        //each model weight
        var weight = 1.0f / KFoldSample.n_splits;
        var columnNames = new List<string>();
        foreach (var m in _embeddedModels)
        {
            var modelPrediction = m.Predict(dataset, removeAllTemporaryFilesAtEnd);
            columnNames = modelPrediction.Columns.ToList();
            if (res == null)
            {
                res = modelPrediction.FloatCpuTensor();
                res.Update_Multiplying_By_Alpha(weight);
            }
            else
            {
                res.AddTensor(weight, modelPrediction.FloatCpuTensor(), 1.0f);
            }
        }
        return (DataFrame.New(res, columnNames), "");
    }
    public override List<string> Save(string workingDirectory, string modelName)
    {
        var res = ModelSample.Save(workingDirectory, modelName);
        foreach (var embeddedModel in _embeddedModels)
        {
            var newFiles = embeddedModel.Save(workingDirectory, embeddedModel.ModelName);
            res.AddRange(newFiles);    
        }
        return res;
    }
    public override List<string> AllFiles()
    {
        List<string> res = new();
        res.AddRange(ModelSample.SampleFiles(WorkingDirectory, ModelName));
        foreach (var m in _embeddedModels)
        {
            res.AddRange(m.AllFiles());
        }
        return res;
    }
    private KFoldSample KFoldSample => (KFoldSample)ModelSample;
    //TODO add tests
}