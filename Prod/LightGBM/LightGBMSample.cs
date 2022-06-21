// ReSharper disable UnusedMember.Global
// ReSharper disable InconsistentNaming
// ReSharper disable IdentifierTypo
// ReSharper disable CommentTypo

using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using SharpNet.Datasets;
using SharpNet.HPO;
using SharpNet.Hyperparameters;
using SharpNet.Models;
using System.Linq;

namespace SharpNet.LightGBM;

[SuppressMessage("ReSharper", "MemberCanBePrivate.Global")]
public class LightGBMSample : AbstractModelSample
{
    public LightGBMSample() :base(_categoricalHyperparameters)
    {
    }
    public override EvaluationMetricEnum GetLoss()
    {
        switch (objective)
        {
            case objective_enum.DEFAULT_VALUE:
                return EvaluationMetricEnum.DEFAULT_VALUE;
            case objective_enum.regression:
                return EvaluationMetricEnum.Rmse;
            case objective_enum.regression_l1:
                return EvaluationMetricEnum.Mae;
            case objective_enum.binary:
                return EvaluationMetricEnum.BinaryCrossentropy;
            case objective_enum.multiclass:
            case objective_enum.cross_entropy: //TODO to check
                return EvaluationMetricEnum.CategoricalCrossentropy; 
            default:
                throw new NotImplementedException($"can't manage metric {objective}");
        }
    }

    public override EvaluationMetricEnum GetRankingEvaluationMetric()
    {
        return GetAllEvaluationMetrics()[0];
    }
    public static EvaluationMetricEnum ToEvaluationMetricEnum(string metric)
    {
        if (string.IsNullOrEmpty(metric))
        {
            return EvaluationMetricEnum.DEFAULT_VALUE;
        }
        switch (metric.ToLowerInvariant())
        {
            case "rmse":
            case "regression":
                return EvaluationMetricEnum.Rmse;
            case "regression_l1":
            case "mae":
                return EvaluationMetricEnum.Mae;
            case "huber":
                return EvaluationMetricEnum.Huber;
            case "auc":
                return EvaluationMetricEnum.AUC;
            case "average_precision":
                return EvaluationMetricEnum.AveragePrecisionScore;
            case "binary":
                return EvaluationMetricEnum.BinaryCrossentropy;
            case "multiclass":
            case "cross_entropy":
                return EvaluationMetricEnum.CategoricalCrossentropy;
            case "accuracy":
                return EvaluationMetricEnum.Accuracy;
            default:
                throw new NotImplementedException($"can't manage metric {metric}");
        }
    }

    public static string ToStringMetric(EvaluationMetricEnum metric)
    {
        switch (metric)
        {
            case EvaluationMetricEnum.DEFAULT_VALUE: return "";
            case EvaluationMetricEnum.Rmse: return "rmse";
            case EvaluationMetricEnum.Mae:  return "mae";
            case EvaluationMetricEnum.Huber : return "huber";
            case EvaluationMetricEnum.AUC: return  "auc";
            case EvaluationMetricEnum.AveragePrecisionScore: return "average_precision";
            case EvaluationMetricEnum.BinaryCrossentropy: return  "binary" ;
            case EvaluationMetricEnum.CategoricalCrossentropy: return "cross_entropy";
            case EvaluationMetricEnum.Accuracy: return  "accuracy";
            default:
                throw new NotImplementedException($"can't manage metric {metric}");
        }
    }

    protected override List<EvaluationMetricEnum> GetAllEvaluationMetrics()
    {
        if (string.IsNullOrEmpty(metric))
        {
            return new List<EvaluationMetricEnum> { EvaluationMetricEnum.DEFAULT_VALUE };
        }
        return metric.Split(',').Select(ToEvaluationMetricEnum).ToList();
    }


    public override bool FixErrors()
    {
        if (boosting == boosting_enum.rf)
        {
            if (bagging_freq <= 0 || bagging_fraction >= 1.0f || bagging_fraction <= 0.0f)
            {
                return false;
            }
        }
        if (boosting != boosting_enum.dart)
        {
            drop_rate = DEFAULT_VALUE;
            max_drop = DEFAULT_VALUE;
            skip_drop = DEFAULT_VALUE;
            xgboost_dart_mode = false;
            uniform_drop = false;
            drop_seed = DEFAULT_VALUE;
        }

        if (path_smooth > 0 && min_data_in_leaf<2)
        {
            min_data_in_leaf = 2;
        }

        //if (objective == objective_enum.DEFAULT_VALUE)
        //{
        //    throw new ArgumentException("objective must always be set");
        //}

        if (IsMultiClassClassificationProblem())
        {
            if (num_class < 2)
            {
                throw new ArgumentException($"{nameof(num_class)} must be set for multi class problem (was:{num_class})");
            }
        }
        else
        {
            if (num_class != DEFAULT_VALUE)
            {
                throw new ArgumentException($"{nameof(num_class)} should be set only for multi class problem (was: {num_class})");
            }
        }

        if (bagging_freq <= 0)
        {
            //bagging is disabled
            //bagging_fraction must be equal to 1.0 (100%)
            if (bagging_fraction < 1)
            {
                return false;
            }
        }
        else
        {
            //bagging is enabled
            //bagging_fraction must be stricly less then 1.0 (100%)
            if (bagging_fraction >= 1)
            {
                return false;
            }
        }
        return true;
    }

    public override void Use_All_Available_Cores()
    {
        num_threads = Utils.CoreCount;
    }
    public void UpdateForDataset(DataSet dataset)
    {
        var categoricalFeatures = dataset.ColumnNames.Where(dataset.IsCategoricalColumn).ToList();
        if (!string.IsNullOrEmpty(dataset.IdColumn))
        {
            categoricalFeatures.Remove(dataset.IdColumn);
        }
        //if (dataset.GetDatasetSample() != null)
        //{
        //    foreach (var column in dataset.GetDatasetSample().TargetLabels)
        //    {
        //        categoricalFeatures.Remove(column);
        //    }
        //}
        categorical_feature = (categoricalFeatures.Count >= 1) ? ("name:" + string.Join(',', categoricalFeatures)) : "";
    }

    public (IScore trainLossIfAvailable, IScore validationLossIfAvailable, IScore trainRankingMetricIfAvailable, IScore validationRankingMetricIfAvailable) ExtractScores(IEnumerable<string> linesFromLog)
    {
        //the first element will always be the loss, the following will be the ranking metrics
        var lossAndMetrics = new List<string> { ToStringMetric(GetLoss()) };
        var allMetrics = (metric ?? "").Split(',');
        if (allMetrics.Length >= 1 && allMetrics[0].Length >= 1 && !lossAndMetrics.Contains(allMetrics[0]))
        {
            lossAndMetrics.Add(allMetrics[0]);
        }
        List<string> tokenAndMandatoryTokenAfterToken = new();
        foreach (var m in lossAndMetrics)
        {
            tokenAndMandatoryTokenAfterToken.Add("training");
            tokenAndMandatoryTokenAfterToken.Add(m);
            tokenAndMandatoryTokenAfterToken.Add("valid_1");
            tokenAndMandatoryTokenAfterToken.Add(m);
        }
        var extractedScoresFromLogs = Utils.ExtractValuesFromOutputLog(linesFromLog, 2, tokenAndMandatoryTokenAfterToken.ToArray());

        var trainLossValue = extractedScoresFromLogs[0];
        var trainLossIfAvailable = double.IsNaN(trainLossValue) ? null : new Score((float)trainLossValue, GetLoss());
        var validationLossValue = extractedScoresFromLogs[1];
        var validationLossIfAvailable = double.IsNaN(validationLossValue) ? null : new Score((float)validationLossValue, GetLoss());

        var trainMetricValue = (extractedScoresFromLogs.Length >= 3) ? extractedScoresFromLogs[2] : double.NaN;
        Score trainRankingMetricIfAvailable = null;
        if (!double.IsNaN(trainMetricValue) && lossAndMetrics.Count>=2)
        {
            trainRankingMetricIfAvailable = new Score((float)trainMetricValue, ToEvaluationMetricEnum(lossAndMetrics[1]));
        }
        var validationMetricValue = (extractedScoresFromLogs.Length >= 4) ? extractedScoresFromLogs[3] : double.NaN;
        Score validationRankingMetricIfAvailable = null;
        if (!double.IsNaN(validationMetricValue) && lossAndMetrics.Count >= 2)
        {
            validationRankingMetricIfAvailable = new Score((float)validationMetricValue, ToEvaluationMetricEnum(lossAndMetrics[1]));
        }
        return (trainLossIfAvailable, validationLossIfAvailable, trainRankingMetricIfAvailable, validationRankingMetricIfAvailable);
    }

    #region Core Parameters

    #region CLI specific

    /// <summary>
    /// path of config file
    /// aliases: config_file
    /// </summary>
    public string config;

    /// <summary>
    ///  train: for training, aliases: training
    ///  predict: for prediction, aliases: prediction, test
    ///  convert_model: for converting model file into if-else format, see more information in Convert Parameters
    ///  refit: for refitting existing models with new data, aliases: refit_tree
    ///         save_binary, load train(and validation) data then save dataset to binary file.Typical usage: save_binary first, then run multiple train tasks in parallel using the saved binary file
    /// aliases: task_type
    /// </summary>
    public enum task_enum { train, predict, convert_model, refit, DEFAULT_VALUE = AbstractSample.DEFAULT_VALUE } ;
    // ReSharper disable once MemberCanBePrivate.Global
    public task_enum task = task_enum.DEFAULT_VALUE;

    //path of training data, LightGBM will train from this data
    //aliases: train, train_data, train_data_file, data_filename
    public string data;

    //path(s) of validation/test data, LightGBM will output metrics for these data
    //support multiple validation data, separated by ,
    //aliases: test, valid_data, valid_data_file, test_data, test_data_file, valid_filenames
    public string valid;
    #endregion


   

    public enum objective_enum { regression, regression_l1, huber, fair, poisson, quantile, mape, gamma, tweedie, binary, multiclass, multiclassova, cross_entropy, cross_entropy_lambda, lambdarank, rank_xendcg, DEFAULT_VALUE = AbstractSample.DEFAULT_VALUE } 
    //aliases: objective_type, app, application, loss
    // ReSharper disable once MemberCanBePrivate.Global
    public objective_enum objective = objective_enum.DEFAULT_VALUE;

    /// <summary>
    /// true if we face a classification problem
    /// </summary>
    public bool IsClassification => 
        objective == objective_enum.binary
        || objective == objective_enum.multiclass
        || objective == objective_enum.multiclassova;

    public enum boosting_enum { gbdt, rf, dart, goss, DEFAULT_VALUE = AbstractSample.DEFAULT_VALUE } 
    //gbdt:  traditional Gradient Boosting Decision Tree, aliases: gbrt
    //rf:    Random Forest, aliases: random_forest
    //dart:  Dropouts meet Multiple Additive Regression Trees
    //goss:  Gradient-based One-Side Sampling
    //Note: internally, LightGBM uses gbdt mode for the first 1 / learning_rate iterations
    public boosting_enum boosting = boosting_enum.DEFAULT_VALUE;

    //number of boosting iterations
    //Note: internally, LightGBM constructs num_class* num_boost_round trees for multi-class classification problems
    //aliases: num_iteration, n_iter, num_tree, num_trees, num_round, num_rounds, nrounds, num_boost_round, n_estimators, max_iter
    //constraints: num_iterations >= 0
    public int num_iterations = DEFAULT_VALUE;

    //shrinkage rate
    //in dart, it also affects on normalization weights of dropped trees
    //aliases: shrinkage_rate, eta, 
    //constraints: learning_rate > 0.0
    public double learning_rate = DEFAULT_VALUE;

    //max number of leaves in one tree
    //aliases: num_leaf, max_leaves, max_leaf, max_leaf_nodes
    //constraints: 1 < num_leaves <= 131072
    //default: 31
    public int num_leaves = DEFAULT_VALUE;

    //used only in train, prediction and refit tasks or in correspondent functions of language-specific packages
    //number of threads for LightGBM
    //0 means default number of threads in OpenMP
    //for the best speed, set this to the number of real CPU cores, not the number of threads (most CPUs use hyper-threading to generate 2 threads per CPU core)
    //do not set it too large if your dataset is small (for instance, do not use 64 threads for a dataset with 10,000 rows)
    //be aware a task manager or any similar CPU monitoring tool might report that cores not being fully utilized.
    //This is normal for distributed learning, do not use all CPU cores because this will cause poor performance for the network communication
    //    Note: please don’t change this during training, especially when running multiple jobs simultaneously by external packages, otherwise it may cause undesirable errors
    //aliases: num_thread, nthread, nthreads, n_jobs
    public int num_threads = DEFAULT