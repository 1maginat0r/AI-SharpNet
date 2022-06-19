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
        var c