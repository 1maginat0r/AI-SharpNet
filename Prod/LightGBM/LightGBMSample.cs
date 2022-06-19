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
      