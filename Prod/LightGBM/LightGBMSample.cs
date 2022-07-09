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
    public int num_threads = DEFAULT_VALUE;

    //device for the tree learning, you can use GPU to achieve the faster learning
    //Note: it is recommended to use the smaller max_bin(e.g. 63) to get the better speed up
    //Note: for the faster speed, GPU uses 32-bit float point to sum up by default, so this may affect the accuracy for some tasks.You can set gpu_use_dp= true to enable 64-bit float point, but it will slow down the training
    //Note: refer to Installation Guide to build LightGBM with GPU support
    //aliases: device
    public enum device_type_enum {cpu,gpu, DEFAULT_VALUE = AbstractSample.DEFAULT_VALUE}
    public device_type_enum device_type = device_type_enum.DEFAULT_VALUE;


    //this seed is used to generate other seeds, e.g. data_random_seed, feature_fraction_seed, etc.
    //by default, this seed is unused in favor of default values of other seeds
    //this seed has lower priority in comparison with other seeds, which means that it will be overridden, if you set other seeds explicitly
    //aliases: random_seed, seed 
    public int random_state = DEFAULT_VALUE;

    //used only with cpu device type
    //setting this to true should ensure the stable results when using the same data and the same parameters (and different num_threads)
    //when you use the different seeds, different LightGBM versions, the binaries compiled by different compilers, or in different systems, the results are expected to be different
    //you can raise issues in LightGBM GitHub repo when you meet the unstable results
    //Note: setting this to true may slow down the training
    //Note: to avoid potential instability due to numerical issues, please set force_col_wise=true or force_row_wise=true when setting deterministic=true
    public bool deterministic = false;

    //public enum tree_learner_enum { serial, feature, data, voting }
    ////serial:   single machine tree learner
    ////feature:  feature parallel tree learner, aliases: feature_parallel
    ////data:     data parallel tree learner, aliases: data_parallel
    ////voting:   voting parallel tree learner, aliases: voting_parallel
    ////refer to Distributed Learning Guide to get more details
    ////aliases: tree, tree_type, tree_learner_type
    //public tree_learner_enum tree_learner = tree_learner_enum.serial;

    #endregion

    #region Learning Control Parameters

    #region CLI specific
    //filename of input model
    //for prediction task, this model will be applied to prediction data
    //for train task, training will be continued from this model
    //aliases: model_input, model_in
    public string input_model;

    //filename of output model in training
    //aliases: model_output, model_out
    public string output_model;

    //the feature importance type in the saved model file
    //0: count-based feature importance (numbers of splits are counted);
    //1: gain-based feature importance (values of gain are counted)
    public int saved_feature_importance_type = DEFAULT_VALUE;

    //frequency of saving model file snapshot
    //set this to positive value to enable this function. For example, the model file will be snapshotted at each iteration if snapshot_freq=1
    //aliases: save_period
    public int snapshot_freq = DEFAULT_VALUE;
    #endregion

    //used only with cpu device type
    //set this to true to force col-wise histogram building
    //enabling this is recommended when:
    //  the number of columns is large, or the total number of bins is large
    //  num_threads is large, e.g. > 20
    //  you want to reduce memory cost
    //Note: when both force_col_wise and force_row_wise are false, LightGBM will firstly try them both,
    //and then use the faster one.
    //To remove the overhead of testing set the faster one to true manually
    //Note: this parameter cannot be used at the same time with force_row_wise, choose only one of them
    public bool force_col_wise = false;

    //used only with cpu device type
    //set this to true to force row-wise histogram building
    //enabling this is recommended when:
    //  the number of data points is large, and the total number of bins is relatively small
    //  num_threads is relatively small, e.g. <= 16
    //  you want to use small bagging_fraction or goss boosting to speed up
    //Note: setting this to true will double the memory cost for Dataset object.
    //If you have not enough memory, you can try setting force_col_wise=true
    //Note: when both force_col_wise and force_row_wise are false, LightGBM will firstly try them both,
    //and then use the faster one.
    //To remove the overhead of testing set the faster one to true manually
    //Note: this parameter cannot be used at the same time with force_col_wise, choose only one of them
    public bool force_row_wise = false;

    //max cache size in MB for historical histogram
    //< 0 means no limit
    public double histogram_pool_size = DEFAULT_VALUE;

    //limit the max depth for tree model.
    //This is used to deal with over-fitting when #data is small.
    //Tree still grows leaf-wise
    //<= 0 means no limit
    public int max_depth = DEFAULT_VALUE;

    //minimal number of data in one leaf. Can be used to deal with over-fitting
    //aliases: min_data_per_leaf, min_data, min_child_samples, min_samples_leaf, constraints: min_data_in_leaf >= 0
    //Note: this is an approximation based on the Hessian, so occasionally you may observe splits
    //      which produce leaf nodes that have less than this many observations
    // default: 20
    public int min_data_in_leaf = DEFAULT_VALUE;

    // minimal sum hessian in one leaf.Like min_data_in_leaf, it can be used to deal with over-fitting
    // aliases:     min_sum_hessian_per_leaf, min_sum_hessian, min_hessian, min_child_weight,
    // constraints: min_sum_hessian_in_leaf >= 0.0
    public double min_sum_hessian_in_leaf = DEFAULT_VALUE;

    //like feature_fraction, but this will randomly select part of data without resampling
    //can be used to speed up training
    //can be used to deal with over-fitting
    //Note: to enable bagging, bagging_freq should be set to a non zero value as well
    // aliases: sub_row, subsample, bagging
    // constraints: 0.0 < bagging_fraction <= 1.0
    public double bagging_fraction = DEFAULT_VALUE;

    // aliases: pos_sub_row, pos_subsample, pos_bagging
    // constraints: 0.0 < pos_bagging_fraction <= 1.0
    //used only in binary application
    //used for imbalanced binary classification problem, will randomly sample #pos_samples * pos_bagging_fraction positive samples in bagging
    //should be used together with neg_bagging_fraction
    //set this to 1.0 to disable
    //Note: to enable this, you need to set bagging_freq and neg_bagging_fraction as well
    //Note: if both pos_bagging_fraction and neg_bagging_fraction are set to 1.0, balanced bagging is disabled
    //Note: if balanced bagging is enabled, bagging_fraction will be ignored
    public double pos_bagging_fraction = DEFAULT_VALUE;

    //used only in binary application
    //used for imbalanced binary classification problem, will randomly sample #neg_samples * neg_bagging_fraction negative samples in bagging
    //should be used together with pos_bagging_fraction
    //set this to 1.0 to disable
    //Note: to enable this, you need to set bagging_freq and pos_bagging_fraction as well
    //Note: if both pos_bagging_fraction and neg_bagging_fraction are set to 1.0, balanced bagging is disabled
    //Note: if balanced bagging is enabled, bagging_fraction will be ignored
    // aliases: neg_sub_row, neg_subsample, neg_bagging, constraints: 0.0 < neg_bagging_fraction <= 1.0
    public double neg_bagging_fraction = DEFAULT_VALUE;

    //frequency for bagging
    //0 means disable bagging; k means perform bagging at every k iteration.
    //Every k-th iteration, LightGBM will randomly select bagging_fraction * 100 % of the data to use for the next k iterations
    //Note: to enable bagging, bagging_fraction should be set to value smaller than 1.0 as well
    //aliases: subsample_freq
    public int bagging_freq = DEFAULT_VALUE;

    //random seed for bagging
    //aliases: bagging_fraction_seed
    public int bagging_seed = DEFAULT_VALUE;

    //LightGBM will randomly select a subset of features on each iteration (tree) if feature_fraction is smaller than 1.0.
    //For example, if you set it to 0.8, LightGBM will select 80% of features before training each tree
    //can be used to speed up training
    //can be used to deal with over-fitting
    //aliases:      sub_feature, feature_fraction 
    //constraints:  0.0 < feature_fraction <= 1.0
    public double colsample_bytree= DEFAULT_VALUE;

    //LightGBM will randomly select a subset of features on each tree node if feature_fraction_bynode is smaller than 1.0.
    //For example, if you set it to 0.8, LightGBM will select 80% of features at each tree node
    //can be used to deal with over-fitting
    //Note: unlike feature_fraction, this cannot speed up training
    //Note: if both feature_fraction and feature_fraction_bynode are smaller than 1.0, the final fraction of each node is feature_fraction * feature_fraction_bynode
    // aliases: sub_feature_bynode, feature_fraction_bynode  
    // constraints: 0.0 < feature_fraction_bynode <= 1.0
    public double colsample_bynode = DEFAULT_VALUE;

    //random seed for feature_fraction
    public int feature_fraction_seed = DEFAULT_VALUE;

    //use extremely randomized trees
    //if set to true, when evaluating node splits LightGBM will check only one randomly-chosen threshold for each feature
    //can be used to speed up training
    //can be used to deal with over-fitting
    // aliases: extra_tree
    public bool extra_trees = false;

    //random seed for selecting thresholds when extra_trees is true
    public int extra_seed = DEFAULT_VALUE;

    //will stop training if one metric of one validation data doesn’t improve in last early_stopping_round rounds
    //<= 0 means disable
    //can be used to speed up training
    // aliases: early_stopping_rounds, early_stopping, n_iter_no_change
    public int early_stopping_round = DEFAULT_VALUE;

    //LightGBM allows you to provide multiple evaluation metrics.
    //Set this to true, if you want to use only the first metric for early stopping
    public bool first_metric_only = false;

    //used to limit the max output of tree leaves
    //<= 0 means no constraint
    //the final max output of leaves is learning_rate * max_delta_step
    //aliases: max_tree_output, max_leaf_output
    public double max_delta_step = DEFAULT_VALUE;

    // L1 regularization
    // aliases: reg_alpha, l1_regularization
    // constraints: lambda_l1 >= 0.0
    public double lambda_l1 = DEFAULT_VALUE;

    //L2 regularization
    // aliases: reg_lambda, lambda, l2_regularization
    // constraints: lambda_l2 >= 0.0
    public double lambda_l2 = DEFAULT_VALUE;


    //linear tree regularization, corresponds to the parameter lambda in Eq. 3 of Gradient Boosting with Piece-Wise Linear Regression Trees
    //constraints: linear_lambda >= 0.0
    public double linear_lambda = DEFAULT_VALUE;

    //the minimal gain to perform split
    //can be used to speed up training
    // aliases: min_split_gain
    // constraints: min_gain_to_split >= 0.0
    public double min_gain_to_split = DEFAULT_VALUE;

    #region used only in dart
    // dropout rate: a fraction of previous trees to drop during the dropout
    // used only in dart
    // aliases: rate_drop
    // constraints: 0.0 <= drop_rate <= 1.0
    // default: 0.1
    public double drop_rate = DEFAULT_VALUE;

    //max number of dropped trees during one boosting iteration
    //<=0 means no limit
    // default: 50
    public int max_drop = DEFAULT_VALUE;

    //probability of skipping the dropout procedure during a boosting iteration
    //constraints: 0.0 <= skip_drop <= 1.0 
    // default: 0.50
    public double skip_drop = DEFAULT_VALUE;

    //set this to true, if you want to use xgboost dart mode
    public bool xgboost_dart_mode = false;

    //set this to true, if you want to use uniform drop
    public bool uniform_drop = false;

    //random seed to choose dropping models
    public int drop_seed = DEFAULT_VALUE;
    #endregion


    //used only in goss
    //the retain ratio of large gradient data
    //constraints: 0.0 <= top_rate <= 1.0
    public double top_rate = DEFAULT_VALUE;

    // used only in goss
    // the retain ratio of small gradient data
    // constraints: 0.0 <= other_rate <= 1.0
    public double other_rate = DEFAULT_VALUE;

    // minimal number of data per categorical group
    // constraints: min_data_per_group > 0
    public int min_data_per_group = DEFAULT_VALUE;

    //used for the categorical features
    //limit number of split points considered for categorical features. See the documentation on how LightGBM finds optimal splits for categorical features for more details
    //can be used to speed up training
    // constraints: max_cat_threshold > 0
    public int max_cat_threshold = DEFAULT_VALUE;

    // used for the categorical features
    // L2 regularization in categorical split
    // constraints: cat_l2 >= 0.0
    public double cat_l2 = DEFAULT_VALUE;

    // used for the categorical features
    // this can reduce the effect of noises in categorical features, especially for categories with few data
    // constraints: cat_smooth >= 0.0
    public double cat_smooth = DEFAULT_VALUE;

    //when number of categories of one feature smaller than or equal to max_cat_to_onehot, 
    //one-vs-other split algorithm will be used
    //constraints: max_cat_to_onehot > 0
    public int max_cat_to_onehot = DEFAULT_VALUE;

    //// used only in voting tree learner, refer to Voting parallel
    //// set this to larger value for more accurate result, but it will slow down the training speed
    //// aliases: topk
    //// constraints: top_k > 0
    //public int top_k = DEFAULT_VALUE;

    ////used for constraints of monotonic features
    ////1 means increasing, -1 means decreasing, 0 means non-constraint
    ////you need to specify all features in order. For example, mc=-1,0,1 means decreasing for 1st feature, non-constraint for 2nd feature and increasing for the 3rd feature
    ////aliases: mc, monotone_constraint, monotonic_cst
    //public int[] monotone_constraints = null;

    //used only if monotone_constraints is set
    //monotone constraints method
    //basic, the most basic monotone constraints method. It does not slow the library at all, but over-constrains the predictions
    //intermediate, a more advanced method, which may slow the library very slightly. However, this method is much less constraining than the basic method and should significantly improve the results
    //advanced, an even more advanced method, which may slow the library. However, this method is even less constraining than the intermediate method and should again significantly improve the results
    // options: basic, intermediate, advanced, aliases: monotone_constraining_method, mc_method
    //enum monotone_constraints_method = basic; 

    ////used only if monotone_constraints is set
    ////monotone penalty: a penalization parameter X forbids any monotone splits on the first X (rounded down) level(s) of the tree. The penalty applied to monotone splits on a given depth is a continuous, increasing function the penalization parameter
    //// if 0.0 (the default), no penalization is applied
    //// aliases: monotone_splits_penalty, ms_penalty, mc_penalty, 
    //// constraints: monotone_penalty >= 0.0
    //public double monotone_penalty = DEFAULT_VALUE;

    ////used to control feature’s split gain, will use gain[i] = max(0, feature_contri[i]) * gain[i] to replace the split gain of i-th feature
    ////you need to specify all features in order
    ////aliases: feature_contrib, fc, fp, feature_penalty
    //public double[] feature_contri = null;

    ////path to a .json file that specifies splits to force at the top of every decision tree before best-first learning commences
    ////.json file can be arbitrarily nested, and each split contains feature, threshold fields, as well as left and right fields representing subsplits
    ////categorical splits are forced in a one-hot fashion, with left representing the split containing the feature value and right representing other values
    ////Note: the forced split logic will be ignored, if the split makes gain worse
    ////see this file as an example
    //// aliases: fs, forced_splits_filename, forced_splits_file, forced_splits
    //public string forcedsplits_filename;

    ////decay rate of refit task, will use leaf_output = refit_decay_rate * old_leaf_output + (1.0 - refit_decay_rate) * new_leaf_output to refit trees
    ////used only in refit task in CLI version or as argument in refit function in language-specific package
    ////constraints: 0.0 <= refit_decay_rate <= 1.0
    //public double refit_decay_rate = 0.9;

    #region cost-effective gradient boosting
    //cost-effective gradient boosting mul