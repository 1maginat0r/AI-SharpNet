﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using SharpNet.Data;
using SharpNet.DataAugmentation;
using SharpNet.Datasets;
using SharpNet.GPU;
using SharpNet.Hyperparameters;
using SharpNet.Models;
using SharpNet.Optimizers;
using static SharpNet.GPU.GPUWrapper;
// ReSharper disable UnusedMember.Global
// ReSharper disable AutoPropertyCanBeMadeGetOnly.Global
// ReSharper disable MemberCanBePrivate.Global
// ReSharper disable ConvertToConstant.Global
// ReSharper disable FieldCanBeMadeReadOnly.Global

namespace SharpNet.Networks
{
    public class NetworkSample : AbstractModelSample
    {
        #region constructors
        // ReSharper disable once EmptyConstructor
        public NetworkSample()
        {
        }
        static NetworkSample()
        {
            Utils.ConfigureGlobalLog4netProperties(DefaultWorkingDirectory, "SharpNet");
        }
        #endregion

        
        /// <summary>
        /// The convolution algo to be used
        /// </summary>
        public ConvolutionAlgoPreference ConvolutionAlgoPreference = ConvolutionAlgoPreference.FASTEST_DETERMINIST;

        public double AdamW_L2Regularization = 0.01;
        public double Adam_beta1 = 0.9;
        public double Adam_beta2 = 0.999;
        public double Adam_epsilon = 1e-8;
        public double SGD_momentum = 0.9;
        public double lambdaL2Regularization;
        public bool SGD_usenesterov = true;
        public bool ShuffleDatasetBeforeEachEpoch = true;
        //when ShuffleDatasetBeforeEachEpoch is true, consider that the dataset is built from block of 'ShuffleDatasetBeforeEachEpochBlockSize' element (that must be kept in same order)
        public int ShuffleDatasetBeforeEachEpochBlockSize = 1;
        public Optimizer.OptimizationEnum OptimizerType = Optimizer.OptimizationEnum.VanillaSGD;


        #region debugging options

        /// <summary>
        /// when set to true, will log all forward and backward propagation
        /// </summary>
        public bool LogNetworkPropagation { get; set; } = false;
        #endregion


        #region Learning Rate Hyperparameters

        public double InitialLearningRate;

        /// <summary>
        /// minimum value for the learning rate
        /// </summary>
        public double MinimumLearningRate = 1e-9;

        #region learning rate scheduler fields
        public enum LearningRateSchedulerEnum { Cifar10ResNet, Cifar10DenseNet, OneCycle, CyclicCosineAnnealing, Cifar10WideResNet, Linear, Constant }

        public LearningRateSchedulerEnum LearningRateSchedulerType = LearningRateSchedulerEnum.Constant;
        public int CyclicCosineAnnealing_nbEpochsInFirstRun = 10;

        public int CyclicCosineAnnealing_nbEpochInNextRunMultiplier = 2;
        /// <summary>
        /// for one cycle policy: by how much we have to divide the max learning rate to reach the min learning rate
        /// </summary>
        public int OneCycle_DividerForMinLearningRate = 10;
        public double OneCycle_PercentInAnnealing = 0.2;


        public int Linear_DividerForMinLearningRate = 100;

        /// <summary>
        /// the minimum value for the learning rate (default value:  1e-6)
        /// </summary>
        public double CyclicCosineAnnealing_MinLearningRate = 1e-6;

        public bool DisableReduceLROnPlateau;
        public bool DivideBy10OnPlateau = true; // 'true' : validated on 19-apr-2019: +20 bps
        public bool LinearLearningRate;
        #endregion
        #endregion


        public double MinimumRankingScoreToSaveModel = double.NaN;


        /// <summary>
        /// if set:
        ///  we'll only save the model after the epoch that gives the better results:
        ///     better ranking score in validation dataset (if a validation dataset is provided)
        ///     better ranking score in training dataset (if no validation dataset is provided)
        /// </summary>
        public bool use_best_model = true;

        public override IScore GetMinimumRankingScoreToSaveModel()
        {
            if (double.IsNaN(MinimumRankingScoreToSaveModel))
            {
                return null;
            }
            return new Score((float)MinimumRankingScoreToSaveModel, GetRankingEvaluationMetric());

        }


        /// <summary>
        /// all resources (CPU or GPU) available for the current network
        /// values superior or equal to 0 means GPU resources (device)
        /// values strictly less then 0 mean CPU resources (host)
        /// 
        /// if ResourceIds.Count == 1
        ///     if masterNetworkIfAny == null:
        ///         all computation will be done in a single network (using resource ResourceIds[0])
        ///     else:
        ///         we are in a slave network (using resource ResourceIds[0]) doing part of the parallel computation
        ///         the master network is 'masterNetworkIfAny'.
        /// else: (ResourceIds.Count >= 2)
        ///     we are the master network (using resource ResourceIds[0]) doing part of the parallel computation
        ///     slaves network will use resourceId ResourceIds[1:]
        /// 
        /// for each resourceId in this list:
        ///     if resourceId strictly less then 0:
        ///         use CPU resource (no GPU usage)
        ///     else:
        ///         run the network on the GPU with device Id = resourceId
        /// </summary>
        public List<int> ResourceIds = new() { 0 };
        public void SetResourceId(int resourceId)
        {
            if (resourceId == int.MaxValue)
            {
                //use multi GPU
                ResourceIds = Enumerable.Range(0, GetDeviceCount()).ToList();
            }
            else
            {
                //single resource
                ResourceIds = new List<int> { resourceId };
            }
        }

        public int NumEpochs;
        public int BatchSize;
        public override EvaluationMetricEnum GetLoss() => LossFunction;
        public override EvaluationMetricEnum GetRankingEvaluationMetric()
        {
            var metrics = GetAllEvaluationMetrics();
            return metrics.Count != 0 ? metrics[0] : EvaluationMetricEnum.DEFAULT_VALUE;
        }

        protected override List<EvaluationMetricEnum> GetAllEvaluationMetrics()
        {
            return EvaluationMetrics;
        }
        public EvaluationMetricEnum LossFunction = EvaluationMetricEnum.DEFAULT_VALUE;


        public float MseOfLog_Epsilon = 0.0008f;
        public float Huber_Delta = 1.0f;

        /// <summary>
        /// the percent of elements in the True (y==1) class
        /// the goal is to recalibrate the loss if one class (y==1 or y==0) is over-represented
        /// </summary>
        public float BCEWithFocalLoss_PercentageInTrueClass = 0.5f;
        public float BCEWithFocalLoss_Gamma = 0;

        public List<EvaluationMetricEnum> EvaluationMetrics = new();

        public CompatibilityModeEnum CompatibilityMode = CompatibilityModeEnum.SharpNet;
        public string DataSetName;
        /// <summary>
        /// if true
        ///     we'll always use the full test data set to compute the loss and accuracy of this test data set
        /// else
        ///     we'll use the full test data set for some specific epochs (the first, the last, etc.)
        ///     and a small part of this test data set for other epochs:
        ///         DataSet.PercentageToUseForLossAndAccuracyFastEstimate
        /// </summary>
        public bool AlwaysUseFullTestDataSetF