using System;
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
        public bool AlwaysUseFullTestDataSetForLossAndAccuracy = true;
      
        /// <summary>
        /// true if we want to display statistics about the weights tensors.
        /// Used only for debugging 
        /// </summary>
        public bool DisplayTensorContentStats = false;

        public bool SaveNetworkStatsAfterEachEpoch = false;
        /// <summary>
        /// Interval in minutes for saving the network
        /// If less than 0
        ///     => this option will be disabled
        /// If == 0
        ///     => the network will be saved after each iteration
        /// </summary>
        public int AutoSaveIntervalInMinutes = 3*60;
        /// <summary>
        /// number of consecutive epochs with a degradation of the validation loss to
        /// stop training the network.
        /// A value less or equal then 0 means no early stopping
        /// </summary>
        public int EarlyStoppingRounds = 0;
        /// <summary>
        /// name of the the first layer for which we want ot freeze the weights
        /// if 'FirstLayerNameToFreeze' is valid and 'LastLayerNameToFreeze' is empty
        ///     we'll freeze all layers in the network from 'FirstLayerNameToFreeze' to the last network layer
        /// if both 'FirstLayerNameToFreeze' and 'LastLayerNameToFreeze' are valid
        ///     we'll freeze all layers in the network between 'FirstLayerNameToFreeze' and 'LastLayerNameToFreeze'
        /// if 'FirstLayerNameToFreeze' is empty and 'LastLayerNameToFreeze' is valid
        ///     we'll freeze all layers from the start of the network to layer 'LastLayerNameToFreeze'
        /// if both 'FirstLayerNameToFreeze' and 'LastLayerNameToFreeze' are empty
        ///     no layers will be freezed
        /// </summary>
        public string FirstLayerNameToFreeze = "";
        /// <summary>
        /// name of the the last layer for which we want to freeze the weights
        /// </summary>
        public string LastLayerNameToFreeze = "";

        #region logging
        public static string DefaultWorkingDirectory => Utils.ChallengesPath;
        public static string DefaultDataDirectory => Path.Combine(DefaultWorkingDirectory, "Data");
        #endregion

        #region Learning Rate Scheduler
        public NetworkSample WithCyclicCosineAnnealingLearningRateScheduler(int nbEpochsInFirstRun, int nbEpochInNextRunMultiplier, double minLearningRate = 0.0)
        {
            DisableReduceLROnPlateau = true;
            LearningRateSchedulerType = LearningRateSchedulerEnum.CyclicCosineAnnealing;
            CyclicCosineAnnealing_nbEpochsInFirstRun = nbEpochsInFirstRun;
            CyclicCosineAnnealing_nbEpochInNextRunMultiplier = nbEpochInNextRunMultiplier;
            CyclicCosineAnnealing_MinLearningRate = minLearningRate;
            return this;
        }
        public NetworkSample WithLinearLearningRateScheduler(int dividerForMinLearningRate)
        {
            DisableReduceLROnPlateau = true;
            LearningRateSchedulerType = LearningRateSchedulerEnum.Linear;
            Linear_DividerForMinLearningRate = dividerForMinLearningRate;
            return this;
        }
        public NetworkSample WithOneCycleLearningRateScheduler(int dividerForMinLearningRate, double percentInAnnealing)
        {
            LearningRateSchedulerType = LearningRateSchedulerEnum.OneCycle;
            DisableReduceLROnPlateau = true;
            OneCycle_DividerForMinLearningRate = dividerForMinLearningRate;
            OneCycle_PercentInAnnealing = percentInAnnealing;
            return this;
        }
        public NetworkSample WithCifar10ResNetLearningRateScheduler(bool disableReduceLROnPlateau, bool divideBy10OnPlateau, bool linearLearningRate)
        {
            LearningRateSchedulerType = LearningRateSchedulerEnum.Cifar10ResNet;
            DisableReduceLROnPlateau = disableReduceLROnPlateau;
            DivideBy10OnPlateau = divideBy10OnPlateau;
            LinearLearningRate = linearLearningRate;
            return this;
        }
        public NetworkSample WithCifar10WideResNetLearningRateScheduler(bool disableReduceLROnPlateau, bool divideBy10OnPlateau, bool linearLearningRate)
        {
            LearningRateSchedulerType = LearningRateSchedulerEnum.Cifar10WideResNet;
            DisableReduceLROnPlateau = disableReduceLROnPlateau;
            DivideBy10OnPlateau = divideBy10OnPlateau;
            LinearLearningRate = linearLearningRate;
            return this;
        }
        public NetworkSample WithCifar10DenseNetLearningRateScheduler(bool disableReduceLROnPlateau, bool divideBy10OnPlateau, bool linearLearningRate)
        {
            LearningRateSchedulerType = LearningRateSchedulerEnum.Cifar10DenseNet;
            DisableReduceLROnPlateau = disableReduceLROnPlateau;
            DivideBy10OnPlateau = divideBy10OnPlateau;
            LinearLearningRate = linearLearningRate;
            return this;
        }

        public NetworkSample WithConstantLearningRateScheduler(double learningRate)
        {
            LearningRateSchedulerType = LearningRateSchedulerEnum.Constant;
            DisableReduceLROnPlateau = true;
            LinearLearningRate = false;
            InitialLearningRate = learningRate;
            return this;
        }

        public ReduceLROnPlateau ReduceLROnPlateau()
        {
            if (DisableReduceLROnPlateau)
            {
                return null;
            }
            var factorForReduceLrOnPlateau = DivideBy10OnPlateau ? 0.1 : Math.Sqrt(0.1);
            return new ReduceLROnPlateau(factorForReduceLrOnPlateau, 5, 5);
        }


        public ILearningRateComputer GetLearningRateComputer()
        {
            return new LearningRateComputer(GetLearningRateScheduler(), ReduceLROnPlateau(), MinimumLearningRate);
        }

        private ILearningRateScheduler GetLearningRateScheduler()
        {
            switch (LearningRateSchedulerType)
            {
                case LearningRateSchedulerEnum.OneCycle:
                    return new OneCycleLearningRateScheduler(InitialLearningRate, OneCycle_DividerForMinLearningRate, OneCycle_PercentInAnnealing, NumEpochs);
                case LearningRateSchedulerEnum.CyclicCosineAnnealing:
                    return new CyclicCosineAnnealingLearningRateScheduler(CyclicCosineAnnealing_MinLearningRate, InitialLearningRate, CyclicCosineAnnealing_nbEpochsInFirstRun, CyclicCosineAnnealing_nbEpochInNextRunMultiplier, NumEpochs);
                case LearningRateSchedulerEnum.Cifar10DenseNet:
                    return LearningRateScheduler.ConstantByInterval(1, InitialLearningRate, 150, InitialLearningRate / 10, 225, InitialLearningRate / 100);
                case LearningRateSchedulerEnum.Cifar10ResNet:
                    return LinearLearningRate
                        ? LearningRateScheduler.InterpolateByInterval(1, InitialLearningRate, 80, InitialLearningRate / 10, 120, InitialLearningRate / 100, 200, InitialLearningRate / 100)
                        : LearningRateScheduler.ConstantByInterval(1, InitialLearningRate, 80, InitialLearningRate / 10, 120, InitialLearningRate / 100, 200, InitialLearningRate / 100);
                case LearningRateSchedulerEnum.Cifar10WideResNet:
                    return LearningRateScheduler.ConstantByInterval(1, InitialLearningRate, 60, InitialLearningRate / 5, 120, InitialLearningRate / 25, 160, InitialLearningRate / 125);
                case LearningRateSchedulerEnum.Linear:
                    return LearningRateScheduler.Linear(InitialLearningRate, NumEpochs, InitialLearningRate / Linear_DividerForMinLearningRate);
                case LearningRateSchedulerEnum.Constant:
                    return LearningRateScheduler.Constant(InitialLearningRate);
                default:
                    throw new Exception("unknown LearningRateSchedulerType: " + LearningRateSchedulerType);
            }
        }
        #endregion

        public override void Use_All_Available_Cores()
        {
            if (GetDeviceCount() == 0)
            {
                SetResourceId(-1); //we'll use all available CPU
            }
            else
            {
                SetResourceId(int.MaxValue); //we'll use all available GPU
            }
        }


        public override bool FixErrors()
        {
            switch (OptimizerType)
            {
                case Optimizer.OptimizationEnum.AdamW:
                    WithAdamW(AdamW_L2Regularization, Adam_beta1, Adam_beta2, Adam_epsilon);
                    //lambdaL2Regularization = SGD_momentum = 0;
                    //SGD_usenesterov = false;
                    break;
                case Optimizer.OptimizationEnum.SGD:
                    WithSGD(SGD_momentum, SGD_usenesterov);
                    //AdamW_L2Regularization = Adam_beta1 = Adam_beta2 = Adam_epsilon = 0;
                    break;
                case Optimizer.OptimizationEnum.Adam:
                    WithAdam(Adam_beta1, Adam_beta2, Adam_epsilon);
                    //AdamW_L2Regularization = SGD_momentum = 0;
                    //SGD_usenesterov = false;
                    break;
                case Optimizer.OptimizationEnum.VanillaSGD:
                case Optimizer.OptimizationEnum.VanillaSGDOrtho:
                    //SGD_momentum = AdamW_L2Regularization = Adam_beta1 = Adam_beta2 = Adam_epsilon = 0;
                    //SGD_usenesterov = false;
                    break; // no extra configuration needed
            }

            switch (LearningRateSchedulerType)
            {
                case LearningRateSchedulerEnum.CyclicCosineAnnealing:
                    WithCyclicCosineAnnealingLearningRateScheduler(CyclicCosineAnnealing_nbEpochsInFirstRun, CyclicCosineAnnealing_nbEpochInNextRunMultiplier, CyclicCos