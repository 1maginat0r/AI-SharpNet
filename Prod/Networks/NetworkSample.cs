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
        public