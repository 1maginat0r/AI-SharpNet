
﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using SharpNet.Data;

namespace SharpNet.Optimizers
{
    /// <summary>
    /// Compute the learning for a specific batch in a specific epoch
    /// </summary>
    public class LearningRateComputer : ILearningRateComputer
    {
        private readonly ILearningRateScheduler _lrScheduler;
        private readonly ReduceLROnPlateau _reduceLrOnPlateauIfAny;
        private readonly double _minimumLearningRate;

        public LearningRateComputer(ILearningRateScheduler lrScheduler, ReduceLROnPlateau reduceLrOnPlateauIfAny, double minimumLearningRate)
        {
            _lrScheduler = lrScheduler;
            _reduceLrOnPlateauIfAny = reduceLrOnPlateauIfAny;
            _minimumLearningRate = minimumLearningRate;
        }

        /// <summary>
        /// Return the learning rate associated with epoch 'epoch' / batch size 'blockIdInEpoch'
        /// </summary>
        /// <param name="epoch">id of epoch. The first epoch is number 1 (not 0)</param>
        /// <param name="percentagePerformedInEpoch"></param>
        /// <param name="learningRateMultiplicativeFactorFromReduceLrOnPlateau"></param>
        /// <returns></returns>
        public double LearningRate(int epoch, double percentagePerformedInEpoch, double learningRateMultiplicativeFactorFromReduceLrOnPlateau)
        {
            Debug.Assert(epoch>= 1);
            Debug.Assert(percentagePerformedInEpoch >= 0);
            Debug.Assert(percentagePerformedInEpoch <= 1);
            var learningRateFromScheduler = _lrScheduler.LearningRate(epoch, percentagePerformedInEpoch);
            var learningRateWithPlateauReduction = learningRateFromScheduler * learningRateMultiplicativeFactorFromReduceLrOnPlateau;
            learningRateWithPlateauReduction = Math.Max(learningRateWithPlateauReduction, _minimumLearningRate);
            return learningRateWithPlateauReduction;
        }

        /// <summary>
        /// Check if we should reduce the learning rate because we have reached a plateau
        /// (a plateau: no improvement in several epochs in a row)
        /// this method is called at the beginning of each new epoch
        /// </summary>
        /// <param name="previousEpochsData">stats associated with the previous computed epochs</param>
        /// <param name="loss"></param>
        /// <returns>true if we should reduce the learning rate</returns>
        public bool ShouldReduceLrOnPlateau(List<EpochData> previousEpochsData, EvaluationMetricEnum loss)
        {
            return _reduceLrOnPlateauIfAny != null && _reduceLrOnPlateauIfAny.ShouldReduceLrOnPlateau(previousEpochsData, loss);
        }

        /// <summary>
        /// The multiplier we should use that take into account all plateaus reached from the beginning of the search
        /// </summary>
        /// <param name="previousEpochsData">stats associated with the previous computed epochs</param>
        /// <param name="loss"></param>
        /// <returns>
        /// a value strictly less then 1.0 if we should reduce the learning rate because of several previous plateaux
        /// 1.0 is we should not reduce the learning rate
        /// </returns>
        public double MultiplicativeFactorFromReduceLrOnPlateau(List<EpochData> previousEpochsData, EvaluationMetricEnum loss)
        {
            var result = previousEpochsData.Count >= 1
                ? previousEpochsData.Last().LearningRateMultiplicativeFactorFromReduceLrOnPlateau
                : 1.0;
            if (ShouldReduceLrOnPlateau(previousEpochsData, loss))
            {
                result *= _reduceLrOnPlateauIfAny.FactorForReduceLrOnPlateau;
            }
            return result;
        }
        public bool ShouldCreateSnapshotForEpoch(int epoch)
        {
            return _lrScheduler.ShouldCreateSnapshotForEpoch(epoch);
        }

        public double MaxLearningRate => _lrScheduler.MaxLearningRate;
    }
}