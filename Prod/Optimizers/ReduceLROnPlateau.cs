using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using SharpNet.Data;

namespace SharpNet.Optimizers
{
    /// <summary>
    /// This class is used for plateau detection (: )several epochs in a row without progress)
    /// One trick in this case is to reduce the learning rate
    /// </summary>
    public class ReduceLROnPlateau
    {
        #region fields & properties
        public double FactorForReduceLrOnPlateau { get; }
        private readonly int _patienceForReduceLrOnPlateau;
        private readonly int _cooldownForReduceLrOnPlateau;
        #endregion


        public ReduceLROnPlateau(double factorForReduceLrOnPlateau, int patienceForReduceLrOnPlateau, int cooldownForReduceLrOnPlateau)
        {
            FactorForReduceLrOnPlateau = factorForReduceLrOnPlateau;
            _patienceForReduceLrOnPlateau = patienceForReduceLrOnPlateau;
            _cooldownForReduceLrOnPlateau = cooldownForReduceLrOnPlateau;
        }


        /// <summary>
        /// return the distance (# epochs) between the last epoch and the best epoch found so far (in term of training loss)
        /// </summary>
        /// <param name="epochData"></param>
        /// <param name="loss"></param>
        /// <param name="maxNbConsecutiveEpochsToReport">stops as soon as the returned result is >= 'maxNbConsecutiveEpochsToReport'</param>
        /// <returns>distance (in number of epochs) between the last epoch and the best epoch found so far (in term of training loss)
        /// it will return 0 if the best epoch was the last processed, 1 if the best epoch was just before the last epoch, etc...</returns>
        public static int NbConsecutiveEpochsWithoutProgress(List<EpochData> epochData, EvaluationMetricEnum loss, int maxNbConsecutiveEpochsToReport = int.MaxValue)
        {
            Debug.Assert(maxNbConsecutiveEpochsToReport>=1);
            if (epochData.Count <= 1)
            {
                return 0;
            }
