using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace SharpNet.Optimizers
{
    /// <summary>
    /// Implementation of Cyclic Cosine Annealing Learning Rate = SGDR = stochastic gradient descent with warm restarts
    /// (see https://arxiv.org/pdf/1608.03983.pdf)
    /// </summary>
    public class CyclicCosineAnnealingLearningRateScheduler : ILearningRateScheduler
    {
        #region private fields
        private readonly double _minLearningRate;
        private readonly List<Tuple<double, double>> _values = new ();
        /// <summary>
        /// the last epoch of each cycle.
        /// For instance, if we have 3 cycles :
        ///     1=>10 then 11=>30 then 31=>70
        /// then it will contains
        ///     10, 30, 70
        /// </summary>
        private List<int> EndEpochForEachCycle { get; } = new ();
        #endregion

        /// <summary>
        /// see https://arxiv.org/pdf/1608.03983.pdf
        /// </summary>
        /// <param name="minLearningRate"></param>
        /// <param name="maxLearningRate"></param>
        /// <param name="nbEpochsInFirstRun">Number of epochs in the first warm started run</param>
        /// <param name="nbEpochInNextRunMultiplier">factor to multiply the number of epochs in the previous run</param>
        /// <param name="nbEpochs">total number of epochs to get the number of epochs in the next run</param>
        public CyclicCosineAnnealingLearningRateScheduler(double minLearningRate, double maxLearningRate, int nbEpochsInFirstRun, int nbEpochInNextRunMultiplier, int nbEpochs)
        {
            Debug.Assert(nbEpochsInFirstRun>=1);
            Debug.Assert(nbEpochInNextRunMultiplier >= 1);
            int firstEpochInCurrentRun = 1;
            int nbEpochsInCurrentRun = nbEpochsInFirstRun;
            for (;;)
            {
                _values.Add(Tuple.Create((double)firstEpochInCurrentRun,0.0));
                var lastEpochInCurrentRun = firstEpochInCurrentRun+ nbEpochsInCurrentRun-1;
                var firstEpochInNextRun = firstEpochInCurrentRun+ nbEpochsInCurrentRun;
                var nbEpochsInNextRun = nbEpochsInCurrentRun*nbEpochInNextRunMultiplier;
                var lastEpochInNextRun = firstEpochInNextRun + nbEpochsInNextRun-1;
                //if it is not possible to finish entirely the next run
                if (lastEpochInNextRun > nbEpochs)
                {
                    //we'll increase the length (in epoch count) of the current run to make it finish at exactly the last epoch
                    //(no need to do a new run after it because it will have a smaller size)
                    lastEpochInCurrentRun = nbEpochs;
                }
                EndEpochForEachCycle.Add(lastEpochInCurrentRun);
                _values.Add(Tuple.Create(lastEpochInCurrentRun+1 - 1e-6, 1.0));
       