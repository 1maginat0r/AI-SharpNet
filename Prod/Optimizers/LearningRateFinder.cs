using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using System.Text;
using SharpNet.Data;

namespace SharpNet.Optimizers
{
    /// <summary>
    /// see https://www.jeremyjordan.me/nn-learning-rate/
    /// </summary>
    public class LearningRateFinder : ILearningRateComputer
    {
        #region private fields
        private readonly double _momentum;
        private readonly double _minLearningRate;
        /// <summary>
        ///between each block, we'll multiply the learning rate by 'multiplicativeCoeff'
        /// </summary>
        private readonly double _multiplicativeCoeff;
        private readonly List<double> _loss = new ();
        private readonly List<double> _avgLosses = new ();
        private readonly List<double> _smoothedLosses = new ();
        private readonly int _nbBlocksPerEpoch;

        #endregion

        public LearningRateFinder(int miniBatchSize, int entireBatchSize, doub