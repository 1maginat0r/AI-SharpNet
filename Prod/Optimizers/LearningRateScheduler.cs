using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using SharpNet.Data;

namespace SharpNet.Optimizers
{
    public class LearningRateScheduler : ILearningRateScheduler
    {
        #region private fields
        private readonly List<Tuple<double, double>> Values;
        private readonly bool _constantByInterval;
        /// <summary>
        /// if true:
        ///     in each epoch, use the same learning rate
        /// else:
        ///     use a linear learning rate from the start to the end of the epoch
        /// </summary>
        private readonly bool ConstantInEa