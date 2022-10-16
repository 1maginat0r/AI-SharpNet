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
        #regio