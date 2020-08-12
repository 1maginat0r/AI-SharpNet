
﻿using System;
using System.Collections.Generic;
using SharpNet.CPU;
using SharpNet.DataAugmentation.Operations;

namespace SharpNet.DataAugmentation
{
    public class AutoAugment : AbstractDataAugmentationStrategy
    {
        private const float MAGNITUDE_MAX = 9f;
        private readonly double _widthShiftRangeInPercentage;
        private readonly double _heightShiftRangeInPercentage;
        private readonly bool _horizontalFlip;
        private readonly bool _verticalFlip;
        private readonly bool _rotate180Degrees;

        public AutoAugment(int indexInMiniBatch, 
            CpuTensor<float> xOriginalMiniBatch,
            List<Tuple<float, float>> meanAndVolatilityForEachChannel, Lazy<ImageStatistic> stats, Random rand,
            double widthShiftRangeInPercentage, double heightShiftRangeInPercentage, double cutoutPatchPercentage,
            double alphaCutMix, double alphaMixup, bool horizontalFlip, bool verticalFlip, bool rotate180Degrees) : 
            base(indexInMiniBatch, xOriginalMiniBatch, meanAndVolatilityForEachChannel, stats, rand, cutoutPatchPercentage, alphaCutMix, alphaMixup)