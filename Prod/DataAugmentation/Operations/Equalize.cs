using System;
using System.Collections.Generic;
using SharpNet.CPU;

namespace SharpNet.DataAugmentation.Operations
{
    public class Equalize : Operation
    {
        private readonly List<int[]> _originalPixelToEqualizedPixelByChannel;
        private readonly List<Tuple<float, float>> _meanAndVolatilityForEachChannel;


        public Equalize(List<int[]> originalPixelToEqualizedPixelByChannel, List<Tuple<float, float>> meanAndVolatilityForEachChannel)
        {
            _originalPixelToEqualizedPixelByChannel = originalPixelToEqualizedPixelByChannel;
            _meanAndVolatilityForEachChannel = meanAndVolatilityForEachChannel;
        }

        public override float AugmentedValue(int indexInMiniBatch, int channel,
    