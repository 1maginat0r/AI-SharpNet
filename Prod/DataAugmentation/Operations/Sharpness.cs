using System;
using SharpNet.CPU;

namespace SharpNet.DataAugmentation.Operations
{
    public class Sharpness : Operation
    {
        private readonly float _enhancementFactor;

        public Sharpness(float enhancementFactor)
        {
            _enhancementFactor = enhancementFactor;
        }

        public override float AugmentedValue(int indexInMiniBatch, int channel,
            CpuTensor<float> xInputMiniBatch, int rowInput, int colInput, 
            CpuTensor<float> xOutputMiniBatch, int rowOutput, int colOutput)
        {
            var initialValue = xInputMiniBatch.Get(indexInMiniBatch, channel, rowInput, colInput);
            var nbRows = xInputMiniBatch.Shape[2];
            var nbCols = xInputMiniBatch.Shape[3];
            int count = 0;
            var smoothValue = 0f;
            /*
             * We use the following weight value centered on the pixel
             *      [ [ 1 1 1 ]
             *        [ 1 5 1 ]
             *        [ 1 1 1 ] 