using System.Collections.Generic;
using System.Diagnostics;
using SharpNet.Data;
using SharpNet.GPU;
using SharpNet.Networks;

namespace SharpNet.Layers
{
    public class DropoutLayer : Layer
    {
        #region fields
        private readonly double _dropoutRate;
        private Tensor _dropoutReservedSpaceForTraining;
        #endregion

        public DropoutLayer(double dropoutRate, Network network, string layerName) : base(network, layerName)
        {
            _dropoutRate = dropoutRate;
        }

        #region forward and backward propagation
        public override void ForwardPropagation(List<Tensor> allX, Tensor y, bool isTraining)
        {
            Debug.Assert(allX.Count == 1);
            var x = allX[0];

            if (isTraining)
            {
                //we initialize the dropout reserved space buffer
                InitializeDropoutReservedSpaceForTraining(x);
            }
            else
            {
                //no need of dropout reserved space for inference
                FreeFloatTensor(ref _dropoutReservedSpaceForTraining);
            }
            x.Dr