using System.Collections.Generic;
using System.Diagnostics;
using SharpNet.Data;
using SharpNet.Networks;

namespace SharpNet.Layers
{
    public class ZeroPadding2DLayer : Layer
    {
        #region private fields
        private readonly int _paddingTop;
        private readonly int _paddingBottom;
        private readonly int _paddingLeft;
        private readonly int _paddingRight;
        #endregion

        public ZeroPadding2DLayer(int paddingTop, int paddingBottom, int paddingLeft, int paddingRight, int previousLayerIndex, Network network, string layerName) : base(network, new[] { previousLayerIndex}, layerName)
        {
            _paddingTop = paddingTop;
            _paddingBottom = paddingBottom;
            _paddingLeft = paddingLeft;
            _paddingRight = paddingRight;
        }

        #region forward and backward propagation
        public override void ForwardPropagation(List<Tensor> allX, Tensor y, bool isTraining)
        {
            Debug.Assert(allX.Count == 1);
            var x = allX[0];
      