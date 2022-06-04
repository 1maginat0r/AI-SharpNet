using System.Collections.Generic;
using System.Diagnostics;
using SharpNet.Data;
using SharpNet.Networks;

namespace SharpNet.Layers
{
    public class UpSampling2DLayer : Layer
    {
        public enum InterpolationEnum {Nearest, Bilinear};

        #region private fields
        private readonly int _rowFactor;
        private readonly int _colFactor;
        private readonly InterpolationEnum _interpolation;

        #endregion

        public UpSampling2DLayer(int rowFactor, int colFactor, InterpolationEnum interpolation, Network network, string layerName) : base(network, layerName)
        {
            Debug.Assert(LayerIndex >= 2);
            _rowFactor = rowFactor;
            _colFactor = colFactor;
            _interpolation = interpolation;
        }

        #region forward and backward propagation
        public override void ForwardPropagation(List<Tensor> allX, Tensor y, bool isTraining)
        {
            Debug.Assert(allX.Count == 1);
            var x = allX[0];
            y.UpSampling2D(x, _rowFactor, _colFactor, _interpolation);
        }
        public override void BackwardPropagation(List<Tensor> allX_NotUsed, Tensor y_NotUsed, Tensor dy, List<Tensor> allDx)
        {
            Debug.Assert(allX_NotUsed.Count == 0);
            Debug.Assert(y_NotUsed == null);
            Debug.Assert(allDx.Count == 1);
            var dx = allDx[0];
            dx.DownSampling2D(dy, _rowFactor, _colFactor);