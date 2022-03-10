using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using SharpNet.Data;
using SharpNet.Networks;

namespace SharpNet.Layers
{
    public class MultiplyLayer : Layer
    {
        public MultiplyLayer(int mainMatrixLayerIndex, int diagonalMatrixLayerIndex, Network network, string layerName) : base(network, new []{ mainMatrixLayerIndex, diagonalMatrixLayerIndex }, layerName)
        {
            Debug.Assert(LayerIndex >= 2);
            if (!ValidLayerShapeToMultiply(PreviousLayerMainMatrix.OutputShape(1), PreviousLayerDiagonalMatrix.OutputShape(1)))
            {
                throw new ArgumentException("invalid layers to multiply between " + PreviousLayerMainMatrix + " and " + diagonalMatrixLayerIndex);
            }
        }

        #region forward and backward propagation
        public override void ForwardPropagation(List<Tensor> allX, Tensor y, bool isTraining)
        {
            Debug.Assert(allX.Count == 2);
            var a = allX[0];
            var diagonalMatrix = allX[1]; //vector with the content of the diagonal matrix
            y.MultiplyTensor(a, diagonalMatrix);
        }
        public override void BackwardPropagation(List<Tensor> allX, Tensor y_NotUsed, Tensor dy, List<Tensor> allDx)
        {
            Debug.Assert(allDx.Count == 2);
            Debug.Assert(y_NotUsed == null);
            var dx1 = allDx[0];
            var dx2 = allDx[1];
            var a = allX[0];
            var diagonalMatrix = allX[1];
            Debug.Assert(dx1.SameShape(dy));
            Debug.Assert(dx1.SameShape(a));
            Debug.Assert(dx2.SameShape(diagonalMatrix));


            StartBackwardTimer(LayerType() + ">SameShape");
            dx1.MultiplyTensor(dy, diagonalMatrix);
            StopBackwardTimer(LayerType() + ">SameShape");
            if (dx2.SameShape(dy))
            {
                StartBackwa