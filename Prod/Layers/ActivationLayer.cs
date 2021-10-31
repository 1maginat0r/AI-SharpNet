using System;
using System.Collections.Generic;
using System.Diagnostics;
using SharpNet.Data;
using SharpNet.GPU;
using SharpNet.Networks;

namespace SharpNet.Layers
{
    public class ActivationLayer : Layer
    {
        #region private fields
        private readonly Tensor _activationParameter;
        #endregion

        #region public fields and properties
        public cudnnActivationMode_t ActivationFunction { get; }
        #endregion

        //No need to configure the number of channels by filter: it is always the same as in previous layer
        public ActivationLayer(cudnnActivationMode_t activationFunctionType, Tensor activationParameter, Network network, string layerName) : base(network, layerName)
        {
            if (activationParameter != null && activationParameter.UseGPU != Network.UseGPU)
            {
                activationParameter = Network.UseGPU ? activationParameter.ToGPU<float>(Network.GpuWrapper) : activationParameter.ToCpuFloat();
            }
            _activationParameter = activationParameter;
            ActivationFunction = activationFunctionType;
        }

        #region forward and backward propagation
        pub