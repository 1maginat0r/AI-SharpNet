using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using SharpNet.Data;
using SharpNet.GPU;
using SharpNet.Networks;

namespace SharpNet.Layers
{
    /// <summary>
    /// x shape:
    ///     (batchSize, x.C, x.H, x.W)      if 4D
    ///     (batchSize, x.H, x.W)           if 3D
    /// y shape:
    ///     (batchSize, x.C, y.H, y.W)      if 4D
    ///     (batchSize, y.H, y.W)           if 3D
    ///          y.H = (x.H − poolingHeight) / poolingStride + 1
    ///          y.W = (x.W − poolingWidth) / poolingStride + 1
    /// </summary>
    public class PoolingLayer : Layer
    {
        #region Fields
        private readonly cudnnPoolingMode_t _poolingMode;
        /// <summary>
        /// pooling height
        /// -1 if we are using global vertical pooling
        /// </summary>
        private readonly int _poolingHeight;
        /// <summary>
        /// pooling width
        /// -1 if we are using global horizontal pooling
        /// </summary>
        private readonly int _poolingWidth;
        /// <summary>
        /// vertical (height) stride
        /// -1 if we are using global vertical pooling
        /// </summary>
        private readonly int _verticalStride;
        /// <summary>
        /// horizontal (width) stride
        /// -1 if we are using global horizontal pooling
        /// </summary>
        private readonly int _horizontalStride;
  