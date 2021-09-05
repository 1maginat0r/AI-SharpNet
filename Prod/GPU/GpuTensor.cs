
﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using JetBrains.Annotations;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.Layers;
// ReSharper disable IdentifierTypo

namespace SharpNet.GPU
{
    public sealed unsafe class GPUTensor<T> : Tensor
    {
        #region Private fields
        private readonly GPUWrapper _wrapper;
        /// <summary>
        /// pointer to the start of the tensor in the device (GPU) memory 
        /// </summary>
        private readonly IntPtr _pointerToStartOfTensor;
        #endregion
        #region public properties
        /// <summary>
        /// true if the 'this' tensor is the owner of the associate memory
        ///     => in this case the memory associated with the 'this' tensor should be freed by the tensor when it is disposed
        /// false if the 'this' tensor is a span to an already allocated memory area in the device
        ///     => in this case this memory associated with the 'this' tensor should not be de allocated by the tensor when it is disposed
        /// </summary>
        public override bool IsOwnerOfMemory {get;}
        #endregion

        #region constructors
        public GPUTensor(int[] shape, Memory<T>? unpinnedHostMemory, GPUWrapper wrapper) : base(shape, Marshal.SizeOf(typeof(T)), true)
        {
            _wrapper = wrapper;
            _wrapper.CheckThreadId();
            CapacityInBytes = ReallyNeededMemoryInBytes;
            CapacityInBytes = Math.Max(CapacityInBytes, 8);
            _pointerToStartOfTensor = IntPtr.Zero;

            var res = NVCudaWrapper.cuMemAlloc_v2(out _pointerToStartOfTensor, CapacityInBytes);
            GPUWrapper.CheckStatus(res);
            if (unpinnedHostMemory.HasValue)
            {
                InitializeFromHostMemory(unpinnedHostMemory.Value);
            }
            IsOwnerOfMemory = true;
        }
        /// <summary>
        /// construct a tensor that is a span to an already allocated memory
        /// this memory should not be de allocated 
        /// </summary>
        /// <param name="shape">shape of the tensor</param>
        /// <param name="pointerToMemoryOwner">the already allocated memory area that the tensor will use</param>
        /// <param name="wrapper"></param>
        public GPUTensor(int[] shape, IntPtr pointerToMemoryOwner, GPUWrapper wrapper) : base(shape, Marshal.SizeOf(typeof(T)), true)
        {
            _wrapper = wrapper;
            _wrapper.CheckThreadId();
            CapacityInBytes = ReallyNeededMemoryInBytes;
            _pointerToStartOfTensor = pointerToMemoryOwner;
            IsOwnerOfMemory = false;
        }
        #endregion

        /// <summary>
        /// copy from CPU (Host) pinned memory to GPU (Device) memory
        /// </summary>
        /// <param name="hostPinnedPointer">pointer to host (pinned) memory (in CPU) </param>
        public void InitializeFromHostPinnedMemory(IntPtr hostPinnedPointer)
        {
            AssertIsNotDisposed();
            Debug.Assert(hostPinnedPointer != IntPtr.Zero);
            _wrapper.SwCopyHostToDevice.Start();
            _wrapper.LogCopyHostToDeviceCall(ReallyNeededMemoryInBytes);
            //asynchronous copy
            var res = NVCudaWrapper.cuMemcpyHtoDAsync_v2(Pointer, hostPinnedPointer, ReallyNeededMemoryInBytes, _wrapper.DefaultStream.StreamHandle);
            //for synchronous copy: var res = NVCudaWrapper.cuMemcpyHtoD_v2(Pointer, hostPinnedPointer, ReallyNeededMemoryInBytes)
            GPUWrapper.CheckStatus(res);
            _wrapper.SwCopyHostToDevice.Stop();
        }
        /// <summary>
        /// copy the first 'Count' element of 'buffer into the 'this' Tensor
        /// </summary>
        /// <param name="buffer">a buffer to read from
        /// It must contains at least 'Count' elements
        /// </param>
        public void InitializeFromHostMemory(Memory<T> buffer)
        {
            Debug.Assert(buffer.Length >= Count);
            using var m = new HostPinnedMemory<T>(buffer);
            InitializeFromHostPinnedMemory(m.Pointer);
        }
        /// <summary>
        /// copy from 'src' GPU (Device) to 'this' GPU (Device) memory
        /// the source and target device (GPU) may be different
        /// </summary>
        /// <param name="src">the tensor stored in device memory that we should copy into the current tensor</param>
        public void InitializeFromDeviceMemory(GPUTensor<T> src)
        {
            AssertIsNotDisposed();
            Debug.Assert(Count == src.Count);
            var srcDeviceId = src._wrapper.DeviceId;
            if (_wrapper.DeviceId == srcDeviceId)
            {
                //copy in the same device (GPU)
                _wrapper.SwCopyDeviceToSameDevice.Start();
                _wrapper.LogCopyDeviceToSameDeviceCall(ReallyNeededMemoryInBytes);
                src.CopyTo(0, this, 0, Count);
                _wrapper.SwCopyDeviceToSameDevice.Stop();
            }
            else
            {
                //copy between 2 distinct devices (GPU)
                _wrapper.SwCopyDeviceToOtherDevice.Start();
                _wrapper.LogCopyDeviceToOtherDeviceCall(ReallyNeededMemoryInBytes);
                //asynchronous copy
                var res = CudartWrapper.cudaMemcpyPeerAsync(Pointer, _wrapper.DeviceId, src.Pointer, srcDeviceId, ReallyNeededMemoryInBytes, IntPtr.Zero);
                //for synchronous copy:
                //var res = CudartWrapper.cudaMemcpyPeer(Pointer, _wrapper.DeviceId, src.Pointer, srcDeviceId, ReallyNeededMemoryInBytes);
                GPUWrapper.CheckStatus(res);
                _wrapper.SwCopyDeviceToOtherDevice.Stop();
            }
        }

        #region Tensor implementation
        public override void BatchNormalization(Tensor y, Tensor scale, Tensor bias, double exponentialAverageSmoothingFactor, Tensor runningInputMean, Tensor runningInputVariance, cudnnBatchNormMode_t mode, double epsilon, Tensor meanBuffer, Tensor invertOfUnbiasedVolatilityBuffer, bool isTraining)
        {
            var x = this;
            Debug.Assert(AreCompatible(new List<Tensor> { x, y, scale, bias, runningInputMean, runningInputVariance, meanBuffer, invertOfUnbiasedVolatilityBuffer }));
            var xDesc = TensorDesc(x);
            var yDesc = xDesc;
            var bnScaleBiasMeanVarDesc = TensorDesc(scale);

            float oneFloat = 1f, zeroFloat = 0f;
            var zero = &zeroFloat;
            var one = &oneFloat;

            if (isTraining)
            {
               var res = CudnnWrapper.cudnnBatchNormalizationForwardTraining(CudnnHandle, mode, one, zero,
                        xDesc, x, yDesc, y,
                        bnScaleBiasMeanVarDesc, scale, bias, exponentialAverageSmoothingFactor,
                        runningInputMean,
                        runningInputVariance, epsilon, meanBuffer,
                        invertOfUnbiasedVolatilityBuffer);
                CheckStatus(res);
            }
            else
            {
                var res = CudnnWrapper.cudnnBatchNormalizationForwardInference(CudnnHandle, mode, one, zero,
                        xDesc, x, yDesc, y,
                        bnScaleBiasMeanVarDesc, scale, bias, 
                        runningInputMean,
                        runningInputVariance, epsilon);
                CheckStatus(res);
            }
        }
        public override void BatchNormalizationBackward(Tensor dy, Tensor dx, Tensor scale, Tensor scaleGradient, Tensor biasGradient, cudnnBatchNormMode_t mode, double epsilon, Tensor meanBuffer, Tensor invertOfUnbiasedVolatilityBuffer)
        {
            var x = this;
            Debug.Assert(AreCompatible(new List<Tensor> { x, dy, dx, scale, scaleGradient, biasGradient, meanBuffer, invertOfUnbiasedVolatilityBuffer }));
            var xDesc = TensorDesc(x);
            var bnScaleBiasDiffDesc = TensorDesc(scale);

            float oneFloat = 1f, zeroFloat = 0f;
            var zero = &zeroFloat;
            var one = &oneFloat;

            var res = CudnnWrapper.cudnnBatchNormalizationBackward(CudnnHandle, mode, 
                one, zero, one, zero,
                xDesc, x,
                xDesc, dy,
                xDesc, dx,
                bnScaleBiasDiffDesc, scale, scaleGradient, biasGradient,
                epsilon, meanBuffer,
                invertOfUnbiasedVolatilityBuffer);
            CheckStatus(res);
        }


        #region Layer Normalization
        public override void Compute_Row_Mean_Variance(Tensor row_mean, Tensor row_variance, bool unbiasedVariance)
        {
            var x = this;
            Debug.Assert(AreCompatible(new List<Tensor> { this, row_mean, row_variance }));
            Debug.Assert(row_mean.SameShape(row_variance));
            int rows = row_mean.Count;
            if (x.Count % rows != 0)
            {
                throw new ArgumentException("x.Count % rows != 0");
            }
            int cols = x.Count / rows;
            _wrapper.RunKernel("Compute_Row_Mean_Variance", rows, new object[] { cols, x, row_mean, row_variance, unbiasedVariance });
        }

        // ReSharper disable once UnusedMember.Local
        private void Compute_Row_Mean_VarianceV2(Tensor row_mean, Tensor row_variance, bool unbiasedVariance)
        {
            var x = this;
            Debug.Assert(AreCompatible(new List<Tensor> { this, row_mean, row_variance }));
            Debug.Assert(row_mean.SameShape(row_variance));
            int rows = row_mean.Count;
            if (x.Count % rows != 0)
            {
                throw new ArgumentException("x.Count % rows != 0");
            }
            int cols = x.Count / rows;
            var (blocksPerGrid, threadsPerBlock) = KernelManager.Compute_BlocksPerGrid_ThreadsPerBlock_From_rows_cols(rows, cols, _wrapper.ThreadsByMultiprocessor);
            int dynamicSharedMemory = sizeof(float) * (cols + 1 + cols);
            int nextColsPowerOf2 = Utils.NextPowerOf2(cols);
            _wrapper.RunKernel("Compute_Row_Mean_Variance_V2", blocksPerGrid * threadsPerBlock, new object[] { x, row_mean, row_variance, cols, nextColsPowerOf2, false }, blocksPerGrid, threadsPerBlock, dynamicSharedMemory);
        }



        public override void StandardizeInPlace(Tensor row_mean, Tensor row_variance, int axis, float epsilon)
        {
            var x = this;
            Debug.Assert(AreCompatible(new List<Tensor> { this, row_mean, row_variance }));
            Debug.Assert(row_mean.SameShape(row_variance));
            if (axis == 1)
            {
                //we'll standardize each row
                int rows = row_mean.Count;
                if (x.Count % rows != 0)
                {
                    throw new ArgumentException("The number of elements in the tensor must be a multiple of the number of rows");
                }
                int cols = x.Count / rows;

                const int pointsByThread = 8;
                int threadsByRow = (cols + pointsByThread - 1) / pointsByThread;
                _wrapper.RunKernel("StandardizeInPlaceByRow", rows * threadsByRow, new object[] { cols, pointsByThread, threadsByRow, x, row_mean, row_variance, epsilon});
                return;
            }
            throw new NotSupportedException("Only axis=1 is supported");
        }
        public override void StandardizeRowsInPlaceBroadcastGammasBetas(Tensor row_mean, Tensor row_variance, float epsilon, Tensor col_gammas, Tensor col_betas)
        {
            var x = this;
            Debug.Assert(AreCompatible(new List<Tensor> { this, row_mean, row_variance, col_gammas, col_betas }));
            Debug.Assert(row_mean.SameShape(row_variance));
            Debug.Assert(col_gammas.SameShape(col_betas));
            //we'll standardize each row
            int rows = row_mean.Count;
            if (x.Count % rows != 0)
            {
                throw new ArgumentException("The number of elements in the tensor must be a multiple of the number of rows");
            }
            int cols = col_gammas.Count;
            if (x.Count != rows * cols)
            {
                throw new ArgumentException("x.Count != rows * cols");
            }

            const int pointsByThread = 8;
            int threadsByRow = (cols + pointsByThread - 1) / pointsByThread;
            _wrapper.RunKernel("StandardizeRowsInPlaceBroadcastGammasBetas", rows * threadsByRow, new object[] { cols, pointsByThread, threadsByRow, x, row_mean, row_variance, epsilon, col_gammas, col_betas });
        }

        public override void LayerNormalizationBackward(Tensor dy, Tensor dx, Tensor col_gammas, Tensor row_mean, Tensor row_variance, float epsilon, Tensor dmean_row, Tensor dvariance_row)
        {
            var x = this;
            Debug.Assert(AreCompatible(new List<Tensor> { x, dy, dx, col_gammas, row_mean, row_variance }));
            Debug.Assert(x.SameShape(dy, dx));
            Debug.Assert(row_mean.SameShape(row_variance));
            int rows = row_mean.Count;
            int cols = col_gammas.Count;
            if (x.Count != rows * cols)
            {
                throw new ArgumentException("x.Count != rows * cols");
            }

            _wrapper.RunKernel("LayerNormalizationBackward_dmean_dvariance", rows, new object[] { cols, x, dy, col_gammas, row_mean, row_variance, epsilon, dmean_row, dvariance_row });
            const int pointsByThread = 8;
            int threadsByRow = (cols + pointsByThread - 1) / pointsByThread;
            _wrapper.RunKernel("LayerNormalizationBackward_dx", rows * threadsByRow, new object[] { cols, pointsByThread, threadsByRow, x, dy, dx, col_gammas, row_mean, row_variance, epsilon, dmean_row, dvariance_row });
        }
        #endregion

        public override void numpy_sum(Tensor sum_result, int axis)
        {
            var a = this;
            Debug.Assert(AreCompatible(new List<Tensor> { a, sum_result }));
            sum_result.ZeroMemory();
            if (axis == 1)
            {
                int rows = sum_result.Count;
                if (a.Count % rows != 0)
                {
                    throw new ArgumentException("x.Count % rows != 0");
                }
                int cols = a.Count / rows;
                _wrapper.RunKernel("numpy_sum_RowByRow", rows, new object[] { cols, a, sum_result });
                return;
            }
            if (axis == 0)
            {
                int cols = sum_result.Count;
                if (a.Count % cols != 0)
                {
                    throw new ArgumentException("x.Count % cols != 0");
                }
                int rows = a.Count / cols;
                _wrapper.RunKernel("numpy_sum_ColByCol", cols, new object[] { rows, a, sum_result });
                return;
            }
            throw new ArgumentException("axis != 0 && axis != 1");
        }




        public override void ActivationForward(cudnnActivationMode_t activationType, Tensor activationParameter, Tensor y)
        {
            AssertIsNotDisposed();
            y.AssertIsNotDisposed();

            var x = this;
            Debug.Assert(AreCompatible(new List<Tensor> {x, y}));
            Debug.Assert(x.SameShape(y));
            var xDesc = TensorDesc(x);
            var yDesc = TensorDesc(y);

            float oneFloat = 1f, zeroFloat = 0f;
            var zero = &zeroFloat;
            var one = &oneFloat;

            switch (activationType)
            {
                case cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX:
                    var resSoftMax = CudnnWrapper.cudnnSoftmaxForward(CudnnHandle, cudnnSoftmaxAlgorithm_t.CUDNN_SOFTMAX_ACCURATE, cudnnSoftmaxMode_t.CUDNN_SOFTMAX_MODE_INSTANCE, one, xDesc, x, zero, yDesc, y);
                    CheckStatus(resSoftMax);
                    break;
                case cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX_LAST_DIMENSION:
                    Reshape(-1, x.Shape[^1]).ActivationForward(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX, null, y.Reshape(-1, y.Shape[^1]));
                    return;
                case cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX_WITH_HIERARCHY:
                    Debug.Assert(activationParameter.UseGPU);
                    x.CopyTo(y);
                    _wrapper.RunKernel("ComputeSoftmaxWithHierarchy", y.Shape[0], new object[] { y.MultDim0, activationParameter, y});
                    break;
                case cudnnActivationMode_t.CUDNN_ACTIVATION_LN:
                    Debug.Assert(activationParameter.UseGPU);
                    _wrapper.RunKernel("ComputeLn", x.Count, new object[] { x, y });
                    break;
                case cudnnActivationMode_t.CUDNN_ACTIVATION_LEAKY_RELU:
                    Debug.Assert(activationParameter.UseGPU);
                    var activationDes = ActivationDesc(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU);
                    var resLeakyRelu = CudnnWrapper.cudnnActivationForward(CudnnHandle, activationDes, one, xDesc, x, zero, yDesc, y);
                    CheckStatus(resLeakyRelu);
                    var alphaActivation = activationParameter.ContentAsFloatArray()[0];
                    y.AddTensor(alphaActivation, x, 1- alphaActivation);
                    break;
                default:
                    var activationDescriptor = ActivationDesc(activationType);
                    var res = CudnnWrapper.cudnnActivationForward(CudnnHandle, activationDescriptor, one, xDesc, x, zero, yDesc, y);
                    CheckStatus(res);
                    break;
            }
        }

        public override void ActivationBackward(cudnnActivationMode_t activationType, Tensor activationParameter, Tensor dy, Tensor x, Tensor y)
        {
            var dx = this;
            Debug.Assert(AreCompatible(new List<Tensor> {dx, dy, x, y }));
            Debug.Assert(dx.SameShape(dy, x, y));
            var dxDesc = TensorDesc(dx);
            var dyDesc = dxDesc;
            var xDesc = dxDesc;
            var yDesc = dxDesc;

            float oneFloat = 1f, zeroFloat = 0f;
            var zero = &zeroFloat;
            var one = &oneFloat;

            switch (activationType)
            {
                case cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX:
                    var resSoftMax = CudnnWrapper.cudnnSoftmaxBackward(CudnnHandle, cudnnSoftmaxAlgorithm_t.CUDNN_SOFTMAX_ACCURATE, cudnnSoftmaxMode_t.CUDNN_SOFTMAX_MODE_INSTANCE, one, yDesc, y, dyDesc, dy, zero, dxDesc, dx);
                    CheckStatus(resSoftMax);
                    return;
                case cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX_LAST_DIMENSION:
                    Reshape(-1, Shape[^1]).ActivationBackward(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX, null, dy.Reshape(-1, dy.Shape[^1]), x?.Reshape(-1, x.Shape[^1]), y.Reshape(-1, y.Shape[^1]));
                    return;
                case cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX_WITH_HIERARCHY:
                    Debug.Assert(activationParameter.UseGPU);
                    Debug.Assert(dx.MultDim0 == activationParameter.Count);
                    _wrapper.RunKernel("ComputeSoftmaxGradientWitHierarchy", dx.Count, new object[] { dx.MultDim0, activationParameter, y, dy, dx});
                    return;
                case cudnnActivationMode_t.CUDNN_ACTIVATION_LEAKY_RELU:
                    var activationDescLeakyRelu = ActivationDesc(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU);
                    var resLeakyRelu = CudnnWrapper.cudnnActivationBackward(CudnnHandle, activationDescLeakyRelu, one, yDesc, y, dyDesc, dy, xDesc, x, zero, dxDesc, dx);
                    CheckStatus(resLeakyRelu);
                    var alphaActivation = activationParameter.ContentAsFloatArray()[0];
                    dx.AddTensor(alphaActivation, dy, 1 - alphaActivation);
                    return;
                case cudnnActivationMode_t.CUDNN_ACTIVATION_LN:
                    _wrapper.RunKernel("LnGradient", dx.Count, new object[] { dy, x, dx });
                    return;
                default:
                    var activationDesc = ActivationDesc(activationType);
                    var res = CudnnWrapper.cudnnActivationBackward(CudnnHandle, activationDesc, one, yDesc, y, dyDesc, dy, xDesc, x, zero, dxDesc, dx);
                    CheckStatus(res);
                    return;
            }
        }
        public override void Pooling(Tensor y, cudnnPoolingMode_t poolingMode, int poolingHeight, int poolingWidth, int verticalStride, int horizontalStride)
        {
            var x = this;

            Debug.Assert(AreCompatible(new List<Tensor> { x, y }));
            var poolingDesc = PoolingDesc(poolingMode, poolingHeight, poolingWidth, verticalStride, horizontalStride);
            var xDesc = TensorDesc(x);
            var yDesc = TensorDesc(y);

            float oneFloat = 1f, zeroFloat = 0f;
            var zero = &zeroFloat;
            var one = &oneFloat;

            var res = CudnnWrapper.cudnnPoolingForward(CudnnHandle, poolingDesc, one, xDesc, x, zero, yDesc, y);
            CheckStatus(res);
        }
        public override void PoolingGradient(Tensor y, Tensor x, Tensor dx, cudnnPoolingMode_t poolingMode, int poolingHeight, int poolingWidth, int verticalStride, int horizontalStride)
        {
            var dy = this;
            Debug.Assert(AreCompatible(new List<Tensor> { dy, y, x, dx }));
            Debug.Assert(x.Shape.Length == 4);
            Debug.Assert(SameDimension(new List<Tensor> { dy, y, x, dx }));
            var poolingDesc = PoolingDesc(poolingMode, poolingHeight, poolingWidth, verticalStride, horizontalStride);
            var xDesc = TensorDesc(x);
            var dxDesc = xDesc;
            var yDesc = TensorDesc(y);
            var dyDesc = yDesc;

            float oneFloat = 1f, zeroFloat = 0f;
            var zero = &zeroFloat;
            var one = &oneFloat;

            var res = CudnnWrapper.cudnnPoolingBackward(CudnnHandle, poolingDesc, one, yDesc, y, dyDesc, dy, xDesc, x, zero, dxDesc, dx);
            CheckStatus(res);
        }

        public override void LinearFunction(float slope, Tensor x, float intercept)
        {
            var y = this;
            Debug.Assert(y.Count == x.Count);
            if (Math.Abs(intercept) < 1e-8)
            {
                AddTensor(slope, x, 0f);
                return;
            }
            _wrapper.RunKernel("LinearFunction", Count, new object[] { y, slope, x, intercept });
        }
     
        public override void Concatenate(IList<Tensor> tensors)
        {
#if DEBUG
            CheckConcatenate(tensors);
            Debug.Assert(tensors.Count >= 2);
            Debug.Assert(tensors.Count <= 3);
#endif
            var concat = this;
            var a = tensors[0];
            var b = tensors[1];
            if (tensors.Count == 2)
            {
                _wrapper.RunKernel("Concatenate", Count, new object[] { Shape[0], concat, concat.MultDim0, a, a.MultDim0, b, b.MultDim0 });
            }
            else
            {
                var c = tensors[2];
                _wrapper.RunKernel("Concatenate3", Count, new object[] { Shape[0], concat, concat.MultDim0, a, a.MultDim0, b, b.MultDim0, c, c.MultDim0 });
            }

        }

        public override void Split(IList<Tensor> tensors)
        {
#if DEBUG
            CheckConcatenate(tensors);
            Debug.Assert(tensors.Count>=2);
            Debug.Assert(tensors.Count<=3);
#endif
            var concat = this;
            var a = tensors[0];
            var b = tensors[1];
            if (tensors.Count == 2)
            {
                _wrapper.RunKernel("Split", Count, new object[] {Shape[0], concat, concat.MultDim0, a, a.MultDim0, b, b.MultDim0});
            }
            else
            {
                var c = tensors[2];
                _wrapper.RunKernel("Split3", Count, new object[] { Shape[0], concat, concat.MultDim0, a, a.MultDim0, b, b.MultDim0, c, c.MultDim0 });
            }
        }
        /// <summary>
        /// resize the current GPU tensor to a different shape
        /// </summary>
        /// <param name="newShape"></param>
        public override void ReshapeInPlace(params int[] newShape)
        {
            AssertIsNotDisposed();
            newShape = FillMinusOneIfAny(Shape, newShape);
            if (SameShape(newShape))
            {
                //nothing to do
            }
            else if (ReallyNeededMemoryInBytesForShape(newShape) <= CapacityInBytes)
            {
                //smaller shape
                Shape = newShape;
                RecomputeMultDim();
            }
            else
            {
                //bigger shape : we do not have enough space to store it
                throw new ArgumentException("CapacityInBytes: " + CapacityInBytes + " but need memory  " + ReallyNeededMemoryInBytesForShape(newShape) + " for " + this);
            }
        }

        public override Tensor Reshape(params int[] newShape)
        {
            AssertIsNotDisposed();
            newShape = FillMinusOneIfAny(Shape, newShape);
            if (SameShape(newShape))
            {
                return this;
            }
            if (ReallyNeededMemoryInBytesForShape(newShape) <= CapacityInBytes)
            {
                return new GPUTensor<T>(newShape, Pointer, _wrapper);
            }
            //bigger shape : we do not have enough space to store it
            throw new ArgumentException("CapacityInBytes: " + CapacityInBytes + " but need memory  " + ReallyNeededMemoryInBytesForShape(newShape) + " for " + this);
        }

        /// <summary>
        /// compute: this = alpha * this 
        /// </summary>
        /// <param name="alphaFloat"></param>
        public override void Update_Multiplying_By_Alpha(float alphaFloat)
        {
            var y = this;
            var yDesc = TensorDesc(y);
            var res = CudnnWrapper.cudnnScaleTensor(CudnnHandle, yDesc, y, &alphaFloat);
            CheckStatus(res);
        }
        public override void BroadcastAddVectorToOutput(Tensor y)
        {
            var bias = this;
            Debug.Assert(AreCompatible(new List<Tensor> { bias, y }));
            Debug.Assert(y.Dimension >= 2);
            Debug.Assert(y.MultDim0 == Count);
            y.Update_Adding_Alpha_X(1, bias);
        }
        public override void Switch_First_2_axis(Tensor target)
        {
            Debug.Assert(Shape.Length >= 2);
            int aLength = Shape[0];
            int bLength = Shape[1];
            int cLength = MultDim1;
            _wrapper.RunKernel("Switch_First_2_axis", Count, new object[] { aLength, bLength, cLength, this, target});
            var targetShape = (int[])Shape.Clone();
            targetShape[0] = bLength;
            targetShape[1] = aLength;
            target.ReshapeInPlace(targetShape);
        }

        public override void SwitchSecondAndThirdDimension(Tensor target)
        {
            Debug.Assert(Shape.Length == 3 || (Shape.Length==4&&Shape[3]==1));
            Debug.Assert(target.Shape.Length == 3 || (target.Shape.Length==4&& target.Shape[3]==1));
            Debug.Assert(Shape[0] == target.Shape[0]);
            Debug.Assert(Shape[1] == target.Shape[2]);
            Debug.Assert(Shape[2] == target.Shape[1]);
            int n = Shape[0];
            int c = Shape[1];
            int h = Shape[2];
            _wrapper.RunKernel("SwitchSecondAndThirdDimension", n*c, new object[] { n, c, h, this, target });
        }

        public override void TransposeSecondAndThirdDimension(Tensor target)
        {
            Debug.Assert(Shape.Length >= 3);
            var targetShape = (int[])Shape.Clone();
            (targetShape[1], targetShape[2]) = (targetShape[2], targetShape[1]);
            target.ReshapeInPlace(targetShape);
            int A = Shape[0];
            int B = Shape[1];
            int C = Shape[2];
            int D = Count / (A*B*C);
            _wrapper.RunKernel("TransposeSecondAndThirdDimension_V1", A*B*C*D, new object[] { B, C, D, this, target });     //23.0s : 5% faster than TransposeSecondAndThirdDimension_V2 & TransposeSecondAndThirdDimension_V3 
            //_wrapper.RunKernel("TransposeSecondAndThirdDimension_V2", A*B*C, new object[] { B, C, D, this, target });     //24.6s
            //_wrapper.RunKernel("TransposeSecondAndThirdDimension_V3", A*B, new object[] { B, C, D, this, target });    //24.5s
        }

        public override void Compute_BiasGradient_from_dy(Tensor biasGradient)
        {
            var dy = this;
            Debug.Assert(AreCompatible(new List<Tensor> { dy, biasGradient}));
            Debug.Assert(Dimension >= 2);
            var dyDesc = TensorDesc(dy);
            var dbDesc = TensorDesc(biasGradient);

            float oneFloat = 1f, zeroFloat = 0f;
            var zero = &zeroFloat;
            var one = &oneFloat;

            var res = CudnnWrapper.cudnnConvolutionBackwardBias(CudnnHandle, one, dyDesc, dy, zero, dbDesc, biasGradient);
            CheckStatus(res);
        }
        #region Convolution
        public override void Convolution(Tensor filters, int paddingTop, int paddingBottom, int paddingLeft,
            int paddingRight, int stride, Tensor y, bool isDepthwiseConvolution,
            GPUWrapper.ConvolutionAlgoPreference forwardAlgoPreference, TensorMemoryPool memoryPool)
        {
            var x = this;
            Debug.Assert(AreCompatible(new List<Tensor> { x, filters, y }));
            int inputChannelCount = x.Shape[1];
            Debug.Assert(inputChannelCount == filters.Shape[1]);

            if (isDepthwiseConvolution)
            {
                //only depthMultiplier=1 is supported
                int depthMultiplier = filters.Shape[0];
                if (depthMultiplier != 1)
                {
                    throw new NotImplementedException("only depthMultiplier=1 is supported");
                }
                Debug.Assert(inputChannelCount == y.Shape[1]);
            }


            int groupCount = isDepthwiseConvolution ? inputChannelCount : 1;
            var convDesc = ConvDesc(paddingTop, paddingBottom, paddingLeft, paddingRight, stride, groupCount);
            var filterDesc = FilterDesc(filters, isDepthwiseConvolution);
            var xDesc = TensorDesc(x);
            var yDesc = TensorDesc(y);
            var forwardAlgo = _wrapper.ConvolutionForwardAlgorithm(xDesc, filterDesc, convDesc, yDesc, forwardAlgoPreference);
            var res = CudnnWrapper.cudnnGetConvolutionForwardWorkspaceSize(CudnnHandle, xDesc, filterDesc, convDesc, yDesc, forwardAlgo, out var workspaceSize); 
            CheckStatus(res);
            float oneFloat = 1f, zeroFloat = 0f;
            var zero = &zeroFloat;
            var one = &oneFloat;
            var storageBuffer = memoryPool.GetBuffer(workspaceSize);
            res = CudnnWrapper.cudnnConvolutionForward(CudnnHandle, one, xDesc, x, filterDesc, filters, convDesc, forwardAlgo, storageBuffer.Pointer, workspaceSize, zero, yDesc, y);
            memoryPool.FreeFloatTensor(ref storageBuffer);
            CheckStatus(res);
        }
        public override void BroadcastConvolutionBiasToOutput(Tensor y)
        {
            var convolutionBias = this;
            Debug.Assert(AreCompatible(new List<Tensor> { convolutionBias, y }));
            y.Update_Adding_Alpha_X(1, convolutionBias);
        }
        public override void ConvolutionBackwardBias(Tensor bias)
        {
            var dy = this;
            Debug.Assert(AreCompatible(new List<Tensor> { dy, bias }));
            var dyDesc = TensorDesc(dy);
            var dbDesc = TensorDesc(bias);

            float oneFloat = 1f, zeroFloat = 0f;
            var zero = &zeroFloat;
            var one = &oneFloat;

            var res = CudnnWrapper.cudnnConvolutionBackwardBias(CudnnHandle, one, dyDesc, dy, zero, dbDesc, bias);
            CheckStatus(res);
        }
        public override void ConvolutionGradient(Tensor convolution, Tensor dy, int paddingTop, int paddingBottom,
            int paddingLeft, int paddingRight, int stride, Tensor dx, Tensor convGradient, bool isDepthwiseConvolution,
            GPUWrapper.ConvolutionAlgoPreference backwardAlgoPreference, TensorMemoryPool memoryPool)
        {
            var x = this;
            Debug.Assert(AreCompatible(new List<Tensor> { x, convolution, dy, dx, convGradient }));
            Debug.Assert(dx==null || x.SameShape(dx));
            var xDesc = TensorDesc(x);
            var dyDesc = TensorDesc(dy);
            var dwDesc = FilterDesc(convGradient, isDepthwiseConvolution);
            int inputChannelCount = x.Shape[1];
            int groupCount = isDepthwiseConvolution ? inputChannelCount : 1;
            var convDesc = ConvDesc(paddingTop, paddingBottom, paddingLeft, paddingRight, stride, groupCount);
            var backwardFilterAlgo = _wrapper.ConvolutionBackwardFilterAlgorithm(xDesc, dyDesc, convDesc, dwDesc, backwardAlgoPreference);
            var res = CudnnWrapper.cudnnGetConvolutionBackwardFilterWorkspaceSize(CudnnHandle, xDesc, dyDesc, convDesc, dwDesc, backwardFilterAlgo, out var filterWorkspaceSize);
            CheckStatus(res);

            float oneFloat = 1f, zeroFloat = 0f;
            var zero = &zeroFloat;
            var one = &oneFloat;

            //we compute 'convGradient'
            var filterStorageBuffer = memoryPool.GetBuffer(filterWorkspaceSize);
            res = CudnnWrapper.cudnnConvolutionBackwardFilter(CudnnHandle, one, xDesc, x, dyDesc, dy, convDesc, backwardFilterAlgo, filterStorageBuffer.Pointer, filterWorkspaceSize, zero, dwDesc, convGradient);
            memoryPool.FreeFloatTensor(ref filterStorageBuffer);
            CheckStatus(res);

            //we compute 'dx'
            if (dx == null)
            {
                return;
            }
            var dxDesc = TensorDesc(dx);
            var wDesc = FilterDesc(convolution, isDepthwiseConvolution);
            var backwardDataAlgo = _wrapper.ConvolutionBackwardDataAlgorithm(dwDesc, dyDesc, convDesc, xDesc, backwardAlgoPreference);
            res = CudnnWrapper.cudnnGetConvolutionBackwardDataWorkspaceSize(CudnnHandle, dwDesc, dyDesc, convDesc, dxDesc, backwardDataAlgo, out var dataWorkspaceSize);
            CheckStatus(res);
            var dataStorageBuffer = memoryPool.GetBuffer(dataWorkspaceSize);
            res = CudnnWrapper.cudnnConvolutionBackwardData(CudnnHandle, one, wDesc, convolution, dyDesc, dy, convDesc, backwardDataAlgo, dataStorageBuffer.Pointer, dataWorkspaceSize, zero, dxDesc, dx);
            CheckStatus(res);
            memoryPool.FreeFloatTensor(ref dataStorageBuffer);
        }
        #endregion
        public override void UniformDistribution(Random rand, double minValue, double maxValue)
        {
            var array = new float[Count];
            Utils.UniformDistribution(array, rand, minValue, maxValue);
            InitializeFromHostMemory(array as T[]);
        }
        public override void NormalDistribution(Random rand, double mean, double stdDev)
        {
            var array = new float[Count];
            Utils.NormalDistribution(array, rand, mean, stdDev);
            InitializeFromHostMemory(array as T[]);
        }

        public override void SetValue(float sameValue)
        {
            var array = new float[Count];
            var sameValueAsFloat = sameValue;
            for (int i = 0; i < array.Length; ++i)
            {
                array[i] = sameValueAsFloat;
            }
            InitializeFromHostMemory(array as T[]);
        }
        public override float[] ContentAsFloatArray()
        {
            return (DeviceContent() as float[]);
        }
        public override Half[] ContentAsHalfArray()
        {
            return (DeviceContent() as Half[]);
        }

        public override Tensor Clone()
        {
            var cloned  = new GPUTensor<T>(Shape, null, _wrapper);
            CopyTo(cloned);
            return cloned;
        }

        public override void ArgMax(Tensor buffer)
        {
            var input = this;
            if (input.Shape.Length >= 3)
            {
                var input2D = input.Reshape(-1, input.Shape[^1]);
                var buffer2D = buffer.Reshape(-1, 1);
                input2D.ArgMax(buffer2D);
                return;
            }
            // input shape: (rows, numClass)
            // buffer shape: (rows, 1)
            Debug.Assert(input.Shape.Length == 2);
            Debug.Assert(buffer.Shape.Length == 2);
            Debug.Assert(buffer.Shape[1] == 1);
            Debug.Assert(input.Shape[0] == buffer.Shape[0]);
            int rows = input.Shape[0];
            int numClass = input.Shape[1];
            _wrapper.RunKernel("ArgMax", rows, new object[] { numClass, buffer, this });
        }

        #region Compute of Loss and Metrics

        private void CallCudaForLossBuffer(Tensor yExpected, Tensor yPredicted, EvaluationMetricEnum evaluationMetric)
        {
            var buffer = this;
            Debug.Assert(AreCompatible(new List<Tensor> { buffer, yExpected, yPredicted }));
            Debug.Assert(yPredicted.Dimension >= 2);
            int batchSize = yPredicted.Shape[0];
            var numClass = yPredicted.MultDim0;
            var kernelName = evaluationMetric + "LossBuffer";
            _wrapper.RunKernel(kernelName, batchSize, new object[] { numClass, buffer, yExpected, yPredicted });
        }

        protected override void ComputeAccuracyBuffer(Tensor yExpected, Tensor yPredicted)
        {
            var buffer = this;
            Debug.Assert(AreCompatible(new List<Tensor> {yExpected, yPredicted}));
            Debug.Assert(buffer.Shape.Length == 1);
            Debug.Assert(buffer.Shape[0] == yPredicted.Shape[0]);
            Debug.Assert(yExpected.SameShape(yPredicted));
            Debug.Assert(yExpected.Dimension >= 2);
            int nbRows = yExpected.Shape[0];
            var nbCols = yExpected.Shape[1];
            _wrapper.RunKernel("ComputeAccuracy", nbRows, new object[] { nbCols, buffer, yExpected, yPredicted });
        }

        protected override double ComputePearsonCorrelation(Tensor y_pred)
        {
            throw new NotImplementedException($"{nameof(EvaluationMetricEnum.PearsonCorrelation)} can not be used on GPU (only available on CPU)");
        }

        protected override double ComputeSpearmanCorrelation(Tensor y_pred)
        {
            throw new NotImplementedException($"{nameof(EvaluationMetricEnum.SpearmanCorrelation)} can not be used on GPU (only available on CPU)");
        }

        //!D TODO : use GPU for computation
        public override void ComputeAUCBuffer(Tensor yExpected, Tensor yPredicted)
        {
            var buffer = this;
            var cpuBuffer = new CpuTensor<float>(buffer.Shape);
            cpuBuffer.ComputeAUCBuffer(yExpected.ToCpuFloat(), yPredicted.ToCpuFloat());
            cpuBuffer.CopyTo(buffer);
        }

        //!D TODO : use GPU for computation
        public override void ComputeAveragePrecisionScoreBuffer(Tensor yExpected, Tensor yPredicted)
        {
            var buffer = this;
            var cpuBuffer = new CpuTensor<float>(buffer.Shape);
            cpuBuffer.ComputeAveragePrecisionScoreBuffer(yExpected.ToCpuFloat(), yPredicted.ToCpuFloat());
            cpuBuffer.CopyTo(buffer);
        }

        protected override void ComputeSparseAccuracyBuffer(Tensor yExpectedSparse, Tensor yPredicted)
        {
            var buffer = this;
            (yExpectedSparse, yPredicted, _) = ReformatTo2DTensorsSparse(yExpectedSparse, yPredicted);
            //yExpectedSparse shape:    (batchSize*timeSteps, 1)
            //yPredicted shape:         (batchSize*timeSteps, numClass)
            Debug.Assert(buffer.Shape.Length == 1);
            Debug.Assert(yExpectedSparse.Count == buffer.Shape[0]);
            int rows = yPredicted.Shape[0];
            var numClass = yPredicted.Shape[1];
            _wrapper.RunKernel("ComputeSparseAccuracy", rows, new object[] { numClass, buffer, yExpectedSparse, yPredicted });
        }

    protected override void SparseCategoricalCrossentropyLossBuffer(Tensor yExpectedSparse, Tensor yPredicted)
    {
        (yExpectedSparse, yPredicted, _) = ReformatTo2DTensorsSparse(yExpectedSparse, yPredicted);
        //yExpectedSparse shape:    (batchSize*timeSteps, 1)
        //yPredicted shape:         (batchSize*timeSteps, numClass)
        Debug.Assert(this.Shape.Length == 1);
        Debug.Assert(yExpectedSparse.Count == this.Shape[0]);
        CallCudaForLossBuffer(yExpectedSparse, yPredicted, EvaluationMetricEnum.SparseCategoricalCrossentropy);
    }

    protected override void CategoricalCrossentropyLossBuffer(Tensor yExpectedOneHot, Tensor yPredicted)
    {
        CallCudaForLossBuffer(yExpectedOneHot, yPredicted, EvaluationMetricEnum.CategoricalCrossentropy);
    }

    protected override void MaeLossBuffer(Tensor yExpected, Tensor yPredicted)
    {
        CallCudaForLossBuffer(yExpected, yPredicted, EvaluationMetricEnum.Mae);
    }

    protected override void MseLossBuffer(Tensor yExpected, Tensor yPredicted)
    {
        CallCudaForLossBuffer(yExpected, yPredicted, EvaluationMetricEnum.Mse);
    }

    protected override void MeanSquaredLogErrorLossBuffer(Tensor yExpected, Tensor yPredicted)
    {
        CallCudaForLossBuffer(yExpected, yPredicted, EvaluationMetricEnum.MeanSquaredLogError);
    }

    protected override void CategoricalCrossentropyWithHierarchyLossBuffer(Tensor yExpected, Tensor yPredicted)
    {
        CallCudaForLossBuffer(yExpected, yPredicted, EvaluationMetricEnum.CategoricalCrossentropyWithHierarchy);
    }

    protected override void BinaryCrossentropyLossBuffer(Tensor yExpected, Tensor yPredicted)
    {
        CallCudaForLossBuffer(yExpected, yPredicted, EvaluationMetricEnum.BinaryCrossentropy);
    }


    protected override void BCEContinuousYLossBuffer(Tensor yExpected, Tensor yPredicted)
    {
        CallCudaForLossBuffer(yExpected, yPredicted, EvaluationMetricEnum.BCEContinuousY);
    }

    protected override void BCEWithFocalLossLossBuffer(Tensor yExpected, Tensor yPredicted, float percentageInTrueClass, float gamma)
    {
        var bceWithFocalLossLossBuffer = this;
        Debug.Assert(yExpected.Shape.Length == 2);
        Debug.Assert(yExpected.SameShape(yPredicted));
        int rows = yExpected.Shape[0];
        int numClass = yExpected.Shape[1];
        Debug.Assert(bceWithFocalLossLossBuffer.SameShape(new[] { rows }));
        _wrapper.RunKernel("BCEWithFocalLossLossBuffer", rows, new object[] { numClass, percentageInTrueClass, gamma, bceWithFocalLossLossBuffer, yExpected, yPredicted });
    }

    protected override void ComputeAccuracyCategoricalCrossentropyWithHierarchyBuffer(Tensor yExpected, Tensor yPredicted)
    {
        var buffer = this;
        Debug.Assert(AreCompatible(new List<Tensor> { buffer, yExpected, yPredicted }));
        Debug.Assert(buffer.Shape.Length == 1);
        Debug.Assert(buffer.Shape[0] == yPredicted.Shape[0]);
        Debug.Assert(yExpected.SameShape(yPredicted));
        Debug.Assert(yExpected.Dimension >= 2);
        int rows = yExpected.Shape[0];
        var cols = yExpected.Shape[1];
        _wrapper.RunKernel("ComputeSingleAccuracyForCategoricalCrossentropyWithHierarchy", rows, new object[] { cols, buffer, yExpected, yPredicted });
    }

    protected override void MseOfLogLossBuffer(Tensor yExpected, Tensor yPredicted, float epsilon)
        {
            var buffer = this;
            int rows = yExpected.Shape[0];
            Debug.Assert(buffer.SameShape(new[] { rows }));
            Debug.Assert(yExpected.SameShape(yPredicted));
            _wrapper.RunKernel("MseOfLogLossBuffer", rows, new object[] { yExpected.MultDim0, buffer, yExpected, yPredicted, epsilon });
        }

        //!D To write properly for GPU
        public override (float f1, float precision, float recall) F1PrecisionRecallMicro(Tensor yExpected, Tensor yPredicted)
        {
            return ToCpuFloat().F1PrecisionRecallMicro(yExpected.ToCpuFloat(), yPredicted.ToCpuFloat());
        }

        public override void CosineSimilarityLossBuffer(Tensor yExpected, Tensor yPredicted, int timeSeriesLength)
        {
            var cosineSimilarityLoss = this;
            Debug.Assert(cosineSimilarityLoss.SameShape(new[] { timeSeriesLength }));
            Debug.Assert(yExpected.SameShape(yPredicted));
            _wrapper.RunKernel("CosineSimilarityLossBuffer", timeSeriesLength, new object[] { yExpected.Count, cosineSimilarityLoss, yExpected, yPredicted });
        }

        public override void HuberLossBuffer(Tensor yExpected, Tensor yPredicted, float huberDelta)
        {
            var huberLoss = this;
            int rows = yExpected.Shape[0];
            Debug.Assert(huberLoss.SameShape(new[] { rows }));
            Debug.Assert(yExpected.SameShape(yPredicted));
            _wrapper.RunKernel("HuberLossBuffer", rows, new object[] { yExpected.MultDim0, huberDelta, huberLoss, yExpected, yPredicted });
        }

        #endregion

        #region Compute of Gradients (for backward propagation)

        protected override void BCEWithFocalLossGradient(Tensor yExpected, Tensor yPredicted, float percentageInTrueClass, float gamma)
        {
            var bceWithFocalLossGradient = this;
            Debug.Assert(yExpected.Shape.Length == 2);
            Debug.Assert(yExpected.SameShape(yPredicted, bceWithFocalLossGradient));
            int rows = yExpected.Shape[0];
            int numClass = yExpected.Shape[1];
            _wrapper.RunKernel("BCEWithFocalLossGradient", rows, new object[] { numClass, percentageInTrueClass, gamma, bceWithFocalLossGradient, yExpected, yPredicted });
        }

        public override void CategoricalCrossentropyWithHierarchyGradient(Tensor yExpected, Tensor yPredicted)
        {
            var categoricalCrossentropyWithHierarchyGradient = this;
            int rows = yExpected.Shape[0];
            var numClass = yExpected.Shape[1];
            categoricalCrossentropyWithHierarchyGradient.ZeroMemory();
            _wrapper.RunKernel("CategoricalCrossentropyWithHierarchyGradient", rows, new object[] { numClass, categoricalCrossentropyWithHierarchyGradient, yExpected, yPredicted });
        }

        public override void CosineSimilarityGradient(Tensor yExpected, Tensor yPredicted, int timeSeriesLength)
        {
            var cosineSimilarityGradient = this;
            Debug.Assert(yExpected.SameShape(yPredicted));
            Debug.Assert(cosineSimilarityGradient.SameShape(yPredicted));
            _wrapper.RunKernel("CosineSimilarityGradient", timeSeriesLength, new object[] { yExpected.Count, cosineSimilarityGradient, yExpected, yPredicted });
        }

        public override void HuberGradient(Tensor yExpected, Tensor yPredicted, float huberDelta)
        {
            var huberGradient = this;
            int rows = yExpected.Shape[0];
            _wrapper.RunKernel("HuberGradient", rows, new object[] { yExpected.MultDim0, huberDelta, huberGradient, yExpected, yPredicted });
        }

        public override void MseGradient(Tensor yExpected, Tensor yPredicted)
        {
            var mseGradient = this;
            int batchSize = yExpected.Shape[0];
            _wrapper.RunKernel("MseGradient", batchSize, new object[] { yExpected.MultDim0, mseGradient, yExpected, yPredicted });
        }

        public override void SparseCategoricalCrossentropyGradient(Tensor yExpectedSparse, Tensor yPredicted)
        {
            (yExpectedSparse, yPredicted, var sparseCategoricalCrossentropyGradient) = ReformatTo2DTensorsSparse(yExpectedSparse, yPredicted, this);
            //yExpectedSparse shape:    (batchSize*timeSteps, 1)
            //yPredicted shape:         (batchSize*timeSteps, numClass)
            int rows = yPredicted.Shape[0];
            int numClass = yPredicted.Shape[^1];
            _wrapper.RunKernel("SparseCategoricalCrossentropyGradient", rows, new object[] { numClass, sparseCategoricalCrossentropyGradient, yExpectedSparse, yPredicted});
        }

        public override void MseOfLogGradient(Tensor yExpected, Tensor yPredicted, float epsilon)
        {
            var mseGradient = this;
            int rows = yExpected.Shape[0];
            _wrapper.RunKernel("MseOfLogGradient", rows, new object[] { yExpected.MultDim0, mseGradient, yExpected, yPredicted, epsilon });
        }

        public override void MaeGradient(Tensor yExpected, Tensor yPredicted)
        {
            var mseGradient = this;
            int rows = yExpected.Shape[0];
            _wrapper.RunKernel("MaeGradient", rows, new object[] { yExpected.MultDim0, mseGradient, yExpected, yPredicted });
        }
        #endregion


        public override void DropoutForward(Tensor y, double dropoutRate, bool isTraining, Random dropoutRandom, [NotNull] Tensor dropoutReservedSpaceForTraining) 
        {
            var x = this;
            if (!isTraining)
            {
                x.CopyTo(y);
                return;
            }
            Debug.Assert(dropoutReservedSpaceForTraining.UseGPU); 
            var xDesc = TensorDesc(x);
            var yDesc = TensorDesc(y);
            cudnnDropoutDescriptor_t dropoutDesc = _wrapper.DropoutDesc(dropoutRate);
            var res = CudnnWrapper.cudnnDropoutForward(CudnnHandle, dropoutDesc, xDesc, x, yDesc, y, dropoutReservedSpaceForTraining.Pointer, dropoutReservedSpaceForTraining.CapacityInBytes);
            CheckStatus(res);
        }

        /// <summary>
        /// this = x
        /// </summary>
        /// <param name="dy"></param>
        /// <param name="dx"></param>
        /// <param name="dropoutRate"></param>
        /// <param name="dropoutReserveSpace"></param>
        public override void DropoutBackward(Tensor dy, Tensor dx, double dropoutRate, [NotNull] Tensor dropoutReserveSpace)
        {
            Debug.Assert(dropoutReserveSpace.UseGPU);
            var dxDesc = TensorDesc(dx);
            var dyDesc = TensorDesc(dy);
            //no need of memory pool : the descriptor has been already created on forward propagation
            var dropoutDesc = _wrapper.DropoutDesc(dropoutRate);
            var res = CudnnWrapper.cudnnDropoutBackward(CudnnHandle, dropoutDesc, dyDesc, dy, dxDesc, dx, dropoutReserveSpace, dropoutReserveSpace.CapacityInBytes);
            CheckStatus(res);
        }
        public override void UpdateAdamOptimizer(double learningRate, double beta1, double beta2, double epsilon,
            double adamW_l2Regularization, Tensor dW, Tensor adam_vW, Tensor adam_sW, int timeStep)
        {
            var W = this;
            var beta1_power = Math.Pow(beta1, timeStep);
            var beta2_power = Math.Pow(beta2, timeStep);
            var multiplicative_factor = learningRate * (Math.Sqrt(1.0 - beta2_power) / (1.0 - beta1_power));
            _wrapper.RunKernel("UpdateAdamOptimizer", Count, new object[] { beta1, beta2, epsilon, adamW_l2Regularization, multiplicative_factor, dW, W, adam_vW, adam_sW });
        }
        public override void UpdateSGDOptimizer(double learningRate, double momentum, bool usenesterov, Tensor dW, Tensor velocity)
        {
            var W = this;
            //velocity[i] = (momentum * velocity[i]) - (dW[i] * learningRate);
            velocity.AddTensor((float)-learningRate, dW, (float)momentum);
            if (usenesterov)
            {
                //W[i] += momentum * velocity[i] - (dW[i] * learningRate);
                W.Update_Adding_Alpha_X((float)momentum, velocity);
                W.Update_Adding_Alpha_X((float)-learningRate, dW);
            }
            else
            {
                //W[i] += velocity[i];
                W.Update_Adding_Alpha_X(1, velocity);
            }
        }

        public override void BatchMatrixMultiplication(Tensor a_3D, bool transposeA, Tensor b_3D, bool transposeB, float alpha, float beta)
        {
            var c_3D = this;
            a_3D.AssertIsNotDisposed();
            b_3D.AssertIsNotDisposed();
            c_3D.AssertIsNotDisposed();
            Debug.Assert(a_3D.Shape.Length == 3);
            Debug.Assert(b_3D.Shape.Length == 3);
            Debug.Assert(c_3D.Shape.Length == 3);
            Debug.Assert(a_3D.Shape[0] == b_3D.Shape[0]);
            Debug.Assert(a_3D.Shape[0] == c_3D.Shape[0]);
            Debug.Assert(AreCompatible(new List<Tensor> { c_3D, a_3D, b_3D }));
            var batchCount = a_3D.Shape[0];
            var aH = a_3D.Shape[1];
            var aW = a_3D.MultDim1;
            var aStride = a_3D.MultDim0;
            var bH = b_3D.Shape[1];
            var bW = b_3D.MultDim1;
            var bStride = b_3D.MultDim0;
            var cStride = c_3D.MultDim0;
            var transLeft = transposeB ? cublasOperation_t.CUBLAS_OP_T : cublasOperation_t.CUBLAS_OP_N;
            var transRight = transposeA ? cublasOperation_t.CUBLAS_OP_T : cublasOperation_t.CUBLAS_OP_N;
            int N = transposeB ? bH : bW; //number of rows of the matrix op(Left) (= number of rows of matrix C)
            int M = transposeA ? aW : aH; //number of columns of the matrix op(Right) (= number of columns of the matrix C)
            int K = transposeB ? bW : bH; //number of columns of the matrix op(Left) (= number of rows of the matrix op(B))
            int ldb = bW; //number of rows of the matrix B (because order = ColumnMajor)
            int lda = aW; //number of rows of the matrix A (because order = ColumnMajor)
            int ldc = N; //number of rows of the matrix C (because order = ColumnMajor)
            //Cuda is column major : we have to compute B*A instead of A*B
            var res = CublasWrapper.cublasSgemmStridedBatched(CublasHandle, transLeft, transRight, N, M, K, ref alpha, b_3D, ldb, bStride, a_3D, lda, aStride, ref beta, this, ldc, cStride, batchCount);
            GPUWrapper.CheckStatus(res);
        }
        
        public override void Dot(Tensor a, bool transposeA, Tensor b, bool transposeB, float alpha, float beta)
        {
            AssertIsNotDisposed();
            a.AssertIsNotDisposed();
            b.AssertIsNotDisposed();
            Debug.Assert(AreCompatible(new List<Tensor> { this, a, b }));
            Debug.Assert(b.Dimension >= 2);
            Debug.Assert(a.Dimension >= 2);
            Debug.Assert(Dimension >= 2);
            var bH = b.Shape[0];
            var bW = b.MultDim0;
            var aH = a.Shape[0];
            var aW = a.MultDim0;
            var transLeft = transposeB ? cublasOperation_t.CUBLAS_OP_T : cublasOperation_t.CUBLAS_OP_N;
            var transRight = transposeA ? cublasOperation_t.CUBLAS_OP_T : cublasOperation_t.CUBLAS_OP_N;
            int N = transposeB ? bH : bW; //number of rows of the matrix op(Left) (= number of rows of matrix C)
            int M = transposeA ? aW : aH; //number of columns of the matrix op(Right) (= number of columns of the matrix C)
            int K = transposeB ? bW : bH; //number of columns of the matrix op(Left) (= number of rows of the matrix op(B))
            int ldb = bW; //number of rows of the matrix B (because order = ColumnMajor)
            int lda = aW; //number of rows of the matrix y (because order = ColumnMajor)
            int ldc = N; //number of rows of the matrix C (because order = ColumnMajor)
            //Cuda is column major : we have to compute B*y instead of y*B
            var res = CublasWrapper.cublasSgemm_v2(CublasHandle, transLeft, transRight, N, M, K, ref alpha, b, ldb, a, lda, ref beta, this, ldc);

            //The lib may return CUBLAS_STATUS_INTERNAL_ERROR (in version 10.0) in some cases => ignored (see Non Reg test that reproduces the issue)
            if (res == cublasStatus_t.CUBLAS_STATUS_INTERNAL_ERROR)
            {
                return;
            }

            GPUWrapper.CheckStatus(res);
        }

        public override void SetAllElementsAboveMainDiagonal(float valueForElementsAboveMainDiagonal)
        {
            Debug.Assert(Shape.Length == 2 || Shape.Length == 3);
            (int matrices_count, int rows_by_matrix, int cols_by_matrix) = Shape.Length == 3
                ? (Shape[0], Shape[1], Shape[2])
                : (1, Shape[0], Shape[1]);
            _wrapper.RunKernel("SetAllElementsAboveMainDiagonal", matrices_count* rows_by_matrix, new object[] { rows_by_matrix, cols_by_matrix, valueForElementsAboveMainDiagonal, this });
        }

        public override void SetIdentityMatrix()
        {
            Debug.Assert(Shape.Length == 2);
            Debug.Assert(Shape[0] == Shape[1]);
            int nbRows = Shape[0];
            ZeroMemory();
            _wrapper.RunKernel("Set1InMainDiagonal", nbRows, new object[] { nbRows, this });
        }


        public override void SetToZeroAllElementsBelowMainDiagonal()
        {
            Debug.Assert(Shape.Length == 2);
            int nbRows = Shape[0];
            int nbColumns = Shape[1];
            _wrapper.RunKernel("SetToZeroAllElementsBelowMainDiagonal", nbRows, new object[] { nbColumns , this });
        }

        public override void Transpose(Tensor transposed)
        {
            AssertIsNotDisposed();
            transposed.AssertIsNotDisposed();
            Debug.Assert(AreCompatible(new List<Tensor> { this, transposed }));
            Debug.Assert(Dimension == 2);
            if (transposed.CapacityInBytes < ReallyNeededMemoryInBytesForShape(Shape))
            {
                throw new ArgumentException("Can't transpose to tensor: not enough capacity");
            }
            transposed.ReshapeInPlace(new[]{ Shape[1] , Shape[0] });
            Debug.Assert(transposed.Dimension == Dimension);
            Debug.Assert(transposed.Shape[0] == Shape[1]);
            Debug.Assert(transposed.Shape[1] == Shape[0]);

            // because cublas is column major:
            //  (*) the 'this' tensor (=A) is seen (from cublas) as a matrix of shape (Shape[1], Shape[0]) = (lda, M)
            //      =>  lda = Shape[1]
            //          M = Shape[0]
            //  (*) the transpose of the 'this' tensor (=C) is seen as a matrix of shape (Shape[0], Shape[1]) = (M, N) = (ldc, N)