using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using SharpNet.Data;
using SharpNet.GPU;
using SharpNet.Hyperparameters;
using SharpNet.Layers;
using SharpNet.MathTools;
using static SharpNet.GPU.GPUWrapper;

namespace SharpNet.CPU
{
    public unsafe class CpuTensor<T> : Tensor
    {
        #region fields
        public Memory<T> Content { get; private set; }
        /// <summary>
        /// used only if the tensor is NOT the owner of the memory
        /// </summary>
        private readonly IntPtr _ptrToOwnerPinnedMemory;
        /// <summary>
        /// used only if the tensor is the owner of the memory
        /// </summary>
        private HostPinnedMemory<T> _hostPinnedMemory;
        #endregion

        #region constructors
        public CpuTensor(int[] shape, T[] data, int typeSize) : base(shape, typeSize, false)
        {
            Content = data ?? new T[Count];
            CapacityInBytes = (ulong)(Content.Length * TypeSize);
            _ptrToOwnerPinnedMemory = IntPtr.Zero;
        }
        public CpuTensor(int[] shape, T[] data = null) : this(shape, data, typeof(T).IsValueType ?Marshal.SizeOf(typeof(T)) : IntPtr.Size)
        {
        }
        private CpuTensor(int[] shape, CpuTensor<T> memoryOwner, int startIndex) : base(shape, memoryOwner.TypeSize, false)
        {
            Content = memoryOwner.Content.Slice(startIndex, Utils.Product(shape));
            CapacityInBytes = (ulong)(Content.Length * TypeSize);
            _ptrToOwnerPinnedMemory = memoryOwner.Pointer + TypeSize * startIndex;
        }
        public static CpuTensor<T> New(T[] content, int columns)
        {
            if (content == null || content.Length == 0)
            {
                return null;
            }
            Debug.Assert(content.Length % columns == 0);
            int rows = content.Length / columns;
            return new CpuTensor<T>(new[] { rows, columns }, content);
        }
        #endregion

        /// <summary>
        /// pointer to (pinned) host memory (in CPU)
        /// </summary>
        public override IntPtr Pointer
        {
            get
            {
                if (!IsOwnerOfMemory)
                {
                    //the memory owner has its memory already pinned
                    Debug.Assert(_ptrToOwnerPinnedMemory != IntPtr.Zero);
                    return _ptrToOwnerPinnedMemory;
                }
                Debug.Assert(_ptrToOwnerPinnedMemory == IntPtr.Zero);
                if (_hostPinnedMemory == null)
                {
                    _hostPinnedMemory = new HostPinnedMemory<T>(Content);
                }
                return _hostPinnedMemory.Pointer;
            }
        }

        /// <summary>
        /// true if the tensor memory is currently pinned
        /// </summary>
        private bool HasPinnedMemory => !IsOwnerOfMemory || _hostPinnedMemory != null;

        public override void WordEmbeddingForwardPropagation(Tensor x, Tensor wordEmbedding, int xIndexInLastDimensionToUse, int yIndexInLastDimensionToUse, int copyCountBeforeIndex, int copyCountAfterIndex)
        {
            var y = this;
            Debug.Assert(wordEmbedding.Shape.Length == 2);
            Debug.Assert(x.Shape[0] == y.Shape[0]); //same batchSize
            Debug.Assert(x.Shape[1] == y.Shape[1]); //same timeSteps
            Debug.Assert(x.Shape.Length == 3);
            Debug.Assert(y.Shape.Length == 3);

            Debug.Assert(xIndexInLastDimensionToUse>=0);
            Debug.Assert(yIndexInLastDimensionToUse>=0);
            var timeSteps = x.Shape[1];
            var embeddingDim = wordEmbedding.Shape[1];

            void ProcessBatch(int batchIndex)
            {
                var xSpan = x.AsReadonlyFloatCpuSpan;
                var ySpan = y.AsFloatCpuSpan;
                var wordEmbeddingSpan = wordEmbedding.AsReadonlyFloatCpuSpan;

                for (int timeStep = 0; timeStep < timeSteps; ++timeStep)
                {
                    int xTimeStepIndex = x.Idx(batchIndex, timeStep, xIndexInLastDimensionToUse);
                    int yTimeStepIndex = y.Idx(batchIndex, timeStep, yIndexInLastDimensionToUse);


                    //for the current timeStep, we copy the elements from 'x' to 'y' before 'indexInLastDimensionToUse'
                    //int xElementsBeforeEmbeddingIndex = indexInLastDimensionToUse;
                    if (copyCountBeforeIndex > 0)
                    {
                        //we copy 'xElementsBeforeEmbeddingIndex' elements before index 'indexInLastDimensionToUse'
                        xSpan.Slice(xTimeStepIndex- copyCountBeforeIndex, copyCountBeforeIndex).CopyTo(ySpan.Slice(yTimeStepIndex- copyCountBeforeIndex, copyCountBeforeIndex));
                    }

                    int wordIndex = (int)(xSpan[xTimeStepIndex] + 0.1);
                    wordEmbeddingSpan.Slice(wordIndex*embeddingDim, embeddingDim).CopyTo(ySpan.Slice(yTimeStepIndex, embeddingDim));

                    //for the current timeStep, we copy the elements from 'x' to 'y' after 'indexInLastDimensionToUse'
                    //int xElementsAfterEmbeddingIndex = inputSize - indexInLastDimensionToUse - 1;
                    if (copyCountAfterIndex > 0)
                    {
                        //we copy the 'xElementsAfterEmbeddingIndex' elements after index 'indexInLastDimensionToUse'
                        xSpan.Slice(xTimeStepIndex+ 1, copyCountAfterIndex).CopyTo(ySpan.Slice(yTimeStepIndex+ embeddingDim, copyCountAfterIndex));
                    }
                }
            }
            Parallel.For(0, x.Shape[0], ProcessBatch);
        }

        public override void WordEmbeddingBackwardPropagation(/*in*/ Tensor x, /*out*/ Tensor dx, /*in*/ Tensor dy, int dxIndexInLastDimensionToUse, int dyIndexInLastDimensionToUse, int copyCountBeforeIndex, int copyCountAfterIndex)
        {
            var dW = this;

            Debug.Assert(dW.Shape.Length == 2);
            Debug.Assert(x.Shape.Length == 3);
            Debug.Assert(dy.Shape.Length == 3);
            Debug.Assert(x.Shape[0] == dy.Shape[0]); //same batchSize
            Debug.Assert(x.Shape[1] == dy.Shape[1]); //same timeSteps
            Debug.Assert(dxIndexInLastDimensionToUse >= 0);
            Debug.Assert(dyIndexInLastDimensionToUse >= 0);

            var xCpu = (CpuTensor<float>)x;
            var dyCpu = (CpuTensor<float>)dy;

            dW.ZeroMemory();
            var batchSize = dy.Shape[0];
            var timeSteps = x.Shape[1];
            var embeddingDim = dW.Shape[1];

            var xSpan = x.AsReadonlyFloatCpuSpan;
            var dxSpan = dx.AsFloatCpuSpan;
            var dWSpan = dW.AsFloatCpuSpan;
            var dySpan = dy.AsReadonlyFloatCpuSpan;

            for (int batchIndex = 0; batchIndex < batchSize; ++batchIndex)
            {
                for (int timeStep = 0; timeStep < timeSteps; ++timeStep)
                {
                    //we initialize 'dw' for the current batchIndex & timeStep
                    int wordIndex = (int)(xSpan[xCpu.Idx(batchIndex, timeStep, dxIndexInLastDimensionToUse)] + 0.1);
                    int indexInDw = dW.Idx(wordIndex, 0);
                    int indexIndY = dyCpu.Idx(batchIndex, timeStep, dyIndexInLastDimensionToUse);
                    for (int embeddingId = 0; embeddingId < embeddingDim; ++embeddingId)
                    {
                        dWSpan[indexInDw] += dySpan[indexIndY];
                        ++indexInDw;
                        ++indexIndY;
                    }


                    int dyTimeStepIndex = dy.Idx(batchIndex, timeStep, dyIndexInLastDimensionToUse);
                    int dxTimeStepIndex = dx.Idx(batchIndex, timeStep, dxIndexInLastDimensionToUse);

                    //we initialize 'dx' for the current batchIndex & timeStep
                    //for the current timeStep, we copy the elements from 'dy' to 'dx' before 'indexInLastDimensionToUse'
                    //int dyElementsBeforeEmbeddingIndex = prevXIndexInLastDimensionToUse==-1?xIndexInLastDimensionToUse:(xIndexInLastDimensionToUse-prevXIndexInLastDimensionToUse-1);
                    if (copyCountBeforeIndex > 0)
                    {
                        //we copy 'xElementsBeforeEmbeddingIndex' elements before index 'xIndexInLastDimensionToUse'
                        dySpan.Slice(dyTimeStepIndex- copyCountBeforeIndex, copyCountBeforeIndex).CopyTo(dxSpan.Slice(dxTimeStepIndex- copyCountBeforeIndex, copyCountBeforeIndex));
                    }
                    dxSpan[dxTimeStepIndex] = 0;
                    //for the current timeStep, we copy the elements from 'dy' to 'dx' after 'xIndexInLastDimensionToUse'
                    //int dyElementsAfterEmbeddingIndex = nextXIndexInLastDimensionToUse==-1 ? (inputSize - xIndexInLastDimensionToUse - 1):(nextXIndexInLastDimensionToUse-xIndexInLastDimensionToUse-1);
                    if (copyCountAfterIndex > 0)
                    {
                        //we copy the 'xElementsAfterEmbeddingIndex' elements after index 'indexInLastDimensionToUse'
                        dySpan.Slice(dyTimeStepIndex + embeddingDim, copyCountAfterIndex).CopyTo(dxSpan.Slice(dxTimeStepIndex + 1, copyCountAfterIndex));
                    }
                }
            }
        }

        /// <summary>
        /// resize the current Cpu tensor to a different shape (both bigger or smaller)
        /// </summary>
        /// <param name="newShape"></param>
        public override void ReshapeInPlace(params int[] newShape)
        {
            newShape = FillMinusOneIfAny(Shape, newShape);
            if (SameShape(newShape))
            {
                return;
            }
            else if (HasEnoughCapacityForTensor(newShape))
            {
                //smaller shape
                Shape = newShape;
            }
            else
            {
                //bigger shape
                if (!IsOwnerOfMemory)
                {
                    throw new ArgumentException("must be memory owner to increase memory associated with the 'this' Tensor");
                }
                _hostPinnedMemory?.Dispose();
                _hostPinnedMemory = null;
                Content = new T[Utils.Product(newShape)];
                CapacityInBytes = (ulong)(Content.Length * TypeSize);
                Shape = newShape;
            }
            RecomputeMultDim();
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
                return new CpuTensor<T>(newShape, this, 0);
            }
            //bigger shape : we do not have enough space to store it
            throw new ArgumentException("CapacityInBytes: " + CapacityInBytes + " but need memory  " + ReallyNeededMemoryInBytesForShape(newShape) + " for " + this);
        }


        public T this[int i]
        {
            get => ReadonlyContent[i];
            set => SpanContent[i] = value;
        }

        public override void Switch_First_2_axis(Tensor target)
        {
            Debug.Assert(target.Count == Count);
            Debug.Assert(Shape.Length >= 2);
            int aLength = Shape[0];
            int bLength = Shape[1];
            int cLength = MultDim1;
            int multDim0 = bLength * cLength;
            var srcContent = AsReadonlyFloatCpuSpan;
            var targetContent = target.AsFloatCpuSpan;

            for (int idx_src = 0; idx_src < Count; ++idx_src)
            {
                int a_src = idx_src / multDim0;
                int tmp = idx_src % multDim0;
                int b_src = tmp / cLength;
                int c_src = tmp % cLength;
                int idx_target = b_src * aLength * cLength + a_src * cLength + c_src;
                targetContent[idx_target] = srcContent[idx_src];
            }

            var targetShape = (int[]) Shape.Clone();
            targetShape[0] = bLength;
            targetShape[1] = aLength;
            target.ReshapeInPlace(targetShape);
        }

        public override void SwitchSecondAndThirdDimension(Tensor target)
        {
            Debug.Assert(Shape.Length == 3 || (Shape.Length==4&&Shape[3] == 1));
            var srcContent = AsReadonlyFloatCpuSpan;
            var targetContent = target.AsFloatCpuSpan;
            for (int n = 0; n < Shape[0]; ++n)
            {
                for (int c = 0; c < Shape[1]; ++c)
                {
                    for (int h = 0; h < Shape[2]; ++h)
                    {
                        targetContent[target.Idx(n, h, c)] = srcContent[Idx(n, c, h)];
                    }
                }
            }
        }

        public override void TransposeSecondAndThirdDimension(Tensor target)
        {
            Debug.Assert(Shape.Length >= 3);
            var targetShape = (int[])Shape.Clone();
            (targetShape[1], targetShape[2]) = (targetShape[2], targetShape[1]);
            target.ReshapeInPlace(targetShape);
            var src = Reshape(Shape[0], Shape[1], Shape[2], -1);
            var srcSpan = src.AsFloatCpuSpan;
            var targetSpan = target.AsFloatCpuSpan;

            int A = Shape[0];
            int B = Shape[1];
            int C = Shape[2];
            int D = Count/(A*B*C);
            for (int a=0;a<A;++a)
            for(int b=0;b<B;++b)
            for(int c=0;c<C;++c)
            for (int d = 0; d < D; ++d)
            {
                targetSpan[target.Idx(a, c, b, d)] = srcSpan[src.Idx(a, b, c, d)];
            }
        }
        

        public override Tensor ChangeAxis(int[] targetAxisToSrcAxis)
        {
            Debug.Assert(targetAxisToSrcAxis.Length == Dimension);
            Debug.Assert(targetAxisToSrcAxis.Min() == 0);
            Debug.Assert(targetAxisToSrcAxis.Max() == Dimension-1);

            var srcAxisToTargetAxis = new int[Dimension];
            for (int targetAxis = 0; targetAxis < Dimension; ++targetAxis)
            {
                srcAxisToTargetAxis[targetAxisToSrcAxis[targetAxis]] = targetAxis;
            }

            var targetShape = new int[Dimension];
            for (int targetAxis = 0; targetAxis < Dimension; ++targetAxis)
            {
                targetShape[targetAxis] = Shape[targetAxisToSrcAxis[targetAxis]];
            }

            var result = new CpuTensor<T>(targetShape);

            var idxInSrcAxis = new int[Dimension];
            var srcMultDim = new int[Dimension];
            var idxInTargetAxis =  new int[Dimension];
            var targetMultDim = new int[Dimension];
            srcMultDim[^1] = 1;
            targetMultDim[^1] = 1;
            for (int dim = Dimension - 2; dim >= 0; --dim)
            {
                srcMultDim[dim] = Shape[dim + 1] * srcMultDim[dim + 1];
                targetMultDim[dim] = targetShape[dim + 1] * targetMultDim[dim + 1];
            }

            void ProcessDimension(int axisSrc)
            {
                for (int idxInAxis = 0; idxInAxis < Shape[axisSrc]; ++idxInAxis)
                {
                    idxInTargetAxis[srcAxisToTargetAxis[axisSrc]] = idxInAxis;
                    idxInSrcAxis[axisSrc] = idxInAxis;
                    if (axisSrc == Dimension - 1)
                    {
                        int targetIdx = 0;
                        int srcIdx = 0;
                        for (int axis = 0; axis < Dimension; ++axis)
                        {
                            srcIdx += idxInSrcAxis[axis] * srcMultDim[axis];
                            targetIdx += idxInTargetAxis[srcAxisToTargetAxis[axis]] * targetMultDim[srcAxisToTargetAxis[axis]];
                        }
                        result[targetIdx] = this[srcIdx];
                    }
                    else
                    {
                        ProcessDimension(axisSrc + 1);
                    }
                }
            }

            ProcessDimension(0);
            return result;
        }

        public override bool IsOwnerOfMemory => _ptrToOwnerPinnedMemory == IntPtr.Zero;
        public ReadOnlySpan<T> ReadonlyContent => Content.Slice(0, Count).Span;
        public Span<T> SpanContent => Content.Slice(0, Count).Span;
        public T Get(int n, int c)
        {
            return this[Idx(n, c)];
        }
        public T Get(int n, int c, int h)
        {
            Debug.Assert(Dimension == 3);
            return this[Idx(n, c, h)];
        }
        public T Get(int n, int c, int h, int w)
        {
            Debug.Assert(Dimension == 4);
            return this[Idx(n, c, h, w)];
        }
        public void Set(int n, int c, T t)
        {
            Debug.Assert(Dimension == 2);
            this[Idx(n, c)] = t;
        }
        // ReSharper disable once MemberCanBeProtected.Global
        public void Set(int n, int c, int h, T t)
        {
            Debug.Assert(Dimension == 3);
            this[Idx(n, c, h)] = t;
        }
        public void Set(int n, int c, int h, int w, T t)
        {
            Debug.Assert(Dimension == 4);
            this[Idx(n, c, h, w)] = t;
        }
        public void Map(Func<T, T> func, CpuTensor<T> result)
        {
            Debug.Assert(Count == result.Count);
            for (int i = 0; i < Count; ++i)
            {
                result[i] = func(this[i]);
            }
        }


        /// <summary>
        /// Transform the 'this' tensor into another tensor by transforming:
        ///   each element 'val' of the  'this' tensor at position (m,c,h,w)
        /// into
        ///   the value returned by the method func(m,c,val)
        /// </summary>
        /// <typeparam name="TY"></typeparam>
        /// <param name="func"></param>
        /// <returns></returns>
        public CpuTensor<TY> Select<TY>(Func<int,int, T, TY> func) where TY : struct
        {
            var result = new CpuTensor<TY>(Shape);
            Debug.Assert(SameShape(result));
            var content = ReadonlyContent;
            for (int m = 0; m < Shape[0]; ++m)
            {
                for (int c = 0; c < Shape[1]; ++c)
                {
                    int startIdx = Idx(m, c);
                    for (int idx = startIdx; idx < (startIdx + MultDim1); ++idx)
                    {
                        result[idx] = func(m, c, content[idx]);
                    }
                }
            }
            return result;
        }

        public CpuTensor<TY> Select<TY>(Func<T, TY> func) where TY : struct
        {
            var result = new CpuTensor<TY>(Shape);
            Debug.Assert(SameShape(result));
            var content = ReadonlyContent;
            var resultSpan = result.SpanContent;
            for (int i = 0; i < Count; ++i)
            {
                resultSpan[i] = func(content[i]);
            }
            return result;
        }

        #region Tensor implementation
        public override void UpdateSGDOptimizer(double learningRate, double momentum, bool usenesterov, Tensor dW, Tensor velocity)
        {
            var W = this;
            var wContent = W.AsFloatCpuSpan;
            var dWContent = dW.AsFloatCpuSpan;
            var velocityContent = velocity.AsFloatCpuSpan;
            var learningRateFloat = (float) learningRate;
            var momentumFloat = (float)momentum;
            for (int i = 0; i < W.Count; ++i)
            {
                velocityContent[i] = (momentumFloat * velocityContent[i]) - (dWContent[i] * learningRateFloat);
                if (usenesterov)
                {
                    wContent[i] += momentumFloat * velocityContent[i] - (dWContent[i] * learningRateFloat);
                }
                else
                {
                    wContent[i] += velocityContent[i];
                }
            }
        }
        public override void BatchNormalization(Tensor y, Tensor scale, Tensor bias, double exponentialAverageSmoothingFactor, Tensor runningInputMean, Tensor runningInputVariance, cudnnBatchNormMode_t mode, double epsilon, Tensor meanBuffer, Tensor invertOfUnbiasedVolatilityBuffer, bool isTraining)
        {
            var x = this;
            Debug.Assert(AreCompatible(new List<Tensor>{x,y,scale,bias,runningInputMean,runningInputVariance,meanBuffer,invertOfUnbiasedVolatilityBuffer}));
            Debug.Assert(x.SameShape(y));
            Debug.Assert(scale.SameShape(bias, runningInputMean, runningInputVariance, meanBuffer, invertOfUnbiasedVolatilityBuffer));
            bool is1C11Shape = bias.Count == bias.Shape[1];
            var meanDivider = Count / bias.Count;  // = batchSize if (1,C,H,W) , and = batchSize*H*W if (1,C,1,1)

            
            var batchSize = x.Shape[0];

            var xContent = x.AsFloatCpuSpan;
            var yContent = y.AsFloatCpuSpan;
            var scaleContent = scale.AsFloatCpuSpan;
            var biasContent = bias.AsFloatCpuSpan;

            // 'meanBuffer' & 'invertOfUnbiasedVolatilityBuffer' will only be used when isTraining = true
            var meanContent = isTraining?meanBuffer.AsFloatCpuSpan:null;
            var invertOfUnbiasedVolatility = isTraining ? invertOfUnbiasedVolatilityBuffer.AsFloatCpuSpan:null;

            var runningInputMeanContent = runningInputMean.AsFloatCpuSpan;
            var runningInputVarianceContent = runningInputVariance.AsFloatCpuSpan;


            if (isTraining)
            {
                //'invertOfUnbiasedVolatilityBuffer' will temporary store the variance of the input 
                Compute_Column_Mean_Variance(meanBuffer, invertOfUnbiasedVolatilityBuffer);
                var variance = invertOfUnbiasedVolatilityBuffer.AsFloatCpuSpan;

                //we need to update 'runningInputMean' and 'runningInputVariance'
                for (int j = 0; j < runningInputVariance.Count; ++j)
                {
                    runningInputMeanContent[j] = (float) (meanContent[j] * exponentialAverageSmoothingFactor + runningInputMeanContent[j] * (1 - exponentialAverageSmoothingFactor));
                    runningInputVarianceContent[j] = (float)(variance[j] * exponentialAverageSmoothingFactor + runningInputVarianceContent[j] * (1 - exponentialAverageSmoothingFactor));
                }

                //we update 'invertOfUnbiasedVolatilityBuffer' so that it stores the invert of the unbiased volatility of the input
                for (int j = 0; j < invertOfUnbiasedVolatilityBuffer.Count; ++j)
                {
                    invertOfUnbiasedVolatility[j] = (float)(1.0 / Math.Sqrt(((meanDivider - 1) * variance[j]) / meanDivider + epsilon));
                }
            }

            int idx = 0;
            for (int n = 0; n < batchSize; ++n)
            {
                for (int j = 0; j < MultDim0; ++j)
                {
                    int scaleIndex = is1C11Shape ? (j / MultDim1) : j;
                    var xTarget = isTraining
                        ? ((xContent[idx] - meanContent[scaleIndex]) * invertOfUnbiasedVolatility[scaleIndex])
                        : (float)((xContent[idx] - runningInputMeanContent[scaleIndex]) / Math.Sqrt(runningInputVarianceContent[scaleIndex] + epsilon));
                    yContent[idx++] = scaleContent[scaleIndex] * xTarget + biasContent[scaleIndex];
                }
            }
        }
        public override void BatchNormalizationBackward(Tensor dy, Tensor dx, Tensor scale, Tensor scaleGradient, Tensor biasGradient, cudnnBatchNormMode_t mode, double epsilon, Tensor meanBuffer, Tensor invertOfUnbiasedVolatilityBuffer)
        {
            var x = this;
            var batchSize = x.Shape[0];
            Debug.Assert(AreCompatible(new List<Tensor> {x, dy, dx, scale, scaleGradient, biasGradient, meanBuffer, invertOfUnbiasedVolatilityBuffer}));
            Debug.Assert(x.SameShape(dy, dx));
            Debug.Assert(scale.SameShape(scaleGradient, biasGradient, meanBuffer, invertOfUnbiasedVolatilityBuffer));
            bool is1C11Shape = scale.Count == scale.Shape[1];
            var meanDivider = Count / scale.Count;  // = batchSize if (1,C,H,W) , and = batchSize*H*W if (1,C,1,1)
            scaleGradient.ZeroMemory();
            dx?.ZeroMemory();

            //we compute resultBnBiasDiff
            dy.AsFloatCpu.ComputeSumByColumn(biasGradient);
            //we compute resultBnScaleDiff
            var xContent = x.AsFloatCpuSpan;
            var dyContent = dy.AsFloatCpuSpan;
            Span<float> dxContent = null;
            if (dx != null)
            {
                dxContent = dx.AsFloatCpuSpan;
            }

            var biasGradientContent = biasGradient.AsFloatCpuSpan;
            var scaleGradientContent = scaleGradient.AsFloatCpuSpan;
            var scaleContent = scale.AsFloatCpuSpan;
            var meanBufferContent = meanBuffer.AsFloatCpuSpan;
            var invertOfUnbiasedVolatility = invertOfUnbiasedVolatilityBuffer.AsFloatCpuSpan;
            for (int j = 0; j < MultDim0; ++j)
            {
                int meanIndex = is1C11Shape ? (j / MultDim1) : j;
                double result = 0.0;
                for (int n = 0; n < batchSize; ++n)
                {

                    int idx = n * MultDim0 + j;
                    result += dyContent[idx] * (xContent[idx] - meanBufferContent[meanIndex]);
                }
                scaleGradientContent[meanIndex] += (float) (result * invertOfUnbiasedVolatility[meanIndex]);
            }
            //we compute dx
            for (int i = 0; i < batchSize; ++i)
            {
                for (int j = 0; j < MultDim0; ++j)
                {
                    int meanIndex = is1C11Shape ? (j / MultDim1) : j;
                    int idx = i * MultDim0 + j;
                    double result = meanDivider * dyContent[idx] - biasGradientContent[meanIndex] - scaleGradientContent[meanIndex] * invertOfUnbiasedVolatility[meanIndex] * (xContent[idx] - meanBufferContent[meanIndex]);
                    if (dxContent != null)
                    {
                        dxContent[idx] += (float) ((scaleContent[meanIndex] * invertOfUnbiasedVolatility[meanIndex] * result) /meanDivider);
                    }
                }
            }
        }


        public override void StandardizeInPlace(Tensor row_mean, Tensor row_variance, int axis, float epsilon)
        {
            var x = this;
            Debug.Assert(AreCompatible(new List<Tensor> { this, row_mean, row_variance}));
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
                void ProcessRow(int row)
                {
                    var xSpan = x.AsFloatCpuSpan;
                    var row_mean_value = row_mean.AsFloatCpuSpan[row];
                    var row_variance_value = row_variance.AsFloatCpuSpan[row];
                    int startIndex = row * cols;
                    int endIndex = startIndex + cols - 1;
                    for (int i = startIndex; i <= endIndex; ++i)
                    {
                        xSpan[i] = (xSpan[i] - row_mean_value) / MathF.Sqrt(row_variance_value + epsilon);
                    }
                }
                Parallel.For(0, rows, ProcessRow);
                return;
            }
            throw new NotSupportedException("Only axis=1 is supported");
        }

        public override void StandardizeRowsInPlaceBroadcastGammasBetas(Tensor row_mean, Tensor row_variance, float epsilon, Tensor col_gammas, Tensor col_betas)
        {
            var x = this;
            Debug.Assert(AreCompatible(new List<Tensor> { this, row_mean, row_variance }));
            Debug.Assert(row_mean.SameShape(row_variance));
            //we'll standardize each row
            int rows = row_mean.Count;
            if (x.Count % rows != 0)
            {
                throw new ArgumentException("The number of elements in the tensor must be a multiple of the number of rows");
            }
            int cols = x.Count / rows;
            void ProcessRow(int row)
            {
                var xSpan = x.AsFloatCpuSpan;
                var row_mean_value = row_mean.AsFloatCpuSpan[row];
                var row_variance_value = row_variance.AsFloatCpuSpan[row];
                var col_gammas_span = col_gammas.AsReadonlyFloatCpuSpan;
                var col_betas_span = col_betas.AsReadonlyFloatCpuSpan;

                int startIndex = row * cols;
                int endIndex = startIndex + cols - 1;
                int col = 0;
                for (int i = startIndex; i <= endIndex; ++i)
                {
                    xSpan[i] = (xSpan[i] - row_mean_value) / MathF.Sqrt(row_variance_value + epsilon);
                    xSpan[i] = col_gammas_span[col]* xSpan[i] + col_betas_span[col];
                    ++col;
                }
            }
            Parallel.For(0, rows, ProcessRow);
        }

        public override void numpy_sum(Tensor sum_result, int axis)
        {
            var a = this;
            Debug.Assert(AreCompatible(new List<Tensor> { a, sum_result}));
            sum_result.ZeroMemory();
            var sum_result_as_span = sum_result.AsFloatCpuSpan;
            var aSpan = a.AsReadonlyFloatCpuSpan;
            if (axis == 1)
            {
                int rows = sum_result.Count;
                if (a.Count % rows != 0)
                {
                    throw new ArgumentException("x.Count % rows != 0");
                }
                int cols = a.Count / rows;
                for (int row = 0; row < rows; ++row)
                {
                    var row_sum = 0.0f;
                    for (int col = 0; col < cols; ++col)
                    {
                        row_sum += aSpan[row * cols + col];
                    }
                    sum_result_as_span[row] = row_sum;
                }

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
                for (int row = 0; row < rows; ++row)
                {
                    for (int col = 0; col < cols; ++col)
                    {
                        sum_result_as_span[col] += aSpan[row * cols + col];
                    }
                }
                return;
            }
            throw new ArgumentException("axis != 0 && axis != 1");
        }
        

        public override void Compute_Row_Mean_Variance(Tensor row_mean, Tensor row_variance, bool unbiasedVariance)
        {
            var x = this;
            Debug.Assert(AreCompatible(new List<Tensor> { this, row_mean, row_variance}));
            Debug.Assert(row_mean.SameShape(row_variance));
            int rows = row_mean.Count;
            if (x.Count % rows != 0)
            {
                throw new ArgumentException("x.Count % rows != 0");
            }
            int cols = x.Count / rows;

            void ProcessBlock(int rowId)
            {
                var xSpan = x.AsFloatCpuSpan;
                int startIndex = rowId * cols;
                int endIndex = startIndex + cols - 1;
                double sum = 0.0;
                double sumSquare = 0.0;
                for (int i = startIndex; i <= endIndex; ++i)
                {
                    double xValue = xSpan[i];
                    sum += xValue;
                    sumSquare += xValue * xValue;
                }
                var row_mean_value = (sum / cols);
                var divider = unbiasedVariance ? (cols - 1) : cols;
                var row_variance_value = Math.Abs(sumSquare - cols * row_mean_value * row_mean_value) / divider;
                row_mean.AsFloatCpuSpan[rowId] = (float)row_mean_value;
                row_variance.AsFloatCpuSpan[rowId] = (float)row_variance_value;
            }
            Parallel.For(0, rows, ProcessBlock);
        }

        public override void LayerNormalizationBackward(Tensor dy, Tensor dx, Tensor col_gammas, Tensor row_mean, Tensor row_variance, float epsilon, Tensor dmean, Tensor dvariance)
        {
            var x = this;
            Debug.Assert(AreCompatible(new List<Tensor> { x, dy, dx, col_gammas, row_mean, row_variance}));
            Debug.Assert(x.SameShape(dy, dx));
            Debug.Assert(row_mean.SameShape(row_variance));
            int rows = row_mean.Count;
            int cols= col_gammas.Count;
            if (x.Count != rows * cols)
            {
                throw new ArgumentException("x.Count != rows * cols");
            }

            void ComputeDxForRow(int row)
            {
                var gammaSpan = col_gammas.AsFloatCpuSpan;
                var mean_row = row_mean.AsFloatCpuSpan[row];
                var variance_row = row_variance.AsFloatCpuSpan[row];
                var volatility_row = MathF.Sqrt(variance_row + epsilon);
                var x_row = x.AsFloatCpu.SpanSlice(row * cols, cols);
                var dy_row = dy.AsFloatCpu.SpanSlice(row * cols, cols);
                var dvariance_row = 0f;
                var dmean_row = 0f;
                for (int col = 0; col < cols; ++col)
                {
                    var tmp0 = (dy_row[col] * gammaSpan[col]);
                    dvariance_row += tmp0 * (x_row[col]-mean_row);
                    dmean_row -= tmp0;
                }
                dvariance_row *= (-0.5f * MathF.Pow(variance_row + epsilon, -1.5f));
                dmean_row /= volatility_row;
                for (int col = 0; col < cols; ++col)
                {
                    dmean_row += dvariance_row*(x_row[col] -mean_row) * (-2f/cols);
                }
                var dx_row = dx.AsFloatCpu.SpanSlice(row * cols, cols);
                for (int col = 0; col < cols; ++col)
                {
                    dx_row[col] = (dy_row[col] * gammaSpan[col]) /volatility_row
                                + dvariance_row * (2f / cols) * (x_row[col] - mean_row)
                                + dmean_row / cols;
                }
            }
            Parallel.For(0, rows, ComputeDxForRow);
        }

        public override void DropoutForward(Tensor y, double dropoutRate, bool isTraining, Random dropoutRandom, Tensor dropoutReservedSpaceForTraining)
        {
            var x = this;
            if (!isTraining)
            {
                x.CopyTo(y);
                return;
            }
            Debug.Assert(!dropoutReservedSpaceForTraining.UseGPU);
            var dropoutRateFloat = (float)dropoutRate;
            Utils.UniformDistribution(dropoutReservedSpaceForTraining.AsFloatCpuSpan, dropoutRandom, 0.0, 1.0);
            y.AsFloatCpu.BuildEntirelyFromInput(x, dropoutReservedSpaceForTraining, (prevLayer, prob) => prob < dropoutRate ? 0f : prevLayer / (1 - dropoutRateFloat));
        }
        public override void DropoutBackward(Tensor dy, Tensor dx, double dropoutRate, Tensor dropoutReserveSpace)
        {
            Debug.Assert(!dropoutReserveSpace.UseGPU);
            var dropoutRateFloat = (float)dropoutRate;
            dx.AsFloatCpu.BuildEntirelyFromInput(dy, dropoutReserveSpace, (dOutput, prob) => prob < dropoutRateFloat ? 0f : dOutput / (1 - dropoutRateFloat));
        }
        //this = dy

        public override void ActivationForward(cudnnActivationMode_t activationType, Tensor activationParameter, Tensor y)
        {
            var x = this;
            Debug.Assert(AreCompatible(new List<Tensor> {x, y}));
            switch (activationType)
            {
                case cudnnActivationMode_t.CUDNN_ACTIVATION_RELU:
                    CpuTensorActivationFunctions.Relu(x, y);
                    return;
                case cudnnActivationMode_t.CUDNN_ACTIVATION_LEAKY_RELU:
                    Debug.Assert(activationParameter.Dimension == 1);
                    Debug.Assert(activationParameter.Count == 1);
                    CpuTensorActivationFunctions.LeakyRelu(x, y, activationParameter.AsReadonlyFloatCpuSpan[0]);
                    return;
                case cudnnActivationMode_t.CUDNN_ACTIVATION_ELU:
                    CpuTensorActivationFunctions.Elu(x, y, 1.0);
                    return;
                case cudnnActivationMode_t.CUDNN_ACTIVATION_TANH:
                    CpuTensorActivationFunctions.Tanh(x, y);
                    return;
                case cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID:
                    CpuTensorActivationFunctions.Sigmoid(x, y);
                    return;
                case cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX:
                    CpuTensorActivationFunctions.Softmax(x, y);
                    return;
                case cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX_LAST_DIMENSION:
                    CpuTensorActivationFunctions.SoftmaxLastDimension(x, y);
                    return;
                case cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX_WITH_HIERARCHY:
                    Debug.Assert(activationParameter.Dimension == 1);
                    CpuTensorActivationFunctions.SoftmaxWithHierarchy(x, y, activationParameter);
                    return;
                case cudnnActivationMode_t.CUDNN_ACTIVATION_SWISH:
                    CpuTensorActivationFunctions.Swish(x, y);
                    return;
                case cudnnActivationMode_t.CUDNN_ACTIVATION_LN:
                    CpuTensorActivationFunctions.Ln(x, y);
                    return;
                case cudnnActivationMode_t.CUDNN_ACTIVATION_IDENTITY:
                    x.CopyTo(y);
                    return;
                default:
                    throw new ArgumentException("invalid activation mode " + activationType);
            }
        }
        public override void ActivationBackward(cudnnActivationMode_t activationType, Tensor activationParameter, Tensor dy, Tensor x, Tensor y)
        {
            var dx = this;
            Debug.Assert(AreCompatible(new List<Tensor> { y, dy, x, dx }));
            switch (activationType)
            {
                case cudnnActivationMode_t.CUDNN_ACTIVATION_RELU:
                    CpuTensorActivationFunctions.ReluGradient(y, dy, dx);
                    return;
                case cudnnActivationMode_t.CUDNN_ACTIVATION_LEAKY_RELU:
                    Debug.Assert(activationParameter.Dimension == 1);
                    Debug.Assert(activationParameter.Count == 1);
                    CpuTensorActivationFunctions.LeakyReluGradient(y, dy, dx, activationParameter.AsReadonlyFloatCpuSpan[0]);
                    return;
                case cudnnActivationMode_t.CUDNN_ACTIVATION_ELU:
                    CpuTensorActivationFunctions.EluGradient(y, dy, x, dx, 1f);
                    return;
                case cudnnActivationMode_t.CUDNN_ACTIVATION_TANH:
                    CpuTensorActivationFunctions.TanhGradient(y, dy, dx);
                    return;
                case cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID:
                    CpuTensorActivationFunctions.SigmoidGradient(y, dy, dx);
                    return;
                case cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX_LAST_DIMENSION:
                    CpuTensorActivationFunctions.SoftmaxGradientLastDimension(y, dy, dx);
                    return;
                case cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX:
                    CpuTensorActivationFunctions.SoftmaxGradient(y, dy, dx);
                    return;
                case cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX_WITH_HIERARCHY:
                    Debug.Assert(activationParameter.Dimension == 1);
                    CpuTensorActivationFunctions.SoftmaxGradientWitHierarchy(y, dy, dx, activationParameter);
                    return;
                case cudnnActivationMode_t.CUDNN_ACTIVATION_SWISH:
                    CpuTensorActivationFunctions.SwishGradient(y, dy, x, dx);
                    return;
                case cudnnActivationMode_t.CUDNN_ACTIVATION_LN:
                    CpuTensorActivationFunctions.LnGradient(dy, x, dx);
                    return;
                case cudnnActivationMode_t.CUDNN_ACTIVATION_IDENTITY:
                    dy.CopyTo(dx);
                    return;
                default:
                    throw new ArgumentException("invalid activation mode " + activationType);
            }
        }

        public override void Update_Adding_Alpha_X(float alpha, Tensor x)
        {
            var y = this;
            Debug.Assert(AreCompatible(new List<Tensor> {y, x}));
            Debug.Assert(x.Count == y.Count);
            MKL_BLAS.cblas_saxpy(x.Count, alpha, x.AsFloatPointer, 1, y.AsFloatPointer, 1);
        }

        // compute: this = alpha * x + beta * this
        public override void AddTensor(float alpha, Tensor x, float beta)
        {
            // this = beta * this
            Update_Multiplying_By_Alpha(beta);
            // this = alpha * x + beta * this
            Update_Adding_Alpha_X(alpha, x);
        }

        public override void LinearFunction(float slope, Tensor x, float intercept)
        {
            Debug.Assert(this.SameShape(x));
            var yAsSpan = AsFloatCpuSpan;
            var xAsSpan = x.AsReadonlyFloatCpuSpan;
            for (int i = 0; i < xAsSpan.Length; ++i)
            {
                yAsSpan[i] = slope * xAsSpan[i] + intercept;
            }
        }

        public override void MultiplyTensor(Tensor a, Tensor diagonalMatrix)
        {
            Debug.Assert(this.SameShape(a));
            Debug.Assert(a.Count >= diagonalMatrix.Count);
            Debug.Assert(Count % diagonalMatrix.Count == 0);

            var aFloat = a.AsFloatCpuSpan;
            var xFloat = diagonalMatrix.AsFloatCpuSpan;
            var thisFloat = AsFloatCpuSpan;
            if (a.Count == diagonalMatrix.Count)
            {
                for (int i = 0; i < diagonalMatrix.Count; ++i)
                {
                    thisFloat[i] = aFloat[i] * xFloat[i];
                }
            }
            else
            {
                Debug.Assert(diagonalMatrix.Shape[0]*diagonalMatrix.Shape[1] == diagonalMatrix.Count);
                int indexInX = 0;
                int indexInThis = 0;
                int toAddInThis = a.Count / diagonalMatrix.Count;
                while(indexInThis<Count)
                {
                    int endIndexInThis = indexInThis + toAddInThis;
                    var alphaFromX = xFloat[indexInX++];
                    while (indexInThis< endIndexInThis)
                    {
                        thisFloat[indexInThis] = aFloat[indexInThis] * alphaFromX;
                        indexInThis++;
                    }
                }
            }
        }

        public override void UpSampling2D(Tensor tensorBeforeUpSampling, int rowFactor, int colFactor, UpSampling2DLayer.InterpolationEnum interpolation)
        {
            Debug.Assert(rowFactor >= 1);
            Debug.Assert(colFactor >= 1);
            if (interpolation == UpSampling2DLayer.InterpolationEnum.Bilinear)
            {
                throw new NotImplementedException("only " + UpSampling2DLayer.InterpolationEnum.Nearest + " interpolation is supported (not " + interpolation + ")");
            }
            var beforeUpSampling = (CpuTensor<T>)tensorBeforeUpSampling;
            var afterUpSampling = this;
            Debug.Assert(rowFactor * beforeUpSampling.Shape[2] == afterUpSampling.Shape[2]);
            Debug.Assert(colFactor * beforeUpSampling.Shape[3] == afterUpSampling.Shape[3]);
            for (int m = 0; m < afterUpSampling.Shape[0]; ++m)
            for (int c = 0; c < afterUpSampling.Shape[1]; ++c)
            for (int row = 0; row < afterUpSampling.Shape[2]; ++row)
            for (int col = 0; col < afterUpSampling.Shape[3]; ++col)
            {
                afterUpSampling.Set(m, c, row, col, beforeUpSampling.Get(m, c, row / rowFactor, col / colFactor));
            }
        }
        public override void DownSampling2D(Tensor tensorBeforeDownSampling, int rowFactor, int colFactor)
        {
            var beforeDownSampling = (CpuTensor<float>)tensorBeforeDownSampling;
            var afterDownSampling = AsFloatCpu;
            afterDownSampling.ZeroMemory();
            Debug.Assert(rowFactor >= 1);
            Debug.Assert(colFactor >= 1);
            Debug.Assert(rowFactor * afterDownSampling.Shape[2] == beforeDownSampling.Shape[2]);
            Debug.Assert(colFactor * afterDownSampling.Shape[3] == beforeDownSampling.Shape[3]);
            for (int m = 0; m < beforeDownSampling.Shape[0]; ++m)
            for (int c = 0; c < beforeDownSampling.Shape[1]; ++c)
            for (int row = 0; row < beforeDownSampling.Shape[2]; ++row)
            for (int col = 0; col < beforeDownSampling.Shape[3]; ++col)
            {
                var toAdd = beforeDownSampling.Get(m, c, row , col );
                var prevValue = afterDownSampling.Get(m, c, row / rowFactor, col / colFactor);
                afterDownSampling.Set(m, c, row / rowFactor, col / colFactor, toAdd + prevValue);
            }
        }

        public override void MultiplyEachRowIntoSingleValue(Tensor a, Tensor b)
        {
            Debug.Assert(a.SameShape(b));
            int nbRows = Count;
            Debug.Assert(nbRows <= a.Count);
            Debug.Assert(a.Count % nbRows == 0);
            int nbColumns_in_a_and_b = b.Count / nbRows;
            var thisFloat = AsFloatCpuSpan;
            var aFloat = a.AsFloatCpuSpan;
            var bFloat = b.AsFloatCpuSpan;
            int indexIn_a_or_b = 0;
            for (int row = 0; row < nbRows; ++row)
            {
                float rowSum = 0;
                for (int col = 0; col < nbColumns_in_a_and_b; ++col)
                {
                    rowSum += aFloat[indexIn_a_or_b] * bFloat[indexIn_a_or_b];
                    ++indexIn_a_or_b;
                }
                thisFloat[row] = rowSum;
            }
        }

        public override void Clip(float lower, float upper)
        {
            Debug.Assert(upper >= lower);
            var thisFloat = AsFloatCpuSpan;
            for (int i = 0; i < Count; ++i)
            {
                var curValue = thisFloat[i];
                thisFloat[i] = Math.Min(Math.Max(curValue, lower), upper);
            }
        }

        public override void ZeroPadding(Tensor unpaddedTensor, int paddingTop, int paddingBottom, int paddingLeft, int paddingRight)
        {
            //we are adding padding to 'unpaddedTensor' to initialize 'paddedTensor'
            var paddedTensor = this;
            paddedTensor.ZeroMemory();
            ZeroPadding_and_Unpadding(unpaddedTensor, paddingTop, paddingLeft, false);
        }
        public override void ZeroUnpadding(Tensor paddedTensor, int paddingTop, int paddingBottom, int paddingLeft, int paddingRight)
        {
            ((CpuTensor<T>)paddedTensor).ZeroPadding_and_Unpadding(this, paddingTop, paddingLeft, true);
        }

        private void ZeroPadding_and_Unpadding(Tensor unpaddedTensor, int paddingTop, int paddingLeft, bool isUnpadding)
        {
            var paddedTensor = this;
            Debug.Assert(AreCompatible(new List<Tensor> { paddedTensor, unpaddedTensor }));
            Debug.Assert(paddedTensor.Dimension == 4);
            Debug.Assert(paddedTensor.Dimension == unpaddedTensor.Dimension);
            Debug.Assert(paddedTensor.Shape[0] == unpaddedTensor.Shape[0]); //same batch size
            Debug.Assert(paddedTensor.Shape[1] == unpaddedTensor.Shape[1]); //same number of channels
            int h_src = unpaddedTensor.Shape[2];
            int w_src = unpaddedTensor.Shape[3];
            // copy the row 'srcRowId' from 'src' tensor (n, c, h_src, w_src) to dest tensor (n, c, h_dest, w_dest)
            // the number of distinct rows in 'src' tensor is : n*c*h_src

            void ApplyZeroPaddingForRowId(int srcRowId)
            {
                // 0 <= srcRowId < n*c*h_src
                int row_src = (srcRowId % h_src);
                int unpaddedRowIndex = srcRowId * w_src;
                int paddedRowIndex = ((srcRowId / h_src) * paddedTensor.Shape[2] + row_src + paddingTop) * paddedTensor.Shape[3] + paddingLeft;
                if (isUnpadding)
                {
                    paddedTensor.CopyTo(paddedRowIndex, unpaddedTensor, unpaddedRowIndex, w_src);
                }
                else
                {
                    unpaddedTensor.CopyTo(unpaddedRowIndex, paddedTensor, paddedRowIndex, w_src);
                }
            }
            Parallel.For(0, unpaddedTensor.Shape[0] * unpaddedTensor.Shape[1] * unpaddedTensor.Shape[2], ApplyZeroPaddingForRowId);
        }

        public override void AssertIsNotDisposed()
        {
            if (_disposed)
            {
                throw new Exception("Tensor is disposed " + this);
            }
        }
        public override void Concatenate(IList<Tensor> tensors)
        {
            CheckConcatenate(tensors);
            void ConcatenateSingleRow(int m)
            {
                int startIdx = Idx(m);
                foreach (var t in tensors)
                {
                    t.CopyTo(t.Idx(m), this, startIdx, t.MultDim0);
                    startIdx += t.MultDim0;
                }
            }
            Parallel.For(0, Shape[0], ConcatenateSingleRow);
        }
        public override void Split(IList<Tensor> tensors)
        {
            CheckConcatenate(tensors);
            void SplitSingleRow(int m)
            {
                int startIdx = Idx(m);
                foreach (var t in tensors)
                {
                    CopyTo(startIdx, t, t.Idx(m), t.MultDim0);
                    startIdx += t.MultDim0;
                }
            }
            Parallel.For(0, Shape[0], SplitSingleRow);
        }

        ///// <summary>
        ///// return a (square) diagonal matrix of length (rowCount, rowCount)
        ///// each element in the diagonal will be 1, all other will be 0
        ///// </summary>
        ///// <param name="rowCount">number of rows and columns of the diagonal matrix</param>
        ///// <returns></returns>
        //public static CpuTensor<float> NewFloatDiagonalMatrix(int rowCount)
        //{
        //    var data = new float[rowCount * rowCount];
        //    for (int row = 0; row < rowCount; ++row)
        //    {
        //        data[row * rowCount + row] = 1f;
        //    }
        //    return new CpuTensor<float>(new[] { rowCount, rowCount }, data);
        //}


        // compute:     this = alpha * this
        public override void Update_Multiplying_By_Alpha(float alpha)
        {
            MKL_BLAS.cblas_sscal(Count, alpha, AsFloatPointer, 1);
        }
        #region pooling layers

        public override void Pooling(Tensor y, cudnnPoolingMode_t poolingMode, int poolingHeight, int poolingWidth, int verticalStride, int horizontalStride)
        {
            var x = this;
#if DEBUG
            Debug.Assert(AreCompatible(new List<Tensor> { x, y }));
            Debug.Assert(x.Shape[0] == y.Shape[0]); //same batch size
            Debug.Assert(x.Shape[1] == y.Shape[1]); //same number of channels
            Debug.Assert(x.Dimension == y.Dimension);
            Debug.Assert(x.Dimension == 4);
            int hOutput = y.Shape[2];
            int hInput = x.Shape[2];
            int hExpected = (hInput - poolingHeight) / verticalStride + 1;
            Debug.Assert(hOutput == hExpected);
            int wOutput = y.Shape[3];
            int wInput = x.Shape[3];
            int wExpected = (wInput - poolingWidth) / horizontalStride + 1;
            Debug.Assert(wOutput == wExpected);
#endif
            int batchSize = x.Shape[0];
            if (PoolingLayer.IsMaxPooling(poolingMode))
            {
                Parallel.For(0, batchSize, elementIndex => x.MaxPoolingForSingleElement4D(y, poolingHeight, poolingWidth, verticalStride, horizontalStride, elementIndex ));
            }
            else
            {
                Parallel.For(0, batchSize, elementIndex => x.AvgPoolingForSingleElement4D(y, poolingHeight, poolingWidth, verticalStride, horizontalStride, elementIndex));
            }
        }
        private void AvgPoolingForSingleElement4D(Tensor y, int poolingHeight, int poolingWidth, int verticalStride, int horizontalStride, int elementIndex)
        {
            var x = this;
            Debug.Assert(AreCompatible(new List<Tensor> { x, y }));
            Debug.Assert(x.Dimension == y.Dimension);
            Debug.Assert(x.Dimension == 4);
            int hOutput = y.Shape[2];
            int wOutput = y.Shape.Length>=4?y.Shape[3]:1;
            //the first (top left) point in 'y' is computed from a filter starting at (0,0)
            for (int c = 0; c < x.Shape[1]; ++c)
            {
                int row_filter_start = 0;
                for (int rowAfterPooling = 0; rowAfterPooling < hOutput; ++rowAfterPooling)
                {
                    int col_filter_start = 0;
                    for (int colAfterPooling = 0; colAfterPooling < wOutput; ++colAfterPooling)
                    {
                        //we want to compute the point in y[n, channelId, row_output, col_output]
                        //it is computed by applying an avg filter located (for its top left) in (row_filter_start,col_filter_start) in the x 
                        float outputPointSum = 0f;
                        int count = 0;
                        for (int rowBeforePooling = row_filter_start; rowBeforePooling < (row_filter_start + poolingHeight); ++rowBeforePooling)
                        {
                            for (int colBeforePooling = col_filter_start; colBeforePooling < (col_filter_start + poolingWidth); ++colBeforePooling)
                            {
                                outputPointSum += x.AsFloatCpu.Get(elementIndex, c, rowBeforePooling, colBeforePooling);
                                ++count;
                            }
                        }
                        y.AsFloatCpu.Set(elementIndex, c, rowAfterPooling, colAfterPooling, outputPointSum / count);
                        col_filter_start += horizontalStride;
                    }
                    row_filter_start += verticalStride;
                }
            }
        }
        private void MaxPoolingForSingleElement4D(Tensor y, int poolingHeight, int poolingWidth, int verticalStride, int horizontalStride, int elementIndex)
        {
            var x = this;
            Debug.Assert(AreCompatible(new List<Tensor> { x, y }));
            Debug.Assert(x.Dimension == y.Dimension);
            Debug.Assert(x.Dimension == 4);
            int hOutput = y.Shape[2];
            int wOutput = y.Shape[3];
            //the first (top left) point in 'y' is computed from a filter starting at (0,0)
            for (int c = 0; c < x.Shape[1]; ++c)
            {
                int row_filter_start = 0;
                for (int rowAfterPooling = 0; rowAfterPooling < hOutput; ++rowAfterPooling)
                {
                    int col_filter_start = 0;
                    for (int colAfterPooling = 0; colAfterPooling < wOutput; ++colAfterPooling)
                    {
                        //we want to compute the point in y[n, channelId, row_output, col_output]
                        //it is computed by applying a max filter located (for its top left) in (row_filter_start,col_filter_start) in the x 
                        float outputPointResult = float.MinValue;
                        for (int rowBeforePooling = row_filter_start; rowBeforePooling < (row_filter_start + poolingHeight); ++rowBeforePooling)
                        {
                            for (int colBeforePooling = col_filter_start; colBeforePooling < (col_filter_start + poolingWidth); ++colBeforePooling)
                            {
                                outputPointResult = Math.Max(outputPointResult, x.AsFloatCpu.Get(elementIndex, c, rowBeforePooling, colBeforePooling));
                            }
                        }
                        y.AsFloatCpu.Set(elementIndex, c, rowAfterPooling, colAfterPooling, outputPointResult);
                        col_filter_start += horizontalStride;
                    }
                    row_filter_start += verticalStride;
                }
            }
        }
        public override void PoolingGradient(Tensor yNotUsed, Tensor x, Tensor dx, cudnnPoolingMode_t poolingMode, int poolingHeight, int poolingWidth, int verticalStride, int horizontalStride)
        {
            var dy = this;
            int batchSize = x.Shape[0];
#if DEBUG
            Debug.Assert(AreCompatible(new List<Tensor> { dy, x, dx }));
            Debug.Assert(x.Shape[0] == dy.Shape[0]); //same batchSize
            Debug.Assert(x.Shape[1] == dy.Shape[1]); //same number of channels
            Debug.Assert(dx.SameShape(x));
            Debug.Assert(x.Shape.Length == 4);
            Debug.Assert(dx.Shape.Length == 4);
            Debug.Assert(dy.Shape.Length == 4);
            int hOutput = dy.Shape[2];
            int wOutput = dy.Shape[3];
            Debug.Assert(hOutput == ((x.Shape[2] - poolingHeight) / verticalStride + 1));
            Debug.Assert(wOutput == ((x.Shape[3] - poolingWidth) / horizontalStride + 1));
#endif
            dx.ZeroMemory();
            if (PoolingLayer.IsMaxPooling(poolingMode))
            {
                Parallel.For(0, batchSize, elementIndex => dy.MaxPoolingGradientForSingleElement4D(x, dx, poolingHeight, poolingWidth, verticalStride, horizontalStride, elementIndex));
            }
            else
            {
                Parallel.For(0, batchSize, elementIndex => dy.AvgPoolingGradientForSingleElement4D(x, dx, poolingHeight, poolingWidth, verticalStride, horizontalStride, elementIndex));
            }
        }
        private void AvgPoolingGradientForSingleElement4D(Tensor x, Tensor dx, int poolingHeight, int poolingWidth, int verticalStride, int horizontalStride, int elementIndex)
        {
            var dy = this;
            int hOutput = dy.Shape[2];
            int wOutput = dy.Shape[3];
            double doubleMultiplier = 1.0 / (poolingHeight * poolingWidth);
            float floatMultiplier = (float)doubleMultiplier;

            for (int c = 0; c < x.Shape[1]; ++c)
            {
                int row_filter_start = 0;
                for (int rowAfterPooling = 0; rowAfterPooling < hOutput; ++rowAfterPooling)
                {
                    int col_filter_start = 0;
                    for (int colAfterPooling = 0; colAfterPooling < wOutput; ++colAfterPooling)
                    {
                        for (int rowBeforePooling = row_filter_start; rowBeforePooling < (row_filter_start + poolingHeight); ++rowBeforePooling)
                        {
                            for (int colBeforePooling = col_filter_start; colBeforePooling < (col_filter_start + poolingWidth); ++colBeforePooling)
                            {
                                var pointGradient = dy.AsFloatCpu.Get(elementIndex, c, rowAfterPooling, colAfterPooling);
                                dx.AsFloatCpu.Set(elementIndex, c, rowBeforePooling, colBeforePooling, floatMultiplier * pointGradie