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
            Debug.Assert(x.Shape.Len