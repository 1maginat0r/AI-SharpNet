﻿using System;
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

            in