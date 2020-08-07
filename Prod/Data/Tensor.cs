
﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using JetBrains.Annotations;
using SharpNet.CPU;
using SharpNet.GPU;
using SharpNet.Hyperparameters;
using SharpNet.Layers;

namespace SharpNet.Data
{
    //[DebuggerDisplay("{ToString(true)}")]
    public abstract unsafe class Tensor : IDisposable
    {
        #region fields
        public int[] Shape { get; protected set; }
        public int MultDim0 { get; private set; }
        public int MultDim1 { get; private set; }
        private int _multDim2;
        public bool UseGPU { get; }
        public int TypeSize { get; }
        #endregion

        #region constructors
        protected Tensor(int[] shape, int typeSize, bool useGpu)
        {
            Debug.Assert(shape.Length >= 1);
            Debug.Assert(shape.Length <= 4);
            Debug.Assert(shape.Min() >= 0);
            Shape = shape;
            UseGPU = useGpu;
            TypeSize = typeSize;
            RecomputeMultDim();
        }
        #endregion
        public bool SameShape(params Tensor[] b) { return b.Where(x=>x!=null).All(SameShape); }
        public bool SameShape(Tensor b) {return SameShape(b.Shape);}
        protected bool SameShape(int[] shape) { return Shape.SequenceEqual(shape); }

        public bool SameShapeExceptFirstDimension(Tensor b) { return SameShapeExceptFirstDimension(b.Shape); }
        public override string ToString()
        {
            return ToString(false);
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int Idx(int n) { return MultDim0 * n; }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int Idx(int n, int c)
        {
            return MultDim0 * n + MultDim1 * c;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int Idx(int n, int c, int h, int w) { return MultDim0 * n + MultDim1 * c + _multDim2 * h + w; }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int Idx(int n, int c, int h) { return MultDim0 * n + MultDim1 * c + h; }

        // this = a*b
        public void Dot(Tensor a, Tensor b) { Dot(a, false, b, false, 1, 0); }


        /// <summary>
        /// compute the transpose of 'this' tensor and stores it in 'output'
        /// </summary>
        /// <param name="transposed"></param>
        public abstract void Transpose(Tensor transposed);

        /// <summary>
        /// Orthogonal initializer for Weights
        /// See: https://www.tensorflow.org/api_docs/python/tf/keras/initializers/Orthogonal
        /// </summary>
        /// <param name="rand"></param>
        public abstract void Orthogonal(Random rand);


        /// <summary>
        /// length of the buffer needed to compute the QR Factorization of 'this' tensor
        /// </summary>
        /// <returns></returns>
        public abstract int QRFactorization_FloatBufferLength();

        /// <summary>
        /// compute A (= this) = Q R factorization
        /// this : the A matrix (in row major order) of shape (m, n) (with m>=n)
        /// </summary>
        /// <param name="Q">the orthogonal 'Q' matrix of shape (m, n)</param>
        /// <param name="R">the upper triangular matrix 'R' of shape (n, n)</param>
        /// <param name="buffer">a float tensor of length returned by 'QRFactorization_FloatBufferLength'</param>
        public abstract void QRFactorization(Tensor Q, Tensor R, Tensor buffer);


        /// <summary>
        /// if the 'this' matrix is an orthogonal matrix, then transpose(this) * this = Identity matrix
        /// return the max error between the expected result (identity matrix) and the observed result of transpose(this) * this
        /// </summary>
        /// <returns></returns>
        public float MaxErrorIfOrthogonalMatrix()
        {
            var n = MultDim0;
            var a = ToCpuFloat();
            //var aTranspose = new CpuTensor<float>(new [] { n, m });
            var multResult =  new CpuTensor<float>(new [] { n, n });
            multResult.Dot(a, true, a, false, 1.0f, 0.0f);
            var spanResult = multResult.AsReadonlyFloatCpuSpan;
            float maxError = 0.0f;
            for(int row=0;row<n;++row )
            for (int col = 0; col < n; ++col)
            {
                var expectedResult = (col == row) ? 1.0f : 0.0f;
                var observedError = Math.Abs(spanResult[col + n * row] - expectedResult);
                maxError = Math.Max(maxError, observedError);
            }
            return maxError;
        }

        /// <summary>
        /// set to 0 all the elements below the main diagonal of the matrix
        /// (all elements with row index strictly less then column index)
        /// </summary>
        public abstract void SetToZeroAllElementsBelowMainDiagonal();


        /// <summary>
        /// set to 'valueForElementsAboveMainDiagonal' all the elements strictly above the main diagonal
        /// (all elements with row index strictly higher then column index)
        /// if 'this' if a 2D Tensor of shape (rows_by_matrix, cols_by_matrix)
        ///     each element above the main diagonal will be set to 'valueForElementsAboveMainDiagonal'
        /// if 'this' if a 3D Tensor of shape (matrices_count, rows_by_matrix, cols_by_matrix)
        ///     it will be considered as a list of 'matrices_count' matrices each with shape (rows_by_matrix, cols_by_matrix)
        ///     each individual matrix will be updated
        /// </summary>
        public abstract void SetAllElementsAboveMainDiagonal(float valueForElementsAboveMainDiagonal);

        /// <summary>
        /// set the 'this' square matrix to an identity matrix (1 on diagonals, 0 everywhere else)
        /// constraints: 'this must be a squared matrix (rows == cols)
        /// </summary>
        public abstract void SetIdentityMatrix();




        /// <summary>
        /// this (= 'y') shape :
        ///      (batchSize, timeSteps, outputSize)
        /// </summary>
        /// <param name="x">
        /// 'x' shape:
        ///      (batchSize, timeSteps, inputSize)
        /// </param>
        /// <param name="wordEmbedding">
        ///  'wordEmbedding' shape:
        ///     (vocabularySize, embeddingDim)
        ///     vocabularySize = 1+number of distinct words in the embedding
        /// </param>