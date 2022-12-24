using System.Collections.Generic;
using System.Diagnostics;
using JetBrains.Annotations;
using SharpNet.Data;

namespace SharpNet.Optimizers;

public class VanillaSgdOrtho : Optimizer
{
    #region private fields
    // the orthogonal 'Q' matrix of shape (m, n)
    [CanBeNull] private readonly Tensor Q;
    //[CanBeNull] private readonly Tensor Identity_mm;
    [CanBeNull] private readonly Tensor buffer_mm;
    //[CanBeNull] private readonly Tensor buffer_mn_v1;
    //[CanBeNull] private readonly Tensor buffer_nn_v1;
    //  upper triangular matrix 'R' of shape (n, n)
    [CanBeNull] private readonly Tensor R;
    [CanBeNull] private readonly Tensor QRFactorization_buffer;
    private readonly TensorMemoryPool _memoryPool;

    #endregion

    public VanillaSgdOrtho(TensorMemoryPool memoryPool, int[] weightShape)
    {
        _memoryPool = memoryPool;
        Debug.Assert(weightShape.Length == 2);
        int m = weightShape[0];
        int n = weightShape[1];
        Debug.Assert(m >= n);
        Q = _memoryPool.GetFloatTensor(new[] { m, n });
        R = _memoryPool.GetFloatTensor(new[] { n, n });
        buffer_mm = _memoryPool.GetFloatTensor(new[] { m, m });
        //Identity_mm = _memoryPool.GetFloatTensor(new[] { m, m });
        //buffer_mn_v1 = _memoryPool.GetFloatTensor(new[] { m, n });
        //buffer_nn_v1 = _memoryPool.GetFloatTensor(new[] { n, n });
        //Identity_mm.SetIdentityMatrix();

        QRFactorization_buffer = _memoryPool.GetFloatTensor(new[] { Q.QRFactorization_FloatBufferLength() });
        ZeroMemory();
    }

    public override List<Tensor> EmbeddedTensors
    {
        get
        {
            var result = new List<Tensor> { Q, R, QRFactorization_buffer};
            result.RemoveAll(t => t == null);
            return result;
        }
    }

    public override bool IsOrthogonal => true;


    public override void UpdateWeights(double learningRate, double maxLearningRate, int batchSize, Tensor weights, Tensor weightGradients, Tensor bias, Tensor biasGradients)
    {
        Debug.Assert(weights.SameShape(weightGradients));
        Debug.Assert(bias == null || bias.SameShape(biasGradients));
        var ponderedLearningRate = PonderedLearnin