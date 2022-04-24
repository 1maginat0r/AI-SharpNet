using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using SharpNet.Data;
using SharpNet.GPU;
using SharpNet.Networks;

namespace SharpNet.Layers;


/// <summary>
/// Input:
///  allX[0] = Q of shape (batch_size, query_timeSteps == input_seq_length, embedding_dim)
///  allX[1] = V of shape (batch_size, value_timeSteps, embedding_dim)
///  allX[2] = K of shape (batch_size, value_timeSteps, embedding_dim)
///     in most cases, K and V are the same tensor (K is optional, it's default value is V)
/// Output:
/// y of shape:           (batch_size, value_timeSteps, embedding_dim)
///                       (same as V shape)
/// </summary>
public class ScaledDotProductAttentionLayer : Layer
{
    private readonly bool _use_scale;
    private readonly bool _use_causal_mask;

    private Tensor _weights_buffer = null;
    public ScaledDotProductAttentionLayer(bool use_scale, bool use_causal_mask, int queriesLayerIndex, int valuesLayerIndex, int keysLayerIndex,
        Network network, string layerName = "") : base(network, new[]{queriesLayerIndex, valuesLayerIndex, keysLayerIndex }, layerName)
    {
        _use_scale = use_scale;
        _use_causal_mask = use_causal_mask;
    }

    #region forward and backward propagation
    public override void ForwardPropagation(List<Tensor> allX, Tensor y, bool isTraining)
    {
        Debug.Assert(allX.Count == 3);
        var Q = allX[QUERIES_LAYER_INDEX];      // queries: (batch_size, query_timeSteps == input_seq_length, embedding_dim)
        var V = allX[VALUES_LAYER_INDEX];       // values:  (batch_size, value_timeSteps, embedding_dim)
        var K = allX[KEYS_LAYER_INDEX];         // keys:    (batch_size, value_timeSteps, embedding_dim)

        ScaledDotProductAttentionForwardPropagation(Q, V, K, y, isTraining, ref _weights_buffer, Network.MemoryPool, _use_scale, _use_causal_mask);
    }

    public static void ScaledDotProductAttentionForwardPropagation(Tensor Q, Tensor V, Tensor K, Tensor y, bool isTraining, ref Tensor _weights_buffer, TensorMemoryPool memoryPool, bool use_scale, bool use_causal_mask)
    {

        if (!V.Shape.SequenceEqual(K.Shape))
        {
            throw new ArgumentException($"V.Shape and K.Shape must be equal, but are {Tensor.ShapeToString(V.Shape)} and {Tensor.ShapeToString(K.Shape)}");
        }
        if (!V.Shape.SequenceEqual(y.Shape))
        {
            throw new ArgumentException($"V.Shape and y.Shape must be equal, but are {Tensor.ShapeToString(V.Shape)} and {Tensor.ShapeToString(y.Shape)}");
        }
        if (Q.Shape[2] != V.Shape[2])
        {
            throw new ArgumentException($"queries.Shape[2] and values.Shape[2] must be equal (same embedding dim), but are {Q.Shape[2]} and {V.Shape[2]}");
        }
        var batch_size = K.Shape[0];
        var query_time_steps = Q.Shape[1];
        var value_time_steps = V.Shape[1];
        var embedding_dim = K.Shape[2];

        //Scoring the queries against the keys after transposing the latter, and scaling
        //scores = matmul(Q, K, transpose_keys = True) / math.sqrt(embedding_dim))
        var scores_buffer = memoryPool.GetFloatTensor(new[] { batch_size, query_time_steps, value_time_steps });
        float scaling = (use_scale) ? (1.0f / MathF.Sqrt(embedding_dim)) : 1.0f;

        scores_buffer.BatchMatrixMultiplication(Q, false, K, true, scaling, 0.0f);

        if (use_causal_mask)
        {
            scores_buffer.SetAllElementsAboveMainDiagonal(-1e12f);
        }

        //Computing the weights by a softmax operation
        //weights = softmax(scores)
        memoryPool.GetFloatTensor(ref _weights_buffer, scores_buffer.Shape);       // (batch_size, query_time_steps, value_time_steps)
        scores_buffer.ActivationForward(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX_LAST_DIMENSION, null, _weights_buffer);

        //Computing the attention by a weighted sum of the value vectors
        //y = matmul(weights, V)
        y.BatchMatrixMultiplication(_weights_buffer, false, V, false, 1.0f, 0.0f);
        memoryPool.FreeFloatTensor(scores_buffer);
        if (!isTraining)
        {
            memoryPool.FreeFloatTensor(ref _weights_buffer);
        }
    }

    private const int QUERIES_LAYER_INDEX = 0;
    private const int VALUES_LAYER_INDEX = 1;
    private const int KEYS_LAYER_INDEX = 2;


    public override void BackwardPropagation(List<Tensor> allX, Tensor y_NotUsed, Tensor dy, List<Tensor> allDx)
    {
        //dy:          (batch_size, value_timeSteps, value_embedding_dim)
        var dQ = allDx[QUERIES_LAYER_INDEX];    // queries:    (batch_size, query_timeSteps == input_seq_length, embedding_dim)
        var dV = allDx[VALUES_LAYER_INDEX];     // values:     (batch_size, value_timeSteps, embedding_dim)
        var dK = allDx[KEYS_LAYER_INDEX];       /