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
            throw new ArgumentException($"V.Shape and y.Shape must be equal, bu