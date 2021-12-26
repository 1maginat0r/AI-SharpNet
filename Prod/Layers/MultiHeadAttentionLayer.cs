﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using JetBrains.Annotations;
using SharpNet.Data;
using SharpNet.Networks;
using SharpNet.Optimizers;

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
public class MultiHeadAttentionLayer : Layer
{
    private readonly int _num_heads;
    private readonly int _key_dim;
    private readonly int _value_dim;
    private readonly bool _use_bias_Q_V_K;
    private readonly bool _use_bias_O;
    private readonly bool _use_causal_mask;
    private const bool use_scale = true;
    private const bool flattenInputTensorOnLastDimension = true;
    private const int QUERIES_LAYER_INDEX = 0;
    private const int VALUES_LAYER_INDEX = 1;
    private const int KEYS_LAYER_INDEX = 2;


    #region Private fields
    #region trainable parameters
    [NotNull] private Tensor _weights;
    [CanBeNull] private Tensor _bias;
    #endregion
    #region gradients
    [NotNull] private Tensor _weightGradients;
    [CanBeNull] private Tensor _biasGradients;
    #endregion
    #endregion

    #region trainable parameters
    /// <summary>
    /// shape :   (embedding_dim, 2*num_heads * key_dim+2*num_heads* value_dim)
    /// contains the Weights for Q, K,V & O
    /// </summary>
    public override Tensor Weights => _weights;
    /// <summary>
    /// shape :   (1, 2*num_heads*key_dim + num_heads* value_dim + embedding_dim)
    /// contains the Weights for Q, K,V & O
    /// </summary>
    public override Tensor Bias => _bias;
    public override Tensor WeightGradients => _weightGradients;
    public override Tensor BiasGradients => _biasGradients;


    #endregion

    [NotNull] private readonly Optimizer _w_Q_optimizer;
    [NotNull] private readonly Optimizer _w_K_optimizer;
    [NotNull] private readonly Optimizer _w_V_optimizer;
    [NotNull] private readonly Optimizer _w_O_optimizer;


    //private Tensor Q_heads;             //(batch_size*query_time_steps, _num_heads*_key_dim)
    //private Tensor K_heads;             //(batch_size*value_time_steps, _num_heads*_key_dim)
    //private Tensor V_heads;             //(batch_size*value_time_steps, _num_heads, _value_dim)
    //private Tensor attention_heads;     //(batch_size*_num_heads, value_time_steps, _value_dim)
    private Tensor weights_buffer;      //(batch_size, query_time_steps, value_time_steps)
    private Tensor Q_heads_T;           //(batch_size*_num_heads, query_time_steps, _key_dim)
    private Tensor K_heads_T;           //(batch_size*_num_heads, value_time_steps, _key_dim)
    private Tensor V_heads_T;           //(batch_size*_num_heads, value_time_steps, _value_dim)
    private Tensor attention_heads_T;   //(batch_size, value_time_steps, _num_heads* _value_dim)


    /// <summary>
    /// shape of Weights for Q K V O tensors
    /// </summary>
    private readonly List<int[]> _shapes_w_Q_K_V_O;
    /// <summary>
    /// number of elements in Weights for Q K V O tensors
    /// </summary>
    private readonly List<int> _count_w_Q_K_V_O;

    /// <summary>
    /// shape of Weights Bias for Q K V O tensors
    /// </summary>
    private readonly List<int[]> _shapes_w_bias_Q_K_V_O;
    /// <summary>
    /// number of elements in Weights Bias for Q K V O tensors
    /// </summary>
    private readonly List<int> _count_w_bias_Q_K_V_O;

    /// <summary>
    /// no need to have 'embedding_dim' as a parameter: it is always equal to the last dimension of 'V' (value) Layer
    /// </summary>
    /// <param name="num_heads"></param>
    /// <param name="key_dim"></param>
    /// <param name="value_dim"></param>
    /// <param name="use_bias_Q_V_K"></param>
    /// <param name="use_bias_O"></param>
    /// <param name="use_causal_mask"></param>
    /// <param name="queriesLayerIndex"></param>
    /// <param name="valuesLayerIndex"></param>
    /// <param name="keysLayerIndex"></param>
    /// <param name="network"></param>
    /// <param name="layerName"></param>
    public MultiHeadAttentionLayer(int num_heads, int key_dim, int value_dim, bool use_bias_Q_V_K, bool use_bias_O,
        bool use_causal_mask, int queriesLayerIndex, int valuesLayerIndex, int keysLayerIndex,
        Network network, string layerName = "") : base(network,
        new[] { queriesLayerIndex, valuesLayerIndex, keysLayerIndex }, layerName)
    {
        _num_heads = num_heads;
        _key_dim = key_dim;
        _value_dim = value_dim;
        _use_bias_Q_V_K = use_bias_Q_V_K;
        _use_bias_O = use_bias_O;
        var embedding_dim = network.Layers[valuesLayerIndex].OutputShape(1)[2];
        _use_causal_mask = use_causal_mask;

        _shapes_w_Q_K_V_O = new List<int[]>
        {
            new[] { embedding_dim, num_heads * key_dim },
            new[] { embedding_dim, num_heads * key_dim },
            new[] { embedding_dim, num_heads * value_dim },
            new[] { num_heads * value_dim, embedding_dim }
        };
  