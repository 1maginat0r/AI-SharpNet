
﻿using System;
using System.Linq;
using SharpNet.Datasets;
using SharpNet.GPU;
using SharpNet.Layers;

// ReSharper disable FieldCanBeMadeReadOnly.Global
// ReSharper disable MemberCanBePrivate.Global

namespace SharpNet.Networks.Transformers;

public class TransformerNetworkSample : NetworkSample
{
    // ReSharper disable once EmptyConstructor
    public TransformerNetworkSample()
    {
    }

    #region Hyperparameters

    public float layer_norm_epsilon = LayerNormalizationLayer.DEFAULT_EPSILON;
    public int embedding_dim = -1; // == d_model
    public cudnnActivationMode_t LastActivationLayer = cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX_LAST_DIMENSION;
    public int N_PositionalEncoding = PositionalEncodingAttnIsAllYouNeedLayer.DEFAULT_N_POSITIONAL_ENCODING;


    /// <summary>
    /// if false:
    ///     expect the input layer to be of shape (batch_size, seq_len)
    /// if true:
    ///     expect the input layer to be of shape (batch_size, seq_len, embedding_dim)
    /// </summary>
    public bool input_is_already_embedded = false;


    /// <summary>
    /// if false:
    ///     the predicted shape is (batch_size, seq_len, 1)
    /// else:
    ///     the predicted shape is (batch_size, 1)
    /// </summary>
    public bool output_shape_must_be_scalar = false;

    public bool layer_norm_before_ffd = true;              //should be true
    public bool layer_norm_after_ffd = false;              //should be false
    public bool layer_norm_before_last_dense = true; // must be true

    //encoders Hyperparameters
    public int encoder_num_transformer_blocks = -1;
    public int encoder_num_heads = -1; //must be a divider of 'embedding_dim'
    public bool encoder_mha_use_bias_Q_V_K = false;         
    public bool encoder_mha_use_bias_O = true;
    public float encoder_mha_dropout = 0.2f;
    public int encoder_feed_forward_dim = -1;
    public float encoder_feed_forward_dropout = 0.2f;
    public bool encoder_use_causal_mask = false;
    public bool encoder_add_layer_norm_before_mha = true;   //should be true
    public bool encoder_add_layer_norm_after_mha = false;   //should be false


    public POOLING_BEFORE_DENSE_LAYER pooling_before_dense_layer = POOLING_BEFORE_DENSE_LAYER.NONE;

    //decoders Hyperparameters
    public int decoder_num_transformer_blocks = -1;
    public int decoder_num_heads = -1; //must be a divider of 'embedding_dim'
    public bool decoder_mha_use_bias_Q_V_K = false;
    public bool decoder_mha_use_bias_O = true;
    public float decoder_mha_dropout = 0.2f;
    public int decoder_feed_forward_dim = -1;
    public float decoder_feed_forward_dropout = 0.2f;
    public bool decoder_add_layer_norm_before_mha = true;
    public bool decoder_add_layer_norm_after_mha = false;

    #endregion

    //public static TransformerNetworkSample DefaultFullEncoders(int embedding_dim, int N, int num_heads, bool use_bias, float mha_dropout, int feed_forward_dim, float feed_forward_dropout)
    //{
    //    var sample = (TransformerNetworkSample)new TransformerNetworkSample()
    //        {
    //            LossFunction = EvaluationMetricEnum.CategoricalCrossentropy,
    //            CompatibilityMode = CompatibilityModeEnum.TensorFlow,
    //            lambdaL2Regularization = 0.0005,
    //            NumEpochs = 150,
    //            BatchSize = 128,
    //            InitialLearningRate = 0.1,

    //            //No Data augmentation
    //            DataAugmentationType = ImageDataGenerator.DataAugmentationEnum.NO_AUGMENTATION,
              
    //        }
    //        .WithCommonParams(embedding_dim)
    //        .WithEncoders(N, num_heads, use_bias, mha_dropout, feed_forward_dim, feed_forward_dropout)
    //        .WithSGD(0.9, false)
    //        .WithCyclicCosineAnnealingLearningRateScheduler(10, 2);
    //    return sample;
    //}

    public override void BuildLayers(Network nn, AbstractDatasetSample datasetSample)
    {
        //nn.PropagationManager.LogPropagation = true;
        //nn.Sample.DisplayTensorContentStats = true;

        if (encoder_num_transformer_blocks >= 1 && decoder_num_transformer_blocks<=0)
        {
            //Full encoders
            var inputShapeOfSingleElement = datasetSample.X_Shape(1).Skip(1).ToArray();
            if (!input_is_already_embedded)
            {
                if (inputShapeOfSingleElement.Length != 1)
                {
                    throw new ArgumentException($"inputShape.Length={inputShapeOfSingleElement.Length} != 1");
                }
                int timeSteps = inputShapeOfSingleElement[0];
                nn.Input(timeSteps, -1, -1);
                nn.Embedding(new[] { datasetSample.NumClass }, new[] { embedding_dim }, new[] { -1 }, new[] { 0 }, 0.0);
            }
            else
            {
                if (inputShapeOfSingleElement.Length != 2)
                {
                    throw new ArgumentException($"inputShape.Length={inputShapeOfSingleElement.Length} != 2");
                }
                
                int timeSteps = inputShapeOfSingleElement[0];
                if (inputShapeOfSingleElement[1] != embedding_dim)
                {
                    throw new ArgumentException($"inputShape[1]={inputShapeOfSingleElement[1]} != embedding_dim={embedding_dim}");
                }
                nn.Input(timeSteps, embedding_dim, -1);
            }
            

            nn.PositionalEncodingAttnIsAllYouNeedLayer(N_PositionalEncoding);
            AddTransformers(nn, datasetSample.NumClass, nn.LastLayerIndex, -1);
            return;
        }

        throw new NotSupportedException("only full encoders are currently supported");
    }

    //private TransformerNetworkSample WithEncoders(int N, int num_heads, bool use_bias, float mha_dropout, int feed_forward_dim, float feed_forward_dropout, bool use_causal_mask = false, bool add_layer_norm_before_mha = false, bool add_layer_norm_after_mha = true)
    //{
    //    encoder_num_transformer_blocks = N;
    //    encoder_num_heads = num_heads;
    //    encoder_mha_use_bias = use_bias;
    //    encoder_mha_dropout = mha_dropout;
    //    encoder_feed_forward_dim = feed_forward_dim;
    //    encoder_feed_forward_dropout = feed_forward_dropout;
    //    encoder_use_causal_mask = use_causal_mask;
    //    encoder_add_layer_norm_after_mha = add_layer_norm_after_mha;
    //    return this;
    //}
    //private TransformerNetworkSample WithDecoders(int N, int num_heads, bool use_bias, float mha_dropout, int feed_forward_dim, float feed_forward_dropout, bool add_layer_norm_before_mha, bool add_layer_norm_after_mha)
    //{
    //    decoder_num_transformer_blocks = N;
    //    decoder_num_heads = num_heads;
    //    decoder_mha_use_bias = use_bias;
    //    decoder_mha_dropout = mha_dropout;
    //    decoder_feed_forward_dim = feed_forward_dim;
    //    decoder_feed_forward_dropout = feed_forward_dropout;
    //    decoder_add_layer_norm_before_mha = add_layer_norm_before_mha;
    //    decoder_add_layer_norm_after_mha = add_layer_norm_after_mha;
    //    return this;
    //}
    //private TransformerNetworkSample DisableEncoders()
    //{
    //    encoder_num_transformer_blocks = encoder_num_heads = encoder_feed_forward_dim = -1;
    //    encoder_mha_dropout = encoder_feed_forward_dropout = 0.0f;