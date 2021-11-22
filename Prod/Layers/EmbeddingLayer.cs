using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using JetBrains.Annotations;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.Networks;
using SharpNet.Optimizers;

namespace SharpNet.Layers;

/// <summary>
/// This layer can only be used as the second layer in a model (the first layer being the InputLayer).
/// 
/// =======================================================================================================
/// input 'x' shape                                     output 'y' shape
/// =======================================================================================================
/// (batchSize, timeSteps)                              (batchSize, timeSteps, EmbeddingDim)
/// =======================================================================================================
/// (batchSize, input_length)                           (batchSize, input_length+EmbeddingDim-1)
/// =======================================================================================================
/// (batchSize, timeSteps, input_length)                (batchSize, timeSteps, input_length+EmbeddingDim-1)
/// =======================================================================================================
/// </summary>
public sealed class EmbeddingLayer : Layer
{
    #region Private fields
        
    #region trainable parameters
    /// <summary>
    /// Word Embedding, of shape: (VocabularySize, EmbeddingDim)
    /// </summary>
    [NotNull] private Tensor _weights;
    #endregion
        
    #region gradients
    /// <summary>
    /// same shape as '_weights'
    /// </summary>
    [NotNull] private Tensor _weightGradients;
    /// <summary>
    /// Adam or SGD optimizer or Vanilla SGD
    /// </summary>
    #endregion

    [NotNull] private readonly Optimizer _optimizer;


    /// <summary>
    /// each element is the description of an embedding:
    ///     vocabularySize:
    ///         Size of the vocabulary, i.e. maximum integer index + 1
    ///         In the input 'x' tensor:
    ///         each wordIndex element must be in [0, VocabularySize-1]
    ///     embeddingDim:
    ///         Dimension of the dense embedding
    ///     featureIndexInLastDimensionToUse:
    ///         index in last dimension of input tensor where to find the index of the feature to embed
    ///     embeddingTen