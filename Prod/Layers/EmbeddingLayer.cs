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
    ///     embeddingTensorIndex:
    ///         index of the embedding tensor to use (in field 'EmbeddingTensors')
    /// </summary>
    private readonly List<(int vocabularySize, int embeddingDim, int indexInLastDimensionToUse, int embeddingTensorIndex)> EmbeddingDescriptions;

    private readonly List<(int vocabularySize, int embeddingDim)> EmbeddingTensorShapes;


    /// <summary>
    /// regularization hyper parameter. 0 if no L2 regularization
    /// </summary>
    private readonly double LambdaL2Regularization;
    /// <summary>
    /// if value > 0 
    ///     clip values of weights gradients in range [-ClipValueForGradients, ClipValueForGradients]
    /// else
    ///     do not clip values
    /// </summary>
    private readonly float ClipValueForGradients;
    /// <summary>
    /// true if we should divide the weight gradients by the time steps
    /// </summary>
    private readonly bool DivideGradientsByTimeSteps;
    #endregion



    public static List<(int vocabularySize, int embeddingDim, int indexInLastDimensionToUse, int embeddingTensorIndex)> ToEmbeddingLayerDescription(
        int[] vocabularySizes,
        int[] embeddingDims,
        int[] indexesInLastDimensionToUse, 
        int[] embeddingTensorIndex)
    {
        List<(int vocabularySize, int embeddingDim, int indexInLastDimensionToUse, int embeddingTensorIndex)> result = new();
        if (vocabularySizes.Length != embeddingDims.Length || vocabularySizes.Length != indexesInLastDimensionToUse.Length)
        {
            throw new ArgumentException($"input are not the same length : {vocabularySizes.Length} vs {embeddingDims.Length} vs {indexesInLastDimensionToUse.Length}");
        }
        for (int i = 0; i < vocabularySizes.Length; i++)
        {
            result.Add((vocabularySizes[i], embeddingDims[i], indexesInLastDimensionToUse[i], embeddingTensorIndex[i]));
        }
        return result;
    }

    

    #region constructor
    public EmbeddingLayer(
        IEnumerable<(int vocabularySize, int embeddingDim, int indexInLastDimensionToUse, int embeddingTensorIndex)> embeddingDescriptions,
        double lambdaL2Regularization,
        float clipValueForGradients,
        bool divideGradientsByTimeSteps,
        bool trainable, Network network, string layerName) : base(network, layerName)
    {
        EmbeddingDescriptions = embeddingDescriptions.OrderBy(t => t.indexInLastDimensionToUse).ToList();
        EmbeddingTenso