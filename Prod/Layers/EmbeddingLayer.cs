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
        EmbeddingTensorShapes = ExtractEmbeddingTensorShapes(EmbeddingDescriptions);

        if (EmbeddingDescriptions[0].indexInLastDimensionToUse < 0 && EmbeddingDescriptions.Count != 1)
        {
            throw new ArgumentException($"only 1 element is allowed if indexesInLastDimensionToUse = {EmbeddingDescriptions[0].indexInLastDimensionToUse}");
        }
        LambdaL2Regularization = lambdaL2Regularization;
        ClipValueForGradients = clipValueForGradients;
        DivideGradientsByTimeSteps = divideGradientsByTimeSteps;

        Trainable = trainable;

        //trainable params
        int weightColumns = EmbeddingTensorShapes.Select(t=>t.vocabularySize*t.embeddingDim).Sum();
        _weights = GetFloatTensor(new[] { 1, weightColumns });
        _weightGradients = GetFloatTensor(_weights.Shape);

        _optimizer = Sample.GetOptimizer(_weights.Shape, null, MemoryPool);
        ResetParameters(false);
    }

    private static List<(int vocabularySize, int embeddingDim)> ExtractEmbeddingTensorShapes(List<(int vocabularySize, int embeddingDim, int indexInLastDimensionToUse, int embeddingTensorIndex)> embeddingDescriptions)
    {
        IDictionary<int, (int vocabularySize, int embeddingDim)> allEmbeddingTensors = new Dictionary<int, (int vocabularySize, int embeddingDim)>();
        foreach (var c in embeddingDescriptions)
        {
            if (!allEmbeddingTensors.ContainsKey(c.embeddingTensorIndex))
            {
                allEmbeddingTensors[c.embeddingTensorIndex] = (c.vocabularySize, c.embeddingDim);
            }
            else
            {
                var observedTensor = allEmbeddingTensors[c.embeddingTensorIndex];
                if (observedTensor.vocabularySize != c.vocabularySize || observedTensor.embeddingDim != c.embeddingDim)
                {
                    throw new ArgumentException($"embedding tensor {c.embeddingTensorIndex} has already been defined with different vocabularySize or embeddingDim");
                }
            }
        }
        return allEmbeddingTensors.OrderBy(t => t.Key).Select(t => t.Value).ToList();
    }

    #endregion

    #region forward and backward propagation
    public override void ForwardPropagation(List<Tensor> allX, Tensor y, bool isTraining)
    {
        Debug.Assert(allX.Count == 1);
        var x = allX[0];
        Debug.Assert(x.Shape[0] == y.Shape[0]); //same batchSize
        Debug.Assert(y.Shape.Length != 3 || x.Shape[1] == y.Shape[1]); //same timeSteps
        Debug.Assert(!ShouldEmbedEachElementOfLastDimension || x.Shape[1] == y.Shape[1]); //same timeSteps
        int deltaForIndexesInLastDimensionToUse = 0;
        var allEmbeddingTensors = Split(_weights);

        var xOriginalShape = (int[])x.Shape.Clone();
        var yOriginalShape = (int[])y.Shape.Clone();

        // we'll ensure that in all cases:
        //  the x shape is (batchSize, timeSteps, input_length)
        //  the y shape is (batchSize, timeSteps, input_length+EmbeddingDim-1)
        if (x.Shape.Length == 2)
        {
            if (ShouldEmbedEachElementOfLastDimension)
            {
                //x shape from (batchSize, timeSteps) to (batchSize, timeSteps, 1)
                x.ReshapeInPlace(new [] { x.Shape[0], x.Shape[1], 1});
            }
            else
            {
                //x shape from (batchSize, input_length) to (batchSize, 1, input_length)
                x.ReshapeInPlace(new [] { x.Shape[0], 1, x.Shape[1] });
                //y shape from (batchSize, input_length+EmbeddingDim-1) to (batchSize, 1, input_length+EmbeddingDim-1)
                y.ReshapeInPlace(new [] { y.Shape[0], 1, y.Shape[1] });
            }
        }

        if (ShouldEmbedEachElementOfLastDimension)
        {
            Debug.Assert(allEmbeddingTensors.Count == 1);
            y.WordEmbeddingForwardPropagation(x, allEmbeddingTensors[0], 0, 0, 0, 0);
        }
        else
        {
            for (var i = 0; i < EmbeddingDescriptions.Count; i++)
            {
                var embeddingTensor = allEmbeddingTensors[EmbeddingDescriptions[i].embeddingTensorIndex];
                var xIndexInLastDimensionToUse = EmbeddingDescriptions[i].indexInLastDimensionToUse;
                int copyCountBeforeIndex = (i == 0) ? xIndexInLastDimensionToUse : (xIndexInLastDimensionToUse - EmbeddingDescriptions[i-1].indexInLastDimensionToUse - 1);
                int copyCountAfterIndex = (i == EmbeddingDescriptions.Count - 1) ? x.Shape[2] - xIndexInLastDimensionToUse - 1 : 0;
                y.WordEmbeddingForwardPropagation(x, embeddingTensor, xIndexInLastDimensionToUse, deltaForIndexesInLastDimensionToUse + xIndexInLastDimensionToUse, copyCountBeforeIndex, copyCountAfterIndex);
                deltaForIndexesInLastDimensionToUse += embeddingTensor.Shape[1] - 1;
            }
        }

        x.ReshapeInPlace(xOriginalShape);
       