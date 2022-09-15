using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using SharpNet.Datasets;
using SharpNet.GPU;
using SharpNet.HPO;
using SharpNet.Layers;
// ReSharper disable FieldCanBeMadeReadOnly.Global

// ReSharper disable MemberCanBePrivate.Global

namespace SharpNet.Networks;

public class NetworkSample_1DCNN : NetworkSample
{
    // ReSharper disable once EmptyConstructor
    public NetworkSample_1DCNN()
    {
    }

    #region Hyper-Parameters
    //Embedding of the categorical features
    public int EmbeddingDim = 10;