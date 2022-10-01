using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using log4net;
using SharpNet.CPU;
using SharpNet.Datasets;
using SharpNet.HPO;
using SharpNet.Hyperparameters;
// ReSharper disable UnusedMember.Global

namespace SharpNet.Networks.Transformers;

public static class TextTransformersUtils
{
    private const string NAME = "TextTransformers";


    #region public fields & properties
    private static readonly ILog Log = LogManager.GetLogger(typeof(Tex