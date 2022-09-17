using System.IO;
using log4net;
using SharpNet.Datasets;
using SharpNet.TextPreprocessing;
// ReSharper disable MemberCanBePrivate.Global

namespace SharpNet.Networks.Transformers;

public class CharLevelTransformersDatasetSample : TransformerDatasetSample
{
    #region private static fields

    private readonly string _fullText;

    #endregion

    #region public fields & properties
    // ReSharper disable once UnusedMember.Global
    public static readonly ILog Log = LogManager.GetLogger(typeof(CharLevelTransformersDatasetSample));
    #endregion

    #region Hyperparameters
    /// <summary>
    /// the maximum size fo the text that will be used for training the network
    /// -1 means the full text (default value)
    /// other values are used to speed up the training process (usually for testing purpose)
    /// </summary>
    // ReSharper disable once ConvertToConstant.Global
    // ReSharper disable once FieldCanBeMadeReadOnly.Global
    public int MaxCharacterLengthForTraining = -1;

    #endregion

    static CharLevelTransformersDatasetSample()
    {
        Utils.ConfigureGlobalLog4netProperties(TextTransformersUtils.WorkingDirectory, "log");
        Utils.ConfigureThreadLog4netProperties(TextTransformersUtils.WorkingDi