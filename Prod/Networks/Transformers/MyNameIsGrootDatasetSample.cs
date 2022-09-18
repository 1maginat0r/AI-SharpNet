using System.Text;
using log4net;
using SharpNet.Datasets;
using SharpNet.TextPreprocessing;
// ReSharper disable MemberCanBePrivate.Global

namespace SharpNet.Networks.Transformers;

public class MyNameIsGrootDatasetSample : TransformerDatasetSample
{
    #region private static fields

    private readonly string _fullText;

    #endregion

    #region public fields & properties
    // ReSharper disable once UnusedMember.Global
    public static readonly ILog Log = LogManager.GetLogger(typeof(MyNameIsGrootDatasetSample));
    #endregion


    static MyNameIsGrootDatasetSample()
    {
        Utils.ConfigureGlobalLog4netProperties(MyNameIsGrootUtils.WorkingDirectory, "log");
        Utils.ConfigureThreadLog4netProperties(MyNameIsGrootUtils.WorkingDirectory, "log");
    }

    private const string sentence = "my name is groot";

    public MyNameIsG