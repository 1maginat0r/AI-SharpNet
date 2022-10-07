using SharpNet.GPU;
using SharpNet.Layers;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.Linq;
using SharpNet.Data;
using System.Globalization;

namespace SharpNet.Networks
{
    [SuppressMessage("ReSharper", "InconsistentNaming")]
    [SuppressMessage("ReSharper", "ConvertToConstant.Local")]
    [SuppressMessage("ReSharper", "FieldCanBeMadeReadOnly.Local")]
    public class Yolov3NetworkSample : NetworkSample
    {
        #region fields & properties
        private List<Tuple<string, Dictionary<string, string>>> _blocks;
        private static List<Tuple<string, Dictionary<string, string>>> YOLOV3Config => ExtractConfigFileFromContent(Utils.LoadResourceContent(typeof(Yolov3NetworkSample).Assembly, "SharpNet.ObjectDetection.yolov3.cfg"));
        private readonly IDictionary<int, int> _blockIdToLastLayerIndex = new Dictionary<int, int>();
        private int[] InputShape_CHW = { 3, 608, 608 };
        private double BatchNormMomentum = 0.99;
        private double BatchNormEpsilon = 0.001;
        private double Alpha_LeakyRelu = 0.1;
        
        private float MinScore = 0.5f;
        private float IOU_threshold_for_duplicate = 0.5f;

        private int MaxOutputSize = int.MaxValue;
        private int MaxOutputSizePerClass = int.MaxValue;
        #endregion

        #region constructor

        private Yolov3NetworkSample([JetBrains.Annotations.NotNull] List<Tuple<string, Dictionary<string, string>>> blocks)
        {
            _blocks = blocks;
        }

        // ReSharper disable once UnusedMember.Global
        public static Yolov3NetworkSample ValueOf(List<int> resourceIds, List<Tuple<string, Dictionary<string, string>>> blocks = null)
        {
            var config = (Yolov3NetworkSample)new Yolov3NetworkSample(blocks ?? YOLOV3Config)
            {
                    LossFunction = EvaluationMetricEnum.CategoricalCrossentropy,
                    CompatibilityMode = Compatibility