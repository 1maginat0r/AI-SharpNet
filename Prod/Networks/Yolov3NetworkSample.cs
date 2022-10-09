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
                    CompatibilityMode = CompatibilityModeEnum.TensorFlow,
                    lambdaL2Regularization = 0.0005,
                    ResourceIds = resourceIds.ToList(),
            }
                .WithSGD(0.9, false)
                .WithCyclicCosineAnnealingLearningRateScheduler(10, 2);
            return config;
        }
        #endregion

        // ReSharper disable once UnusedMember.Global
        public Network Build()
        {
            LoadNetDescription();
            var network = BuildNetworkWithoutLayers(Path.Combine(DefaultWorkingDirectory, "YOLO"), "YOLO V3");

            network.Input(InputShape_CHW[0], InputShape_CHW[1], InputShape_CHW[2], "input_1");

            for (int i = 1; i < _blocks.Count; ++i)
            {
                switch (_blocks[i].Item1)
                {
                    case "convolutional":
                        AddConvolution(network, i);
                        break;
                    case "shortcut":
                        AddShortcut(network, i);
                        break;
                    case "upsample":
                        AddUpSample(network, i);
                        break;
                    case "route":
                        AddRoute(network, i);
                        break;
                    case "yolo":
                        AddYolo(network, i);
                        break;
                }
            }

            var yoloLayers = network.Layers.Where(l => l.GetType() == typeof(YOLOV3Layer)).Select(l => l.LayerIndex).ToArray();
            Debug.Assert(yoloLayers.Length == 3);

            //all predictions of the network
            network.ConcatenateLayer(yoloLayers, "tf_op_layer_concat_6");

            //we remove (set box confidence to 0) low predictions  after non max suppression
            network.NonMaxSuppression(MinScore, IOU_threshold_for_duplicate, MaxOutputSize, MaxOutputSizePerClass, "NonMaxSuppression");

            return network;
        }

        private void LoadNetDescription()
        {
            if (_blocks[0].Item1 != "net")
            {
                throw new ArgumentException("the first item in the config must be 'net'");
            }
            var block = _blocks[0].Item2;
            InputShape_CHW = new[] { int.Parse(block["channels"]), int.Parse(block["height"]), int.Parse(block["width"]) };
            if (block.TryGetValue("momentum", out var value))
            {
                BatchNormMomentum = double.Parse(value, CultureInfo.InvariantCulture);
            }
            _blockIdToLastLayerIndex[0] = 0;
            //TODO: support decay
            //other fields (ignored) : 
            //        batch=64
            //        subdivisions=16
            //        decay=0.0005
            //        angle=0
            //        saturation = 1.5
            //        exposure = 1.5
            //        hue=.1

        }
        private void AddConvolution(Network network, int blockId)
        {
            var block = _blocks[blockId].Item2;
            bool batch_normalize = block.ContainsKey("batch_normalize") && int.Parse(block["batch_normalize"]) >= 1;
            var stride = int.Parse(block["stride"]);
            int lastLayerIndex = _blockIdToLastLayerIndex[blockId - 1];
            if (stride > 1)
            {
                var layerName = GetLayerName(network, "zero_padding2d", typeof(ZeroPadding2DLayer));
                network.ZeroPadding2D(1, 0, 1, 0, lastLayerIndex, layerName);
                lastLayerIndex = network.LastLayerIndex;
            }
            network.Convolution(
                int.Parse(block["filters"]),
                int.Parse(block["size"]),
                stride,
                (stride > 1) ? ConvolutionLayer.PADDING_TYPE.VALID:ConvolutionLayer.PADDING_TYPE.SAME,
                0.0, //lambdaL2Regularization
                !batch_normalize,
                lastLayerIndex,
                "conv_" + (blockId - 1));
            if (batch_normalize)
            {
                network.BatchNorm(BatchNormMomentum, BatchNormEpsilon, "bnorm_" + (blockId - 1));
            }
            if (block.ContainsKey("activation") && !block["activation"].Equals("linear"))
            {
                var activationFunction = ExtractActivation(block["activation"], out var alphaActivation);
                network.Activation(activationFunction, alphaActivation, "leaky_" + (blockId - 1));
            }
            _blockIdToLastLayerIndex[blockId] = network.LastLayerIndex;
        }
        private void AddShortcut(Network network, int blockId)
        {
            var block = _blocks[blockId].Item2;
            var layerName = GetLayerName(network, "tf_op_layer_AddV2", typeof(AddLayer));
            network.AddLayer(_blockIdToLastLayerIndex[blockId - 1], BlockOffsetToLastLayerIndex(int.Parse(block["from"]), block