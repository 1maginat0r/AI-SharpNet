﻿using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using JetBrains.Annotations;
using SharpNet.CatBoost;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.DataAugmentation;
using SharpNet.GPU;
using SharpNet.HPO;
using SharpNet.Hyperparameters;
using SharpNet.LightGBM;
using SharpNet.Networks;
using SharpNet.Networks.Transformers;
using static SharpNet.Networks.NetworkSample;

namespace SharpNet.Datasets.Biosonar85;

public static class Biosonar85Utils
{
    public const string NAME = "Biosonar85";
    public static string WorkingDirectory => Path.Combine(Utils.ChallengesPath, NAME);
    public static string DataDirectory => Path.Combine(WorkingDirectory, "Data");

    public static void Run()
    {
        Utils.ConfigureGlobalLog4netProperties(WorkingDirectory, NAME, true);
        Utils.ConfigureThreadLog4netProperties(WorkingDirectory, NAME, true);

        //Mixup.DisplayStatsForAlphaMixup();return;

        //var trainDataset = Load("X_train_23168_101_64_1024_512.bin", "Y_train_23168_1_64_1024_512.bin", true);
        //var testDataset = Load("X_test_950_101_64_1024_512.bin", "Y_test_950_1_64_1024_512.bin", false);

        //var bin_file = Path.Combine(DataDirectory, "Y_train_ofTdMHi.csv.bin");
        //var tensor = CpuTensor<float>.LoadFromBinFile(bin_file, new[] { -1, 101, 64});
        //ParseLogFile(); return;

        //Log.Info(AllCombinations(0.4, 0.7));return;
        //Log.Info(AllCombinations(0.701, 0.85));return;

        //ChallengeTools.Retrain(Path.Combine(WorkingDirectory, "Dump"), "2A4F619211", null, percentageInTraining:0.5, retrainOnFullDataset:false, useAllAvailableCores:true);return;
        //ChallengeTools.Retrain(Path.Combine(WorkingDirectory, "Dump"), "FCB789043E_18", null, percentageInTraining: 0.5, retrainOnFullDataset: false, useAllAvailableCores: true); return;
        //ChallengeTools.Retrain(Path.Combine(WorkingDirectory, "Dump"), "9811DDD19E_19", null, percentageInTraining: 0.5, retrainOnFullDataset: false, useAllAvailableCores: true); return;
        //ChallengeTools.Retrain(Path.Combine(WorkingDirectory, "Dump"), "E5BA77E393", null, percentageInTraining: 0.5, retrainOnFullDataset: false, useAllAvailableCores: true); return;


        //ChallengeTools.ComputeAndSaveFeatureImportance(WorkingDirectory, "674CD08C52", true); return;


        //ComputeAverage_avg();return;

        //Launch_HPO_spectrogram(20); return;

        //Launch_HPO_MEL_SPECTROGRAM_256_801_BinaryCrossentropy(20); return;
        Launch_HPO_MEL_SPECTROGRAM_256_801_BCEWithFocalLoss(20); return;
        //Launch_HPO_MEL_SPECTROGRAM_64_401(10); return;
        //Launch_HPO_MEL_SPECTROGRAM_SMALL_128_401(10); return;
        //Launch_HPO_MEL_SPECTROGRAM_128_401_BinaryCrossentropy(20); return;
        //Launch_HPO_MEL_SPECTROGRAM_128_401_BCEWithFocalLoss(10); return;

        //LaunchCatBoostHPO(1000); return;

        //LaunchLightGBMHPO(350);return;

        //Launch_HPO_Transformers(10); return;
        //Launch_HPO(10);return;
        //OptimizeModel_AE0F13543C();
    }


    public static void OptimizeModel_AE0F13543C()
    {
        var eachUpdate = new List<Tuple<string, string>>
                         {
                             Tuple.Create("HeightShiftRangeInPercentage","0.03"),
                             Tuple.Create("TopDropoutRate","0.2"),
                             Tuple.Create("HeightShiftRangeInPercentage","0.1"),
                             Tuple.Create("SkipConnectionsDropoutRate","0.5"),
                             Tuple.Create("SkipConnectionsDropoutRate","0.3"),
                             Tuple.Create("AlphaMixup","1"),
                             Tuple.Create("InitialLearningRate","0.002"),
                             Tuple.Create("InitialLearningRate","0.003"),
                             Tuple.Create("CutoutPatchPercentage","0.1"),
                             Tuple.Create("CutoutPatchPercentage","0.2"),
                             Tuple.Create("RowsCutoutPatchPercentage","0.06"),
                             Tuple.Create("RowsCutoutPatchPercentage","0.18"),
                             //Tuple.Create("TopDropoutRate","0.4"), //done
                         };
        void ContentUpdate(string key, string newValue, IDictionary<string, string> content)
        {
            if (content.ContainsKey(key))
            {
                content[key] = newValue;
            }
        }

        foreach(var (key, newValue) in eachUpdate)
        {
            ISample.Log.Info($"Updating {key} to {newValue}");
            ChallengeTools.RetrainWithContentUpdate(Path.Combine(WorkingDirectory, "OptimizeModel", "1EF57E45FC_16"), "1EF57E45FC_16", a => ContentUpdate(key, newValue, a));
        }
    }

    // ReSharper disable once UnusedMember.Global
    public static string AllCombinations(double minPercentage, double maxPercentage)
    {
        var siteToCount = SiteToCount();

        double totalCount = siteToCount.Select(m => m.Value).Sum();
        List<Tuple<string, double>> allSites = new();
        foreach( var (site,count) in siteToCount)
        {
            allSites.Add(Tuple.Create(site,count/totalCount));
        }
        allSites = allSites.OrderByDescending(s => s.Item2).ToList();

        List<Tuple<string, double>> res = new();
        AllCombinations_Helper(0, 0, new List<string>(), allSites, res);
        res = res.Where(s => s.Item2 >= minPercentage && s.Item2 <= maxPercentage).ToList();
        var allAvailableCombinations = res.OrderBy(s=>s.Item2).Select(s=>s.Item1).ToList();
        return string.Join(", ", allAvailableCombinations);
    }

    private static Dictionary<string, int> SiteToCount()
    {
        var siteToCount = new Dictionary<string, int>();
        var ids = DataFrame.read_string_csv(Biosonar85DatasetSample.Y_train_path).StringColumnContent("id");
        foreach (var id in ids)
        {
            var site = IdToSite(id);
            if (!siteToCount.ContainsKey(site))
            {
                siteToCount[site] = 0;
            }
            ++siteToCount[site];
        }
        return siteToCount;
    }

    // ReSharper disable once UnusedMember.Local
    private static void ParseLogFile()
    {
        const string log_directory = @"C:\Projects\Challenges\Biosonar85\MandatorySitesForTraining";
        var csvPath = Path.Combine(log_directory, "data.csv");
        Dictionary<string, int> siteToCount = SiteToCount();
        foreach (var logPath in Directory.GetFiles(log_directory, "*.log"))
        {
            var lines = File.ReadAllLines(logPath);
            var currentBlock = new List<string>();

            for (int i = 0; i < lines.Length; ++i)
            {
                if (lines[i].Contains("sites for training"))
                {
                    ProcessBlock(currentBlock, siteToCount, csvPath);
                    currentBlock.Clear();
                    currentBlock.Add(lines[i]);
                }
                else
                {
                    currentBlock.Add(lines[i]);
                    if (i == lines.Length - 1)
                    {
                        ProcessBlock(currentBlock, siteToCount, csvPath);
                        currentBlock.Clear();
                    }
                }
            }
        }

    }
    private static void ProcessBlock(List<string> lines, Dictionary<string, int> siteToCount, string csvPath)
    {
        var sites = lines.FirstOrDefault(l => l.Contains("sites for training"));
        if (string.IsNullOrEmpty(sites))
        {
            return; 
        }
        sites = sites.Split().Last();
        double totalCount = siteToCount.Select(m => m.Value).Sum();
        double siteCount = 0;
        foreach (var site in sites.Split(','))
        {
            siteCount += siteToCount[site];
        }   
        var networkName = lines.FirstOrDefault(l => l.Contains("Network Name:"));
        if (string.IsNullOrEmpty(networkName))
        {
            return ;
        }
        networkName = networkName.Split().Last();
        int bestEpoch = -1;
        float bestScore = 0;
        for (int epoch = 1;; ++epoch)
        {
            var epochLine = lines.FirstOrDefault(l => l.Contains($"Epoch {epoch}/20"));
            if (string.IsNullOrEmpty(epochLine))
            {
                break;
            }
            const string toFind = "val_accuracy:";
            var index = epochLine.IndexOf(toFind, StringComparison.Ordinal);
            if (index < 0)
            {
                break;
            }
            var score = float.Parse(epochLine.Substring(index + toFind.Length).Trim().Split()[0]);
            if (bestEpoch == -1 || score > bestScore)
            {
                bestEpoch = epoch;
                bestScore = score;
            }
        }

        if (bestEpoch == -1)
        {
            return;
        }

        if (!File.Exists(csvPath))
        {
            File.AppendAllText(csvPath, "Sep=;"+Environment.NewLine);
            File.AppendAllText(csvPath, $"NetworkName;Sites;% in training;BestEpoch;BestScore" + Environment.NewLine);
        }
        File.AppendAllText(csvPath, $"{networkName};{sites};{siteCount / totalCount};{bestEpoch};{bestScore}" + Environment.NewLine);
    }



    private static void AllCombinations_Helper(int depth, double totalCount, List<string> currentPath, List<Tuple<string, double>> allSites, List<Tuple<string, double>> res)
    {
        if (depth>=allSites.Count)
        {
            res.Add(Tuple.Create("\""+string.Join(",", currentPath)+"\"", totalCount));
            return;
        }

        //without site at index 'depth'
        AllCombinations_Helper(depth + 1, totalCount, currentPath, allSites, res);

        //with site at index 'depth'
        currentPath.Add(allSites[depth].Item1);
        AllCombinations_Helper(depth + 1, totalCount+ allSites[depth].Item2, currentPath, allSites, res);
        currentPath.RemoveAt(currentPath.Count - 1);
    }

    public static string IdToSite(string id) 
    {
        return id.Split(new[] { '-', '.' })[1]; 
    }



    // ReSharper disable once UnusedMember.Global
    public static void ComputeAverage_avg()
    {
        var dfs = new List<DataFrame>();
        const string path = @"\\RYZEN2700X-DEV\Challenges\Biosonar85\Submit\";
        foreach (var file in new[]
                 {
                     @"7E45F84676_predict_test_0,9353867531264475.csv",
                     @"569C5C14D2_predict_test_0.936063704706595.csv",
                 })
        {
            dfs.Add(DataFrame.read_csv(Path.Combine(path, file), true, x => x == "id" ? typeof(string) : typeof(float)));
        }
        DataFrame.Average(dfs.ToArray()).to_csv(Path.Combine(path, "7E45F84676_569C5C14D2_avg.csv"));
    }


    public static (int[] shape, string n_fft, string hop_len, string f_min, string f_max, string top_db) ProcessXFileName(string xPath)
    {
        var xSplitted = Path.GetFileNameWithoutExtension(xPath).Split("_");
        var xShape = new[] { int.Parse(xSplitted[^8]), int.Parse(xSplitted[^7]), int.Parse(xSplitted[^6]) };
        var n_fft = xSplitted[^5];
        var hop_len = xSplitted[^4];
        var f_min = xSplitted[^3];
        var f_max = xSplitted[^2];
        var top_db = xSplitted[^1];
        return (xShape, n_fft, hop_len, f_min, f_max, top_db);
    }

    public static InMemoryDataSet Load(string xFileName, [CanBeNull] string yFileNameIfAny, string csvPath, float mean = 0f, float stdDev = 1f)
    {
        
        var meanAndVolatilityForEachChannel = new List<Tuple<float, float>> { Tuple.Create(mean, stdDev) };
        
        var xPath = Path.Join(DataDirectory, xFileName);
        (int[] xShape, var  _, var _, var _, var _, var _) = ProcessXFileName(xPath);

        ISample.Log.Info($"Loading {xShape[0]} tensors from {xPath} with shape {Tensor.ShapeToString(xShape)} (Y file:{yFileNameIfAny})");


        var xTensor = CpuTensor<float>.LoadFromBinFile(xPath, xShape);
        var yTensor = string.IsNullOrEmpty(yFileNameIfAny)
            ?null //no Y available for Dataset
            :CpuTensor<float>.LoadFromBinFile(Path.Join(DataDirectory, yFileNameIfAny), new []{ xShape[0], 1 });

        /*
        //mean = 0f; stdDev = 1f; //!D //no standardization
        // we disable standardization
        var xAccBefore = new DoubleAccumulator();
        xAccBefore.Add(xTensor.SpanContent);
        Log.Info($"Stats for {xFileName} before standardization: {xAccBefore}");

        //We standardize the input
        Log.Info($"Mean: {mean}, StdDev: {stdDev}");
        xTensor.LinearFunction(1f / stdDev, xTensor, -mean / stdDev);

        var xAccAfter = new DoubleAccumulator();
        xAccAfter.Add(xTensor.SpanContent);
        Log.Info($"Stats for {xFileName} after standardization: {xAccAfter}");
        */

        var yID = DataFrame.read_string_csv(csvPath).StringColumnContent("id");

        var dataset = new Biosonar85InMemoryDataSet(
            xTensor,
            yTensor,
            xFileName,
            meanAndVolatilityForEachChannel,
            yID);
        return dataset;
    }


    // ReSharper disable once RedundantAssignment
    // ReSharper disable once RedundantAssignment
    public static TensorListDataSet LoadTensorListDataSet(string xFileName, string xAugmentedFileNameIfAny, [CanBeNull] string yFileNameIfAny, string csvPath, float mean, float stdDev)
    {
        //!D we disable standardization
        mean = 0; stdDev = 1;

        var xPath = Path.Join(DataDirectory, xFileName);

        (int[] xShape, var _, var _, var _, var _, var _) = ProcessXFileName(xPath);

        List<CpuTensor<float>> xTensorList = CpuTensor<float>.LoadTensorListFromBinFileAndStandardizeIt(xPath, xShape , mean, stdDev);

        List<CpuTensor<float>> augmentedXTensorList = null;
        if (!string.IsNullOrEmpty(xAugmentedFileNameIfAny))
        {
            augmentedXTensorList = CpuTensor<float>.LoadTensorListFromBinFileAndStandardizeIt(Path.Join(DataDirectory, xAugmentedFileNameIfAny), xShape, mean, stdDev);
        }

        var yTensor = string.IsNullOrEmpty(yFileNameIfAny)
            ? null //no Y available for Dataset
            : CpuTensor<float>.LoadFromBinFile(Path.Join(DataDirectory, yFileNameIfAny), new[] { xShape[0], 1 });

        /*
        if (string.IsNullOrEmpty(yFileNameIfAny) && xShape[0] == 950)
        {
            var y_test_expected = DataFrame.read_csv(@"\\RYZEN2700X-DEV\Challenges\Biosonar85\Submit\2E3950406D_19_D560131427_45_avg_predict_test_0.956247550504151.csv", true, (col) => col == "id" ? typeof(string) : typeof(float));
            yTensor = new CpuTensor<float>(new[] { xShape[0], 1 }, y_test_expected.FloatColumnContent(y_test_expected.Columns[1]));
        }
        */

        /*
        ISample.Log.Info($"Standardization between -1 and +1");
        foreach (var t in xTensorList)
        {
            var xAccBefore = new MathTools.DoubleAccumulator();
            xAccBefore.Add(t.SpanContent);

            if (xAccBefore.Max > xAccBefore.Min)
            {
                mean = (float)xAccBefore.Average;
                var max = xAccBefore.Max-mean;
                var min = xAccBefore.Min-mean;
                var divider = (float) Math.Max(Math.Abs(min), Math.Abs(max));
                // x= (x-mean)/divider

                //We standardize the input between -1 and +1
                t.LinearFunction(1f / divider, t, -mean / divider);
            }
        }
        */

        var yID = DataFrame.read_string_csv(csvPath).StringColumnContent("id");

        var dataset = new TensorListDataSet(
            xTensorList,
            augmentedXTensorList,
            yTensor,
            xFileName,
            Objective_enum.Classification,
            null,
            null /* columnNames*/, 
            null, /* isCategoricalColumn */
            yID,
            "id",
            ',');
        return dataset;
    }


    // ReSharper disable once UnusedMember.Global
    // ReSharper disable once UnusedMember.Local
    private static void Launch_HPO_Transformers(int numEpochs = 10, int maxAllowedSecondsForAllComputation = 0)
    {
        var searchSpace = new Dictionary<string, object>
        {
            //{ nameof(AbstractDatasetSample.PercentageInTraining), 0.5},
            { nameof(AbstractDatasetSample.PercentageInTraining), new[]{0.5,0.8}},
            
            { nameof(AbstractDatasetSample.ShuffleDatasetBeforeSplit), true},
            { nameof(Biosonar85DatasetSample.InputDataType), nameof(Biosonar85DatasetSample.InputDataTypeEnum.TRANSFORMERS_3D)},
            { nameof(NetworkSample.MinimumRankingScoreToSaveModel), 0.87},

            //related to model
            { nameof(NetworkSample.LossFunction), nameof(EvaluationMetricEnum.BCEWithFocalLoss)},
            { nameof(NetworkSample.EvaluationMetrics), nameof(EvaluationMetricEnum.Accuracy)/*+","+nameof(EvaluationMetricEnum.AUC)*/},
            { nameof(NetworkSample.BCEWithFocalLoss_Gamma), new []{0, /*0.35*/}},
            { nameof(NetworkSample.BCEWithFocalLoss_PercentageInTrueClass), 0.5},
            //{ nameof(NetworkSample.BatchSize), new[] {1024} },
            { nameof(NetworkSample.BatchSize), new[] {256} },

            {nameof(TransformerNetworkSample.embedding_dim), 64},
            {nameof(TransformerNetworkSample.input_is_already_embedded), true },
            {nameof(TransformerNetworkSample.encoder_num_transformer_blocks), new[]{4} },
            {nameof(TransformerNetworkSample.encoder_num_heads), new[]{8} },
            {nameof(TransformerNetworkSample.encoder_mha_use_bias_Q_V_K), new[]{false /*,true*/ } },
            {nameof(TransformerNetworkSample.encoder_mha_use_bias_O), true  }, // must be true
            {nameof(TransformerNetworkSample.encoder_mha_dropout), new[]{0.2, 0.4 } },
            {nameof(TransformerNetworkSample.encoder_feed_forward_dim), 4*64},
            {nameof(TransformerNetworkSample.encoder_feed_forward_dropout), new[]{ 0.2,0.4 }}, //0.2
            {nameof(TransformerNetworkSample.encoder_use_causal_mask), true},
            {nameof(TransformerNetworkSample.output_shape_must_be_scalar), true},
            {nameof(TransformerNetworkSample.pooling_before_dense_layer), new[]{ nameof(POOLING_BEFORE_DENSE_LAYER.NONE) /*,nameof(POOLING_BEFORE_DENSE_LAYER.GlobalAveragePooling), nameof(POOLING_BEFORE_DENSE_LAYER.GlobalMaxPooling)*/ } }, //must be NONE
            {nameof(TransformerNetworkSample.layer_norm_before_last_dense), false}, // must be false

            { nameof(NetworkSample.NumEpochs), new[] { numEpochs } },

            { nameof(NetworkSample.ShuffleDatasetBeforeEachEpoch), true},
            // Optimizer 
            { nameof(NetworkSample.OptimizerType), new[] { "AdamW" } },
            //{ nameof(NetworkSample.lambdaL2Regularization), 0.0005 },
            { nameof(NetworkSample.lambdaL2Regularization), 0},
            
            { nameof(NetworkSample.AdamW_L2Regularization), 0.00005},
            //{ nameof(NetworkSample.AdamW_L2Regularization), new[]{0.00001,0.00005, 0.0001,0.0005,0.001, 0.005,0.01}},

            // Learning Rate
            { nameof(NetworkSample.InitialLearningRate),0.005},
            //{ nameof(NetworkSample.InitialLearningRate),new[]{0.0001,0.0005,0.001, 0.005, 0.01, 0.05, 0.1}},

            // Learning Rate Scheduler
            //{nameof(NetworkSample.LearningRateSchedulerType), new[]{"OneCycle", "CyclicCosineAnnealing" } },
            {nameof(NetworkSample.LearningRateSchedulerType), "OneCycle" },
            //{nameof(NetworkSample.LearningRateSchedulerType), new[]{ "CyclicCosineAnnealing" } },
            {nameof(TransformerNetworkSample.LastActivationLayer), nameof(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID)},
            {nameof(NetworkSample.DisableReduceLROnPlateau), true},
            {nameof(NetworkSample.OneCycle_DividerForMinLearningRate), 20},
            {nameof(NetworkSample.OneCycle_PercentInAnnealing),0.1},
            {nameof(NetworkSample.CyclicCosineAnnealing_MinLearningRate), 1e-6},
            {nameof(NetworkSample.CyclicCosineAnnealing_nbEpochsInFirstRun), 10},
            {nameof(NetworkSample.CyclicCosineAnnealing_nbEpochInNextRunMultiplier), 2},

            
            // DataAugmentation
            { nameof(NetworkSample.DataAugmentationType), nameof(ImageDataGenerator.DataAugmentationEnum.DEFAULT) },
            { nameof(NetworkSample.AlphaCutMix), new[]{0,0.5215608}},
            { nameof(NetworkSample.AlphaMixup), new[]{0,1.2}},
            { nameof(NetworkSample.CutoutPatchPercentage), new[]{0,0.06450188,0.1477559}},
            { nameof(NetworkSample.CutoutCount), 1 },
            //{ nameof(NetworkSample.RowsCutoutPatchPercentage), new[]{0.12661687}  },
            { nameof(NetworkSample.RowsCutoutPatchPercentage), new[]{ 0, 0.047216333 }  },
            { nameof(NetworkSample.RowsCutoutCount), 1 },
            { nameof(NetworkSample.ColumnsCutoutPatchPercentage), new[] {  0, 0.007914994} },
            { nameof(NetworkSample.ColumnsCutoutCount), new[] { 1} },
            //{ nameof(NetworkSample.FillMode),new[]{ nameof(ImageDataGenerator.FillModeEnum.Reflect)} },
            { nameof(NetworkSample.FillMode),new[]{ nameof(ImageDataGenerator.FillModeEnum.Nearest)} },
            { nameof(NetworkSample.WidthShiftRangeInPercentage), new[] { 0, 0.034252543 } },
            { nameof(NetworkSample.HeightShiftRangeInPercentage), new[] {0, 0.06299627, 0.008304262 } },
            


        };

        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(new TransformerNetworkSample(), new Biosonar85DatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        const bool retrainOnFullDatasetIfBetterModelFound = false;
        hpo.Process(t => SampleUtils.TrainWithHyperparameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, retrainOnFullDatasetIfBetterModelFound, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
    }

    // ReSharper disable once UnusedMember.Global
    // ReSharper disable once UnusedMember.Local
    private static void Launch_HPO(int numEpochs = 10, int maxAllowedSecondsForAllComputation = 0)
    {
        var searchSpace = new Dictionary<string, object>
        {
            //{"KFold", 2},
            {nameof(AbstractDatasetSample.PercentageInTraining), 0.8}, //will be automatically set to 1 if KFold is enabled
            
            //Related to model
            { nameof(NetworkSample.LossFunction), nameof(EvaluationMetricEnum.BinaryCrossentropy)},
            { nameof(NetworkSample.EvaluationMetrics), nameof(EvaluationMetricEnum.AUC)},
            { nameof(NetworkSample.BatchSize), new[] {256} },
            { nameof(NetworkSample.NumEpochs), new[] { numEpochs } },
            { nameof(NetworkSample.ShuffleDatasetBeforeEachEpoch), true},
            // Optimizer 
            { nameof(NetworkSample.OptimizerType), new[] { "AdamW" } },
            //{ nameof(NetworkSample.OptimizerType), new[] { "SGD"} },
            //{ nameof(NetworkSample.AdamW_L2Regularization), AbstractHyperparameterSearchSpace.Range(0.00001f,0.01f, AbstractHyperparameterSearchSpace.range_type.normal) },
            //{ nameof(NetworkSample.AdamW_L2Regularization), AbstractHyperparameterSearchSpace.Range(0.00001f,0.01f, AbstractHyperparameterSearchSpace.range_type.normal) },
            { nameof(NetworkSample.AdamW_L2Regularization), 0.01 },

            //Dataset
            { nameof(AbstractDatasetSample.ShuffleDatasetBeforeSplit), true},
            { nameof(Biosonar85DatasetSample.InputDataType), nameof(Biosonar85DatasetSample.InputDataTypeEnum.NETWORK_4D)},


            //{ "Use_MaxPooling", new[]{true,false}},
            //{ "Use_AvgPooling", new[]{/*true,*/false}}, //should be false
                

            // DataAugmentation
            { nameof(NetworkSample.DataAugmentationType), nameof(ImageDataGenerator.DataAugmentationEnum.DEFAULT) },
            { nameof(NetworkSample.AlphaCutMix), 0.5}, //must be > 0
            { nameof(NetworkSample.AlphaMixup), new[] { 0 /*, 0.25*/} }, // must be 0
            { nameof(NetworkSample.CutoutPatchPercentage), new[] {0, 0.1,0.2} },
            { nameof(NetworkSample.RowsCutoutPatchPercentage), 0.2 },
            { nameof(NetworkSample.ColumnsCutoutPatchPercentage), new[] {0.1, 0.2} },
            //{ nameof(NetworkSample.HorizontalFlip),new[]{true,false } },
            //{ nameof(NetworkSample.VerticalFlip),new[]{true,false } },
            //{ nameof(NetworkSample.Rotate180Degrees),new[]{true,false } },
            { nameof(NetworkSample.FillMode),nameof(ImageDataGenerator.FillModeEnum.Modulo) },
            { nameof(NetworkSample.WidthShiftRangeInPercentage), 0.1 },
            //{ nameof(NetworkSample.HeightShiftRangeInPercentage), new[] { 0.0 , 0.1,0.2 } }, //0
            //{ nameof(NetworkSample.ZoomRange), new[] { 0.0 , 0.05 } },

            

            //{ nameof(NetworkSample.SGD_usenesterov), new[] { true, false } },
            //{ nameof(NetworkSample.lambdaL2Regularization), new[] { 0.0005, 0.001, 0.00005 } },
            //{ nameof(NetworkSample.lambdaL2Regularization), new[] {0.001, 0.0005, 0.0001, 0.00005 } }, // 0.0001 or 0.001
            //{ nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount), new[]{5}},
            //{ nameof(EfficientNetNetworkSample.LastActivationLayer), nameof(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID)},

            // Learning Rate
            //{ nameof(NetworkSample.InitialLearningRate), new []{0.01, 0.1 }}, //SGD: 0.01 //AdamW: 0.01 or 0.001
            //{ nameof(NetworkSample.InitialLearningRate), AbstractHyperparameterSearchSpace.Range(0.001f,0.2f,AbstractHyperparameterSearchSpace.range_type.normal)},
            { nameof(NetworkSample.InitialLearningRate), 0.005}, 
            // Learning Rate Scheduler
            //{ nameof(NetworkSample.LearningRateSchedulerType), new[] { "OneCycle" } },
            { nameof(NetworkSample.LearningRateSchedulerType), "CyclicCosineAnnealing" },
        };

        //var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(DefaultEfficientNetNetworkSample(), new Biosonar85DatasetSample()), WorkingDirectory);
        //var hpo = new RandomSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(DefaultEfficientNetNetworkSample(), new Biosonar85DatasetSample()), WorkingDirectory);
         var hpo = new RandomSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(new Biosonar85NetworkSample(), new Biosonar85DatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        const bool retrainOnFullDatasetIfBetterModelFound = false;
        hpo.Process(t => SampleUtils.TrainWithHyperparameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, retrainOnFullDatasetIfBetterModelFound, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
    }

    // ReSharper disable once UnusedMember.Global
    // ReSharper disable once UnusedMember.Local
    private static void Launch_HPO_spectrogram(int numEpochs = 10, int maxAllowedSecondsForAllComputation = 0)
    {
        var searchSpace = new Dictionary<string, object>
        {
            //related to Dataset 
            //{"KFold", 2},
            
            { nameof(AbstractDatasetSample.PercentageInTraining), 0.8}, //will be automatically set to 1 if KFold is enabled

            { nameof(AbstractDatasetSample.ShuffleDatasetBeforeSplit), true},
            //{ nameof(Biosonar85DatasetSample.InputDataType), nameof(Biosonar85DatasetSample.InputDataTypeEnum.PNG_1CHANNEL)},
            { nameof(Biosonar85DatasetSample.InputDataType), nameof(Biosonar85DatasetSample.InputDataTypeEnum.PNG_1CHANNEL_V2)},
            { nameof(NetworkSample.MinimumRankingScoreToSaveModel), 0.94},

            //related to model
            { nameof(NetworkSample.LossFunction), nameof(EvaluationMetricEnum.BinaryCrossentropy)},
            { nameof(NetworkSample.EvaluationMetrics), nameof(EvaluationMetricEnum.Accuracy)/*+","+nameof(EvaluationMetricEnum.AUC)*/},
            { nameof(NetworkSample.BatchSize), new[] {128} },
            
            { nameof(NetworkSample.NumEpochs), new[] { numEpochs } },
            
            { nameof(NetworkSample.ShuffleDatasetBeforeEachEpoch), true},
            // Optimizer 
            { nameof(NetworkSample.OptimizerType), new[] { "AdamW" /*, "SGD"*/ } },
            //{ nameof(NetworkSample.SGD_usenesterov), new[] { true, false } },
            { nameof(NetworkSample.lambdaL2Regularization), new[] { 0.0005 /*, 0.001*/} },
            { nameof(NetworkSample.AdamW_L2Regularization), new[] { /*0.005,0.001,, 0.05*/ 0.0005, 0.00025 } }, // to discard: 0.005, 0.05 0.001
            
            { nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount), -1 },
            //{ nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount), 4 },

            // Learning Rate
            //{ nameof(NetworkSample.InitialLearningRate), AbstractHyperparameterSearchSpace.Range(0.003f, 0.03f)},
            
            { nameof(NetworkSample.InitialLearningRate), new[]{0.0025, 0.005 , 0.01} }, //0.005 or 0.01

            // Learning Rate Scheduler
            //{ nameof(NetworkSample.LearningRateSchedulerType), new[] { "OneCycle" } },
            //{ nameof(NetworkSample.LearningRateSchedulerType), "CyclicCosineAnnealing" },
            {nameof(EfficientNetNetworkSample.LastActivationLayer), nameof(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID)},
            //{ nameof(NetworkSample.LearningRateSchedulerType), "CyclicCosineAnnealing" },
            {nameof(NetworkSample.LearningRateSchedulerType), new[]{"OneCycle"} },
            {nameof(NetworkSample.DisableReduceLROnPlateau), true},
            {nameof(NetworkSample.OneCycle_DividerForMinLearningRate), 20},
            {nameof(NetworkSample.OneCycle_PercentInAnnealing), new[]{ 0.1} }, //discard: 0.4
            {nameof(NetworkSample.CyclicCosineAnnealing_nbEpochsInFirstRun), 10},
            {nameof(NetworkSample.CyclicCosineAnnealing_nbEpochInNextRunMultiplier), 2},
            {nameof(NetworkSample.CyclicCosineAnnealing_MinLearningRate), 1e-5},


            // DataAugmentation
            { nameof(NetworkSample.DataAugmentationType), nameof(ImageDataGenerator.DataAugmentationEnum.DEFAULT) },
            { nameof(NetworkSample.AlphaCutMix), new[] { 0.0,  1.0} }, //0
            { nameof(NetworkSample.AlphaMixup), new[] { 0.0,  1.0} }, // 1 or 0 , 1 seems better
            { nameof(NetworkSample.CutoutPatchPercentage), new[] {/*0,*/ 0.05, 0.1} }, //0.1 or 0.2
            { nameof(NetworkSample.RowsCutoutPatchPercentage), new[] {/*0 ,*/ 0.1} }, //0 or 0.1
            { nameof(NetworkSample.ColumnsCutoutPatchPercentage),  0 }, // must be 0            
            { nameof(NetworkSample.HorizontalFlip),new[]{true,false } },
            
            //{ nameof(NetworkSample.VerticalFlip),new[]{true,false } },
            //{ nameof(NetworkSample.Rotate180Degrees),new[]{true,false } },
            { nameof(NetworkSample.FillMode),new[]{ nameof(ImageDataGenerator.FillModeEnum.Reflect) /*, nameof(ImageDataGenerator.FillModeEnum.Modulo)*/ } }, //Reflect
            //{ nameof(NetworkSample.HeightShiftRangeInPercentage), AbstractHyperparameterSearchSpace.Range(0.05f, 0.30f) }, // must be > 0 , 0.1 seems good default
            { nameof(NetworkSample.HeightShiftRangeInPercentage), new[]{0.05, 0.1   } }, //to discard: 0.2
            { nameof(NetworkSample.WidthShiftRangeInPercentage), new[]{0}}, // must be 0

        };

        //model: FB0927A468_17
        searchSpace[nameof(NetworkSample.HeightShiftRangeInPercentage)] = 0.05;
        searchSpace[nameof(NetworkSample.AdamW_L2Regularization)] = 0.00025;
        searchSpace[nameof(NetworkSample.AlphaCutMix)] = 0;
        searchSpace[nameof(NetworkSample.AlphaMixup)] = 1;
        searchSpace[nameof(NetworkSample.CutoutPatchPercentage)] = 0.1;
        searchSpace[nameof(NetworkSample.RowsCutoutPatchPercentage)] = 0.1;
        searchSpace[nameof(NetworkSample.InitialLearningRate)] = 0.0025;
        searchSpace[nameof(NetworkSample.HorizontalFlip)] = true;
        searchSpace[nameof(AbstractDatasetSample.PercentageInTraining)] = 0.5;
        searchSpace[nameof(NetworkSample.NumEpochs)] = 20;
        searchSpace[nameof(NetworkSample.MinimumRankingScoreToSaveModel)] = 0.93;


        //!D
        searchSpace[nameof(NetworkSample.HeightShiftRangeInPercentage)] = HyperparameterSearchSpace.Range(0.025f, 0.075f);
        searchSpace[nameof(NetworkSample.AdamW_L2Regularization)] = HyperparameterSearchSpace.Range(0.0002f, 0.0003f);
        searchSpace[nameof(NetworkSample.AlphaMixup)] = HyperparameterSearchSpace.Range(0.75f, 1.25f);
        searchSpace[nameof(NetworkSample.InitialLearningRate)] = HyperparameterSearchSpace.Range(0.002f, 0.03f);
        searchSpace[nameof(NetworkSample.CutoutPatchPercentage)] = HyperparameterSearchSpace.Range(0.05f, 0.15f);
        searchSpace[nameof(NetworkSample.RowsCutoutPatchPercentage)] = HyperparameterSearchSpace.Range(0.05f, 0.15f);
        searchSpace[nameof(NetworkSample.MinimumRankingScoreToSaveModel)] = 0.94;
        searchSpace[nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount)] = new[] { -1,5,6 };


        //searchSpace["MandatorySitesForTraining"] = new string[] { "GUA,StEUS", "BON,JAM,ARUBA,BAHAMAS,BERMUDE", "GUA,ARUBA", "StMARTIN,JAM,ARUBA,StEUS,BERMUDE", "StMARTIN,JAM,ARUBA,StEUS,BAHAMAS", "BON,JAM,ARUBA,StEUS", "BON,StMARTIN", "GUA,StEUS,BERMUDE", "GUA,StEUS,BAHAMAS", "GUA,ARUBA,BERMUDE", "GUA,ARUBA,BAHAMAS", "StMARTIN,JAM,ARUBA,StEUS,BAHAMAS,BERMUDE", "BON,JAM,ARUBA,StEUS,BERMUDE", "BON,JAM,ARUBA,StEUS,BAHAMAS", "BON,StMARTIN,BERMUDE", "BON,StMARTIN,BAHAMAS", "GUA,StEUS,BAHAMAS,BERMUDE", "GUA,ARUBA,BAHAMAS,BERMUDE", "BON,JAM,ARUBA,StEUS,BAHAMAS,BERMUDE", "GUA,ARUBA,StEUS", "BON,StMARTIN,BAHAMAS,BERMUDE", "BON,StMARTIN,StEUS", "BON,StMARTIN,ARUBA", "GUA,ARUBA,StEUS,BERMUDE", "GUA,ARUBA,StEUS,BAHAMAS", "BON,StMARTIN,StEUS,BERMUDE", "BON,StMARTIN,StEUS,BAHAMAS", "BON,StMARTIN,ARUBA,BERMUDE", "BON,StMARTIN,ARUBA,BAHAMAS", "GUA,ARUBA,StEUS,BAHAMAS,BERMUDE", "BON,StMARTIN,StEUS,BAHAMAS,BERMUDE", "BON,StMARTIN,ARUBA,BAHAMAS,BERMUDE", "BON,StMARTIN,ARUBA,StEUS", "GUA,JAM", "BON,StMARTIN,ARUBA,StEUS,BERMUDE", "BON,StMARTIN,ARUBA,StEUS,BAHAMAS", "GUA,JAM,BERMUDE", "GUA,JAM,BAHAMAS", "BON,StMARTIN,ARUBA,StEUS,BAHAMAS,BERMUDE", "GUA,JAM,BAHAMAS,BERMUDE", "GUA,JAM,StEUS", "GUA,JAM,ARUBA", "BON,StMARTIN,JAM", "GUA,JAM,StEUS,BERMUDE", "GUA,JAM,StEUS,BAHAMAS", "GUA,JAM,ARUBA,BERMUDE", "GUA,JAM,ARUBA,BAHAMAS", "BON,StMARTIN,JAM,BERMUDE", "BON,StMARTIN,JAM,BAHAMAS", "GUA,JAM,StEUS,BAHAMAS,BERMUDE", "GUA,JAM,ARUBA,BAHAMAS,BERMUDE", "GUA,JAM,ARUBA,StEUS", "BON,StMARTIN,JAM,BAHAMAS,BERMUDE", "GUA,StMARTIN", "BON,StMARTIN,JAM,StEUS", "BON,StMARTIN,JAM,ARUBA", "GUA,JAM,ARUBA,StEUS,BERMUDE", "GUA,JAM,ARUBA,StEUS,BAHAMAS", "GUA,StMARTIN,BERMUDE", "GUA,StMARTIN,BAHAMAS", "GUA,BON", "BON,StMARTIN,JAM,StEUS,BERMUDE", "BON,StMARTIN,JAM,StEUS,BAHAMAS", "BON,StMARTIN,JAM,ARUBA,BERMUDE", "BON,StMARTIN,JAM,ARUBA,BAHAMAS", "GUA,JAM,ARUBA,StEUS,BAHAMAS,BERMUDE", "GUA,StMARTIN,BAHAMAS,BERMUDE", "GUA,BON,BERMUDE", "GUA,BON,BAHAMAS", "BON,StMARTIN,JAM,StEUS,BAHAMAS,BERMUDE", "GUA,StMARTIN,StEUS", "BON,StMARTIN,JAM,ARUBA,BAHAMAS,BERMUDE", "GUA,StMARTIN,ARUBA", "GUA,BON,BAHAMAS,BERMUDE", "BON,StMARTIN,JAM,ARUBA,StEUS", "GUA,StMARTIN,StEUS,BERMUDE", "GUA,StMARTIN,StEUS,BAHAMAS", "GUA,StMARTIN,ARUBA,BERMUDE", "GUA,StMARTIN,ARUBA,BAHAMAS", "GUA,BON,StEUS", "GUA,BON,ARUBA", "BON,StMARTIN,JAM,ARUBA,StEUS,BERMUDE", "BON,StMARTIN,JAM,ARUBA,StEUS,BAHAMAS", "GUA,StMARTIN,StEUS,BAHAMAS,BERMUDE", "GUA,StMARTIN,ARUBA,BAHAMAS,BERMUDE", "GUA,BON,StEUS,BERMUDE", "GUA,BON,StEUS,BAHAMAS", "GUA,BON,ARUBA,BERMUDE", "GUA,BON,ARUBA,BAHAMAS", "BON,StMARTIN,JAM,ARUBA,StEUS,BAHAMAS,BERMUDE", "GUA,StMARTIN,ARUBA,StEUS", "GUA,BON,StEUS,BAHAMAS,BERMUDE", "GUA,BON,ARUBA,BAHAMAS,BERMUDE", "GUA,StMARTIN,ARUBA,StEUS,BERMUDE", "GUA,StMARTIN,ARUBA,StEUS,BAHAMAS", "GUA,BON,ARUBA,StEUS", "GUA,StMARTIN,ARUBA,StEUS,BAHAMAS,BERMUDE", "GUA,BON,ARUBA,StEUS,BERMUDE", "GUA,BON,ARUBA,StEUS,BAHAMAS", "GUA,BON,ARUBA,StEUS,BAHAMAS,BERMUDE", "GUA,StMARTIN,JAM" };
        // the 39 missings
        //searchSpace["MandatorySitesForTraining"] = new string[] { "GUA,StEUS", "GUA,ARUBA","StMARTIN,JAM,ARUBA,StEUS,BAHAMAS","BON,JAM,ARUBA,StEUS","GUA,ARUBA,BERMUDE","BON,JAM,ARUBA,StEUS,BERMUDE","BON,StMARTIN,BERMUDE","BON,StMARTIN,StEUS","GUA,ARUBA,StEUS,BERMUDE","BON,StMARTIN,StEUS,BERMUDE","BON,StMARTIN,ARUBA,BERMUDE","BON,StMARTIN,ARUBA,BAHAMAS,BERMUDE","BON,StMARTIN,ARUBA,StEUS","BON,StMARTIN,ARUBA,StEUS,BERMUDE","BON,StMARTIN,ARUBA,StEUS,BAHAMAS","GUA,JAM,BAHAMAS","GUA,JAM,BAHAMAS,BERMUDE","GUA,JAM,StEUS","GUA,JAM,ARUBA,BERMUDE","GUA,JAM,ARUBA,BAHAMAS","BON,StMARTIN,JAM,BERMUDE","GUA,JAM,ARUBA,BAHAMAS,BERMUDE","BON,StMARTIN,JAM,BAHAMAS,BERMUDE","BON,StMARTIN,JAM,StEUS","GUA,JAM,ARUBA,StEUS,BAHAMAS","GUA,BON","GUA,JAM,ARUBA,StEUS,BAHAMAS,BERMUDE","GUA,StMARTIN,StEUS,BERMUDE","GUA,StMARTIN,StEUS,BAHAMAS","GUA,StMARTIN,ARUBA,BAHAMAS","GUA,BON,StEUS","GUA,BON,ARUBA","BON,StMARTIN,JAM,ARUBA,StEUS,BERMUDE","GUA,StMARTIN,ARUBA,BAHAMAS,BERMUDE","GUA,BON,ARUBA,BAHAMAS,BERMUDE","GUA,BON,ARUBA,StEUS","GUA,BON,ARUBA,StEUS,BAHAMAS", "GUA,StMARTIN,JAM" };
        //searchSpace["MandatorySitesForTraining"] = new string[] { "GUA,ARUBA","StMARTIN,JAM,ARUBA,StEUS,BAHAMAS","BON,JAM,ARUBA,StEUS","BON,StMARTIN,BERMUDE","BON,StMARTIN,StEUS","BON,StMARTIN,StEUS,BERMUDE","BON,StMARTIN,ARUBA,StEUS","BON,StMARTIN,ARUBA,StEUS,BERMUDE","BON,StMARTIN,ARUBA,StEUS,BAHAMAS","GUA,JAM,BAHAMAS","GUA,JAM,BAHAMAS,BERMUDE","BON,StMARTIN,JAM","BON,StMARTIN,JAM,BERMUDE","BON,StMARTIN,JAM,BAHAMAS,BERMUDE","BON,StMARTIN,JAM,StEUS","GUA,JAM,ARUBA,StEUS,BAHAMAS","GUA,BON","GUA,StMARTIN,StEUS,BERMUDE","GUA,StMARTIN,StEUS,BAHAMAS","GUA,StMARTIN,ARUBA,BAHAMAS","GUA,BON,StEUS","GUA,BON,ARUBA,BAHAMAS,BERMUDE","GUA,BON,ARUBA,StEUS", "GUA,StMARTIN,JAM,BERMUDE", "GUA,StMARTIN,JAM,BAHAMAS", "GUA,BON,JAM", "GUA,StMARTIN,JAM,BAHAMAS,BERMUDE", "GUA,BON,JAM,BERMUDE", "GUA,BON,JAM,BAHAMAS", "GUA,StMARTIN,JAM,StEUS", "GUA,StMARTIN,JAM,ARUBA", "GUA,BON,JAM,BAHAMAS,BERMUDE", "GUA,StMARTIN,JAM,StEUS,BERMUDE", "GUA,StMARTIN,JAM,StEUS,BAHAMAS", "GUA,StMARTIN,JAM,ARUBA,BERMUDE", "GUA,StMARTIN,JAM,ARUBA,BAHAMAS", "GUA,BON,JAM,StEUS", "GUA,BON,JAM,ARUBA", "GUA,StMARTIN,JAM,StEUS,BAHAMAS,BERMUDE", "GUA,StMARTIN,JAM,ARUBA,BAHAMAS,BERMUDE", "GUA,BON,JAM,StEUS,BERMUDE", "GUA,BON,JAM,StEUS,BAHAMAS", "GUA,BON,JAM,ARUBA,BERMUDE", "GUA,BON,JAM,ARUBA,BAHAMAS", "GUA,StMARTIN,JAM,ARUBA,StEUS", "GUA,BON,JAM,StEUS,BAHAMAS,BERMUDE", "GUA,BON,JAM,ARUBA,BAHAMAS,BERMUDE", "GUA,StMARTIN,JAM,ARUBA,StEUS,BERMUDE", "GUA,StMARTIN,JAM,ARUBA,StEUS,BAHAMAS", "GUA,BON,JAM,ARUBA,StEUS", "GUA,BON,StMARTIN", "GUA,StMARTIN,JAM,ARUBA,StEUS,BAHAMAS,BERMUDE", "GUA,BON,JAM,ARUBA,StEUS,BERMUDE", "GUA,BON,JAM,ARUBA,StEUS,BAHAMAS", "GUA,BON,StMARTIN,BERMUDE", "GUA,BON,StMARTIN,BAHAMAS", "GUA,BON,JAM,ARUBA,StEUS,BAHAMAS,BERMUDE", "GUA,BON,StMARTIN,BAHAMAS,BERMUDE", "GUA,BON,StMARTIN,StEUS", "GUA,BON,StMARTIN,ARUBA", "GUA,BON,StMARTIN,StEUS,BERMUDE", "GUA,BON,StMARTIN,StEUS,BAHAMAS", "GUA,BON,StMARTIN,ARUBA,BERMUDE", "GUA,BON,StMARTIN,ARUBA,BAHAMAS", "GUA,BON,StMARTIN,StEUS,BAHAMAS,BERMUDE", "GUA,BON,StMARTIN,ARUBA,BAHAMAS,BERMUDE", "GUA,BON,StMARTIN,ARUBA,StEUS" };
        //searchSpace["MandatorySitesForTraining"] = "GUA,BON,ARUBA";
        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(DefaultEfficientNetNetworkSample(), new Biosonar85DatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        const bool retrainOnFullDatasetIfBetterModelFound = false;
        hpo.Process(t => SampleUtils.TrainWithHyperparameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, retrainOnFullDatasetIfBetterModelFound, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
    }


    // ReSharper disable once UnusedMember.Local
    private static void Launch_HPO_MEL_SPECTROGRAM_SMALL_128_401(int numEpochs = 10, int maxAllowedSecondsForAllComputation = 0)
    {
     

        var searchSpace = new Dictionary<string, object>
        {
            //related to Dataset 
            //{"KFold", 2},
            
            { nameof(AbstractDatasetSample.PercentageInTraining), 1.0}, //will be automatically set to 1 if KFold is enabled

            { nameof(AbstractDatasetSample.ShuffleDatasetBeforeSplit), false},
            { nameof(Biosonar85DatasetSample.InputDataType), nameof(Biosonar85DatasetSample.InputDataTypeEnum.MEL_SPECTROGRAM_SMALL_128_401)},
            { nameof(NetworkSample.MinimumRankingScoreToSaveModel), 0.93},

            //related to model
            { nameof(NetworkSample.LossFunction), nameof(EvaluationMetricEnum.BinaryCrossentropy)},
            { nameof(NetworkSample.EvaluationMetrics), nameof(EvaluationMetricEnum.Accuracy)/*+","+nameof(EvaluationMetricEnum.AUC)*/},
            { nameof(NetworkSample.BatchSize), 32}, //because of memory issues, we have to use small batches

            { nameof(NetworkSample.NumEpochs), 1},

            { nameof(NetworkSample.ShuffleDatasetBeforeEachEpoch), false},
            // Optimizer 
            { nameof(NetworkSample.OptimizerType), new[] { "AdamW" } },
            { nameof(NetworkSample.lambdaL2Regularization), new[] { 0.0005} },


            { nameof(NetworkSample.AdamW_L2Regularization), 0.0005 },

            { nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount), 2 },
            //{ nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount), 5 },

            // Learning Rate
            { nameof(NetworkSample.InitialLearningRate), 0.01},

            // Learning Rate Scheduler
            //{ nameof(NetworkSample.LearningRateSchedulerType), "CyclicCosineAnnealing" },
            {nameof(NetworkSample.LearningRateSchedulerType), new[]{"OneCycle"} },
            {nameof(EfficientNetNetworkSample.LastActivationLayer), nameof(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID)},
            {nameof(NetworkSample.DisableReduceLROnPlateau), true},
            {nameof(NetworkSample.OneCycle_DividerForMinLearningRate), 20},
            {nameof(NetworkSample.OneCycle_PercentInAnnealing), new[]{ 0.1} },
            {nameof(NetworkSample.CyclicCosineAnnealing_nbEpochsInFirstRun), 10},
            {nameof(NetworkSample.CyclicCosineAnnealing_nbEpochInNextRunMultiplier), 2},
            {nameof(NetworkSample.CyclicCosineAnnealing_MinLearningRate), 1e-5},


            // DataAugmentation
            { nameof(NetworkSample.DataAugmentationType), nameof(ImageDataGenerator.DataAugmentationEnum.DEFAULT) },
            //{ nameof(NetworkSample.AlphaCutMix), 0 },
            { nameof(NetworkSample.AlphaMixup), 0},
            //{ nameof(NetworkSample.CutoutPatchPercentage), 0.0 },
            //{ nameof(NetworkSample.CutoutCount), 0 },
            //{ nameof(NetworkSample.ColumnsCutoutPatchPercentage), 0.2},
            { nameof(NetworkSample.RowsCutoutCount), 0},
            { nameof(NetworkSample.RowsCutoutPatchPercentage),  0.0 },
            { nameof(NetworkSample.HorizontalFlip),true},
            //{ nameof(NetworkSample.VerticalFlip),true},

            { nameof(NetworkSample.FillMode),new[]{ nameof(ImageDataGenerator.FillModeEnum.Reflect) /*, nameof(ImageDataGenerator.FillModeEnum.Modulo)*/ } }, //Reflect
            { nameof(NetworkSample.HeightShiftRangeInPercentage), 0.0},
            { nameof(NetworkSample.WidthShiftRangeInPercentage), 0.0},

        };

     
        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(DefaultEfficientNetNetworkSample(), new Biosonar85DatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        const bool retrainOnFullDatasetIfBetterModelFound = false;
        hpo.Process(t => SampleUtils.TrainWithHyperparameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, retrainOnFullDatasetIfBetterModelFound, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
    }



    // ReSharper disable once UnusedMember.Local
    private static void Launch_HPO_MEL_SPECTROGRAM_128_401_BinaryCrossentropy(int numEpochs = 10, int maxAllowedSecondsForAllComputation = 0)
    {
        //when LossFunction = BinaryCrossentropy
        //  InitialLearningRate ==
        //  AdamW_L2Regularization == 0.00025
        //  OneCycle_PercentInAnnealing == 0
        //  AlphaMixup == 1.2
        //  SkipConnectionsDropoutRate >= 0.4



        // stats from FindBestLearningRate
        //     DefaultMobileBlocksDescriptionCount     BatchSize   Best learning rate                           Free GPU Memory     
        //     B0 + -1                                 64          0.002 => 0.030 or 0.08                       12700MB             A305BB2D2F
        //     B0 + 5                                  64          0.003 => 0.025 or 0.06                       14587MB             93898B4B73  
        //     B0 + 5                                  128         ?     => 0.002                               7268MB              C05FD055A3
        //     B0 + -1                                 128         ?     => 0.01 or 0.1                         5500MB             A305BB2D2F                                       

        #region previous tests
        var searchSpace = new Dictionary<string, object>
        {
            //related to Dataset 
            //{"KFold", 2},
            
            { nameof(AbstractDatasetSample.PercentageInTraining), 0.5}, //will be automatically set to 1 if KFold is enabled

            { nameof(AbstractDatasetSample.ShuffleDatasetBeforeSplit), true},
            { nameof(Biosonar85DatasetSample.InputDataType), nameof(Biosonar85DatasetSample.InputDataTypeEnum.MEL_SPECTROGRAM_128_401)},
            { nameof(NetworkSample.MinimumRankingScoreToSaveModel), 0.93},

            //related to model
            { nameof(NetworkSample.LossFunction), nameof(EvaluationMetricEnum.BinaryCrossentropy)},
            { nameof(NetworkSample.EvaluationMetrics), nameof(EvaluationMetricEnum.Accuracy)/*+","+nameof(EvaluationMetricEnum.AUC)*/},
            { nameof(NetworkSample.BatchSize), new[] {128} }, //because of memory issues, we have to use small batches

            { nameof(NetworkSample.NumEpochs), new[] { numEpochs } },

            { nameof(NetworkSample.ShuffleDatasetBeforeEachEpoch), true},
            // Optimizer 
            { nameof(NetworkSample.OptimizerType), new[] { "AdamW" } },
            { nameof(NetworkSample.lambdaL2Regularization), new[] { 0.0005} },

            
            { nameof(NetworkSample.AdamW_L2Regularization), new[] {0.00015,  0.00025,  0.0005} },

            { nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount), -1 },
            //{ nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount), 5 },

            // Learning Rate
            { nameof(NetworkSample.InitialLearningRate), new[]{0.00125, 0.0025, 0.005 } },

            // Learning Rate Scheduler
            //{ nameof(NetworkSample.LearningRateSchedulerType), "CyclicCosineAnnealing" },
            {nameof(NetworkSample.LearningRateSchedulerType), new[]{"OneCycle"} },
            {nameof(EfficientNetNetworkSample.LastActivationLayer), nameof(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID)},
            {nameof(NetworkSample.DisableReduceLROnPlateau), true},
            {nameof(NetworkSample.OneCycle_DividerForMinLearningRate), 20},
            {nameof(NetworkSample.OneCycle_PercentInAnnealing), new[]{ 0.1} },
            {nameof(NetworkSample.CyclicCosineAnnealing_nbEpochsInFirstRun), 10},
            {nameof(NetworkSample.CyclicCosineAnnealing_nbEpochInNextRunMultiplier), 2},
            {nameof(NetworkSample.CyclicCosineAnnealing_MinLearningRate), 1e-5},


            // DataAugmentation
            { nameof(NetworkSample.DataAugmentationType), nameof(ImageDataGenerator.DataAugmentationEnum.DEFAULT) },
            { nameof(NetworkSample.AlphaCutMix), 0 },
            { nameof(NetworkSample.AlphaMixup), new[] { 1.0, 1.2} },
            { nameof(NetworkSample.CutoutPatchPercentage), new[] {0, 0.1, 0.2} },
            { nameof(NetworkSample.RowsCutoutPatchPercentage), new[] {0.1, 0.2} },
            { nameof(NetworkSample.ColumnsCutoutPatchPercentage),  0 },
            { nameof(NetworkSample.HorizontalFlip),true/*new[]{true,false } */},

            { nameof(NetworkSample.FillMode),new[]{ nameof(ImageDataGenerator.FillModeEnum.Reflect) /*, nameof(ImageDataGenerator.FillModeEnum.Modulo)*/ } }, //Reflect
            { nameof(NetworkSample.HeightShiftRangeInPercentage), new[]{0, 0.05} },
            { nameof(NetworkSample.WidthShiftRangeInPercentage), new[]{0}},

        };

        //model: FB0927A468_17
        searchSpace[nameof(NetworkSample.AdamW_L2Regularization)] = 0.00025;
        searchSpace[nameof(NetworkSample.AlphaCutMix)] = 0;
        searchSpace[nameof(NetworkSample.AlphaMixup)] = 1;
        searchSpace[nameof(NetworkSample.CutoutPatchPercentage)] = 0.1;
        searchSpace[nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount)] = -1;
        searchSpace[nameof(NetworkSample.HeightShiftRangeInPercentage)] = 0.05;
        searchSpace[nameof(NetworkSample.HorizontalFlip)] = true;
        searchSpace[nameof(NetworkSample.InitialLearningRate)] = 0.0025;
        searchSpace[nameof(NetworkSample.MinimumRankingScoreToSaveModel)] = 0.94;
        searchSpace[nameof(NetworkSample.NumEpochs)] = 20;
        searchSpace[nameof(AbstractDatasetSample.PercentageInTraining)] = 0.5;
        searchSpace[nameof(NetworkSample.RowsCutoutPatchPercentage)] = 0.1;




        searchSpace[nameof(NetworkSample.AdamW_L2Regularization)] = new[] { 0.00025, 0.0005 };
        searchSpace[nameof(NetworkSample.AlphaCutMix)] = 0;
        searchSpace[nameof(NetworkSample.AlphaMixup)] = 1.0; //new[] { 1, 1.2, 1.5, 2.0,3 };
        searchSpace[nameof(NetworkSample.ColumnsCutoutPatchPercentage)] = 0;
        searchSpace[nameof(NetworkSample.ColumnsCutoutCount)] = 0;
        searchSpace[nameof(NetworkSample.CutoutPatchPercentage)] = 0.1;
        searchSpace[nameof(NetworkSample.CutoutCount)] = 1;
        searchSpace[nameof(NetworkSample.HeightShiftRangeInPercentage)] = 0.05;
        searchSpace[nameof(NetworkSample.InitialLearningRate)] = 0.02;
        searchSpace[nameof(NetworkSample.LossFunction)] = nameof(EvaluationMetricEnum.BinaryCrossentropy);
        searchSpace[nameof(NetworkSample.NumEpochs)] = 20;
        searchSpace[nameof(NetworkSample.OneCycle_DividerForMinLearningRate)] = 20;
        searchSpace[nameof(NetworkSample.OneCycle_PercentInAnnealing)] = new[]{0,0.1};
        searchSpace[nameof(NetworkSample.RowsCutoutPatchPercentage)] = 0.1;
        searchSpace[nameof(NetworkSample.RowsCutoutCount)] = 1; //new[] { 1, 10 };
        searchSpace[nameof(NetworkSample.WidthShiftRangeInPercentage)] = 0;
        searchSpace[nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount)] = new[]{5,-1}; //new[]{2,3,4,5,6, -1};
        //best: 0.9440      393C258A23
        //AdamW_L2Regularization = 0.00025
        //LossFunction = BinaryCrossentropy
        //DefaultMobileBlocksDescriptionCount = 5
        //OneCycle_PercentInAnnealing = 0



        searchSpace[nameof(NetworkSample.AdamW_L2Regularization)] = 0.00025; //new[] { 0.00025, 0.0005 };
        searchSpace[nameof(NetworkSample.AlphaCutMix)] = 0;
        searchSpace[nameof(NetworkSample.AlphaMixup)] = 1.0; //new[] { 1, 1.2, 1.5, 2.0,3 };
        searchSpace[nameof(NetworkSample.ColumnsCutoutPatchPercentage)] = 0;
        searchSpace[nameof(NetworkSample.ColumnsCutoutCount)] = 0;
        searchSpace[nameof(NetworkSample.CutoutPatchPercentage)] = 0.1;
        searchSpace[nameof(NetworkSample.CutoutCount)] = 1;
        searchSpace[nameof(NetworkSample.HeightShiftRangeInPercentage)] = 0.05;
        searchSpace[nameof(NetworkSample.InitialLearningRate)] = 0.02;
        searchSpace[nameof(NetworkSample.LossFunction)] = nameof(EvaluationMetricEnum.BinaryCrossentropy);
        searchSpace[nameof(NetworkSample.NumEpochs)] = 20;
        searchSpace[nameof(NetworkSample.OneCycle_DividerForMinLearningRate)] = 20;
        searchSpace[nameof(NetworkSample.OneCycle_PercentInAnnealing)] = 0; //new[] { 0, 0.1 };
        searchSpace[nameof(NetworkSample.RowsCutoutPatchPercentage)] = 0.1;
        searchSpace[nameof(NetworkSample.RowsCutoutCount)] = 1; //new[] { 1, 10 };
        searchSpace[nameof(NetworkSample.WidthShiftRangeInPercentage)] = 0;
        searchSpace[nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount)] = 5; //new[] { 5, -1 }; //new[]{2,3,4,5,6, -1};
        searchSpace[nameof(EfficientNetNetworkSample.TopDropoutRate)] = new[] { 0.1f, 0.2f, 0.3f, 0.4f };
        searchSpace[nameof(EfficientNetNetworkSample.SkipConnectionsDropoutRate)] = new[] { 0.1f, 0.2f, 0.3f, 0.4f };
        //best: 0.9514 89B2D5CEDF
        //AdamW_L2Regularization = 0.00025
        //DefaultMobileBlocksDescriptionCount = 5
        //LossFunction = BinaryCrossentropy
        //OneCycle_PercentInAnnealing = 0
        //TopDropoutRate = 0.2
        //SkipConnectionsDropoutRate = 0.2
        //TopDropoutRate = 0.1


        searchSpace[nameof(NetworkSample.LossFunction)] = nameof(EvaluationMetricEnum.BinaryCrossentropy);
        searchSpace[nameof(EfficientNetNetworkSample.TopDropoutRate)] = new[] { 0.0f, 0.2f, 0.4f };
        searchSpace[nameof(EfficientNetNetworkSample.SkipConnectionsDropoutRate)] = new[] { 0.0f, 0.2f, 0.4f, 0.5f };
        searchSpace[nameof(NetworkSample.OneCycle_PercentInAnnealing)] = new[] { 0, 0.1 };
        searchSpace[nameof(NetworkSample.AlphaMixup)] = new[] { 1, 1.2, 1.5};
        searchSpace[nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount)] = 5; //new[] { 5, -1 }; //new[]{2,3,4,5,6, -1};
        //best: 0.9511 9CEE8F0073
        //AlphaMixup = 1.2
        //LossFunction = BinaryCrossentropy
        //SkipConnectionsDropoutRate = 0.5
        //TopDropoutRate = 0.4
        //OneCycle_PercentInAnnealing = 0
        #region stats (hpo_16380.csv)
        /*
         * Stats for AlphaMixup:
         1.2:-0.9425161921459696 +/- 0.005026113550600352 (23 evals at 549.6s/eval) (target Time: 44.9%)
         1.5:-0.9396843284368515 +/- 0.00708116208620747 (20 evals at 546.7s/eval) (target Time: 30%)
         1:-0.9385012310484181 +/- 0.007097887046079005 (23 evals at 545.1s/eval) (target Time: 25.1%)
        Stats for BCEWithFocalLoss_Gamma:
         0:-0.9402588985183022 +/- 0.0066713671938491695 (66 evals at 547.1s/eval) (target Time: 50%)
         0.7:empty (target Time: 50%)
        Stats for BCEWithFocalLoss_PercentageInTrueClass:
         0.5:-0.9402588985183022 +/- 0.0066713671938491695 (66 evals at 547.1s/eval) (target Time: 33.3%)
         0.4:empty (target Time: 33.3%)
         0.6:empty (target Time: 33.3%)
        Stats for OneCycle_PercentInAnnealing:
         0:-0.9420196516173226 +/- 0.0062835303533355265 (35 evals at 551.3s/eval) (target Time: 64.3%)
         0.1:-0.9382709514710211 +/- 0.006536636545131606 (31 evals at 542.4s/eval) (target Time: 35.7%)
        Stats for SkipConnectionsDropoutRate:
         0.4:-0.9408757798373699 +/- 0.0073601562871463844 (16 evals at 546.1s/eval) (target Time: 26.9%)
         0.2:-0.9406338754822227 +/- 0.004946303973481242 (17 evals at 553s/eval) (target Time: 26.1%)
         0.5:-0.9405490942299366 +/- 0.006036560713137301 (16 evals at 544s/eval) (target Time: 25.8%)
         0:-0.939030201996074 +/- 0.0078074921220547475 (17 evals at 545.3s/eval) (target Time: 21.2%)
        Stats for TopDropoutRate:
         0:-0.941954721575198 +/- 0.005152757288117706 (23 evals at 551s/eval) (target Time: 42%)
         0.4:-0.9401502013206482 +/- 0.0074750716188162385 (22 evals at 545.6s/eval) (target Time: 33%)
         0.2:-0.938515441758292 +/- 0.006786811112259223 (21 evals at 544.5s/eval) (target Time: 25%)
         */
        #endregion




        searchSpace[nameof(NetworkSample.AdamW_L2Regularization)] = 0.00025; //new[] { 0.00025, 0.0005 };
        searchSpace[nameof(NetworkSample.AlphaCutMix)] = 0;
        searchSpace[nameof(NetworkSample.AlphaMixup)] = 1.2; //1.0; //new[] { 1, 1.2, 1.5, 2.0,3 };
        searchSpace[nameof(NetworkSample.ColumnsCutoutPatchPercentage)] = 0;
        searchSpace[nameof(NetworkSample.ColumnsCutoutCount)] = 0;
        searchSpace[nameof(NetworkSample.CutoutPatchPercentage)] = 0.1;
        searchSpace[nameof(NetworkSample.CutoutCount)] = 1;
        searchSpace[nameof(NetworkSample.HeightShiftRangeInPercentage)] = 0.05;
        searchSpace[nameof(NetworkSample.InitialLearningRate)] = new[]{0.01,0.02,0.04};
        searchSpace[nameof(NetworkSample.LossFunction)] = nameof(EvaluationMetricEnum.BinaryCrossentropy);
        searchSpace[nameof(NetworkSample.NumEpochs)] = 20;
        searchSpace[nameof(NetworkSample.OneCycle_DividerForMinLearningRate)] = 20;
        searchSpace[nameof(NetworkSample.OneCycle_PercentInAnnealing)] = 0; //new[] { 0, 0.1 };
        searchSpace[nameof(NetworkSample.RowsCutoutPatchPercentage)] = 0.1;
        searchSpace[nameof(NetworkSample.RowsCutoutCount)] = 1; //new[] { 1, 10 };
        searchSpace[nameof(NetworkSample.WidthShiftRangeInPercentage)] = 0;
        searchSpace[nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount)] = 5; //new[] { 5, -1 }; //new[]{2,3,4,5,6, -1};
        searchSpace[nameof(EfficientNetNetworkSample.TopDropoutRate)] = 0.2f; //new[] { 0.2f, 0.4f };
        searchSpace[nameof(EfficientNetNetworkSample.SkipConnectionsDropoutRate)] = new[] { 0f, 0.2f, 0.4f};
        //best 0.9502 7046B1D862
        //SkipConnectionsDropoutRate = 0
        //InitialLearningRate = 0.02
        //TopDropoutRate = 0.2
        #region stats (hpo_25492.csv)
        /*
        Stats for InitialLearningRate:
         0.02:-0.9446551203727722 +/- 0.003921505446990086 (3 evals at 559.1s/eval) (target Time: 55.7%)
         0.01:-0.9432544112205505 +/- 0.0018052004396709764 (3 evals at 563.8s/eval) (target Time: 38.8%)
         0.04:-0.9311263759930929 +/- 0.00738073029813322 (3 evals at 565.6s/eval) (target Time: 5.6%)
        Stats for SkipConnectionsDropoutRate:
         0:-0.9420586824417114 +/- 0.006738606448944757 (3 evals at 555s/eval) (target Time: 41.8%)
         0.4:-0.9406238198280334 +/- 0.0014258936491357123 (3 evals at 551.9s/eval) (target Time: 33.8%)
         0.2:-0.9363534053166708 +/- 0.010897142170991284 (3 evals at 581.6s/eval) (target Time: 24.5%)
         */
        #endregion
        #endregion

        //with Data Augmentation on waveform
        searchSpace