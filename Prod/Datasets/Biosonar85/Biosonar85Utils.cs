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
        double