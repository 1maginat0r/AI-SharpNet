using System;
using System.Collections.Generic;
using NUnit.Framework;
using SharpNet;
using SharpNet.Data;
using SharpNet.Datasets;
using SharpNet.GPU;
using SharpNet.Layers;
using SharpNet.Models;
using SharpNet.Networks;
using log4net;
using System.IO;
using System.Linq;
using System.Reflection;
using SharpNet.CPU;

namespace SharpNetTests.NonReg
{


    /// <summary>
    /// sandbox to make // run with PyTorch on several kind of networks
    /// </summary>
    //[TestFixture]
    public class ParallelRunWithPyTorch
    {
        private static readonly ILog Log = LogManager.GetLogger(typeof(ParallelRunWithPyTorch));


        private static string GetDefaultWorkingDirectory()
        {
            // ReSharper disable once AssignNullToNotNullAttribute
            return Path.Combine(Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location), "../../../NonReg");
        }
        static ParallelRunWithPyTorch()
        {
            var log_directory = GetDefaultWorkingDirectory();
            Utils.TryDelete(Path.Join(log_directory, "ParallelRunWithPyTorch.log"));
            // ReSharper disable once PossibleNullReferenceException
            Console.WriteLine(Path.GetDirectoryName(Assembly.GetEntryAssembly().Location));
            Utils.ConfigureGlobalLog4netProperties(log_directory, "ParallelRunWithPyTorch");
            Utils.ConfigureThreadLog4netProperties(log_directory, "ParallelRunWithPyTorch");
        }
        private static (float loss_before, float loss_after) Train(Network model,
            CpuTensor<float> X,
            CpuTensor<float> Y,
            double lr,
            int numEpochs,
            int batch_size,
            double momentum  = 0.0,
            double lambdaL2Regularization = 0.0
        )
        {
            Log.Info(model.Summary() + Environment.NewLine);

            Log.Info(model.ToPytorchModule() + Environment.NewLine);


            var predict_before = model.Predict(X, false).ToNumpy();

            using var trainingDataSet = new InMemoryDataSet(X, Y);

            var lossAccuracyBefore = model.ComputeMetricsForValidationDataSet(batch_size, trainingDataSet);
            var loss_before = (float)lossAccuracyBefore.First(t => t.Key == model.NetworkSample.LossFunction).Value;

            Log.Info("-");
            Log.Info("----------------------------------------