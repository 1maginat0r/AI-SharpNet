using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using SharpNet.Datasets;
using SharpNet.Models;

namespace SharpNet.LightGBM
{
    public class LightGBMModel : Model
    {
        #region public fields & properties
        public LightGBMSample LightGbmSample => (LightGBMSample)ModelSample;
        #endregion

        #region constructor
        /// <summary>
        /// 
        /// </summary>
        /// <param name="lightGbmModelSample"></param>
        /// <param name="workingDirectory"></param>
        /// <param name="modelName">the name of the model to use</param>
        /// <exception cref="Exception"></exception>
        [SuppressMessage("ReSharper", "VirtualMemberCallInConstructor")]
        public LightGBMModel(LightGBMSample lightGbmModelSample, string workingDirectory, [JetBrains.Annotations.NotNull] string modelName): base(lightGbmModelSample, workingDirectory, modelName)
        {
            if (!File.Exists(ExePath))
            {
                throw new Exception($"Missing exe {ExePath}");
            }
            if (!Directory.Exists(TempPath))
            {
                Directory.CreateDirectory(TempPath);
            }
        }
        #endregion

        public override (string train_XDatasetPath_InModelFormat, string train_YDatasetPath_InModelFormat, string train_XYDatasetPath_InModelFormat, string validation_XDatasetPath_InModelFormat, string validation_YDatasetPath_InModelFormat, string validation_XYDatasetPath_InModelFormat, IScore trainLossIfAvailable, IScore validationLossIfAvailable, IScore trainRankingMetricIfAvailable, IScore validationRankingMetricIfAvailable)
            Fit(DataSet trainDataset, DataSet validationDatasetIfAny, Func<bool, bool, DataSet, DataSet, string, List<string>> save = null)
        {
            if (ModelSample.GetLoss()  == EvaluationMetricEnum.DEFAULT_VALUE)
            {
                throw new ArgumentException("Loss Function not set");
            }

            var sw = Stopwatch.StartNew();
            const bool addTargetColumnAsFirstColumn = true;
            const bool includeIdColumns = false;
            const bool overwriteIfExists = false;
            var train_XYDatasetPath_InModelFormat = trainDataset.to_csv_in_directory(DatasetPath, addTargetColumnAsFirstColumn, includeIdColumns, overwriteIfExists);
            var validation_XYDatasetPath_InModelFormat = validationDatasetIfAny?.to_csv_in_directory(DatasetPath, addTargetColumnAsFirstColumn, includeIdColumns, overwriteIfExists);
            LightGbmSample.UpdateForDataset(trainDataset);
            //we save in 'tmpLightGBMSamplePath' the model sample used for training
            var tmpLightGBMSamplePath = LightGbmSample.ToPath(TempPath, ModelName);
            var tmpLightGBMSample = (LightGBMSample)LightGbmSample.Clone();
            tmpLightGBMSample.Set(new Dictionary<string, object> {
                {"task", LightGBMSample.task_enum.train},
                {"data", train_XYDatasetPath_InModelFormat},
                {"valid", validation_XYDatasetPath_InModelFormat??""},
                {"output_model", ModelPath},
                {"input_model", ""},
                {"prediction_result", ""},
                {"header", true},
                {"save_binary", false},
                //this is needed to retrieve the train and validation metrics
                {nameof(LightGbmSample.verbosity), "1"}, 
                //this is need to retrieve the train metric
                {nameof(LightGbmSample.is_provide_training_metric), true},
            });

            tmpLightGBMSample.Save(tmpLightGBMSamplePath);
            LogForModel($"Training model '{ModelName}' with training dataset '{Path.GetFileNameWithoutExtension(train_XYDatasetPath_InModelFormat)}" + (string.IsNullOrEmpty(validation_XYDatasetPath_InModelFormat) ? "" : $" and validation dataset {Path.GetFileNameWithoutExtension(validation_XYDatasetPath_InModelFormat)}'"));
            var linesFromLog = Utils.Launch(WorkingDirectory, ExePath, "config=" + tmpLightGBMSamplePath, Log, true);
            var (trainLossIfAvailable, validationLossIfAvailable, trainRankingMetricIfAvailable, validationRankingMetricIfAvailable) = tmpLightGBMSample.ExtractScores(linesFromLog);
            //(IScore trainLossIfAvailable, IScore validationLossIfAvailable, IScore trainRankingMetricIfAvailable, IScore validationRankingMetricIfAvailable) = (null, null, null, null);

            Utils.TryDelete(tmpLightGBMSamplePath);

            LogForModel($"Model '{ModelName}' trained with dataset '{Path.GetFileNameWithoutExtension(train_XYDatasetPath_InModelFormat)}' in {sw.Elapsed.TotalSeconds}s (trainScore = {trainLossIfAvailable} / validationScore = {validationLossIfAvailable} / trainMetric = {trainRankingMetricIfAvailable} / validationMetric = {validationRankingMetricIfAvailable})");
            return (null, null, train_XYDatasetPath_InModelFormat, null, null, validation_XYDatasetPath_InModelFormat, trainLossIfAvailable, validationLossIfAvailable, tra