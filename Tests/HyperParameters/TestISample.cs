using System;
using NUnit.Framework;
using SharpNet;
using SharpNet.Datasets;
using SharpNet.Hyperparameters;
using SharpNet.LightGBM;

// ReSharper disable FieldCanBeMadeReadOnly.Local
// ReSharper disable MemberCanBePrivate.Local
// ReSharper disable ConvertToConstant.Local

namespace SharpNetTests.Hyperparameters;

[TestFixture]

public class TestISample
{
    private class TestClass : AbstractDatasetSample
    {
        // ReSharper disable once EmptyConstructor
        public TestClass(){ }
        public override Objective_enum GetObjective() => Objective_enum.Regression;
        public override string IdColumn => throw new NotImplementedException();
        public override string[] TargetLabels => throw new NotImplementedException();
        public override bool IsCategoricalColumn(string columnName) => throw new NotImplementedException();
        public override DataSet TestDataset() => throw new NotImplementedException();
        public override DataSet FullTrainingAndValidation() => throw new NotImplementedException();
        public override int[] X_Shape(int batchSize) => throw new NotImplementedException();
        public override int[] Y_Shape(int batchSize) => throw new NotImplementedException();
        public override int NumClass => 1;
    }

    [Test]
    public void TestCopyWithNewPercentageInTrainingAndKFold()
    {
        var sample =  new TestClass();
        sample.PercentageInTraining = 0.5;
        sample.KFold = 3;
        sample.Train_XDatasetPath_InTargetFormat = "Train_XDatasetPath_InTargetFormat";
        sample.Train_YDataset