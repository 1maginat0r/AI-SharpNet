
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using NUnit.Framework;
using SharpNet;

namespace SharpNetTests;

[TestFixture]
public class TestUtils
{
    [TestCase(100, 99, 20)]
    [TestCase(20, 19, 20)]
    [TestCase(20, 20, 20)]
    [TestCase(40, 21, 20)]
    [TestCase(3, 3, 1)]
    public void TestFirstMultipleOfAtomicValueAboveOrEqualToMinimum(int expected, int minimum, int atomicValue)
    {
        Assert.AreEqual(expected, Utils.FirstMultipleOfAtomicValueAboveOrEqualToMinimum(minimum, atomicValue));
    }

    [Test]
    public void TestRepeat()
    {
        Assert.IsTrue(new[]{1,1,1,3,3,3,1,1,1}.SequenceEqual(Utils.Repeat(new []{1,3,1},3)));
    }
        

    [TestCase(0, null)]
    [TestCase(0, new int[0])]
    [TestCase(0, new int[0])]
    [TestCase(3, new[] {3})]
    [TestCase(3*5, new[] {3, 5})]
    [TestCase(3*5*7, new[] {3, 5, 7})]
    [TestCase(0, new[] {3, 5, 7, 0})]
    public void TestProduct(int expectedResult, int[] data)
    {
        Assert.AreEqual(expectedResult, Utils.Product(data));
    }

    [TestCase(true, 0f, float.NaN, EvaluationMetricEnum.Mae)]
    [TestCase(false, float.NaN, 0, EvaluationMetricEnum.Mae)]
    [TestCase(false, float.NaN, float.NaN, EvaluationMetricEnum.Mae)]
    [TestCase(true, 0f, 1f, EvaluationMetricEnum.Mae)]
    [TestCase(false, 1f, 0f, EvaluationMetricEnum.Mae)]
    [TestCase(false, 0f, 0f, EvaluationMetricEnum.Mae)]
    [TestCase(true, 1f, 0f, EvaluationMetricEnum.Accuracy)]
    [TestCase(false, 0f, 1f, EvaluationMetricEnum.Accuracy)]
    [TestCase(false, 0f, 0f, EvaluationMetricEnum.Accuracy)]
    [TestCase(true, 1f, 0f, EvaluationMetricEnum.SparseAccuracy)]
    [TestCase(false, 0f, 1f, EvaluationMetricEnum.SparseAccuracy)]
    [TestCase(false, 0f, 0f, EvaluationMetricEnum.SparseAccuracy)]
    [TestCase(true, 1f, 0f, EvaluationMetricEnum.AUC)]
    [TestCase(false, 0f, 1f, EvaluationMetricEnum.AUC)]
    [TestCase(false, 0f, 0f, EvaluationMetricEnum.AUC)]
    [TestCase(true, 1f, 0f, EvaluationMetricEnum.AveragePrecisionScore)]
    [TestCase(false, 0f, 1f, EvaluationMetricEnum.AveragePrecisionScore)]
    [TestCase(false, 0f, 0f, EvaluationMetricEnum.AveragePrecisionScore)]
    [TestCase(true, 1f, -0.5f, EvaluationMetricEnum.PearsonCorrelation)]
    [TestCase(false, 0f, 0.5f, EvaluationMetricEnum.PearsonCorrelation)]
    [TestCase(false, 0f, 0f, EvaluationMetricEnum.PearsonCorrelation)]
    [TestCase(true, 1f, -0.5f, EvaluationMetricEnum.SpearmanCorrelation)]
    [TestCase(false, 0f, 0.5f, EvaluationMetricEnum.SpearmanCorrelation)]
    [TestCase(false, 0f, 0f, EvaluationMetricEnum.SpearmanCorrelation)]
    public void TestIsBetterScore(bool expectedResult, float a, float b, EvaluationMetricEnum metric)
    {
        Assert.AreEqual(expectedResult, Utils.IsBetterScore(a, b, metric));
    }

    [TestCase(true, EvaluationMetricEnum.Accuracy)]
    [TestCase(true, EvaluationMetricEnum.SparseAccuracy)]
    [TestCase(true, EvaluationMetricEnum.AccuracyCategoricalCrossentropyWithHierarchy)]
    [TestCase(true, EvaluationMetricEnum.F1Micro)]
    [TestCase(true, EvaluationMetricEnum.PearsonCorrelation)]
    [TestCase(true, EvaluationMetricEnum.SpearmanCorrelation)]
    [TestCase(true, EvaluationMetricEnum.AUC)]
    [TestCase(true, EvaluationMetricEnum.AveragePrecisionScore)]
    [TestCase(false, EvaluationMetricEnum.Huber)]
    [TestCase(false, EvaluationMetricEnum.Mae)]
    [TestCase(false, EvaluationMetricEnum.Mse)]
    [TestCase(false, EvaluationMetricEnum.MseOfLog)]
    [TestCase(false, EvaluationMetricEnum.MeanSquaredLogError)]
    [TestCase(false, EvaluationMetricEnum.Rmse)]
    [TestCase(false, EvaluationMetricEnum.BinaryCrossentropy)]
    [TestCase(false, EvaluationMetricEnum.BCEContinuousY)]
    [TestCase(false, EvaluationMetricEnum.BCEWithFocalLoss)]
    [TestCase(false, EvaluationMetricEnum.CategoricalCrossentropy)]
    [TestCase(false, EvaluationMetricEnum.CategoricalCrossentropyWithHierarchy)]
    public void TestHigherScoreIsBetter(bool expectedResult, EvaluationMetricEnum metric)
    {
        Assert.AreEqual(20, Enum.GetNames(typeof(EvaluationMetricEnum)).Length, $"expecting {Enum.GetNames(typeof(EvaluationMetricEnum)).Length} distinct tests for each {typeof(EvaluationMetricEnum)}");
        Assert.AreEqual(expectedResult, Utils.HigherScoreIsBetter(metric));
    }
    

        [Test]
    public void TestBetaDistribution()
    {
        double sum = 0.0;
        double sumSquare = 0.0;
        var rand = new Random(0);
        const int count = 100000;
        for (int i = 0; i < count; ++i)
        {
            var val = Utils.BetaDistribution(1.0, 1.0, rand);
            sum += val;
            sumSquare += val * val;
        }
        var mean = sum / count;
        var meanOfSquare = sumSquare / count;
        var variance = Math.Max(0,meanOfSquare - mean * mean);
        const double epsilon = 0.01;
        Assert.AreEqual(0.5, mean, epsilon);
        Assert.AreEqual(1/12.0, variance, epsilon);
    }
    [Test]
    public void TestNewVersion()
    {
        Assert.AreEqual(new Version(7,6,5), Utils.NewVersion(7605));
        Assert.AreEqual(new Version(7,6,0), Utils.NewVersion(7600));
    }

        
    [Test]
    public void TestNewVersionXXYY0()
    {
        Assert.AreEqual(new Version(9, 2), Utils.NewVersionXXYY0(9020));
        Assert.AreEqual(new Version(10, 1), Utils.NewVersionXXYY0(10010));
    }


    [Test]
    public void TestTargetCpuInvestmentTime()
    {
        AssertAreEqual(Array.Empty<double>(), Utils.TargetCpuInvestmentTime(new List<Tuple<double, double, long>>()), 1e-6);

        var empty = Tuple.Create(0.0, 0.0, 0L);
        var single_lowError = Tuple.Create(5.0, 0.0, 1L);
        var single_highError = Tuple.Create(10.0, 0.0, 1L);
        var lowError = Tuple.Create(5.0, 5.0, 99L);
        var highError = Tuple.Create(10.0, 5.0, 99L);

        foreach (var e in new [] { empty, single_highError, single_lowError, lowError, highError })
        {
            AssertAreEqual(new[] { 1.0 },  Utils.TargetCpuInvestmentTime(new List<Tuple<double, double, long>> { e }), 1e-6);
        }

        foreach (var e in new[] { empty, single_highError, single_lowError, lowError, highError })
        {
            AssertAreEqual(new[] { 0.5, 0.5 }, Utils.TargetCpuInvestmentTime(new List<Tuple<double, double, long>> { empty, e }), 1e-6);
        }

        foreach (var e in new[] { empty, single_highError, single_lowError, lowError, highError })
        {
            AssertAreEqual(new[] { 0.25, 0.25, 0.25 , 0.25 }, Utils.TargetCpuInvestmentTime(new List<Tuple<double, double, long>> { empty, e, single_highError, single_lowError }), 1e-6);
        }

        AssertAreEqual(new[] { 0.75, 0.25}, Utils.TargetCpuInvestmentTime(new List<Tuple<double, double, long>> { lowError, highError }), 1e-6);
        AssertAreEqual(new[] { 0.25, 0.75}, Utils.TargetCpuInvestmentTime(new List<Tuple<double, double, long>> { highError, lowError  }), 1e-6);
        AssertAreEqual(new[] { 0.25, 0.25/2, 0.75/2, 0.25}, Utils.TargetCpuInvestmentTime(new List<Tuple<double, double, long>> { single_highError, highError, lowError, single_lowError  }), 1e-6);
    }

    private static void AssertAreEqual(double[] expected, double[] observed, double epsilon)
    {
        Assert.AreEqual(expected.Length, observed.Length);
        for (int i = 0; i < expected.Length; ++i)
        {
            Assert.AreEqual(expected[i], observed[i], epsilon);
        }
    }

    [Test]
    public void TestRandomIndexBasedOnWeights()
    {
        var rand = new Random();
        var weights = new[] { 3 * 1.0, 3 * 7.0, 3 * 2.0 };
        var indexes = new int[weights.Length];
        for (int i = 0; i < 10000; ++i)
        {
            ++indexes[Utils.RandomIndexBasedOnWeights(weights, rand)];
        }
        Debug.Assert(Math.Abs(indexes[0] - 1000) < 100);
        Debug.Assert(Math.Abs(indexes[1] - 7000) < 500);
        Debug.Assert( Math.Abs(indexes[2]-2000)<200 );
    }

    [TestCase("", "")]
    [TestCase("", " ")]
    [TestCase("", "\t")]
    [TestCase("a b", "\t \"a b;")]
    [TestCase("a b", " a b,;;;")]
    [TestCase("a b", " a\tb")]
    [TestCase("a  b", "a\n\rb")]
    public void TestNormalizeCategoricalFeatureValue(string expected, string str)
    {
        Assert.AreEqual(expected, Utils.NormalizeCategoricalFeatureValue(str));
    }


    [TestCase(1, 0)]
    [TestCase(1, 1)]
    [TestCase(2, 2)]
    [TestCase(256, 256)]
    [TestCase(512, 257)]
    [TestCase(256, 255)]
    public void TestNextPowerOf2(int expected, int n)
    {
        Assert.AreEqual(expected, Utils.NextPowerOf2(n));

    }

    [TestCase(0, 0)]
    [TestCase(1, 1)]
    [TestCase(2, 2)]
    [TestCase(2, 3)]
    [TestCase(256, 256)]
    [TestCase(256, 257)]
    [TestCase(128, 255)]
    public void TestPrevPowerOf2(int expected, int n)
    {
        Assert.AreEqual(expected, Utils.PrevPowerOf2(n));
    }

    [TestCase(0, 100, 100)]
    [TestCase(100, 100, 150)]
    [TestCase(0, 100, 50)]
    [TestCase(1, 7,3)]
    [TestCase(1, -99, 10)]
    public void AlwaysPositiveModulo(int expected, int x, int modulo)
    {
        Assert.AreEqual(expected, Utils.AlwaysPositiveModulo(x, modulo));

    }


}