﻿using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using NUnit.Framework;
using SharpNet;
using SharpNet.Datasets;
using SharpNet.Layers;
using SharpNet.Models;
using SharpNet.Pictures;

namespace SharpNetTests.Datasets
{
    [TestFixture]
    public class TestCategoryHierarchy
    {

        [Test]
        public void TestCategoryNameToPrediction()
        {
            var stars = StarExample();
            foreach (var expectedCancelName in new[]{ "etoile_pleine", "etoile*", "etoile7", "etoile**", "etoile*7", "etoile3*", "etoile37"})
            {
                var path = ToPathForStars(expectedCancelName);
                var prediction = stars.ExpectedPrediction(path);
                var observedCancelName = stars.ExtractPredictionWithProba(prediction).Item1;
                Assert.AreEqual(expectedCancelName, observedCancelName);
            }
        }


        [TestCase(new[] { "full" }, "etoile_pleine")]
        [TestCase(new[] { "1digit"}, "etoile*")]
        [TestCase(new[] { "1digit", "7" }, "etoile7")]
        [TestCase(new[] { "2digits", "3", "*" }, "etoile3*")]
        [TestCase(new[] { "2digits", "*", "7" }, "etoile*7")]
        [TestCase(new[] { "2digits", "3", "7" }, "etoile37")]
        [TestCase(new[] { "2digits"}, "etoile**")]
        public void TestToPathForStars(string[] expectedPath, string cancelName)
        {
            Assert.AreEqual(expectedPath, ToPathForStars(cancelName));
        }

        private static string[] ToPathForStars(string cancelName)
        {
            if (cancelName.StartsWith("etoile_pleine")) {return new[] {"full" };}
            if (cancelName.Equals("etoile*")) {return new[] {"1digit" };}
            if (cancelName.Equals("etoile**")) {return new[] {"2digits" };}
            switch (cancelName.Length)
            {
                case 7: return new[] {"1digit", cancelName[6].ToString()};
                case 8: return new[] {"2digits", cancelName[6].ToString(), cancelName[7].ToString() };
                default: return null;
            }
        }

        [Test]
        public void TestGetExpected()
        {
            var stars = StarExample();
            Assert.IsTrue(Utils.SameContent(new float[] { -300, 0, 21, 30, 0, 0, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 91, 0, 0, 0, 0, 0, 0, 0, 0, 0 },  
                stars.ExpectedPrediction(new string[] { }), 1e-6));
            Assert.IsTrue(Utils.SameContent(new float[] { 30, 0, -160, 30, 0, 0, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -100, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
                stars.ExpectedPrediction(new [] {"full" }), 1e-6));
            Assert.IsTrue(Utils.SameContent(new float[] { 30, 0, -160, 30, 0, 0, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -100, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
                stars.ExpectedPrediction(new [] {"1digit" }), 1e-6));
            Assert.IsTrue(Utils.SameContent(new float[] { 30, 0, -160, 30, 0, 0, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 91, 0, 0, 0, 0, 1, 0, 0, 0, 0 },
                stars.ExpectedPrediction(new [] {"1digit", "5" }), 1e-6));
            Assert.IsTrue(Utils.SameContent(new float[] { 30, 0, -160, 30, 0, 0, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -100, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
                stars.ExpectedPrediction(new[] { "1digit", "*" }), 1e-6));
            Assert.IsTrue(Utils.