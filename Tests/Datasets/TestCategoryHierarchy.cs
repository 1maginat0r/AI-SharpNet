using System.Collections.Generic;
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


        [TestCase(new