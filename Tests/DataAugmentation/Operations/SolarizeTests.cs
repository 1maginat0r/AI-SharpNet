using System;
using System.Collections.Generic;
using System.Linq;
using NUnit.Framework;
using SharpNet.DataAugmentation;
using SharpNet.DataAugmentation.Operations;

namespace SharpNetTests.DataAugmentation.Operations
{
    [TestFixture]
    public class SolarizeTests
    {


        [Test]
        public void TestSolarize()
        {
            // 4x4 matrix, 1 channel, no normalization
            var input