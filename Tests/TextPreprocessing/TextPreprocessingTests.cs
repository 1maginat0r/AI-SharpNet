using System;
using System.Collections.Generic;
using NUnit.Framework;
using SharpNet.CPU;
using SharpNet.TextPreprocessing;

namespace SharpNetTests.TextPreprocessing
{
    [TestFixture]
    public class TextPreprocessingTests
    {
        [Test]
        public void TestPadding()
        {
            var sequences = new List<List<int>> { new() { 0, 1, 2 }, new() { 3, 4 } };

            //pre and post padding keeping the same length
            //pre padding
            var observed = PadSequenceTools.PadSequence(sequences);
            var expected = new CpuTensor<int>(new[] { 2, 3 }, new[] { 0, 1, 2, 0, 3, 4});
            Assert.IsTrue(expected.SpanContent.SequenceEqual(observed.SpanContent));
            observed = PadSequenceTools.PadSequence(sequences, -1, true, true);
            Assert.IsTrue(expected.SpanContent.SequenceEqual(observed.SpanContent));
            //post padding
            observed = PadSequenceTools.PadSequence(sequences, -1, false, true);
            expected = new CpuTensor<int>(new[] { 2, 3 }, new[] { 0, 1, 2, 3, 4,0 });
            Assert.IsTrue(expected.SpanContent.SequenceEqual(observed.SpanContent));

            //pre and post padding changing the length
            //pre padding
            observed = PadSequenceTools.PadSequence(sequences, 4, true, true);
            expected = new CpuTensor<int>(new[] { 2, 4 }, new[] { 0, 0, 1, 2, 0, 0, 3, 4 });
            Assert.IsTrue(expected.SpanContent.SequenceEqual(observed.SpanContent));
            //post padding
            observed = PadSequenceTools.PadSequence(sequences, 4, false, true);
            expected = new CpuTensor<int>(new[] { 2, 4 }, new[] { 0, 1, 2, 0, 3, 4, 0, 0 });
            Assert.IsTrue(expected.SpanContent.SequenceEqual(observed.SpanContent));

            //pre and post truncating changing the length
            //pre truncating
         