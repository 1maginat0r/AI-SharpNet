using NUnit.Framework;
using SharpNet.DataAugmentation;
using SharpNet.DataAugmentation.Operations;

namespace SharpNetTests.DataAugmentation.Operations
{
    [TestFixture]
    public class BrightnessTests
    {
        [Test]
        public void TestBrightness()
        {
            // 1x1 matrix, 3 channels, no normalization
            var input = new[] { 250f, 150f, 50f };
            const float blackMean = 0f;

            var inputShape = new[] { 1, 3, 1, 1 };
            var expected = new[] { (input[0] + blackMean) / 2, (input[1] + blackMean) / 2, (input[2] + blackMean) / 2 };
            var operation = new Brightness(0.5f, blackMean);
            Operatio