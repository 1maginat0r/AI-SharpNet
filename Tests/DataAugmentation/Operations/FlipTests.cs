using System;
using System.Linq;
using NUnit.Framework;
using SharpNet.DataAugmentation;
using SharpNet.DataAugmentation.Operations;

namespace SharpNetTests.DataAugmentation.Operations
{
    [TestFixture]
    public class FlipTests
    {

        [Test]
        public void TestHorizontalFlip()
        {
            //single element
            var input = new[] { 12f };
            var inputShape = new[] { 1, 1, 1, 1 };
            var expected = new[] { 12f };
            OperationTests.Check(new HorizontalFlip(inputShape[3]), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest);

            //single line
            input = new[] { 12f, 13, 14 };
            inputShape = new[] { 1, 1, 1, 3 };
            expected = new[] { 14f, 13, 12 };
            OperationTests.Check(new HorizontalFlip(inputShape[3]), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest);

            //single column
            input = new[] {