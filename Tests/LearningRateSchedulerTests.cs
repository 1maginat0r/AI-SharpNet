
﻿using NUnit.Framework;
using SharpNet.Data;
using SharpNet.Optimizers;


namespace SharpNetTests
{
    [TestFixture]
    public class LearningRateSchedulerTests
    {
        [Test]
        public void LearningRateSchedulerTest()
        {
        }

        private const double epsilon = 1e-8;
        [Test()]
        public void ConstantTest()
        {
            var lr = LearningRateScheduler.Constant(0.5);
            Assert.AreEqual(0.5, lr.LearningRate(1, 0), epsilon);
            Assert.AreEqual(0.5, lr.LearningRate(100, 0), epsilon);
            Assert.AreEqual(0.5, lr.LearningRate(0, 0), epsilon);
            Assert.AreEqual(0.5, lr.LearningRate(-1, 0), epsilon);
        }

        [Test]
        public void ConstantByIntervalTest1()
        {
            var lr = LearningRateScheduler.ConstantByInterval(1, 0.1, 3, 0.3);
            Assert.AreEqual(0.1, lr.LearningRate(-1, 0), epsilon);
            Assert.AreEqual(0.1, lr.LearningRate(0, 0), epsilon);
            Assert.AreEqual(0.1, lr.LearningRate(1, 0), epsilon);
            Assert.AreEqual(0.1, lr.LearningRate(2, 0), epsilon);
            Assert.AreEqual(0.3, lr.LearningRate(3, 0), epsilon);
            Assert.AreEqual(0.3, lr.LearningRate(100, 0), epsilon);
        }

        [Test]
        public void ConstantByIntervalTest2()
        {
            var lr = LearningRateScheduler.ConstantByInterval(1, 0.1, 3, 0.3, 6, 0.9);
            Assert.AreEqual(0.1, lr.LearningRate(-1, 0), epsilon);
            Assert.AreEqual(0.1, lr.LearningRate(0, 0), epsilon);
            Assert.AreEqual(0.1, lr.LearningRate(1, 0), epsilon);
            Assert.AreEqual(0.1, lr.LearningRate(2, 0), epsilon);
            Assert.AreEqual(0.3, lr.LearningRate(3, 0), epsilon);
            Assert.AreEqual(0.3, lr.LearningRate(4, 0), epsilon);
            Assert.AreEqual(0.3, lr.LearningRate(5, 0), epsilon);
            Assert.AreEqual(0.9, lr.LearningRate(6, 0), epsilon);
            Assert.AreEqual(0.9, lr.LearningRate(100, 0), epsilon);
        }

        [TestCase(true)]
        [TestCase(false)]
        public void TestSerialization(bool isConstantByInterval)
        {
            var lr = isConstantByInterval
                    ?LearningRateScheduler.ConstantByInterval(1, 0.1, 3, 0.3, 6, 0.9)
                    : LearningRateScheduler.InterpolateByInterval(1, 0.1, 3, 0.3, 6, 0.9);
            var serialized = Serializer.Deserialize(lr.Serialize());
            var lr2 = LearningRateScheduler.ValueOf(serialized);
            for (int epoch = -1; epoch <= 100; ++epoch)
            {
                Assert.AreEqual(lr2.LearningRate(epoch, 0), lr.LearningRate(epoch, 0), epsilon);
            }
        }

        [Test]
        public void InterpolateByIntervalTest1()
        {
            var lr = LearningRateScheduler.InterpolateByInterval(1,0.1,3,0.3);
            Assert.AreEqual(0.1, lr.LearningRate(-1, 0), epsilon);
            Assert.AreEqual(0.1, lr.LearningRate(0, 0), epsilon);
            Assert.AreEqual(0.1, lr.LearningRate(1, 0), epsilon);
            Assert.AreEqual(0.1, lr.LearningRate(1, 0.5), epsilon);
            Assert.AreEqual(0.2, lr.LearningRate(2, 0), epsilon);
            Assert.AreEqual(0.2, lr.LearningRate(2, 1.0), epsilon);
            Assert.AreEqual(0.3, lr.LearningRate(3, 0), epsilon);
            Assert.AreEqual(0.3, lr.LearningRate(100, 0), epsilon);
        }

        [Test]
        public void InterpolateByIntervalTest2()
        {
            var lr = LearningRateScheduler.InterpolateByInterval(1, 0.1, 3, 0.3,6,0.9);
            Assert.AreEqual(0.1, lr.LearningRate(-1, 0), epsilon);
            Assert.AreEqual(0.1, lr.LearningRate(0, 0), epsilon);
            Assert.AreEqual(0.1, lr.LearningRate(1, 0), epsilon);
            Assert.AreEqual(0.2, lr.LearningRate(2, 0), epsilon);
            Assert.AreEqual(0.2, lr.LearningRate(2, 0.5), epsilon);
            Assert.AreEqual(0.3, lr.LearningRate(3, 0), epsilon);
            Assert.AreEqual(0.5, lr.LearningRate(4, 0), epsilon);
            Assert.AreEqual(0.5, lr.LearningRate(4, 0.75), epsilon);
            Assert.AreEqual(0.7, lr.LearningRate(5, 0), epsilon);
            Assert.AreEqual(0.9, lr.LearningRate(6, 0), epsilon);
            Assert.AreEqual(0.9, lr.LearningRate(100, 0), epsilon);
        }


        [Test]
        public void InterpolateByIntervalTest3()
        {
            var startLearningRate = 0.0;
            var lastLearningRate = 100.0;
            var lr = LearningRateScheduler.Linear(startLearningRate, 100, lastLearningRate);
            Assert.AreEqual(startLearningRate, lr.LearningRate(-1, 0), epsilon);
            Assert.AreEqual(startLearningRate, lr.LearningRate(0, 0), epsilon);
            Assert.AreEqual(0.5, lr.LearningRate(1, 0.5), epsilon);
            Assert.AreEqual(1.0, lr.LearningRate(1, 1.0), epsilon);
            Assert.AreEqual(1.0, lr.LearningRate(2, 0), epsilon);
            Assert.AreEqual(2.0, lr.LearningRate(3, 0), epsilon);
            Assert.AreEqual(2.75, lr.LearningRate(3, 0.75), epsilon);
            Assert.AreEqual(98, lr.LearningRate(99, 0), epsilon);
            Assert.AreEqual(98.5, lr.LearningRate(99, 0.5), epsilon);
            Assert.AreEqual(99, lr.LearningRate(100, 0), epsilon);
            Assert.AreEqual(99.5, lr.LearningRate(100, 0.5), epsilon);
            Assert.AreEqual(lastLearningRate, lr.LearningRate(100, 1.0), epsilon);
            Assert.AreEqual(lastLearningRate, lr.LearningRate(500, 0), epsilon);
        }
    }
}