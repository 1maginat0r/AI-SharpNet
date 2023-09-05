using System;
using System.Collections.Generic;
using System.Linq;
using NUnit.Framework;
using SharpNet;

// ReSharper disable FieldCanBeMadeReadOnly.Local
// ReSharper disable ConvertToConstant.Local
// ReSharper disable NonReadonlyMemberInGetHashCode

namespace SharpNetTests
{
    [TestFixture]
    public class TestClassFieldSetter
    {
        private enum NotUsedEnum {A,B,C};

        private class TestClass
        {
            public bool Bool = true;
            public int Int = 42;
            public float Float = 42;
            public double Double = 42;
            public List<double> Doubles = new() {50,51,52};
            public string String = "42";
            public NotUsedEnum Enum = NotUsedEnum.C;
            public List<NotUsedEnum> Enums1 = new() { NotUsedEnum.C, NotUsedEnum.B };
            public NotUsedEnum[] Enums2 = new[] { NotUsedEnum.A, NotUsedEnum.B, NotUsedEnum.B, NotUsedEnum.C };
        }

        [Test]
        public void TestSet()
        {
            var p = new TestClass();

            ClassFieldSetter.Set(p, "Bool", false);
            Assert.AreEqual(false, p.Bool);
            ClassFieldSetter.Set(p, "Int", 41);
            Assert.AreEqual(41, p.Int);
            ClassFieldSetter.Set(p, "Float", 40);
            Assert.AreEqual(40, p.Float, 1e-6);
            ClassFieldSetter.Set(p, "Double", 39);
            Assert.AreEqual(39, p.Double, 1e-6);
            ClassFieldSetter.Set(p, "Doubles", new List<double>{10.0,11.0,12.0});
            CollectionAssert.AreEqual(new List<double> { 10.0, 11.0, 12.0 }, p.Doubles);
            ClassFieldSetter.Set(p, "String", "38");
            Assert.AreEqual("38", p.String);
            ClassFieldSetter.Set(p, "Enum", "B");
            Assert.AreEqual(NotUsedEnum.B, p.Enum);
            ClassFieldSetter.Set(p, "Enum", NotUsedEnum.A);
            Assert.AreEqual(NotUsedEnum.A, p.Enum);

            var listEnums = new[] { NotUsedEnum.B, NotUsedEnum.A };
            ClassFieldSetter.Set(p, "Enums1", listEnums.ToList());
            CollectionAssert.AreEqual(listEnums.ToList(), p.Enums1);
            ClassFieldSetter.Set(p, "Enums1", "B,A");
            CollectionAssert.AreEqual(listEnums.ToList(), p.Enums1);
            ClassFieldSetter.Set(p, "Enums2", "B,A");
            CollectionAssert.AreEqual(listEnums.ToArray(), p.Enums2);
        }

        [Test]
        public void TestGet()
        {
            var p = new TestClass();

            p.Bool = false;
            Assert.AreEqual(false, ClassFieldSetter.Get(p, "Bool"));
            p.Int = 41;
            Assert.AreEqual(41, ClassFieldSetter.Get(p, "Int"));
            p.Float = 40;
            Assert.AreEqual(40, ClassFieldSetter.Get(p, "Float"));
            p.Double = 39;
            Assert.AreEqual(39, ClassFieldSetter.Get(p, "Double"));

            p.Doubles = new List<double> { 10.0, 11.