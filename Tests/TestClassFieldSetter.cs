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
