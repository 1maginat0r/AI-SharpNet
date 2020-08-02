using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.Linq;
using System.Reflection;

namespace SharpNet
{
    public static class ClassFieldSetter
    {
        #region private fields
        private static readonly Dictionary<Type,  Dictionary<string, FieldInfo>> type2name2FieldInfo = new ();
        #endregion

        public static void Set(object o, string fieldName, object fieldValue)
        {
            Set(o, o.GetType(), fieldName, fieldValue);
        }
        public static object Get(object o, string fieldName)
        {
            return GetFieldInfo(o.GetType(), fieldName).GetValue(o);
        }
        /// <summary>
        /// public for testing purpose only
        /// </summary>
        /// <param name="t"></param>
        /// <returns></returns>
        public static string ToConfigContent(object t)
        {
            var type = t.GetType();
            var result = new List<string>();
            foreach (var (parameterName, fieldInfo) in GetFieldName2FieldInfo(type).OrderBy(f => f.Key))
            {
                result.Add($"{parameterName} = {Utils.FieldValueToStri