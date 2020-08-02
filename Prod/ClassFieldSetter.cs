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
                result.Add($"{parameterName} = {Utils.FieldValueToString(fieldInfo.GetValue(t))}");
            }
            return string.Join(Environment.NewLine, result) + Environment.NewLine;
        }
        public static Type GetFieldType(Type t, string fieldName)
        {
            return GetFieldName2FieldInfo(t)[fieldName].FieldType;
        }
        public static IEnumerable<string> FieldNames(Type t)
        {
            return GetFieldName2FieldInfo(t).Keys;
        }


        #region private methods
        private static void Set(object o, Type objectType, string fieldName, object fieldValue)
        {
            var f = GetFieldInfo(objectType, fieldName);
            if (fieldValue.GetType() == f.FieldType)
            {
                f.SetValue(o, fieldValue);
                return;
            }
            if (f.FieldType == typeof(bool))
            {
                SetBoolField(o, objectType, fieldName, fieldValue);
            }
            else if (f.FieldType == typeof(int))
            {
                SetIntField(o, objectType, fieldName, fieldValue);
            }
            else if (f.FieldType == typeof(float))
            {
                SetFloatField(o, objectType, fieldName, fieldValue);
            }
            else if (f.FieldType == typeof(double))
            {
                SetDoubleField(o, objectType, fieldName, fieldValue);
            }
            else if (f.FieldType == typeof(string))
            {
                SetStringField(o, objectType, fieldName, fieldValue);
            }
            else if (f.FieldType.IsEnum)
            {
                // ReSharper disable once AssignNullToNotNullAttribute
                f.SetValue(o, Enum.Parse(f.FieldType, fieldValue.ToString()));
            }
            else if (fieldValue is string && f.FieldType.IsGenericType && f.FieldType.GetGenericTypeDefinition() == typeof(List<>))
            {
                var res = ParseStringToListOrArray((string)fieldValue, f.FieldType.GetGenericArguments()[0], true);
                f.SetValue(o, res);
            }
            else if (fieldValue is string && f.FieldType.IsArray)
            {
                var res = ParseStringToListOrArray((string)fieldValue, f.FieldType.GetElementType(), false);
                f.SetValue(o, res);
            }
            else
            {
                throw new Exception($"invalid field {fieldName} with value {fieldValue}");
            