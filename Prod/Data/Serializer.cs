using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using SharpNet.CPU;

namespace SharpNet.Data
{
    public class Serializer
    {
        private readonly StringBuilder _sb = new ();
        public Serializer Add(string description, int value)
        {
            _sb.Append("int;" + description + ";" + value + ";");
            return this;
        }
        public Serializer Add(string description, int[] values)
        {
            _sb.Append("intVector;" + description + ";" + values.Length + ";" + ((values.Length==0)?"":(ToString(values)+ ";")));
            return this;
        }
        public Serializer Add(string description, double value)
        {
            _sb.Append("double;" + description + ";" + value.ToString(CultureInfo.InvariantCulture) + ";");
            return this;
        }
        public Serializer Add(string description, double[] values)
        {
            _sb.Append("doubleVector;" + description + ";" + values.Length + ";" + ((values.Length == 0) ? "" : (ToString(values) + ";")));
            return this;
        }
        public Serializer Add(string description, Type value)
        {
            _sb.Append("Type;" + description + ";" + value.Name + ";");
            return this;
        }
        public Serializer Add(string description, string value)
        {
            value = (value ?? "").Replace(';', '_');
            _sb.Append("string;" + description + ";" + value + ";");
            return this;
        }
        public Serializer Add(string description, EpochData[] data)
        {
            return Add(description, data, x => Split(x.Serialize()));
        }

        private static string[] Split(string s)
        {
            return s.TrimEnd(';').Split(';');
        }

        private Serializer Add<T>(string description, T[] array, Func<T, string[]> serializer)
        {
            _sb.Append(typeof(T).Name+ "Vector;" + description + ";" + array.Length + ";");
            foreach (var t in array)
            {
                var strArray = serializer(t);
                _sb.Append(strArray.Length + ";" + string.Join(";", strArray)+";");
            }
            return this;
        }
        private static T[] DeserializeArray<T>(string[] splitted, int count, ref int startIndex, Func<string, T> convert)
        {
            var data = new T[count];
            for (int i = 0; i < count; ++i)
            {
                var elementCount = int.Parse(splitted[startIndex++]);
                var elementValue = "";
                for (int j = startIndex; j < startIndex + elementCount; ++j)
                {
                    elementValue += splitted[j] + ";";
                }
                startIndex += elementCount; 
                data[i] = convert(elementValue);
            }
            return data;
        }
        // ReSharper disable once UnusedMember.Global
        public Serializer Add(string description, float value)
        {
            _sb.Append("single;" + description + ";" + value.ToString("G9", CultureInfo.InvariantCulture) + ";");
            return this;
        }
        // ReSharper disable once UnusedMember.Global
        public Serializer Add(string description, float[] values)
        {
            _sb.Append("singleVector;" + description + ";" + values.Length + ";" + ((values.Length == 0) ? "" : (ToString(values) + ";")));
            return this;
        }
        public Serializer Add(string description, bool value)
        {
            _sb.Append("bool;" + description + ";" + value.ToString(CultureInfo.InvariantCulture) + ";");
            return this;
        }
        public Serializer Add(string description, Tensor value)
        {
            if (value != null)
            {
                _sb.Append(Serialize(description, value) + ";");
            }
            return this;
        }
        public override string ToString()
        {
            return _sb.ToString();
        }

     

        