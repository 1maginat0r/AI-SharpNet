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
            ret