using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using ProtoBuf;
// ReSharper disable MemberCanBePrivate.Global

namespace SharpNet.Datasets.CFM60
{
    [ProtoContract]
    public class CFM60Entry : TimeSeriesSinglePoint
    {
        public const int POINTS_BY_DAY = 61;
        public const int DISTINCT_PID_COUNT = 900;  //total number of distinct companies (pid)


        public static bool IsInterpolatedId(int id) {return id < 0;}


        /// <summary>
        /// parameter less constructor needed for ProtoBuf serialization 
        /// </summary>
        // ReSharper disable once UnusedMember.Global
        public CFM60Entry() { }

        private CFM60Entry(string line)
        {
            var splitted = line.Split(',');
            int index = 0;
            ID = int.Parse(splitted[index++]);
            pid = int.Parse(splitted[index++]);
            day = int.Parse(splitted[index++]);
            abs_ret = new float[POINTS_BY_DAY];
            for (int i = 0; i < POINTS_BY_DAY; ++i)
            {
                if (double.TryParse(splitted[index++], out var tmp))
                {
                    abs_ret[i] = (float)tmp;
                }
            }
            rel_vol = new float[POINTS_BY_DAY];
            for (int i = 0; i < POINTS_BY_DAY; ++i)
            {
                if (double.TryParse(splitted[index++], out var tmp))
                {
  