using System;
using System.Collections.Generic;

namespace SharpNet.MathTools
{
    public sealed class DoubleAccumulator
	{
		#region private fields
		private double sum;
		private double sumSquare;
        private double min;
        private double max;
        #endregion

        public long Count { get; private set; }
        public double Average => (Count == 0) ? 0 : (sum / Count);
        public double Min => min;
 