
using System;

namespace SharpNet.MathTools
{
    /// <summary>
    /// Compute a linear regression 
    ///     y = Slope * x + Intercept
    /// with:
    ///     x: the independent variable
    ///     y: the dependent variable
    /// </summary>
    public sealed class LinearRegression
    {
        #region private fields
        private double xy_sum;
        private double x_sum;
        private double xx_sum;
        private double y_sum;
        private double yy_sum;
        private int count;
        #endregion


        /// <summary>
        /// add the observation (x, y) that will be used for the linear regression
        /// </summary>
        /// <param name="x">independent variable</param>
        /// <param name="y">dependent variable</param>
        public void Add(double x, double y)
        {
            if (double.IsNaN(y))
            {
                return;
            }
            ++count;
            xy_sum += x * y;
            x_sum += x;