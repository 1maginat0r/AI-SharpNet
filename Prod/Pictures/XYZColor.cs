using System;
using System.Diagnostics;

namespace SharpNet.Pictures
{
    public class XYZColor
    {
        public XYZColor(double x, double y, double z)
        {
            X = x;
            Y = y;
            Z = z;
        }
        public double X {get;}
        public double Y {get;}
        public double Z {get;}
        private static double GetGammaCorrectedV2(double r_OR_g_OR_b)
        {
            Debug.Assert(r_OR_g_OR_b >= -0.00001);
            Debug.Assert(r_OR_g_OR_b <= 1.00001);
            if (r_OR_g_OR_b > 0.04045)
            {
                return Math.Pow(((r_OR_g_OR_b + 0.055) / 1.055), 2.4);
            }
            else
            {
                return r_OR_g_OR_b / 12.92;
            }
        }
        private static double[] InvertGetGammaCorrected(double r_GammaCorrected, double g_GammaCorrected, double b_GammaCorrected)
        {
            var input = new[] { r_GammaCorrected, g_GammaCorrected, b_GammaCorrected };
            var result = new double[input.Length];
            for (int i = 0; i < input.Length; ++i)
            {
                input[i] = Math.Max(input[i], 0.0);
                input[i] = Math.Min(input[i], 1.0);
                if ((12.92 * input[i]) > 0.04045)
                {
                    result[i] = 1.055 * Math.Pow(input[i], 1.0 / 2.4) - 0.055;
                }
                else
                {
                    result[i] = input[i] * 12.92;
                }
            }
            return result;
        }

        //XYZ Color Space
        public RGBColor XYZ2RGB()
        {
            double r = 3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z;
            double g = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z;
            double b = 0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z;
            //return new RGBColor(ToRGBColor(r / 100.0), ToRGBColor(g / 100.0), ToRGBColor(b / 100.0));
            double[] notGammaCorrected = InvertGetGammaCorrected(r / 100.0, g / 100.0, b / 100.0);
            return new RGBColor(RGBColor.ToRGBColor(notGammaCorrected[0]), RGBColor.ToRGBColor(notGammaCorrected[1]), RGBColor.ToRGBColor(notGammaCorrected[2]));
        }
        public static XYZColor RGB2XYZ(byte r, byte g, byte b)
        {
            doubl