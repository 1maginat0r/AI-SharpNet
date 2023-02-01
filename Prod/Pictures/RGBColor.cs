using System;
using System.Collections.Generic;

namespace SharpNet.Pictures;

public class RGBColor : IEquatable<RGBColor>
{
    #region private fields
    private readonly byte red;
    private readonly byte green;
    private readonly byte blue;
    private int Index24Bits { get; }
    private LabColor _lazyLab;
    #endregion

    public static byte ToRGBColor(double color_255)
    {
        return (byte)Utils.NearestInt(255 * color_255);
    }


    private static readonly RGBColor Black = new (0, 0, 0);
    private static readonly RGBColor White = new (255, 255, 255);
    //public static readonly RGBColor RedRef = new (255, 0, 0);
    //public static readonly RGBColor OrangeRef = new (255, 165, 0);
    //public static readonly RGBColor GreenRef = new (0, 255, 0);
    //public static readonly RGBColor PinkRef = new (255, 102, 204);
    //public static readonly RGBColor GreyRef = new (161, 169, 164);
    //public static readonly RGBColor DarkBlue = new (0, 0, 139);
    //public static readonly RGBColor ClearBlue = new (135, 206, 235);
    public RGBColor(byte red, byte green, byte blue)
    {
        this.red = red;
        this.green = green;
        this.blue = blue;
        Index24Bits = red + (green << 8) + (blue << 16);
    }
    public byte Red => red;
    public byte Green => green;
    public byte Blue => blue;

    // ReSharper disable UnusedMember.Global
    public double DistanceToWhite { get { return ColorDistance(this, White); } }
    public double DistanceToBlack { get { return ColorDistance(this, Black); } }


    public LabColor Lab
    {
        get
        {
            if (_lazyLab == null)
            {
                _lazyLab = LabColor.RGB2Lab(red, green, blue);
            }
            return _lazyLab;
        }
    }
    
    public List<double> Distance(IEnumerable<RGBColor> colors)
    {
        var result = new List<double>();
        foreach (var c in colors)
        {
            result.Add(ColorDistance(this, c));
        }

        return result;
    }

    private static double ColorDistance(RGBColor a, RGBColor b)
    {
        return a.Lab.Distance(b.Lab);
    }
    public double ColorDistance(RGBColor b)
    {
        return ColorDistance(this,b);
    }

    public static double HueDistanceInDegrees(double hue1InDegrees, double hue2InDegrees)
    {
        double hueDistanceInDegrees = Math.Abs(hue1InDegrees - hue2InDegrees);
        if (hueDistanceInDegrees >= 180)
        {
            hue