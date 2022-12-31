using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using PhotoSauce.MagicScaler;
using SharpNet.CPU;
using System;
using System.Runtime.InteropServices;

namespace SharpNet.Pictures
{
    public class BitmapContent : CpuTensor<byte>
    {
        /// <summary>
        /// Load a RGB bitmap (with 3 channels)
        /// </summary>
        /// <param name="filename"></param>
        /// <returns></returns>
        public static BitmapContent ValueFomSingleRgbBitmap(string filename)
        {
            using var bmp = new Bitmap(filename);
            return ValueFomSingleRgbBitmap(bmp);
        }
        /// <summary>
        /// Construct an element stacking several bitmaps, each bitmap containing a single channel
        /// </summary>
        /// <param name="singleChannelBitmaps">list of bitmap files, each containing a single channel (meaning R=G=B)</param>
        /// <returns></returns>
        public static BitmapContent ValueFromSeveralSingleChannelBitmaps(IEnumerable<string> singleChannelBitmaps)
        {
            var bmps = singleChannelBitmaps.Select(filename => new Bitmap(filename)).ToList();
            var result = ValueFromSeveralSingleChannelBitmaps(bmps);
            bmps.ForEach(bmp=>bmp.Dispose());
            return result;
        }
        public BitmapContent(int[] shape, byte[] data) : base(shape, data)
        {
        }
        public int GetChannels() => Shape[0];
        public int GetHeight() => Shape[1];
        public int GetWidth() => Shape[2];
        public BitmapContent MakeSquarePictures(bool alwaysUseBiggestSideForWidthSide, bool alwaysCropInsidePicture, Tuple<byte, byte, byte> fillingColor)
        {
            int heightAndWidth = alwaysCropInsidePicture
                ? Math.Min(GetHeight(), GetWidth())
                : Math.Max(GetHeight(), GetWidth());
     