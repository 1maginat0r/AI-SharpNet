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
            var content = new byte[3 * heightAndWidth * heightAndWidth];
            var result = new BitmapContent(new[] { Shape[0], heightAndWidth, heightAndWidth }, content);
            bool swapWidthAndHeight = alwaysUseBiggestSideForWidthSide && GetHeight() > GetWidth();
            
            for (int channel = 0; channel < Shape[0]; ++channel)
            {
                var filling = (channel == 0) ? fillingColor.Item1 : (channel == 1 ? fillingColor.Item2 : fillingColor.Item3);
                for (int row = 0; row < heightAndWidth; ++row)
                {
                    int originalRow = row + (GetHeight()-heightAndWidth) / 2;
                    for (int col = 0; col < heightAndWidth; ++col)
                    {
                        int originalCol = col + (GetWidth() - heightAndWidth) / 2;
                        var originalVal = (originalRow >= 0 && originalCol >= 0 && originalRow < GetHeight() && originalCol < GetWidth())
                            ? Get(channel, originalRow, originalCol)
                            : filling;
                        if (swapWidthAndHeight)
                        {
                            result.Set(channel, col, row, originalVal);
                        }
                        else
                        {
                            result.Set(channel, row, col, originalVal);
                        }
                    }
                }
            }
            return result;

        }
        /// <summary>
        /// return the SHA-1 of a bitmap (160 bits stored in a string of 40 bytes in hexadecimal format: 0=>f)
        /// </summary>
        public string SHA1()
        {
            var byteArray = Content.ToArray();
            using var sha1 = new SHA1CryptoServiceProvider();
            var hashBytes = sha1.ComputeHash(byteArray);
            return BitConverter.ToString(hashBytes).Replace("-", "");
        }
        public void Save(params string[] fileNames)
        {
            if (fileNames.Length== 1)
            {
                Debug.Assert(GetChannels() == 3);
                var bmp = AsBitmap();
                PictureTools.SavePng(bmp, fileNames[0]);
                bmp.Dispose();
            }
            else
            {
                Debug.Assert(GetChannels() == fileNames.Length);
                for (int channel = 0; channel < GetChannels(); ++channel)
                {
