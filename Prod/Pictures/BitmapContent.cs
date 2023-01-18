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
                    var bmp = AsBitmapForChannel(channel);
                    PictureTools.SavePng(bmp, fileNames[channel]);
                    bmp.Dispose();
                }
            }
        }
        public RGBColor AverageColor(RGBColorFactoryWithCache cache)
        {
            var acc = new ColorAccumulator();
            Debug.Assert(3 == Shape[0]);
            var content = SpanContent;
            for (int i = 0; i < MultDim0; ++i)
            {
                acc.Add(cache.Build(content[i], content[i+MultDim0], content[i + 2*MultDim0]));
            }
            return acc.Average;
        }
        /// <summary>
        /// Compute Sum / Sum^2 / Count of each channel.
        /// THis will be used to compute Mean/Volatility of each channel
        /// </summary>
        /// <param name="_sum_SumSquare_Count_For_Each_Channel">
        /// _Sum_SumSquare_Count_For_Each_Channel[3*channel+0] : sum of all elements in channel 'channel'
        /// _Sum_SumSquare_Count_For_Each_Channel[3*channel+1] : sum of squares of all elements in channel 'channel'
        /// _Sum_SumSquare_Count_For_Each_Channel[3*channel+2] : count of all elements in channel 'channel'
        /// </param>
        public void UpdateWith_Sum_SumSquare_Count_For_Each_Channel(float[] _sum_SumSquare_Count_For_Each_Channel)
        {
            var content = ReadonlyContent;
            for (int channel = 0; channel < GetChannels(); ++channel)
            {
                var sum = 0f;
                var sumSquare = 0f;
                var startIdx = Idx(channel, 0, 0);
                var endIdxExcluded = Idx(channel+1, 0, 0);
                for (int idx = startIdx; idx< endIdxExcluded; ++idx)
                {
                    var val = content[idx];
                    sum += val;
                    sumSquare += val * val;
                }
                lock (_sum_SumSquare_Count_For_Each_Channel)
                {
                    _sum_SumSquare_Count_For_Each_Channel[3 * channel] += sum;
                    _sum_SumSquare_Count_For_Each_Channel[3 * channel + 1] += sumSquare;
                    _sum_SumSquare_Count_For_Each_Channel[3 * channel + 2] += GetWidth()*GetHeight();
                }
            }
        }


        public static BitmapContent Resize(string path, int targetWidth, int targetHeight)
        {
            return CropAndResize(