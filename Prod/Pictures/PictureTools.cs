
﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Runtime.InteropServices;
using SharpNet.CPU;
using SharpNet.MathTools;
using Image = System.Drawing.Image;
using Rectangle = System.Drawing.Rectangle;
using Size = System.Drawing.Size;

namespace SharpNet.Pictures
{
    public static class PictureTools
    {
        public static void SavePng(Bitmap bmp, string filename)
        {
            var dir = new FileInfo(filename).Directory;
            if ((dir != null) && (!dir.Exists))
            {
                dir.Create();
            }
            bmp.Save(Utils.UpdateFilePathChangingExtension(filename, "", "", ".png"));
        }
        public static List<KeyValuePair<CpuTensor<byte>, int>> ReadInputPictures(string fileData, string fileLabels)
        {
            var result = new List<KeyValuePair<CpuTensor<byte>, int>>();
            var fileDataBytes = File.ReadAllBytes(fileData);
            var fileLabelBytes = File.ReadAllBytes(fileLabels);
            int nbItems = ToInt32(fileDataBytes, 4);
            int height = ToInt32(fileDataBytes, 8);
            int width = ToInt32(fileDataBytes, 12);
            for (int index = 0; index < nbItems; ++index)
            {

                var matrix = Load(fileDataBytes, 16 + index * (height * width), width, height);
                var target = fileLabelBytes[8 + index];
                result.Add(new KeyValuePair<CpuTensor<byte>, int>(new CpuTensor<byte>(new[] { width, height }, matrix), target));
            }
            return result;
        }
        // ReSharper disable once UnusedMember.Global
        public static void SaveBitmap<T>(CpuTensor<T> xTrain, int pictureIndex, string directory, string filePrefix, string fileSuffix)
        {
            switch (Marshal.SizeOf(typeof(T)))
            {
                case 1: SaveBitmap(xTrain as CpuTensor<byte>, pictureIndex, directory, filePrefix, fileSuffix);break;
                case 4: SaveBitmap(xTrain as CpuTensor<float>, pictureIndex, directory, filePrefix, fileSuffix);break;
                default: throw new ArgumentException(Marshal.SizeOf(typeof(T)) + " size is not supported");
            }
        }

        // ReSharper disable once UnusedMember.Global
        public static void SaveTensor2D_to_GreyscaleBitmap(CpuTensor<float> tensor2D, string fileName)
        {
            using var bmp = Tensor2D_to_GreyscaleBitmap(tensor2D.Reshape(-1, tensor2D.Shape[^1]).AsFloatCpu);
            Save(bmp, fileName);
        }

        public static bool IsPicture(string fullName)
        {
            var toLower = (fullName ?? "").ToLowerInvariant();
            foreach (var extension in new[] { ".jpeg", ".jpg", ".bmp", ".png" })
            {
                if (toLower.EndsWith(extension))
                {
                    return true;
                }
            }
            return false;
        }
        // ReSharper disable once UnusedMember.Global
        public static Size ImageSize(string imagePath)
        {
            using var stream = File.OpenRead(imagePath);
            using var sourceImage = Image.FromStream(stream, false, false);
            return sourceImage.Size;
        }

        private static void SaveBitmap(CpuTensor<byte> xTrain, int pictureIndex, string directory, string filePrefix, string fileSuffix)
        {
            SaveBitmap(xTrain, (x => x), pictureIndex, directory, filePrefix, fileSuffix);
        }
        private static void SaveBitmap(CpuTensor<float> xTrain, int pictureIndex, string directory, string filePrefix, string fileSuffix)
        {
            SaveBitmap(xTrain, (x => (byte)(255 * x)), pictureIndex, directory, filePrefix, fileSuffix);
        }
        private static void SaveBitmap<T>(CpuTensor<T> xTrain, Func<T, byte> toByte, int pictureIndex, string directory, string filePrefix, string fileSuffix) where T:struct
        {
            Save(AsBitmap(xTrain, toByte, pictureIndex), Path.Combine(directory, filePrefix + "_" + pictureIndex.ToString("D3") + "_" + fileSuffix));
        }
        private static Bitmap AsBitmap<T>(CpuTensor<T> xTrain, Func<T, byte> toByte, int pictureIndex) where T: struct
        {
            if (xTrain == null)
            {
                return null;
            }
            Debug.Assert(xTrain.Dimension == 4); //NCHW
            var h = xTrain.Shape[2];
            var w = xTrain.Shape[3];
            var bmp = new Bitmap(w, h, PixelFormat.Format24bppRgb);
            var rect = new Rectangle(0, 0, w, h);
            // Lock the bitmap's bits.  
            var bmpData = bmp.LockBits(rect, ImageLockMode.WriteOnly, bmp.PixelFormat);
            var rgbValues = new byte[bmpData.Stride * bmpData.Height];
            int index = 0;
            for (int rowIndex = 0; rowIndex < h; rowIndex++)
            {
                for (int colIndex = 0; colIndex < w; colIndex++)
                {
                    rgbValues[index + 2] = toByte(xTrain.Get(pictureIndex, 0, rowIndex, colIndex)); //R
                    rgbValues[index + 1] = toByte(xTrain.Get(pictureIndex, 1, rowIndex, colIndex)); //G
                    rgbValues[index] = toByte(xTrain.Get(pictureIndex, 2, rowIndex, colIndex)); //B
                    index += 3;
                }
                index += bmpData.Stride - w * 3;
            }
            Marshal.Copy(rgbValues, 0, bmpData.Scan0, rgbValues.Length);
            // Unlock the bits.
            bmp.UnlockBits(bmpData);
            return bmp;
        }
        private static void Save(Bitmap a, string filePath)
        {
            if (string.IsNullOrEmpty(filePath) || a == null)
            {
                return;
            }
            var dir = new FileInfo(filePath).Directory;
            if ((dir != null) && (!dir.Exists))
            {
                dir.Create();
            }
            a.Save(Utils.UpdateFilePathChangingExtension(filePath, "", "", ".png"));
        }
        private static int ToInt32(byte[] fileDataBytes, int startIndex)
        {
            byte[] buffer = new byte[4];
            for (int i = startIndex; i < startIndex + 4; ++i)
            {
                buffer[3 - (i - startIndex)] = fileDataBytes[i];
            }
            return BitConverter.ToInt32(buffer, 0);
        }
        private static byte[] Load(byte[] fileDataBytes, int startIndex, int width, int height)
        {
            var matrix = new byte[width * height];
            for (int x = 0; x < width; ++x)
            {
                for (int y = 0; y < height; ++y)
                {
                    matrix[x * height + y] = fileDataBytes[startIndex + x + y * width];
                }
            }
            return matrix;
        }
        private static Bitmap Tensor2D_to_GreyscaleBitmap(CpuTensor<float> tensor2D)
        {
            Debug.Assert(tensor2D.Shape.Length == 2);
            var h = tensor2D.Shape[0];
            var w = tensor2D.Shape[1];
            var bmp = new Bitmap(w, h, PixelFormat.Format24bppRgb);
            var rect = new Rectangle(0, 0, w, h);
            // Lock the bitmap's bits.  
            var bmpData = bmp.LockBits(rect, ImageLockMode.WriteOnly, bmp.PixelFormat);
            var rgbValues = new byte[bmpData.Stride * bmpData.Height];
            int index = 0;
            var spanContent = tensor2D.SpanContent;
            var acc = new DoubleAccumulator();
            acc.Add(spanContent);
            var minValue = acc.Min;
            var maxValue = acc.Max;
            int idx = 0;
            for (int rowIndex = 0; rowIndex < h; rowIndex++)
            {
                for (int colIndex = 0; colIndex < w; colIndex++)
                {
                    float val = spanContent[idx++];

                    byte byteVal = 0;
                    if (minValue != maxValue)
                    {
                        byteVal = (byte)(255 * (val - minValue) / (maxValue - minValue));
                    }
                    rgbValues[index + 2] = byteVal; //R
                    rgbValues[index + 1] = byteVal; //G
                    rgbValues[index] = byteVal; //B
                    index += 3;
                }
                index += bmpData.Stride - w * 3;
            }
            Marshal.Copy(rgbValues, 0, bmpData.Scan0, rgbValues.Length);
            // Unlock the bits.
            bmp.UnlockBits(bmpData);
            return bmp;
        }

    }
}