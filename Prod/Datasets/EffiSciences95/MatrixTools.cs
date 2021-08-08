
﻿using System;
using System.Collections.Generic;
using System.Linq;
using SharpNet.Pictures;

namespace SharpNet.Datasets.EffiSciences95;


/// <summary>
/// different tools used to make fast computation on a matrix (2D array)
/// it is mainly based on precomputed sub matrix sum (see https://www.techiedelight.com/calculate-sum-elements-sub-matrix-constant-time/)
/// </summary>
public static class MatrixTools
{
    public static int[,] CreateCountMatrix(bool[,] m)
    {
        var countMatrix = new int[m.GetLength(0), m.GetLength(1)];
        for (int row = 0; row < m.GetLength(0); ++row)
        {
            for (int col = 0; col < m.GetLength(1); ++col)
            {
                countMatrix[row, col] =
                    (m[row, col] ? 1 : 0)
                    + Default(countMatrix, row, col - 1, 0)
                    + Default(countMatrix, row - 1, col, 0)
                    - Default(countMatrix, row - 1, col - 1, 0);
            }
        }
        return countMatrix;
    }
    public static List<Tuple<int, int>> ExtractValidIntervals(bool[] validCols, int first_idx, int minLengthOfInterval, Func<int, bool> isEmpty)
    {
        List<Tuple<int, int>> res = new();
        int start = -1;
        for (int row = first_idx; row < first_idx + validCols.Length; ++row)
        {
            int idx_in_validCols = row - first_idx;
            if (validCols[idx_in_validCols])
            {
                if (start == -1)
                {
                    if (isEmpty(row))
                    {
                        continue;
                    }
                    start = row;
                }
                if (idx_in_validCols == validCols.Length - 1)
                {
                    if ((row - start + 1) >= minLengthOfInterval)
                    {
                        res.Add(new Tuple<int, int>(start, row));
                    }
                    start = -1;
                }
            }
            else
            {
                if (start != -1)
                {
                    int end = row - 1;
                    while (isEmpty(end) && end > start)
                    {
                        --end;
                    }
                    if ((end - start + 1) >= minLengthOfInterval)
                    {
                        res.Add(new Tuple<int, int>(start, end));
                    }
                    start = -1;
                }
            }
        }
        return res;
    }
    public static void SetRow(bool[,] m, int row, int col_start, int col_end, bool newValue)
    {
        for (int col = col_start; col <= col_end; ++col)
        {
            m[row, col] = newValue;
        }
    }
    public static void SetCol(bool[,] m, int col, int row_start, int row_end, bool newValue)
    {
        for (int row = row_start; row <= row_end; ++row)
        {
            m[row, col] = newValue;
        }
    }
    public static int[,] ToKMeanIndex(RGBColor[,] allColors, IList<RGBColor> KMeanRoots, Func<RGBColor, RGBColor, double>[] ColorDistances, double[] computeMainColorFromPointsWithinDistances, int remainingTries)
    {
        int[] counts = new int[KMeanRoots.Count];
        var result = new int[allColors.GetLength(0), allColors.GetLength(1)];
        for (int row = 0; row < allColors.GetLength(0); ++row)
            for (int col = 0; col < allColors.GetLength(1); ++col)
            {
                int i = -1;
                result[row, col] = -1;
                double minDistance = 2 * computeMainColorFromPointsWithinDistances.Max();
                for (int j = 0; j < KMeanRoots.Count; ++j)
                {
                    double distance = ColorDistances[j](KMeanRoots[j], allColors[row, col]);
                    if (distance <= minDistance)
                    {
                        minDistance = distance;
                        i = j;
                    }
                }
                if (i >= 0 && minDistance <= computeMainColorFromPointsWithinDistances[i])
                {
                    result[row, col] = i;
                    ++counts[i];
                }
            }

        var to_check_for_count = new int[] { 0, 1, 2, 4};
        if (remainingTries > 0 && to_check_for_count.Any(i => counts[i] >= 1700))
        {

            computeMainColorFromPointsWithinDistances = (double[])computeMainColorFromPointsWithinDistances.Clone();
            foreach (var j in to_check_for_count)
            {
                if (counts[j] > 1700)
                {
                    computeMainColorFromPointsWithinDistances[j] /= 1.5;
                }
            }
            return ToKMeanIndex(allColors, KMeanRoots, ColorDistances, computeMainColorFromPointsWithinDistances, remainingTries - 1);
        }

        return result;
    }
    public static int RowCount(int[,] countMatrix, int row, int col_start, int col_end)
    {
        return Count(countMatrix, row, row, col_start, col_end);
    }
    public static int ColCount(int[,] countMatrix, int col, int row_start, int row_end)
    {
        return Count(countMatrix, row_start, row_end, col, col);
    }
    public static double Density(int[,] countMatrix, int row_start, int row_end, int col_start, int col_end)
    {
        int count = Count(countMatrix, row_start, row_end, col_start, col_end);
        return count / ((row_end - row_start + 1.0) * (col_end - col_start + 1.0));
    }
    public static bool[,] ExtractBoolMatrix(int[,] kmeanTextColorIndex, int index)
    {