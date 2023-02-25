
﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using JetBrains.Annotations;
using log4net;
using Porter2StemmerStandard;
using SharpNet.CPU;
using SharpNet.Datasets;

namespace SharpNet.TextPreprocessing;

public static class TfIdfEncoding
{
    private static readonly IStemmer _stemmer = new EnglishPorter2Stemmer();

    private static readonly ILog Log = LogManager.GetLogger(typeof(DataSet));

    /// <summary>
    /// return the list of columns in 'df' that contains TfIdf encoding of column 'columnToEncode'
    /// </summary>
    /// <param name="df"></param>
    /// <param name="columnToEncode"></param>
    /// <returns></returns>
    private static string[] EncodedColumns(DataFrame df, string columnToEncode)
    {
        return df.Columns.Where(c => c.StartsWith(columnToEncode + TFIDF_COLUMN_NAME_KEY)).ToArray();
    }


    /// <summary>
    /// reduces the number of TfIdf features encoding the column 'columnToAdjustEncoding' in DataFrame 'df' to 'targetEmbeddingDim' columns,
    /// and return the new DataFrame.
    /// If the number of encoding columns for column 'columnToAdjustEncoding' is less than 'targetEmbeddingDim',
    /// then do nothing and returns the same the DataFrame
    /// </summary>
    /// <param name="df"></param>
    /// <param name="columnToAdjustEncoding"></param>
    /// <param name="targetEmbeddingDim"></param>
    /// <returns>a DataFrame with 'targetEmbeddingDim' (or less) columns used to encode 'columnToAdjustEncoding'</returns>
    public static DataFrame ReduceEncodingToTargetEmbeddingDim(DataFrame df, string columnToAdjustEncoding,
        int targetEmbeddingDim)
    {
        var existingEncodingColumns = EncodedColumns(df, columnToAdjustEncoding);
        if (existingEncodingColumns.Length <= targetEmbeddingDim)
        {
            Log.Debug(
                $"can't reduce encoding of column of {columnToAdjustEncoding} to {targetEmbeddingDim} because existing encoding is {existingEncodingColumns.Length}");
            return df;
        }

        return df.Drop(existingEncodingColumns.Skip(targetEmbeddingDim).ToArray());
    }

    public static IEnumerable<string> ColumnToRemoveToFitEmbedding(DataFrame df, string columnToAdjustEncoding,
        int targetEmbeddingDim, bool keepOriginalColumnNameWhenUsingEncoding)
    {
        var existingEncodingColumns = EncodedColumns(df, columnToAdjustEncoding).ToList();
        var toDrop = new List<string>();
        if (existingEncodingColumns.Count > targetEmbeddingDim)
        {
            toDrop = existingEncodingColumns.Skip(targetEmbeddingDim).ToList();
        }

        if (existingEncodingColumns.Count > toDrop.Count && !keepOriginalColumnNameWhenUsingEncoding)
        {
            toDrop.Add(columnToAdjustEncoding);
        }

        return toDrop;
    }


    public enum TfIdfEncoding_norm
    {
        None,
        L1,
        L2,
        Standardization // we scale the embedding with 0 mean and 1 as standard deviation (= volatility). This is not a norm
    };
    public static List<DataFrame> Encode(IList<DataFrame> dfs, string columnToEncode, int embeddingDim, bool keepEncodedColumnName = false, bool reduceEmbeddingDimIfNeeded = false, TfIdfEncoding_norm norm = TfIdfEncoding_norm.L2, bool scikitLearnCompatibilityMode = false)
    {
        var documents = new List<string>();
        foreach (var df in dfs)
        {
            documents.AddRange(df?.StringColumnContent(columnToEncode) ?? Array.Empty<string>());
        }

        var documents_tfidf_encoded = Encode(documents, embeddingDim, columnToEncode, reduceEmbeddingDimIfNeeded, norm, scikitLearnCompatibilityMode);
        var dfs_encoded = new List<DataFrame>();
        var startRowIndex = 0;
        for (var index = 0; index < dfs.Count; index++)
        {
            var df = dfs[index];
            if (df == null)
            {
                dfs_encoded.Add(null);
                continue;
            }

            if (!keepEncodedColumnName)
            {
                df = df.Drop(columnToEncode);
            }

            var df_tfidf_encoded = documents_tfidf_encoded.RowSlice(startRowIndex, df.Shape[0], false);
            dfs_encoded.Add(DataFrame.MergeHorizontally(df, df_tfidf_encoded));
            startRowIndex += df.Shape[0];
        }

        return dfs_encoded;
    }



    public static DataFrame Encode(IList<string> documents, int embeddingDim, [NotNull] string columnNameToEncode, bool reduceEmbeddingDimIfNeeded = false, TfIdfEncoding_norm norm = TfIdfEncoding_norm.L2, bool scikitLearnCompatibilityMode = false)
    {
        var tokenizer = new Tokenizer(oovToken: null, lowerCase: true, stemmer: _stemmer);
       
        tokenizer.FitOnTexts(documents);
        var training_sequences = tokenizer.TextsToSequences(documents);
        if (reduceEmbeddingDimIfNeeded && tokenizer.DistinctWords < embeddingDim)
        {
            embeddingDim = Math.Max(1, tokenizer.DistinctWords);
        }

        Log.Info($"number of distinct words for {columnNameToEncode}: {tokenizer.DistinctWords}");

        var tfIdf = new float[documents.Count * embeddingDim];
        // documentsContainingWord[wordIdx] : number of distinct documents containing the word at index 'wordIdx'
        var documentsContainingWord = new int[embeddingDim];

        //first step: we compute the Text Frequency (tf)
        //  tfIdf[documentId * embeddingDim + wordIdx] : number of time the word at 'wordIdx' appears in document 'documentId'
        for (int documentId = 0; documentId < documents.Count; ++documentId)
        {