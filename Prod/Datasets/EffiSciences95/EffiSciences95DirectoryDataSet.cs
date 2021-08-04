using System;
using System.Collections.Generic;
using System.Drawing;
using SharpNet.Pictures;

namespace SharpNet.Datasets.EffiSciences95;

/// <summary>
/// this class is used to load each picture of a dataset (see method: OriginalElementContent).
/// Before returning the picture, it will remove the label (old or young) appearing in the picture
/// </summary>
public class EffiSciences95DirectoryDataSet : DirectoryDataSet
{
    private readonly List<EffiSciences95LabelCoordinates> _labelCoordinates;
    private readonly EffiSciences95DatasetSample _datasetSample;
    private readonly Random _r = new (Utils.RandomSeed());


    public static EffiSciences95DirectoryDataSet ValueOf(EffiSciences95DatasetSample datasetSample, string directory, int maxElementCount = -1)
    {
        //we load the coordinates of the box containing the label ("old" or "young") for each picture (as computed in method FindLabelCoordinates in class LabelFinder)
        var idToLabelCoordinatesInPicture = new EffiSciences95LabelCoordinatesDataset(directory).Content;
        // the known label from the Labeled (train) dataset
        var pictureId_to_TrainLabel = (directory == "Labeled")?EffiSciences95Utils.LoadPredictionFile("Labeled_TextTarget.csv") :null;
        
        List<List<string>> elementIdToPaths = new();
        List<string> elementId_to_pictureId = new();
        List<int> elementId_to_label = new();
        List<EffiSciences95LabelCoordinates> allLabelCoordinatesInPictures = new();

        for(int pictureId=0;pictureId<=EffiSciences95Utils.MaxPictureId(directory);++pictureId)
        {
            var labelCoordinatesInPicture = idToLabelCoordinatesInPicture.TryGetValue(pictureId, out var tmp) ? tmp : null;
            if (labelCoordinatesInPicture!= null && (labelCoordinatesInPicture.IsEmpty || !labelCoordinatesInPicture.HasKnownLabel))
            {
                labelCoordinatesInPicture = null;
            }
            // for Labeled (train) dataset:
            //    0 or 1
            // for Unlabeled (test) dataset:
            //    -1
            int label = -1; 
            if (directory=="Labeled")
            {
                if (labelCoordinatesInPicture == null)
                {
                    continue; //for training, we need to know the box shape for each picture to remove it while training
                }
                // we extract the label for this training pictureId
                label = pictureId_to_TrainLabel[pictureId];
            }
            elementIdToPaths.Add(new List<string> { EffiSciences95Utils.PictureIdToPath(pictureId, directory) });
            elementId_to_pictureId.Add(pictureId.ToString());
            elementId_to_label.Add(label);
            allLabelCoordinatesInPictures.Add(labelCoordinatesInPicture);
            if (maxElementCount != -1 && elementIdToPaths.Count >= maxElementCount)
            {
                break;
            }
        }

        return new EffiSciences95DirectoryDataSet(
            datasetSample,
            allLabelCoordinatesInPictures,
            elementIdToPaths,
            elementId_to_pictureId,
            elementId_to_label);
    }

    private EffiSciences95DirectoryDataSet(
        EffiSciences95DatasetSample datasetSample,
        List<EffiSciences95LabelCoordinates> labelCoordinates,
        List<List<string>> elementIdToPaths,
        List<string> y_IDs,
        List<int> elementIdToCategoryIndex
    )
        : base(
            datasetSample,
            elementIdToPaths, 
            elementIdToCategoryIndex, 
            null,
            EffiSciences95Utils.NAME, 
            PrecomputedMeanAndVolatilityForEachChannel,
            ResizeStrategyEnum.None, 
            null,
            y_IDs.ToArray())
    {
        _datasetSample = datasetSample;
        _labelCoordinates = labelCoordinates;
    }



    /// <summary>
    /// return a picture (from the training (Labeled) or test (Unlabeled) dataset) after removing the box that seems to contain the label 
    /// </summary>
    /// <returns></returns>
    public override BitmapContent OriginalElementCo