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
            var labelCoordinatesIn