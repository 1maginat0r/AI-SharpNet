using System.Drawing;
using System.Globalization;
namespace SharpNet.Datasets.EffiSciences95;


/// <summary>
/// for each picture, this class contains the estimate coordinates of the box around the label ("old" or "young") found in this picture
/// </summary>
public class EffiSciences95LabelCoordinates
{
    public EffiSciences95LabelCoordinates(int no, string label, int col_start, int row_start, int width, int height, double density, string descDensity, double confidenceLevel)
    {
        this.No = no;
        this.Label = label;
        this.Col_start = col_start;
        this.Row_start = row_start;
        this.Width = width;
        this.Height = height;
        this.Density = density;
        this.DescDensity = descDensity;
        this.ConfidenceLevel = confidenceLevel;
    }

    /// <summary>
    /// id of the picture
    /// </summary>
    public readonly int No; 
    /// <summary>
    /// for the training dataset:
    ///     label of the picture ("y" for "young" label , "o" for "old" label)
    /// fot the test dataset:
    ///     empty string
    /// </summary>
    public readonly string Label;
    /// <summary>
    /// coordinates of the start of the label box (from the left)
    /// </summary>
    private readonly int Col_start;
    /// <summary>
    /// coordinates of the start of the label box (from the top)
    /// </summary>
    private readonly int Row_start;
    /// <summary>
    /// with of the label box
    /// </summary>
 