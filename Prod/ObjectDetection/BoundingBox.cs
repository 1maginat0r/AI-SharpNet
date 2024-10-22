
﻿using System;

namespace SharpNet.ObjectDetection
{
    public class BoundingBox
    {
        #region private fields
        private readonly double _colCenter;
        private readonly double _rowCenter;
        private readonly double _width;
        private readonly double _height;
        #endregion

        #region constructor
        public BoundingBox(double colCenter, double rowCenter, double width, double height)
        {
            _colCenter = colCenter;
            _rowCenter = rowCenter;
            _width = width;
            _height = height;
        }
        #endregion

        /// <summary>
        /// Intersection over union
        /// </summary>
        /// <param name="other"></param>
        /// <returns></returns>
        public double IoU(BoundingBox other)
        {
            var intersection = Intersection(other);
            if (intersection < 1e-6)
            {
                return 0.0;
            }
            return intersection / Union(other);
        }
        
        public BoundingBox UpSampling(double rowFactor, double colFactor)
        {
            return new BoundingBox(colFactor*_colCenter, rowFactor*_rowCenter, colFactor*_width, rowFactor*_height);
        }
        public double Intersection(BoundingBox other)
        {
            double maxLeft = Math.Max(Left, other.Left);
            double minRight = Math.Min(Right, other.Right);
            if (maxLeft >= minRight)
            {
                return 0.0;
            }
            double maxTop = Math.Max(Top, other.Top);
            double minBottom = Math.Min(Bottom, other.Bottom);
            if (maxTop >= minBottom)
            {
                return 0.0;
            }
            return (minRight - maxLeft) * (minBottom - maxTop);
        }
        public double Union(BoundingBox other)
        {
            return Area + other.Area - Intersection((other));
        }
        public double Area => _height * _width;
        public double Right => _colCenter+_width/2;
        public double Left => _colCenter-_width/2;
        public double Top => _rowCenter - _height / 2;
        public double Bottom => _rowCenter + _height/ 2;
        public override string ToString()
        {
            return "Center: ("+Math.Round(_rowCenter,3)+", "+ Math.Round(_colCenter, 3)+") Height:"+  Math.Round(_height, 3)+" Width: "+ Math.Round(_width, 3);
        }
        public override bool Equals(object obj)
        {
            if (ReferenceEquals(null, obj))
            {
                return false;
            }
            if (ReferenceEquals(this, obj))
            {
                return true;
            }
            if (obj.GetType() != GetType())
            {
                return false;
            }
            return Equals((BoundingBox)obj, 1e-6);
        }
        public override int GetHashCode()
        {
            return ToString().GetHashCode();
        }
        public bool Equals(BoundingBox other, double epsilon)
        {
            return    Math.Abs(_colCenter - other._colCenter) <= epsilon
                   && Math.Abs(_rowCenter - other._rowCenter) <= epsilon
                   && Math.Abs(_width - other._width) <= epsilon
                   && Math.Abs(_height - other._height) <= epsilon;
        }
    }
}