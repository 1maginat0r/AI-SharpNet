using System;
using System.Collections.Generic;

namespace SharpNet.Networks
{
    public class MobileBlocksDescription
    {
        #region public properties
        public int KernelSize { get; }
        public int NumRepeat { get; }
        public int OutputFilters { get; }
        public int ExpandRat