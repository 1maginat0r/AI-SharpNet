using System.Collections.Generic;
using System;

namespace SharpNet.Datasets
{
    public abstract class AbstractTrainingAndTestDataset : ITrainingAndTestDataset
    {
        #region public properties
        public abstract DataSet Training { get; }
        public abstract DataSet Test { get; }
        protected string Name { get; }
        #endregion

        #region constructor
        protected AbstractTrainingAndTestDataset(string name)
        {
            Name = name;
        }
        #endregion

        public virtual void Dispose()
        {
            Training?.Dispose();
            Test?.Dispose();
        }

        protected virtual int CategoryByteToCategoryIndex(byte categoryByte)
        {
            return categoryByte;
        }

        public AbstractTrainingAndTestDataset WithRandomizeColumnDataSet(List<string> columnsToRandomize, Random r)
        {
            if (columnsToRandomize.Count == 0)
            {
      