using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace SharpNet.Hyperparameters;

public abstract class AbstractSample : ISample
{
    #region private fields
    private readonly HashSet<string> _mandatoryCategoricalHyperparameters;
    #endregion


    public virtual string ToPath(string workingDirectory, string sampleName)
    {
        return Path.Combine(workingDirectory, sampleName + "."+GetType().Name+".conf");
    }
    public const int DEFAULT_VALUE = -6666;

    #region constructors
    protected AbstractSample(HashSet<string> mandatoryCategoricalHyperparameters = null)
    {
        _mandatoryCategoricalHyperparameters = mandatoryCategoricalHyperparameters ?? new HashSet<string>();
    }
    #endregion

    #region ISample methods
    public HashSet<string> HyperparameterNames()
    {
        var type = GetType();
        return new HashSet<string>(ClassFieldSetter.FieldNames(type));
    }
    public string ComputeHash()
    {
        var fieldsToDiscardInComputeHash = FieldsToDiscardInComputeHash();
        bool Accept(string fieldName, object fieldValue)
        {
            return DefaultAcceptForConfigContent(fieldName, fieldValue) && !fieldsToDiscardInComputeHash.Contains(fieldName);
        }
        return Utils.ComputeHash(ToConfigContent(Accept), 10);
    }
    public virtual ISample Clone()
    {
        var clonedInstance = (ISample)Activator.CreateInstance(GetType(), true);
        clonedInstance?.Set(ToDictionaryConfigContent(DefaultAcceptForConfigContent));
        return clonedInstance;
    }
    public virtual List<string> Save(string workingDirectory, string modelName)
    {
        var path = ToPath(workingDirectory, modelName);
        Save(path);
        return new List<string> { path };
    }
    public virtual bool MustUseGPU => false;

    public void Save(string path)
    {
        var content = ToConfigContent(DefaultAcceptForConfigContent);
        File.WriteAllText(path, content);
    }
    public virtual List<string> SampleFiles(string workingDirectory, string modelName)
    {
        return new List<string> { ToPath(workingDirectory, modelName) };
    }
    public virtual void Set(string fieldName, object fieldValue)
    {
        ClassFieldSetter.Set(this, fieldName, fieldValue);
    }
    public virtual object Get(string fieldName)
    {
        return ClassFieldSetter.Get(this, fieldName);
    }
    public virtual bool FixErrors()
    {
        return true;
    }
    public Type GetFieldType(string HyperparameterName)
    {
        return ClassFieldSetter.GetFieldType(GetType(), HyperparameterName);
    }
    public virtual void Set(IDictionary<