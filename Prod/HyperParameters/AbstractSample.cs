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
    public virtual void Set(IDictionary<string, object> dico)
    {
        foreach (var (key, value) in dico)
        {
            Set(key, value);
        }
    }
    //TODO add tests
    /// <summary>
    /// to determine if an Hyper-Parameter is categorical, we'll use the following rules:
    ///     1/ if the Hyper-Parameter name is in '_mandatoryCategoricalHyperparameters'
    ///             => it is categorical
    ///     2/ if the Hyper-Parameter is a int/float/double
    ///             => it is numerical
    ///     3/ if the Hyper-Parameter is a bool/string/enum
    ///             => it is categorical
    ///     4/ in all other cases
    ///             => an exception is thrown (can't determine if the Hyper-Parameter is categorical)
    /// </summary>
    /// <param name="HyperparameterName"></param>
    /// <returns></returns>
    /// <exception cref="ArgumentException"></exception>
    public virtual bool IsCategoricalHyperparameter(string HyperparameterName)
    {
        if (_mandatoryCategoricalHyperparameters.Contains(HyperparameterName))
        {
            return true;
        }
        var HyperparameterType = GetFieldType(HyperparameterName);
        if (HyperparameterType == typeof(double) || HyperparameterType == typeof(float) || HyperparameterType == typeof(int))
        {
            return false;
        }
        if (HyperparameterType == typeof(string) || HyperparameterType == typeof(bool) || HyperparameterType.IsEnum || HyperparameterType.IsClass)
        {
            return true;       
        }
        throw new ArgumentException($"can't determine if {HyperparameterName} ({HyperparameterType}) field of class {GetType()} is categorical");
    }
    #endregion

    public virtual HashSet<string> FieldsToDiscardInComputeHash()
    {
        return new HashSet<string>();
    }

    protected virtual string ToConfigContent(Func<string, object, bool> accept)
    {
        var result = new List<string>();
        foreach (var (parameterName,fieldValue) in ToDictionaryConfigContent(accept).OrderBy(f => f.Key))
        {
            var fieldValueToString = Utils.FieldValueToString(fieldValue);
            result.Add($"{parameterName} = {fieldValueToString}");
        }
        return string.Join(Environment.NewLine, result) + Environment.NewLine;
    }
    protected IDictionary<string,object> ToDictionaryConfigContent(Func<string, object, bool> accept)
    {
        var result = n