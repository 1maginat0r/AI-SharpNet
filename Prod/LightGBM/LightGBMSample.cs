// ReSharper disable UnusedMember.Global
// ReSharper disable InconsistentNaming
// ReSharper disable IdentifierTypo
// ReSharper disable CommentTypo

using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using SharpNet.Datasets;
using SharpNet.HPO;
using SharpNet.Hyperparameters;
using SharpNet.Models;
using System.Linq;

namespace SharpNet.LightGBM;

[SuppressMessage("ReSharper", "MemberCanBePrivate.Global")]
public class LightGBMSample : AbstractModelSample
{
    public LightGBMSample() :base(_categoricalHyperparameters)
    {
    }
    public override EvaluationMetricEnum GetLoss()
    {
        switch (objective)
        {
            case objective_enum.DEFAULT_VALUE:
                return EvaluationMetricEnum.DEFAULT_VALUE;
            case objective_enum.regression:
                return EvaluationMetricEnum