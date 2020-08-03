using System.Runtime.InteropServices;
using System.Security;
// ReSharper disable UnusedMember.Global

namespace SharpNet.Data
{
    public static unsafe class BlasServices
    {
        #region Matrix multiplication
        public static void DotMkl(float* A, int aH, int aW, bool transposeA, float* B, int bH, int