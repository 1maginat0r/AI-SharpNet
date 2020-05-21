using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using SharpNet.Data;
using SharpNet.GPU;
using SharpNet.Hyperparameters;
using SharpNet.Layers;
using SharpNet.MathTools;
using static SharpNet.GPU.GPUWrapper;

namespace SharpNet.CPU
{
    public unsafe class CpuTensor<T> : Tensor
    {
        #region fields
        public Memory<T> Content { get; private set; }
        /// <summary>
        /// used only if the tensor is NOT the owner of the memory
        /// </summary>
        private readonly IntPtr _ptrToOwnerPinnedMemory;
        /// <summary>
        /// used only if the tensor is the owner of the memory
        /// </summary>
        private HostPinnedMemory<T> _hostPinnedMemory;
        #endregion

        #region constructors
        public CpuTensor(int[] shape, T[] data, int typeSize) : base(shape, typeSize, false)
        {
            Content = data ?? new T[Count];
            CapacityInBytes = (ulong)(Content.Length * TypeSize);
            _ptrToOwnerPinnedMemory = IntPtr.Zero;
        }
        public CpuTensor(int[] shape, T[] data = null) : this(shape, data, typeof(T).IsValueType ?Marshal.SizeOf(typeof(T)) : IntPtr.Size)
        {
        }
        private CpuTensor(int[] shape, CpuTensor<T> m