
﻿using System;
using System.Runtime.InteropServices;
using SharpNet.Data;
// ReSharper disable UnusedMember.Global
// ReSharper disable IdentifierTypo

namespace SharpNet.GPU
{
    public enum CUresult
    {
        CUDA_SUCCESS = 0,
        CUDA_ERROR_INVALID_VALUE = 1,
        CUDA_ERROR_OUT_OF_MEMORY = 2,
        CUDA_ERROR_NOT_INITIALIZED = 3,
        CUDA_ERROR_DEINITIALIZED = 4,
        CUDA_ERROR_PROFILER_DISABLED = 5,
        CUDA_ERROR_PROFILER_NOT_INITIALIZED = 6,
        CUDA_ERROR_PROFILER_ALREADY_STARTED = 7,
        CUDA_ERROR_PROFILER_ALREADY_STOPPED = 8,
        CUDA_ERROR_NO_DEVICE = 100,
        CUDA_ERROR_INVALID_DEVICE = 101,
        CUDA_ERROR_INVALID_IMAGE = 200,
        CUDA_ERROR_INVALID_CONTEXT = 201,
        CUDA_ERROR_CONTEXT_ALREADY_CURRENT = 202,
        CUDA_ERROR_MAP_FAILED = 205,
        CUDA_ERROR_UNMAP_FAILED = 206,
        CUDA_ERROR_ARRAY_IS_MAPPED = 207,
        CUDA_ERROR_ALREADY_MAPPED = 208,
        CUDA_ERROR_NO_BINARY_FOR_GPU = 209,
        CUDA_ERROR_ALREADY_ACQUIRED = 210,
        CUDA_ERROR_NOT_MAPPED = 211,
        CUDA_ERROR_NOT_MAPPED_AS_ARRAY = 212,
        CUDA_ERROR_NOT_MAPPED_AS_POINTER = 213,
        CUDA_ERROR_ECC_UNCORRECTABLE = 214,
        CUDA_ERROR_UNSUPPORTED_LIMIT = 215,
        CUDA_ERROR_CONTEXT_ALREADY_IN_USE = 216,
        CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = 217,
        CUDA_ERROR_INVALID_PTX = 218,
        CUDA_ERROR_INVALID_GRAPHICS_CONTEXT = 219,
        CUDA_ERROR_NVLINK_UNCORRECTABLE = 220,
        CUDA_ERROR_JIT_COMPILER_NOT_FOUND = 221,
        CUDA_ERROR_INVALID_SOURCE = 300,
        CUDA_ERROR_FILE_NOT_FOUND = 301,
        CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302,
        CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = 303,
        CUDA_ERROR_OPERATING_SYSTEM = 304,
        CUDA_ERROR_INVALID_HANDLE = 400,
        CUDA_ERROR_ILLEGAL_STATE = 401,
        CUDA_ERROR_NOT_FOUND = 500,
        CUDA_ERROR_NOT_READY = 600,
        CUDA_ERROR_ILLEGAL_ADDRESS = 700,
        CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701,
        CUDA_ERROR_LAUNCH_TIMEOUT = 702,
        CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703,
        CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704,
        CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = 705,
        CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = 708,
        CUDA_ERROR_CONTEXT_IS_DESTROYED = 709,
        CUDA_ERROR_ASSERT = 710,
        CUDA_ERROR_TOO_MANY_PEERS = 711,
        CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712,
        CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = 713,
        CUDA_ERROR_HARDWARE_STACK_ERROR = 714,
        CUDA_ERROR_ILLEGAL_INSTRUCTION = 715,
        CUDA_ERROR_MISALIGNED_ADDRESS = 716,
        CUDA_ERROR_INVALID_ADDRESS_SPACE = 717,
        CUDA_ERROR_INVALID_PC = 718,
        CUDA_ERROR_LAUNCH_FAILED = 719,
        CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE = 720,
        CUDA_ERROR_NOT_PERMITTED = 800,
        CUDA_ERROR_NOT_SUPPORTED = 801,
        CUDA_ERROR_UNKNOWN = 999
    }

    public enum CUdevice_attribute
    {
        CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
        CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,
        CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,
        CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,
        CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,
        CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,
        CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,
        CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
        CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK = 8,
        CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,
        CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,
        CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,
        CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,
        CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK = 12,
        CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,
        CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,
        CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,
        CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
        CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17,
        CU_DEVICE_ATTRIBUTE_INTEGRATED = 18,
        CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19,
        CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = 21,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = 22,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = 23,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = 24,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = 25,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = 26,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH = 27,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT = 28,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES = 29,
        CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 30,
        CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31,
        CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32,
        CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33,
        CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34,
        CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35,
        CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36,
        CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37,
        CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38,
        CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39,
        CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40,
        CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43,
        CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER = 44,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49,
        CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50,
        CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 51,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = 52,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = 55,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = 56,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = 57,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = 58,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = 59,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = 60,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = 66,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74,
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77,
        CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78,
        CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79,
        CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80,
        CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,
        CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82,
        CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83,
        CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84,
        CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85,
        CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 86,
        CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87,
        CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88,
        CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89,
        CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 90,
        CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91,
        CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS = 92,
        CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS = 93,
        CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR = 94,
        CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95,
        CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH = 96,
        CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,
        CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES = 98,
        CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED = 99,
        CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100,
        CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = 101,
        CU_DEVICE_ATTRIBUTE_MAX
    }

    public static class NVCudaWrapper
    {
        private const string DLL_NAME = "nvcuda";

        [DllImport(DLL_NAME)]
        public static extern CUresult cuMemAllocHost_v2(
            ref IntPtr pp,
            size_t bytesize);

        [DllImport(DLL_NAME)]
        public static extern CUresult cuMemFreeHost(
            IntPtr p);

        [DllImport(DLL_NAME)]
        public static extern CUresult cuMemFree_v2(
            IntPtr dptr);

        [DllImport(DLL_NAME)]
        public static extern CUresult cuMemAlloc_v2(
            out IntPtr dptr,
            size_t bytesize);


        [DllImport(DLL_NAME)]
        public static extern CUresult cuMemsetD8_v2(
            IntPtr dptr,
            char c,
            size_t N);

        [DllImport(DLL_NAME)]
        public static extern CUresult cuMemsetD32_v2(
            IntPtr dptr,
            uint ui,
            size_t N);

        [DllImport(DLL_NAME)]
        public static extern CUresult cuMemGetAddressRange_v2(
            out IntPtr pbase,
            out size_t psize,
            IntPtr dptr);

        [DllImport(DLL_NAME)]
        public static extern CUresult cuMemcpyHtoD_v2(
            IntPtr dstDevice, 
            [In] IntPtr srcHost,
            size_t byteCount);

        [DllImport(DLL_NAME)]
        public static extern CUresult cuMemcpyHtoDAsync_v2(
            IntPtr dstDevice,
            [In] IntPtr srcHost,
            size_t byteCount,
            IntPtr hStream);

        [DllImport(DLL_NAME)]
        public static extern CUresult cuMemcpyDtoD_v2(
            IntPtr dstDevice,
            [In] IntPtr srcDevice,
            size_t byteCount);

        [DllImport(DLL_NAME)]
        public static extern CUresult cuMemcpyDtoDAsync_v2(
            IntPtr dstDevice,
            [In] IntPtr srcDevice,
            size_t byteCount,
            IntPtr hStream);

        [DllImport(DLL_NAME)]
        public static extern CUresult cuMemcpyDtoH_v2(
            IntPtr dstHost,
            IntPtr srcDevice,
            size_t byteCount);

        [DllImport(DLL_NAME)]
        public static extern CUresult cuStreamCreate(
            out IntPtr streamHandle, 
            uint flags
        );

        [DllImport(DLL_NAME)]
        public static extern CUresult cuStreamSynchronize(
            IntPtr streamHandle
        );

        [DllImport(DLL_NAME)]
        public static extern CUresult cuStreamDestroy_v2(
            IntPtr streamHandle
        );

        [DllImport(DLL_NAME)]
        public static extern CUresult cuLaunchKernel(
            IntPtr kernelHandle, 
            uint gridDimX,
            uint gridDimY, 
            uint gridDimZ, 
            uint blockDimX, 
            uint blockDimY, 
            uint blockDimZ, 
            uint sharedMemBytes,  //this is really uint, not size_t
            IntPtr streamHandle,
            IntPtr[] kernelParams,
            IntPtr[] extra);

        [DllImport(DLL_NAME)]
        public static extern CUresult cuDeviceGet(out IntPtr deviceHandle, int ordinal);

        [DllImport(DLL_NAME)]
        public static extern CUresult cuDeviceGetCount(out int count);

        [DllImport(DLL_NAME)]
        public static extern CUresult cuInit(uint flags);

        [DllImport(DLL_NAME)]
        public static extern CUresult cuDevicePrimaryCtxRetain(out IntPtr contextHandle, IntPtr deviceHandle);
        
        [DllImport(DLL_NAME)]
        public static extern CUresult cuDevicePrimaryCtxRelease(IntPtr contextHandle);
        
        [DllImport(DLL_NAME)]
        public static extern CUresult cuDeviceGetAttribute(out int pi, CUdevice_attribute attrib, IntPtr deviceHandle);

        [DllImport(DLL_NAME)]
        public static extern CUresult cuDeviceGetName(
            byte[] name, 
            int len,
            IntPtr deviceHandle);

        [DllImport(DLL_NAME)]
        public static extern CUresult cuModuleLoadFatBinary(out IntPtr moduleHandle, byte[] fatBinaryObject);

        [DllImport(DLL_NAME)]
        public static extern CUresult cuDriverGetVersion(out int driverVersion);

        [DllImport(DLL_NAME)]
        public static extern CUresult cuDeviceTotalMem_v2(out size_t bytes, IntPtr deviceHandle);

        [DllImport(DLL_NAME)]
        public static extern CUresult cuMemGetInfo_v2(out size_t free, out size_t total);

        [DllImport(DLL_NAME)]
        public static extern CUresult cuModuleGetFunction(
            out IntPtr functionHandle, 
            IntPtr moduleHandle, 
            string name );

    }
}