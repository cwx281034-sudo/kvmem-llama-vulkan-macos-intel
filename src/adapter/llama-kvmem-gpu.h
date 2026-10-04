#pragma once

// KVMem GPU runtime portability shim.
//
// The adapter is written against the CUDA runtime API because that is what the
// upstream KVMem port targets. ROCm builds of this tree reuse the very same
// source: llama.cpp's HIP backend compiles ggml/src/ggml-cuda/*.cu with AMD
// clang (see ggml/src/ggml-hip/CMakeLists.txt) and relies on its own
// CUDA -> HIP name mapping in ggml/src/ggml-cuda/vendors/hip.h.
//
// This header does the same job for the out-of-tree adapter so the adapter
// sources stay single-source for every backend the tree can build. It is the
// only place that is allowed to name a vendor runtime: everything else in
// src/adapter/ speaks the CUDA spelling, and a non-CUDA backend simply supplies
// a compatible facade for that spelling.
//
// Three layers, selected in this order:
//
//   1. GGML_USE_VULKAN            Vulkan  -> llama-kvmem-runtime-vulkan.h
//   2. GGML_USE_HIP | __HIP_PLATFORM_AMD__   HIP -> inline name mapping
//   3. otherwise                 CUDA  -> cuda_runtime.h
//
// Layer 1 must be tested first. A Vulkan build also defines no HIP macro, but a
// future build that combines Vulkan with a hipified CUDA tree would otherwise
// silently take the HIP branch and never reach the Vulkan facade.
//
// Backend selection macros are PUBLIC definitions of the ggml target
// (ggml_add_backend() in llama.cpp/ggml/src/CMakeLists.txt), so every libllama
// translation unit - including this adapter - sees the same branch the ggml
// backend was built with.
//
// Only the subset of the runtime API actually used by the adapter is mapped.
// If new CUDA calls are added to the adapter, add the matching mapping here or
// the HIP/Vulkan build will fail to compile (which is the intended, loud
// failure). KVMEM_GPU_BACKEND_VULKAN is the build-time assertion for that: see
// llama-kvmem-runtime-vulkan.h for the list it must keep satisfied.

#if defined(GGML_USE_VULKAN)
#   define KVMEM_GPU_BACKEND_VULKAN 1
#endif

#if defined(KVMEM_GPU_BACKEND_VULKAN)

#   include "llama-kvmem-runtime-vulkan.h"

#elif defined(GGML_USE_HIP) || defined(__HIP_PLATFORM_AMD__)

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

// The kernels spell bfloat16 the CUDA way; HIP's equivalent is __hip_bfloat16.
// ggml-cuda/vendors/hip.h defines nv_bfloat16 the same way.
typedef __hip_bfloat16 __nv_bfloat16;
typedef __hip_bfloat16 nv_bfloat16;

// --- runtime API ----------------------------------------------------------
#define cudaError_t                hipError_t
#define cudaSuccess                hipSuccess
#define cudaErrorMemoryAllocation  hipErrorMemoryAllocation
#define cudaGetErrorString         hipGetErrorString
#define cudaGetLastError           hipGetLastError

#define cudaMemcpyKind             hipMemcpyKind
#define cudaMemcpyHostToDevice     hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost     hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice   hipMemcpyDeviceToDevice
#define cudaMemcpy                 hipMemcpy
#define cudaMemcpyAsync            hipMemcpyAsync

#define cudaMalloc                 hipMalloc
#define cudaFree                   hipFree
#define cudaMallocHost(ptr, size)  hipHostMalloc(ptr, size, hipHostMallocDefault)
#define cudaFreeHost               hipHostFree

#define cudaMemsetAsync            hipMemsetAsync

#define cudaStream_t               hipStream_t
#define cudaStreamPerThread        hipStreamPerThread
#define cudaStreamCreateWithFlags  hipStreamCreateWithFlags
#define cudaStreamDestroy          hipStreamDestroy
#define cudaStreamSynchronize      hipStreamSynchronize
#define cudaStreamWaitEvent        hipStreamWaitEvent
#define cudaStreamNonBlocking      hipStreamNonBlocking

#define cudaEvent_t                hipEvent_t
#define cudaEventCreateWithFlags   hipEventCreateWithFlags
#define cudaEventDisableTiming     hipEventDisableTiming
#define cudaEventRecord            hipEventRecord
#define cudaEventSynchronize       hipEventSynchronize
#define cudaEventDestroy           hipEventDestroy

#define cudaSetDevice              hipSetDevice
#define cudaGetDevice              hipGetDevice
#define cudaDeviceSynchronize      hipDeviceSynchronize

#define cudaPointerAttributes      hipPointerAttribute_t
#define cudaPointerGetAttributes   hipPointerGetAttributes
#define cudaMemoryTypeDevice       hipMemoryTypeDevice

#else

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#endif  // GGML_USE_VULKAN / GGML_USE_HIP