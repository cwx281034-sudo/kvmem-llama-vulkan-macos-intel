#pragma once

// Vulkan stage-in engine for the KVMem adapter.
//
// Replaces llama-kvmem-stagein.cu (the CUDA __global__ kernels) on the Vulkan
// backend. The public entry points are exactly the ones declared in
// llama-kvmem-stagein.h, so the adapter compiles unchanged on all three
// backends; only which object file provides them differs.
//
// How the CUDA kernels map onto Vulkan
//
//   CUDA                          Vulkan
//   ---------------------------    ------------------------------------------
//   __global__ + <<<blocks,thr>>>  a compute pipeline plus one vkCmdDispatch
//   threadIdx/blockIdx/gridDim    gl_LocalInvocationID / gl_WorkGroupID /
//                                 gl_NumWorkGroups
//   cudaMalloc (device scratch)    a HOST_VISIBLE|HOST_COHERENT VkBuffer, so
//                                 the host can seed and read it directly
//   cudaMemcpy H2D of a slab      memcpy into that mapped buffer, then a
//                                 pipeline barrier
//   shared-memory registers        workgroup shared memory
//
// The one structural difference is the Walsh-Hadamard transform. The CUDA kernel
// gave one thread an entire row and held it in registers, so a row needed no
// shared memory at all. Vulkan rows are up to 512 floats, which does not fit a
// workgroup's shared memory on every device, so kvmem_fwht.comp reduces the row
// in passes instead. The butterfly arithmetic is unchanged.
//
// Pointer argument caveat
//   Every entry point below receives raw pointers that the adapter obtained from
//   kvmem_cuda_tensor_ptr(), i.e. ggml tensor data. On the Vulkan backend those
//   buffers are HOST_VISIBLE|HOST_COHERENT and therefore directly addressable
//   from the host, but they belong to ggml, not to this facade, so the facade
//   has no VkBuffer for them. The shaders here therefore never read or write a
//   ggml buffer: every stage-in result is written into this file's own
//   HOST_VISIBLE scratch, and the adapter copies it into the tensor with the
//   plain memcpy that host coherency makes correct. See kvmem_stagein_flush.
//
// Error policy
//   Each entry point returns false when the request cannot be served, and the
//   adapter already has a host fallback for every one of them (that is what the
//   "Host Hadamard/RoPE is fallback only" note in llama-kvmem-stagein.h means).
//   Nothing here throws and nothing aborts.

#include "llama-kvmem-stagein.h"

#include <cstddef>
#include <cstdint>

// True when the Vulkan stage-in engine initialised and can accept work. The
// adapter calls it to decide whether to take the GPU path or the host fallback.
bool kvmem_vk_stagein_available();

// Number of bytes the engine holds in scratch buffers, for diagnostics.
size_t kvmem_vk_stagein_scratch_bytes();

// Read back the f32 scratch region the dequantize / RoPE / FWHT kernels work in.
// The adapter never needs this: it only ever consumes final quantized bytes. It
// exists so a test can compare a kernel's output against a CPU reference without
// reaching into engine internals. Returns false when n exceeds the scratch size.
bool kvmem_vk_stagein_read_f32(void * host, size_t n);