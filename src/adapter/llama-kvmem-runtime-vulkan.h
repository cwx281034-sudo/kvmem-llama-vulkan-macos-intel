#pragma once

// Vulkan facade for the KVMem adapter runtime shim.
//
// Selected by llama-kvmem-gpu.h when GGML_USE_VULKAN is defined. It gives the
// adapter the exact CUDA spelling it already uses (cudaMalloc, cudaMemcpyAsync,
// cudaStream_t, ...) on top of plain Vulkan, so src/adapter/*.cpp stays
// single-source across CUDA, HIP and Vulkan.
//
// What this header is
//   * Type and constant definitions only. Every entry point below is
//     declared, not defined, so the adapter links against one implementation
//     (llama-kvmem-runtime-vulkan.cpp) and the same device state is shared by
//     every translation unit instead of being rebuilt per file.
//
// What this header is NOT
//   * It is not a Vulkan compute engine. It carries no shader, no pipeline and
//     no descriptor management. The transfer facade below only needs queue
//     submission and buffer copies; kernels are the separate stage-in
//     translation unit's job.
//
// Design notes that the rest of the adapter depends on
//   * cudaStream_t is an opaque queue handle, and cudaStreamPerThread is that
//     handle for the calling thread. The adapter stores stream handles in its
//     staging structs and hands them to ggml backend events, so the handle has
//     to be cheap to copy and stable for the process.
//   * The transfer-kind values keep CUDA's numbering (1, 2, 3):
//     kvmem_record_transfer() in llama-memory-kvmem.cpp indexes
//     transfer_bytes[kind - 1] and rejects anything outside 1..3, so the
//     facade must not renumber them or the KVMEM_PERF counters silently
//     corrupt (or discard) adapter traffic.

#include <cstddef>
#include <cstdint>

// The facade's own implementation file defines this to include the header
// directly; every other translation unit must go through llama-kvmem-gpu.h so
// the backend branch stays the single place that selects a runtime.
#if !defined(KVMEM_GPU_BACKEND_VULKAN) && !defined(KVMEM_VK_FACADE_IMPL)
#   error "include llama-kvmem-gpu.h instead of reaching for this header directly"
#endif

// --- error type -----------------------------------------------------------
// CUDA-compatible numbering for the subset the adapter tests. The adapter only
// compares against kvmem_cuda_success and kvmem_cuda_error_memory_allocation,
// so any distinct nonzero value works; these mirror CUDA's own layout to keep
// the facade drop-in.
typedef int kvmem_vk_error;

enum {
    kvmem_vk_success                = 0,
    kvmem_vk_error_invalid_value    = 1,
    kvmem_vk_error_out_of_memory    = 2,
    kvmem_vk_error_not_initialized  = 3,
    kvmem_vk_error_invalid_device   = 10,
    kvmem_vk_error_unknown          = 999,
};

// --- CUDA spelling --------------------------------------------------------
#define cudaError_t                kvmem_vk_error
#define cudaSuccess                kvmem_vk_success
#define cudaErrorMemoryAllocation  kvmem_vk_error_out_of_memory
#define cudaErrorInvalidValue      kvmem_vk_error_invalid_value

// cudaGetErrorString / cudaGetLastError are defined in the implementation file.
cudaError_t cudaGetLastError(void);
const char * cudaGetErrorString(cudaError_t e);

// --- streams --------------------------------------------------------------
// Opaque, one handle per (device, queue). The adapter only ever stores,
// passes and destroys them; it never inspects a stream.
struct kvmem_vk_stream;
typedef struct kvmem_vk_stream * cudaStream_t;

// The per-thread default queue, matching CUDA's cudaStreamPerThread. There is
// one implicit queue per device here: every submission an adapter thread makes
// is already ordered against the ggml backend's work on the same device by the
// fences the facade installs around cross-graph hand-offs, so the distinction
// between "this thread's stream" and "the device's queue" is not observable to
// the adapter. Keeping it a real (non-null) handle matters: the adapter uses
// `stream == nullptr` to mean "no stream yet" (see MultiD2hPipe::Device) and
// treats nullptr as a reason to create one.
extern cudaStream_t cudaStreamPerThread;

cudaError_t cudaStreamCreateWithFlags(cudaStream_t * stream, unsigned int flags);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);

// --- events ---------------------------------------------------------------
struct kvmem_vk_event;
typedef struct kvmem_vk_event * cudaEvent_t;

cudaError_t cudaEventCreateWithFlags(cudaEvent_t * event, unsigned int flags);
cudaError_t cudaEventDestroy(cudaEvent_t event);
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
cudaError_t cudaEventSynchronize(cudaEvent_t event);

// cudaStreamWaitEvent needs both handle types, so it is declared here with the
// event flags it forwards.
cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags);

// --- memcpy kinds ---------------------------------------------------------
// Values are CUDA's: HostToDevice 1, DeviceToHost 2, DeviceToDevice 3.
// kvmem_record_transfer() depends on this exact 1..3 range.
typedef int cudaMemcpyKind;

enum {
    cudaMemcpyHostToDevice   = 1,
    cudaMemcpyDeviceToHost   = 2,
    cudaMemcpyDeviceToDevice = 3,
};

cudaError_t cudaMemcpy(void * dst, const void * src, size_t n, cudaMemcpyKind kind);
cudaError_t cudaMemcpyAsync(void * dst, const void * src, size_t n,
                            cudaMemcpyKind kind, cudaStream_t stream);

// --- device memory --------------------------------------------------------
cudaError_t cudaMalloc(void ** ptr, size_t n);
cudaError_t cudaFree(void * ptr);
cudaError_t cudaMallocHost(void ** ptr, size_t n);
cudaError_t cudaFreeHost(void * ptr);
cudaError_t cudaMemsetAsync(void * dst, int value, size_t n, cudaStream_t stream);

enum {
    cudaStreamDefault      = 0,
    cudaStreamNonBlocking  = 1,
};

enum {
    cudaEventDefault       = 0,
    cudaEventDisableTiming = 2,
};

// --- device selection -----------------------------------------------------
cudaError_t cudaSetDevice(int device);
cudaError_t cudaGetDevice(int * device);
cudaError_t cudaDeviceSynchronize(void);

// --- pointer attributes ---------------------------------------------------
// The adapter uses this to answer "is this ggml tensor pointer a real device
// allocation, and which device owns it" (kvmem_cuda_tensor_ptr callers and the
// multi-GPU D2H pipe). Vulkan has no equivalent query, so the facade tracks
// allocations it made itself in a registry and reports unregistered host
// pointers as cudaMemoryTypeUnregistered - which is what CUDA itself reports for
// malloc'd host memory, and what the adapter already treats as "not a device
// pointer".
enum {
    cudaMemoryTypeUnregistered = 0,
    cudaMemoryTypeHost         = 1,
    cudaMemoryTypeDevice       = 2,
};

struct cudaPointerAttributes {
    int  type;    // one of the cudaMemoryType* values above
    int  device;  // owning device ordinal; -1 when not a device allocation
    void * devicePointer;
    void * hostPointer;
};

cudaError_t cudaPointerGetAttributes(struct cudaPointerAttributes * attrs, const void * ptr);

// Can this pointer be dereferenced by host code with plain memcpy/memmove?
//
// This is NOT the same question as "did this facade allocate it". The adapter
// routinely hands ggml-owned tensor pointers to the memcpy facade, and ggml's
// Vulkan backend allocates those DEVICE_LOCAL when the GPU is discrete
// (ggml_vk_create_buffer_device: the eDeviceLocal|HostVisible request falls
// back to plain eDeviceLocal unless GGML_VK_PREFER_HOST_MEMORY or
// GGML_VK_ALLOW_SYSMEM_FALLBACK is set). Such a pointer is neither in this
// facade's registry nor host addressable, and dereferencing it faults.
//
// Callers use this to reject an unreadable device pointer before a copy, so the
// portable fallback (ggml_backend_tensor_get / _set) is taken instead of
// segfaulting. CUDA needs no such test - every device pointer there is
// unmapped host memory - so the CUDA/HIP branches do not declare it.
bool kvmem_vk_ptr_is_host_readable(const void * ptr);
bool kvmem_vk_ptr_is_host_writable(const void * ptr);

// --- floating point --------------------------------------------------------
// The CUDA/HIP branches pull in the vendor fp16/bf16 headers so device code
// can use __half / __nv_bfloat16. The adapter's host translation units only
// need the scalar conversions (llama-kvmem-stagein.cu uses them inside kernels,
// which is a CUDA/HIP-only file), so the facade declares the small, portable
// subset those conversions need rather than depending on <cuda_fp16.h>.
typedef uint16_t half;

inline float __half2float(half h) {
    const uint32_t sign = (uint32_t) (h >> 15) & 0x1u;
    const uint32_t exp  = (uint32_t) (h >> 10) & 0x1fu;
    const uint32_t mant = (uint32_t) h & 0x3ffu;

    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign << 31;  // +-0
        } else {
            // Subnormal: normalize it into an fp32 normal.
            uint32_t e = 0;
            uint32_t m = mant;
            while ((m & 0x400u) == 0) {
                m <<= 1;
                e += 1;
            }
            m &= 0x3ffu;
            bits = (sign << 31) | ((127 - 15 - e + 1) << 23) | (m << 13);
        }
    } else if (exp == 0x1f) {
        bits = (sign << 31) | (0xffu << 23) | (mant << 13);  // inf / nan
    } else {
        bits = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    }

    float out;
    static_assert(sizeof(out) == sizeof(bits), "fp32 assumed");
    __builtin_memcpy(&out, &bits, sizeof(out));
    return out;
}

inline half __float2half(float f) {
    uint32_t bits;
    __builtin_memcpy(&bits, &f, sizeof(bits));

    const uint32_t sign = (bits >> 31) & 0x1u;
    const int32_t  exp  = (int32_t) ((bits >> 23) & 0xffu) - 127;
    const uint32_t mant = bits & 0x7fffffu;

    if (((bits >> 23) & 0xffu) == 0xffu) {
        return (half) ((sign << 15) | 0x7c00u | (mant ? 0x200u : 0u));
    }
    if (exp > 15) {
        return (half) ((sign << 15) | 0x7c00u);  // overflow -> inf
    }
    if (exp < -14) {
        // Flush tiny magnitudes to zero; the adapter only scales block
        // scales, which are always well inside the normal range.
        return (half) (sign << 15);
    }

    const uint32_t h_exp  = (uint32_t) (exp + 15);
    // Round to nearest, ties to even.
    const uint32_t h_mant = (mant + 0xfffu + ((mant >> 13) & 1u)) >> 13;
    return (half) ((sign << 15) | (h_exp << 10) | (h_mant & 0x3ffu));
}

inline float __bfloat162float(uint16_t h) {
    const uint32_t bits = (uint32_t) h << 16;
    float out;
    __builtin_memcpy(&out, &bits, sizeof(out));
    return out;
}

// The CUDA/HIP branches alias the two bfloat16 spellings onto one vendor type.
// The adapter's host code only reads them through __bfloat162float, so the
// facade keeps them as distinct 16-bit storage types and relies on the implicit
// conversion for the two spellings to resolve to the overload above.
typedef uint16_t __nv_bfloat16;
typedef uint16_t nv_bfloat16;