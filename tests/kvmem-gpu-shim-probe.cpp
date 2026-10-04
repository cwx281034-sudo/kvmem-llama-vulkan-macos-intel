// Self-contained proof that the KVMem runtime shim is backend-selectable and
// does not pull a vendor header on the Vulkan path.
//
// This TU includes ONLY llama-kvmem-gpu.h. It touches every CUDA spelling the
// adapter uses, so a missing mapping is a compile error. It deliberately
// includes no llama.cpp header, so it stays compilable while the submodule
// patch baseline (t1) is still being restored.
//
// Every entry point is checked in an unevaluated context (static_assert on
// decltype), which type-checks the declaration without emitting a reference to
// it. That keeps the probe linkable before a backend's runtime implementation
// exists, which is what lets it run on every backend including one whose
// runtime is still being written.

#include "llama-kvmem-gpu.h"

#include <cstddef>
#include <cstdio>
#include <type_traits>

#if defined(KVMEM_GPU_BACKEND_VULKAN)
#   define KVMEM_PROBE_BACKEND "VULKAN"
#elif defined(GGML_USE_HIP) || defined(__HIP_PLATFORM_AMD__)
#   define KVMEM_PROBE_BACKEND "HIP"
#else
#   define KVMEM_PROBE_BACKEND "CUDA"
#endif

// Signatures of llama-memory-kvmem.cpp / llama-memory-kvmem-mtp.cpp.
#define KVMEM_CHECK_SIGNATURE(fn, signature) \
    static_assert(std::is_same<decltype(&fn), signature>::value, \
        "KVMem adapter calls " #fn " with a signature the shim does not provide")

KVMEM_CHECK_SIGNATURE(cudaMalloc,            cudaError_t (*)(void **, size_t));
KVMEM_CHECK_SIGNATURE(cudaFree,              cudaError_t (*)(void *));
KVMEM_CHECK_SIGNATURE(cudaFreeHost,          cudaError_t (*)(void *));
KVMEM_CHECK_SIGNATURE(cudaMemsetAsync,       cudaError_t (*)(void *, int, size_t, cudaStream_t));
KVMEM_CHECK_SIGNATURE(cudaMemcpy,            cudaError_t (*)(void *, const void *, size_t, cudaMemcpyKind));
KVMEM_CHECK_SIGNATURE(cudaMemcpyAsync,       cudaError_t (*)(void *, const void *, size_t, cudaMemcpyKind, cudaStream_t));
KVMEM_CHECK_SIGNATURE(cudaStreamCreateWithFlags, cudaError_t (*)(cudaStream_t *, unsigned int));
KVMEM_CHECK_SIGNATURE(cudaStreamDestroy,     cudaError_t (*)(cudaStream_t));
KVMEM_CHECK_SIGNATURE(cudaStreamSynchronize, cudaError_t (*)(cudaStream_t));
KVMEM_CHECK_SIGNATURE(cudaStreamWaitEvent,   cudaError_t (*)(cudaStream_t, cudaEvent_t, unsigned int));
KVMEM_CHECK_SIGNATURE(cudaEventCreateWithFlags, cudaError_t (*)(cudaEvent_t *, unsigned int));
KVMEM_CHECK_SIGNATURE(cudaEventDestroy,      cudaError_t (*)(cudaEvent_t));
KVMEM_CHECK_SIGNATURE(cudaEventRecord,       cudaError_t (*)(cudaEvent_t, cudaStream_t));
KVMEM_CHECK_SIGNATURE(cudaEventSynchronize,  cudaError_t (*)(cudaEvent_t));
KVMEM_CHECK_SIGNATURE(cudaSetDevice,         cudaError_t (*)(int));
KVMEM_CHECK_SIGNATURE(cudaGetDevice,         cudaError_t (*)(int *));
KVMEM_CHECK_SIGNATURE(cudaDeviceSynchronize, cudaError_t (*)(void));
KVMEM_CHECK_SIGNATURE(cudaPointerGetAttributes, cudaError_t (*)(cudaPointerAttributes *, const void *));
KVMEM_CHECK_SIGNATURE(cudaGetLastError,      cudaError_t (*)(void));
KVMEM_CHECK_SIGNATURE(cudaGetErrorString,    const char * (*)(cudaError_t));

// cudaMallocHost is the one entry the shim maps as a function-like macro, because
// HIP's hostMalloc takes a flags argument CUDA's does not:
//   #define cudaMallocHost(ptr, size) hipHostMalloc(ptr, size, hipHostMallocDefault)
// so `&cudaMallocHost` does not name anything on that backend. Check it through
// a two-argument call instead, which is exactly how the adapter calls it
// (llama-memory-kvmem.cpp: kvmem_cuda_ok(cudaMallocHost(&dev.pin, dev.used), ...)).
// That spelling expands correctly on both CUDA and HIP.
static_assert(std::is_same<decltype(cudaMallocHost((void **) nullptr, (size_t) 0)),
                           cudaError_t>::value,
    "cudaMallocHost(ptr, size) must accept the adapter's two-argument call");

int main() {
    // kvmem_record_transfer() rejects kind outside 1..3 and indexes
    // transfer_bytes[kind - 1], so the numbering has to survive a backend
    // switch byte-for-byte.
    static_assert(cudaMemcpyHostToDevice   == 1, "transfer kind renumbered");
    static_assert(cudaMemcpyDeviceToHost   == 2, "transfer kind renumbered");
    static_assert(cudaMemcpyDeviceToDevice == 3, "transfer kind renumbered");
    static_assert(cudaSuccess == 0, "success must be falsy");
    static_assert(cudaErrorMemoryAllocation != 0, "OOM must be a distinct error");

    // The adapter uses `stream == nullptr` as "not created yet" (MultiD2hPipe)
    // and cudaStream_t must be a pointer so that comparison compiles.
    static_assert(std::is_pointer<cudaStream_t>::value, "stream handle must be a pointer");
    static_assert(std::is_pointer<cudaEvent_t>::value,  "event handle must be a pointer");

    // cudaStreamPerThread must be a defined object of that pointer type. Checked
    // with decltype only: reading it here would force the probe to link against
    // the backend runtime, which is exactly what this probe must not need.
    static_assert(std::is_same<decltype(cudaStreamPerThread), cudaStream_t>::value,
        "cudaStreamPerThread must be a cudaStream_t object, not a call or alias");

    cudaStream_t stream = nullptr;
    cudaEvent_t  event  = nullptr;
    cudaError_t  err    = cudaSuccess;
    cudaPointerAttributes attrs{};
    attrs.type   = cudaMemoryTypeDevice;
    attrs.device = 0;

    // fp16/bf16 conversions are used by the adapter's mean-K accounting, which
    // reads block scales straight out of packed device memory
    // (*reinterpret_cast<const half *>(k + off)). Exercise that exact spelling,
    // over known bit patterns, because it is the only construction that is
    // portable: on CUDA/HIP these are vendor classes, so a literal cast does not
    // compile there, and on Vulkan they are 16-bit integers, where casting a
    // float would not produce the right bit pattern. The bit patterns are what
    // the quantised blocks actually contain: fp16/bf16 1.0.
    static_assert(sizeof(half) == 2, "half must be a 2-byte storage type");
    static_assert(sizeof(__nv_bfloat16) == 2, "bfloat16 must be a 2-byte storage type");
    const uint16_t h_bits  = 0x3c00;  // fp16 1.0
    const uint16_t bf_bits = 0x3f80;  // bf16 1.0
    const half *     h_ptr = reinterpret_cast<const half *>(&h_bits);
    const __nv_bfloat16 * b_ptr = reinterpret_cast<const __nv_bfloat16 *>(&bf_bits);
    if (__half2float(*h_ptr) != 1.0f || __bfloat162float(*b_ptr) != 1.0f) {
        fprintf(stderr, "KVMEM_PROBE FAIL backend=%s: fp16/bf16 conversion wrong\n",
                KVMEM_PROBE_BACKEND);
        return 1;
    }
    // nv_bfloat16 is the second spelling the stage-in kernel uses; it must be
    // the same storage type so both paths convert identically.
    const nv_bfloat16 * b2_ptr = reinterpret_cast<const nv_bfloat16 *>(&bf_bits);
    if (__bfloat162float(*b2_ptr) != 1.0f) {
        fprintf(stderr, "KVMEM_PROBE FAIL backend=%s: nv_bfloat16 conversion wrong\n",
                KVMEM_PROBE_BACKEND);
        return 1;
    }

    fprintf(stderr, "KVMEM_PROBE PASS backend=%s\n", KVMEM_PROBE_BACKEND);
    (void) stream; (void) event; (void) err; (void) attrs;
    return 0;
}