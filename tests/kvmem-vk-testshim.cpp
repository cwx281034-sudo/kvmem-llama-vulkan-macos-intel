// Stubs for the two adapter symbols the Vulkan stage-in test needs.
//
// Both live in src/adapter/llama-memory-kvmem.cpp, which cannot be compiled
// until the llama.cpp patch baseline (t1) is restored. The stage-in engine only
// calls them for KVMEM_PERF accounting and timestamp reads, so a test build can
// supply its own definitions and stay independent of that baseline.
//
// This file is deliberately a test-only target (kvmem-vk-testshim). It is never
// linked into libllama; the real definitions win there.

#include <chrono>
#include <cstdint>

// KVMEM_PERF accounting: the test does not assert on the counters.
void kvmem_record_transfer(int /*cuda_kind*/, uint64_t /*bytes*/) {}

// ggml.h declares this; only the adapter's timing paths call it.
extern "C" int64_t ggml_time_us(void) {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}