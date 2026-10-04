// Differential test: the Vulkan stage-in kernels against a CPU reference.
//
// The CPU reference is transcribed from the CUDA kernels in
// llama-kvmem-stagein.cu, so this compares the Vulkan port against the code it
// replaces rather than against itself.
//
// It needs only ggml.h (for the enum) and the stage-in entry points, so it runs
// before the llama.cpp patch baseline (t1) is restored.

#include "llama-kvmem-stagein-vulkan.h"
#include "llama-kvmem-gpu.h"

#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

int g_fail = 0;

void check(bool ok, const std::string & what) {
    printf("%-52s %s\n", what.c_str(), ok ? "PASS" : "FAIL");
    if (!ok) {
        ++g_fail;
    }
}

// --- CPU reference, transcribed from llama-kvmem-stagein.cu ------------------

constexpr int QK8_0 = 32;
constexpr int QK4_0 = 32;

struct block_q8_0 { uint16_t d; int8_t  qs[QK8_0]; };
struct block_q4_0 { uint16_t d; uint8_t qs[QK4_0 / 2]; };

// Total bytes of q8_0 payload for n elements: 2 byte scale + 32 int8 per block.
size_t ntotal_bytes_q8_0(int64_t n) { return (size_t) (n / QK8_0) * sizeof(block_q8_0); }

uint16_t f2h(float f) {
    uint32_t bits;
    __builtin_memcpy(&bits, &f, 4);
    const uint32_t sign = (bits >> 31) & 1u;
    const int32_t  exp  = (int32_t) ((bits >> 23) & 0xffu) - 127;
    const uint32_t mant = bits & 0x7fffffu;
    if (((bits >> 23) & 0xffu) == 0xffu) {
        return (uint16_t) ((sign << 15) | 0x7c00u | (mant ? 0x200u : 0u));
    }
    if (exp > 15)  return (uint16_t) ((sign << 15) | 0x7c00u);
    if (exp < -14) return (uint16_t) (sign << 15);
    const uint32_t h_mant = (mant + 0xfffu + ((mant >> 13) & 1u)) >> 13;
    return (uint16_t) ((sign << 15) | ((uint32_t) (exp + 15) << 10) | (h_mant & 0x3ffu));
}

float h2f(uint16_t h) {
    const uint32_t sign = (uint32_t) (h >> 15) & 1u;
    const uint32_t exp  = (uint32_t) (h >> 10) & 0x1fu;
    const uint32_t mant = (uint32_t) h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign << 31;
        } else {
            uint32_t e = 0, m = mant;
            while ((m & 0x400u) == 0) { m <<= 1; ++e; }
            m &= 0x3ffu;
            bits = (sign << 31) | ((127 - 15 - e + 1) << 23) | (m << 13);
        }
    } else if (exp == 0x1f) {
        bits = (sign << 31) | (0xffu << 23) | (mant << 13);
    } else {
        bits = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float out;
    __builtin_memcpy(&out, &bits, 4);
    return out;
}

void ref_dequant_q8_0(const block_q8_0 * x, float * y, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        const float d = h2f(x[i].d);
        for (int j = 0; j < QK8_0; ++j) {
            y[i * QK8_0 + j] = (float) x[i].qs[j] * d;
        }
    }
}

void ref_dequant_q4_0(const block_q4_0 * x, float * y, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        const float d = h2f(x[i].d);
        for (int j = 0; j < QK4_0 / 2; ++j) {
            const int x0 = (x[i].qs[j] & 0x0F) - 8;
            const int x1 = (x[i].qs[j] >> 4) - 8;
            y[i * QK4_0 + j]             = (float) x0 * d;
            y[i * QK4_0 + j + QK4_0 / 2] = (float) x1 * d;
        }
    }
}

void ref_rope_neox(float * x, int64_t n_tokens, int n_head, int n_embd_head,
                   int n_rot, int32_t pos0, const float * theta) {
    const int n_half = n_rot / 2;
    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int h = 0; h < n_head; ++h) {
            float * row = x + (t * n_head + h) * n_embd_head;
            const float pos = (float) (pos0 + t);
            for (int i = 0; i < n_half; ++i) {
                float s, c;
                s = sinf(pos * theta[i]);
                c = cosf(pos * theta[i]);
                const float x0 = row[i];
                const float x1 = row[i + n_half];
                row[i]        = x0 * c - x1 * s;
                row[i + n_half] = x0 * s + x1 * c;
            }
        }
    }
}

template <int N>
void ref_fwht(float * x, int64_t n_rows, float scale) {
    for (int64_t r = 0; r < n_rows; ++r) {
        float * row = x + r * N;
        float tmp[N];
        for (int i = 0; i < N; ++i) tmp[i] = row[i] * scale;
        for (int h = 1; h < N; h *= 2) {
            for (int i = 0; i < N; i += 2 * h) {
                for (int j = 0; j < h; ++j) {
                    const float a = tmp[i + j];
                    const float b = tmp[i + j + h];
                    tmp[i + j]     = a + b;
                    tmp[i + j + h] = a - b;
                }
            }
        }
        for (int i = 0; i < N; ++i) row[i] = tmp[i];
    }
}

void ref_quant_q8_0(const float * x, block_q8_0 * y, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        const float * src = x + i * QK8_0;
        float amax = 0.0f;
        for (int j = 0; j < QK8_0; ++j) amax = fmaxf(amax, fabsf(src[j]));
        const float d  = amax / 127.0f;
        const float id = d ? 1.0f / d : 0.0f;
        y[i].d = f2h(d);
        for (int j = 0; j < QK8_0; ++j) y[i].qs[j] = (int8_t) roundf(src[j] * id);
    }
}

void ref_quant_q4_0(const float * x, block_q4_0 * y, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        const float * src = x + i * QK4_0;
        float amax = 0.0f, vmax = 0.0f;
        for (int j = 0; j < QK4_0; ++j) {
            const float v = src[j];
            if (amax < fabsf(v)) { amax = fabsf(v); vmax = v; }
        }
        const float d  = vmax / -8.0f;
        const float id = d ? 1.0f / d : 0.0f;
        y[i].d = f2h(d);
        for (int j = 0; j < QK4_0 / 2; ++j) {
            const float x0 = src[j] * id;
            const float x1 = src[j + QK4_0 / 2] * id;
            int xi0 = (int) (x0 + 8.5f);
            int xi1 = (int) (x1 + 8.5f);
            if (xi0 > 15) xi0 = 15;
            if (xi1 > 15) xi1 = 15;
            y[i].qs[j] = (uint8_t) ((xi0 & 15) | ((xi1 & 15) << 4));
        }
    }
}

// --- helpers ----------------------------------------------------------------

std::mt19937 rng(1234);

std::vector<float> rand_f32(size_t n, float lo = -1.0f, float hi = 1.0f) {
    std::uniform_real_distribution<float> d(lo, hi);
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = d(rng);
    return v;
}

template <typename T>
size_t byte_diff(const std::vector<T> & a, const std::vector<T> & b) {
    const uint8_t * pa = (const uint8_t *) a.data();
    const uint8_t * pb = (const uint8_t *) b.data();
    const size_t n = a.size() * sizeof(T);
    size_t bad = 0;
    for (size_t i = 0; i < n; ++i) {
        if (pa[i] != pb[i]) ++bad;
    }
    return bad;
}

double max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    double worst = 0.0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        const double d = fabs((double) a[i] - (double) b[i]);
        if (d > worst) worst = d;
    }
    return worst;
}

// tolerance: dequant/rope/fwht are exact per-element float ops, but the GPU may
// contract multiply-add differently, so allow a small relative error.
bool close(const std::vector<float> & a, const std::vector<float> & b, double tol) {
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        const double want = b[i];
        const double got  = a[i];
        const double tol_abs = tol * fmax(1.0, fabs(want));
        if (fabs(got - want) > tol_abs) {
            return false;
        }
    }
    return true;
}

void report(const std::string & what, bool ok, const char * detail) {
    printf("%-52s %s%s%s\n", what.c_str(), ok ? "PASS" : "FAIL",
           detail && detail[0] ? "  " : "", detail ? detail : "");
    if (!ok) {
        ++g_fail;
    }
}

}  // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    printf("KVMem Vulkan stage-in differential test\n");
    printf("=======================================\n");

    if (!kvmem_vk_stagein_available()) {
        printf("FATAL: Vulkan stage-in engine unavailable\n");
        return 2;
    }
    printf("engine available\n\n");

    // --- dequant q8_0 -------------------------------------------------------
    {
        const int64_t n_blocks = 512;
        const int64_t n = n_blocks * QK8_0;
        std::vector<block_q8_0> q(n_blocks);
        for (int64_t i = 0; i < n_blocks; ++i) {
            q[i].d = f2h(0.005f + (float) (rng() % 200) / 100.0f);
            for (int j = 0; j < QK8_0; ++j) {
                q[i].qs[j] = (int8_t) ((int) (rng() % 255) - 128);
            }
        }
        check(kvmem_stagein_gpu_ready(n, n_blocks * (int64_t) sizeof(block_q8_0)),
              "gpu_ready for q8_0");
        check(kvmem_stagein_h2d_packed(q.data(), n_blocks * sizeof(block_q8_0)),
              "h2d_packed q8_0");
        check(kvmem_stagein_dequant(GGML_TYPE_Q8_0, 1, n), "dequant q8_0");

        std::vector<float> want(n), got(n);
        ref_dequant_q8_0(q.data(), want.data(), n_blocks);
        const bool ok = kvmem_vk_stagein_read_f32(got.data(), n * sizeof(float))
                     && close(got, want, 1e-6);
        char buf[128];
        snprintf(buf, sizeof(buf), "max_abs_diff=%.3e", max_abs_diff(got, want));
        report("dequant q8_0 matches CPU reference", ok, buf);
    }

    // --- dequant q4_0 -------------------------------------------------------
    {
        const int64_t n_blocks = 512;
        const int64_t n = n_blocks * QK4_0;
        std::vector<block_q4_0> q(n_blocks);
        for (int64_t i = 0; i < n_blocks; ++i) {
            q[i].d = f2h(0.005f + (float) (rng() % 200) / 100.0f);
            for (int j = 0; j < QK4_0 / 2; ++j) {
                q[i].qs[j] = (uint8_t) (rng() & 0xff);
            }
        }
        check(kvmem_stagein_h2d_packed(q.data(), n_blocks * sizeof(block_q4_0)),
              "h2d_packed q4_0");
        check(kvmem_stagein_dequant(GGML_TYPE_Q4_0, 1, n), "dequant q4_0");

        std::vector<float> want(n), got(n);
        ref_dequant_q4_0(q.data(), want.data(), n_blocks);
        const bool ok = kvmem_vk_stagein_read_f32(got.data(), n * sizeof(float))
                     && close(got, want, 1e-6);
        char buf[128];
        snprintf(buf, sizeof(buf), "max_abs_diff=%.3e", max_abs_diff(got, want));
        report("dequant q4_0 matches CPU reference", ok, buf);
    }

    // --- quantize q8_0 ------------------------------------------------------
    {
        const int64_t n_blocks = 256;
        const int64_t n = n_blocks * QK8_0;
        std::vector<float> src = rand_f32(n, -4.0f, 4.0f);
        check(kvmem_stagein_gpu_ready(n, n_blocks * QK8_0), "gpu_ready for quantize q8_0");
        check(kvmem_stagein_h2d_f32(src.data(), n), "h2d_f32 for quantize q8_0");

        std::vector<block_q8_0> got(n_blocks), want(n_blocks);
        check(kvmem_stagein_quantize(GGML_TYPE_Q8_0, got.data(), n_blocks, QK8_0),
              "quantize q8_0");
        ref_quant_q8_0(src.data(), want.data(), n_blocks);
        const size_t nd = byte_diff(got, want);
        char buf[128];
        snprintf(buf, sizeof(buf), "diff_bytes=%zu/%llu", nd, (unsigned long long)(n_blocks * sizeof(block_q8_0)));
        report("quantize q8_0 byte-exact vs CPU reference", nd == 0, buf);
    }

    // --- quantize q4_0 ------------------------------------------------------
    {
        const int64_t n_blocks = 256;
        const int64_t n = n_blocks * QK4_0;
        std::vector<float> src = rand_f32(n, -4.0f, 4.0f);
        check(kvmem_stagein_h2d_f32(src.data(), n), "h2d_f32 for quantize q4_0");

        std::vector<block_q4_0> got(n_blocks), want(n_blocks);
        check(kvmem_stagein_quantize(GGML_TYPE_Q4_0, got.data(), n_blocks, QK4_0),
              "quantize q4_0");
        ref_quant_q4_0(src.data(), want.data(), n_blocks);
        const size_t nd = byte_diff(got, want);
        char buf[128];
        snprintf(buf, sizeof(buf), "diff_bytes=%zu/%llu", nd, (unsigned long long)(n_blocks * sizeof(block_q4_0)));
        report("quantize q4_0 byte-exact vs CPU reference", nd == 0, buf);
    }

    // --- fwht, every width the adapter allows -------------------------------
    for (int nrot : {64, 128, 256, 512}) {
        const int64_t n_rows  = 7;
        const int64_t n_embd  = nrot * 2;
        const int64_t n_total = n_rows * n_embd;
        std::vector<float> src = rand_f32(n_total, -2.0f, 2.0f);
        kvmem_stagein_h2d_f32(src.data(), n_total);
        if (!kvmem_stagein_fwht(n_rows, n_embd, nrot)) {
            report("fwht nrot=" + std::to_string(nrot), false, "dispatch refused");
            continue;
        }
        std::vector<float> want = src, got(n_total);
        const int64_t nrows = n_rows * (n_embd / nrot);
        const float scale = 1.0f / std::sqrt((float) nrot);
        switch (nrot) {
            case 64:  ref_fwht<64> (want.data(), nrows, scale); break;
            case 128: ref_fwht<128>(want.data(), nrows, scale); break;
            case 256: ref_fwht<256>(want.data(), nrows, scale); break;
            case 512: ref_fwht<512>(want.data(), nrows, scale); break;
        }
        const bool ok = kvmem_vk_stagein_read_f32(got.data(), n_total * sizeof(float))
                     && close(got, want, 1e-4);
        char buf[128];
        snprintf(buf, sizeof(buf), "max_abs_diff=%.3e", max_abs_diff(got, want));
        report("fwht nrot=" + std::to_string(nrot) + " matches CPU reference", ok, buf);
    }

    // --- rope ---------------------------------------------------------------
    {
        const int64_t n_tokens = 5;
        const int n_head = 4, n_embd_head = 64, n_rot = 32;
        const int64_t n_total = n_tokens * n_head * n_embd_head;
        std::vector<float> src = rand_f32(n_total, -2.0f, 2.0f);
        std::vector<float> theta(n_rot / 2);
        for (size_t i = 0; i < theta.size(); ++i) {
            theta[i] = 1.0f / (float) (10000.0 * std::pow(10000.0, -2.0 * i / (float) n_rot));
        }
        kvmem_stagein_h2d_f32(src.data(), n_total);
        check(kvmem_stagein_rope_neox(n_tokens, n_head, n_embd_head, n_rot, 7,
                                      theta.data(), n_rot / 2),
              "rope_neox dispatch");
        std::vector<float> want = src, got(n_total);
        ref_rope_neox(want.data(), n_tokens, n_head, n_embd_head, n_rot, 7, theta.data());
        const bool ok = kvmem_vk_stagein_read_f32(got.data(), n_total * sizeof(float))
                     && close(got, want, 1e-4);
        char buf[128];
        snprintf(buf, sizeof(buf), "max_abs_diff=%.3e", max_abs_diff(got, want));
        report("rope_neox matches CPU reference", ok, buf);
    }

    // --- end-to-end enqueue_k: the real stage-in path -----------------------
    {
        const int64_t nt = 16, n_embd = 128, n_head = 2, n_embd_head = 64, n_rot_rope = 32;
        const int nrot = 0;
        const int64_t n_total = nt * n_embd;
        std::vector<float> src = rand_f32(n_total, -2.0f, 2.0f);
        std::vector<block_q8_0> src_q(n_total / QK8_0);
        ref_quant_q8_0(src.data(), src_q.data(), n_total / QK8_0);
        const size_t packed_n = src_q.size() * sizeof(block_q8_0);
        const int32_t pos0 = 5;
        std::vector<float> theta(n_rot_rope / 2);
        for (size_t i = 0; i < theta.size(); ++i) theta[i] = 0.01f * (float) (i + 1);

        check(kvmem_stagein_gpu_ready(n_total, packed_n), "gpu_ready for enqueue_k");
        // The engine caches theta and, like the CUDA original it replaces, only
        // refreshes that cache when the length changes (llama-kvmem-stagein.cu:
        // if (theta && n_theta > 0 && g_st.theta.size() != n_theta)). The earlier
        // rope case cached 16 values, and this case has the same n_rot, so the
        // cache would still hold the earlier values. Flush it first so the
        // comparison starts from this case's theta.
        kvmem_stagein_gpu_free();
        check(kvmem_stagein_gpu_ready(n_total, packed_n), "gpu_ready for enqueue_k");
        std::vector<uint8_t> dst(ntotal_bytes_q8_0(n_total), 0);
        check(kvmem_stagein_enqueue_k(GGML_TYPE_Q8_0, src_q.data(), packed_n,
                                      dst.data(), nt, n_embd, nrot,
                                      n_head, n_embd_head, n_rot_rope, pos0,
                                      theta.data(), n_rot_rope / 2,
                                      nullptr, nullptr, nullptr, nullptr),
              "enqueue_k");
        int64_t copy_us = 0, rope_us = 0, had_us = 0, set_us = 0;
        check(kvmem_stagein_flush(&copy_us, &rope_us, &had_us, &set_us), "flush");

        // CPU reference: dequant -> rope -> quant
        std::vector<float> want_f32(n_total);
        ref_dequant_q8_0(src_q.data(), want_f32.data(), n_total / QK8_0);
        ref_rope_neox(want_f32.data(), nt, n_head, n_embd_head, n_rot_rope, pos0, theta.data());
        std::vector<block_q8_0> want(n_total / QK8_0);
        ref_quant_q8_0(want_f32.data(), want.data(), n_total / QK8_0);

        std::vector<uint8_t> wantb((uint8_t *) want.data(),
                                  (uint8_t *) want.data() + ntotal_bytes_q8_0(n_total));
        size_t nd = 0;
        for (size_t i = 0; i < wantb.size(); ++i) {
            if (wantb[i] != dst[i]) ++nd;
        }
        char buf[128];
        snprintf(buf, sizeof(buf), "diff_bytes=%zu/%zu", nd, wantb.size());
        report("enqueue_k end-to-end matches CPU reference", nd == 0, buf);
    }

    // --- meank --------------------------------------------------------------
    {
        const uint32_t n_layer = 4, n_embd = 128, n_keep = 6, tok0 = 2;
        const int64_t ne0 = 32;
        const size_t nb0 = 4, nb1 = ne0 * nb0, nb2 = n_embd * nb0;
        std::vector<float> k((tok0 + n_keep + 2) * n_embd);
        for (auto & v : k) v = (float) ((int) (rng() % 200) - 100) / 50.0f;

        check(kvmem_meank_ready(n_layer, n_embd), "meank_ready");
        // The K tensor lives in host coherent memory, same as a ggml Vulkan
        // buffer, which is what kvmem_meank_add reads through.
        check(kvmem_meank_add(1, GGML_TYPE_F32, k.data(), tok0, n_keep, n_embd,
                              ne0, nb0, nb1, nb2),
              "meank_add dispatch");
        std::vector<float> got(n_embd, 0.0f), want(n_embd, 0.0f);
        for (uint32_t d = 0; d < n_embd; ++d) {
            const uint32_t head = d / ne0, dim = d % ne0;
            float s = 0.0f;
            for (uint32_t t = 0; t < n_keep; ++t) {
                s += k[(tok0 + t) * (n_embd) + head * ne0 + dim];
            }
            want[d] = s;
        }
        check(kvmem_meank_d2h(1, got.data(), n_embd), "meank_d2h");
        report("meank_add matches CPU reference", close(got, want, 1e-5), "");

        kvmem_meank_zero(1);
        std::vector<float> after(n_embd, 1.0f);
        kvmem_meank_d2h(1, after.data(), n_embd);
        bool zeroed = true;
        for (float v : after) zeroed = zeroed && v == 0.0f;
        check(zeroed, "meank_zero clears the layer");
        kvmem_meank_free();
    }

    // --- batched d2d --------------------------------------------------------
    {
        std::vector<uint8_t> a(1024), b(1024, 0);
        for (size_t i = 0; i < a.size(); ++i) a[i] = (uint8_t) (i * 7 + 3);
        const void * src[1] = { a.data() };
        void *       dst[1] = { b.data() };
        const size_t nb[1]  = { a.size() };
        check(kvmem_d2d_batched(src, dst, nb, 1), "d2d_batched dispatch");
        check(memcmp(a.data(), b.data(), a.size()) == 0, "d2d_batched content");

        // several ops at once
        std::vector<std::vector<uint8_t>> s(4, std::vector<uint8_t>(256));
        std::vector<std::vector<uint8_t>> d(4, std::vector<uint8_t>(256, 0));
        const void * ss[4];
        void * dd[4];
        size_t nn[4];
        for (int i = 0; i < 4; ++i) {
            for (size_t j = 0; j < 256; ++j) s[i][j] = (uint8_t) (i * 31 + j);
            ss[i] = s[i].data(); dd[i] = d[i].data(); nn[i] = 256;
        }
        check(kvmem_d2d_batched(ss, dd, nn, 4), "d2d_batched 4 ops");
        bool all = true;
        for (int i = 0; i < 4; ++i) all = all && memcmp(s[i].data(), d[i].data(), 256) == 0;
        check(all, "d2d_batched 4 ops content");
    }

    // --- capability queries must match the CUDA contract --------------------
    check(kvmem_stagein_fwht_ok(64) && kvmem_stagein_fwht_ok(512) &&
          !kvmem_stagein_fwht_ok(96) && !kvmem_stagein_fwht_ok(0),
          "fwht_ok matches CUDA accepted widths");
    check(kvmem_stagein_quant_ok(GGML_TYPE_Q8_0) && kvmem_stagein_quant_ok(GGML_TYPE_Q4_0) &&
          !kvmem_stagein_quant_ok(GGML_TYPE_F16),
          "quant_ok matches CUDA accepted types");

    kvmem_stagein_gpu_free();
    printf("\n%s (%d failures)\n", g_fail == 0 ? "ALL PASS" : "FAILURES", g_fail);
    return g_fail == 0 ? 0 : 1;
}