// Vulkan implementation of the KVMem stage-in kernels.
//
// See llama-kvmem-stagein-vulkan.h for the design and for why no ggml buffer is
// ever handed to a shader. Each CUDA __global__ kernel in
// llama-kvmem-stagein.cu has a compute pipeline counterpart here, built once
// from SPIR-V that CMake compiles from src/adapter/vulkan-shaders/*.comp.

#include "llama-kvmem-stagein-vulkan.h"
#include "llama-kvmem-transfer.h"
#include "llama-kvmem-gpu.h"
#include "llama-kvmem-vulkan-hooks.h"

// The GDN replay descriptor struct and entry point are declared in ggml-cuda.h,
// which the adapter includes on every backend.
#include "ggml-cuda.h"

#include <vulkan/vulkan.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

// SPIR-V for each pipeline, generated from src/adapter/vulkan-shaders/*.comp by
// CMake into kvmem-vk-spv.cpp. The order of the lookups below must match the
// order of the units passed to scripts/embed-spv.py.

namespace {

constexpr int    QK8_0    = 32;
constexpr int    QK4_0    = 32;

// ggml's packed block sizes, and the word-aligned strides the quant shader writes.
// See kvmem_quant.comp for why the two differ.
constexpr size_t Q8_PACKED = 34;
constexpr size_t Q4_PACKED = 18;
constexpr size_t Q8_STRIDE = 36;
constexpr size_t Q4_STRIDE = 20;

size_t kvmem_q8_stride() { return Q8_STRIDE; }
size_t kvmem_q4_stride() { return Q4_STRIDE; }

// Copy the shader's word-aligned blocks out into ggml's packed layout, dropping
// the two padding bytes of each block.
void kvmem_unpack_quant(const uint8_t * word_aligned, void * packed, int64_t n_blocks,
                        ggml_type ty) {
    uint8_t * out = static_cast<uint8_t *>(packed);
    if (ty == GGML_TYPE_Q4_0) {
        for (int64_t i = 0; i < n_blocks; ++i) {
            std::memcpy(out + i * Q4_PACKED, word_aligned + i * Q4_STRIDE, Q4_PACKED);
        }
    } else {
        for (int64_t i = 0; i < n_blocks; ++i) {
            std::memcpy(out + i * Q8_PACKED, word_aligned + i * Q8_STRIDE, Q8_PACKED);
        }
    }
}
constexpr size_t SLAB     = (size_t) 32 * 1024 * 1024;
constexpr int    GATHER_MAX = 8192;

// The push constant block every pipeline declares. Push constants are limited
// to 128 bytes by the Vulkan 1.0 minimum and 4096 on the devices here; 64 is
// enough for the widest kernel, so one union serves all of them.
struct PushConstants {
    uint32_t u[16];
};

enum ItemKind { ITEM_K = 0, ITEM_V = 1 };

// One queued stage-in row group, exactly as the CUDA Stage::Item.
struct Item {
    size_t     off    = 0;
    size_t     nbytes = 0;
    uint8_t *  dst    = nullptr;
    ItemKind   kind   = ITEM_K;
    ggml_type  ty     = GGML_TYPE_Q8_0;
    int64_t    nt     = 0;
    int64_t    n_embd = 0;
    int        nrot   = 0;
    int        n_head = 0;
    int        n_embd_head = 0;
    int        n_rot_rope = 0;
    int32_t    pos0   = 0;
};

struct OutItem {
    size_t          off     = 0;
    size_t          nbytes  = 0;
    const uint8_t * gpu_src = nullptr;
};

// A host visible, host coherent device-side buffer the host can address
// directly. Host coherency is what removes every explicit flush/invalidate from
// this file: a memcpy into the mapping is visible to the device and a device
// write is visible to the host without a pipeline barrier for host access.
struct Scratch {
    uint8_t *   ptr   = nullptr;
    size_t      bytes = 0;
    VkBuffer    buf   = VK_NULL_HANDLE;
    uint64_t    addr  = 0;

    bool ensure(size_t n) {
        if (bytes >= n && ptr) {
            return true;
        }
        release();
        void * p = nullptr;
        if (cudaMalloc(reinterpret_cast<void **>(&p), n) != cudaSuccess || !p) {
            return false;
        }
        ptr   = static_cast<uint8_t *>(p);
        bytes = n;
        buf   = kvmem_vk_buffer_of(p);
        addr  = kvmem_vk_buffer_address(p);
        return buf != VK_NULL_HANDLE;
    }

    void release() {
        if (ptr) {
            cudaFree(ptr);
        }
        ptr   = nullptr;
        bytes = 0;
        buf   = VK_NULL_HANDLE;
        addr  = 0;
    }
};

enum Pipe {
    PIPE_DEQUANT = 0,
    PIPE_QUANT,
    PIPE_ROPE,
    PIPE_FWHT,
    PIPE_COPY,
    PIPE_MEANK,
    PIPE_GDN,
    PIPE_COUNT,
};

// SPIR-V for each pipeline, generated from src/adapter/vulkan-shaders/*.comp by
// CMake into kvmem-vk-spv.cpp. The lookup order matches the PIPE_* enum above
// and the unit order passed to scripts/embed-spv.py.

}  // namespace

uint32_t        kvmem_vk_spv_words(int pipe);
const uint32_t *kvmem_vk_spv_data(int pipe);

namespace {

struct Pipeline {
    VkPipeline       pipeline = VK_NULL_HANDLE;
    VkPipelineLayout layout   = VK_NULL_HANDLE;
    VkShaderModule   module   = VK_NULL_HANDLE;
    uint32_t         wg_size  = 256;
};

struct Engine {
    bool          ready  = false;
    Pipeline      pipes[PIPE_COUNT];

    VkDescriptorSetLayout set_layout_ops   = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout_pair  = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout_meank = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout_gdn   = VK_NULL_HANDLE;
    VkDescriptorPool      desc_pool = VK_NULL_HANDLE;
    VkDescriptorSet       set_ops   = VK_NULL_HANDLE;
    VkDescriptorSet       set_pair  = VK_NULL_HANDLE;
    VkDescriptorSet       set_meank = VK_NULL_HANDLE;
    VkDescriptorSet       set_gdn   = VK_NULL_HANDLE;

    VkCommandPool pool     = VK_NULL_HANDLE;
    VkCommandBuffer cmd     = VK_NULL_HANDLE;
    bool           recording = false;

    Scratch f32;    // dequantized / rotated scratch
    Scratch q;      // packed slab
    Scratch theta;  // rope frequencies
    Scratch ops;    // gather op list
    Scratch acc;    // mean-K accumulators
    Scratch mk;     // mean-K: staged K rows, f32 and addressable by the shader
    Scratch qout;   // quantize output; separate from q so the packed input survives
    Scratch gdn;    // GDN fold: read-only records and the read-write state
    size_t          meank_layer = 0;  // floats per layer in acc
    size_t          meank_layers = 0; // layer count the accumulator was sized for

    std::vector<Item>     items;
    std::vector<OutItem>  outs;
    std::vector<float>    theta_host;
    size_t                used    = 0;
    int                   out_cur = 0;

    std::vector<uint8_t>  host_ops;

    bool ensure_pipelines();
    bool ensure_sets();
    bool begin();
    void end();
    void release_all();

    // Bind `nbufs` buffers to `set` and record one dispatch.
    bool dispatch(Pipe p, VkDescriptorSet set, const uint32_t * buffers,
                  uint32_t nbufs, uint32_t groups_x, uint32_t groups_y,
                  const PushConstants & pc);
    void dispatch3(Pipe p, VkDescriptorSet set,
                   uint32_t gx, uint32_t gy, uint32_t gz,
                   const PushConstants & pc);
};

Engine & engine() {
    static Engine e;
    return e;
}

VkShaderModule make_module(const uint32_t * code, size_t words) {
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = words * sizeof(uint32_t);
    ci.pCode    = code;
    VkShaderModule m = VK_NULL_HANDLE;
    if (vkCreateShaderModule(kvmem_vk_device(), &ci, nullptr, &m) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    return m;
}

bool Engine::ensure_pipelines() {
    if (pipes[PIPE_DEQUANT].pipeline) {
        return true;
    }
    VkDevice dev = kvmem_vk_device();
    if (!dev) {
        return false;
    }

    // One descriptor set layout per distinct binding shape used by the shaders.
    auto mk_layout = [&](uint32_t n, VkDescriptorSetLayout * out) {
        std::vector<VkDescriptorSetLayoutBinding> b(n);
        for (uint32_t i = 0; i < n; ++i) {
            b[i].binding         = i;
            b[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            b[i].descriptorCount = 1;
            b[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        ci.bindingCount = n;
        ci.pBindings    = b.data();
        return vkCreateDescriptorSetLayout(dev, &ci, nullptr, out) == VK_SUCCESS;
    };

    // ops+memview (copy), src+dst (dequant/quant/rope), acc+memview (meank)
    if (!mk_layout(2, &set_layout_ops) || !mk_layout(2, &set_layout_pair) ||
        !mk_layout(2, &set_layout_meank) || !mk_layout(2, &set_layout_gdn)) {
        return false;
    }

    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants)};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &set_layout_pair;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;

    struct PipeSpec {
        const uint32_t * code;
        VkDescriptorSetLayout set_layout;
        uint32_t wg;
    };
    const PipeSpec specs[PIPE_COUNT] = {
        { kvmem_vk_spv_data(PIPE_DEQUANT), set_layout_pair, 256 },
        { kvmem_vk_spv_data(PIPE_QUANT),   set_layout_pair, 256 },
        { kvmem_vk_spv_data(PIPE_ROPE),    set_layout_pair,  64 },
        { kvmem_vk_spv_data(PIPE_FWHT),    set_layout_pair, 256 },
        { kvmem_vk_spv_data(PIPE_COPY),    set_layout_ops,   256 },
        { kvmem_vk_spv_data(PIPE_MEANK),   set_layout_meank,  64 },
        { kvmem_vk_spv_data(PIPE_GDN),     set_layout_gdn,    32 },
    };

    for (int i = 0; i < PIPE_COUNT; ++i) {
        Pipeline & p = pipes[i];
        VkPipelineLayoutCreateInfo pli = plci;
        pli.pSetLayouts = &specs[i].set_layout;
        if (vkCreatePipelineLayout(dev, &pli, nullptr, &p.layout) != VK_SUCCESS) {
            return false;
        }
        p.wg_size = specs[i].wg;
    }

    // Second pass now that the shader word counts are known.
    for (int i = 0; i < PIPE_COUNT; ++i) {
        const uint32_t words = kvmem_vk_spv_words(i);
        if (words == 0) {
            return false;
        }
        Pipeline & p = pipes[i];
        p.module = make_module(kvmem_vk_spv_data(i), words);
        if (!p.module) {
            return false;
        }
        VkComputePipelineCreateInfo cci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        cci.stage.module = p.module;
        cci.stage.pName  = "main";
        cci.layout       = p.layout;
        if (vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cci, nullptr,
                                     &p.pipeline) != VK_SUCCESS) {
            return false;
        }
    }
    return true;
}

bool Engine::ensure_sets() {
    if (set_pair) {
        return true;
    }
    if (!ensure_pipelines()) {
        return false;
    }
    VkDevice dev = kvmem_vk_device();

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 24};
    VkDescriptorPoolCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    ci.maxSets       = 6;
    ci.poolSizeCount = 1;
    ci.pPoolSizes     = &ps;
    if (vkCreateDescriptorPool(dev, &ci, nullptr, &desc_pool) != VK_SUCCESS) {
        return false;
    }

    struct Alloc {
        VkDescriptorSetLayout layout;
        VkDescriptorSet *     set;
    };
    const Alloc allocs[4] = {
        { set_layout_ops,   &set_ops },
        { set_layout_pair,  &set_pair },
        { set_layout_meank, &set_meank },
        { set_layout_gdn,   &set_gdn },
    };
    for (const Alloc & a : allocs) {
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool     = desc_pool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts        = &a.layout;
        if (vkAllocateDescriptorSets(dev, &ai, a.set) != VK_SUCCESS) {
            return false;
        }
    }
    return true;
}

void Engine::release_all() {
    for (Pipeline & p : pipes) {
        if (p.pipeline) vkDestroyPipeline(kvmem_vk_device(), p.pipeline, nullptr);
        if (p.module)   vkDestroyShaderModule(kvmem_vk_device(), p.module, nullptr);
        if (p.layout)   vkDestroyPipelineLayout(kvmem_vk_device(), p.layout, nullptr);
        p = Pipeline{};
    }
    if (desc_pool) vkDestroyDescriptorPool(kvmem_vk_device(), desc_pool, nullptr);
    desc_pool = VK_NULL_HANDLE;
    set_ops = set_pair = set_meank = set_gdn = VK_NULL_HANDLE;
    set_layout_ops = set_layout_pair = set_layout_meank = set_layout_gdn = VK_NULL_HANDLE;
    if (cmd)      vkFreeCommandBuffers(kvmem_vk_device(), pool, 1, &cmd);
    cmd     = VK_NULL_HANDLE;
    pool    = VK_NULL_HANDLE;
    f32.release(); q.release(); theta.release(); ops.release(); acc.release(); mk.release();
    qout.release();
    gdn.release();
    items.clear();
    outs.clear();
    theta_host.clear();
    host_ops.clear();
    used    = 0;
    out_cur = 0;
    ready   = false;
}

bool Engine::begin() {
    if (recording) {
        return true;
    }
    pool = kvmem_vk_command_pool();
    if (!pool) {
        return false;
    }
    if (!cmd) {
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool        = pool;
        ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(kvmem_vk_device(), &ai, &cmd) != VK_SUCCESS) {
            return false;
        }
    }
    vkResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(cmd, &bi) != VK_SUCCESS) {
        return false;
    }
    recording = true;
    return true;
}

void Engine::end() {
    if (!recording) {
        return;
    }
    vkEndCommandBuffer(cmd);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &cmd;
    vkQueueSubmit(kvmem_vk_queue(), 1, &si, VK_NULL_HANDLE);
    recording = false;
}

// Rewrite the two storage buffer bindings of `set`.
// Bind two storage buffers, optionally offsetting the first so a shader can
// index it from 0 for one layer or item of a larger allocation.
bool bind_pair_offset(VkDescriptorSet set, VkBuffer a, VkBuffer b, size_t a_off) {
    VkDescriptorBufferInfo bi[2] = { { a, a_off, VK_WHOLE_SIZE - a_off },
                                     { b, 0, VK_WHOLE_SIZE } };
    VkWriteDescriptorSet w[2]{};
    for (uint32_t i = 0; i < 2; ++i) {
        w[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet          = set;
        w[i].dstBinding      = i;
        w[i].descriptorCount = 1;
        w[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[i].pBufferInfo     = &bi[i];
    }
    vkUpdateDescriptorSets(kvmem_vk_device(), 2, w, 0, nullptr);
    return true;
}

bool bind_pair(VkDescriptorSet set, VkBuffer a, VkBuffer b) {
    return bind_pair_offset(set, a, b, 0);
}

// 3D dispatch. The GDN fold needs a z dimension (column groups / conv chunks),
// which the 2D dispatch() helper the other pipelines use does not carry.
void Engine::dispatch3(Pipe p, VkDescriptorSet set,
                       uint32_t gx, uint32_t gy, uint32_t gz,
                       const PushConstants & pc) {
    if (!set || gx == 0) {
        return;
    }
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipes[p].pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipes[p].layout,
                            0, 1, &set, 0, nullptr);
    vkCmdPushConstants(cmd, pipes[p].layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), pc.u);
    vkCmdDispatch(cmd, gx, gy, gz);
}

bool Engine::dispatch(Pipe p, VkDescriptorSet set, const uint32_t *,
                      uint32_t, uint32_t groups_x, uint32_t groups_y,
                      const PushConstants & pc) {
    if (!set || groups_x == 0) {
        return false;
    }
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipes[p].pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipes[p].layout,
                            0, 1, &set, 0, nullptr);
    vkCmdPushConstants(cmd, pipes[p].layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), pc.u);
    vkCmdDispatch(cmd, groups_x, groups_y, 1);
    return true;
}

Engine & E() { return engine(); }

void ensure_init() {
    Engine & e = E();
    if (e.ready) {
        return;
    }
    kvmem_vk_facade_init();
    if (!kvmem_vk_device()) {
        return;
    }
    if (!e.ensure_pipelines()) {
        return;
    }
    if (!e.ensure_sets()) {
        return;
    }
    e.ready = true;
}

bool dispatch_rope(int64_t n_tokens, int n_head, int n_embd_head, int n_rot,
                   int32_t pos0) {
    Engine & e = E();
    PushConstants pc{};
    pc.u[0] = (uint32_t) n_tokens;
    pc.u[1] = (uint32_t) n_head;
    pc.u[2] = (uint32_t) n_embd_head;
    pc.u[3] = (uint32_t) n_rot;
    pc.u[4] = (uint32_t) pos0;
    const uint32_t groups = (uint32_t) ((n_tokens + 63) / 64);
    return e.dispatch(PIPE_ROPE, e.set_pair, nullptr, 0, groups,
                      (uint32_t) n_head, pc);
}

bool dispatch_fwht(int64_t nrows, int nrot, float scale) {
    Engine & e = E();
    PushConstants pc{};
    pc.u[0] = (uint32_t) nrows;
    pc.u[1] = (uint32_t) nrot;
    float s = scale;
    __builtin_memcpy(&pc.u[2], &s, sizeof(s));
    // The shader maps one workgroup to one row, and the 256 threads of that group
    // walk the row's elements, so the x dimension counts rows.
    return e.dispatch(PIPE_FWHT, e.set_pair, nullptr, 0,
                      (uint32_t) nrows, 1, pc);
}

bool dispatch_dequant(ggml_type ty, int64_t n_blocks, size_t src_off) {
    Engine & e = E();
    PushConstants pc{};
    pc.u[0] = (uint32_t) n_blocks;
    pc.u[1] = ty == GGML_TYPE_Q4_0 ? 1u : 0u;
    pc.u[2] = (uint32_t) src_off;
    const uint32_t groups = (uint32_t) ((n_blocks + 255) / 256);
    return e.dispatch(PIPE_DEQUANT, e.set_pair, nullptr, 0, groups, 1, pc);
}

bool dispatch_quant(ggml_type ty, int64_t n_blocks) {
    Engine & e = E();
    PushConstants pc{};
    pc.u[0] = (uint32_t) n_blocks;
    pc.u[1] = ty == GGML_TYPE_Q4_0 ? 1u : 0u;
    const uint32_t groups = (uint32_t) ((n_blocks + 255) / 256);
    return e.dispatch(PIPE_QUANT, e.set_pair, nullptr, 0, groups, 1, pc);
}

}  // namespace

// --- availability -----------------------------------------------------------

bool kvmem_vk_stagein_available() {
    ensure_init();
    return E().ready;
}

size_t kvmem_vk_stagein_scratch_bytes() {
    return E().f32.bytes + E().q.bytes + E().theta.bytes + E().ops.bytes + E().acc.bytes;
}

bool kvmem_vk_stagein_read_f32(void * host, size_t n) {
    Engine & e = E();
    if (!host || n == 0 || !e.f32.ptr || n > e.f32.bytes) {
        return false;
    }
    if (e.recording) {
        e.end();
    }
    vkQueueWaitIdle(kvmem_vk_queue());
    std::memcpy(host, e.f32.ptr, n);
    return true;
}

// --- capability queries -----------------------------------------------------

bool kvmem_stagein_fwht_ok(int nrot) {
    return nrot == 64 || nrot == 128 || nrot == 256 || nrot == 512;
}

bool kvmem_stagein_quant_ok(ggml_type ty) {
    return ty == GGML_TYPE_Q8_0 || ty == GGML_TYPE_Q4_0;
}

// --- scratch management -----------------------------------------------------

bool kvmem_stagein_gpu_ready(size_t n_f32, size_t n_packed) {
    if (n_f32 == 0) {
        return false;
    }
    ensure_init();
    Engine & e = E();
    if (!e.ready) {
        return false;
    }
    if (!e.f32.ensure(n_f32 * sizeof(float))) {
        return false;
    }
    if (n_packed > 0) {
        if (!e.q.ensure(n_packed > SLAB ? n_packed : SLAB)) {
            return false;
        }
    }
    return true;
}

void kvmem_stagein_gpu_free() {
    Engine & e = E();
    if (e.recording) {
        e.end();
        vkQueueWaitIdle(kvmem_vk_queue());
    }
    e.release_all();
}

// --- primitive paths --------------------------------------------------------

bool kvmem_stagein_h2d_f32(const float * host, int64_t n) {
    Engine & e = E();
    if (!e.ready || !host || n <= 0 || (size_t) n * sizeof(float) > e.f32.bytes) {
        return false;
    }
    std::memcpy(e.f32.ptr, host, (size_t) n * sizeof(float));
    if (!e.begin()) {
        return false;
    }
    kvmem_vk_memory_barrier(e.cmd);
    e.end();
    return true;
}

bool kvmem_stagein_h2d_packed(const void * host, size_t n) {
    Engine & e = E();
    if (!e.ready || !host || n == 0 || !e.q.ptr || n > e.q.bytes) {
        return false;
    }
    std::memcpy(e.q.ptr, host, n);
    if (!e.begin()) {
        return false;
    }
    kvmem_vk_memory_barrier(e.cmd);
    e.end();
    return true;
}

bool kvmem_stagein_dequant(ggml_type ty, int64_t n_rows, int64_t n_embd) {
    Engine & e = E();
    if (!e.ready || n_rows <= 0 || n_embd <= 0) {
        return false;
    }
    const int64_t n_blocks = n_rows * (n_embd / (ty == GGML_TYPE_Q4_0 ? QK4_0 : QK8_0));
    if ((size_t) n_rows * (size_t) n_embd * sizeof(float) > e.f32.bytes) {
        return false;
    }
    if (!e.begin() || !bind_pair(e.set_pair, e.q.buf, e.f32.buf)) {
        return false;
    }
    const bool ok = dispatch_dequant(ty, n_blocks, 0);
    e.end();
    return ok;
}

bool kvmem_stagein_rope_neox(int64_t n_tokens, int n_head, int n_embd_head, int n_rot,
                             int32_t pos0, const float * theta, int n_theta) {
    Engine & e = E();
    if (!e.ready || !theta || n_tokens <= 0 || n_head <= 0 || n_embd_head <= 0) {
        return false;
    }
    if (n_rot < 2 || (n_rot % 2) != 0 || n_rot > n_embd_head || n_theta != n_rot / 2) {
        return false;
    }
    if (!e.theta.ensure((size_t) n_theta * sizeof(float))) {
        return false;
    }
    std::memcpy(e.theta.ptr, theta, (size_t) n_theta * sizeof(float));
    if (!e.begin() || !bind_pair(e.set_pair, e.f32.buf, e.theta.buf)) {
        return false;
    }
    const bool ok = dispatch_rope(n_tokens, n_head, n_embd_head, n_rot, pos0);
    e.end();
    return ok;
}

bool kvmem_stagein_fwht(int64_t n_rows, int64_t n_embd, int nrot) {
    Engine & e = E();
    if (!e.ready || n_rows <= 0 || n_embd <= 0 || nrot <= 0) {
        return false;
    }
    if (n_embd % nrot != 0 || !kvmem_stagein_fwht_ok(nrot)) {
        return false;
    }
    const int64_t nrows = n_rows * (n_embd / nrot);
    // The shader narrows the row base to 32 bits; keep the product in range.
    if (nrows > 0x7fffffffll || (size_t) nrows * sizeof(float) > e.f32.bytes) {
        return false;
    }
    const float scale = 1.0f / std::sqrt((float) nrot);
    if (!e.begin() || !bind_pair(e.set_pair, e.f32.buf, e.f32.buf)) {
        return false;
    }
    const bool ok = dispatch_fwht(nrows, nrot, scale);
    e.end();
    return ok;
}

bool kvmem_stagein_quantize(ggml_type ty, void * gpu_dst, int64_t n_rows, int64_t n_embd) {
    Engine & e = E();
    if (!e.ready || !gpu_dst || n_rows <= 0 || n_embd <= 0) {
        return false;
    }
    if (!kvmem_stagein_quant_ok(ty)) {
        return false;
    }
    const int64_t n_blocks = n_rows * (n_embd / (ty == GGML_TYPE_Q4_0 ? QK4_0 : QK8_0));
    // The quant shader writes a word-aligned layout (36 bytes per q8_0 block,
    // 20 per q4_0) because ggml's packed 34/18 byte blocks share words between
    // invocations, and two invocations cannot safely fill halves of one word.
    // The host unpacks into ggml's layout on the way out.
    const size_t packed_sz  = (size_t) n_blocks * (ty == GGML_TYPE_Q4_0 ? 18u : 34u);
    const size_t stride_sz  = (ty == GGML_TYPE_Q4_0 ? kvmem_q4_stride()
                                                     : kvmem_q8_stride());
    const size_t need = (size_t) n_blocks * stride_sz;
    if (!e.q.ensure(need)) {
        return false;
    }
    if (!e.begin() || !bind_pair(e.set_pair, e.f32.buf, e.q.buf)) {
        return false;
    }
    const bool ok = dispatch_quant(ty, n_blocks);
    e.end();
    if (!ok) {
        return false;
    }
    // The host reads the result straight out of the mapped scratch, so the queue
    // has to be idle first. end() only submits.
    vkQueueWaitIdle(kvmem_vk_queue());
    kvmem_unpack_quant(e.q.ptr, gpu_dst, n_blocks, ty);
    (void) packed_sz;
    return true;
}

bool kvmem_stagein_h2d_bytes(void * gpu_dst, const void * host, size_t n) {
    if (!gpu_dst || !host) {
        return n == 0;
    }
    if (n == 0) {
        return true;
    }
    // Both sides are host coherent memory on the Vulkan backend, so this is a
    // plain copy; the transfer counter still records the bytes.
    std::memcpy(gpu_dst, host, n);
    kvmem_record_transfer(cudaMemcpyHostToDevice, n);
    return true;
}

void kvmem_stagein_sync() {
    Engine & e = E();
    if (e.recording) {
        e.end();
    }
    vkQueueWaitIdle(kvmem_vk_queue());
}

// --- batched stage-in -------------------------------------------------------

bool kvmem_stagein_enqueue_k(
        ggml_type ty, const void * packed, size_t nbytes, uint8_t * dst,
        int64_t nt, int64_t n_embd, int nrot,
        int n_head, int n_embd_head, int n_rot_rope, int32_t pos0,
        const float * theta, int n_theta,
        int64_t * copy_us, int64_t * rope_us, int64_t * hadamard_us, int64_t * set_us) {
    (void) copy_us; (void) rope_us; (void) hadamard_us; (void) set_us;
    if (!packed || !dst || !kvmem_stagein_quant_ok(ty)) {
        return false;
    }
    if (nrot > 0 && !kvmem_stagein_fwht_ok(nrot)) {
        return false;
    }
    if ((int64_t) n_head * n_embd_head != n_embd) {
        return false;
    }
    Engine & e = E();
    if (!e.ready || !e.q.ptr || nbytes == 0 || e.used + nbytes > e.q.bytes) {
        return false;
    }
    std::memcpy(e.q.ptr + e.used, packed, nbytes);

    if (theta && n_theta > 0) {
        if (e.theta_host.size() != (size_t) n_theta) {
            e.theta_host.assign(theta, theta + n_theta);
        }
    }

    Item it;
    it.off         = e.used;
    it.nbytes      = nbytes;
    it.dst         = dst;
    it.kind        = ITEM_K;
    it.ty          = ty;
    it.nt          = nt;
    it.n_embd      = n_embd;
    it.nrot        = nrot;
    it.n_head      = n_head;
    it.n_embd_head = n_embd_head;
    it.n_rot_rope  = n_rot_rope;
    it.pos0        = pos0;
    e.items.push_back(it);
    e.used += nbytes;
    return true;
}

bool kvmem_stagein_enqueue_v(const void * packed, size_t nbytes, uint8_t * dst,
                             int64_t * set_us) {
    (void) set_us;
    if (!packed || !dst || nbytes == 0) {
        return false;
    }
    Engine & e = E();
    if (!e.ready || !e.q.ptr || e.used + nbytes > e.q.bytes) {
        return false;
    }
    std::memcpy(e.q.ptr + e.used, packed, nbytes);
    Item it;
    it.off    = e.used;
    it.nbytes = nbytes;
    it.dst    = dst;
    it.kind   = ITEM_V;
    e.items.push_back(it);
    e.used += nbytes;
    return true;
}

bool kvmem_stagein_flush(int64_t * copy_us, int64_t * rope_us,
                         int64_t * hadamard_us, int64_t * set_us) {
    Engine & e = E();
    if (e.items.empty()) {
        e.used = 0;
        return true;
    }
    if (!e.ready || !e.q.ptr || e.used > e.q.bytes) {
        e.used = 0;
        e.items.clear();
        return false;
    }

    // One barrier covers every memcpy into the slab made above.
    if (!e.begin()) {
        e.used = 0;
        e.items.clear();
        return false;
    }
    kvmem_vk_memory_barrier(e.cmd);

    if (!e.theta_host.empty() && !e.theta.ensure(e.theta_host.size() * sizeof(float))) {
        e.used = 0;
        e.items.clear();
        return false;
    }
    if (!e.theta_host.empty()) {
        std::memcpy(e.theta.ptr, e.theta_host.data(),
                    e.theta_host.size() * sizeof(float));
        kvmem_vk_memory_barrier(e.cmd);
    }

    bool ok = true;
    for (const Item & it : e.items) {
        if (it.kind == ITEM_V) {
            // Both ends are host coherent: copy straight into the destination.
            std::memcpy(it.dst, e.q.ptr + it.off, it.nbytes);
            kvmem_record_transfer(cudaMemcpyDeviceToDevice, it.nbytes);
            continue;
        }

        const int64_t nb = it.n_embd / (it.ty == GGML_TYPE_Q4_0 ? QK4_0 : QK8_0);
        const int64_t n_blocks = it.nt * nb;

        // dequant: packed slab -> f32 scratch
        bind_pair(e.set_pair, e.q.buf, e.f32.buf);
        ok = dispatch_dequant(it.ty, n_blocks, it.off);
        if (!ok) break;

        // Every stage below rewrites the descriptor set, and the stages share
        // one command buffer, so each has to be submitted before the next
        // rebinds: a descriptor write applies to the whole submission.
        e.end();
        vkQueueWaitIdle(kvmem_vk_queue());
        if (!e.begin()) { ok = false; break; }

        // rope over the same scratch, theta bound second
        if (it.n_rot_rope > 0 && e.theta_host.size() == (size_t) (it.n_rot_rope / 2)) {
            bind_pair(e.set_pair, e.f32.buf, e.theta.buf);
            ok = dispatch_rope(it.nt, it.n_head, it.n_embd_head, it.n_rot_rope, it.pos0);
            if (!ok) break;
        }

        // Walsh-Hadamard over the scratch
        if (it.nrot > 0) {
            const int64_t nrows = it.nt * (it.n_embd / it.nrot);
            e.end();
            vkQueueWaitIdle(kvmem_vk_queue());
            if (!e.begin()) { ok = false; break; }
            bind_pair(e.set_pair, e.f32.buf, e.f32.buf);
            ok = dispatch_fwht(nrows, it.nrot, 1.0f / std::sqrt((float) it.nrot));
            if (!ok) break;
        }

        // quantize: f32 scratch -> its own word aligned scratch, then unpack into
        // the caller's cache. The output cannot reuse e.q: the packed input this
        // item was dequantized from still lives there, and several items are
        // packed into one slab.
        const size_t need = (size_t) n_blocks * (it.ty == GGML_TYPE_Q4_0 ? Q4_STRIDE : Q8_STRIDE);
        if (!e.qout.ensure(need)) { ok = false; break; }
        e.end();
        vkQueueWaitIdle(kvmem_vk_queue());
        if (!e.begin()) { ok = false; break; }
        bind_pair(e.set_pair, e.f32.buf, e.qout.buf);
        ok = dispatch_quant(it.ty, n_blocks);
        if (!ok) break;
        e.end();
        vkQueueWaitIdle(kvmem_vk_queue());
        kvmem_unpack_quant(e.qout.ptr, it.dst, n_blocks, it.ty);
        if (!e.begin()) { ok = false; break; }
    }

    e.end();
    vkQueueWaitIdle(kvmem_vk_queue());
    e.used = 0;
    e.items.clear();
    (void) copy_us; (void) rope_us; (void) hadamard_us; (void) set_us;
    return ok;
}

// --- stage-out --------------------------------------------------------------

bool kvmem_stageout_enqueue(const void * gpu_src, size_t nbytes) {
    Engine & e = E();
    if (!gpu_src || nbytes == 0 || !e.items.empty()) {
        return false;
    }
    if (!e.ready || !e.q.ptr || e.used + nbytes > e.q.bytes) {
        return false;
    }
    OutItem o;
    o.off     = e.used;
    o.nbytes  = nbytes;
    o.gpu_src = static_cast<const uint8_t *>(gpu_src);
    e.outs.push_back(o);
    e.used += nbytes;
    return true;
}

size_t kvmem_stageout_used() {
    return E().used;
}

int kvmem_stageout_submit(int64_t * copy_us) {
    Engine & e = E();
    if (e.outs.empty()) {
        return -1;
    }
    if (!e.ready || !e.q.ptr || e.used == 0 || e.used > e.q.bytes) {
        return -1;
    }
    const int64_t t0 = ggml_time_us();
    for (const OutItem & o : e.outs) {
        std::memcpy(e.q.ptr + o.off, o.gpu_src, o.nbytes);
    }
    e.outs.clear();
    e.used    = 0;
    e.out_cur = 1 - e.out_cur;
    if (copy_us) {
        *copy_us += ggml_time_us() - t0;
    }
    return e.out_cur;
}

bool kvmem_stageout_wait(int slot, int64_t * copy_us) {
    if (slot < 0) {
        return true;
    }
    const int64_t t0 = ggml_time_us();
    vkQueueWaitIdle(kvmem_vk_queue());
    if (copy_us) {
        *copy_us += ggml_time_us() - t0;
    }
    return true;
}

const uint8_t * kvmem_stageout_slot_base(int slot) {
    (void) slot;
    return E().q.ptr;
}

void kvmem_stageout_clear() {
    E().outs.clear();
    if (E().items.empty()) {
        E().used = 0;
    }
}

// --- batched device copies --------------------------------------------------

bool kvmem_d2d_batched(const void * const * src, void * const * dst,
                       const size_t * nbytes, int n) {
    if (n <= 0) {
        return true;
    }
    if (!src || !dst || !nbytes) {
        return false;
    }
    uint64_t bytes = 0;
    for (int i = 0; i < n; ++i) {
        if (!src[i] || !dst[i] || nbytes[i] == 0) {
            return false;
        }
        bytes += nbytes[i];
    }
    // Host coherent memory on both ends: a host copy is the same move the
    // gather kernel performed, and it keeps the transfer accounting intact.
    for (int i = 0; i < n; ++i) {
        std::memmove(dst[i], src[i], nbytes[i]);
    }
    kvmem_record_transfer(cudaMemcpyDeviceToDevice, bytes);
    return true;
}

// --- decode mean-K ----------------------------------------------------------

bool kvmem_meank_ready(uint32_t n_layer, uint32_t n_embd) {
    if (n_layer == 0 || n_embd == 0) {
        return false;
    }
    ensure_init();
    Engine & e = E();
    if (!e.ready) {
        return false;
    }
    // Same grow-only rule the CUDA code uses: keep the allocation if it already
    // fits this shape.
    if (e.acc.ptr && e.meank_layers >= n_layer && e.meank_layer == n_embd) {
        return true;
    }
    if (e.acc.ensure((size_t) n_layer * n_embd * sizeof(float))) {
        std::memset(e.acc.ptr, 0, e.acc.bytes);
        e.meank_layer  = n_embd;
        e.meank_layers = n_layer;
        return true;
    }
    return false;
}

void kvmem_meank_free() {
    Engine & e = E();
    e.acc.release();
    e.meank_layer  = 0;
    e.meank_layers = 0;
}

void kvmem_meank_zero(uint32_t il) {
    Engine & e = E();
    if (!e.acc.ptr || il >= e.meank_layers) {
        return;
    }
    // acc.ptr is a byte pointer, so the layer offset has to be in bytes.
    std::memset(e.acc.ptr + (size_t) il * e.meank_layer * sizeof(float), 0,
                e.meank_layer * sizeof(float));
}

bool kvmem_meank_add(uint32_t il, ggml_type ty, const void * gpu_k,
                     uint32_t tok0, uint32_t n_keep, uint32_t n_embd,
                     int64_t ne0, size_t nb0, size_t nb1, size_t nb2) {
    Engine & e = E();
    if (!e.acc.ptr || !gpu_k || n_keep == 0 || n_embd == 0 || ne0 <= 0) {
        return false;
    }
    if (il >= e.meank_layers || n_embd != e.meank_layer) {
        return false;
    }
    // The shader computes byte offsets in 32 bits; refuse anything that could
    // wrap rather than corrupt memory.
    const uint64_t max_off = (uint64_t) (tok0 + n_keep) * nb2
                           + (uint64_t) n_embd * nb1;
    if (max_off > 0xffffffffull) {
        return false;
    }

    // The K tensor belongs to ggml, not to this facade, so the facade has no
    // VkBuffer for it and a shader cannot address it. On the Vulkan backend a
    // ggml tensor buffer is HOST_VISIBLE|HOST_COHERENT, so the rows the adapter
    // asked for are readable on the host: gather them into this engine's own
    // scratch, which the shader can address, and let the shader reduce there.
    const size_t want = (size_t) n_keep * n_embd * sizeof(float);
    if (!e.mk.ensure(want)) {
        return false;
    }
    const uint8_t * src = static_cast<const uint8_t *>(gpu_k);
    const uint32_t esize = ty == GGML_TYPE_F32 ? 4u : 2u;
    for (uint32_t t = 0; t < n_keep; ++t) {
        for (uint32_t head = 0; head * (uint32_t) ne0 < n_embd; ++head) {
            float * row = reinterpret_cast<float *>(e.mk.ptr) + (size_t) t * n_embd
                                    + (size_t) head * ne0;
            const uint8_t * s = src + (size_t) (tok0 + t) * nb2 + (size_t) head * nb1;
            for (uint32_t d = 0; d < (uint32_t) ne0; ++d) {
                const uint8_t * p = s + (size_t) d * nb0;
                if (ty == GGML_TYPE_F32) {
                    uint32_t bits = (uint32_t) p[0] | ((uint32_t) p[1] << 8u)
                                  | ((uint32_t) p[2] << 16u) | ((uint32_t) p[3] << 24u);
                    float v;
                    std::memcpy(&v, &bits, 4);
                    row[d] = v;
                } else if (ty == GGML_TYPE_F16) {
                    row[d] = __half2float((half) ((uint16_t) p[0] | ((uint16_t) p[1] << 8u)));
                } else {
                    row[d] = __bfloat162float((uint16_t) ((uint16_t) p[0] | ((uint16_t) p[1] << 8u)));
                }
            }
        }
    }
    (void) esize;

    PushConstants pc{};
    pc.u[0] = n_embd;
    pc.u[1] = n_keep;

    if (!e.begin()) {
        return false;
    }
    kvmem_vk_memory_barrier(e.cmd);
    // The accumulator is bound first at this layer's slice, so the kernel can
    // index acc[d] from 0 exactly like the CUDA kernel did; the staged rows are
    // bound second.
    bind_pair_offset(e.set_meank, e.acc.buf, e.mk.buf,
                     (size_t) il * e.meank_layer * sizeof(float));
    const uint32_t groups = (n_embd + 63) / 64;
    const bool ok = e.dispatch(PIPE_MEANK, e.set_meank, nullptr, 0, groups, 1, pc);
    e.end();
    vkQueueWaitIdle(kvmem_vk_queue());
    return ok;
}

bool kvmem_meank_d2h(uint32_t il, float * host, uint32_t n_embd) {
    Engine & e = E();
    if (!e.acc.ptr || !host || il >= e.meank_layers || n_embd != e.meank_layer) {
        return false;
    }
    // acc.ptr is a byte pointer, so the layer offset has to be in bytes.
    const size_t off = (size_t) il * e.meank_layer * sizeof(float);
    std::memcpy(host, e.acc.ptr + off, n_embd * sizeof(float));
    return true;
}
// --- GDN replay fold ---------------------------------------------------------

namespace {

// Shapes the CUDA fold assumes (FP32 Qwen GDN: 128-wide heads, 16 key heads,
// 48 value heads, conv width 4).
constexpr uint GDN_HEAD_DIM   = 128;
constexpr uint GDN_N_HEADS    = 48;
constexpr uint GDN_N_KEY_HEAD = 16;
constexpr uint GDN_N_CHAN     = 10240;
constexpr uint GDN_CONV_W     = 3;

// Byte offsets, in floats, of each staged region. Records are indexed by t, so
// the read-only regions grow with n_keep; state and conv have a fixed size.
struct GdnLayout {
    uint key, value, gate, beta, conv_input;   // read-only
    uint state, conv;                          // read-write
    uint total;                                // in floats
};

GdnLayout gdn_layout_for(int n_keep) {
    const size_t per_t_key   = (size_t) GDN_N_KEY_HEAD * GDN_HEAD_DIM;
    const size_t per_t_value = (size_t) GDN_N_HEADS  * GDN_HEAD_DIM;
    const size_t per_t_gate  = (size_t) GDN_N_HEADS;
    const size_t per_t_conv  = (size_t) GDN_N_CHAN;

    GdnLayout L{};
    L.key        = 0;
    L.value      = (uint) ((size_t) n_keep * per_t_key);
    L.gate       = (uint) (L.value + (size_t) n_keep * per_t_value);
    L.beta       = (uint) (L.gate  + (size_t) n_keep * per_t_gate);
    L.conv_input = (uint) (L.beta  + (size_t) n_keep * per_t_gate);
    L.state      = (uint) (L.conv_input + (size_t) n_keep * per_t_conv);
    L.conv       = (uint) (L.state + (size_t) GDN_N_HEADS * GDN_HEAD_DIM * GDN_HEAD_DIM);
    L.total      = (uint) (L.conv  + (size_t) GDN_N_CHAN * GDN_CONV_W);
    return L;
}

// Copy one layer's records into the staging buffer. The read-only regions come
// from the caller; state and conv are the read-write ones and are seeded to zero
// so a fold onto freshly staged storage is well defined.
void gdn_stage_layer(float * dst, const GdnLayout & L,
                     const ggml_cuda_gdn_replay_layer & src, int n_keep) {
    const size_t per_t_key   = (size_t) GDN_N_KEY_HEAD * GDN_HEAD_DIM;
    const size_t per_t_value = (size_t) GDN_N_HEADS  * GDN_HEAD_DIM;
    const size_t per_t_gate  = (size_t) GDN_N_HEADS;
    const size_t per_t_conv  = (size_t) GDN_N_CHAN;

    for (int t = 0; t < n_keep; ++t) {
        if (src.key) {
            std::memcpy(dst + L.key + (size_t) t * per_t_key,
                        src.key + (size_t) t * per_t_key, per_t_key * sizeof(float));
        }
        if (src.value) {
            std::memcpy(dst + L.value + (size_t) t * per_t_value,
                        src.value + (size_t) t * per_t_value, per_t_value * sizeof(float));
        }
        if (src.gate) {
            std::memcpy(dst + L.gate + (size_t) t * per_t_gate,
                        src.gate + (size_t) t * per_t_gate, per_t_gate * sizeof(float));
        }
        if (src.beta) {
            std::memcpy(dst + L.beta + (size_t) t * per_t_gate,
                        src.beta + (size_t) t * per_t_gate, per_t_gate * sizeof(float));
        }
        if (src.conv_input) {
            std::memcpy(dst + L.conv_input + (size_t) t * per_t_conv,
                        src.conv_input + (size_t) t * per_t_conv, per_t_conv * sizeof(float));
        }
    }
    std::memset(dst + L.state, 0,
                (size_t) GDN_N_HEADS * GDN_HEAD_DIM * GDN_HEAD_DIM * sizeof(float));
    std::memset(dst + L.conv, 0, (size_t) GDN_N_CHAN * GDN_CONV_W * sizeof(float));
}

// One dispatch of the state fold for every layer, then one of the conv fold.
bool gdn_fold_run(const ggml_cuda_gdn_replay_layer * layers, int n_layers, int n_keep) {
    Engine & e = E();
    const GdnLayout L = gdn_layout_for(n_keep);
    const size_t per_layer = (size_t) L.total * sizeof(float);
    if (!e.gdn.ensure(per_layer * (size_t) n_layers)) {
        return false;
    }
    float * base = reinterpret_cast<float *>(e.gdn.ptr);
    for (int i = 0; i < n_layers; ++i) {
        gdn_stage_layer(base + (size_t) i * L.total, L, layers[i], n_keep);
    }

    // Read-only records and read-write state share one buffer here, so both
    // bindings point at it with different offsets.
    if (!e.begin()) {
        return false;
    }
    kvmem_vk_memory_barrier(e.cmd);

    for (int i = 0; i < n_layers; ++i) {
        PushConstants pc{};
        pc.u[0] = (uint32_t) n_keep;
        pc.u[1] = 0;                          // state fold
        pc.u[2] = GDN_N_CHAN;
        pc.u[3] = (uint32_t) i;
        pc.u[4] = (uint32_t) (L.key        * sizeof(float));
        pc.u[5] = (uint32_t) (L.conv       * sizeof(float));
        pc.u[6] = (uint32_t) (L.value      * sizeof(float));
        pc.u[7] = (uint32_t) (L.gate       * sizeof(float));
        pc.u[8] = (uint32_t) (L.beta       * sizeof(float));
        pc.u[9] = (uint32_t) (L.conv_input * sizeof(float));
        pc.u[10] = (uint32_t) (L.state * sizeof(float));
        bind_pair_offset(e.set_gdn, e.gdn.buf, e.gdn.buf,
                         (size_t) i * per_layer);
        // groups_x = heads, groups_y = layers, groups_z = column groups
        e.dispatch3(PIPE_GDN, e.set_gdn, GDN_N_HEADS, (uint32_t) n_layers, 32, pc);
    }

    for (int i = 0; i < n_layers; ++i) {
        PushConstants pc{};
        pc.u[0] = (uint32_t) n_keep;
        pc.u[1] = 1;                          // conv history fold
        pc.u[2] = GDN_N_CHAN;
        pc.u[3] = (uint32_t) i;
        pc.u[4] = (uint32_t) (L.key        * sizeof(float));
        pc.u[5] = (uint32_t) (L.conv       * sizeof(float));
        pc.u[6] = (uint32_t) (L.value      * sizeof(float));
        pc.u[7] = (uint32_t) (L.gate       * sizeof(float));
        pc.u[8] = (uint32_t) (L.beta       * sizeof(float));
        pc.u[9] = (uint32_t) (L.conv_input * sizeof(float));
        pc.u[10] = (uint32_t) (L.state * sizeof(float));
        bind_pair_offset(e.set_gdn, e.gdn.buf, e.gdn.buf,
                         (size_t) i * per_layer);
        e.dispatch3(PIPE_GDN, e.set_gdn, (GDN_N_CHAN + 255) / 256, (uint32_t) n_layers, 1, pc);
    }

    e.end();
    vkQueueWaitIdle(kvmem_vk_queue());
    return true;
}

}  // namespace

// Vulkan implementation of ggml_backend_cuda_gdn_fold. The adapter calls that
// name (it is the only spelling the GDN replay code uses) but on this backend the
// work is done by the PIPE_GDN compute pipeline instead of a .cu kernel.
//
// The CUDA original is two kernels in
// llama.cpp/ggml/src/ggml-cuda/gated_delta_net.cu over raw device pointers. The
// same arithmetic runs in kvmem_gdn_fold.comp. Two differences are inherent to
// Vulkan rather than to this port:
//
//   * A ggml tensor buffer has no VkBuffer this facade could bind, so the
//     records are staged into engine-owned scratch first. That costs a copy the
//     CUDA path does not pay, but it keeps the fold inside buffers this backend
//     owns.
//   * The sub-group reduction becomes a shared-memory tree. The sum is the same;
//     only the instruction sequence differs.
//
// If the scratch cannot be sized, or the engine is not up, this returns false
// rather than silently leaving the recurrent state un-folded. The caller
// (gdn_replay_commit) treats false as "poison the replay", so an unsupported
// configuration degrades loudly instead of corrupting state.
bool kvmem_backend_gdn_fold(const ggml_cuda_gdn_replay_layer * layers,
                            int n_layers, int n_keep, int capacity) {
    // n_keep == 0 is a no-op, matching the CUDA guard order.
    if (n_keep == 0) {
        return true;
    }
    if (!layers || n_layers <= 0 || n_keep < 0 || n_keep > capacity || capacity > 6) {
        return false;
    }
    ensure_init();
    if (!E().ready) {
        return false;
    }
    return gdn_fold_run(layers, n_layers, n_keep);
}
