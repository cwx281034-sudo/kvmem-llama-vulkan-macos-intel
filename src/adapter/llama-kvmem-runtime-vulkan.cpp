// Vulkan implementation of the KVMem adapter runtime shim.
//
// src/adapter/llama-kvmem-runtime-vulkan.h declares the CUDA spelling the
// adapter uses. This file defines it on top of the Vulkan loader API.
//
// What this file is responsible for
//   * device memory (cudaMalloc/cudaFree) and host-visible pinned memory
//     (cudaMallocHost/cudaFreeHost),
//   * buffer copies in all three directions, synchronous and on a queue,
//   * queue and fence handles (cudaStream_t / cudaEvent_t) the adapter stores in
//     its staging structs.
//
// What this file is NOT
//   * It runs no compute. The stage-in kernels are
//     llama-kvmem-stagein-vulkan.cpp, which binds its own pipelines against the
//     shared device and queue created here.
//
// Why the adapter's raw device pointers are NOT always host addressable here
//   An earlier version of this file assumed ggml's Vulkan backend always hands
//   out host visible memory, so any pointer the adapter produced could be
//   dereferenced with memcpy. That assumption is wrong on a discrete GPU.
//   ggml_vk_create_buffer_device() in ggml-vulkan-buffers.cpp requests
//   eDeviceLocal | eHostVisible | eHostCoherent, but falls back to plain
//   eDeviceLocal when the driver cannot satisfy the host visible request and
//   neither GGML_VK_PREFER_HOST_MEMORY nor GGML_VK_ALLOW_SYSMEM_FALLBACK is
//   set - which is the default. On such a device, tensor->data is a VRAM
//   address that faults the moment host code dereferences it.
//
//   ggml does not expose the memory type through its buffer interface either:
//   ggml_backend_vk_buffer_type_interface leaves .is_host NULL, so
//   ggml_backend_buffer_is_host() is false for every Vulkan buffer regardless of
//   whether it happens to be mapped. The adapter therefore cannot infer
//   readability from the buffer, and this facade cannot infer it from a bare
//   pointer either - only the allocation history is known here.
//
//   So this file does not guess. cudaMemcpyAsync routes every copy through a
//   copy command when either side is a facade allocation, and for pointers it
//   did not allocate it refuses the copy rather than dereferencing them
//   (see host_accessible()). Callers that hold a ggml tensor use the portable
//   ggml_backend_tensor_get/_set path, which lets ggml pick the right copy
//   command for its own buffer.

#define KVMEM_VK_FACADE_IMPL 1
#include "llama-kvmem-runtime-vulkan.h"
#include "llama-kvmem-vulkan-hooks.h"

#include <vulkan/vulkan.h>

#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

// One device, one queue. The adapter calls cudaSetDevice() with small ordinals
// for its multi-GPU paths, but on a single-Vulkan-device build there is nothing
// to switch to: the facade maps every ordinal onto the one device and reports
// the requested ordinal back, so the adapter's own bookkeeping stays consistent.
struct DeviceState {
    VkInstance                 instance = VK_NULL_HANDLE;
    VkPhysicalDevice           physical = VK_NULL_HANDLE;
    VkDevice                   device   = VK_NULL_HANDLE;
    VkQueue                    queue    = VK_NULL_HANDLE;
    uint32_t                   family   = 0;
    VkCommandPool              pool     = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties mem{};
    std::vector<VkCommandBuffer> cmd;
    std::vector<VkFence>          fences;
    size_t                       next_fence = 0;

    // allocations this facade made, so cudaFree/cudaPointerGetAttributes can
    // recognise them and report a device pointer.
    std::unordered_map<void *, VkBuffer> buffers;
    std::unordered_map<void *, VkDeviceMemory> memories;
    std::unordered_map<void *, bool>   host;      // true: host visible allocation
    std::unordered_map<void *, void *> mapped;    // host pointer -> VkDeviceMemory

    // The host address range of every mapped allocation this facade made.
    // Only a pointer inside one of these ranges may be dereferenced by host
    // code; anything else is either a VRAM address or unrelated host memory and
    // must go through a copy command. Kept as ranges rather than exact keys so
    // a pointer to the middle of a buffer - which is what the adapter always
    // passes, since it offsets by rows - is recognised too.
    std::vector<std::pair<uintptr_t, uintptr_t>> readable;

    int32_t current = 0;

    bool ensure();
    void destroy();

    uint32_t find_memory(uint32_t type_bits, VkMemoryPropertyFlags want) const {
        for (uint32_t i = 0; i < mem.memoryTypeCount; ++i) {
            if ((type_bits & (1u << i)) &&
                (mem.memoryTypes[i].propertyFlags & want) == want) {
                return i;
            }
        }
        return UINT32_MAX;
    }

    // Is [ptr, ptr+bytes) inside a mapping this facade created and still owns?
    bool host_accessible(const void * ptr, size_t bytes) const {
        if (!ptr) {
            return false;
        }
        const uintptr_t lo = reinterpret_cast<uintptr_t>(ptr);
        // A zero length copy at the very end of a buffer is still a valid
        // address to form, but nothing is read or written, so accept it.
        if (bytes == 0) {
            for (const auto & r : readable) {
                if (lo >= r.first && lo <= r.second) {
                    return true;
                }
            }
            return false;
        }
        if (lo > UINTPTR_MAX - bytes) {
            return false;
        }
        const uintptr_t hi = lo + bytes;
        for (const auto & r : readable) {
            if (lo >= r.first && hi <= r.second) {
                return true;
            }
        }
        return false;
    }
};

DeviceState & state() {
    static DeviceState s;
    return s;
}

bool init_instance(DeviceState & s) {
    if (s.instance) {
        return true;
    }

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName  = "kvmem";
    app.apiVersion       = VK_API_VERSION_1_0;

    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    if (vkCreateInstance(&ici, nullptr, &s.instance) != VK_SUCCESS) {
        fprintf(stderr, "KVMem vulkan: vkCreateInstance failed\n");
        return false;
    }

    uint32_t n = 0;
    vkEnumeratePhysicalDevices(s.instance, &n, nullptr);
    if (n == 0) {
        fprintf(stderr, "KVMem vulkan: no Vulkan physical device\n");
        vkDestroyInstance(s.instance, nullptr);
        s.instance = VK_NULL_HANDLE;
        return false;
    }
    std::vector<VkPhysicalDevice> pds(n);
    vkEnumeratePhysicalDevices(s.instance, &n, pds.data());

    // Prefer a discrete GPU; fall back to whatever is present.
    uint32_t pick = 0;
    VkPhysicalDeviceProperties props{};
    for (uint32_t i = 0; i < n; ++i) {
        VkPhysicalDeviceProperties p{};
        vkGetPhysicalDeviceProperties(pds[i], &p);
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            pick = i;
            break;
        }
    }
    s.physical = pds[pick];
    vkGetPhysicalDeviceProperties(s.physical, &props);
    vkGetPhysicalDeviceMemoryProperties(s.physical, &s.mem);

    uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(s.physical, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qf(qn);
    vkGetPhysicalDeviceQueueFamilyProperties(s.physical, &qn, qf.data());
    for (uint32_t i = 0; i < qn; ++i) {
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            s.family = i;
            break;
        }
    }

    const float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = s.family;
    qci.queueCount       = 1;
    qci.pQueuePriorities = &prio;

    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos    = &qci;
    if (vkCreateDevice(s.physical, &dci, nullptr, &s.device) != VK_SUCCESS) {
        fprintf(stderr, "KVMem vulkan: vkCreateDevice failed on %s\n", props.deviceName);
        vkDestroyInstance(s.instance, nullptr);
        s.instance = VK_NULL_HANDLE;
        return false;
    }
    vkGetDeviceQueue(s.device, s.family, 0, &s.queue);

    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.queueFamilyIndex = s.family;
    pci.flags           = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vkCreateCommandPool(s.device, &pci, nullptr, &s.pool);
    return true;
}

bool DeviceState::ensure() {
    if (device) {
        return true;
    }
    return init_instance(*this);
}

void DeviceState::destroy() {
    if (pool) {
        vkDestroyCommandPool(device, pool, nullptr);
        pool = VK_NULL_HANDLE;
    }
    for (const auto & kv : buffers) {
        vkDestroyBuffer(device, kv.second, nullptr);
    }
    for (const auto & kv : memories) {
        if (mapped.count(kv.second)) {
            vkUnmapMemory(device, kv.second);
        }
        vkFreeMemory(device, kv.second, nullptr);
    }
    buffers.clear();
    memories.clear();
    host.clear();
    mapped.clear();
    if (device) {
        vkDestroyDevice(device, nullptr);
        device = VK_NULL_HANDLE;
    }
    if (instance) {
        vkDestroyInstance(instance, nullptr);
        instance = VK_NULL_HANDLE;
    }
}

// Allocate a VkBuffer plus memory. is_host selects the memory property set.
VkBuffer alloc_buffer(DeviceState & s, size_t n, void ** mapped_out, bool is_host) {
    *mapped_out = nullptr;
    if (!s.ensure() || n == 0) {
        return VK_NULL_HANDLE;
    }

    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size  = n;
    // Buffer device address is what lets a shader reach a ggml tensor buffer;
    // it needs the extended usage flag. It is optional, so the caller checks.
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
              | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
              | VK_BUFFER_USAGE_TRANSFER_DST_BIT
              | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
              | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    VkBuffer buf = VK_NULL_HANDLE;
    if (vkCreateBuffer(s.device, &bci, nullptr, &buf) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(s.device, buf, &req);

    // Every allocation here is host visible, because the adapter reaches its
    // scratch through plain memcpy instead of a copy command. Preferring
    // DEVICE_LOCAL would pick a memory type that cannot be mapped on a discrete
    // GPU, so host visibility is the requirement and DEVICE_LOCAL only wins
    // when it happens to satisfy it.
    const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    uint32_t type = s.find_memory(req.memoryTypeBits, want);
    if (type == UINT32_MAX) {
        type = s.find_memory(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    }
    if (type == UINT32_MAX) {
        vkDestroyBuffer(s.device, buf, nullptr);
        return VK_NULL_HANDLE;
    }

    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = type;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    if (vkAllocateMemory(s.device, &mai, nullptr, &mem) != VK_SUCCESS) {
        vkDestroyBuffer(s.device, buf, nullptr);
        return VK_NULL_HANDLE;
    }
    vkBindBufferMemory(s.device, buf, mem, 0);

    void * ptr = nullptr;
    if (s.mem.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        if (vkMapMemory(s.device, mem, 0, VK_WHOLE_SIZE, 0, &ptr) != VK_SUCCESS) {
            ptr = nullptr;
        }
        s.mapped[mem] = ptr;
    }
    if (!ptr) {
        // A buffer the host cannot map is unusable to every caller here: the
        // adapter addresses its scratch directly with memcpy rather than going
        // through a copy command. Reporting success with a null mapping would
        // turn the failure into a null-pointer write far from the cause, so
        // fail the allocation instead.
        vkDestroyBuffer(s.device, buf, nullptr);
        vkFreeMemory(s.device, mem, nullptr);
        return VK_NULL_HANDLE;
    }
    s.buffers[ptr] = buf;
    s.memories[ptr] = mem;
    s.host[ptr] = is_host;
    if (ptr) {
        s.readable.push_back(std::make_pair(
            reinterpret_cast<uintptr_t>(ptr),
            reinterpret_cast<uintptr_t>(ptr) + n));
    }
    *mapped_out = ptr;
    return buf;
}

void free_buffer(DeviceState & s, void * ptr) {
    if (!ptr) {
        return;
    }
    auto bit = s.buffers.find(ptr);
    auto mit = s.memories.find(ptr);
    if (bit == s.buffers.end() || mit == s.memories.end()) {
        return;
    }
    if (s.mapped.count(mit->second)) {
        vkUnmapMemory(s.device, mit->second);
        s.mapped.erase(mit->second);
    }
    vkDestroyBuffer(s.device, bit->second, nullptr);
    vkFreeMemory(s.device, mit->second, nullptr);
    s.buffers.erase(bit);
    s.memories.erase(mit);
    if (s.host.count(ptr)) {
        s.host.erase(ptr);
    }
    const uintptr_t lo = reinterpret_cast<uintptr_t>(ptr);
    s.readable.erase(std::remove_if(s.readable.begin(), s.readable.end(),
        [&](const std::pair<uintptr_t,uintptr_t> & r) {
            return r.first == lo;
        }),
        s.readable.end());
}

std::mutex & lock() {
    static std::mutex m;
    return m;
}

// Submit a finished command buffer and wait for it. The adapter's copy paths
// are host-visible memory operations whose results it reads immediately, so the
// facade does not need to overlap them.
void submit_and_wait(DeviceState & s, VkCommandBuffer cb) {
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &cb;
    vkQueueSubmit(s.queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(s.queue);
}

struct StreamState {
    // A CUDA stream handle is a queue; here every stream is the one queue. The
    // flag records whether the caller asked for a non-blocking stream, which
    // changes nothing here because there is only ever one queue.
    bool created = false;
};

struct EventState {
    VkFence fence = VK_NULL_HANDLE;
    bool    armed = false;
};

}  // namespace

// --- stream / event handles -------------------------------------------------

struct kvmem_vk_stream { StreamState st; };
struct kvmem_vk_event  { EventState  st; };

cudaStream_t cudaStreamPerThread = nullptr;

namespace {
kvmem_vk_stream & default_stream() {
    static kvmem_vk_stream s;
    return s;
}
}  // namespace

// --- stage-in TU hooks ------------------------------------------------------
// The stage-in kernels build their pipelines on the device and queue created
// here. Declarations live in llama-kvmem-vulkan-hooks.h.

void kvmem_vk_facade_init() {
    std::lock_guard<std::mutex> g(lock());
    DeviceState & s = state();
    if (!s.ensure()) {
        return;
    }
    if (!cudaStreamPerThread) {
        cudaStreamPerThread = &default_stream();
    }
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t * stream, unsigned int) {
    if (!stream) {
        return kvmem_vk_error_invalid_value;
    }
    std::lock_guard<std::mutex> g(lock());
    state().ensure();
    // Every stream is the same queue; the handle exists so the adapter's
    // "stream == nullptr means not created yet" logic keeps working.
    static kvmem_vk_stream owned;
    *stream = &owned;
    return kvmem_vk_success;
}

cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    (void) stream;
    return kvmem_vk_success;
}

cudaError_t cudaStreamSynchronize(cudaStream_t) {
    std::lock_guard<std::mutex> g(lock());
    DeviceState & s = state();
    if (!s.device) {
        return kvmem_vk_success;
    }
    vkDeviceWaitIdle(s.device);
    return kvmem_vk_success;
}

cudaError_t cudaDeviceSynchronize(void) {
    return cudaStreamSynchronize(cudaStreamPerThread);
}

cudaError_t cudaEventCreateWithFlags(cudaEvent_t * event, unsigned int) {
    if (!event) {
        return kvmem_vk_error_invalid_value;
    }
    std::lock_guard<std::mutex> g(lock());
    DeviceState & s = state();
    if (!s.ensure()) {
        return kvmem_vk_error_not_initialized;
    }
    kvmem_vk_event * e = new kvmem_vk_event();
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    vkCreateFence(s.device, &fci, nullptr, &e->st.fence);
    *event = e;
    return kvmem_vk_success;
}

cudaError_t cudaEventDestroy(cudaEvent_t event) {
    if (!event) {
        return kvmem_vk_success;
    }
    std::lock_guard<std::mutex> g(lock());
    if (event->st.fence) {
        vkDestroyFence(state().device, event->st.fence, nullptr);
    }
    delete event;
    return kvmem_vk_success;
}

// Recording an event fences the queue, which is what the adapter means by
// "the work up to this point is done".
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t) {
    if (!event) {
        return kvmem_vk_error_invalid_value;
    }
    std::lock_guard<std::mutex> g(lock());
    DeviceState & s = state();
    if (!s.device) {
        return kvmem_vk_success;
    }
    vkResetFences(s.device, 1, &event->st.fence);
    event->st.armed = true;
    return kvmem_vk_success;
}

cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    if (!event || !event->st.armed) {
        return kvmem_vk_success;
    }
    std::lock_guard<std::mutex> g(lock());
    DeviceState & s = state();
    if (!s.device) {
        return kvmem_vk_success;
    }
    vkWaitForFences(s.device, 1, &event->st.fence, VK_TRUE, UINT64_MAX);
    return kvmem_vk_success;
}

cudaError_t cudaStreamWaitEvent(cudaStream_t, cudaEvent_t event, unsigned int) {
    if (!event) {
        return kvmem_vk_success;
    }
    return cudaEventSynchronize(event);
}

// --- memory -----------------------------------------------------------------

cudaError_t cudaMalloc(void ** ptr, size_t n) {
    if (!ptr) {
        return kvmem_vk_error_invalid_value;
    }
    std::lock_guard<std::mutex> g(lock());
    void * mapped = nullptr;
    if (!alloc_buffer(state(), n, &mapped, /*is_host=*/false)) {
        *ptr = nullptr;
        return kvmem_vk_error_out_of_memory;
    }
    *ptr = mapped;
    return kvmem_vk_success;
}

cudaError_t cudaFree(void * ptr) {
    std::lock_guard<std::mutex> g(lock());
    free_buffer(state(), ptr);
    return kvmem_vk_success;
}

cudaError_t cudaMallocHost(void ** ptr, size_t n) {
    if (!ptr) {
        return kvmem_vk_error_invalid_value;
    }
    std::lock_guard<std::mutex> g(lock());
    void * mapped = nullptr;
    if (!alloc_buffer(state(), n, &mapped, /*is_host=*/true)) {
        *ptr = nullptr;
        return kvmem_vk_error_out_of_memory;
    }
    *ptr = mapped;
    return kvmem_vk_success;
}

cudaError_t cudaFreeHost(void * ptr) {
    return cudaFree(ptr);
}

cudaError_t cudaMemsetAsync(void * dst, int value, size_t n, cudaStream_t) {
    if (!dst || n == 0) {
        return kvmem_vk_success;
    }
    // Host coherent memory makes a plain memset visible to the device.
    std::memset(dst, value, n);
    return kvmem_vk_success;
}

cudaError_t cudaMemcpy(void * dst, const void * src, size_t n, cudaMemcpyKind) {
    return cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToHost, nullptr);
}

cudaError_t cudaMemcpyAsync(void * dst, const void * src, size_t n,
                            cudaMemcpyKind kind, cudaStream_t) {
    if (!dst || !src || n == 0) {
        return kvmem_vk_success;
    }
    std::lock_guard<std::mutex> g(lock());
    DeviceState & s = state();
    if (!s.ensure()) {
        return kvmem_vk_error_not_initialized;
    }

    const bool from_device = (kind == cudaMemcpyDeviceToHost ||
                              kind == cudaMemcpyDeviceToDevice);
    const bool to_device   = (kind == cudaMemcpyHostToDevice ||
                              kind == cudaMemcpyDeviceToDevice);

    // Only pointers this facade allocated are known VkBuffers. A ggml tensor
    // buffer is a different allocation and is not in the registry, so this
    // facade never issues a vkCmdCopyBuffer for one; the stage-in TU reaches
    // those buffers directly instead.
    auto dit = s.buffers.find(dst);
    auto sit = s.buffers.find(const_cast<void *>(src));
    const bool dst_is_buffer = dit != s.buffers.end();
    const bool src_is_buffer = sit != s.buffers.end();

    // Both ends are plain host memory: a memcpy is all that is needed.
    if (!src_is_buffer && !dst_is_buffer) {
        // Do not blindly dereference arbitrary pointers. A ggml tensor's
        // device pointer cannot be read or written on the host (e.g. discrete
        // GPU with DEVICE_LOCAL-only VRAM). The caller must use
        // ggml_backend_tensor_get/_set instead.
        if (!s.host_accessible(src, n) || !s.host_accessible(dst, n)) {
            return kvmem_vk_error_invalid_value;
        }
        std::memmove(dst, src, n);
        return kvmem_vk_success;
    }

    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool        = s.pool;
    cai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(s.device, &cai, &cb);

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb, &bi);

    VkBufferCopy region{0, 0, n};
    void * stage = nullptr;

    if (src_is_buffer && dst_is_buffer) {
        vkCmdCopyBuffer(cb, sit->second, dit->second, 1, &region);
        vkEndCommandBuffer(cb);
        submit_and_wait(s, cb);
        vkFreeCommandBuffers(s.device, s.pool, 1, &cb);
        return kvmem_vk_success;
    }

    if (dst_is_buffer) {
        // host to buffer: memcpy into a staging buffer, then one device copy
        if (!s.host_accessible(src, n)) {
            vkEndCommandBuffer(cb);
            vkFreeCommandBuffers(s.device, s.pool, 1, &cb);
            return kvmem_vk_error_invalid_value;
        }
        VkBuffer sb = alloc_buffer(s, n, &stage, /*is_host=*/true);
        if (!sb) {
            vkEndCommandBuffer(cb);
            vkFreeCommandBuffers(s.device, s.pool, 1, &cb);
            return kvmem_vk_error_out_of_memory;
        }
        std::memcpy(stage, src, n);
        vkCmdCopyBuffer(cb, sb, dit->second, 1, &region);
        vkEndCommandBuffer(cb);
        submit_and_wait(s, cb);
        free_buffer(s, stage);
        vkFreeCommandBuffers(s.device, s.pool, 1, &cb);
        return kvmem_vk_success;
    }

    // buffer to host: copy into a staging buffer and read it back
    VkBuffer sb = alloc_buffer(s, n, &stage, /*is_host=*/true);
    if (!sb) {
        vkEndCommandBuffer(cb);
        vkFreeCommandBuffers(s.device, s.pool, 1, &cb);
        return kvmem_vk_error_out_of_memory;
    }
    vkCmdCopyBuffer(cb, sit->second, sb, 1, &region);
    vkEndCommandBuffer(cb);
    submit_and_wait(s, cb);
    std::memcpy(dst, stage, n);
    free_buffer(s, stage);
    vkFreeCommandBuffers(s.device, s.pool, 1, &cb);
    (void) to_device;
    (void) from_device;
    return kvmem_vk_success;
}

// --- device selection -------------------------------------------------------

cudaError_t cudaSetDevice(int) {
    std::lock_guard<std::mutex> g(lock());
    state().ensure();
    return kvmem_vk_success;
}

cudaError_t cudaGetDevice(int * device) {
    if (!device) {
        return kvmem_vk_error_invalid_value;
    }
    std::lock_guard<std::mutex> g(lock());
    *device = state().current;
    return kvmem_vk_success;
}

cudaError_t cudaPointerGetAttributes(struct cudaPointerAttributes * attrs,
                                     const void * ptr) {
    if (!attrs) {
        return kvmem_vk_error_invalid_value;
    }
    std::lock_guard<std::mutex> g(lock());
    DeviceState & s = state();
    auto it = s.buffers.find(const_cast<void *>(ptr));
    if (it == s.buffers.end()) {
        attrs->type          = cudaMemoryTypeUnregistered;
        attrs->device        = -1;
        attrs->devicePointer = nullptr;
        attrs->hostPointer   = const_cast<void *>(ptr);
        return kvmem_vk_success;
    }
    // Host coherent memory is both: report it the way CUDA reports malloc'd
    // host memory, which is what the adapter already treats as "not a device
    // pointer" for its own pinned buffers.
    const bool is_host = s.host[const_cast<void *>(ptr)];
    attrs->type          = is_host ? cudaMemoryTypeHost : cudaMemoryTypeDevice;
    attrs->device        = is_host ? -1 : 0;
    attrs->devicePointer = const_cast<void *>(ptr);
    attrs->hostPointer   = const_cast<void *>(ptr);
    return kvmem_vk_success;
}

// --- errors -----------------------------------------------------------------

cudaError_t cudaGetLastError(void) {
    return kvmem_vk_success;
}

const char * cudaGetErrorString(cudaError_t e) {
    switch (e) {
        case kvmem_vk_success:             return "no error";
        case kvmem_vk_error_invalid_value: return "invalid value";
        case kvmem_vk_error_out_of_memory: return "out of memory";
        case kvmem_vk_error_not_initialized: return "initialization error";
        default:                           return "unknown error";
    }
}

// --- stage-in TU hooks ------------------------------------------------------
// The stage-in kernels need the raw Vulkan handles to build their own
// pipelines on the same device and queue. The declarations are in
// llama-kvmem-vulkan-hooks.h; the definitions follow.

VkInstance    kvmem_vk_instance()            { return state().instance; }
VkPhysicalDevice kvmem_vk_physical_device()  { return state().physical; }
VkDevice      kvmem_vk_device()               { return state().device; }
VkQueue       kvmem_vk_queue()                { return state().queue; }
uint32_t      kvmem_vk_queue_family()         { return state().family; }
VkCommandPool kvmem_vk_command_pool()         { return state().pool; }

VkDeviceMemory kvmem_vk_device_memory(void * host_ptr) {
    auto & s = state();
    auto it = s.memories.find(host_ptr);
    return it == s.memories.end() ? VK_NULL_HANDLE : it->second;
}

VkBuffer kvmem_vk_buffer_of(void * host_ptr) {
    auto & s = state();
    auto it = s.buffers.find(host_ptr);
    return it == s.buffers.end() ? VK_NULL_HANDLE : it->second;
}

bool kvmem_vk_is_host(void * host_ptr) {
    auto & s = state();
    auto it = s.host.find(host_ptr);
    return it != s.host.end() && it->second;
}

// Returns the buffer's device address. The stage-in shaders use this to convert
// a host visible pointer into the 32 bit offset their SSBO view is bound at.
uint64_t kvmem_vk_buffer_address(void * host_ptr) {
    VkBuffer b = kvmem_vk_buffer_of(host_ptr);
    if (b == VK_NULL_HANDLE) {
        return 0;
    }
    DeviceState & s = state();
    VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    info.buffer = b;
    return vkGetBufferDeviceAddress(s.device, &info);
}

void kvmem_vk_memory_barrier(VkCommandBuffer cb) {
    // The producer is usually the host, not a transfer: the adapter memcpy's
    // into host visible memory and then submits. A barrier whose source stage
    // only covers TRANSFER would leave that host write unordered against the
    // shader read, so HOST has to be in the source stage.
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT  | VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cb,
                         VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, nullptr, 0, nullptr);
}
// --- ggml-cuda.h symbols the Vulkan backend must still provide ---------------
//
// The GDN replay code calls ggml_backend_cuda_gdn_fold, which upstream declares
// in ggml-cuda.h and the CUDA backend implements. ggml-cuda.h is still included
// by the adapter on every backend (it carries the descriptor struct), so the
// symbol has to resolve somewhere. On Vulkan it is implemented by the stage-in
// engine's GDN fold pipeline; see llama-kvmem-stagein-vulkan.cpp for why that is
// a real implementation rather than a stub, and for the degradation behaviour
// when the engine is unavailable.
bool kvmem_backend_gdn_fold(const struct ggml_cuda_gdn_replay_layer * layers,
                            int n_layers, int n_keep, int capacity);

extern "C" bool ggml_backend_cuda_gdn_fold(
        const struct ggml_cuda_gdn_replay_layer * layers,
        int n_layers, int n_keep, int capacity, void * stream) {
    (void) stream;   // the Vulkan path submits and waits on its own queue
    return kvmem_backend_gdn_fold(layers, n_layers, n_keep, capacity);
}

bool kvmem_vk_ptr_is_host_readable(const void * ptr) {
    auto & s = state();
    std::lock_guard<std::mutex> lk(lock());
    return s.host_accessible(ptr, 1);
}

bool kvmem_vk_ptr_is_host_writable(const void * ptr) {
    auto & s = state();
    std::lock_guard<std::mutex> lk(lock());
    return s.host_accessible(ptr, 1);
}
