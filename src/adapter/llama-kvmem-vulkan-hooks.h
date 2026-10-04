#pragma once

// Handles shared between the Vulkan runtime facade and the stage-in engine.
//
// llama-kvmem-runtime-vulkan.cpp owns the device, queue, command pool and the
// allocation registry. The stage-in engine (llama-kvmem-stagein-vulkan.cpp) needs
// those handles to build its compute pipelines on the same device and queue, so
// they are declared here instead of as loose externs in each translation unit.
//
// This header is Vulkan-backend internal: it is not part of the CUDA spelling the
// adapter speaks, and nothing outside src/adapter should include it.

#include <vulkan/vulkan.h>

#include <cstdint>

// Brings the shared device up and makes cudaStreamPerThread a real handle. Call
// once before anything else on this backend.
void kvmem_vk_facade_init();

VkInstance   kvmem_vk_instance();
VkPhysicalDevice kvmem_vk_physical_device();
VkDevice     kvmem_vk_device();
VkQueue      kvmem_vk_queue();
uint32_t     kvmem_vk_queue_family();
VkCommandPool kvmem_vk_command_pool();

// Allocation registry lookups. Both return a null handle for a pointer the
// facade does not own, which includes any ggml tensor buffer.
VkBuffer       kvmem_vk_buffer_of(void * host_ptr);
VkDeviceMemory kvmem_vk_device_memory(void * host_ptr);
bool           kvmem_vk_is_host(void * host_ptr);

// Buffer device address, or 0 when unavailable.
uint64_t kvmem_vk_buffer_address(void * host_ptr);

// Records a memory barrier making prior host writes to host visible memory
// visible to a shader that reads it.
void kvmem_vk_memory_barrier(VkCommandBuffer cb);