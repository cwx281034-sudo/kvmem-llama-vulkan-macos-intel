# Vulkan KVMem 崩溃诊断（2026-10-03）

## 现象

`llama-kvmem-cli --kvmem --device Vulkan0` 在首次 decode 阶段 SIGSEGV（exit 139）。
不带 `--kvmem` 时 2B/3.5B/4B/8B 模型均正常推理，说明 Vulkan 后端与模型本身无问题。

崩溃报告（`~/Library/Logs/DiagnosticReports/llama-kvmem-cli-*.ips`）：

```
EXC_BAD_ACCESS / SIGSEGV / KERN_INVALID_ADDRESS at 0x0000000000001000
#0 _platform_memmove
#1 cudaMemcpyAsync(void*, void const*, unsigned long, int, kvmem_vk_stream*)
#2 kvmem_copy_async(void*, void const*, unsigned long, int, kvmem_vk_stream*)
#3 llama_memory_kvmem::d2h_submit(ggml_backend*)
#4 llama_memory_kvmem::harvest_pending(ggml_backend_sched*)
#5 llama_context::process_ubatch(...)
#6 llama_context::decode(...)
```

## 根因

`src/adapter/llama-kvmem-runtime-vulkan.cpp` 的 `cudaMemcpyAsync` 用
`s.buffers.find(ptr)` 判断一个指针是不是本 facade 分配的 `VkBuffer`：

- 命中注册表 → 走 `vkCmdCopyBuffer`；
- 未命中 → 走 `std::memmove(dst, src, n)`。

这个"注册表兜底"的假设在 CUDA 上成立，在 Vulkan 上不成立。

`d2h_submit()` 里被复制的**源**是 ggml 张量指针（`n.t->data`，见
llama-memory-kvmem.cpp:3011），它属于 ggml Vulkan 后端自己的
`VkBuffer`，**从来不在 facade 的注册表里**。因此每个这类源都被误判成
"两边都是普通主机内存"，于是执行 `memmove` 把一张 Vulkan 显存设备的
KGUF KV 张量当成主机内存拷贝。

更糟的是 `s.gpu`（`cudaMalloc` 得到的暂存区）**确实**在注册表里，于是
`src_is_buffer=false, dst_is_buffer=true` 的分支本来该被正确识别，
但源码在 line 526 先做了 `if (!src_is_buffer && !dst_is_buffer)` 判断——
当 src 是 ggml 指针、dst 是 staging 时会走 `vkCmdCopyBuffer`，可
`alloc_buffer` 对 `is_host=true` 的分配同样注册进 `s.buffers`。

实测崩溃点是 `_platform_memmove`，即走了 line 527 的
`std::memmove(dst, src, n)` 分支，说明 src 与 dst 都未命中注册表：
`src` 是 ggml Vulkan 设备指针（预期），`dst` 是 `s.pin`
（`cudaMallocHost` 的 pinned host 区）——而 `d2h_->slots[].pin`
经由 `cudaMallocHost` 分配后被释放/重建时指针可能落在 0x1000 附近的
未映射页，说明 `s.cap < bytes` 的重建路径上 `cudaMallocHost` 实际返回了
一个被本 facade 判为"不可映射就当作失败"的分配，或者 `s.gpu`
分配失败后仍继续使用。

无论具体是哪一个指针，**设计缺陷是同一个**：

> facade 把"未注册指针"等同于"可被主机直接读写的内存"，
> 而在 Vulkan 上 ggml 的设备缓冲区完全可能不可主机寻址。

## 修复方向

facade 必须区分三类指针，不能用"在不在注册表"二分：

1. **facade 自己分配的**（`s.buffers` 命中）→ 真 `VkBuffer`，走 `vkCmdCopyBuffer`；
2. **主机可寻址的**（ggml Vulkan 后端的设备缓冲区是
   HOST_VISIBLE|HOST_COHERENT，见 llama-kvmem-stagein-vulkan.cpp:1076-1080
   的既有注释与用法）→ 可以 `memmove`，但必须先校验指针确实落在
   一个已映射的 Vulkan 内存区间内；
3. **既非 facade 分配、又不可主机寻址** → 返回错误，绝不 memmove。

配套问题（同一批修复）：

- `kvmem-server-devices.h:88` 的多 GPU 探测把 `IGPU` 之外的设备全算作
  `discrete`。macOS 上 Metal + Vulkan 各注册一块同物理 GPU，
  `props.device_id` 去重只处理 `discrete` 分支，导致 `count==2`，
  触发 "select multi-GPU explicitly with --device CUDA0,CUDA1" 而拒绝启动。
  这条报错信息里的 CUDA 字样本身就是 CUDA 假设的泄漏。
