# macOS Intel Vulkan Release

This repository packages the KVMem fork with a Vulkan/MoltenVK build for Intel macOS.

## Upstream references

- KVMem project: https://github.com/kvmem/kvmem-llama.cpp
- llama.cpp upstream: https://github.com/ggml-org/llama.cpp
- Embedded llama.cpp baseline: `7fe450e19` (`llama.cpp : bump version to 0.5.0`)
- Source version: `0.17.0`

The `llama.cpp` directory is tracked as an upstream gitlink. KVMem integration changes are kept in `patches/`, including `patches/llama-kvmem-current.patch` and the Vulkan adapter sources under `src/adapter/`.

## Standalone package

The release archive is:

`artifacts/kvmem-llama-vulkan-macos-standalone.tar.gz`

It contains `llama-kvmem-server`, `llama-kvmem-cli`, KVMem/llama/ggml dynamic libraries, MoltenVK, OpenSSL runtime libraries, startup scripts, licenses, provenance and SHA256 checksums. GGUF model files are intentionally not included.

Verify the archive with:

```bash
sha256sum -c artifacts/kvmem-llama-vulkan-macos-standalone.tar.gz.sha256
```

## Example

```bash
./run-server.sh \
  -m /path/to/Ling-3.0-tiny-Q3_K_S.gguf \
  --device Vulkan0 --kvmem \
  -ngl 8 -c 262144 \
  --kvmem-budget 2048 \
  --kvmem-block-tokens 128 \
  --kvmem-gen-reserve 4096 \
  --kvmem-cpu-gb 32 \
  -b 64 -ub 32 --host 127.0.0.1 --port 8080
```

The tested target is macOS Intel with Vulkan through MoltenVK. The package does not claim portability to Apple Silicon or other operating systems.
