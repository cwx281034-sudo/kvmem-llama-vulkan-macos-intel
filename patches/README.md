# llama.cpp patch replay

`llama-kvmem-current.patch` is the cumulative diff against pinned
`7fe450e19305b828c199d602c23a8337aaa1f03b` (`llama.cpp : bump version to 0.5.0`,
2026-09-23). It was previously based on `b81c99b`; it was rebased forward with
`git cherry-pick` so the KVMem hooks track the newer upstream tree instead of
pinning an older one.

It includes the existing KVMem hooks, multimodal batch, MTP, media
parser and mtmd helper extensions, plus FP32 GDN Record/Fold for ReplaySSM.
It also fixes reasoning-budget initialization from a template's generation prefix.
`scripts/apply-patches.sh` applies it
without creating commits and checks for an already applied tree.

Four hunks needed rebase context rewritten against the newer upstream:

- `common/speculative.cpp`: upstream renamed the draft field `n_past` to `pos0`,
  so the `batch.logical_pos[...]` assignments now read `dp.pos0`.
- `src/llama-context.cpp`: upstream added the `gf_res_prev_active == res` guard to
  graph reuse. The KVMem `llama_kvmem_capture_can_reuse()` condition is kept as an
  additional clause alongside it, not in place of it.
- `src/models/qwen35.cpp`: upstream replaced the separate Q/K/V projections with a
  joint `build_qkv()` call that already yields `Kcur` and `Vcur`, so the patch's
  `build_lora_mm` block for those two is obsolete. Only `kvmem_capture_q()` is kept.
- `tools/server/server-common.cpp`: upstream rewrote `handle_media()` to detect the
  `data:` media type from the URI itself, which retired the patch's per-call
  `accept_base64_uri` flag. The `oaicompat_chat_process_media()` split the patch
  introduces is preserved, and its `handle_media()` calls use the 3-argument form.

`0005-hip-rdna2-quantized-kv-fa-vec.patch` is @zintown's PR #58 RDNA2
quantized-KV Flash Attention dispatch fix. It selects the existing VEC kernel
for supported dimensions, avoiding the larger tile kernel's zero-occupancy
assertion. It is applied after the cumulative patch by both Linux and Windows
build entry points. F16 draft KV is not covered by this fix.

`reasoning-budget-upgrade.patch` upgrades the v0.15.0 ReplaySSM tree.
`replayssm-upgrade.patch` upgrades the preceding multimodal/query-replay tree.
`multimodal-upgrade.patch` upgrades the KVMem working tree recorded before
the 2026-09-14 implementation to the same current code. The script checks applicability before
changing files. Unrelated local changes are preserved; conflicting changes
require review.

The numbered `0001` through `0004` files are historical patches, retained for
reference. They are superseded by the cumulative diff: the old series did
not cleanly replay on the current pin and must not be applied together with it.

To check a clean extraction without changing the active submodule:

```bash
mkdir -p /tmp/kvmem-llama-patch-check
git -C llama.cpp archive 7fe450e19305b828c199d602c23a8337aaa1f03b | tar -x -C /tmp/kvmem-llama-patch-check
KVMEM_LLAMA_DIR=/tmp/kvmem-llama-patch-check scripts/apply-patches.sh
KVMEM_LLAMA_DIR=/tmp/kvmem-llama-patch-check scripts/apply-patches.sh
```

Note: `llama.cpp/src/CMakeLists.txt` hardcodes `UNITY_BUILD ON` upstream, so
CMake absorbs the adapter into a `Unity/unity_*_cxx.cxx` translation unit. The
per-source object names (`llama-memory-kvmem.cpp.o`) only appear in
`build.ninja` when that property is turned off; the sources are compiled either
way.
