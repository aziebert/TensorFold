# Volta (sm_70) port — MLX checkpoints on Volta NVIDIA GPUs

This fork adds a native CUDA path for NVIDIA **Volta** (compute capability 7.0,
e.g. the Quadro GV100). Volta is two generations behind the sm_80+ paths this
codebase targets: no bf16 or fp8 tensor ops, no `wgmma`, and exactly one
tensor-core shape (`mma.m8n8k4`). Everything here is built around that
constraint, and every kernel is validated on a GV100.

Validated on a **Quadro GV100 (32 GB)**, CUDA 12.4, with MLX 4-bit and NVFP4
checkpoints of Qwen3.8-27B and Nemotron 3.5.

## Build

Zig 0.17.0 (the repo's pinned toolchain) and nvcc from CUDA 12.4:

```sh
zig build -Dnvcc=/usr/bin/nvcc -Dsm=70
```

The production binary for a GV100 is **always `-Dsm=70` only**. A multi-arch
build (`-Dsm=70,80,86,89`) compiles clean but does not run on Volta: kernels
gated for sm_80+ emit fatbins with no sm_70 cubin, and the eager loader
hard-fails the whole module (`CUDA_ERROR_NO_BINARY_FOR_GPU`, 209) instead of
skipping them. Build multi-arch binaries for newer GPUs only.

## What is in this fork

Each capability is isolated behind the `zig/src/cuda/volta.zig` dispatch
(Lane) and the `zig/kernels/cuda/*_volta.cu` sources, so the pieces can be
lifted into other engines:

| Capability | Where | Notes |
|---|---|---|
| bf16 dense GEMM | `zig/kernels/cuda/volta_qmm.cu` (`tf_vqmm884_{xs}_{y}_{z}`) | Exact `code·scale + bias` per 64-group in fp16 on `mma.m8n8k4` (Volta's only tensor-core shape), fp32 accumulation, bf16 in/out. `y` selects the weight layout (stored MLX column-runs-of-K/8 vs tiled `[N/32][K/64][32 cols][8 words]`), `z` selects f32/bf16 out. Split-K partials reduce **in-kernel** via atomic CNT counters — do not launch a separate reduce kernel. |
| MLX 4-bit (affine4) prefill/decode | `zig/src/families/qwen35/cuda_engine.zig` (`loadVolta`/`voltaA4`, dispatched in `project()`) | Dense affine4 matmuls route to the Volta lane on `major == 7`. **Layout rule:** the lane reads the *raw stored* safetensors layout (W `[N][K/8]` words, S/B `[N][K/64]` bf16 group-major). Skip `pack4` on Volta — feeding sm_80-style packed tiles to the kernel is a *silent* failure (exit 0, finite logits, flat distribution, garbage text). |
| NVFP4 MoE experts | `zig/kernels/cuda/experts.cuh` + host-side quantizer | e2m1 FP4 codes (8 per 32-bit word) with e4m3 per-group scales, dequant `value = e2m1 · 2⁻¹⁴ · e4m3`. `mma.m16n8k16` does not exist on Volta, so the MMA is **software-emulated** (shuffle-gather + fp32 dot). X tiles are `Xt[tile][kcol8][row][col]`, kcol8-major; W is `[n·8 + k/8]` nibble words; S is `[n]` 4-byte e4m3 groups. Output is bf16. |
| Attention + draft on sm_70 | Nemotron family wiring | Attention and draft routed to the sm_70 kernels, including the FP8→bf16 dequant fix. |
| MTP drafter (Qwen3.8-27B) | `cuda_mtp.zig`, `cuda_mtp_weights.zig` | The standalone MLX 4-bit MTP checkpoint loads on the dense engine: dequantize-to-bf16 at load and route through the bf16 path (`onVoltaBf16`), plus the MoE variant (`mtpExperts`). 4-bit grouped experts have no native sm_70 path, so bf16-dequant is the supported route for the 4-bit MoE head. |
| CUDA graph capture on Volta | persistent decode-window scratch pool | A lazy `cuMemAlloc` inside a capture window is illegal (`STREAM_CAPTURE_UNSUPPORTED`, 900). The decode window uses a persistent scratch pool, and **every** separately-initialized component (main window, then the MTP head) is pre-warmed at its widest row count before its record window — warming one component does not cover another. |

## Validation

- `zig/tests/cuda/volta_qmm_test.cpp` — **263 checks, all green** on the GV100
  (the compiled `volta_attn_test` and `volta_qmm_test` are kept in-tree).
- Full kernel suite validated (exit 0): attention, qmm, qmmf (matmul F16 + FP4
  e2m1 dequant with GPU==CPU **maxrel 0.0038**, reduce, unscale2, gather_rows),
  experts, draft.
- End-to-end: eager and graph-captured generation, no-drafts and with MTP,
  producing coherent output (Qwen3.8-27B greedy decode verified).

> **Semantic check, not exit codes.** For quantized-GEMM wiring, "no launch
> error + finite logits" is a false positive — it passed both while producing
> garbage. The real check is a short greedy decode: coherent text with a
> peaked top-1 (≈0.6 here). A flat top-k (top-1 softmax < 0.1) is the garbage
> signature.

## Benchmarks — honest numbers

Qwen3.8-27B, 128K context, q8_0 KV cache, 128-token greedy cells, GV100 32 GB.
Both engines run in the same session.

**Decode tok/s (p0 / p100 / p400):**

| Engine | MTP off | MTP on |
|---|---|---|
| TensorFold (MLX 4-bit, sm_70 build) | 10.5 / 10.5 / 10.5 | 15.0 / 17.3 / 36.7 |
| llama.cpp (Q4_K_M, full offload, FA, `--spec-type draft-mtp -n-max 4`) | 28.4 / 28.3 / 28.6 | 49.3 / 36.6 / 61.2 |

**Prefill tok/s (llama.cpp):** p0 ≈ 100, p100 ≈ 129–172, p400 ≈ 238–293.
(Note: the two engines' tokenizers differ — the "400" prompt is 805 tokens in
TensorFold vs 643 in llama.cpp, so prefill cells are comparable in spirit
only.)

**VRAM:** llama.cpp full config 24.3/32 GB (16.5 GB model + 128K q8 KV +
buffers).

**Verdict, no dressing:** llama.cpp is roughly **2.7–4× faster** in decode on
this hardware, and wins prefill decisively. It is the right engine to run on a
GV100 today. TensorFold's MTP decode does scale with context (15.0 → 36.7 tok/s
across the cells), but it does not close the gap.

What this port still buys you:

1. **MLX checkpoints run directly on Volta** — no re-quantization to GGUF.
2. A validated **sm_70 kernel + launch-config reference** (263-check suite,
   in-kernel reduction, exact layouts above) for anyone else targeting Volta.
3. The graph-capture + MTP machinery, reusable by newer-GPU users.

## Limitations

- No bf16/fp8 tensor ops on Volta — the fp16-accumulate path carries bf16 I/O.
- 4-bit grouped experts fall back to bf16 dequant at load (no native
  grouped-affine4 on sm_70).
- Production binary is sm_70-only (multi-arch builds do not run on Volta).
- Speed: see the verdict above.

## Credits

- **@gilby** — thank you. The CUDA work in the
  [gilby/TensorFold](https://github.com/gilby/TensorFold) fork
  (`cuda-engine`, `cuda-dspark`, `ampere-sm80`, `b16-mma` branches) and the
  upstream pull requests (#360 dense Gemma 4 decode, #148, #68, #147/#149/#164
  Apple Silicon) laid the ground this port stands on.
- @ashhart and the TensorFold maintainers — the engine, the contributor skill,
  and the in-tree test culture this port leaned on.
- The port itself was developed in ~3 continuous days with
  [Hermes Agent](https://hermes-agent.nousresearch.com) (Nous Research) — the
  full story, including the bugs that cost the most, is in
  [volta-sm70-story.md](volta-sm70-story.md).
