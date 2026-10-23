# Three days on Volta — how this port happened

> A development log for the curious. This port ran as a continuous loop: a
> human setting the goals and reviewing (Alexandre Ziebert), and an AI agent —
> [Hermes Agent](https://hermes-agent.nousresearch.com) (Nous Research) —
> driving the terminal, the editor and a Quadro GV100, from the first commit
> on Oct 21 to the last on Oct 23, 2026, with almost no breaks. It ended with
> a working engine that is *not* the fastest on the box. That is fine — here
> is the whole story anyway.

## The premise

[TensorFold](https://github.com/ashhart/TensorFold) is a Zig binary that serves
MLX checkpoints natively — Apple Silicon (Metal) and NVIDIA (CUDA), no Python,
no MLX install. The CUDA path targets sm_80 and up (Ampere and newer). A
Quadro GV100 is **sm_70** — Volta, 2018: no bf16 tensor cores, no FP8, no
`wgmma`, 48 KB of static shared memory per block, and exactly one tensor-core
primitive, `mma.m8n8k4` with fp16 inputs.

The question: can a 27B MoE model with NVFP4 (4-bit) experts, attention and an
MTP draft head actually run on a 2018 GPU through a runtime written for the
MLX checkpoint format — and can an agent do that port end-to-end, with
verification, in a few days?

## Day 1 — the lane (Oct 21)

The first commit of the branch is unglamorous on purpose:
`volta_qmm: pass the full suite on a GV100 (263 checks)`.

The architecture decision early on: one dispatch type per kernel family
(`zig/src/cuda/volta.zig`, the "Lane"), pure-CPU `plan(M,N,K)` functions, one
launch method per kernel entry, and fatbins embedded at compile time via
`@embedFile` + `cuModuleLoadData`. The consumer side dispatches on device
capability in `qlinear.zig` — families that don't set a lane field skip the
Volta path automatically, so nothing breaks on other GPUs.

The contract: the C++ reference test (`zig/tests/cuda/volta_qmm_test.cpp`) is
the truth. Every Zig launch config (grid, block, argument order) is copied
**verbatim** from the validated C++ test, never derived from the Python
reference logic. That discipline paid off immediately — the 263-check qmm
suite (4-bit affine matmul: `code·scale + bias` per 64-group, exact fp16 on
`mma.m8n8k4`, fp32 accumulation) went green, and the first coherent decode of a
27B model on Volta happened that night.

## Day 2 — MoE, FP4, and the silent-garbage bug (Oct 23, early hours)

NVFP4 grouped experts next. The formats: e2m1 FP4 codes packed 8-per-32-bit
word, e4m3 per-group scales, dequant `value = e2m1 · 2⁻¹⁴ · e4m3`. The hard
part: the reference code emits `mma.m16n8k16`, which Volta simply does not
have. Solution: a **software emulation** of that MMA — shuffle-gather of the
fragments plus fp32 dot products in the registers (`experts.cuh`). Slow, but
exact: GPU vs CPU **maxrel 0.0038** across the whole experts kernel. A
host-side quantizer packs the weights into the layout the kernel expects
(`W[n·8 + k/8]` nibble words, `S[n]` 4-byte e4m3 groups).

Then the bug of the project. The standard 4-bit dense path packs weights into
sm_80 tile format (`pack4`). When the Volta path started feeding those packed
tiles to `tf_vqmm884`, **everything passed** — exit 0, no launch errors, all
logits finite. The output was coherent-*looking* garbage. The tell was a flat
top-k distribution (top-1 softmax under 0.1). The kernel expects the *raw
stored* MLX layout (W `[N][K/8]`, S/B `[N][K/64]` group-major) — the fix lived
in the weight loader (skip `pack4`, upload raw parts), not in the kernel.

The rule ever since: **verify semantics, not exit codes.** A short greedy
decode with a peaked top-1 (≈0.6 on "The capital of France is Paris…") is the
pass; a flat distribution is the fail. Same family of lesson: an undersized
intermediate buffer passed small shapes and only broke at production shapes —
the signature is "only the leading K groups/columns are correct, the rest
zero". Re-derive every scratch size from the kernel's own fetch arithmetic
before trusting a launch.

Also on day 2: attention and draft for the Nemotron family routed to the sm_70
kernels, with an FP8→bf16 dequant fix along the way.

## Day 3 — MTP, graph capture, and the honest benchmark (Oct 23)

The MTP (multi-token-prediction) head is a standalone MLX 4-bit sub-network.
Two problems: dense MLPs were gated to MoE engines, and 4-bit *grouped*
experts have no native sm_70 path. Both solved the pragmatic way: drop the
gate, dequantize-to-bf16 at load, and route through the validated bf16 path
(`onVoltaBf16`) — plus the MoE NVFP4 variant (`mtpExperts`), validated the same
way.

CUDA graph capture was the last real blocker. A lazy `cuMemAlloc` inside a
capture window is illegal — it fails with `STREAM_CAPTURE_UNSUPPORTED` (900),
not a silent miscompute. Two fixes: a **persistent decode-window scratch pool**
(allocations outlive capture), and pre-warming. The subtle part: warming the
main decode window does **not** cover the MTP head — every separately
initialized component needs its own warmup at its widest row count, before its
record window. (The warmup data can be garbage; the replayed graph reads what
the caller writes at replay — only the allocations matter.)

With eager and graph-captured generation both coherent, came the benchmark —
the part nobody wants to write. A 12-cell matrix (p0/p100/p400 prefill × MTP
off/on) against **llama.cpp built from source** for Volta (gcc-13, CUDA 12.4,
arch `70-real`, flash-attention quant pairs; MTP via `--spec-type
draft-mtp`, which builds the draft context against the target model's own
weights — the `-md <same gguf>` variant loads a full second copy and OOMs on
32 GB).

The numbers (full table in [volta-sm70.md](volta-sm70.md)):

| Engine | MTP off | MTP on |
|---|---|---|
| TensorFold (MLX 4-bit) | 10.5 | 15.0–36.7 |
| llama.cpp (Q4_K_M, FA) | 28.4–28.6 | 49.3–61.2 |

**llama.cpp wins by 2.7–4×.** The report says so plainly, because a
benchmark whose verdict is hidden isn't a benchmark. What the port still
delivers: MLX checkpoints run directly on Volta with no re-quantization, a
validated sm_70 kernel reference (the 263-check suite and the exact layouts
documented in the feature guide), and the graph-capture/MTP machinery.

## What I'd do differently

- **Check shared memory first.** Two different `CUDA_ERROR_LAUNCH_FAILED` (719)
  faults looked identical: one was the sm_80 GEMM's ~62 KB static shared
  memory exceeding Volta's 48 KB without opt-in (deterministic, at launch —
  settle it by sizing, don't debug the kernel); the other was an async out-of-
  bounds that only surfaces at a later `streamSynchronize` (needs
  `CUDA_LAUNCH_BLOCKING=1` + the sanitizer before anyone suspects the GPU).
- **Never let a multi-arch binary reach a GV100.** It compiles clean and dies
  at `cuModuleLoadData` with `NO_BINARY_FOR_GPU` (209). Production binary:
  `-Dsm=70` only.
- **Trust the in-tree C++ test over everything else** — the Python reference
  logic is for planning, the kernel source and the C++ launch config are for
  launching.
- **Treat non-monotonic single-cell dips (< 20%) as graph/warmup noise** until
  repeated — it saved a day of chasing a ghost in the p100 MTP cell.

## Thanks

- **@gilby** — the CUDA work in the
  [gilby/TensorFold](https://github.com/gilby/TensorFold) fork and the upstream
  pull requests (#360, #148, #68, #147/#149/#164) were the ground this port
  stood on. Thank you.
- @ashhart and the TensorFold maintainers — the engine, the contributor skill,
  and a test culture that made an agent-run port verifiable instead of
  vibes-based.
- And yes — an honest footnote: on this hardware, run llama.cpp. The port's
  value is the direct MLX checkpoint path, the validated sm_70 reference, and
  three days of hard-won lessons now written down here.
