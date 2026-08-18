# LLM Inference Terminology: Kernels, Quantization, and Hardware

*A working reference for self-hosted vLLM deployment. Written against evidence gathered August 2026. Kernel support status changes fast — re-verify against your pinned vLLM version.*

---

## 1. The stack, top to bottom

```
Model weights (safetensors on disk)
        ↓
vLLM / SGLang        — orchestration: batching, scheduling, KV cache,
                        and DECIDING WHICH KERNEL TO CALL
        ↓
Kernel libraries     — CUTLASS / Marlin / Triton / FlashInfer / DeepGEMM / cuBLAS
        ↓
PTX → SASS           — compilation chain (this is where builds break)
        ↓
GPU                  — tensor cores, memory bandwidth
```

A **kernel** is a function that executes on the GPU. Everything above it is bookkeeping.

Almost all LLM compute is one operation: **GEMM** (General Matrix Multiply). The kernel libraries below are competing implementations of that same matrix multiply, each tuned for a specific combination of *hardware generation* and *number format*. At startup, vLLM inspects your GPU's compute capability and your checkpoint's quantization config, then selects one.

When a log line says "falls back to Marlin," that is vLLM's selection logic choosing a different implementation than the preferred fast path — usually because no fast path exists for your architecture.

---

## 2. The notation that unlocks everything

**W8A8**, **W4A16** — Weights and Activations, with their bit widths.

| Notation | Weights | Activations | What actually happens |
|---|---|---|---|
| **W16A16** | 16-bit | 16-bit | BF16/FP16 baseline. No quantization. |
| **W8A8** | 8-bit | 8-bit | The matmul executes in 8-bit on FP8 tensor cores. Genuinely 8-bit compute. |
| **W4A16** | 4-bit | 16-bit | Weights *stored* 4-bit, **dequantized to BF16 in registers**, matmul runs in BF16. |
| **W4A8** | 4-bit | 8-bit | Rare. Requires Machete/CUTLASS — **SM90 only**. |

### Why W4A16 is the single most important concept here

W4A16 is called **weight-only quantization**. It gives you the memory saving *without requiring 4-bit compute hardware*.

- You read **a quarter of the bytes** from GPU memory — this is why 4-bit decodes faster than FP8 even with no FP4 tensor cores.
- You **forfeit the 4-bit math throughput** — this is why it does not help prefill.

Nearly every "fallback" you will encounter is a native format degrading to W4A16.

---

## 3. Quantization formats

### The number formats

| Format | Bits | Type | Value set |
|---|---|---|---|
| **BF16** | 16 | float | Full range, the reference |
| **FP8 (E4M3)** | 8 | float | 4 exponent, 3 mantissa bits |
| **FP8 (E5M2)** | 8 | float | 5 exponent, 2 mantissa — wider range, less precision |
| **INT8** | 8 | integer | −128 to +127, uniformly spaced |
| **INT4** | 4 | integer | −8 to +7, **uniformly spaced** |
| **FP4 (E2M1)** | 4 | float | **Non-uniformly spaced** (see below) |

### INT4 vs FP4 — not the same thing

Both are 4 bits. They differ fundamentally.

**INT4** is a 4-bit signed integer: 16 evenly spaced values. Dequantization is `w = scale × (q − zero)` — unpack a nibble, convert integer to float, multiply. Integer→float conversion is universal, available on every CUDA GPU since Kepler.

**FP4 (E2M1)** is a 4-bit *float*: 1 sign bit, 2 exponent bits, 1 mantissa bit. Its 16 values are non-uniformly spaced:

```
0, ±0.5, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0
```

The non-uniform spacing is the point — dense resolution near zero where weight distributions concentrate, coarse in the tails. In theory this makes FP4 more accurate than INT4 at equal bit width.

**The catch:** interpreting an E2M1 bit pattern as a float is *not* a standard operation. NVIDIA added dedicated instructions on Blackwell — `cvt.e2m1x2.f32` and `cvt.rn.satfinite.e2m1x2.f32` — and gated them behind specific compilation targets. This is the root of most FP4 deployment failures (see §5).

### FP4 sub-formats

Both use E2M1 elements; they differ in scaling metadata.

| | Block size | Scale format | Notes |
|---|---|---|---|
| **NVFP4** | 16 elements | FP8 E4M3 + global FP32 | NVIDIA's format. Finer blocks, higher-precision scales. |
| **MXFP4** | 32 elements | E8M0 (power-of-two only) | OCP microscaling standard. Used by gpt-oss. |

NVFP4's finer blocks and richer scales generally make it more accurate than MXFP4.

### Quantization algorithms (distinct from formats)

The *format* is how bits are laid out. The *algorithm* is how you choose the values.

| Algorithm | What it does |
|---|---|
| **RTN** (round-to-nearest) | Naive. Just round. Baseline quality. |
| **GPTQ** | Layer-wise second-order error compensation. Widely used. |
| **AWQ** | Activation-aware — identifies salient weight channels via activation statistics and protects them. |
| **ModelOpt PTQ** | NVIDIA's toolkit. Produces NVFP4 checkpoints. |
| **llm-compressor** | Red Hat / Neural Magic toolkit, maintained under the vLLM project org. Produces compressed-tensors checkpoints. |

**A good algorithm on a "worse" format can beat a mediocre one on a "better" format.** Measured on RTX PRO 6000: AWQ-Int4 achieved **KLD 0.024** versus NVFP4's **0.035** (lower = closer to the unquantized reference), while also being faster. Theory favored NVFP4; measurement favored AWQ-Int4.

---

## 4. Kernel libraries

| Name | What it is | Availability |
|---|---|---|
| **cuBLAS** | NVIDIA's closed-source classic BLAS. Safe default. | Everywhere |
| **CUTLASS** | NVIDIA's open-source C++ GEMM template library. Hand-tuned per architecture. The reference fast path. | Arch-specific — separate code paths per SM90 / SM100 / SM120 |
| **Marlin** | A specific **mixed-precision W4A16 kernel** from IST-DASLab (the GPTQ authors). Overlaps dequantization with async memory loads. ~4× over FP16 at batch ≤32. | Broadly portable |
| **GPTQ-Marlin / AWQ-Marlin** | Marlin variants specialized for GPTQ and AWQ INT4 checkpoints. | SM80+ |
| **moe_wna16** | The MoE-specific W4A16 path (grouped/expert version of Marlin). | SM80+ |
| **Machete** | vLLM's Hopper-native successor to Marlin. Uses `wgmma` PTX instructions. | **SM90 only** — will not load elsewhere |
| **Triton** | OpenAI's Python-like DSL that JIT-compiles GPU kernels. Portable, autotuned. Usually slower than hand-tuned CUTLASS. | Anywhere — but pays a cold-start compile penalty |
| **DeepGEMM** | DeepSeek's fine-grained FP8 GEMM library. | SM90 / SM100 only — **no SM120 support** |
| **FlashInfer** | LLM-serving kernel library: attention, sampling, increasingly quantized MoE. JIT-based. | Where kernels exist for your arch |

### Marlin is not inherently a "fallback"

This is the most commonly confused point.

- For **INT4 weight-only**, Marlin is the **purpose-built, intended kernel**. That is what it was written for. It is fast.
- For **NVFP4/MXFP4 on hardware without native FP4**, Marlin is pressed into service for a format it was not designed for. It works and produces correct output, but it is a substitute.

On SM120 specifically, the "native" CUTLASS FP4 MoE path is *broken* (garbage output at 5–7 tok/s; CUTLASS issue #3096), so **Marlin is the fastest correct path** — measured 50.5 tok/s TP=4 on Qwen3.5-397B-NVFP4. There, forcing Marlin is the right call, not a compromise.

### Dense GEMM vs grouped GEMM (MoE)

A dense layer is one large matrix multiply.

An MoE layer routes each token to a few of many experts (e.g. 10 of 512), producing **many small matmuls of varying sizes** — a **grouped GEMM**. This is a completely separate kernel from the dense one.

This distinction matters enormously:

```
vLLM gate: cutlass_group_gemm_supported()
           accepts compute capability 90–109

SM89 (L40S)   → 89 < 90   → FAILS  → Triton fused_moe
SM90 (H100)   → 90        → PASSES → CUTLASS grouped GEMM
SM100 (B200)  → 100       → PASSES → CUTLASS grouped GEMM
SM120 (RTX PRO 6000) → 120 ≥ 110 → FAILS → Triton fused_moe
```

**Consequence:** "FP8 is native on my GPU" is only fully true for **dense** models on SM89 and SM120. On both, FP8 MoE experts fall to Triton.

---

## 5. Compilation — where builds break

```
CUDA C++  →  PTX  →  SASS
             ↑        ↑
      virtual ISA   real machine code
      (portable)    (per-architecture)
```

- **PTX** — portable intermediate assembly. Forward-compatible.
- **SASS** — actual machine code for one specific architecture.
- **cubin** — compiled SASS baked into a binary.

A container can ship either. If it ships PTX only, the driver JIT-compiles to SASS **on first kernel launch** — not at model load.

### SM numbers = compute capability

| SM | Architecture | Example GPU |
|---|---|---|
| SM70 | Volta | V100 |
| SM75 | Turing | T4 |
| SM80 | Ampere | A100 |
| SM86 | Ampere | A10G |
| SM89 | Ada Lovelace | L40S, L4 |
| SM90 | Hopper | H100, H200 |
| SM100 | Blackwell (datacenter) | B200, B300, GB200 |
| SM120 | Blackwell (workstation) | RTX PRO 6000, RTX 5090 |
| SM121 | Blackwell | GB10 (DGX Spark) |

**Critical:** SM120 is **not** "newer than SM100." Both are Blackwell. SM100 is the datacenter line; SM120 is the workstation/pro line. They are *different feature sets*, not successive generations. Assuming SM120 ⊇ SM100 is the source of most SM120 confusion.

This split is not new. Ampere shipped SM80 (A100, datacenter) alongside SM86 (A10G, consumer) — and SM86, despite the higher number, had worse FP64, GDDR6 instead of HBM2e, and no NVSwitch. Ada (SM89) had no datacenter part; Hopper (SM90) had no consumer part. The lines don't advance in lockstep, so **you cannot compare capability across generations by number**.

### What the silicon actually supports — from the vendor spec sheets

Kernel availability and hardware capability are constantly confused. The official specification tables settle the hardware half:

| | **L40S** (SM89) | **RTX PRO 6000 BSE** (SM120) |
|---|---|---|
| BF16 tensor | 362.05 TFLOPS | 1 PFLOP |
| FP8 tensor | 733 TFLOPS | 2 PFLOPS |
| **FP4 tensor** | **— no row exists —** | **4 PFLOPS** |
| INT8 tensor | 733 TOPS | not published |
| INT4 tensor | 733 TOPS | not published |
| Memory / bandwidth | 48 GB GDDR6 / 864 GB/s | 96 GB GDDR7 / 1,597 GB/s |

Sources: NVIDIA L40S specifications (https://www.nvidia.com/en-in/data-center/l40s/#specifications) and NVIDIA RTX PRO 6000 Blackwell Server Edition specifications (https://www.nvidia.com/en-in/data-center/rtx-pro-6000-blackwell-server-edition/#specs). L40S figures are dense; the page also lists with-sparsity doubles.

Three readings:

- **SM120 genuinely has FP4 tensor cores** — 4 PFLOPS, exactly 2× its FP8 and 4× its BF16. That clean doubling per precision halving is what real hardware support looks like. NVIDIA's product copy states plainly that 5th-gen Tensor Cores "add support for FP4 precision." Claims that SM120 lacks FP4 hardware are **wrong**; what it lacks is complete kernel coverage and default arch-target exposure.
- **SM89 genuinely does not.** The L40S table runs FP32 → TF32 → BF16 → FP16 → FP8 → INT8 → INT4 and stops. Ada predates the format. This is a real hardware ceiling.
- **SM89 has native INT4 at 733 TOPS**, though vLLM's W4A16 path dequantizes to BF16 and never uses it — which is precisely why INT4 is so portable across architectures.

### The `a` and `f` suffixes

Blackwell introduced additional compilation targets:

| Target | Meaning | Compatibility |
|---|---|---|
| `sm_120` | Baseline feature set | Forward-compatible |
| `sm_120a` | **Architecture-specific** — unlocks specialized tensor-core features | **Not** forward-compatible |
| `sm_120f` | **Family-specific** (new in CUDA 12.9) — superset of baseline | Forward-compatible within the 12.x family |

**The FP4 conversion instructions live in `a`/`f`, not in bare `sm_120`.**

### Anatomy of a real failure

Running gpt-oss-20b (MXFP4) on an AWS DLC image on g7e:

1. Checkpoint ships MXFP4 weights
2. vLLM sees SM120, finds no working native FP4 MoE kernel, selects **Marlin W4A16**
3. Marlin must dequantize FP4 → BF16, so it emits `cvt.e2m1x2.f32`
4. The container's PTX was compiled targeting bare **`sm_120`**
5. **Weights load successfully (1.92s)** — nothing FP4-specific has run yet
6. First MoE forward pass → driver JITs the PTX → instruction not valid for declared target → **`cudaErrorUnsupportedPtxVersion`**

The GPU physically *has* FP4 capability. The build declared a target that does not expose it. **A toolchain failure, not a hardware limitation.**

Fix: build with `TORCH_CUDA_ARCH_LIST=12.0f`, `CMAKE_CUDA_ARCHITECTURES=120f`, `FLASHINFER_CUDA_ARCH_LIST=12.0f` (single value — a multi-arch list has been reported to break FlashInfer import).

Verify: `cuobjdump -lelf <vllm>/_C.abi3.so | grep sm_120` — you want `sm_120a` or `sm_120f` cubins, not bare `sm_120`.

**INT4 never enters this chain.** Integer→float conversion needs no arch-gated instruction.

### Proof that FP4 dequant does not *require* the instruction

There are only 16 possible FP4 values. You can dequantize with a **16-entry lookup table** on any GPU ever made. The hardware `cvt` is an optimization, not a necessity.

NVIDIA's Nemotron-3.5-Lightning NVFP4 checkpoint ships a validated W4A16 recipe for **A100 (SM80)** — hardware that predates FP4 entirely. That dequant path provably cannot use Blackwell instructions. Portable FP4 dequant exists; not every kernel uses it.

---

## 6. Why decode and prefill behave differently

This governs every quantization decision.

| Phase | What happens | Bottleneck |
|---|---|---|
| **Prefill** | Process the whole input prompt at once | **Compute-bound** — thousands of tokens share one weight read; arithmetic throughput dominates |
| **Decode** | Generate one token at a time | **Memory-bandwidth-bound** — every weight streamed from VRAM per token; tensor cores mostly idle waiting |

### Measured: Qwen3.6-27B on RTX PRO 6000

| Format | Decode tok/s | Prefill tok/s |
|---|---|---|
| BF16 | 59 | 4,359 |
| FP8 | 97–100 | **4,747** |
| NVFP4 | **163–169** | 4,732 |

**Decode swings ~70% across formats. Prefill swings 0.3%.**

Quantization choice is effectively a **decode-only decision**.

### Why the slower-compute format wins decode

For a 27B dense model, bytes swept per token step:
- FP8 → ~29 GB
- INT4 → ~15 GB

At 1,597 GB/s that is roughly 18 ms vs 9 ms. The dequantization that 4-bit pays happens **in registers, hidden behind the memory stall**. Effectively free.

FP8's native tensor cores cannot help when the GPU is starved for weights.

### The formula that predicts decode speed

```
decode speed  ∝  1 / (active_params × bytes_per_param)
```

**Active** params, not total — this is why MoE models decode so fast.

| Model | Active | Bytes/param | Bytes/token | Relative decode |
|---|---|---|---|---|
| 27B dense FP8 | 27B | 1.0 | ~29 GB | Slowest |
| 27B dense INT4 | 27B | 0.5 | ~15 GB | ~2× |
| 122B MoE INT4 | 10B | 0.5 | ~5–8 GB | ~4–6× |
| 35B-A3B MoE INT4 | 3B | 0.5 | ~2 GB | ~15× |

**A 122B MoE decodes faster than a 27B dense.** Sparsity beats size.

### Where prefill still matters

Agent workloads re-send system prompt + tool schemas + history every turn, so prefill drives **TTFT**. But the fix is `--enable-prefix-caching`, not quantization — prefill is nearly format-insensitive.

---

## 7. KV cache

Alongside weights, you must store attention keys and values for every token in context.

```
KV bytes/token = 2 (K and V) × num_kv_heads × head_dim × num_full_attention_layers × dtype_bytes
```

### Worked example — Qwen3.5-397B-A17B (verified from config.json)

- 60 layers, but `layer_types` = 15 repetitions of `[linear, linear, linear, full_attention]`
- → only **15 full-attention layers**; the other 45 are Gated DeltaNet (linear attention)
- `num_key_value_heads` = 2, `head_dim` = 256

```
2 × 2 × 256 × 15 = 15,360 elements/token
→ ~15 KB/token at FP8
→ ~30 KB/token at BF16
→ ~4.0 GB for a full 262,144-token sequence at FP8
```

**Architectures that shrink KV:**

| Technique | Effect |
|---|---|
| **GQA** (grouped-query attention) | Fewer KV heads than query heads |
| **MLA** (multi-head latent attention) | Compresses KV into a low-rank latent |
| **Linear attention** (Gated DeltaNet, Mamba-2) | **Constant-size recurrent state** — does not grow with sequence length |
| **Sliding window** | Only caches a fixed window |

Hybrid models (Qwen3.5/3.6, Nemotron-3.5) interleave linear and full attention, so KV grows far more slowly than layer count suggests. **For these, weights — not KV — are the binding constraint.**

### FP8 KV cache

`--kv-cache-dtype fp8` roughly halves KV memory. Caveats:

- Default per-tensor scale is **1.0** if the checkpoint has no calibrated scales — the worst case for accuracy, with reported degradation on reasoning-heavy tasks and long generations.
- Calibrate with llm-compressor rather than relying on defaults.
- `--calculate-kv-scales` is **buggy on hybrid GatedDeltaNet+Attention models** (vLLM #37554) — produces corrupted frozen scales. Do not use it on Qwen3.5.
- It buys *capacity*, which is worthless at low concurrency and decisive at high batch.

---

## 8. Parallelism

| Type | What it splits | Communication cost |
|---|---|---|
| **TP** (tensor parallel) | Each layer's matrices across GPUs | **High** — all-reduce every layer |
| **PP** (pipeline parallel) | Different layers on different GPUs | Low — activations at boundaries |
| **EP** (expert parallel) | MoE experts across GPUs | **Very high** — all-to-all per token |
| **DP** (data parallel) | Independent replicas | None |

**Interconnect determines what is viable:**

| Fabric | Bandwidth |
|---|---|
| PCIe Gen4 | ~64 GB/s |
| PCIe Gen5 + GPUDirect P2P | ~128 GB/s |
| NVLink 3 (A100) | 600 GB/s |
| NVLink 4 (H100) | 900 GB/s |
| NVLink 5 (B200) | 1.8 TB/s |

g7e has **no NVLink** — PCIe Gen5 only, ~7× less than H100. Measured consequences:
- TP=8 on 8× RTX PRO 6000 ≈ **one-third** the aggregate throughput of 8× H100 SXM
- **Expert-parallel over PCIe is catastrophic: 1.4–2.6 tok/s.** Never enable `--enable-expert-parallel` on g7e.

**Rule: use the smallest TP that fits.** TP=1 ideal, TP=2 fine, TP=4 tolerable, TP=8 heavily penalized.

---

## 9. Speculative decoding

A small draft model proposes several tokens; the main model verifies them in one forward pass. Accepted tokens are free.

| Name | What it is |
|---|---|
| **MTP** (multi-token prediction) | Draft heads trained into the model itself |
| **EAGLE** | Separate lightweight draft head |
| **DSpark / DFlash** | NVIDIA Nemotron variants |

**Efficacy depends on the kernel path.** On a **Marlin dequantize fallback**, the verification pass is disproportionately expensive *and* dequantized activations differ from what the draft heads were trained on. Measured on Qwen3.5-397B-NVFP4/SM120: **−22%**, with acceptance dropping to 61–85%.

On a native FP4/FP8 path, the same technique gives **+38% or more**.

**Rule: enable speculative decoding only after confirming you are on a native path. If you see the Marlin fallback warning, disable it.**

---

## 10. Serving and structured output

| Term | Meaning |
|---|---|
| **Continuous batching** | Add/remove requests from a running batch dynamically |
| **Chunked prefill** | Split long prefills into chunks so decode is not blocked. Reported ~10–20% cost at low concurrency. |
| **Prefix caching** | Reuse KV for shared prefixes. **Large win for agents** re-sending system prompts and tool schemas. |
| **CUDA graphs** | Pre-record kernel launch sequences. Eager mode costs 50–70% throughput. |
| **PagedAttention** | vLLM's non-contiguous KV allocation. Reduces fragmentation. |
| **TTFT** | Time to first token — prefill latency |
| **TPOT / ITL** | Time per output token — decode latency |

### Structured output

| Term | Meaning |
|---|---|
| **Guided/constrained decoding** | Mask logits so only schema-valid tokens can be emitted |
| **xgrammar** | vLLM's default backend. Grammar compile is one-time; <3% per-token overhead on cached schemas. |
| **outlines / llguidance** | Alternative backends |
| **Tool-call parser** | Extracts function calls from model output into the OpenAI `tool_calls` format |
| **Reasoning parser** | Separates chain-of-thought from the user-visible answer |

**Parsers are model-family-specific** and a common source of silent failure:

| Family | Tool parser | Reasoning parser |
|---|---|---|
| Qwen3.x | `qwen3_xml` (preferred) or `qwen3_coder` | `qwen3` |
| GLM-4.5 / 5.2 | `glm45` / `glm47` | `glm45` |
| Kimi K2/K3 | `kimi_k2` / `kimi_k3` | same |
| gpt-oss | `openai` (harmony format) | `gpt_oss` |
| Nemotron 3.5 | `qwen3_coder` | `nemotron_v3` |

Known issues worth knowing: `qwen3_coder` can emit an infinite `!!!!` stream on long inputs and does not stream tool-call arguments (#30439); `tool_choice='required'` + MTP + thinking mode can produce XML instead of JSON (50–70% failure) — use `tool_choice='auto'` or disable thinking.

---

## 11. Quick reference

| If you see... | It means... |
|---|---|
| "Marlin kernel... may degrade performance" | No native kernel for your format+arch. Correct output, sub-optimal speed. |
| `cudaErrorUnsupportedPtxVersion` | Kernel built for wrong arch target. Likely bare `sm_120` needing `sm_120f`. |
| "Int8 not supported for this architecture" | You are on SM120. Use FP8 or INT4 instead. |
| "No compiled cutlass_scaled_mm for CUDA device capability: 120" | FP8 MoE grouped GEMM unavailable. Triton fallback. |
| "Using default MoE config... Performance might be sub-optimal" | No tuned Triton config for your device string. Cold-start autotune penalty. |
| "sink setting not supported" | Forced FlashInfer/FlashAttn on gpt-oss. Use TRITON_ATTN. |

| Rule of thumb | Why |
|---|---|
| Decode speed ∝ 1/(active params × bytes per param) | Decode is bandwidth-bound |
| Prefill is nearly quantization-independent | Prefill is compute-bound; formats compute similarly |
| Prefer MoE for decode latency | Active ≪ total parameters |
| Prefer INT4 over FP4 on non-SM100 | Same benefit, mature kernel, no arch-gated instruction |
| Use the smallest TP that fits | All-reduce cost scales with TP degree |
| Never expert-parallel over PCIe | Measured 1.4–2.6 tok/s |
| Verify cubin arch before trusting any FP4 path | `cuobjdump -lelf ... \| grep sm_120` |

---

*Companion document: `02-g6e-g7e-quantization-and-model-selection.md` applies all of this to the specific g6e/g7e decision.*

---

## 12. References

**Vendor silicon specifications (primary)**
- NVIDIA RTX PRO 6000 Blackwell Server Edition — https://www.nvidia.com/en-in/data-center/rtx-pro-6000-blackwell-server-edition/#specs
- NVIDIA L40S — https://www.nvidia.com/en-in/data-center/l40s/#specifications
- NVIDIA Blackwell architecture — https://www.nvidia.com/en-in/data-center/technologies/blackwell-architecture/
- NVIDIA Ada Lovelace architecture — https://www.nvidia.com/en-in/technologies/ada-architecture/
- NVIDIA Tensor Cores — https://www.nvidia.com/en-in/data-center/tensor-cores/
- Data Center GPU Line Card — https://docs.nvidia.com/data-center-gpu/line-card.pdf

**Kernel libraries**
- CUTLASS — https://github.com/NVIDIA/cutlass
- vLLM — https://github.com/vllm-project/vllm
- FlashInfer — https://github.com/flashinfer-ai/flashinfer
- DeepGEMM — https://github.com/deepseek-ai/DeepGEMM
- llm-compressor (quantization toolkit, vLLM project) — https://github.com/vllm-project/llm-compressor

**Companion**
- `02-g6e-g7e-quantization-and-model-selection.md` — applies this terminology to the specific g6e/g7e decision, with model recommendations and deployment configs.
