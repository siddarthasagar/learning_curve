# AWS g6e & g7e: Quantization Support and Model Selection

**Scope:** AWS g6e (NVIDIA L40S, SM89) and g7e (NVIDIA RTX PRO 6000 Blackwell Server Edition, SM120) only.
**Use case:** Assistive agent product — tool/function-calling accuracy, structured JSON reliability, multi-step reasoning. Coding deprioritized.
**Stack:** EKS + vLLM + Karpenter, Strands Agents SDK, MCP tool transport.
**Regions:** us-east-1 / us-west-2 (pricing is us-east-1 EC2 on-demand, observed August 2026).

**Primary hardware references** (all silicon claims in this document are sourced from these):
- NVIDIA RTX PRO 6000 Blackwell Server Edition — https://www.nvidia.com/en-in/data-center/rtx-pro-6000-blackwell-server-edition/#specs
- NVIDIA L40S — https://www.nvidia.com/en-in/data-center/l40s/#specifications

---

## 1. Official silicon specifications

### 1.1 NVIDIA RTX PRO 6000 Blackwell Server Edition (g7e, SM120)

Verbatim from the NVIDIA specifications table:

| Spec | Value |
|---|---|
| GPU Architecture | NVIDIA Blackwell Architecture |
| CUDA Parallel Processing Cores | 24,064 |
| NVIDIA RT Cores | 188 (4th Gen) |
| **FP4 Tensor Core** | **4 PFLOPS** |
| **FP8 Tensor Core** | **2 PFLOPS** |
| **FP16 / BF16 Tensor Core** | **1 PFLOP** |
| TF32 Tensor Core | 234 TFLOPS |
| Single-Precision (FP32) | 120 TFLOPS |
| Peak RT Core Performance | 355 TFLOPS |
| GPU Memory | 96 GB GDDR7 (with ECC) |
| Memory Interface | 512-bit |
| **Memory Bandwidth** | **1597 GB/s** |
| Max Power | Up to 600W (configurable) |

Additional documented features: 5th Gen Tensor Cores that "add support for FP4 precision"; second-generation Transformer Engine; PCIe Gen 5; Multi-Instance GPU (MIG) supporting "up to four fully isolated instances," each with its own memory, cache, and compute cores.

**Two observations that matter:**

1. **FP4 hardware is real and officially specified.** 4 PFLOPS FP4 against 2 PFLOPS FP8 against 1 PFLOP BF16 — a clean 2:1 ratio at each precision halving. That doubling is the signature of genuine tensor-core support at each level; emulation would not produce it. Any claim that SM120 "has no FP4 hardware" is wrong.
2. **The spec table contains no INT8 or INT4 row.** NVIDIA does not advertise integer tensor throughput for this part. This is consistent with (though not proof of) vLLM's hard block on INT8 W8A8 for SM120.

### 1.2 NVIDIA L40S (g6e, SM89)

Verbatim from the NVIDIA specifications table (`*` = with sparsity):

| Spec | Value |
|---|---|
| GPU Architecture | NVIDIA Ada Lovelace architecture |
| GPU Memory | 48 GB GDDR6 with ECC |
| **Memory Bandwidth** | **864 GB/s** |
| **Interconnect Interface** | **PCIe Gen4 x16: 64 GB/s bidirectional** |
| CUDA Cores | 18,176 |
| Third-Generation RT Cores | 142 |
| Fourth-Generation Tensor Cores | 568 |
| RT Core Performance | 212 TFLOPS |
| FP32 | 91.6 TFLOPS |
| TF32 Tensor Core | 183 / 366* TFLOPS |
| **BFLOAT16 Tensor Core** | **362.05 / 733* TFLOPS** |
| FP16 Tensor Core | 362.05 / 733* TFLOPS |
| **FP8 Tensor Core** | **733 / 1,466* TFLOPS** |
| **Peak INT8 Tensor TOPS** | **733 / 1,466*** |
| **Peak INT4 Tensor TOPS** | **733 / 1,466*** |
| Max Power | 350W |
| **Multi-Instance GPU (MIG)** | **No** |
| **NVLink Support** | **No** |

**Three observations that matter:**

1. **No FP4 row exists.** The table runs FP32 → TF32 → BF16 → FP16 → FP8 → INT8 → INT4 and stops. Ada Lovelace predates FP4 entirely. **This is the one place where "no FP4 hardware" is literally true.**
2. **INT4 is natively supported at 733 TOPS dense** — identical throughput to FP8 and INT8. Ada has real integer-4 tensor silicon.
3. **FP8 : BF16 = 733 : 362.05 = 2.02:1.** The same clean doubling. The occasional claim that "Ada FP8 doesn't get Hopper's 2× ratio" is not supported by the datasheet.

### 1.3 Side by side

| | g6e (L40S) | g7e (RTX PRO 6000 BSE) | Ratio |
|---|---|---|---|
| Architecture | Ada Lovelace, SM89 | Blackwell, SM120 | |
| Memory | 48 GB GDDR6 (44.7 GiB usable) | 96 GB GDDR7 | **2.0×** |
| **Bandwidth** | **864 GB/s** | **1,597 GB/s** | **1.85×** |
| BF16 tensor | 362 TFLOPS | 1,000 TFLOPS | 2.76× |
| FP8 tensor | 733 TFLOPS | 2,000 TFLOPS | 2.73× |
| FP4 tensor | **none** | 4,000 TFLOPS | — |
| INT8 / INT4 tensor | 733 TOPS each | not published | — |
| Interconnect | PCIe Gen4, 64 GB/s | PCIe Gen5 + GPUDirect P2P | 2× |
| NVLink | No | No | |
| MIG | No | Up to 4 instances | |
| Power | 350W | up to 600W | |

The 1.85× bandwidth ratio is the number that predicts decode throughput, and it matches AWS's own claim of "1.85x the GPU memory bandwidth" for G7e versus G6e.

---

## 2. Instance pricing

| | g6e.xlarge | g7e.2xlarge |
|---|---|---|
| GPU | 1× L40S | 1× RTX PRO 6000 BSE |
| vCPU / RAM | 4 / 32 GiB | 8 / 64 GiB |
| **On-demand** | **$1.8610/hr** (~$1,359/mo) | **$3.3631/hr** (~$2,455/mo) |

**Price ratio 1.81× against bandwidth ratio 1.85×** — throughput-per-dollar is roughly a wash. The real differentiator is **96 GB vs 44.7 GiB usable**, which decides which models run at all.

AWS's own SageMaker benchmark (Qwen3-32B BF16, concurrency 32): **ml.g7e.2xlarge = $0.79 per 1M output tokens** vs **ml.g6e.12xlarge = $2.06** — a 2.6× reduction. Latency scaling from concurrency 1→32 is 22% on G7e vs 62% on G6e.

### Full family (us-east-1 EC2 on-demand)

| g6e | GPUs | $/hr | | g7e | GPUs | $/hr |
|---|---|---|---|---|---|---|
| xlarge | 1 | 1.8610 | | 2xlarge | 1 | 3.3631 |
| 2xlarge | 1 | 2.2421 | | 4xlarge | 1 | 3.9982 |
| 4xlarge | 1 | 3.0042 | | 8xlarge | 1 | 5.2682 |
| 8xlarge | 1 | 4.52856 | | 12xlarge | 2 | 8.2861 |
| 12xlarge | **4** | 10.4926 | | 24xlarge | 4 | 16.5722 |
| 16xlarge | **1** | 7.57719 | | 48xlarge | 8 | 33.1443 |
| 24xlarge | 4 | 15.0655 | | | | |
| 48xlarge | 8 | 30.1312 | | | | |

Two traps: **g6e has no 2-GPU size** (1 → 4), so a 2-GPU need is better served by g7e.12xlarge ($8.29) than g6e.12xlarge ($10.49). And **g6e.16xlarge is a 1-GPU instance at $7.58/hr** — worse per-GPU than the 4-GPU 12xlarge. Never buy it for inference.

SageMaker adds ~25% (ml.g7e.2xlarge ≈ $4.204/hr, ml.g6e.xlarge ≈ $2.326/hr). SageMaker AI Savings Plans offer up to 64% off for 1–3 year commitments.

---

## 3. Quantization support matrix

**Critical distinction throughout: hardware capability ≠ kernel availability.** The silicon tables in §1 state what the chip can do. What follows states what vLLM will actually run on it today.

### Dense layers

| Format | g6e (SM89) — HW | g6e kernel | g7e (SM120) — HW | g7e kernel |
|---|---|---|---|---|
| BF16 | ✅ 362 TFLOPS | cuBLAS/CUTLASS | ✅ 1 PFLOP | cuBLAS/CUTLASS |
| **FP8 W8A8** | ✅ **733 TFLOPS** | CUTLASS `scaled_mm` | ✅ **2 PFLOPS** | CUTLASS `scaled_mm` (PR #22131, v0.9.2) |
| **INT8 W8A8** | ✅ **733 TOPS** | CUTLASS INT8 | not published | ❌ **BLOCKED** — `RuntimeError` (#28856) |
| **INT4 (AWQ/GPTQ)** | ✅ **733 TOPS** | GPTQ-Marlin (W4A16) | not published | GPTQ-Marlin (W4A16) |
| NVFP4 / MXFP4 | ❌ **no FP4 silicon** | Marlin W4A16 | ✅ **4 PFLOPS** | SM120 kernel exists; **needs `sm_120f` build** |

### MoE layers

| Format | g6e (SM89) | g6e kernel | g7e (SM120) | g7e kernel |
|---|---|---|---|---|
| BF16 | ✅ | Triton/CUTLASS | ✅ | Triton/CUTLASS |
| **FP8 grouped GEMM** | HW yes, **kernel no** | Triton `fused_moe` | HW yes, **kernel no** | Triton `fused_moe` |
| **INT4 (AWQ/GPTQ)** | ✅ | `moe_wna16` / Marlin | ✅ | `moe_wna16` / Marlin |
| NVFP4 / MXFP4 | ❌ no FP4 silicon | Marlin W4A16 | ✅ **HW present** | **kernel gap** → Marlin |
| DeepGEMM | ❌ | — | ❌ | — |

### Six things this matrix tells you

**1. g7e HAS FP4 hardware — 4 PFLOPS, officially specified.** What it lacks is complete kernel coverage, for two separate reasons. First, the FP4 conversion instructions (`cvt.e2m1x2.f32`) are gated behind the `sm_120a`/`sm_120f` compilation targets (CUDA 12.9+); a binary built for bare `sm_120` cannot emit them even though the transistors exist. Second, SM100's FP4 MoE kernels are built on `tcgen05` staging operands through Tensor Memory (TMEM) — a memory tier SM120 does not have — so they need rewriting, not recompiling. Dense FP4 got a bespoke SM120 kernel (`nvfp4_scaled_mm_sm120_kernels.cu`); MoE grouped FP4 largely did not.

**2. g6e genuinely has no FP4.** The L40S spec table has no FP4 row. This is a hardware ceiling, not a software gap, and no toolchain change will lift it.

**3. FP8 MoE grouped GEMM is unavailable on both.** vLLM's `cutlass_group_gemm_supported()` accepts compute capability 90–109. SM89 fails below (89 < 90), SM120 above (120 ≥ 110). Both route MoE experts to Triton. **"FP8 is native on my hardware" is only fully true for dense models.**

**4. INT8 is native on g6e (733 TOPS) but blocked on g7e** in vLLM. NVIDIA also does not publish integer throughput for RTX PRO 6000. One of the few capabilities the older card has and the newer lacks.

**5. INT4 works everywhere — and note what it actually uses.** L40S has native INT4 tensor cores at 733 TOPS, but vLLM's GPTQ-Marlin is **W4A16**: it unpacks 4-bit weights, dequantizes to BF16, and multiplies on BF16 tensor cores. So the 733 TOPS INT4 figure is not the path being exercised. This is *why* INT4 is so portable — it needs only integer→float conversion (universal since Kepler) plus BF16 MMA, so the absence of an INT4 row on the RTX PRO 6000 spec sheet costs nothing.

**6. Only FP4 formats carry build risk.** A container built for bare `sm_120` loads weights fine, then crashes at first FP4 kernel launch with `cudaErrorUnsupportedPtxVersion`. INT4 never touches that instruction.

### Consolidated

| Model type | Format | g6e | g7e |
|---|---|---|---|
| Dense | FP8 | ✅ CUTLASS | ✅ **CUTLASS** ← best dense |
| Dense | INT4 | ✅ Marlin | ✅ Marlin |
| Dense | INT8 | ✅ CUTLASS | ❌ blocked |
| Dense | NVFP4 | ⚠️ no HW → Marlin | ⚠️ HW present, build-gated |
| MoE | FP8 | ⚠️ Triton | ⚠️ Triton |
| MoE | INT4 | ✅ `moe_wna16` | ✅ **`moe_wna16`** ← best MoE |
| MoE | NVFP4/MXFP4 | ⚠️ no HW → Marlin | ⚠️ kernel gap → Marlin |

**Decision rule: dense → FP8. MoE → INT4.**

---

## 4. Performance: what actually drives speed

Measured on RTX PRO 6000, Qwen3.6-27B:

| Format | Decode tok/s | Prefill tok/s |
|---|---|---|
| BF16 | 59 | 4,359 |
| FP8 | 97–100 | **4,747** |
| NVFP4 | **163–169** | 4,732 |

**Decode varies ~70% by format. Prefill varies 0.3%.** Quantization is a decode-only decision.

Also measured on the same GPU: **AWQ-Int4 at 259 tok/s tuned, KLD 0.024** versus NVFP4's 163–169 tok/s and KLD 0.035. INT4 was both faster *and* closer to the unquantized reference.

Sanity-check that against the silicon: at 4,747 prefill tok/s on a 27B model, effective throughput is ~256 TFLOPS — about **25% of the card's 1 PFLOP BF16 peak**. Prefill is not compute-saturated at agent batch sizes, which is exactly why the format barely moves it, and why native FP4's extra arithmetic would buy little here.

```
decode speed  ∝  1 / (ACTIVE params × bytes per param)
```

| Config | Active | Bytes/token | Relative decode |
|---|---|---|---|
| 27B dense FP8 | 27B | ~29 GB | 1× (slowest) |
| 27B dense INT4 | 27B | ~15 GB | ~2× |
| 122B MoE INT4 | 10B | ~5–8 GB | ~4–6× |
| 35B-A3B MoE INT4 | 3B | ~2 GB | ~15× |

**A 122B MoE decodes faster than a 27B dense.** Sparsity beats size, and it collapses the usual smart-vs-fast tradeoff.

---

## 5. Model inventory

Sizes marked `*` are **estimated** from published ratios (verified anchor: Qwen3.5-397B-GPTQ-Int4 = 236 GB / 397B ≈ 0.59 bytes/param). Smaller models keep proportionally more weight in BF16, so real sizes skew **higher**. **Verify shard totals on the HF file tree before committing.**

| Model | Format | Size | Active | g6e | g7e | Notes |
|---|---|---|---|---|---|---|
| `Qwen/Qwen3.5-122B-A10B-GPTQ-Int4` | INT4 MoE | ~70–85 GB* | 10B | ❌ | ✅ tight | 570k downloads; BFCL-V4 **72.2**, IFEval **93.4** |
| `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` | INT4 MoE | ~20–22 GB* | 3B | ✅ | ✅ | 362k downloads |
| `Qwen/Qwen3.6-35B-A3B-FP8` | FP8 MoE | ~37 GB | 3B | ⚠️ ~7 GB KV | ✅ | 9.47M downloads |
| `Qwen/Qwen3.6-27B-FP8` | FP8 dense | ~29 GB | 27B | ⚠️ ~7 GB KV | ✅ | 7.95M downloads; fully native |
| `Qwen/Qwen3.8-27B-FP8` | FP8 dense | ~29 GB | 27B | ⚠️ tight | ✅ | Newest; 64-layer hybrid, `attn_output_gate` |
| `Qwen/Qwen3.5-27B-GPTQ-Int4` | INT4 dense | ~17–19 GB* | 27B | ✅ | ✅ | 111k downloads |
| `Qwen/Qwen3.5-27B-FP8` | FP8 dense | ~29 GB | 27B | ⚠️ | ✅ | BFCL-V4 0.685 |
| `nvidia/Nemotron-3.5-Lightning-30B-A3B-NVFP4` | NVFP4 MoE | ~15–20 GB* | 3B | ✅ | ✅ | 1M ctx; τ³-Banking only **9.48** |
| `openai/gpt-oss-120b` | MXFP4 MoE | 60.8 GB | 5.1B | ❌ | ⚠️ FP4 risk | τ-bench Retail 67.8% |
| `openai/gpt-oss-20b` | MXFP4 MoE | 12.8 GB | 3.6B | ⚠️ FP4 risk | ⚠️ FP4 risk | τ-bench Retail 54.8% |
| `Qwen/Qwen3.5-397B-A17B-GPTQ-Int4` | INT4 MoE | **236 GB** (verified) | 17B | 8 GPU | 4 GPU | BFCL-V4 72.9, TAU2 86.7 |
| `nvidia/Qwen3.5-397B-A17B-NVFP4` | NVFP4 MoE | **251 GB** (verified) | 17B | 8 GPU | 4 GPU | 50.5 tok/s TP=4 measured |
| `Qwen/Qwen3.5-397B-A17B-FP8` | FP8 MoE | **406 GB** (verified) | 17B | ❌ >384 GB | 8 GPU | |

### Availability notes

- **Qwen3.6 and Qwen3.8 ship FP8 only** — no first-party INT4. Qwen3.6 caps at 36B; Qwen3.8-27B is dense. The Qwen3.5 GPTQ-Int4 checkpoints appeared ~2 months after the base release, so INT4 for newer generations may simply be pending.
- **`nvidia/Qwen3.5-397B-A17B-NVFP4-V2` does not exist.** Verified twice against NVIDIA's collection listing; only DeepSeek-R1 carries a `-v2` suffix.
- Qwen3.5/3.8 checkpoints are **multimodal**. Pass `--language-model-only` to reclaim VRAM if you don't need vision.

### Not viable on either family

| Model | Size | Reason |
|---|---|---|
| Kimi K3 | 1,560 GB | Exceeds 768 GB (8× g7e) and 384 GB (8× g6e) |
| Kimi K2.6 | 552–585 GB | g7e.48xlarge only; not viable on g6e |
| GLM-5.2-NVFP4 | 446 GB | g7e.48xlarge only |
| GLM-4.5-FP8 | 377 GB | Would fill 8× L40S with zero KV headroom |
| Qwen3.5-397B BF16 | 807 GB | Exceeds both families |

---

## 6. Recommendations by category

### g7e.2xlarge — $3.3631/hr (~$2,455/mo)

| Category | Model | Why |
|---|---|---|
| **Smartest** | `Qwen3.5-122B-A10B-GPTQ-Int4` | Best verified tool-calling (BFCL-V4 72.2, IFEval 93.4). Only model >36B that fits one GPU. |
| **Fastest decode** | `Qwen3.5-35B-A3B-GPTQ-Int4` | 3B active × 4-bit ≈ 2 GB/token |
| **Best prefill / high batch** | `Qwen3.6-27B-FP8` or `Qwen3.8-27B-FP8` | Native CUTLASS dense end-to-end |
| **Safest deployment** | `Qwen3.6-27B-FP8` | Zero fallbacks, no FP4, 7.95M downloads |
| **BEST OVERALL** | **`Qwen3.5-122B-A10B-GPTQ-Int4`** | Smartest *and* ~4× better decode economics than the 27B dense |

### g6e.xlarge — $1.8610/hr (~$1,359/mo)

| Category | Model | Why |
|---|---|---|
| **Smartest that fits** | `Qwen3.5-35B-A3B-GPTQ-Int4` | ~20–22 GB, most total knowledge that fits comfortably |
| **Fastest decode** | Same, or Nemotron-3.5-Lightning | Both 3B active |
| **Best prefill** | `Qwen3.6-27B-FP8` | Native FP8 — but ~7 GB KV, low concurrency only |
| **Safest / most KV headroom** | `Qwen3.5-27B-GPTQ-Int4` | ~17–19 GB → ~25 GB for KV |
| **BEST OVERALL** | **`Qwen3.5-35B-A3B-GPTQ-Int4`** | Fits comfortably, fastest decode, no FP4 risk |

**INT4 is what makes g6e usable.** At FP8 the 27B is ~29 GB in 44.7 GiB → ~7 GB KV (cramped). At INT4 it is ~17–19 GB → ~25 GB KV. The difference between "technically fits" and "comfortable with real batching."

---

## 7. Overall recommendation

### Primary: `Qwen/Qwen3.5-122B-A10B-GPTQ-Int4` on g7e.2xlarge — ~$2,455/mo

1. **Best verified agentic scores** in the viable set — BFCL-V4 72.2, IFEval 93.4
2. **First-party checkpoint** from Qwen, 570k downloads
3. **INT4 uses `moe_wna16`/GPTQ-Marlin** — the *intended* kernel for MoE INT4, not a fallback
4. **No FP4 PTX exposure** — immune to the `sm_120f` crash class
5. **~4× better decode economics** than the dense 27B
6. **Single GPU, TP=1** — no PCIe all-reduce penalty

**Must verify:** the ~70–85 GB estimate against 96 GB actual. At the top of that range KV headroom gets thin.

### Runner-up: `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4`

Fastest decode available, runs on **either** instance. Develop on g6e.xlarge at $1.86/hr, promote the identical checkpoint to g7e.2xlarge for production.

### Control: `Qwen/Qwen3.6-27B-FP8` on g7e.2xlarge

Fully native CUTLASS path, zero deployment risk. Keep as your known-good A/B baseline — if a 4-bit variant misbehaves on tool calls, this isolates model from quantization.

### Wildcard: `nvidia/Nemotron-3.5-Lightning-30B-A3B-NVFP4`

Cheapest viable at ~15–20 GB, 3B active, 1M context. NVIDIA ships a validated W4A16 recipe for **A100 (SM80)** — hardware with no FP4 at all — which strongly implies its dequant path avoids Blackwell-only instructions. But the only published multi-turn tool-use score is τ³-bench Banking **9.48**, with no BFCL. Benchmark before trusting it.

---

## 8. Deployment configuration

### INT4 MoE on g7e.2xlarge

```bash
export VLLM_CACHE_ROOT=/opt/vllm-cache      # bake into image
export VLLM_USE_DEEP_GEMM=0                 # no SM120 support

vllm serve Qwen/Qwen3.5-122B-A10B-GPTQ-Int4 \
  --tensor-parallel-size 1 \
  --quantization moe_wna16 \
  --max-model-len 262144 \
  --language-model-only \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,32]}'
```

### FP8 dense on g7e.2xlarge

```bash
vllm serve Qwen/Qwen3.6-27B-FP8 \
  --tensor-parallel-size 1 \
  --quantization fp8 \
  --max-model-len 262144 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3
```

**Verify FP8 actually loaded as FP8.** A community report claims Qwen's DeepSeek-format block-128 FP8 must be dequantized to BF16 before running on Blackwell — which contradicts vLLM PR #22131 ("Add support for block FP8 on SM120"). If a ~29 GB checkpoint consumes ~58 GB, it dequantized. Check before trusting the native-FP8 assumption.

### Flags to A/B rather than assume

| Flag | Consideration |
|---|---|
| `--kv-cache-dtype fp8` | Buys capacity; reported 10–15% throughput cost at low batch. Never use `--calculate-kv-scales` on Qwen3.5 hybrids (#37554 corruption bug). |
| `--enable-chunked-prefill` | Reported 10–20% cost at low concurrency |
| `--speculative-config` | **Only on a confirmed native path.** −22% on Marlin fallback. |
| `--enable-expert-parallel` | **Never on g7e.** 1.4–2.6 tok/s over PCIe. |

### Containers

Prefer **BYOC vLLM** built for `sm_120a`/`sm_120f`, the NVIDIA NGC vLLM container (25.09+ explicitly lists RTX PRO 6000 Blackwell Server Edition support), or the AWS standalone vLLM CUDA-13 DLC. Stock DJL-LMI images have shipped kernels compiled for bare `sm_120`.

Before deploying any FP4 model:
```bash
cuobjdump -lelf $(python -c "import vllm,os;print(os.path.dirname(vllm.__file__))")/_C.abi3.so | grep sm_120
# want sm_120a or sm_120f — bare sm_120 means FP4 will crash
```
INT4 checkpoints do not depend on this.

### EKS / Karpenter

- Pin node classes to multiple AZs — GPU capacity errors are common
- Stage weights on local NVMe rather than pulling from S3 on every scale-up
- Bake tuned Triton `fused_moe` configs into the image (device string must match exactly)
- Warmup request during pod init so `torch.compile` and CUDA-graph capture finish before traffic
- g6e Spot is broadly available (~$1.38/hr xlarge); g7e Spot is thin
- g7e supports **MIG up to 4 instances** — a possible route to co-locating small models or isolating tenants on one 96 GB card

---

## 9. Validation before you commit

The biggest gap across all published sources: **nobody has measured quantization impact on tool calling.** NVIDIA's NVFP4-vs-FP8 comparisons cover MMLU-Pro, GPQA, AIME, IFBench. RedHatAI publishes OpenLLM, Arena-Hard, HumanEval. Neither covers BFCL, τ²-bench, or JSON-schema adherence.

The strongest available proxy is **HumanEval**, because code generation shares tool calling's structural property — syntactically exact output where one wrong token breaks everything. Measured on Qwen3-32B GPTQ-Int4: MMLU-Pro dropped 1.6 points while **HumanEval dropped 8**. A 5× difference on the same checkpoint. Expect tool calling to behave more like HumanEval than like MMLU.

Also note: quantization damage is **worse for smaller models**. Recovery tables show W4A16 on Llama-3.1-8B at 83–94% on some tasks, but 99–100% at 70B and 405B. Your 27B sits nearer the fragile end.

Run before shipping:

1. **Same model, FP8 vs INT4** — a few hundred multi-step trajectories against your real MCP tool schemas
2. **Measure:** JSON validity rate, argument correctness, correct-tool-selection, and **negative tool discipline** (correctly declining when no tool applies — a measured axis in BFCL's irrelevance-detection category that degrades as your registry grows)
3. **Measure end-to-end trajectory latency**, not tok/s in isolation
4. **Watch startup logs** for the Marlin fallback warning on any model you expect to be native

If JSON validity drops below ~95%, enable `guided_json` (xgrammar, <3% per-token overhead on cached schemas) before changing models.

---

## 10. Caveats

- Sizes marked `*` are estimates from the verified 397B ratio; real sizes skew higher. Verify shard totals.
- **No measured L40S throughput** exists publicly for 27–32B dense or 30–35B MoE. The 1.85× bandwidth ratio is the best predictor.
- **GPTQ-Int4 throughput on either card is not independently benchmarked** — estimates come by analogy to AWQ-Int4 on the same W4A16 kernel class.
- No GPTQ-Int4 accuracy comparison (KLD, BFCL, τ²) is published for these models.
- RTX PRO 6000 tensor figures (4/2/1 PFLOPS) carry no explicit dense-vs-sparse label on the NVIDIA spec page; L40S figures are explicitly marked dense / with-sparsity. Treat the g7e-vs-g6e compute ratios as approximate.
- SM120 FP4 support is a **moving target**. If native SM120 FP4 MoE stabilizes upstream, NVFP4 gains roughly 2× compute-bound throughput plus functional speculative decoding. Re-check each vLLM release.
- Pricing observed August 2026; AWS list prices change and Spot fluctuates by AZ.
- Community benchmarks (rtx6kpro, vLLM forum) are directional, not vendor-verified.

---

## 11. References

**Primary — NVIDIA official specifications**
- RTX PRO 6000 Blackwell Server Edition — https://www.nvidia.com/en-in/data-center/rtx-pro-6000-blackwell-server-edition/#specs
- RTX PRO 6000 datasheet — https://resources.nvidia.com/en-us-rtx-pro-6000
- L40S — https://www.nvidia.com/en-in/data-center/l40s/#specifications
- L40S datasheet — https://resources.nvidia.com/en-us-l40s/l40s-datasheet-28413

**AWS**
- EC2 G7e instances — https://aws.amazon.com/ec2/instance-types/g7e/
- G7e launch announcement — https://aws.amazon.com/blogs/aws/announcing-amazon-ec2-g7e-instances-accelerated-by-nvidia-rtx-pro-6000-blackwell-server-edition-gpus
- SageMaker AI with G7e (cost benchmark) — https://aws.amazon.com/blogs/machine-learning/accelerate-generative-ai-inference-on-amazon-sagemaker-ai-with-g7e-instances/

**vLLM issues and PRs cited**
- #22131 — block FP8 support on SM120
- #28856 — INT8 not supported on SM120
- #43507, #32109 — CUTLASS MoE unavailable on SM120
- #31085 — feature request: native NVFP4 MoE kernels for SM120
- #37725 — preserve CUDA arch suffix (a/f) for SM12x
- #37554 — `--calculate-kv-scales` corruption on hybrid models
- #30439 — `qwen3_coder` does not stream tool-call arguments
- CUTLASS #3096 — NVFP4 MoE on SM120, fixed via `compute_120f`

*Issue numbers were gathered across multiple research passes; the patterns are well-corroborated but verify specific identifiers before acting on one.*

---

*Companion document: `01-llm-inference-terminology.md` explains the kernel, quantization, and hardware terminology used throughout.*
