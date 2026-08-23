# NVIDIA GPU Hardware & Quantization Kernel Support — Reference Document

**Revision 2 — corrected 21 August 2026.** Supersedes the August 2026 draft. See [§7 Changelog](#7-changelog-what-changed-in-revision-2) for a diff of what was wrong and why.

**Verification status of this revision:**

| Section | Status |
|---|---|
| §1 L40S, RTX PRO 6000 Blackwell SE | Re-verified against vendor sources, Aug 2026 |
| §1 A100 / H100 / H200 / GB200 / GB300 / HGX / Blackwell arch | Carried over from rev 1; **not** independently re-fetched. Consistent with NVIDIA's published datasheet figures, but treat as unaudited. |
| §2–§5 | Re-verified; corrections applied |
| §4 SM120 rows | Re-verified against vLLM release notes through v0.22.0. **This is the section that rots fastest — see §6.** |

---

## 1. GPU Hardware Specifications

### NVIDIA L40S — *verified*
- **Architecture**: Ada Lovelace (AD102), compute capability **8.9 (SM89)**
- **Form factor**: PCIe Gen4 x16, dual-slot, full-height full-length, passive
- **GPU memory**: 48 GB GDDR6 with ECC, 864 GB/s, 384-bit bus
- **Cores**: 18,176 CUDA; 568 4th-gen Tensor; 142 3rd-gen RT
- **Compute**: FP32 91.6 TFLOPS; TF32 Tensor Core 183 (366 sparse); FP16/BF16 Tensor Core 362.05 (733 sparse); FP8 Tensor Core 733 (1,466 sparse); INT8 733 TOPS (1,466 sparse); RT Core 209 TFLOPS
- **Interconnect**: PCIe Gen4 x16 (64 GB/s bidirectional). **No NVLink. No MIG.**
- **Max TDP**: 350W
- **Sources**: [NVIDIA L40S product page](https://www.nvidia.com/en-us/data-center/l40s/) · [L40S datasheet (PDF)](https://acecloud.ai/wp-content/uploads/2025/06/l40s-datasheet.pdf) · [Lenovo Press ThinkSystem L40S product guide](https://lenovopress.lenovo.com/lp1812-nvidia-l40s-48gb-pcie-gen4-passive-gpu)

> **Note on the "Transformer Engine" framing.** Ada's 4th-gen Tensor Cores support FP8 natively, but Ada does not have Hopper's runtime amax-tracking Transformer Engine. FP8 calibration on L40S is a build-time step in TensorRT-LLM or a one-shot calibration pass in vLLM, not a per-layer runtime decision.

### NVIDIA RTX PRO 6000 Blackwell Server Edition — *verified*
- **Architecture**: Blackwell (full GB202 die), compute capability **12.0 (SM120)**
- **Form factor**: PCIe 5.0 x16, dual-slot, FHFL, passive/fanless
- **GPU memory**: 96 GB GDDR7 ECC, 512-bit bus, **1,597 GB/s**
- **Cores**: 24,064 CUDA across 188 SMs; 752 5th-gen Tensor; 188 4th-gen RT
- **Compute**: FP32 120 TFLOPS; FP4 AI peak 4 PFLOPS; RT Core 355 TFLOPS
- **Multi-Instance GPU**: up to 4 instances @ 24 GB each
- **Max TDP**: up to 600W (PNY lists 400–600W configurable)
- **Sources**: [NVIDIA RTX PRO 6000 Blackwell Server Edition](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/) · [Lenovo Press product guide](https://lenovopress.lenovo.com/lp2263-thinksystem-nvidia-rtx-pro-6000-blackwell-server-edition-pcie-gen5-gpu)

> **Don't mix up the three editions.** Workstation, Max-Q, and Server share identical silicon (24,064 CUDA cores, 96 GB GDDR7) and differ in power limit, cooling, and memory data rate. Server Edition is **1,597 GB/s and 120 TFLOPS FP32**; Workstation Edition is **1,792 GB/s and 125 TFLOPS FP32**. Benchmarks quoted for "RTX PRO 6000" without an edition are ambiguous.

### NVIDIA A100 (Ampere) — *carried over, unaudited*
> Distinct from A800, the China-export variant with NVLink capped at 400 GB/s.

| Variant | Memory | Bandwidth | Interconnect | Max TDP |
|---|---|---|---|---|
| SXM4 40GB | 40 GB HBM2 | 1555 GB/s | NVLink 600 GB/s, PCIe Gen4 64 GB/s | 400W |
| SXM4 80GB | 80 GB HBM2e | 2039 GB/s | NVLink 600 GB/s, PCIe Gen4 64 GB/s | 400W |
| PCIe 80GB | 80 GB HBM2e | 1935 GB/s | NVLink Bridge (2 GPU) 600 GB/s, PCIe Gen4 64 GB/s | 300W |
| PCIe 40GB | 40 GB HBM2 | 1555 GB/s | PCIe Gen4 64 GB/s | 250W |

- Compute (common): FP64 9.7 TFLOPS; FP64 Tensor Core 19.5; FP32 19.5; TF32 Tensor Core 156 (312 sparse); BF16/FP16 Tensor Core 312 (624 sparse); INT8 Tensor Core 624 TOPS (1248 sparse)
- Compute capability **8.0 (SM80)**. Server options: HGX A100 (4/8 GPU); DGX A100 (8 GPU, 320 GB total)
- **Sources**: [A100 product page](https://www.nvidia.com/en-us/data-center/a100/#specifications) · [A100 datasheet r4 (PDF)](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf) · [Ampere architecture whitepaper (PDF)](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf)

### NVIDIA H100 (Hopper) — *carried over, unaudited*
| Variant | Memory | Bandwidth | Max TDP | Interconnect |
|---|---|---|---|---|
| SXM | 80 GB HBM3 | 3.35 TB/s | up to 700W | NVLink 900 GB/s, PCIe Gen5 128 GB/s |
| NVL (PCIe) | 94 GB | 3.9 TB/s | 350–400W | NVLink 600 GB/s, PCIe Gen5 128 GB/s |

- Compute (SXM): FP64 34 TFLOPS; FP64 Tensor Core 67; FP32 67; TF32 Tensor Core 989 (sparse); BF16/FP16 Tensor Core 1979 (sparse); FP8 Tensor Core 3958 (sparse)
- Compute capability **9.0 (SM90)**. **No FP4 tensor cores** — see §4 MXFP4/NVFP4 notes.
- **Source**: [NVIDIA H100 product page](https://www.nvidia.com/en-us/data-center/h100/#specifications)

### NVIDIA H200 (Hopper) — *carried over, unaudited*
- First GPU with HBM3e. **141 GB HBM3e, 4.8 TB/s** (both SXM and NVL)
- SXM: up to 700W, NVLink 900 GB/s. NVL (PCIe): up to 600W, 2/4-way NVLink bridge 900 GB/s per GPU
- Compute capability **9.0 (SM90)**. Confidential Computing on both variants.
- **Source**: [NVIDIA H200 product page](https://www.nvidia.com/en-us/data-center/h200/#specifications)

### NVIDIA GB200 NVL72 — *carried over, unaudited*
- 36 Grace CPUs + 72 Blackwell GPUs, fully liquid-cooled rack
- NVFP4 Tensor Core 1440 PFLOPS (sparse) / 720 PFLOPS (dense); FP8/FP6 720 PFLOPS; FP16/BF16 360 PFLOPS
- 13.4 TB HBM3E, 576 TB/s; NVLink 130 TB/s (72-GPU domain)
- CPU: 2592 Arm Neoverse V2 cores, 17 TB LPDDR5X, 14 TB/s
- **Source**: [GB200 NVL72 product page](https://www.nvidia.com/en-us/data-center/gb200-nvl72/#specs)

### NVIDIA GB300 NVL72 — *carried over, unaudited*
- 72 Blackwell Ultra GPUs + 36 Grace CPUs, liquid-cooled
- FP4 Tensor Core 1440 PFLOPS (sparse) / 1080 PFLOPS (dense); FP8/FP6 720 PFLOPS
- 20 TB GPU memory, up to 576 TB/s; NVLink 130 TB/s
- Quantum-X800 IB or Spectrum-X Ethernet, ConnectX-8 SuperNICs (800 Gb/s per GPU)
- vs GB200: 1.5x dense FP4 FLOPS, 2x attention; target is test-time-scaling inference
- **Source**: [GB300 NVL72 product page](https://www.nvidia.com/en-us/data-center/gb300-nvl72/#specs)

### NVIDIA HGX B200 / B300 — *carried over, unaudited*
| Spec | HGX B200 | HGX B300 |
|---|---|---|
| GPU | 8x Blackwell SXM | 8x Blackwell Ultra SXM |
| FP4 Tensor Core | 144 PFLOPS (sparse) | 144 PFLOPS (sparse), 108 dense |
| Total memory | 1.4 TB | 2.1 TB |
| NVLink gen / total BW | 5th / 14.4 TB/s | 5th / 14.4 TB/s |
| Networking bandwidth | 0.8 TB/s | 1.6 TB/s |
| Attention perf (relative) | 1x | 2x |

- **Source**: [NVIDIA HGX platform page](https://www.nvidia.com/en-us/data-center/hgx/#specifications)

### NVIDIA Blackwell Architecture — *carried over, unaudited*
- Custom TSMC 4NP, 208 billion transistors
- Two reticle-limited dies, 10 TB/s chip-to-chip, presented as a single GPU
- 2nd-gen Transformer Engine, adds FP4 microscaling (NVFP4)
- 5th-gen NVLink, scales to 576 GPUs, 130 TB/s in a 72-GPU domain
- Confidential Computing; first TEE-I/O capable GPU
- **Source**: [Blackwell Architecture page](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)

> **Datacenter Blackwell ≠ workstation Blackwell.** GB100/GB200-class parts are **SM100**; RTX PRO 6000 and RTX 50-series are **SM120**; GB10 (DGX Spark) is **SM121**. These are different compute capabilities with different kernel availability, and SM100 support does *not* imply SM120 support. Most of §4's pain lives in exactly this gap.

---

## 2. CUDA Kernel Languages & Where "Missing Kernel Support" Lives

### Languages CUDA kernels are written in
- **CUDA C/C++** — majority of kernels, compiled via `nvcc` to PTX then SASS
- **PTX** — architecture-agnostic intermediate assembly; JIT-compiled to SASS at runtime if no precompiled binary matches the GPU
- **SASS** — native machine code, locked to a specific compute capability (sm_100 and sm_120 do not cross-run)
- **CUTLASS** — NVIDIA's open-source C++ template library for hand-tuned GEMM/attention; where most FP4/FP8 MoE GEMM implementations live
- **CuTe DSL** — Python-embedded CUTLASS DSL; the newer FP4 MoE work for SM12x (FlashInfer's `b12x`) is written here rather than in C++ CUTLASS
- **Triton** — Python-embedded DSL used heavily in vLLM as a portable fallback when hand-tuned kernels aren't available for an architecture
- **Hand-written SASS** — appears in community projects when the stock toolchain lags. Example: an independent SM120 vLLM fork hand-wrote SASS for 2-bit MoE and FP4-delta kernels, building its own SM120 ISA database and assembler first because consumer Blackwell has no public SASS toolchain ([kacper-daftcode/vLLM-Moet](https://github.com/kacper-daftcode/vLLM-Moet)). *Low-weight source: a 3-star personal repo. Cited as an existence proof of the technique, not as a benchmark authority.*

### Four layers to check for "kernel support is missing on SM120"

Rev 1 listed three. There is a fourth, and it is the dangerous one.

1. **Kernel implementation** — does a working tile config exist for this compute capability and precision? NVFP4 MoE grouped GEMM hit a real CUTLASS bug on SM120 (TMA Warp-Specialized grouped GEMM initialization failure) even though the same path works on SM100 — [CUTLASS issue #3096](https://github.com/NVIDIA/cutlass/issues/3096), [CUTLASS #2800](https://github.com/NVIDIA/cutlass/issues/2800) (BlockScaledMmaOp restricts FP4 to sm_100a).
2. **Framework dispatch / capability detection** — separate Python-level checks (`is_device_capability_family`) decide which backend a model+precision combination routes to. Several "unsupported on SM120" errors were dispatch bugs, not missing kernels — [vLLM issue #31085](https://github.com/vllm-project/vllm/issues/31085).
3. **Build / packaging** — official wheels and Docker images are compiled with a fixed `TORCH_CUDA_ARCH_LIST`. If SM120 wasn't in the list for a release, the kernel source exists but wasn't compiled into the binary you downloaded.
4. **Silent architecture mismatch (new)** — the kernel is present, `is_supported()` returns True, the launch succeeds, and the output is garbage. On GB10 (SM12.1), FP8 CUTLASS kernels compiled for SM12.0 passed the support check and produced all-NaN logits with no Python exception; the `enable_sm120_only` guard let CUDA silently PTX-JIT the sm_120f binary onto SM12.1 — [spark-vllm-docker issue #143](https://github.com/eugr/spark-vllm-docker/issues/143), [NVIDIA developer forum thread](https://forums.developer.nvidia.com/t/gb10-sm12-1-vllm-fp8-inference-any-progress-on-native-sm12-1-kernels/363514). **A green support matrix cell does not mean correct numerics. Validate outputs, not just startup.**

---

## 3. Source-of-Truth Methodology for Building Support Matrices

Track four independent, versioned sources — a "yes" at one layer does not guarantee a "yes" at the next.

| Layer | What it answers | Source |
|---|---|---|
| Kernel exists (C++) | Does CUTLASS have a working tile config for this capability/precision? | [CUTLASS docs](https://docs.nvidia.com/cutlass/latest/overview.html) + the CUTLASS issue tracker filtered by SM number |
| Kernel exists (alternate) | Does FlashInfer have an independent implementation for this arch/precision? | [FlashInfer docs](https://docs.flashinfer.ai/) |
| Framework dispatches correctly | Does vLLM route this quant+arch combination to a working backend? | [vLLM release notes](https://github.com/vllm-project/vllm/releases) (see caveat below) + version-pinned supported-hardware page |
| Numerics are correct | Does the dispatched kernel produce non-garbage output on *this exact* SM? | Your own eval. Nothing else answers this. |

**Rules for confidence:**
- **Version-pin every doc URL.** `docs.vllm.ai/en/latest/...` and `/en/stable/...` both currently 404 for the quantization supported-hardware page; version-pinned paths such as `/en/v0.8.5/features/quantization/supported_hardware.html` resolve. Pin to the exact release running in production — the matrix shape changes release to release.
- **Read the release notes, not just the docs page.** In practice the vLLM release notes are more current and more SM-specific than the supported-hardware table, which still enumerates only Volta through Hopper.
- Cross-check GitHub issues and PRs by SM number + op type for gaps the docs haven't caught up with.
- Distinguish "kernel exists" from "kernel is dispatched to" from "kernel is numerically correct" in every matrix cell.
- Empirically test as the final tiebreaker. CUTLASS, FlashInfer, and vLLM ship on independent cadences.

---

## 4. Verified Quantization Support Matrix — AWS G6e vs. G7e

**Hardware mapping:**
- **AWS G6e** = up to 8x NVIDIA L40S, Ada Lovelace, compute capability **SM89** — [AWS EC2 G6e](https://aws.amazon.com/ec2/instance-types/g6e/)
- **AWS G7e** = up to 8x NVIDIA RTX PRO 6000 Blackwell Server Edition, compute capability **SM120** — [AWS EC2 G7e](https://aws.amazon.com/ec2/instance-types/g7e/) · [G7e GA announcement, 20 Jan 2026](https://aws.amazon.com/about-aws/whats-new/2026/01/amazon-g7e-instances-generally-available)

**Matrix state as of vLLM v0.22.0 (see §6 for the version timeline).**

| Quant | Dense — SM89 (G6e) | MoE — SM89 (G6e) | Dense — SM120 (G7e) | MoE — SM120 (G7e) |
|---|---|---|---|---|
| **BF16** | ✅ Native | ✅ Native (Triton `fused_experts`, arch-agnostic) | ✅ Native | ✅ Native (Triton, arch-agnostic) |
| **FP8 (W8A8)** | ✅ Native | ✅ Native (`cutlass_fp8` experts) | ✅ Native (CUTLASS SM120 kernels since v0.9.2; blockwise GEMM optimized v0.19.0, swapAB v0.20.0) | ✅ Native |
| **MXFP4** | ⚠️ No native FP4 tensor-core path on Ada — Marlin dequant fallback | ⚠️ Marlin dequant fallback only | ✅ Native Blackwell MXFP4 tensor cores; Triton backend also enabled for SM120 | ⚠️ Native kernels exist; dispatch gap [#31085](https://github.com/vllm-project/vllm/issues/31085) **still open** — verify you aren't silently on Marlin |
| **GPTQ-Int4 (Marlin)** | ✅ Native | ✅ Native (`fused_marlin_moe`) | ✅ Works; not in the official vLLM hardware table | ⚠️ Kernel exists, not explicitly verified for SM120 |
| **NVFP4** | ❌ No native kernel; software emulation only (dequant→BF16) | ❌ No native path; emulation impractical for MoE | ✅ Native (AWS runs NVFP4 models on G7e in production) | ⚠️ **Two paths, different status** — see the NVFP4 note below |

### Citations per row

**BF16** — arch-agnostic baseline dtype. [vLLM Fused MoE kernel features](https://docs.vllm.ai/en/latest/design/moe_kernel_features/); [vLLM GPU installation guide](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/) (compute capability 7.5+ floor).

**FP8** — the vLLM supported-hardware table marks FP8 W8A8 ✅ for Ada. SM120 confirmed by [vLLM v0.9.2 release notes](https://github.com/vllm-project/vllm/releases/tag/v0.9.2): *"SM120: CUTLASS W8A8/FP8 kernels and related tuning, added to Dockerfile (#17280, #19566, #20071, #19794)"*, originating in [PR #17280](https://github.com/vllm-project/vllm/pull/17280) ("Support Cutlass w8a8 FP8 for Blackwell Geforce GPUs (sm120)"). Subsequently optimized: [v0.19.0](https://github.com/vllm-project/vllm/releases/tag/v0.19.0) (SM120 CUTLASS blockwise FP8 GEMM, #37970), [v0.20.0](https://github.com/vllm-project/vllm/releases/tag/v0.20.0) (swapAB for SM120 blockwise FP8, #38325), [v0.22.0](https://github.com/vllm-project/vllm/releases/tag/v0.22.0) (per-tensor FP8 CUTLASS on SM12.1, #41215). Independently corroborated by the [TensorRT-LLM release notes](https://nvidia.github.io/TensorRT-LLM/release-notes.html) ("Added FP8 support for SM120 architecture").

**MXFP4** — ⚠️ **Corrected in rev 2.** Rev 1 stated that native MXFP4 tensor-core acceleration "requires compute capability ≥9.0." That is wrong, and it conflated a software gate with hardware capability. The facts:

- **Native FP4 tensor cores start with Blackwell.** Hopper does not have them. Per the [vLLM gpt-oss launch post](https://blog.vllm.ai/2025/08/05/gpt-oss.html), vLLM ships two MXFP4 MoE kernels: a FlashInfer kernel using *Blackwell's native MXFP4 tensor cores* on B200, and the OpenAI Triton `matmul_ogs` kernel on H100/H200. NVIDIA states this directly in the [Nemotron 3 Ultra paper](https://arxiv.org/pdf/2606.15007): the single NVFP4 checkpoint "targets both Blackwell, where it runs with native FP4 math, and Hopper, where it runs as W4A16… because Hopper lacks native FP4 tensor cores."
- **The ≥9.0 number is a framework capability check**, not a tensor-core threshold — the `MXFP4 quantized models is only supported on GPUs with compute capability >= 9.0` error originates in transformers/vLLM's guard logic ([vLLM issue #23203](https://github.com/vllm-project/vllm/issues/23203) tracks relaxing it toward 7.5).
- **The SM89 conclusion is unchanged** — Ada has no native FP4 tensor-core path and falls back to Marlin dequantization. Only the stated reason was wrong.
- SM120 Triton backend enabled in [PR #31089](https://github.com/vllm-project/vllm/pull/31089), with the author noting Triton performed *worse* than Marlin on RTX PRO 6000 at batch 1. SM120 dispatch gap tracked in [#31085](https://github.com/vllm-project/vllm/issues/31085), opened 20 Dec 2025 and **still open as of Aug 2026** (labelled `unstale`).

**GPTQ-Int4** — vLLM supported-hardware table shows GPTQ and Marlin (GPTQ/AWQ/FP8) ✅ for Ada. SM120 support attested by community reports requiring a `--attention-backend TRITON_ATTN` workaround. *Rev 2 note: the community source was not re-verified in this pass. Treat SM120 GPTQ as "reported working, unaudited."*

**NVFP4** — ⚠️ **Substantially revised in rev 2.**

*Dense:* no native off-Blackwell path — [vLLM NVFP4 emulation kernel docs](https://docs.vllm.ai/en/latest/api/vllm/model_executor/kernels/linear/nvfp4/emulation/) ("Software emulation fallback for NVFP4 (dequant → BF16 matmul)"). SM120 native production use confirmed by the [AWS SageMaker G7e blog](https://aws.amazon.com/blogs/machine-learning/accelerate-generative-ai-inference-on-amazon-sagemaker-ai-with-g7e-instances/), which names GPT-OSS-120B, **Nemotron-3-Super-120B-A12B (NVFP4 variant)**, and Qwen3.5-35B-A3B as single-node G7e.2xlarge targets.

*MoE:* rev 1 said "broken/unstable, FlashInfer `b12x` is the working alternative," citing a forum post as evidence. **That was internally inconsistent** — the cited post says FlashInfer was broken. These are two different FlashInfer backends:

| Path | Status |
|---|---|
| FlashInfer **CUTLASS** fused MoE (TMA WS grouped GEMM) | ❌ Broken on SM120. All 80 TMA WS tactics fail at initialization; falls through to non-TMA tactics producing garbage (6–7 tok/s) or degraded throughput. Root cause is upstream in CUTLASS ([#3096](https://github.com/NVIDIA/cutlass/issues/3096), no NVIDIA response at time of report). Source: [SM120 NVFP4 MoE performance report, 11 Apr 2026](https://discuss.vllm.ai/t/sm120-rtx-pro-6000-nvfp4-moe-performance-report-qwen3-5-397b/2536) — at that date the *only* correct-output backend was Marlin W4A16 at 50.5 tok/s on 4x RTX PRO 6000. |
| FlashInfer **b12x** CuTe-DSL fused MoE | ✅ Merged upstream. A separate, later implementation: [`b12x_fused_moe`](https://docs.flashinfer.ai/generated/flashinfer.fused_moe.b12x_fused_moe.html) — "Run fused MoE on SM120/SM121 using b12x CuTe-DSL kernels." Integrated into vLLM by [PR #40082](https://github.com/vllm-project/vllm/pull/40082) (FlashInfer PRs #3051, #3066, #3080) and shipped in [v0.22.0](https://github.com/vllm-project/vllm/releases/tag/v0.22.0): *"FlashInfer b12x MoE + FP4 GEMM for SM120/121 (#40082)."* PR reports 24/24 kernel tests passing on SM120/SM121 hardware. |

So the honest current state is: **the CUTLASS NVFP4 MoE path on SM120 is still broken; there is now a merged native alternative via b12x.** Known rough edge: [FlashInfer issue #3383](https://github.com/flashinfer-ai/flashinfer/issues/3383) reports b12x illegal-address failures on sm_121a with EP>1. Also note [vLLM #35566](https://github.com/vllm-project/vllm/issues/35566) (CUDA illegal memory access, MoE, NVFP4, SM120).

> **Benchmark hygiene.** The same forum report debunks community claims of 130–150 tok/s on 4x RTX PRO 6000, finding the forks contained *zero kernel-level changes* and that the numbers likely counted proposed-then-rejected speculative tokens. When evaluating SM120 throughput claims, confirm the figure is sustained output tokens delivered to the client, not burst or speculative-inclusive counts.

---

## 5. Key Caveats

1. **vLLM's official "Supported Hardware for Quantization Kernels" page still has no Blackwell/SM120 column.** It enumerates Volta 7.0, Turing 7.5, Ampere 8.0/8.6, Ada 8.9, Hopper 9.0. Every SM120 claim in §4 is verified against release notes, PRs, and issues rather than that page.
2. **The page URL is unstable.** `/en/latest/` and `/en/stable/` variants 404 as of this revision. Use version-pinned paths.
3. **SM120 ≠ SM121.** RTX PRO 6000 / RTX 50-series are SM12.0; GB10 (DGX Spark) is SM12.1. Kernels compiled for 12.0 can pass support checks on 12.1 and then produce NaN. If your fleet includes both, test both.
4. **Anything marked ⚠️ is "best available evidence, pending official confirmation."** Re-verify whenever vLLM, CUTLASS, or FlashInfer versions are bumped.
5. **Unaudited in this revision:** the A100/H100/H200/GB200/GB300/HGX/Blackwell-arch spec blocks in §1, the community GPTQ-on-SM120 report, and one Hacker News reference from rev 1 that has been dropped rather than carried forward unverified.

---

## 6. Version Timeline — Why §4 Rots

Rev 1's SM120 assessments were anchored to vLLM 0.13 / December 2025 while the document was dated August 2026. Roughly nine releases of SM12x work landed in between:

| Release | SM12x-relevant changes |
|---|---|
| v0.9.2 (Jul 2025) | SM120 CUTLASS W8A8/FP8 kernels + tuning, added to Dockerfile |
| v0.13.0 (Dec 2025) | MXFP4 backend selection does not recognize SM120 → Marlin fallback (#31085 filed) |
| v0.19.0 | Optimized SM120 CUTLASS blockwise FP8 GEMM; fix NVFP4 NaN on desktop Blackwell |
| v0.20.0 | swapAB for SM120 CUTLASS blockwise FP8 GEMM; tuned `fused_moe` config for RTX PRO 6000 Blackwell; sm_110 (Jetson Thor) added to CUDA 13.0 build targets |
| v0.22.0 | **FlashInfer b12x MoE + FP4 GEMM for SM120/121 (#40082)**; per-tensor FP8 CUTLASS on SM12.1 |

**Practical implication:** date-stamp every cell, not just the document. A cell reading "⚠️ broken (Dec 2025)" in an August 2026 document is not conservative — it is wrong in the direction of under-claiming, which costs you hardware utilisation.

---

## 7. Changelog — What Changed in Revision 2

| # | Rev 1 claim | Verdict | Rev 2 |
|---|---|---|---|
| 1 | "MXFP4 native tensor-core acceleration requires compute capability ≥9.0" | **Wrong** | Native FP4 tensor cores start with Blackwell; Hopper uses the Triton `matmul_ogs` kernel. ≥9.0 is a framework capability check. SM89 conclusion unchanged, reasoning replaced. |
| 2 | NVFP4 MoE SM120: "FlashInfer `b12x` path is the working alternative," cited to the Apr 2026 forum post | **Internally contradictory** | The cited post says FlashInfer *CUTLASS* is broken. Split into two rows: CUTLASS (broken) vs b12x CuTe-DSL (merged in v0.22.0 via #40082). |
| 3 | SM120 status "as of Dec 2025" | **Stale** | Added §6 version timeline through v0.22.0. |
| 4 | Three layers of "missing kernel support" | **Incomplete** | Added a fourth: silent architecture mismatch (passes `is_supported()`, returns NaN). |
| 5 | SM120 treated as monolithic | **Incomplete** | SM120 vs SM121 distinguished throughout. |
| 6 | `docs.vllm.ai/en/latest/.../supported_hardware.html` | **404** | Version-pinning guidance added to §3. |
| 7 | L40S, RTX PRO 6000 SE, G6e/G7e mapping, issue #31085, forum post, v0.9.2 note, SageMaker blog, SASS repo | **All accurate** | Retained; source links normalised, SASS repo flagged as low-weight. |
| 8 | Hacker News reference | **Unverified** | Dropped. |
| 9 | A100/H100/H200/GB200/GB300/HGX specs | **Not re-checked** | Retained and explicitly labelled unaudited. |
