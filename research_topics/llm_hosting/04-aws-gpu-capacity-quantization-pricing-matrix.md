# AWS GPU Instance Matrix: Capacity, Bandwidth, Quantization Support, and Pricing

**Purpose:** one lookup table for "what model fits where, at what quantization, for what price."
**Region:** us-east-1, Linux on-demand. **Observed:** August 2026.
**SageMaker column:** EC2 × 1.25 (the ~25% premium, verified against four anchor instances — see §6).

---

## How to read this

Two quantization columns, and the distinction between them is the single most useful thing in this document:

- **Native HW** — what the tensor cores physically support, per the vendor spec sheet. A hardware property. Doesn't change.
- **Kernel** — what vLLM will actually execute today. A software property. Changes with every release.

They diverge constantly, in both directions. g7e has FP4 silicon but incomplete FP4 kernels. L40S has INT4 silicon that vLLM's W4A16 path never touches. **Buy on Native HW; plan your deployment on Kernel.**

---

## 1. Master table

| Instance | GPUs | SM | GPU mem | Agg. bandwidth | Interconnect | Native HW quants | Kernel-supported (vLLM) | Practical weight ceiling | EC2 $/hr | SageMaker $/hr |
|---|---|---|---|---|---|---|---|---|---|---|
| **g6e.xlarge** | 1× L40S | 89 | 48 GB | 0.86 TB/s | — | BF16, FP8, INT8, INT4 | FP8d✅ FP8moe⚠️T INT8✅ INT4✅ FP4❌ | ~30 GB | **1.8610** | **2.3263** |
| g6e.2xlarge | 1× L40S | 89 | 48 GB | 0.86 TB/s | — | BF16, FP8, INT8, INT4 | same as above | ~30 GB | 2.2421 | 2.8026 |
| g6e.4xlarge | 1× L40S | 89 | 48 GB | 0.86 TB/s | — | BF16, FP8, INT8, INT4 | same | ~30 GB | 3.0042 | 3.7553 |
| g6e.8xlarge | 1× L40S | 89 | 48 GB | 0.86 TB/s | — | BF16, FP8, INT8, INT4 | same | ~30 GB | 4.5286 | 5.6607 |
| g6e.16xlarge | **1**× L40S | 89 | 48 GB | 0.86 TB/s | — | BF16, FP8, INT8, INT4 | same | ~30 GB | 7.5772 | 9.4715 |
| g6e.12xlarge | 4× L40S | 89 | 192 GB | 3.46 TB/s | PCIe Gen4 64 GB/s | BF16, FP8, INT8, INT4 | same | ~145 GB | 10.4926 | 13.1158 |
| g6e.24xlarge | 4× L40S | 89 | 192 GB | 3.46 TB/s | PCIe Gen4 | BF16, FP8, INT8, INT4 | same | ~145 GB | 15.0655 | 18.8319 |
| g6e.48xlarge | 8× L40S | 89 | 384 GB | 6.91 TB/s | PCIe Gen4 | BF16, FP8, INT8, INT4 | same | ~290 GB | 30.1312 | 37.6640 |
| **g7e.2xlarge** | 1× RTX PRO 6000 | 120 | **96 GB** | 1.60 TB/s | — | BF16, FP8, **FP4** | FP8d✅ FP8moe⚠️T INT8❌ INT4✅ FP4⚠️B | ~75 GB | **3.3631** | **4.2039** |
| g7e.4xlarge | 1× RTX PRO 6000 | 120 | 96 GB | 1.60 TB/s | — | BF16, FP8, FP4 | same | ~75 GB | 3.9982 | 4.9978 |
| g7e.8xlarge | 1× RTX PRO 6000 | 120 | 96 GB | 1.60 TB/s | — | BF16, FP8, FP4 | same | ~75 GB | 5.2682 | 6.5853 |
| g7e.12xlarge | 2× RTX PRO 6000 | 120 | 192 GB | 3.19 TB/s | PCIe Gen5 P2P ~128 GB/s | BF16, FP8, FP4 | same | ~150 GB | 8.2861 | 10.3576 |
| g7e.24xlarge | 4× RTX PRO 6000 | 120 | 384 GB | 6.39 TB/s | PCIe Gen5 P2P | BF16, FP8, FP4 | same | ~300 GB | 16.5722 | 20.7153 |
| g7e.48xlarge | 8× RTX PRO 6000 | 120 | 768 GB | 12.78 TB/s | PCIe Gen5 P2P | BF16, FP8, FP4 | same | ~600 GB | 33.1443 | 41.4304 |
| p4d.24xlarge | 8× A100 40GB | 80 | 320 GB | 12.4 TB/s | NVLink3 600 GB/s | BF16, INT8, INT4 (**no FP8**) | FP8❌em INT8✅ INT4✅ FP8moe⚠️T FP4❌ | ~240 GB | ~32.77 | ~40.97 |
| p4de.24xlarge | 8× A100 80GB | 80 | 640 GB | 16.3 TB/s | NVLink3 600 GB/s | BF16, INT8, INT4 (**no FP8**) | same | ~500 GB | ~40.97* | ~51.21* |
| **p5.4xlarge** | **1× H100** | 90 | 80 GB | 3.35 TB/s | — (no GDR) | BF16, FP8, INT8, INT4 | FP8d✅ **FP8moe✅** INT8✅ INT4✅ **W4A8✅** FP4❌ | ~60 GB | **6.88** | **8.60** |
| p5.48xlarge | 8× H100 | 90 | 640 GB | 26.8 TB/s | **NVLink4 900 GB/s** | BF16, FP8, INT8, INT4 | same | ~500 GB | 55.04 | 68.80 |
| p5e.48xlarge | 8× H200 | 90 | 1,128 GB | 38.4 TB/s | NVLink4 900 GB/s | BF16, FP8, INT8, INT4 | same | ~900 GB | *unverified* | *unverified* |
| p5en.48xlarge | 8× H200 | 90 | 1,128 GB | 38.4 TB/s | NVLink4 900 GB/s | BF16, FP8, INT8, INT4 | same | ~900 GB | ~63.30 | ~79.13 |
| **p6-b200.48xlarge** | 8× B200 | 100 | **1,432 GB** | ~64 TB/s | **NVLink5 1.8 TB/s** | BF16, FP8, **FP4**, INT8, INT4 | FP8d✅ **FP8moe✅** INT8✅ INT4✅ **FP4✅native** | ~1,150 GB | **113.93** | **142.42** |
| **p6-b300.48xlarge** | 8× B300 | 100 | **2,144 GB** | ~64 TB/s | NVLink5, PCIe Gen6 host | BF16, FP8, FP4, INT8, INT4 | same, FP4 native | ~1,750 GB | **142.42** | **178.02** |
| p6e-gb200.36xlarge | 4× GB200 | 100 | 740 GB | ~32 TB/s | NVL72 domain | BF16, FP8, FP4, INT8, INT4 | same, FP4 native | ~590 GB | UltraServer only | — |
| u-p6e-gb200x36 | 36× GB200 | 100 | 6,660 GB | ~288 TB/s | **NVL72, 130 TB/s agg** | BF16, FP8, FP4, INT8, INT4 | same, FP4 native | ~5,300 GB | 380.95 (CB) | CB only |
| u-p6e-gb200x72 | 72× GB200 | 100 | 13,320 GB | ~576 TB/s | **NVL72, 130 TB/s agg** | BF16, FP8, FP4, INT8, INT4 | same, FP4 native | ~10,600 GB | 761.90 (CB) | CB only |

`*` p4de price not re-verified this pass. CB = Capacity Blocks (Dallas Local Zone), no standard on-demand rate.

### Legend for the kernel column

| Code | Meaning |
|---|---|
| **FP8d✅** | FP8 dense W8A8 on native CUTLASS `scaled_mm` |
| **FP8moe✅** | FP8 MoE grouped GEMM on native CUTLASS — *only SM90 and SM100* |
| **FP8moe⚠️T** | FP8 MoE falls back to Triton `fused_moe`; correct, not peak |
| **FP8❌em** | No FP8 silicon; runs as BF16 emulation — no benefit |
| **INT8✅ / INT8❌** | INT8 W8A8 native / hard-blocked (vLLM #28856 on SM120) |
| **INT4✅** | GPTQ-Marlin / AWQ-Marlin / `moe_wna16` W4A16 — the purpose-built path |
| **W4A8✅** | Machete int4+fp8 — *SM90 only*, needs `wgmma` |
| **FP4✅native** | NVFP4/MXFP4 on FP4 tensor cores via FlashInfer/CUTLASS |
| **FP4⚠️B** | FP4 silicon present, but kernels gated — needs `sm_120f` build; MoE lands on Marlin |
| **FP4❌** | No FP4 silicon; NVFP4/MXFP4 dequantize to W4A16 |

---

## 2. Monthly cost (730 hours)

| Instance | EC2 $/mo | SageMaker $/mo |
|---|---|---|
| g6e.xlarge | $1,359 | $1,698 |
| g6e.2xlarge | $1,637 | $2,046 |
| g6e.4xlarge | $2,193 | $2,741 |
| g6e.12xlarge | $7,660 | $9,575 |
| g6e.48xlarge | $21,996 | $27,495 |
| **g7e.2xlarge** | **$2,455** | **$3,069** |
| g7e.8xlarge | $3,846 | $4,807 |
| g7e.12xlarge | $6,049 | $7,561 |
| g7e.24xlarge | $12,098 | $15,122 |
| g7e.48xlarge | $24,195 | $30,244 |
| p4d.24xlarge | ~$23,924 | ~$29,905 |
| **p5.4xlarge** | **$5,022** | **$6,278** |
| p5.48xlarge | $40,179 | $50,224 |
| p5en.48xlarge | ~$46,209 | ~$57,762 |
| **p6-b200.48xlarge** | **$83,171** | **$103,964** |
| **p6-b300.48xlarge** | **$103,964** | **$129,955** |
| u-p6e-gb200x36 | $278,095 | CB only |
| u-p6e-gb200x72 | $556,190 | CB only |

**SageMaker caveat that overrides the table:** SageMaker endpoints document a **500 GB model-size limit**. That binds long before the hardware does on p5e/p5en/p6 — a 1.56 TB Kimi K3 cannot be deployed to a SageMaker endpoint regardless of the 2,144 GB sitting in the box. Also set the container download/health-check timeout to its 60-minute maximum for any multi-hundred-GB pull.

---

## 3. Cost per GPU-hour

Normalizes across instance sizes — useful for spotting bad-value shapes.

| GPU | $/GPU-hr (EC2) | $/GPU-hr (SM) | Memory/GPU | $/GB-hr |
|---|---|---|---|---|
| L40S (g6e.xlarge) | **1.86** | 2.33 | 48 GB | 0.039 |
| L40S (g6e.12xlarge) | 2.62 | 3.28 | 48 GB | 0.055 |
| L40S (g6e.16xlarge) | **7.58** ⚠️ | 9.47 | 48 GB | 0.158 ⚠️ |
| RTX PRO 6000 (g7e.2xl) | **3.36** | 4.20 | 96 GB | **0.035** |
| RTX PRO 6000 (multi-GPU) | 4.14 | 5.18 | 96 GB | 0.043 |
| A100 40GB (p4d) | ~4.10 | ~5.12 | 40 GB | 0.102 |
| H100 (p5) | 6.88 | 8.60 | 80 GB | 0.086 |
| H200 (p5en) | ~7.91 | ~9.89 | 141 GB | 0.056 |
| B200 (p6-b200) | 14.24 | 17.80 | 179 GB | 0.080 |
| B300 (p6-b300) | 17.80 | 22.25 | 268 GB | 0.066 |
| GB200 (UltraServer) | 10.58 (CB) | — | 185 GB | 0.057 |

**g7e.2xlarge is the cheapest GPU memory on AWS at $0.035/GB-hr** — better than H100 ($0.086) and B200 ($0.080). That is the entire economic case for the G-series: you are buying capacity, not interconnect.

**g6e.16xlarge is the worst value in the table** — one GPU at $7.58/hr, more per GPU than an H100. It exists for CPU-heavy workloads; never buy it for inference.

---

## 4. What fits where — model recommendations by instance

**Scope:** NVIDIA Nemotron v3, Qwen3.8, Qwen3.6, Qwen3.5, GLM-5.2, Kimi K3, Kimi K2.5, NVIDIA Inference-Optimized Checkpoints, and **DeepSeek V4**. No gpt-oss, no MiniMax, no Llama, no Mistral.

**Instance sizes shown** are only those that differ in **GPU count or GPU memory**. Sizes that add vCPU/RAM without adding GPU capacity (g6e.2/4/8/16xlarge, g7e.4/8xlarge, p5en) are omitted — they cannot run a larger model.

### 4.0 Verified model inventory

**All sizes below are GiB, measured from the HuggingFace file tree.** This matters: HuggingFace's web UI displays **decimal GB** (10⁹ bytes) while NVIDIA quotes GPU VRAM in **GiB-equivalent** (a "96 GB" card reports ~95.6 GiB to `nvidia-smi`). The two differ by a factor of **1.0737**, so reading HF's displayed number against VRAM overstates every model by ~7.4%. Earlier drafts of this document made exactly that error. **Use GiB on both sides.**

#### Fits one GPU — g6e (48 GB / ~44.7 GiB usable) or g7e (96 GB / ~95.6 GiB)

| Model | Format | **GiB** | Active | Collection |
|---|---|---|---|---|
| Nemotron-3-Nano-4B-GGUF | GGUF | 2.64 | 4B | Nemotron v3 |
| Nemotron-3-Nano-4B-FP8 | FP8 | 4.92 | 4B | Nemotron v3 |
| Nemotron-3-Nano-4B-BF16 | BF16 | 7.42 | 4B | Nemotron v3 |
| Nemotron-3-Nano-30B-A3B-NVFP4 | NVFP4 | **18.03** | 3B | Nemotron v3 / IOC |
| Nemotron-3.5-Lightning-30B-A3B-NVFP4 | NVFP4 | **20.10** | 3B | Nemotron v3 |
| Qwen3.6-27B-NVFP4 | NVFP4 | **20.43** | 27B dense | IOC |
| Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4 | NVFP4 | 20.89 | 3B | Nemotron v3 |
| Qwen3.6-35B-A3B-NVFP4 | NVFP4 | **21.85** | 3B | IOC |
| **Qwen3.5-35B-A3B-GPTQ-Int4** | INT4 | **22.78** | 3B | Qwen3.5 |
| Qwen3.5-27B-GPTQ-Int4 ⚠️ | INT4 | 28.18 | 27B dense | Qwen3.5 |
| **Qwen3.5 / 3.6 / 3.8-27B-FP8** | FP8 | **28.77** | 27B dense | Qwen3.5/3.6/3.8 |
| Nemotron-3-Nano-30B-A3B-FP8 | FP8 | 30.46 | 3B | Nemotron v3 / IOC |
| Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8 | FP8 | 32.79 | 3B | Nemotron v3 |
| Qwen3.5 / 3.6-35B-A3B-FP8 | FP8 | 34.92 | 3B | Qwen3.5/3.6 |
| Qwen3.5 / 3.6 / 3.8-27B (BF16) | BF16 | 51.77 | 27B dense | Qwen |
| Nemotron-3-Nano-30B-A3B-BF16 | BF16 | 58.84 | 3B | Nemotron v3 |
| Nemotron-3.5-Lightning-30B-A3B-BF16 | BF16 | 61.32 | 3B | Nemotron v3 |
| Qwen3.5 / 3.6-35B-A3B (BF16) | BF16 | 66.99 | 3B | Qwen |
| **Qwen3.5-122B-A10B-GPTQ-Int4** | INT4 | **73.49** | 10B | Qwen3.5 |
| **Nemotron-3-Super-120B-A12B-NVFP4** | NVFP4 | **74.85** | 12B | Nemotron v3 / IOC |
| **Qwen3.5-122B-A10B-NVFP4** | NVFP4 | **77.79** | 10B | IOC |

#### Needs 2–4 GPUs

| Model | Format | **GiB** | Active | Collection |
|---|---|---|---|---|
| Qwen3.5-122B-A10B-FP8 | FP8 | 118.46 | 10B | Qwen3.5 |
| Nemotron-3-Super-120B-A12B-FP8 | FP8 | 119.56 | 12B | Nemotron v3 / IOC |
| **DeepSeek-V4-Flash** (native) | native | **148.67** | — | DeepSeek V4 |
| DeepSeek-V4-Flash-DSpark | native + draft | 155.44 | — | DeepSeek V4 |
| **DeepSeek-V4-Flash-NVFP4** | NVFP4 | **156.75** | — | IOC |
| **Qwen3.5-397B-A17B-GPTQ-Int4** | INT4 | **219.58** | 17B | Qwen3.5 |
| **Qwen3.5-397B-A17B-NVFP4-V2** | NVFP4 | **226.92** | 17B | IOC |
| Nemotron-3-Super-120B-A12B-BF16 | BF16 | 230.27 | 12B | Nemotron v3 |
| Qwen3.5-122B-A10B (BF16) | BF16 | 233.01 | 10B | Qwen3.5 |
| Qwen3.5-397B-A17B-NVFP4 (V1) | NVFP4 | 233.99 | 17B | IOC |
| DeepSeek-V4-Flash-Base | native | 274.45 | — | DeepSeek V4 |
| **Nemotron-3-Ultra-550B-A55B-NVFP4** | NVFP4 | **328.18** | 55B | Nemotron v3 / IOC |
| **Qwen3.5-397B-A17B-FP8** | FP8 | **378.30** | 17B | Qwen3.5 |

#### Needs 8 GPUs or an UltraServer

| Model | Format | **GiB** | Active | Collection |
|---|---|---|---|---|
| **GLM-5.2-NVFP4** | NVFP4 | **432.95** | 40B | IOC |
| Kimi-K2.5 / K2.6 / K2.7-Code | native MXFP4 | **554.33** | 32B | Kimi K2.5 |
| Kimi-K2.6-NVFP4 | NVFP4 | 554.34 | 32B | IOC |
| **GLM-5.2-FP8** | FP8 | **703.77** | 40B | GLM-5.2 |
| Qwen3.5-397B-A17B (BF16) | BF16 | 751.41 | 17B | Qwen3.5 |
| **DeepSeek-V4-Pro** (native) | native | **805.35** | — | DeepSeek V4 |
| DeepSeek-V4-Pro-DSpark | native + draft | 831.45 | — | DeepSeek V4 |
| **DeepSeek-V4-Pro-NVFP4** | NVFP4 | **850.42** | — | IOC |
| Nemotron-3-Ultra-550B-A55B-BF16 | BF16 | 1,044 (1.02 TiB) | 55B | Nemotron v3 |
| GLM-5.2 (BF16) | BF16 | 1,403 (1.37 TiB) | 40B | GLM-5.2 |
| **Kimi-K3** | native MXFP4+MXFP8 | **1,454 (1.42 TiB)** | 104B | Kimi K3 |
| **Kimi-K3-NVFP4** | NVFP4 | **1,495 (1.46 TiB)** | 104B | IOC |
| Qwen3.8-2.4T-A95B-FP8 | FP8 | 2,324 (2.27 TiB) | 95B | Qwen3.8 |
| Qwen3.8-2.4T-A95B (BF16) | BF16 | 4,557 (4.45 TiB) | 95B | Qwen3.8 |

#### Corrections this data forced

**Three checkpoints I said don't exist, do.** `nvidia/Qwen3.5-397B-A17B-NVFP4-V2` (**226.92 GiB**) — asserted absent twice, and it is real *and* ~3% smaller than V1 at 233.99 GiB, so **prefer V2**. `nvidia/GLM-5.2-NVFP4` (432.95 GiB) — asserted absent once. And `nvidia/Kimi-K3-NVFP4` (1,495 GiB) exists, which I never listed at all. Every one of those errors came from asserting a negative against an incomplete listing rather than checking the model URL.

**`nvidia/Qwen3.5-122B-A10B-NVFP4` is 77.79 GiB, not ~65 GB.** The HF parameter-count label was wrong by 20%, as suspected.

**Kimi K2.5, K2.6 and K2.7-Code are byte-identical in size** (554.33 GiB each), and `nvidia/Kimi-K2.6-NVFP4` is 554.34 GiB — within 0.01 GiB of the native checkpoint. The NVIDIA "NVFP4" repack buys essentially **nothing** in footprint over Moonshot's native MXFP4 release. Choose between them on kernel path, not size.

**Compression ratios that actually hold** (measured, GiB, from the BF16 parent):

| Format | Ratio vs BF16 | Example |
|---|---|---|
| FP8 | **~1.80×** | Qwen3.5-35B-A3B 66.99 → 34.92 |
| NVFP4 | **~3.07×** | Nemotron-3-Super 230.27 → 74.85 |
| INT4 (GPTQ) | **~2.94×** | Qwen3.5-35B-A3B 66.99 → 22.78 |

Note FP8 is 1.80×, not the clean 2× — and NVFP4's 3.07× matches NVIDIA's own stated "approximately 3.06x" exactly. INT4-GPTQ compresses slightly *less* than NVFP4 (2.94× vs 3.07×) but lands smaller in absolute terms on the models where both exist, because GPTQ leaves fewer auxiliary tensors in high precision: Qwen3.5-122B is **73.49 GiB at INT4 vs 77.79 GiB at NVFP4**.

**The 27B dense anomaly resolves.** Qwen3.5-27B-GPTQ-Int4 is 28.18 GiB against 28.77 GiB for the FP8 build — a 4-bit checkpoint essentially the same size as its 8-bit sibling. For a *dense* model, so much weight sits in attention, embeddings and `lm_head` (all kept high-precision) that 4-bit buys almost nothing. Combined with the community report of `!!!!` output, there is no reason to use it. **4-bit only pays on MoE**, where the experts dominate.


### g6e.xlarge — 1× L40S, 48 GB (44.7 GiB usable), SM89 · $1.8610/hr · SM $2.3263/hr

| Category | Model | Why |
|---|---|---|
| **Smartest** | `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` (**22.78 GiB**) | Most total parameters that fits with real headroom — ~22 GiB left for KV in 44.7 GiB usable. First-party INT4 on the purpose-built `moe_wna16` kernel. |
| **Fastest decode** | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` (**20.10 GiB**) | 3B active ≈ 2 GB swept/token. Ships matched DSpark and DFlash draft models; 1M context; OpenMDW-1.1. NVIDIA's SM12x recipe uses `--moe-backend marlin`. |
| **Best prefill / high batch** | `Qwen/Qwen3.8-27B-FP8` (**28.77 GiB**) | Dense FP8 is the only fully-native CUTLASS path on SM89. Newest Qwen dense. Leaves ~16 GiB for KV — workable, not generous. |
| **Safest deployment** | `Qwen/Qwen3.6-27B-FP8` (**28.77 GiB**) | 7.95M downloads, native CUTLASS FP8, no broken-checkpoint reports. **Not** `Qwen3.5-27B-GPTQ-Int4` — at 28.18 GiB it saves nothing over FP8 and is community-reported broken. |
| **BEST OVERALL** | **`Qwen/Qwen3.5-35B-A3B-GPTQ-Int4`** (**22.78 GiB**) | 3B active for latency, 35B total for capability, native INT4 kernel, ~20 GB left for KV. Measured ~194–197 tok/s on a same-family SM120 card. The cheapest genuinely useful agent tier on AWS. |

*L40S has no FP4 silicon, so NVFP4 and INT4 both execute as W4A16 Marlin here — operationally equivalent, so choose on checkpoint quality rather than format.*

### g6e.12xlarge — 4× L40S, 192 GB, SM89 · $10.4926/hr · SM $13.1158/hr

| Category | Model | Why |
|---|---|---|
| **Smartest** | `Qwen/Qwen3.5-122B-A10B-FP8` (~122 GB, TP=4) | Verified BFCL-V4 72.2 / IFEval 93.4 — the strongest published agentic scores in this size class. |
| **Fastest decode** | `nvidia/Qwen3.5-122B-A10B-NVFP4` (65 GB, TP=2) | 10B active at 4-bit ≈ 5 GB/token. Halves the TP degree versus FP8. |
| **Best prefill / high batch** | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8` (~124 GB, TP=4) | Native FP8 dense compute; 12B active. |
| **Safest deployment** | `Qwen/Qwen3.5-122B-A10B-GPTQ-Int4` (~70 GB, TP=2) | First-party INT4, 570k downloads, lowest TP degree on PCIe Gen4. |
| **BEST OVERALL** | **Don't buy this shape.** Run `Qwen3.5-122B-A10B-GPTQ-Int4` on **g7e.2xlarge** instead | The same model fits **one** g7e GPU at $3.36/hr versus $10.49/hr here, with no TP and no 64 GB/s PCIe Gen4 all-reduce. 3× cheaper and faster. |

### g6e.48xlarge — 8× L40S, 384 GB, SM89 · $30.1312/hr · SM $37.6640/hr

| Category | Model | Why |
|---|---|---|
| **Smartest** | `Qwen/Qwen3.5-397B-A17B-GPTQ-Int4` (236 GB, TP=8) | BFCL-V4 72.9, TAU2-Bench 86.7 — best agentic scores available at any size in these collections. |
| **Fastest decode** | `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4` (~335 GB, TP=8) | Fits with headroom; but 55B active makes it the *slowest*-decoding option here — pick it for capability, not speed. |
| **Best prefill / high batch** | `Qwen3.5-397B-A17B-GPTQ-Int4` | No FP8 checkpoint of this size fits 384 GB; INT4 is the only viable path. |
| **Safest deployment** | `Qwen3.5-397B-A17B-GPTQ-Int4` | No FP4 PTX dependency, mature `moe_wna16`. |
| **BEST OVERALL** | **`Qwen/Qwen3.5-397B-A17B-GPTQ-Int4`** — but prefer **g7e.24xlarge** | Same model, 4 GPUs instead of 8, $16.57/hr versus $30.13/hr, PCIe Gen5 instead of Gen4. TP=8 over Gen4 is the worst interconnect in this document. |

### g7e.2xlarge — 1× RTX PRO 6000, 96 GB, SM120 · $3.3631/hr · SM $4.2039/hr

| Category | Model | Why |
|---|---|---|
| **Smartest** | `Qwen/Qwen3.5-122B-A10B-GPTQ-Int4` (**73.49 GiB**) | Largest verified-agentic model that fits one GPU. BFCL-V4 72.2, IFEval 93.4. On ~95.6 GiB at `--gpu-memory-utilization 0.9` (~86 GiB) that leaves **~13 GiB for KV** — comfortable with FP8 KV cache. |
| **Fastest decode** | `nvidia/Qwen3.6-35B-A3B-NVFP4` (**21.85 GiB**) | 3B active, 4.48M downloads — most-used checkpoint in the IOC collection. **Measured 168 tok/s on a single RTX PRO 6000 96 GB.** `Nemotron-3.5-Lightning-30B-A3B-NVFP4` (20.10 GiB) matches it and adds DSpark (1.26 GiB) / DFlash (1.10 GiB) draft models. |
| **Best prefill / high batch** | `Qwen/Qwen3.8-27B-FP8` (**28.77 GiB**) | Dense FP8 on native CUTLASS `scaled_mm`; ~67 GiB left for KV and batching. Newest Qwen dense. |
| **Safest deployment** | `Qwen/Qwen3.6-27B-FP8` (**28.77 GiB**) | 7.95M downloads, fully native path, zero fallbacks, no FP4 → immune to the `sm_120f` build trap. |
| **BEST OVERALL** | **`Qwen/Qwen3.5-122B-A10B-GPTQ-Int4`** | Smartest *and* fast — 10B active × 4-bit ≈ 5 GB/token beats the 27B dense FP8's 29 GB/token by ~4×. First-party INT4, no FP4 exposure. **Verified 73.49 GiB — it fits with ~13 GiB of KV headroom**, better than earlier drafts feared. Use `--kv-cache-dtype fp8`, `--dtype bfloat16`, and text-only mode to widen it further. Note the NVFP4 sibling is *larger* at 77.79 GiB, so INT4 wins on both footprint and kernel maturity. |

*`nvidia/Qwen3.5-122B-A10B-NVFP4` measures **77.79 GiB** — 4.3 GiB larger than the GPTQ-Int4 build, not smaller as the HF param label suggested.*

*Alternate: `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` — **74.85 GiB**, 2.28M downloads, Mamba2-Transformer hybrid LatentMoE, pre-trained in NVFP4. A field report on that model's discussion page, **on an RTX 6000 Pro specifically**, states: with MTP disabled vLLM serves at roughly a 77 GB footprint, but enabling speculative decoding OOMs even at 0.95 memory utilization against 96 GB. So it runs on g7e.2xlarge without MTP and only just. NVIDIA's own card points single-GPU users at a **B200 or DGX Spark**, not a 96 GB card. Licence is `nvidia-nemotron-open-model-license`, not Apache/MIT — check it before shipping a product.*

### g7e.12xlarge — 2× RTX PRO 6000, 192 GB, SM120 · $8.2861/hr · SM $10.3576/hr

| Category | Model | Why |
|---|---|---|
| **Smartest** | `Qwen/Qwen3.5-122B-A10B-FP8` (~122 GB, TP=2) | Full FP8 precision of the strongest agentic mid-size model. |
| **Fastest decode** | **Two independent TP=1 replicas** of `Qwen3.5-122B-A10B-GPTQ-Int4` | Each fits one 96 GB GPU. No all-reduce, better fault isolation, ~2× aggregate throughput versus one TP=2 endpoint. |
| **Best prefill / high batch** | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8` (**119.56 GiB**, TP=2) | Native FP8 dense; 12B active. |
| **Biggest model that fits** | `nvidia/DeepSeek-V4-Flash-NVFP4` (**156.75 GiB**, TP=2) | New to this roster — fits 192 GiB with ~35 GiB spare. Native DeepSeek build is 148.67 GiB; the DSpark variant (155.44 GiB) bundles a speculative draft. |
| **Safest deployment** | `Qwen/Qwen3.5-122B-A10B-FP8` (**118.46 GiB**, TP=2) | Native dense FP8, no FP4, single node. |
| **BEST OVERALL** | **Two TP=1 replicas of `Qwen3.5-122B-A10B-GPTQ-Int4`** | On a box with no NVLink, two independent replicas beat one tensor-parallel endpoint. Buy this shape for throughput, not for a bigger model. |

### g7e.24xlarge — 4× RTX PRO 6000, 384 GB, SM120 · $16.5722/hr · SM $20.7153/hr

| Category | Model | Why |
|---|---|---|
| **Smartest** | `Qwen/Qwen3.5-397B-A17B-GPTQ-Int4` (**219.58 GiB**, TP=4) | BFCL-V4 72.9, TAU2 86.7. Leaves ~145 GiB for KV in 384 GiB — far roomier than earlier drafts implied. |
| **Fastest decode** | `Qwen/Qwen3.5-397B-A17B-GPTQ-Int4` (17B active) | Not `Nemotron-3-Ultra-550B-A55B-NVFP4` (328.18 GiB) — it fits, but 55B active makes it ~3× slower per token. |
| **Best prefill / high batch** | `nvidia/Qwen3.5-397B-A17B-NVFP4-V2` (**226.92 GiB**, TP=4) | **V2 exists and is 7 GiB smaller than V1** (233.99 GiB) — prefer it. The FP8 variant at 378.30 GiB technically fits 384 GiB but leaves no KV. |
| **Safest deployment** | `Qwen/Qwen3.5-397B-A17B-GPTQ-Int4` | Smallest of the three 397B builds, no FP4 PTX dependency, no MTP penalty. The NVFP4 sibling measured 50.5 tok/s on Marlin with MTP *regressing* 22%. |
| **BEST OVERALL** | **`Qwen/Qwen3.5-397B-A17B-GPTQ-Int4`** | Only sensible ≥397B deployment on G-series. $12,098/mo. Expect TP=4 PCIe overhead — benchmark before assuming it beats the 122B on one GPU at 1/5 the price. |

### g7e.48xlarge — 8× RTX PRO 6000, 768 GB, SM120 · $33.1443/hr · SM $41.4304/hr

| Category | Model | Why |
|---|---|---|
| **Smartest** | `nvidia/GLM-5.2-NVFP4` (**432.95 GiB**, TP=8) | 753B / 40B active, 1M context, MIT. Fits 768 GiB with ~335 GiB spare. NVFP4 runs Marlin here, not native FP4. |
| **Fastest decode** | `nvidia/Qwen3.5-397B-A17B-NVFP4-V2` (**226.92 GiB**, TP=8) | 17B active, most spare VRAM for KV. TP=8 over PCIe still costs ~⅔ of NVLink throughput. |
| **Best prefill / high batch** | `Qwen/Qwen3.5-397B-A17B-FP8` (**378.30 GiB**, TP=8) | Native dense FP8; MoE experts still fall to Triton on SM120. `GLM-5.2-FP8` at 703.77 GiB also *technically* fits 768 GiB but leaves only ~64 GiB for KV — marginal. |
| **Safest deployment** | `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4` (**328.18 GiB**, TP=8) | Comfortable fit, 376k downloads, NVIDIA-published NVFP4. |
| **BEST OVERALL** | **`nvidia/GLM-5.2-NVFP4`** — or buy `p5.48xlarge` instead | At 432.95 GiB this is the strongest model that genuinely fits G-series, and at $24,195/mo it undercuts p5.48xlarge's $40,179/mo. But TP=8 over PCIe delivers ~⅓ of 8× H100, expert-parallel is off the table, and NVFP4 falls to Marlin. Benchmark both before committing — p5 may win on throughput-per-dollar despite the higher sticker. |

### p4d.24xlarge — 8× A100 40 GB, 320 GB, SM80 · ~$32.77/hr · SM ~$40.97/hr

**Critical: A100 has no FP8 silicon.** Every FP8 checkpoint runs BF16-emulated with zero memory or speed benefit. Only INT4, INT8, and BF16 are real options — and among these collections, **only Qwen3.5 ships GPTQ-Int4**.

| Category | Model | Why |
|---|---|---|
| **Smartest** | `Qwen/Qwen3.5-397B-A17B-GPTQ-Int4` (236 GB, TP=8) | Fits 320 GB tightly (~84 GB for KV). NVLink3 at 600 GB/s makes TP=8 genuinely workable here, unlike on G-series. |
| **Fastest decode** | `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` (~21 GB) | 3B active. Wildly oversized hardware for it, but the fastest thing this box can run. |
| **Best prefill / high batch** | `Qwen/Qwen3.5-122B-A10B-GPTQ-Int4` (~70 GB, TP=2) | No FP8 path exists on SM80, so INT4 W4A16 with a low TP degree is the best compute profile available. |
| **Safest deployment** | `Qwen/Qwen3.5-122B-A10B-GPTQ-Int4` | Comfortable fit, NVLink, mature GPTQ-Marlin — the A100's best-supported quantization. |
| **BEST OVERALL** | **`Qwen/Qwen3.5-122B-A10B-GPTQ-Int4`** — but the family is poor value | ~$4.10/GPU-hr for an architecture that cannot run FP8 or FP4. A single g7e.2xlarge runs the same model at $3.36/hr total. **Only choose P4 if you already hold reserved capacity.** |

### p4de.24xlarge — 8× A100 80 GB, 640 GB, SM80 · ~$40.97/hr · SM ~$51.21/hr

| Category | Model | Why |
|---|---|---|
| **Smartest** | `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4` (**328.18 GiB**, TP=8) | Largest model that fits comfortably in 640 GiB; NVFP4 runs W4A16 Marlin on SM80 (NVIDIA validates this path on A100). |
| **Fastest decode** | `Qwen/Qwen3.5-122B-A10B-GPTQ-Int4` (~70 GB, TP=2) | 10B active with a low TP degree. |
| **Best prefill / high batch** | `Qwen/Qwen3.5-397B-A17B-GPTQ-Int4` (236 GB, TP=8) | Still no FP8; INT4 remains the ceiling. |
| **Safest deployment** | `Qwen/Qwen3.5-397B-A17B-GPTQ-Int4` | 640 GB gives ~400 GB of KV headroom — the most comfortable large-model fit in the P4 family. |
| **BEST OVERALL** | **`Qwen/Qwen3.5-397B-A17B-GPTQ-Int4`** | Genuinely comfortable at TP=8 on NVLink3. Still hobbled by the missing FP8 path — p5.48xlarge costs 34% more and unlocks native FP8 MoE. |

### p5.4xlarge — 1× H100 80 GB, SM90 · $6.88/hr · SM $8.60/hr

**The only single-GPU instance in the P-series, and the cheapest place FP8 MoE grouped GEMM runs natively.**

| Category | Model | Why |
|---|---|---|
| **Smartest** | `Qwen/Qwen3.5-122B-A10B-GPTQ-Int4` (**73.49 GiB**) | Fits 80 GiB but leaves only ~2 GiB after overhead — short context only. The NVFP4 build (77.79 GiB) does **not** practically fit. |
| **Fastest decode** | `nvidia/Qwen3.6-35B-A3B-NVFP4` (**21.85 GiB**) | 3B active on 3.35 TB/s — roughly 2× the bandwidth of g7e. NVFP4 runs W4A16 (no FP4 silicon on Hopper). |
| **Best prefill / high batch** | **`Qwen/Qwen3.6-35B-A3B-FP8`** (**34.92 GiB**) | **Native CUTLASS FP8 MoE grouped GEMM** — the thing neither g6e nor g7e can do. 9.47M downloads. |
| **Safest deployment** | `Qwen/Qwen3.8-27B-FP8` (**28.77 GiB**) | Dense FP8 + DeepGEMM on the most mature architecture; ~50 GiB KV headroom. |
| **BEST OVERALL** | **`Qwen/Qwen3.6-35B-A3B-FP8`** | The single best reason to pay 2× g7e.2xlarge: 3B active, native FP8 *including* the MoE path, 2× bandwidth, TP=1. If your agent is latency-bound rather than capability-bound, this beats anything on G-series. |

### p5.48xlarge — 8× H100, 640 GB, SM90 · $55.04/hr · SM $68.80/hr

| Category | Model | Why |
|---|---|---|
| **Smartest** | `Qwen/Qwen3.5-397B-A17B-FP8` (**378.30 GiB**, TP=8) | Full FP8 of the top-scoring agentic model with **native FP8 MoE grouped GEMM** — impossible on any G-series box. ~260 GiB spare in 640 GiB. |
| **Fastest decode** | `nvidia/Qwen3.5-397B-A17B-NVFP4-V2` (**226.92 GiB**, TP=4) | Fewer bytes, half the TP degree. Runs W4A16 on Hopper (no FP4 silicon). |
| **Best prefill / high batch** | `Qwen/Qwen3.5-397B-A17B-FP8` | Native FP8 dense + MoE, NVLink4 at 900 GB/s, DeepGEMM available. This is the architecture FP8 was designed for. |
| **Safest deployment** | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8` (**119.56 GiB**, TP=2) | Small relative to the box; 418k downloads; every path native. |
| **BEST OVERALL** | **`Qwen/Qwen3.5-397B-A17B-FP8`** or **`nvidia/GLM-5.2-NVFP4`** | The 397B runs as designed with native FP8 MoE. `nvidia/GLM-5.2-NVFP4` at 432.95 GiB also fits (W4A16 on Hopper, ~205 GiB spare) and brings 1M context. `DeepSeek-V4-Flash-NVFP4` (156.75 GiB) fits easily. GLM-5.2-**FP8** at 703.77 GiB does **not** — that needs H200. $40,179/mo. |

### p5e.48xlarge — 8× H200, 1,128 GB, SM90 · pricing unverified

| Category | Model | Why |
|---|---|---|
| **Smartest** | `zai-org/GLM-5.2-FP8` (**703.77 GiB**, TP=8) | 753B / 40B active, 1M context, MIT. Comfortable here — ~1,050 GiB leaves ~345 GiB for KV. Native FP8 MoE. |
| **Fastest decode** | `moonshotai/Kimi-K2.6` (**554.33 GiB**, TP=8) | 32B active vs GLM's 40B, on 4.8 TB/s per GPU. Note `nvidia/Kimi-K2.6-NVFP4` is 554.34 GiB — **identical footprint**, so pick on kernel path, not size. |
| **Best prefill / high batch** | `zai-org/GLM-5.2-FP8` | Native FP8 MoE on 38.4 TB/s aggregate. |
| **Safest deployment** | `Qwen/Qwen3.5-397B-A17B-FP8` (**378.30 GiB**) | Fits with ~670 GiB spare — the most comfortable large deployment on this box. |
| **BEST OVERALL** | **`zai-org/GLM-5.2-FP8`** | H200's 141 GB/GPU is what makes 753B viable on one node. If GLM-5.2 is your target model, this is its natural home. |

### p6e-gb200.36xlarge — 4× GB200, 740 GB, SM100 *(UltraServer only)*

| Category | Model | Why |
|---|---|---|
| **Smartest** | `nvidia/GLM-5.2-NVFP4` (**432.95 GiB**) | 753B / 40B active on **native FP4 tensor cores** for the first time in this table, ~305 GiB spare in 740 GiB. `nvidia/Kimi-K2.6-NVFP4` (554.34 GiB) also fits but leaves far less KV. |
| **Fastest decode** | `nvidia/Qwen3.5-397B-A17B-NVFP4-V2` (**226.92 GiB**) | 17B active, native FP4, and **MTP speculative decoding works here** — the −22% Marlin regression disappears on a native path. |
| **Best prefill / high batch** | `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4` (**328.18 GiB**) | Native FP4 at ~10 PFLOPS/GPU makes 55B active affordable in prefill. |
| **Safest deployment** | `Qwen/Qwen3.5-397B-A17B-FP8` (**378.30 GiB**) | Native FP8 dense + MoE, no FP4 kernel novelty. |
| **BEST OVERALL** | **`nvidia/Qwen3.5-397B-A17B-NVFP4-V2`** | The first instance where your NVFP4 checkpoints execute as designed. Only reachable as part of an UltraServer. |

### p6-b200.48xlarge — 8× B200, 1,432 GB, SM100 · $113.9328/hr · SM $142.4160/hr

| Category | Model | Why |
|---|---|---|
| **Smartest** | `nvidia/DeepSeek-V4-Pro-NVFP4` (**850.42 GiB**, TP=8) | Largest model this node holds, on native FP4. Alternatively `nvidia/GLM-5.2-NVFP4` (432.95 GiB) — NVIDIA's own B200/B300 test target, 40% the footprint of GLM-5.2-FP8 (703.77 GiB) with published accuracy parity. |
| **Fastest decode** | `nvidia/Kimi-K2.6-NVFP4` (**554.34 GiB**) | 32B active on native FP4 with ~8 TB/s per GPU. |
| **Best prefill / high batch** | `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4` (**328.18 GiB**) | ~10 PFLOPS FP4/GPU; 1.8 TB/s NVLink5 removes the all-reduce bottleneck entirely. |
| **Safest deployment** | `Qwen/Qwen3.5-397B-A17B-FP8` (**378.30 GiB**) | Every kernel path native, ~1,050 GiB spare, mature format. |
| **BEST OVERALL** | **`nvidia/GLM-5.2-NVFP4`** | NVIDIA's own reference target for this checkpoint. Native FP4, native FP8 MoE, NVLink5 at 1.8 TB/s, and `--enable-expert-parallel` is finally safe to use. $83,171/mo — needs sustained high utilization to beat any API. |

### p6-b300.48xlarge — 8× B300, 2,144 GB, SM100 · $142.416/hr · SM $178.02/hr

| Category | Model | Why |
|---|---|---|
| **Smartest** | **`moonshotai/Kimi-K3`** (**1,454 GiB / 1.42 TiB**) | 2.8T / 104B active. **The only single node in AWS that holds it** — 2,144 GiB leaves ~690 GiB for KV. `nvidia/Kimi-K3-NVFP4` (1,495 GiB) also exists and also fits., keeping the 16-of-896 expert all-to-all inside one NVLink domain. Its native MXFP4 runs on native FP4 tensor cores. |
| **Fastest decode** | `nvidia/Kimi-K2.6-NVFP4` (**554.34 GiB**) | 32B active versus K3's 104B — roughly 3× fewer bytes per token. |
| **Best prefill / high batch** | `nvidia/DeepSeek-V4-Pro-NVFP4` (**850.42 GiB**) | 1.5× the FP4 compute of B200 with ~1,290 GiB spare. `zai-org/GLM-5.2-FP8` (703.77 GiB) is the FP8 alternative. |
| **Safest deployment** | `zai-org/GLM-5.2-FP8` (**703.77 GiB**) | Native FP8 throughout, ~1,440 GiB headroom, no FP4 novelty. |
| **BEST OVERALL** | **`moonshotai/Kimi-K3`** | AWS's own K3 deployment guide targets `ml.p6-b300.48xlarge`. Note `Qwen3.8-2.4T-A95B-FP8` at 2,324 GiB does **not** fit 2,144 GiB — that needs an UltraServer. $103,964/mo. |

### u-p6e-gb200x36 — 36× GB200, 6,660 GB, SM100 · $380.95/hr (Capacity Blocks)

| Category | Model | Why |
|---|---|---|
| **Smartest** | `Qwen/Qwen3.8-2.4T-A95B-FP8` (**2,324 GiB / 2.27 TiB**) | 2.4T / 95B active — largest FP8 model in these collections. Exceeds p6-b300's 2,144 GiB by ~180 GiB, so this is its entry point. |
| **Fastest decode** | `nvidia/Qwen3.5-397B-A17B-NVFP4-V2` (**226.92 GiB**) | Trivially small here; run many replicas rather than one sharded endpoint. |
| **Best prefill / high batch** | `Qwen/Qwen3.8-2.4T-A95B-FP8` | 36 GPUs in one NVLink domain at 130 TB/s aggregate — no all-to-all ever crosses Ethernet. |
| **Safest deployment** | `moonshotai/Kimi-K3` (**1,454 GiB**) | Fits with ~5,200 GiB spare; the least-stressed configuration for a frontier model. |
| **BEST OVERALL** | **`Qwen/Qwen3.8-2.4T-A95B-FP8`** | Capacity-Blocks only, Dallas Local Zone. Justified only by frontier-scale need, not cost. |

### u-p6e-gb200x72 — 72× GB200, 13,320 GB, SM100 · $761.90/hr (Capacity Blocks)

| Category | Model | Why |
|---|---|---|
| **Smartest** | `Qwen/Qwen3.8-2.4T-A95B` (BF16, **4,557 GiB / 4.45 TiB**) | Full-precision frontier deployment — no quantization compromise at all. |
| **Fastest decode** | `Qwen/Qwen3.8-2.4T-A95B-FP8` (**2,324 GiB**) | Half the bytes of BF16 with 95B active; 576 TB/s aggregate bandwidth. |
| **Best prefill / high batch** | `Qwen/Qwen3.8-2.4T-A95B-FP8` | 360 PFLOPS FP8 dense across the domain. |
| **Safest deployment** | Multiple independent replicas of `GLM-5.2-FP8` or `Kimi-K3` | With 13.3 TB you serve several frontier models concurrently rather than sharding one. |
| **BEST OVERALL** | **`Qwen/Qwen3.8-2.4T-A95B` (BF16)** | The only configuration in AWS that runs a 2.4T model unquantized. $556,190/mo — a research or national-scale proposition, not a product one. |

---

### 4.1 Crossover summary

| Model size | Instance | EC2 $/mo | Why it's the boundary |
|---|---|---|---|
| ≤ 23 GiB | g6e.xlarge | $1,359 | Qwen3.5-35B-A3B-GPTQ-Int4 (22.78) with ~22 GiB KV |
| ≤ 78 GiB | **g7e.2xlarge** | **$2,455** | Qwen3.5-122B-A10B-GPTQ-Int4 (73.49) on one GPU, TP=1 |
| ≤ 35 GiB, latency-critical | p5.4xlarge | $5,022 | Only place FP8 MoE is native at 1 GPU |
| ≤ 300 GiB | g7e.24xlarge | $12,098 | Qwen3.5-397B GPTQ-Int4 (219.58) or NVFP4-V2 (226.92), TP=4 |
| ≤ 600 GiB | g7e.48xlarge | $24,195 | GLM-5.2-NVFP4 (432.95) or Kimi-K2.6 (554.33) — largest that fits G-series |
| ≤ 500 GiB | p5.48xlarge | $40,179 | Qwen3.5-397B-FP8 (378.30) with native FP8 MoE + NVLink |
| ≤ 900 GiB | p5e.48xlarge | — | GLM-5.2-FP8 (703.77) |
| ≤ 1,150 GiB, native FP4 | p6-b200.48xlarge | $83,171 | DeepSeek-V4-Pro-NVFP4 (850.42); all formats native |
| ≤ 1,750 GiB | p6-b300.48xlarge | $103,964 | Kimi K3 (1,454), single NVLink domain |
| ≤ 5,300 GiB | u-p6e-gb200x36 | $278,095 | Qwen3.8-2.4T-A95B-FP8 (2,324) |
| ≤ 10,600 GiB | u-p6e-gb200x72 | $556,190 | Qwen3.8-2.4T-A95B BF16 (4,557) |

**For an assistive agent, the practical answer stays at row two.** `Qwen3.5-122B-A10B-GPTQ-Int4` on g7e.2xlarge carries the best verified agentic scores of anything that fits one GPU, decodes ~4× faster than a 27B dense in FP8, and costs $2,455/mo. Everything above it is a capability purchase that needs measurement to justify.

---

## 5. Quantization by architecture — the normalized view

Support is a property of the SM, not the instance size. This is the same information as the master table's two quant columns, without the repetition.

| | SM80 A100 | SM89 L40S | SM90 H100/H200 | SM100 B200/B300 | SM120 RTX PRO 6000 |
|---|---|---|---|---|---|
| **Native HW** | BF16, INT8, INT4 | BF16, FP8, INT8, INT4 | BF16, FP8, INT8, INT4 | BF16, FP8, **FP4**, INT8, INT4 | BF16, FP8, **FP4** |
| BF16 | ✅ | ✅ | ✅ | ✅ | ✅ |
| FP8 dense | ❌ → BF16 | ✅ CUTLASS | ✅ CUTLASS+DeepGEMM | ✅ CUTLASS+DeepGEMM | ✅ CUTLASS |
| **FP8 MoE grouped** | ❌ Triton | ❌ Triton | ✅ **native** | ✅ **native** | ❌ Triton |
| INT8 W8A8 | ✅ | ✅ | ✅ | ✅ | ❌ **blocked** |
| INT4 W4A16 | ✅ Marlin | ✅ Marlin | ✅ Marlin + Machete | ✅ Marlin | ✅ Marlin |
| W4A8 | ❌ | ❌ | ✅ **SM90 only** | ❌ | ❌ |
| NVFP4/MXFP4 dense | ❌ → W4A16 | ❌ → W4A16 | ❌ → W4A16 | ✅ **native** | ⚠️ needs `sm_120f` |
| NVFP4/MXFP4 MoE | ❌ → Marlin | ❌ → Marlin | ❌ → Marlin | ✅ **native** | ⚠️ → Marlin |
| DeepEP (MoE dispatch) | ✅ NVLink | ❌ no NVLink | ✅ | ✅ | ❌ no NVLink |

**Five facts worth memorizing:**

1. **FP8 starts at SM89.** A100 has no FP8 silicon — an FP8 checkpoint on p4d runs BF16-emulated with zero benefit.
2. **FP8 MoE grouped GEMM works only on SM90 and SM100.** vLLM gates it to compute capability 90–109; SM89 fails below, SM120 fails above. This is the clearest capability P-series has that G-series lacks.
3. **FP4 silicon exists on SM100 and SM120 only** — and only SM100 has complete kernels. Hopper has no FP4 at all.
4. **W4A8 (Machete) is SM90-exclusive.** H100 has a capability B200 does not, because Blackwell replaced `wgmma` with `tcgen05`.
5. **INT4 W4A16 is the only format native on every row.** It needs just integer→float conversion plus BF16 MMA. That universality is why it is the safest default across heterogeneous fleets.

---

## 6. Notes and caveats

**SageMaker pricing** is EC2 × 1.25. The ~25% premium was verified against four anchors (g7e.2xlarge, g7e.12xlarge, g7e.24xlarge, g6e.48xlarge); the rest are computed. P-family SageMaker rates assume the same premium and are estimates. Confirm in the console before budgeting. SageMaker AI Savings Plans offer up to 64% off for 1–3 year commitments.

**Bandwidth figures:** L40S (864 GB/s) and RTX PRO 6000 (1,597 GB/s) are from NVIDIA spec pages. A100/H100/H200 are NVIDIA architecture figures. **B200/B300 at ~8 TB/s is widely reported but not confirmed** in the AWS or NVIDIA pages consulted — NVIDIA publishes no standalone B200 spec page because it ships only inside HGX/DGX systems. Aggregate figures are per-GPU × count.

**"Practical weight ceiling" is a planning heuristic**, not a hard limit. Actual headroom depends on context length, concurrency, and KV dtype. Hybrid-attention models can exceed it; long-context high-concurrency serving will fall short.

**All model sizes are now verified in GiB** from each checkpoint's HuggingFace file tree — no estimates remain in the inventory.

**The GB / GiB trap.** HuggingFace's web UI reports **decimal GB** (10⁹ bytes); NVIDIA quotes GPU VRAM in **GiB-equivalent** (a "96 GB" card shows ~95.6 GiB to `nvidia-smi`). The ratio is **1.0737**, and earlier drafts of this document compared HF's GB against VRAM directly — overstating every model by ~7.4% and making several deployments look tighter than they are. Always convert to GiB before doing fit math.

**Three "does not exist" claims in earlier drafts were wrong**: `nvidia/Qwen3.5-397B-A17B-NVFP4-V2` (226.92 GiB, and *smaller* than V1), `nvidia/GLM-5.2-NVFP4` (432.95 GiB), and `nvidia/Kimi-K3-NVFP4` (1,495 GiB), which was never listed. All three came from asserting absence against a truncated collection listing instead of checking the model URL. Treat any remaining absence claim in these documents with suspicion.

**Prices move.** us-east-1 and us-west-2 are at parity for these families; other regions carry premiums. Spot is broadly available on g6e (~$1.38/hr xlarge), thin on g7e, and interruption-prone on P-series.

**Kernel support is a moving target.** SM120 FP4 fixes land frequently in vLLM/FlashInfer/CUTLASS. If native SM120 FP4 MoE stabilizes upstream, the `FP4⚠️B` cells become `FP4✅` and NVFP4 gains roughly 2× compute-bound throughput plus working speculative decoding. Re-check on each release.

---

## 7. Sources

**AWS instance pages**
- G7e — https://aws.amazon.com/ec2/instance-types/g7e/
- G6e — https://aws.amazon.com/ec2/instance-types/g6e/
- P4 — https://aws.amazon.com/ec2/instance-types/p4/
- P5 / P5e / P5en — https://aws.amazon.com/ec2/instance-types/p5/
- P6 / P6e UltraServers — https://aws.amazon.com/ec2/instance-types/p6/
- Capacity Blocks pricing — https://aws.amazon.com/ec2/capacityblocks/pricing
- SageMaker pricing — https://aws.amazon.com/sagemaker/pricing/

**NVIDIA specifications**
- L40S — https://www.nvidia.com/en-in/data-center/l40s/#specifications
- RTX PRO 6000 Blackwell SE — https://www.nvidia.com/en-in/data-center/rtx-pro-6000-blackwell-server-edition/#specs
- A100 (**not** A800 — that is the China-export variant with NVLink cut to 400 GB/s) — https://www.nvidia.com/en-us/data-center/a100/#specifications
- H100 — https://www.nvidia.com/en-us/data-center/h100/#specifications
- H200 — https://www.nvidia.com/en-us/data-center/h200/#specifications
- GB200 NVL72 — https://www.nvidia.com/en-us/data-center/gb200-nvl72/#specs
- GB300 NVL72 — https://www.nvidia.com/en-us/data-center/gb300-nvl72/#specs
- B200/B300 - https://www.nvidia.com/en-us/data-center/hgx/#specifications
- Blackwell architecture (https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)

**Model collections (recommendations in §4 draw only from these)**
- NVIDIA Nemotron v3 — https://huggingface.co/collections/nvidia/nvidia-nemotron-v3
- NVIDIA Inference-Optimized Checkpoints (ModelOpt) — https://huggingface.co/collections/nvidia/inference-optimized-checkpoints-with-model-optimizer
- Qwen3.8 — https://huggingface.co/collections/Qwen/qwen38
- Qwen3.6 — https://huggingface.co/collections/Qwen/qwen36
- Qwen3.5 — https://huggingface.co/collections/Qwen/qwen35
- GLM-5.2 — https://huggingface.co/collections/zai-org/glm-52
- Kimi K3 — https://huggingface.co/collections/moonshotai/kimi-k3
- Kimi K2.5 — https://huggingface.co/collections/moonshotai/kimi-k25
- DeepSeek V4 — https://huggingface.co/deepseek-ai
- **All sizes verified from `<model-url>/tree/main`, reported in GiB**

**Companion documents**
- `01-llm-inference-terminology.md` — kernels, quantization formats, SM targets, why decode is bandwidth-bound
- `02-g6e-g7e-quantization-and-model-selection.md` — G-series deep dive and model recommendations
- `03-p-series-quantization-and-multinode.md` — P-series deep dive and multi-node deployment
