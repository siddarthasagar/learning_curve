# AWS P-Series (p4 / p5 / p6): Quantization Support, Bandwidth, and Multi-Node Deployment

**Scope:** AWS P-series accelerated instances — p4d/p4de (A100), p5/p5e/p5en (H100/H200), p6-b200/p6-b300/p6e-gb200 (Blackwell).
**Companion to:** `02-g6e-g7e-quantization-and-model-selection.md` (G-series, single-instance SLMs).
**Framing:** G-series for models that fit one instance; P-series for large models needing many GPUs or multiple nodes.
**Observation date:** August 2026. Instance tables are verbatim from AWS product pages.

---

## 0. Two corrections before the data

**The A800 page is not the A100.** The link you supplied (`nvidia.com/.../a800/`) is the **China-export variant**, created to comply with US export controls. Its NVLink is cut from 600 GB/s to 400 GB/s. AWS p4d/p4de use the **standard A100**. Use https://www.nvidia.com/en-us/data-center/a100/ instead — the A800 numbers would understate your interconnect by a third.

**There is no standalone B200 or B300 product page** — which is why you couldn't find one. NVIDIA does not sell these as discrete cards the way it sells L40S or RTX PRO 6000. B200/B300 ship only inside **HGX B200 / HGX B300 baseboards** or **DGX B200** systems, so the specs live on those system pages and in the Blackwell architecture page rather than a per-GPU spec table. GB200/GB300 have their own pages because NVL72 is a rack-scale *product*.

The workaround: **AWS publishes enough to derive the per-GPU figures.** See §2.3.

---

## 1. Official AWS instance tables

### 1.1 P4 — NVIDIA A100 (Ampere, SM80)

| Instance | GPUs | GPU memory | vCPU | System RAM | Network | Local NVMe |
|---|---|---|---|---|---|---|
| p4d.24xlarge | 8× A100 | 320 GB HBM2e (40 GB each) | 96 | 1,152 GiB | 400 Gbps EFA | 8 × 1 TB |
| p4de.24xlarge | 8× A100 | 640 GB HBM2e (80 GB each) | 96 | 1,152 GiB | 400 Gbps EFA | 8 × 1 TB |

NVSwitch at 600 GB/s per GPU. Source: https://aws.amazon.com/ec2/instance-types/p4/

### 1.2 P5 — NVIDIA H100 / H200 (Hopper, SM90)

Verbatim from the AWS P5 product-details table:

| Instance | vCPU | Instance memory | GPU | GPU memory | Network | GPUDirect RDMA | GPU peer-to-peer | Local NVMe | EBS |
|---|---|---|---|---|---|---|---|---|---|
| **p5.4xlarge** | 16 | 256 GiB | **1× H100** | 80 GB HBM3 | 100 Gbps EFA | **No** | **N/A** | 3.84 TB | 10 Gbps |
| p5.48xlarge | 192 | 2 TiB | 8× H100 | 640 GB HBM3 | 3,200 Gbps EFA | Yes | **900 GB/s NVSwitch** | 8 × 3.84 TB | 80 Gbps |
| p5e.48xlarge | 192 | 2 TiB | 8× H200 | 1,128 GB HBM3e | 3,200 Gbps EFA | Yes | 900 GB/s NVSwitch | 8 × 3.84 TB | 80 Gbps |
| p5en.48xlarge | 192 | 2 TiB | 8× H200 | 1,128 GB HBM3e | 3,200 Gbps EFA | Yes | 900 GB/s NVSwitch | 8 × 3.84 TB | 100 Gbps |

**Note the asterisk on p5.4xlarge: GPUDirect RDMA is not supported.** Fine for single-GPU serving; disqualifying if you ever wanted to cluster them.

AWS states 900 GB/s NVSwitch gives "a total of 3.6 TB/s bisectional bandwidth in each instance." P5en additionally pairs H200 with Intel Sapphire Rapids and **PCIe Gen5 CPU↔GPU** (4× the CPU-GPU bandwidth of P5/P5e) plus 3rd-gen EFA on Nitro v5 (~35% lower latency than P5). UltraClusters scale to 20,000 H100/H200 GPUs.

Source: https://aws.amazon.com/ec2/instance-types/p5/

### 1.3 P6 — NVIDIA Blackwell / Blackwell Ultra (SM100)

Verbatim from the AWS P6 product-details table:

| Instance | Blackwell GPUs | GPU memory | vCPU | System RAM | Local NVMe | Network | EBS | UltraServer |
|---|---|---|---|---|---|---|---|---|
| **p6-b300.48xlarge** | 8 Ultra | **2,144 GB HBM3e** | 192 | 4,096 GiB | 8 × 3.84 TB | **6.4 Tbps** | 100 Gbps | No |
| **p6-b200.48xlarge** | 8 | **1,432 GB HBM3e** | 192 | 2,048 GiB | 8 × 3.84 TB | 3.2 Tbps | 100 Gbps | No |
| p6e-gb200.36xlarge | 4 | 740 GB HBM3e | 144 | 960 GiB | 3 × 7.5 TB | 3.2 Tbps | 60 Gbps | Yes (only) |

**UltraServers:**

| UltraServer | GPUs | GPU memory | vCPU | System RAM | Storage | Aggregate EFA | EBS |
|---|---|---|---|---|---|---|---|
| **u-p6e-gb200x72** | 72 | **13,320 GB** | 2,592 | 17,280 GiB | 405 TB | 28,800 Gbps | 1,080 Gbps |
| u-p6e-gb200x36 | 36 | 6,660 GB | 1,296 | 8,640 GiB | 202.5 TB | 14,400 Gbps | 540 Gbps |

Key AWS statements:
- P6e-GB200 gives "up to 72 Blackwell GPUs within one NVLink domain to use **360 petaflops of FP8 compute (without sparsity)** and 13.4 TB of total high-bandwidth memory (HBM3e)... up to **130 terabytes per second** of low-latency NVLink connectivity between GPUs and up to 28.8 terabits per second of total EFAv4."
- Versus P5en: "**20x GPU TFLOPS, 11x GPU memory, and 15x aggregate GPU memory bandwidth under NVLink**."
- p6-b200: "up to **14.4 TB/s of total bidirectional NVLink bandwidth**"; each GPU "supports fifth-generation NVLink... up to **1.8 TB/s** of bandwidth per GPU"; second-generation Transformer Engine "supports new precision formats such as **FP4**."
- Grace Blackwell Superchip: 2 Blackwell GPUs + 1 Grace CPU via NVLink-C2C, "**10 petaflops of FP8 compute (without sparsity) and up to 372 GB of HBM3e**" per superchip.
- p6-b300: "2x networking bandwidth, 1.5x GPU memory size, and 1.5x GPU TFLOPS (at FP4, without sparsity)" vs p6-b200; uses **PCIe Gen6**.
- p6e-gb300: "1.5x GPU memory and 1.5x GPU TFLOPS (FP4, without sparsity)" vs p6e-gb200; "close to 20 TB of GPU memory per UltraServer."

Source: https://aws.amazon.com/ec2/instance-types/p6/

---

## 2. GPU silicon

### 2.1 Per-GPU memory, derived from AWS totals

| GPU | Instance | AWS total | Per GPU |
|---|---|---|---|
| A100 40GB | p4d.24xlarge | 320 GB | 40 GB |
| A100 80GB | p4de.24xlarge | 640 GB | 80 GB |
| H100 | p5.48xlarge | 640 GB | 80 GB |
| H200 | p5e/p5en.48xlarge | 1,128 GB | 141 GB |
| **B200** | p6-b200.48xlarge | 1,432 GB | **179 GB** |
| **B300** | p6-b300.48xlarge | 2,144 GB | **268 GB** |
| GB200 (superchip) | p6e-gb200.36xlarge | 740 GB / 4 GPUs | 185 GB (372 GB per 2-GPU superchip) |

B200 is physically 192 GB HBM3e; AWS provisions ~179 GB usable per GPU. Same pattern for B300 (288 GB physical → 268 GB usable).

### 2.2 Bandwidth and interconnect

| GPU | Memory | Bandwidth | NVLink per GPU | Node interconnect |
|---|---|---|---|---|
| A100 80GB | HBM2e | ~2.0 TB/s | NVLink 3 — 600 GB/s | NVSwitch |
| H100 | HBM3 | ~3.35 TB/s | NVLink 4 — **900 GB/s** | NVSwitch (3.6 TB/s bisectional) |
| H200 | HBM3e | ~4.8 TB/s | NVLink 4 — 900 GB/s | NVSwitch |
| B200 | HBM3e | ~8 TB/s | NVLink 5 — **1.8 TB/s** | NVSwitch (14.4 TB/s total bidirectional) |
| B300 | HBM3e | ~8 TB/s | NVLink 5 — 1.8 TB/s | NVSwitch, PCIe Gen6 host |
| GB200 | HBM3e | ~8 TB/s | NVL72 domain — **130 TB/s aggregate** | NVLink across 72 GPUs |
| *L40S (g6e)* | *GDDR6* | *864 GB/s* | *none* | *PCIe Gen4, 64 GB/s* |
| *RTX PRO 6000 (g7e)* | *GDDR7* | *1,597 GB/s* | *none* | *PCIe Gen5 + GPUDirect P2P* |

NVLink and per-GPU HBM bandwidth figures for A100/H100/H200/B200/B300 are NVIDIA's published architecture numbers; the NVLink-per-GPU and NVL72 aggregate figures for Blackwell are confirmed by the AWS P6 page.

**The interconnect gap is the whole reason P-series exists for multi-GPU work:**

```
g7e   PCIe Gen5 P2P        ~128 GB/s   ← your current ceiling
A100  NVLink 3             600 GB/s    ~4.7×
H100  NVLink 4             900 GB/s    ~7×
B200  NVLink 5           1,800 GB/s    ~14×
GB200 NVL72 domain    130,000 GB/s agg. across 72 GPUs
```

Measured consequence on g7e: 8× RTX PRO 6000 at TP=8 delivers roughly **one-third** the aggregate throughput of 8× H100 SXM, and expert-parallel over PCIe collapses to **1.4–2.6 tok/s**. That is the single technical fact that separates "G-series for SLMs" from "P-series for LLMs."

### 2.3 Deriving B200/B300 compute (your missing spec page)

AWS gives two independent anchors that agree:

```
GB200 superchip = 10 PFLOPS FP8 dense ÷ 2 GPUs      = 5 PFLOPS FP8/GPU
GB200 NVL72     = 360 PFLOPS FP8 dense ÷ 72 GPUs    = 5 PFLOPS FP8/GPU  ✓
```

Applying the standard Blackwell 2:1 precision ladder (confirmed on the RTX PRO 6000 spec sheet: FP4 4 PFLOPS / FP8 2 PFLOPS / BF16 1 PFLOP):

| GPU | BF16 | FP8 | FP4 |
|---|---|---|---|
| **B200 / GB200 (per GPU)** | ~2.5 PFLOPS | **5 PFLOPS** | ~10 PFLOPS |
| **B300 (per GPU)** | ~3.75 PFLOPS | ~7.5 PFLOPS | **~15 PFLOPS** (1.5× B200, per AWS) |

All dense, no sparsity. The B300 figure follows AWS's explicit "1.5x GPU TFLOPS (at FP4, without sparsity)." Treat these as **derived, not vendor-published** — the GB200 die may clock differently from an HGX B200 board.

For comparison: H100 SXM is ~1,979 TFLOPS FP8 dense / ~989 TFLOPS BF16 dense, and **has no FP4 at all**.

---

## 3. Quantization support by architecture

### Dense layers

| Format | SM80 (A100/p4) | SM90 (H100·H200/p5) | SM100 (B200·B300·GB200/p6) |
|---|---|---|---|
| BF16 | ✅ native | ✅ native | ✅ native |
| **FP8 W8A8** | ❌ **no FP8 silicon** → BF16 | ✅ **native** CUTLASS + DeepGEMM | ✅ **native** CUTLASS + DeepGEMM |
| INT8 W8A8 | ✅ native (624 TOPS) | ✅ native | ✅ native |
| INT4 (AWQ/GPTQ) | ✅ Marlin W4A16 | ✅ Marlin + **Machete** | ✅ Marlin W4A16 |
| **W4A8** | ❌ | ✅ **SM90 only** (Machete/CUTLASS, `wgmma`) | ❌ |
| **NVFP4 / MXFP4** | ❌ → W4A16 | ❌ **no FP4 silicon** → W4A16 | ✅ **native FP4 tensor cores** |

### MoE layers (grouped GEMM)

| Format | SM80 | SM90 | SM100 |
|---|---|---|---|
| BF16 | ✅ | ✅ | ✅ |
| **FP8 grouped GEMM** | ❌ → Triton | ✅ **native CUTLASS** | ✅ **native CUTLASS** |
| INT4 | ✅ `moe_wna16` | ✅ `moe_wna16` | ✅ `moe_wna16` |
| **NVFP4 / MXFP4 MoE** | ❌ → Marlin | ❌ → Marlin | ✅ **native** (FlashInfer TRTLLM/CUTLASS) |
| DeepEP (low-latency MoE dispatch) | needs NVLink ✅ | ✅ | ✅ |

### Five dividing lines

**1. FP8 arrives at SM90.** A100 has no FP8 silicon — Ampere predates it. An FP8 checkpoint on p4d runs as BF16 emulation, so you get none of the memory or speed benefit. **Do not buy P4 for FP8 workloads.**

**2. FP4 arrives at SM100 — and only there among P-series.** H100 and H200 cannot do FP4 in hardware. Your NVFP4 and MXFP4 checkpoints (Qwen3.5-397B-NVFP4, GLM-5.2-NVFP4, Kimi-K2.6, gpt-oss) run W4A16 dequant on Hopper. **p6 is the only P-series family where those checkpoints execute as designed.**

**3. FP8 MoE grouped GEMM works on SM90 and SM100 only.** vLLM's `cutlass_group_gemm_supported()` accepts compute capability 90–109. A100 fails below (80), and — as established in the G-series doc — both SM89 and SM120 fail too. **Hopper and datacenter Blackwell are the only architectures inside that window.** This is a genuine capability P-series has that G-series does not.

**4. Machete and W4A8 are SM90-exclusive.** Built on Hopper's `wgmma`, which datacenter Blackwell replaced with `tcgen05`. So **H100 has a W4A8 capability that B200 does not** — newer is not uniformly more capable.

**5. DeepEP's low-latency MoE dispatch requires NVLink.** It works across the whole P-series and on no G-series instance. For large MoE models this is decisive: the expert all-to-all is on the decode critical path.

### Contrast with your current hardware

| | g6e (SM89) | g7e (SM120) | p5 (SM90) | p6 (SM100) |
|---|---|---|---|---|
| FP8 dense | ✅ | ✅ | ✅ | ✅ |
| **FP8 MoE** | ❌ Triton | ❌ Triton | ✅ **native** | ✅ **native** |
| INT8 | ✅ | ❌ blocked | ✅ | ✅ |
| INT4 W4A16 | ✅ | ✅ | ✅ | ✅ |
| **FP4** | ❌ no silicon | ⚠️ HW yes, kernel gap | ❌ no silicon | ✅ **native** |
| NVLink | ❌ | ❌ | ✅ 900 GB/s | ✅ 1.8 TB/s |

Two things stand out. **G-series and Hopper share the same FP4 outcome** (W4A16) for opposite reasons — g6e/p5 lack the silicon, g7e has silicon without kernels. And **INT4 W4A16 is the only format native on every single row** — which is why it travels so well.

---

## 4. What fits where

Weight footprints from the model roster, against per-node capacity:

| Node | Total GPU memory | Practical model ceiling (weights + KV) |
|---|---|---|
| g6e.xlarge (1× L40S) | 44.7 GiB usable | ~20–30 GB — SLM territory |
| g7e.2xlarge (1× RTX PRO 6000) | 96 GB | ~70–85 GB — up to a 122B MoE at INT4 |
| g7e.48xlarge (8×) | 768 GB | ~600 GB, but PCIe-bound at TP=8 |
| p5.48xlarge (8× H100) | 640 GB | ~500 GB with NVLink |
| p5e/p5en.48xlarge (8× H200) | 1,128 GB | ~900 GB |
| **p6-b200.48xlarge (8× B200)** | **1,432 GB** | ~1.2 TB |
| **p6-b300.48xlarge (8× B300)** | **2,144 GB** | ~1.8 TB |
| **u-p6e-gb200x36** | 6,660 GB | multi-trillion-parameter |
| **u-p6e-gb200x72** | 13,320 GB | frontier scale |

Applied to the roster:

| Model | Size | Smallest viable P-series | Notes |
|---|---|---|---|
| Qwen3.5-397B GPTQ-Int4 | 236 GB | p5.48xlarge (TP=8) | Also runs g7e.24xlarge |
| Qwen3.5-397B NVFP4 | 251 GB | **p6-b200** for native FP4 | W4A16 on p5 |
| Qwen3.5-397B FP8 | 406 GB | p5.48xlarge (640 GB) | Native FP8 MoE on Hopper |
| GLM-4.5-FP8 | 377 GB | p5.48xlarge | |
| GLM-5.2-NVFP4 | 446 GB | **p6-b200** | Native FP4 |
| Kimi-K2.6 | 552–585 GB | p5e/p5en (1,128 GB) | comfortable |
| **Kimi K3** | **1,560 GB** | **p6-b300 single node (2,144 GB)** | Or 2× p6-b200 across EFA |

**Kimi K3 is the clean illustration of your framing.** It does not fit any G-series node (768 GB max). It needs 2× p6-b200 over EFA, or — better — **a single p6-b300.48xlarge at 2,144 GB, keeping the entire 16-of-896 expert all-to-all inside one NVLink domain.** AWS's own Kimi K3 deployment guide targets `ml.p6-b300.48xlarge` for exactly this reason.

---

## 5. Your SLM / LLM split, validated

Your instinct is right, and AWS states the underlying reason directly. From their July 2026 disaggregated-inference guidance: for such deployments you should "choose instance types that support both NVLink and EFA… This includes the P5 and P6 instance families on AWS," while G6/G6e/G7e "do support EFA with RDMA read/write" but "performance on multi-GPU instances is bottlenecked by GPU-to-GPU communication over PCIe."

So the split is:

| | G-series | P-series |
|---|---|---|
| **Best at** | Single-GPU models, TP=1 | Multi-GPU and multi-node |
| **Interconnect** | PCIe only | NVLink + NVSwitch + EFA RDMA |
| **Cost per GPU-hr** | $1.86–$4.14 | $6.88 (H100) – $14.24 (B200) |
| **Sweet spot** | ≤96 GB models | ≥250 GB models |
| **FP8 MoE** | Triton fallback | Native CUTLASS |
| **FP4** | g7e: kernel gap | p6: native |

**One refinement worth making.** Multi-node is not simply "more GPUs" — crossing a node boundary drops you from NVLink (1.8 TB/s) to EFA (400 GB/s per GPU on P6). For a 16-of-896-expert model like K3, that boundary sits on the decode critical path. **Prefer the largest single node that fits before going multi-node**, and prefer an UltraServer over 2 nodes when you must scale, because an UltraServer is multi-instance in packaging but single-domain in behaviour.

Ordering by how much pain each option causes a large MoE:

```
1 node, fits in NVLink domain      ← best (p6-b300 for K3)
UltraServer NVL72 domain           ← still NVLink, 9 or 18 instances
2 nodes over EFA                   ← all-to-all crosses Ethernet
8 GPUs over PCIe (g7e.48xlarge)    ← expert-parallel collapses
```

---

## 6. Cost (us-east-1 on-demand, August 2026)

| Instance | GPUs | $/hr | $/GPU-hr | $/mo (730h) |
|---|---|---|---|---|
| g6e.xlarge | 1× L40S | 1.8610 | 1.86 | $1,359 |
| g7e.2xlarge | 1× RTX PRO 6000 | 3.3631 | 3.36 | $2,455 |
| g7e.48xlarge | 8× RTX PRO 6000 | 33.1443 | 4.14 | $24,195 |
| p4d.24xlarge | 8× A100 40GB | ~32.77 | ~4.10 | ~$23,922 |
| **p5.4xlarge** | **1× H100** | **6.88** | 6.88 | $5,022 |
| p5.48xlarge | 8× H100 | 55.04 | 6.88 | $40,179 |
| p5en.48xlarge | 8× H200 | ~63.30 | ~7.91 | ~$46,209 |
| p6-b200.48xlarge | 8× B200 | 113.93 | 14.24 | $83,171 |
| p6-b300.48xlarge | 8× B300 | 142.42 | 17.80 | $103,964 |
| u-p6e-gb200x36 | 36× GB200 | 380.95 (CB) | 10.58 | $278,095 |
| u-p6e-gb200x72 | 72× GB200 | 761.90 (CB) | 10.58 | $556,190 |

UltraServer rates are Capacity Blocks list price (Dallas Local Zone); there is no standard on-demand rate. EC2 Capacity Block reserved: $12.355/accelerator-hr (B200), $14.04 (B300). SageMaker adds ~25%.

**p5.4xlarge is the entry point worth knowing** — a single H100 at $6.88/hr, roughly 2× your g7e.2xlarge. For 2× the bandwidth (3.35 vs 1.597 TB/s) and native FP8 MoE, that is a defensible trade if a specific model needs it. It is the only single-GPU option in the entire P-series.

---

## 7. Recommendations

**Keep G-series for everything that fits one GPU.** Your recommended stack — `Qwen3.5-122B-A10B-GPTQ-Int4` on g7e.2xlarge, `Qwen3.5-35B-A3B-GPTQ-Int4` on g6e.xlarge — runs TP=1 and touches none of the interconnect limits. At $1,359–$2,455/mo it is 4–60× cheaper than any P-series node.

**Reach for P-series only when a model genuinely doesn't fit.** The threshold on your hardware is ~85 GB (one g7e GPU at INT4). Below it, P-series buys you nothing an agent workload can use.

**When you do cross that threshold:**

| Need | Instance | Why |
|---|---|---|
| 250–500 GB model, FP8 or INT4 | **p5.48xlarge** | Native FP8 MoE, NVLink 900 GB/s, $40k/mo |
| Model needs native NVFP4/MXFP4 | **p6-b200.48xlarge** | Only P-series with FP4 silicon |
| 500 GB–1.4 TB | p5en (1,128 GB) or p6-b200 (1,432 GB) | Single node beats two |
| 1.4–2.1 TB (Kimi K3) | **p6-b300.48xlarge** | Single NVLink domain; AWS's own K3 target |
| Multi-trillion parameter | u-p6e-gb200x36/x72 | 72 GPUs, one NVLink domain |

**Avoid p4d/p4de for modern inference.** No FP8 silicon, no FP4, no FP8 MoE grouped GEMM. At ~$4.10/GPU-hr it looks cheap next to H100, but you are paying for an architecture that cannot run the quantization formats every 2026 checkpoint ships in. INT8 and INT4-AWQ are its only competitive paths.

**Format choice within P-series:**
- **SM90 (p5):** FP8 for both dense *and* MoE — this is the architecture FP8 was designed for. Machete gives an extra W4A8 option unavailable anywhere else.
- **SM100 (p6):** FP8 or native NVFP4/MXFP4. This is the only place your existing FP4 checkpoints run at full speed, with working speculative decoding.

**The economics haven't changed.** From the earlier break-even work: self-hosting beats a managed API only at sustained high utilization. A p6-b200 node at $83k/mo needs enormous, steady volume to make sense. The G-series recommendation stands not because P-series is worse hardware, but because your workload doesn't need it.

---

## 8. Caveats

- **B200/B300 compute figures in §2.3 are derived**, not vendor-published — from AWS's GB200 superchip (10 PFLOPS FP8 dense / 2 GPUs) and NVL72 (360 PFLOPS / 72 GPUs) anchors, which agree at 5 PFLOPS FP8 per GPU. HGX B200 boards may clock differently from GB200 dies.
- **Per-GPU HBM bandwidth** for A100/H100/H200/B200/B300 comes from NVIDIA architecture materials, not the AWS pages fetched here. The ~8 TB/s Blackwell figure is widely reported but not confirmed in the sources used for this document.
- **p4d/p4de specs and pricing** were not re-fetched this pass; they carry forward from prior research. Verify before budgeting.
- **p5en.48xlarge at ~$63.30/hr** is carried from prior research, not re-confirmed.
- **A100 INT8 at 624 TOPS** is the published A100 figure, not fetched here.
- UltraServer Capacity Block pricing fluctuates with supply and demand per AWS, and is Dallas Local Zone only — not us-east-1 or us-west-2.
- **p6e-gb300 / u-p6e-gb300x72** are described qualitatively by AWS (1.5× memory and FP4 TFLOPS, "close to 20 TB per UltraServer") but did not appear in the instance tables fetched; treat sizing as provisional.
- SageMaker: ml.p5.48xlarge and ml.p6-b200.48xlarge are confirmed available; the 500 GB model-size cap on SageMaker endpoints may bind before the hardware does.

---

## 9. References

**AWS instance pages (primary — instance tables above are verbatim)**
- P4 — https://aws.amazon.com/ec2/instance-types/p4/
- P5 / P5e / P5en — https://aws.amazon.com/ec2/instance-types/p5/
- P6 / P6e UltraServers — https://aws.amazon.com/ec2/instance-types/p6/
- EC2 Capacity Blocks pricing — https://aws.amazon.com/ec2/capacityblocks/pricing

**NVIDIA GPU specifications**
- A100 (**use this, not A800**) — https://www.nvidia.com/en-us/data-center/a100/
- A800 (China-export variant, NVLink 400 GB/s — *not* what p4d uses) — https://www.nvidia.com/en-us/products/workstations/a800/#specifications
- H100 — https://www.nvidia.com/en-us/data-center/h100/#specifications
- H200 — https://www.nvidia.com/en-us/data-center/h200/#specifications
- GB200 NVL72 — https://www.nvidia.com/en-us/data-center/gb200-nvl72/#specs
- GB300 NVL72 — https://www.nvidia.com/en-us/data-center/gb300-nvl72/#specs
- **B200 / B300 have no standalone page** — see HGX platform (https://www.nvidia.com/en-us/data-center/hgx/) and Blackwell architecture (https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)
- Comparison — L40S: https://www.nvidia.com/en-in/data-center/l40s/#specifications · RTX PRO 6000: https://www.nvidia.com/en-in/data-center/rtx-pro-6000-blackwell-server-edition/#specs

**Companion documents**
- `01-llm-inference-terminology.md` — kernels, quantization formats, SM targets
- `02-g6e-g7e-quantization-and-model-selection.md` — G-series, single-instance model selection
