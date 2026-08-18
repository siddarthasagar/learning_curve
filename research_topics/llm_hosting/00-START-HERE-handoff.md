# Handoff Primer — Self-Hosted Agentic LLM on AWS

**Purpose:** paste this into a fresh conversation to restore full context without replaying the research. The five companion documents hold the detail; this is the map and the conclusions.

---

## Who / what / where

Big Data Architect and MLOps engineer, 15 years, AWS + Databricks. Building an **assistive agent product**.

- **Stack:** EKS + vLLM + Karpenter · Strands Agents SDK · MCP as tool transport · AWS Bedrock/AgentCore · Prometheus/Grafana/Langfuse
- **Regions:** us-east-1 primary, us-west-2 secondary
- **Hardware constraint:** **g6e (L40S, SM89) and g7e (RTX PRO 6000 Blackwell, SM120) only** for experimentation. P-series is exploratory.
- **Selection criteria:** tool/function-calling accuracy, structured JSON reliability, multi-step autonomous reasoning. **Coding is explicitly NOT a criterion.**
- **Current baseline:** Claude Sonnet 5 via API; the goal is to find a self-hostable swap.

## The documents

| # | File | What it's for |
|---|---|---|
| 01 | `01-llm-inference-terminology.md` | Kernels, quantization formats, SM targets, why decode is bandwidth-bound |
| 02 | `02-g6e-g7e-quantization-and-model-selection.md` | G-series deep dive, quantization support matrix, deployment configs |
| 03 | `03-p-series-quantization-and-multinode.md` | P-series (p4/p5/p6), NVLink, multi-node |
| 04 | `04-aws-gpu-capacity-quantization-pricing-matrix.md` | **The master lookup** — capacity, bandwidth, quant support, pricing, what-fits-where |
| 05 | `05-sonnet5-replacement-shortlist.md` | **The decision doc** — which models can replace Sonnet 5 |

---

## The ten facts that drive everything

**1. Decode is memory-bandwidth-bound; prefill is compute-bound.** Measured on RTX PRO 6000 (Qwen3.6-27B): decode varies **70%** across quantization formats (BF16 59 → FP8 97–100 → NVFP4 163–169 tok/s) while prefill varies **0.3%** (4,359 / 4,747 / 4,732). Quantization is effectively a decode-only decision.

**2. Decode speed ∝ 1/(ACTIVE params × bytes per param).** A 122B MoE at 4-bit (10B active, ~5 GiB/token) decodes ~4× faster than a 27B dense at FP8 (27B active, ~29 GiB/token). **Sparsity beats size.**

**3. FP8 MoE grouped GEMM works only on SM90 and SM100.** vLLM's `cutlass_group_gemm_supported()` gates to compute capability 90–109. SM89 fails below (89), SM120 fails above (120). On both g6e and g7e, MoE experts fall back to Triton. "FP8 is native on my hardware" is true only for **dense** models.

**4. g7e HAS FP4 silicon** — 4 PFLOPS per NVIDIA's spec, vs 2 PFLOPS FP8 and 1 PFLOP BF16. What's missing is kernel coverage: the FP4 conversion instructions are gated behind `sm_120a`/`sm_120f` (CUDA 12.9+), and SM100's FP4 MoE kernels use `tcgen05`+TMEM which SM120 lacks. **g6e (Ada) genuinely has no FP4 at all.**

**5. A container built for bare `sm_120` loads weights fine, then crashes at the first FP4 kernel launch** with `cudaErrorUnsupportedPtxVersion`. Verify before deploying any FP4 model:
```bash
cuobjdump -lelf $(python -c "import vllm,os;print(os.path.dirname(vllm.__file__))")/_C.abi3.so | grep sm_120
# want sm_120a or sm_120f — bare sm_120 means FP4 will crash
```
**INT4 never touches this** — integer→float conversion is universal since Kepler.

**6. INT4 is the safest 4-bit path, and often the fastest.** Measured on RTX PRO 6000: AWQ-Int4 **259 tok/s, KLD 0.024** vs NVFP4 163–169 tok/s, KLD 0.035. GPTQ-Marlin/`moe_wna16` is the *purpose-built* kernel for INT4, not a fallback. NVIDIA themselves specify `--moe-backend marlin` for SM12x.

**7. INT8 is native on g6e (733 TOPS) but hard-blocked on g7e** (vLLM #28856). One of the few things the older card does that the newer doesn't.

**8. g7e has no NVLink** — PCIe Gen5 + GPUDirect P2P (~128 GB/s vs H100's 900 GB/s). TP=8 on 8× RTX PRO 6000 delivers roughly **⅓** the aggregate of 8× H100 SXM. **Never `--enable-expert-parallel` on g7e** — measured 1.4–2.6 tok/s.

**9. The GB/GiB trap.** HuggingFace's web UI shows **decimal GB**; NVIDIA quotes VRAM in **GiB-equivalent**. Factor 1.0737. Comparing HF's number against VRAM overstates every model by ~7.4%. **All sizes in these documents are GiB.**

**10. Real compression ratios** (measured, from BF16 parents): FP8 **1.80×** (not 2×), NVFP4 **3.07×** (matches NVIDIA's stated 3.06×), INT4-GPTQ **2.94×**. 4-bit only pays on **MoE** — for a dense 27B, GPTQ-Int4 (28.18 GiB) is nearly the same size as FP8 (28.77 GiB).

---

## Verified model sizes (GiB, from HuggingFace `/tree/main`)

| Model | Format | GiB | Active |
|---|---|---|---|
| Nemotron-3-Nano-30B-A3B-NVFP4 | NVFP4 | 18.03 | 3B |
| Nemotron-3.5-Lightning-30B-A3B-NVFP4 | NVFP4 | 20.10 | 3B |
| Qwen3.6-27B-NVFP4 | NVFP4 | 20.43 | 27B dense |
| Qwen3.6-35B-A3B-NVFP4 | NVFP4 | 21.85 | 3B |
| **Qwen3.5-35B-A3B-GPTQ-Int4** | INT4 | **22.78** | 3B |
| Qwen3.5-27B-GPTQ-Int4 ⚠️ | INT4 | 28.18 | 27B dense |
| **Qwen3.5 / 3.6 / 3.8-27B-FP8** | FP8 | **28.77** | 27B dense |
| Qwen3.5 / 3.6-35B-A3B-FP8 | FP8 | 34.92 | 3B |
| **Qwen3.5-122B-A10B-GPTQ-Int4** | INT4 | **73.49** | 10B |
| Nemotron-3-Super-120B-A12B-NVFP4 | NVFP4 | 74.85 | 12B |
| Qwen3.5-122B-A10B-NVFP4 | NVFP4 | 77.79 | 10B |
| Qwen3.5-122B-A10B-FP8 | FP8 | 118.46 | 10B |
| DeepSeek-V4-Flash-NVFP4 | NVFP4 | 156.75 | — |
| **Qwen3.5-397B-A17B-GPTQ-Int4** | INT4 | **219.58** | 17B |
| Qwen3.5-397B-A17B-NVFP4-**V2** | NVFP4 | 226.92 | 17B |
| Qwen3.5-397B-A17B-NVFP4 (V1) | NVFP4 | 233.99 | 17B |
| Nemotron-3-Ultra-550B-A55B-NVFP4 | NVFP4 | 328.18 | 55B |
| Qwen3.5-397B-A17B-FP8 | FP8 | 378.30 | 17B |
| **GLM-5.2-NVFP4** | NVFP4 | **432.95** | 40B |
| Kimi-K2.5 / K2.6 / K2.7-Code | MXFP4 | 554.33 | 32B |
| Kimi-K2.6-NVFP4 | NVFP4 | 554.34 | 32B |
| **GLM-5.2-FP8** | FP8 | **703.77** | 40B |
| DeepSeek-V4-Pro-NVFP4 | NVFP4 | 850.42 | — |
| **Kimi-K3** | MXFP4+MXFP8 | **1,454** | 104B |
| Qwen3.8-2.4T-A95B-FP8 | FP8 | 2,324 | 95B |

`Qwen3.5-27B-GPTQ-Int4` ⚠️ — community-reported `!!!!` output, and at 28.18 GiB saves nothing over FP8. Avoid.

## Instance pricing (us-east-1 EC2 on-demand; SageMaker ≈ ×1.25)

| Instance | GPUs | VRAM | $/hr | $/mo |
|---|---|---|---|---|
| g6e.xlarge | 1× L40S | 48 GB | 1.8610 | $1,359 |
| **g7e.2xlarge** | 1× RTX PRO 6000 | 96 GB | **3.3631** | **$2,455** |
| g7e.12xlarge | 2 | 192 GB | 8.2861 | $6,049 |
| g6e.12xlarge | 4× L40S | 192 GB | 10.4926 | $7,660 |
| g7e.24xlarge | 4 | 384 GB | 16.5722 | $12,098 |
| g6e.48xlarge | 8× L40S | 384 GB | 30.1312 | $21,996 |
| g7e.48xlarge | 8 | 768 GB | 33.1443 | $24,195 |
| p5.4xlarge | 1× H100 | 80 GB | 6.88 | $5,022 |
| p5.48xlarge | 8× H100 | 640 GB | 55.04 | $40,179 |
| p6-b200.48xlarge | 8× B200 | 1,432 GB | 113.93 | $83,171 |
| p6-b300.48xlarge | 8× B300 | 2,144 GB | 142.42 | $103,964 |

**g7e.2xlarge is the cheapest GPU memory on AWS at $0.035/GB-hr** — better than H100 ($0.086) and B200 ($0.080). Never buy g6e.16xlarge (1 GPU at $7.58/hr, worse per-GPU than an H100).

---

## Current recommendation

**Deploy `Qwen/Qwen3.5-122B-A10B-GPTQ-Int4` on `g7e.2xlarge` — $2,455/mo, TP=1.**

73.49 GiB in ~95.6 GiB leaves ~13 GiB for KV at 0.9 utilization. Best verified agentic scores of anything that fits one GPU: **τ²-Bench 79.5, BFCL-V4 72.2, IFBench 76.1, IFEval 93.4, MAXIFE 87.9**. First-party INT4 on `moe_wna16`, no FP4 exposure, 570k downloads.

```bash
export VLLM_CACHE_ROOT=/opt/vllm-cache
export VLLM_USE_DEEP_GEMM=0                 # no SM120 support

vllm serve Qwen/Qwen3.5-122B-A10B-GPTQ-Int4 \
  --tensor-parallel-size 1 \
  --quantization moe_wna16 \
  --dtype bfloat16 \                        # float16 crashes: mixed weight dtypes
  --kv-cache-dtype fp8 \
  --max-model-len 262144 \
  --language-model-only \                   # skip vision encoder, reclaim KV
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \            # NOT qwen3_coder — infinite "!!!!" bug
  --reasoning-parser qwen3
```

**Escalation ladder:** `Qwen3.5-397B-A17B-GPTQ-Int4` on g7e.24xlarge ($12,098/mo) if long-horizon planning falls short (DeepPlanning 24.1 vs 34.3). `nvidia/GLM-5.2-NVFP4` on g7e.48xlarge ($24,195/mo) as the ceiling. **Kimi K3** has the best independent evidence but needs 8× GB300.

**Control:** keep `Qwen3.6-27B-FP8` (28.77 GiB) running as a known-good A/B baseline — fully native CUTLASS path, zero fallbacks.

---

## Open items

**1. Sonnet 5's tool-use scores don't exist publicly.** No τ², τ³, BFCL, MCP-Atlas, or Tool-Decathlon in Anthropic's System Card or launch post. Independent data (Artificial Analysis) gives it **τ³-Banking 28.25%** — *below* Sonnet 4.6's 30.52%. Sonnet 5 is tuned for agentic **coding**, not multi-turn tool orchestration. **It is a beatable target here.**

**2. Benchmarks themselves are unreliable.** The validity audit (arXiv 2607.02577) found **18.5% evaluator-human misalignment** across BFCL v4, τ²-Bench, LiveMCPBench and MCP-Atlas, and a **18.9-point reproducibility spread** on identical LiveMCPBench runs. Public numbers are directional only.

**3. Quantization vs tool calling is largely unmeasured.** Two models have published data and both are reassuring — GLM-5.2-NVFP4 (τ²-Telecom 98.25 vs FP8's 97.9) and Nemotron-3-Super (≤1.6 pt across τ²-Bench V2 domains). **Nothing exists for Qwen3.5 or Kimi.** The best proxy is HumanEval: on Qwen3-32B GPTQ-Int4, MMLU-Pro dropped 1.6 pts while HumanEval dropped **8** — code and tool calls share the "syntactically exact or fail" property.

**4. FP8 loading on Blackwell needs verification.** A community report claims Qwen's DeepSeek-format block-128 FP8 must dequantize to BF16 on Blackwell, contradicting vLLM PR #22131. If a ~29 GiB checkpoint consumes ~58 GiB, it dequantized.

---

## The next action

**Run your own eval before committing.** Several hundred multi-step trajectories against your real MCP schemas, measuring:

- JSON validity rate
- Argument correctness
- Correct-tool selection
- **Negative tool discipline** — correctly declining when no tool applies (a real BFCL category that degrades as your registry grows)
- End-to-end trajectory latency, not tok/s

Compare **FP8 vs INT4 on the same model**, and both against a live Sonnet 5 endpoint. Pin versions, ≥5 repeated runs. Given the 18.5% evaluator misalignment in public benchmarks, **this is the only decision-grade evidence available** — and a week on one g7e.2xlarge produces it.

Practical gotchas already known: `tool_choice='required'` + MTP + thinking mode yields XML instead of JSON at 50–70% failure — use `tool_choice='auto'` or disable thinking. Never `--calculate-kv-scales` on Qwen3.5 hybrids (vLLM #37554 corruption). Enable `--speculative-config` only on a confirmed native kernel path — it regresses 22% on Marlin.
