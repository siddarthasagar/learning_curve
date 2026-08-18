# Self-Hosted Models That Can Replace Claude Sonnet 5 as an Agent Brain

**Question:** which models from the established roster can I swap in for Claude Sonnet 5 and expect the same or better **agent** performance?
**Not the criterion:** coding. SWE-bench, Terminal-Bench, SWE-Bench Pro, LiveCodeBench and Aider are excluded from the filter.
**The criteria:** multi-turn tool use, MCP workflow execution, instruction/format adherence, structured output reliability, multi-step autonomous reasoning.
**Observation date:** August 2026. Companion to `04-aws-gpu-capacity-quantization-pricing-matrix.md`.

---

## 1. Headline answer

*Revised 17 Aug 2026 after a dedicated research pass. Three findings reshuffled the ranking — see §1.1.*

| | Model | Self-host size | Strongest evidence |
|---|---|---|---|
| ✅ **1** | **Kimi K3** | 1,454 GiB — needs p6-b300 | **τ³-Banking 33.4 (#1) vs Sonnet 5's 28.25** — independent |
| ✅ **2** | **Qwen3.5-397B-A17B** | 219.58 GiB (INT4) | τ²-Bench 86.7, Tool-Decathlon 38.3, BFCL-V4 72.9 |
| ✅ **3** | **Qwen3.5-122B-A10B** | **73.49 GiB (INT4) — one g7e GPU** | τ²-Bench **79.5**, BFCL-V4 72.2, IFBench 76.1 |
| ⚠️ **4** | **GLM-5.2** | 432.95 GiB (NVFP4) | MCP-Atlas 76.8 — but τ³-Banking **26.8, below Sonnet 5** |

**The practical tension is unchanged and now sharper:** the model with the best independent evidence (Kimi K3) needs an 8× GB300 node. The one that fits a single GPU at $2,455/month is now *properly documented* and lands mid-pack. That trade is the decision.

### 1.1 What the research pass changed

**Sonnet 5's tool-use scores are definitively unpublished — and independent data puts it lower than expected.** Confirmed across the System Card, launch post, and third-party explainers: no τ², τ³, BFCL, MCP-Atlas, MCPMark, or Tool-Decathlon. The only independent agentic numbers (Artificial Analysis) give Sonnet 5 **τ³-Banking 28.25%** — *below its own predecessor* Sonnet 4.6 at 30.52%. Anthropic tuned Sonnet 5 for agentic **coding** and knowledge work, not multi-turn tool orchestration. **On public evidence, Sonnet 5 is a beatable target for a non-coding agent.**

**Kimi K3 promoted from "insufficient data" to #1.** τ³-Banking **33.4** (top of board), MCP-Atlas **84.2** (#4, vs Fable 5's 84.7), AA-Briefcase **1548 Elo** (#2), Toolathlon-Verified 73.2, BrowseComp 91.2. The only open model with independent evidence of beating Sonnet 5 on tool use. Two caveats: it needs 8× GB300, and its **hallucination rate is poor** — 49% non-hallucination on AA-Omniscience versus GLM-5.2's 72% and Opus 4.8's 64%. For autonomous multi-step work that is a real risk requiring verification guardrails.

**GLM-5.2 demoted from #1 to #4.** Its MCP-Atlas result is real but the card figure is **76.8**, not 77.0, and τ² is **not on the GLM-5.2 card at all** — the 99.1 Telecom number came from Artificial Analysis, not the vendor. Decisively, on the one independent head-to-head harness, **GLM-5.2 scores τ³-Banking 26.8 — below Sonnet 5's 28.25.** Strong on MCP workflows, weaker on multi-turn banking-style agents. Vals AI still ranks it **#1 among open-weight models** on their independent agentic suites, so it remains a serious contender — just not the evidenced leader.

**Qwen3.5-122B-A10B now has a full agentic profile** (Qwen's card, previously not retrieved): τ²-Bench **79.5**, BFCL-V4 72.2, IFBench **76.1**, MAXIFE **87.9**, IFEval 93.4, MultiChallenge 61.5, VITA-Bench 33.6, DeepPlanning 24.1, HLE w/tool 47.5, BrowseComp 63.8. It sits within ~0.5 pt of the 397B on instruction/format adherence and function calling, and falls behind mainly on long-horizon planning (DeepPlanning 24.1 vs 34.3; τ² 79.5 vs 86.7). **It is no longer "unproven" — it is measured, and mid-pack.**

**Qwen3.5-397B's "TAU2 86.7" is confirmed an aggregate**, not a single domain. Qwen publishes no per-domain split, so the airline discriminator can't be isolated.

**Three models stay rejected, with better evidence:** Nemotron-3-Super now has published τ²-Bench V2 (Airline 56.25 / Retail 62.83 / Telecom 64.36, avg 61.15) — clearly below. DeepSeek-V4-Pro τ³-Banking 25.77 and Flash 22.89 — below. Qwen3.8-27B still has no tool-calling data at all.

**Qwen3.8-Max is a genuine "watch."** Toolathlon-Verified 72.5, CoWorkBench 74.8 (above Opus 4.8's 72.3), IFBench 82.8 — frontier-competitive on agentic *work*, but Qwen has published no BFCL/τ²/MCP-Atlas for it, so it can't be scored on the core criteria.

**A methodology finding that qualifies everything above.** The tool-calling validity audit (arXiv 2607.02577, CoreThink AI + Stanford) reviewed 496 expert-annotated tasks across BFCL v4, τ²-Bench, LiveMCPBench and MCP-Atlas and found **92 evaluator-human disagreements — an 18.5% misalignment rate**. Worse, 23 repeated runs of an identical LiveMCPBench setup returned **57.9%–76.8%, a 18.9-point spread "large enough to change leaderboard conclusions."** Every number in this document inherits that uncertainty. **Your own harness, with pinned versions and ≥5 repeated runs, is the only decision-grade evidence.**

**New quantization evidence.** Nemotron-3-Super publishes BF16 / FP8 / NVFP4 across τ²-Bench V2 domains (arXiv 2604.12374): tool-use degradation is **≤1.6 pts and within run noise** — NVFP4 even wins on Retail and IFBench. Combined with the GLM-5.2-NVFP4 data, that is now two independent models showing 4-bit does not damage tool calling. Still nothing published for Qwen3.5 or Kimi.

---

## 2. The Claude Sonnet 5 baseline — and its limits

Sonnet 5 was released **30 June 2026**. Anthropic's published benchmarks are heavily weighted toward coding and computer use:

| Benchmark | Sonnet 5 | Relevant here? |
|---|---|---|
| SWE-bench Verified | 72.7% | ❌ coding |
| SWE-Bench Pro | 63.2% | ❌ coding |
| Terminal-Bench | 76.1% | ❌ coding |
| **OSWorld-Verified** | **81.2%** | ⚠️ computer use, partly agentic |
| Humanity's Last Exam | published with/without tools | ⚠️ reasoning proxy |
| **τ²-bench** | **not published** | ✅ would be primary |
| **MCP-Atlas** | **not published** | ✅ would be primary |
| **BFCL** | **not published** | ✅ would be primary |
| Tool-Decathlon | not published | ✅ would be primary |

**Anthropic has not published Sonnet 5's scores on any of the four benchmarks that matter most for your use case.** Any claim that a model "beats Sonnet 5 on tool calling" is therefore an inference, not a measurement. This document treats that honestly rather than papering over it.

### Bracketing the baseline

Since Sonnet 5's own tool-use numbers are unavailable, its predecessors bound the range:

| Model | τ²-bench Retail | Airline | Telecom | MCP-Atlas |
|---|---|---|---|---|
| Claude Sonnet 4.5 | **86.2%** | **70.0%** | 98.0% | — |
| Claude Sonnet 4.6 | — | — | — | **61.3%** (max effort) |
| Claude Opus 4.6 | ~91.9% | — | ~99.3% | — |

Sonnet 5 sits above Sonnet 4.6 on every published metric, so a reasonable working assumption is **τ²-bench Retail in the high 80s, Airline in the 70s, MCP-Atlas somewhere in the mid-to-high 60s**. Treat those as inferred bounds, not scores.

### Three methodology traps that invalidate naive comparisons

**1. Anthropic's τ²-bench numbers use prompt addenda.** Per Anthropic's own footnote, scores were achieved "using extended thinking with tool use and a prompt addendum to the Airline and Telecom Agent Policy instructing Claude to better target its known failure modes," plus a further addendum to the Telecom user prompt. Vendor τ² scores are tuned-prompt scores, and a vanilla-prompt comparison would land lower.

**2. τ²-bench Telecom saturates.** Sonnet 4.5 scores 98.0% and GLM-5.2 scores 99.1% — both near ceiling. Retail and Airline are the discriminating domains; Airline is hardest (Sonnet 4.5: 70.0%). **A model advertising only a Telecom score has told you nothing.** Note also that the τ²-bench authors found telecom pass^k declines faster than airline as k rises — less consistent, not just harder.

**3. Harnesses are not shared.** GLM's team re-evaluated MCP-Atlas on the 500-task public set with a 10-minute timeout and Gemini 3 Pro as judge, and applied the Opus 4.5 system-card domain fixes to τ²-Airline. Their numbers are internally consistent but not directly comparable to Anthropic's. An independent 2026 validity audit of BFCL, τ-bench, τ²-Bench, LiveMCPBench and MCP-Atlas found scoring reliability problems across the board.

**Consequence: no public benchmark set can definitively settle this. Your own eval harness is the deciding evidence, and this document is a shortlist for what to put in it.**

---

## 3. Model filter table

Scores are vendor-published unless noted. ✅ = clears the Sonnet-5-class bar on agentic criteria; ⚠️ = plausible but unproven; ❌ = below or wrong-purpose.

| Model | Key agentic scores | vs Sonnet 5 baseline | Verdict | Confidence |
|---|---|---|---|---|
| **GLM-5.2** (753B/40B) | MCP-Atlas **77.0** · τ²-Telecom **99.1** · IFBench 75.81 · AA-LCR 70.13 · GPQA-D 89.39 | MCP-Atlas ~+15 pts over Sonnet 4.6's 61.3; Telecom above Sonnet 4.5's 98.0 | ✅ **MATCHES OR BEATS** | **Medium-High** — different MCP-Atlas harness; no Retail/Airline published |
| **Qwen3.5-397B-A17B** (397B/17B) | BFCL-V4 **72.9** · TAU2 **86.7** · IFEval **92.6** · IFBench **76.5** · MAXIFE 88.2 · MultiChallenge 67.6 · **Tool-Decathlon 38.3** · MCP-Mark 46.1 · DeepPlanning 34.3 | TAU2 86.7 ≈ Sonnet 4.5 Retail 86.2; Tool-Decathlon 38.3 vs Claude-4.5-Sonnet's <40 (first place) | ✅ **MATCHES OR BEATS** | **Medium-High** — broadest coverage of any open model here |
| **Qwen3.5-122B-A10B** (122B/10B) | BFCL-V4 **72.2** · IFEval **93.4** | IFEval leads the 397B; but **no τ², no MCP-Atlas, no Tool-Decathlon** | ⚠️ **CLOSE — NEEDS OWN EVAL** | **Low-Medium** — two benchmarks only |
| **Qwen3.6-35B-A3B** (35B/3B) | BFCL v4 **72.9** · reports TAU3-Bench, MCPMark, MCP-Atlas, VITA-Bench (values not retrieved) | BFCL matches the 397B, but 3B active limits multi-step depth | ⚠️ **CLOSE — NEEDS OWN EVAL** | **Low** |
| Qwen3.8-2.4T-A95B (2.4T/95B) | none retrieved | Newest and largest Qwen; almost certainly strong, but **zero agentic data published** | ⚠️ **INSUFFICIENT DATA** | — |
| Kimi-K3 (2.8T/104B) | none on tool use; coding-led positioning | Frontier scale, but no τ²/BFCL/MCP-Atlas published | ⚠️ **INSUFFICIENT DATA** | — |
| DeepSeek-V4-Pro / Flash | none retrieved for V4 | Predecessor V3.2-Exp led open models on Tool-Decathlon at **20.1%** vs Claude's ~40% — a large historical gap | ⚠️ **INSUFFICIENT DATA** | — |
| Nemotron-3-Ultra-550B-A55B | none retrieved | No agentic benchmarks published | ⚠️ **INSUFFICIENT DATA** | — |
| Nemotron-3-Super-120B-A12B | none retrieved | No agentic benchmarks published | ⚠️ **INSUFFICIENT DATA** | — |
| **Nemotron-3.5-Lightning-30B-A3B** | **τ³-Banking 9.48** · IFBench 72.88 · BrowseComp 36.81 · GPQA-D 75.57 | τ³-Banking 9.48 is very low; IFBench trails the 397B's 76.5 | ❌ **BELOW** | Medium |
| Qwen3.5-27B (dense) | BFCL-V4 **68.5** | ~4 pts behind the 122B/397B on the one shared benchmark | ❌ **BELOW** | Medium |
| Qwen3.6-27B / Qwen3.8-27B | none retrieved | Same class as Qwen3.5-27B; newer but no agentic data | ❌ **BELOW** | Low |
| Kimi-K2.5 / K2.6 / K2.7-Code | Terminal-Bench 2.0 66.7 · SWE-bench Verified 80.2 | Coding-specialised — **exactly the axis excluded here** | ❌ **WRONG PURPOSE** | Medium |
| GLM-4.5 | superseded by 5.2 | — | ❌ **SUPERSEDED** | High |
| Nemotron Nano 4B / 30B, Omni | small-model tier | Not Sonnet-class by size or published scores | ❌ **BELOW** | Medium |

### The two strongest individual data points

**GLM-5.2 on MCP-Atlas: 77.0 vs Claude Sonnet 4.6's 61.3.** MCP-Atlas is the closest published benchmark to your actual workload — multi-step MCP workflows with real servers, error handling and retries. A ~15-point lead is large enough to survive some harness mismatch. Supporting evidence: the GLM-5 paper states that on τ²-Bench, MCP-Atlas and Tool-Decathlon, GLM-5 already achieved "comparable performance to Claude" (Opus 4.5 class) — and 5.2 is a later model.

**Qwen3.5-397B on Tool-Decathlon: 38.3.** The Tool-Decathlon paper found Claude-4.5-Sonnet ranked first at under 40%, with **all open-source models at or below 20.1%**. Qwen3.5-397B at 38.3 nearly doubles the previous open-source ceiling and lands inside Claude's own band. Tool-Decathlon is explicitly "diverse, realistic, long-horizon task execution" — the right shape of test.

### Rejected, with reasons

- **Kimi K2.5 / K2.6 / K2.7-Code** — genuinely strong models, but every published number is coding (Terminal-Bench, SWE-bench). No τ², BFCL or MCP-Atlas. Cut on the stated criterion, not on quality.
- **Nemotron-3.5-Lightning-30B** — the only Nemotron with published tool-use data, and it scores **9.48 on τ³-Banking**. NVIDIA positions it as a "sub-agent workhorse," which matches. Fine as a router or classifier tier; not a Sonnet replacement.
- **All 27B-class dense models** — Qwen3.5-27B's BFCL-V4 68.5 trails the 122B by ~4 points, and quantization damage is worse at small scale (W4A16 recovery on an 8B model runs 83–94% on some tasks versus 99–100% at 70B+). Wrong tier for a Sonnet swap.
- **Nemotron Super/Ultra, Qwen3.8 pair, DeepSeek V4, Kimi K3** — not rejected on merit; simply no published agentic benchmarks. Several may qualify. They cannot be recommended as *verified* swaps today.

---

## 4. Instance recommendations — shortlist only

Same pattern as doc 04, restricted to the three models that passed. Sizes are GiB from the HuggingFace file tree; pricing is us-east-1 EC2 on-demand.

### g6e.xlarge — 1× L40S, 48 GB · $1.8610/hr · $1,359/mo

| Category | Model | Why |
|---|---|---|
| Smartest | — | **No shortlisted model fits.** Smallest qualifier is Qwen3.5-122B-A10B-GPTQ-Int4 at 73.49 GiB. |
| Fastest decode | — | — |
| Best prefill / high batch | — | — |
| Safest deployment | — | — |
| **BEST OVERALL** | **Not viable for a Sonnet swap** | Use this tier for a router/classifier in front of a larger brain, not as the brain. |

### g6e.12xlarge — 4× L40S, 192 GB · $10.4926/hr · $7,660/mo

| Category | Model | Why |
|---|---|---|
| Smartest | `Qwen/Qwen3.5-122B-A10B-FP8` (118.46 GiB, TP=4) | Full-precision-class FP8 of the borderline qualifier. |
| Fastest decode | `Qwen/Qwen3.5-122B-A10B-GPTQ-Int4` (73.49 GiB, TP=2) | 10B active × 4-bit ≈ 5 GiB/token. |
| Best prefill / high batch | `Qwen3.5-122B-A10B-FP8` | Native CUTLASS FP8 dense on SM89; MoE experts fall to Triton. |
| Safest deployment | `Qwen3.5-122B-A10B-GPTQ-Int4` (TP=2) | Lowest TP degree on PCIe Gen4. |
| **BEST OVERALL** | **Don't buy this shape** | The same model runs on **one** g7e GPU at $3.36/hr vs $10.49/hr here. 3× cheaper, no TP. |

### g6e.48xlarge — 8× L40S, 384 GB · $30.1312/hr · $21,996/mo

| Category | Model | Why |
|---|---|---|
| Smartest | `Qwen/Qwen3.5-397B-A17B-GPTQ-Int4` (219.58 GiB, TP=8) | A verified qualifier — BFCL-V4 72.9, TAU2 86.7, Tool-Decathlon 38.3. |
| Fastest decode | Same (17B active) | No faster qualifier fits. |
| Best prefill / high batch | Same | FP8 (378.30 GiB) and NVFP4-V2 both exceed practical capacity at TP=8 with KV. |
| Safest deployment | Same | INT4 avoids FP4 entirely; L40S has no FP4 silicon. |
| **BEST OVERALL** | **`Qwen3.5-397B-A17B-GPTQ-Int4`** — but prefer **g7e.24xlarge** | Same model on 4 GPUs at $16.57/hr vs 8 at $30.13/hr, PCIe Gen5 vs Gen4. |

### g7e.2xlarge — 1× RTX PRO 6000, 96 GB · $3.3631/hr · $2,455/mo

| Category | Model | Why |
|---|---|---|
| Smartest | `Qwen/Qwen3.5-122B-A10B-GPTQ-Int4` (**73.49 GiB**) | **The only shortlisted model that fits a single GPU.** BFCL-V4 72.2, IFEval 93.4 (ahead of the 397B's 92.6). ~13 GiB left for KV at 0.9 utilization. |
| Fastest decode | Same | 10B active × 4-bit ≈ 5 GiB/token — ~4× faster per token than a 27B dense FP8. |
| Best prefill / high batch | Same | No smaller qualifier exists; dense FP8 alternatives don't clear the agentic bar. |
| Safest deployment | Same | First-party INT4, 570k downloads, `moe_wna16` is the purpose-built MoE kernel, no FP4 PTX exposure. |
| **BEST OVERALL** | **`Qwen/Qwen3.5-122B-A10B-GPTQ-Int4`** | **The cheapest plausible Sonnet 5 replacement on AWS at $2,455/mo.** Caveat: its qualification rests on BFCL + IFEval only. **Run your own τ²/MCP eval before committing** — this is the one recommendation in this document that genuinely needs your data. |

### g7e.12xlarge — 2× RTX PRO 6000, 192 GB · $8.2861/hr · $6,049/mo

| Category | Model | Why |
|---|---|---|
| Smartest | `Qwen/Qwen3.5-122B-A10B-FP8` (118.46 GiB, TP=2) | 8-bit precision of the borderline qualifier, if 4-bit degrades your tool-call accuracy. |
| Fastest decode | **Two TP=1 replicas** of `Qwen3.5-122B-A10B-GPTQ-Int4` | Each fits one GPU. No all-reduce on a box with no NVLink; ~2× aggregate. |
| Best prefill / high batch | `Qwen3.5-122B-A10B-FP8` (TP=2) | Native CUTLASS FP8 dense. |
| Safest deployment | `Qwen3.5-122B-A10B-FP8` (TP=2) | No FP4, single node. |
| **BEST OVERALL** | **Two TP=1 replicas of the INT4 build** | Buy this shape for throughput or A/B (INT4 vs FP8), not for a bigger model. |

### g7e.24xlarge — 4× RTX PRO 6000, 384 GB · $16.5722/hr · $12,098/mo

| Category | Model | Why |
|---|---|---|
| Smartest | `Qwen/Qwen3.5-397B-A17B-GPTQ-Int4` (219.58 GiB, TP=4) | The best-evidenced open agentic model that fits G-series. ~145 GiB for KV. |
| Fastest decode | Same (17B active) | `nvidia/Qwen3.5-397B-A17B-NVFP4-V2` (226.92 GiB) is the FP4 alternative — but Marlin on SM120, and MTP *regresses* 22% there. |
| Best prefill / high batch | `nvidia/Qwen3.5-397B-A17B-NVFP4-V2` | V2 is 7 GiB smaller than V1. FP8 (378.30 GiB) technically fits 384 GiB but leaves no KV. |
| Safest deployment | `Qwen3.5-397B-A17B-GPTQ-Int4` | No FP4 PTX dependency, no MTP penalty, smallest of the three 397B builds. |
| **BEST OVERALL** | **`Qwen/Qwen3.5-397B-A17B-GPTQ-Int4`** | **The best-evidenced Sonnet 5 replacement that runs on hardware you have.** $12,098/mo — 5× the 122B tier, for verified τ²/Tool-Decathlon coverage the 122B lacks. |

### g7e.48xlarge — 8× RTX PRO 6000, 768 GB · $33.1443/hr · $24,195/mo

| Category | Model | Why |
|---|---|---|
| Smartest | `nvidia/GLM-5.2-NVFP4` (**432.95 GiB**, TP=8) | **The strongest agentic evidence of any model here** — MCP-Atlas 77.0. 1M context, MIT licence, ~335 GiB spare for KV. |
| Fastest decode | `nvidia/Qwen3.5-397B-A17B-NVFP4-V2` (226.92 GiB, TP=8) | 17B active vs GLM's 40B. |
| Best prefill / high batch | `Qwen/Qwen3.5-397B-A17B-FP8` (378.30 GiB, TP=8) | Native dense FP8; MoE experts still Triton on SM120. GLM-5.2-FP8 at 703.77 GiB leaves only ~64 GiB — marginal. |
| Safest deployment | `Qwen3.5-397B-A17B-GPTQ-Int4` (219.58 GiB, TP=8) | Smallest footprint, no FP4. |
| **BEST OVERALL** | **`nvidia/GLM-5.2-NVFP4`** — or move to `p5.48xlarge` | Best agentic evidence available, but TP=8 over PCIe delivers ~⅓ of 8× H100 and NVFP4 runs Marlin, not native FP4. At $24,195/mo vs p5's $40,179/mo, benchmark both. |

### p5.4xlarge — 1× H100, 80 GB · $6.88/hr · $5,022/mo

| Category | Model | Why |
|---|---|---|
| Smartest | `Qwen/Qwen3.5-122B-A10B-GPTQ-Int4` (73.49 GiB) | Fits, but ~2 GiB spare after overhead — short context only. The NVFP4 build (77.79 GiB) does not practically fit. |
| Fastest decode | Same | 3.35 TB/s bandwidth, ~2× g7e. |
| Best prefill / high batch | Same | H100 has **native FP8 MoE grouped GEMM** — but no FP8 build of a qualifier fits 80 GiB. |
| Safest deployment | Same | Mature SM90 stack. |
| **BEST OVERALL** | **Not recommended for this purpose** | 2× the cost of g7e.2xlarge for the same model with *less* headroom. Buy g7e.2xlarge instead. |

### p5.48xlarge — 8× H100, 640 GB · $55.04/hr · $40,179/mo

| Category | Model | Why |
|---|---|---|
| Smartest | `nvidia/GLM-5.2-NVFP4` (432.95 GiB, TP=8) | Best agentic evidence, on NVLink at 900 GB/s. Runs W4A16 (Hopper has no FP4). |
| Fastest decode | `nvidia/Qwen3.5-397B-A17B-NVFP4-V2` (226.92 GiB, TP=4) | Half the TP degree, 17B active. |
| Best prefill / high batch | `Qwen/Qwen3.5-397B-A17B-FP8` (378.30 GiB, TP=8) | **Native FP8 MoE grouped GEMM** — impossible on any G-series box. ~260 GiB spare. |
| Safest deployment | `Qwen3.5-397B-A17B-FP8` | Every kernel path native on SM90. |
| **BEST OVERALL** | **`Qwen/Qwen3.5-397B-A17B-FP8`** | The 397B running as designed: native FP8 MoE, NVLink, no quantization questions. If the swap must be defensible on evidence, this is it. GLM-5.2-**FP8** (703.77 GiB) does **not** fit — needs H200. |

### p5e.48xlarge — 8× H200, 1,128 GB

| Category | Model | Why |
|---|---|---|
| Smartest | `zai-org/GLM-5.2-FP8` (703.77 GiB, TP=8) | Full FP8 of the best-evidenced agentic model, ~345 GiB spare, native FP8 MoE, 1M context. |
| Fastest decode | `nvidia/Qwen3.5-397B-A17B-NVFP4-V2` (226.92 GiB, TP=4) | 17B active on 4.8 TB/s per GPU. |
| Best prefill / high batch | `zai-org/GLM-5.2-FP8` | Native FP8 MoE on 38.4 TB/s aggregate. |
| Safest deployment | `Qwen/Qwen3.5-397B-A17B-FP8` (378.30 GiB) | ~670 GiB spare — most comfortable large deployment here. |
| **BEST OVERALL** | **`zai-org/GLM-5.2-FP8`** | The strongest agentic model at full 8-bit precision with no quantization caveat at all. Its natural home. |

### p6-b200.48xlarge — 8× B200, 1,432 GB · $113.93/hr · $83,171/mo

| Category | Model | Why |
|---|---|---|
| Smartest | `nvidia/GLM-5.2-NVFP4` (432.95 GiB, TP=8) | **Native FP4 tensor cores** — the hardware NVIDIA tested this checkpoint on. 40% the footprint of FP8 with published accuracy parity. |
| Fastest decode | `nvidia/Qwen3.5-397B-A17B-NVFP4-V2` (226.92 GiB) | Native FP4 *and* MTP speculative decoding works here — the 22% Marlin regression disappears. |
| Best prefill / high batch | `nvidia/GLM-5.2-NVFP4` | ~10 PFLOPS FP4/GPU, NVLink5 at 1.8 TB/s, `--enable-expert-parallel` finally safe. |
| Safest deployment | `Qwen/Qwen3.5-397B-A17B-FP8` (378.30 GiB) | Every path native, ~1,050 GiB spare. |
| **BEST OVERALL** | **`nvidia/GLM-5.2-NVFP4`** | Every format native, NVIDIA's own reference target, best agentic evidence. Also the only tier where quantization is measurably lossless on an agentic benchmark. $83,171/mo needs very high sustained utilization. |

*p6-b300 and the GB200 UltraServers are omitted: no shortlisted model requires more than 704 GiB, so the extra capacity buys nothing for this purpose. They matter only for Kimi K3 or Qwen3.8-2.4T, neither of which currently qualifies on evidence.*

---

## 5. Quantization caveats for the shortlist

**GLM-5.2 — the only model with published quantization-vs-agentic data, and it is reassuring.** NVIDIA's GLM-5.2-NVFP4 card publishes NVFP4 against the FP8 baseline:

| Precision | GPQA-D | SciCode | IFBench | AA-LCR | **τ²-Bench Telecom** |
|---|---|---|---|---|---|
| FP8 baseline | 89.52 | 49.85 | 74.95 | 69.38 | **97.9** |
| **NVFP4** | 89.39 | 49.04 | **75.81** | **70.13** | **98.25** |

NVFP4 is lossless within noise and marginally **ahead** on instruction-following, long-context recall and agentic tool use. This is the only direct evidence anywhere in this research that 4-bit quantization does not damage tool calling.

**Qwen3.5-397B and 122B — no equivalent data.** No published GPTQ-Int4-vs-FP8 comparison on BFCL, τ², or JSON-schema adherence exists for either. Two mitigating signals: quantization damage scales inversely with model size (W4A16 recovers 99–100% at 70B+ versus 83–94% at 8B), and GPTQ leaves attention, embeddings, `lm_head` and the shared expert in high precision. Both are indirect.

**The proxy worth using: HumanEval.** On Qwen3-32B GPTQ-Int4, MMLU-Pro dropped 1.6 points while **HumanEval dropped 8** — a 5× difference on the same checkpoint. Code generation and tool calling share the property of requiring syntactically exact output where one wrong token fails the task. Expect tool calling to behave more like HumanEval than like MMLU, and weight the risk accordingly.

**Practical:** if 4-bit tool-call accuracy disappoints, the FP8 escape hatch costs one instance tier — 122B FP8 on g7e.12xlarge ($6,049/mo), 397B FP8 on p5.48xlarge ($40,179/mo).

---

## 6. Practical swap considerations

| | GLM-5.2 | Qwen3.5-397B | Qwen3.5-122B |
|---|---|---|---|
| Context | 1M | 262K | 262K |
| vs Sonnet 5's 200K | 5× | 1.3× | 1.3× |
| Licence | **MIT** | **Apache-2.0** | **Apache-2.0** |
| vLLM tool parser | `glm47` | `qwen3_coder` (Qwen's rec) or `qwen3_xml` | same |
| vLLM reasoning parser | `glm45` | `qwen3` | `qwen3` |
| Multimodal | text | vision + video | vision + video |
| Min viable instance | g7e.48xlarge | g7e.24xlarge | **g7e.2xlarge** |
| Min $/mo | $24,195 | $12,098 | **$2,455** |

All three carry permissive licences with no revenue or MAU thresholds — unlike Kimi's modified-MIT, which triggers obligations above $20M/month revenue or 100M MAU.

**Known parser issues to handle:** `qwen3_coder` can emit an infinite `!!!!` stream on long tool-call inputs and does not stream arguments (vLLM #30439) — `qwen3_xml` is the documented alternative. On Qwen3.5, `tool_choice='required'` combined with MTP and thinking mode produces XML instead of JSON at a 50–70% failure rate; use `tool_choice='auto'` or disable thinking. GLM parsers have been reported to truncate tool names in streaming near full context.

**Prompt portability:** no published data on how a Sonnet-tuned prompt transfers to any of these. Given Anthropic's own τ² scores rely on model-specific prompt addenda, assume **some re-tuning is required** and budget for it.

---

## 7. Recommendation

**Run a three-way bake-off, not a swap.** The evidence supports a shortlist, not a verdict.

1. **Start with `Qwen3.5-122B-A10B-GPTQ-Int4` on g7e.2xlarge ($2,455/mo).** It is the only qualifier that fits one GPU, and at 1/5 the cost of the next tier. Its case rests on BFCL-V4 72.2 and IFEval 93.4 — real but narrow. **Your τ²/MCP eval decides it.**
2. **Benchmark `Qwen3.5-397B-A17B-GPTQ-Int4` on g7e.24xlarge ($12,098/mo) as the evidence-backed option.** Broadest agentic coverage of any open model: BFCL-V4 72.9, TAU2 86.7, Tool-Decathlon 38.3 inside Claude's own band.
3. **Benchmark `nvidia/GLM-5.2-NVFP4` on g7e.48xlarge ($24,195/mo) as the capability ceiling.** MCP-Atlas 77.0 versus Sonnet 4.6's 61.3 is the single strongest agentic result here, and it is the only shortlisted model with published proof that quantization doesn't hurt tool calling.

**Decision rule:** if the 122B holds ≥95% of Sonnet 5's task success on your trajectories, take it — the cost saving is roughly 10× and it needs one GPU. If it falls short on multi-step depth specifically, go to the 397B before the GLM, because 4 GPUs on PCIe beats 8.

**What to measure:** JSON validity rate, argument correctness, correct-tool selection, **negative tool discipline** (correctly declining when no tool applies — a real BFCL category that degrades as your registry grows), and end-to-end trajectory latency. Several hundred multi-step trajectories against your actual MCP schemas.

**Do not skip this.** Anthropic has not published Sonnet 5's τ²-bench, MCP-Atlas or BFCL scores, so the baseline itself is inferred from Sonnet 4.5/4.6. **The only rigorous comparison available to you is one you run yourself** — and the cheapest way to get it is to point your existing harness at both a Sonnet 5 endpoint and a g7e.2xlarge for a week.

---

## 8. Claims that could not be verified

- **Claude Sonnet 5's τ²-bench, MCP-Atlas, BFCL and Tool-Decathlon scores** — not published by Anthropic. The baseline here is bracketed by Sonnet 4.5 (τ²: 86.2 retail / 70.0 airline / 98.0 telecom) and Sonnet 4.6 (MCP-Atlas 61.3%). Every "beats Sonnet 5" statement is an inference against those bounds.
- **Qwen3.5-397B's TAU2 86.7 domain breakdown** — unclear whether aggregate or a single domain. Given telecom saturation, this matters; treat with caution.
- **GLM-5.2's τ²-bench Retail and Airline** — only Telecom (99.1) published, and Telecom is the saturated domain.
- **MCP-Atlas harness mismatch** — GLM's 77.0 used the 500-task public set, 10-minute timeout, Gemini 3 Pro judge; Anthropic's 61.3 used their own harness at max effort. Not strictly comparable.
- **Any agentic benchmark for** Qwen3.8 (either size), Kimi K3, DeepSeek V4 Pro/Flash, Nemotron-3-Super, Nemotron-3-Ultra — none retrieved. These are unranked, not disqualified.
- **Quantized-vs-full agentic comparison for the Qwen models** — does not exist publicly.
- **Prompt portability from Sonnet 5 to any open model** — no published data.
- An independent 2026 validity audit found reliability problems across BFCL, τ-bench, τ²-Bench, LiveMCPBench and MCP-Atlas. **Every number in this document inherits that uncertainty.**
