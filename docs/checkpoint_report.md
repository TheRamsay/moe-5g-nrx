# Compute-Aware Mixture-of-Experts for Efficient 5G Neural Receiver

**Team:** Dominik Huml (xhumld00), Jakub Kontrik (xkontr02), Martin Vaculik (xvaculm00)

## 1. Problem Statement

Modern 5G neural receivers (NRX) achieve strong decoding performance but apply the same computational effort to every received signal, regardless of channel quality. In practice, most received slots are "easy" (high SNR, line-of-sight) and could be decoded by a lightweight model, while only a fraction require a full-capacity receiver. This uniform compute allocation wastes energy and processing resources — a critical concern for mobile devices with limited battery and thermal budgets.

We propose a **compute-aware NRX** using a Mixture-of-Experts (MoE) architecture with **heterogeneous experts** of different sizes. A learned router examines features extracted from the received OFDM resource grid and selects one expert per sample: a lightweight expert for easy channels, a full-capacity expert for difficult ones. The key metric is the **BLER vs. average FLOPs tradeoff** — we aim to reduce inference compute while maintaining block error rate close to a static dense baseline.

## 2. Related Work

**Wiesmayr et al. (2024)** define the modern neural receiver architecture for OFDM systems, providing our dense baseline design. Their model achieves strong BLER but is static and computationally expensive.

**Van Bolderik et al. (2024) — MEAN** apply MoE to 5G neural receivers as a proof-of-concept. However, MEAN uses homogeneous experts (same architecture, same compute cost) and focuses on per-SNR specialisation rather than compute efficiency. Our work addresses an orthogonal objective: heterogeneous expert sizing for adaptive compute allocation. MEAN's gating network uses raw received samples as input; our router operates on pooled features from a shared stem.

**Van Bolderik et al. (2026) — LOREN** extends the MoE direction with LoRA adapters for different network configurations, focusing on memory efficiency rather than compute efficiency.

**Song et al. (2025)** introduce channel-aware gating for wireless networks. Our router design draws inspiration from this concept, using channel-quality features extracted by the shared stem to inform routing decisions.

## 3. Architecture

Our model consists of three components:

**Shared stem.** A two-layer MLP (hidden dims [64, 64], state_dim=56) that processes the concatenated received signal and LS channel estimate. The stem extracts features used by both the router and the selected expert. Its compute cost is fixed (285M FLOPs) and always paid.

**Channel-aware router.** A small network (hidden_dim=64) that takes pooled stem features and produces a probability distribution over experts. During training, we use Gumbel-Softmax soft gating for differentiability. At inference, hard top-1 selection ensures only one expert executes — guaranteeing real FLOPs savings.

**Heterogeneous experts.** Three CNN-based expert heads with different capacities:

| Expert | Backbone | Params | Expert FLOPs | Total FLOPs | % of large |
|---|---|---:|---:|---:|---:|
| nano | 4 blocks, dim=8 | 90k | 35M | 320M | 20% |
| small | 8 blocks, dim=32 | 168k | 557M | 842M | 52% |
| large | 8 blocks, dim=64 | 450k | 1319M | 1604M | 100% |

Routing nano instead of large saves **80% of expert FLOPs**. The total MoE model has 582k parameters.

**Training objective:**

$$L = L_{BCE} + \gamma \cdot L_{channel} + \alpha \cdot \mathbb{E}[\text{FLOPs ratio}] + \beta \cdot L_{balance}$$

where $L_{BCE}$ is the bit-level cross-entropy for LLR estimation, $L_{channel}$ ($\gamma$=0.05) regularises channel estimation quality, $\alpha$=1e-3 penalises expected compute, and $\beta$=0.1 encourages load balance across experts.

## 4. Evaluation Environment

**Dataset.** We use NVIDIA Sionna to simulate standard 3GPP channel models: UMa (urban macro, outdoor) and TDL-C (tapped delay line, indoor/NLOS). The dataset contains 250k training, 20k validation, and 40k test samples per channel profile, hosted on HuggingFace (`Vack0/moe-5g-nrx`). Each sample is a 5G NR slot: 14 OFDM symbols, 128 subcarriers, 3.5 GHz carrier, 16-QAM modulation, SIMO 1x4 antenna configuration.

**Metrics.** Our primary metric is **BLER** (Block Error Rate) — the fraction of transport blocks with at least one bit error. We evaluate at multiple SNR bins to capture the waterfall curve, with particular attention to the high-SNR transition region (SNR=17 dB for TDLC) where expert quality differences are most pronounced. Compute efficiency is measured as **average realized FLOPs** at inference under hard top-1 routing.

**Infrastructure.** Training runs on the MetaCentrum PBS cluster (1 GPU, 12 CPUs per job). Experiment tracking via Weights & Biases. Checkpoints are versioned as W&B artifacts.

## 5. Baseline Results

We trained three dense (non-MoE) receivers to convergence (20k steps each) as baselines:

| Model | TDLC BLER | UMA BLER | TDLC BLER@SNR=17 | FLOPs |
|---|---|---|---|---|
| dense nano | 0.971 | — | 0.722 | 320M |
| dense small | 0.911 | — | 0.548 | 842M |
| **dense large** | **0.866** | — | **0.284** | **1604M** |

The waterfall region (SNR 15-19 dB on TDLC) shows a **44 percentage point BLER gap** between nano and large at SNR=17, confirming that expert size matters significantly and justifying the MoE approach.

![TDLC Waterfall Region — Dense Baselines](figures/waterfall_dense_baselines.png)
*Figure 1: BLER across the TDLC waterfall region for the three dense baselines. The 44 pp gap at SNR=17 between nano and large justifies heterogeneous MoE routing.*

## 6. First MoE Experiments

### 6.1 Phase 1 — Joint Training from Scratch

All three experts and the router are initialised randomly and trained jointly for 10k steps. The FLOPs penalty ($\alpha$=1e-3) successfully drives the router toward cheaper experts, achieving **48% average FLOPs** — but at the cost of abandoning the large expert entirely. The router learns that nano+small are "good enough" and avoids the expensive expert even when it would help.

**Test result:** avg BLER = 0.926, avg FLOPs = 772M (48% of dense large).

### 6.2 Phase 2 — Warm-Start with Staged Training

Each expert is initialised from its matching pre-trained dense checkpoint. Training proceeds in two stages: 2k steps with experts frozen (router learns to distribute), then 10k steps of joint fine-tuning.

**Result: complete router collapse.** The warm-started large expert is strictly better than nano/small from step 1, so the router sends 100% of traffic to large and never redistributes. The model converges to a fine-tuned dense large — excellent BLER but zero compute savings.

**Test result:** avg BLER = 0.879, avg FLOPs = 1604M (100%).

### 6.3 Anti-Collapse Experiments

We tested six mechanisms to break the Phase 2 router collapse:

| Mechanism | Routing diversity | BLER | Outcome |
|---|---|---|---|
| Stronger load balance ($\beta$=0.5, 1.0) | Collapsed | Good | MSE penalty too weak |
| Very strong load balance ($\beta$=2.0) | 33/33/33 uniform | Poor (0.971) | Forced-uniform, experts can't specialise |
| Soft capacity constraint | Collapsed after unfreeze | Good | Insufficient penalty weight |
| Switch Transformer aux loss | Collapsed | — | Weight too low |
| **Asymmetric warm-start** | **33/29/38 (all active)** | **0.913** | **Positive result** |

### 6.4 Asymmetric Warm-Start — Key Finding

The breakthrough came from attacking the root cause: **remove the warm-start advantage from large.** We warm-start stem + nano + small from dense checkpoints but leave large at random initialisation. Without an initial quality advantage, the router has no reason to collapse onto large.

At 6k steps, the router used only nano+small (large was still untrained). We extended the run to 12k steps using checkpoint resume, and **large "woke up"** — the router discovered it once it became competitive (~step 8000-10000).

![Expert Usage Over Training — Asymmetric Warm-Start](figures/expert_usage_asym_warm.png)
*Figure 2: Expert usage during asymmetric warm-start training. Large (random init) initially dominates briefly, crashes to ~0% by step 2000, then re-emerges at step ~6550 as it trains up to competence. Final routing: 33% large, 29% nano, 38% small.*

## 7. Preliminary Results

| Run | Routing | Avg BLER | Avg FLOPs | FLOPs % |
|---|---|---|---|---|
| Dense large (baseline) | — | ~0.879 | 1604M | 100% |
| Phase 2 v1 (collapsed) | 100% large | **0.879** | 1604M | 100% |
| **Asym warm 12k** | **33/29/38** | **0.913** | **~880M** | **~55%** |
| Phase 1 s56 | 39/36/24 | 0.926 | 772M | 48% |

The asymmetric warm-start run represents a genuine Pareto point: it trades 3.4 pp BLER compared to the dense large baseline for a **45% reduction in average FLOPs**, with all three experts actively contributing.

We also identify a key characterisation finding: **opposite failure modes** in heterogeneous MoE training. Joint-from-scratch (Phase 1) over-penalises the expensive expert, while full warm-start (Phase 2) cannot escape the dominant expert. Asymmetric warm-start resolves this by letting the large expert earn its traffic through training rather than receiving it by default.

![BLER vs FLOPs Pareto Frontier](figures/pareto_bler_flops.png)
*Figure 3: BLER vs FLOPs Pareto frontier. Grey squares are dense baselines. Coloured circles are MoE runs. The dashed line connects Pareto-optimal points. Asymmetric warm-start 12k (green) fills the gap between the collapsed Phase 2 (best BLER, max FLOPs) and Phase 1 (cheapest, worst BLER).*

## 8. Next Steps

1. **Test-split evaluation** of asymmetric warm-start 12k to confirm validation numbers
2. **SNR-binned routing analysis** — verify whether routing is adaptive (difficulty-dependent) or near-uniform
3. **Extended training** (18k-20k steps) to see if BLER continues improving
4. **Out-of-distribution evaluation** on DeepMIMO ray-traced channels to test generalisation
5. **Pareto frontier visualisation** with confidence intervals across seeds

## References

- Wiesmayr, R. et al. (2024). OFDM-based Neural Receivers. arXiv:2409.02912
- van Bolderik, E. et al. (2024). MEAN: Mixture of Experts with Attention for 5G. IEEE PIMRC.
- van Bolderik, E. et al. (2026). LOREN: LoRA-Enhanced Neural Receiver. arXiv:2602.10770
- Song, J. et al. (2025). Channel-Aware Gating for Wireless Networks. IEEE TWC.
