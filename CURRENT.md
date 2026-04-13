# Current Work — 2026-04-13

## Pareto Results (Test Split)

| Run | Routing (large/nano/small) | TDLC BLER | UMA BLER | Avg BLER | FLOPs % | Config |
|---|---|---|---|---|---|---|
| Phase 2 v1 | 100/0/0 (collapsed) | 0.835 | 0.923 | **0.879** | 100% | exp17 |
| **Asym warm 12k** | **46/2/52 (tdlc) 30/31/39 (uma)** | **0.881** | **0.939** | **0.910** | **61%** | exp23 + resume |
| Phase 1 s56 | 39/36/24 | 0.906 | 0.945 | 0.926 | 48% | exp18 |

Asym warm 12k has genuinely **adaptive** routing: TDLC (harder) gets 46% large, UMA (easier)
gets 31% nano. Per-SNR analysis shows the router sends large to the waterfall region where
expert quality matters, and nano/small to low SNR where all experts fail equally.

**Key finding:** routing is SNR-adaptive. At TDLC SNR=18, 93% goes to large. At SNR=-8,
84% goes to small (rational — large can't help at hopeless SNR).

## What Happened This Sprint

### Phase 1 s56 (exp18) — joint from scratch, state_dim=56
- Fixed the stem bottleneck from original Phase 1 (s32 → s56)
- Result: router avoided large entirely (FLOPs penalty too aggressive), settled on nano+small
- Test eval: BLER 0.926 avg, 48% FLOPs, heterogeneous but large abandoned
- Checkpoint: `model-moe_phase1_s56_a1e3_b0p1_s67-2op33pak:best` (step 9000)

### Phase 2 v1 (exp17) — warm-start + staged (2k frozen + 10k joint)
- Router collapsed to 100% large from step 1, never recovered (beta=0.1 too weak)
- Effectively a fine-tuned dense large: best BLER but zero compute savings
- Test eval: BLER 0.879 avg, 100% FLOPs
- Checkpoint: `model-moe_phase2_v1_a1e3_b0p1_s67-89no8f1k:best` (step 12000)

### Anti-Collapse Sweep — 6 experiments, 3 mechanisms that break collapse

| Experiment | Mechanism | Entropy | Routing | BLER | Verdict |
|---|---|---|---|---|---|
| β=0.5, β=1.0 | MSE load balance | ~0 | 100% large | good | **collapsed** |
| β=2.0 | MSE load balance (strong) | 1.097 | 33/33/33 uniform | terrible | breaks collapse but forced-uniform, experts can't specialize |
| capacity (cf=1.5, w=0.5) | Soft capacity constraint | ~0 | 100% large | good | diverse during frozen phase, **collapsed after unfreeze** |
| switch_aux (w=0.01) | Switch Transformer aux loss | ~0 | 100% large | — | **total failure**, weight too weak |
| **asym_warm** | Large starts random | **0.811** | **33/29/38** | **0.913** | **positive result** — large woke up after 6k steps |

### Asym Warm 12k — The Breakthrough
- Asymmetric warm-start: stem + nano + small warm-started, large starts random
- At 6k steps: large was 0% (hadn't trained up yet), router used nano+small only
- **Extended to 12k via resume**: large caught up and router discovered it (33% usage)
- All three experts active with meaningful shares
- Val TDLC BLER@SNR=17: 0.411 (vs Phase 1's 0.588, Phase 2's 0.215)
- Checkpoint: `model-moe_phase2_asym_nlwarm_s67_12k-3witw8yw:best` (step 12000)

### Infrastructure Improvements
- **Checkpoint sync fix**: `metacentrum_job.sh` now syncs `$WORK_ROOT/checkpoints/`
  during cleanup — walltime kills no longer lose checkpoints
- **Resume support**: `training.resume_from` accepts local path or W&B artifact ref.
  Restores model weights + optimizer state + global_step. Used successfully for the
  6k→12k extension runs.
- **Latest checkpoints upload to W&B**: `log_artifact=True` for latest checkpoints,
  not just best. Survives walltime kills even without local sync.
- **Resource tuning**: 24GB RAM is sufficient (was 48GB). Faster queue times.

## Currently Running Jobs

| Job ID | Description | Status |
|---|---|---|
| 18987381 | Asym warm 20k (resume from 12k) | running, step ~16.5k, ~3.5h left |

## Completed Since Last Update

- Asym warm 12k test eval: TDLC 0.881, UMA 0.939, avg 0.910, FLOPs 61%
- Per-SNR routing analysis: router is SNR-adaptive (large dominates waterfall, nano/small at low SNR)
- Checkpoint report drafted with 5 figures, submitted
- Resume support added and tested (12k→20k resume working)
- Checkpoint sync fix (walltime kills no longer lose checkpoints)
- Latest checkpoints now upload to W&B

## Asym Warm 20k Progress (in-flight)

Val BLER is improving as training continues:

| Step | Val TDLC BLER | Val TDLC @SNR=17 | Routing (large/nano/small) |
|---|---|---|---|
| 12000 | 0.881 | 0.411 | 33/29/38 |
| 15000 | 0.858 | 0.304 | 39/21/40 |
| 15500 | — | — | best checkpoint so far |
| 16000 | 0.851 | 0.273 | 39/22/39 |
| 16500 | 0.854 | 0.291 | 39/22/39 |

Routing has stabilised at ~39% large / 22% nano / 39% small. FLOPs ratio ~0.60.
BLER still trending down but oscillating. Best checkpoint at step 15500.

## Immediate Next Steps

1. **Wait for 20k run** to finish, run test eval
2. **Alpha sweep** on asym warm: {5e-4, 2e-3} to map Pareto curve
3. **Multi-seed** (s32, s42) on best config for robustness
4. **DeepMIMO OOD evaluation** — pipeline ready
5. **Difficulty-guided routing** — use per-sample SNR in loss to improve specialisation

## Key Findings

- Nano is underutilised at realistic SNR (mostly absorbs hopeless low-SNR traffic)
- Effective routing is small vs large; nano is a "loss sink" at realistic operating points
- Per-SNR routing is rational: router spends FLOPs where they change BLER outcomes
- Training speed bottleneck is data loading (~3.5s/step, GPU idle 99% of the time)

## Open Questions

- Will 20k steps close the BLER gap further, or has the model plateaued?
- Can alpha tuning find a better BLER/FLOPs operating point?
- Would difficulty-guided routing improve nano utilisation at realistic SNR?
- Data loader optimisation: chunked alternation or single interleaved dataset?
