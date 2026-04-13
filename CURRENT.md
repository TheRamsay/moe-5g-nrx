# Current Work — 2026-04-13

## Pareto Results (Test Split)

| Run | Routing (large/nano/small) | TDLC BLER | UMA BLER | Avg BLER | FLOPs % | Config |
|---|---|---|---|---|---|---|
| Phase 2 v1 | 100/0/0 (collapsed) | 0.835 | 0.923 | **0.879** | 100% | exp17 |
| **Asym warm 12k** | **33/29/38** | **0.881** | **0.944** | **0.913** | **55%** | exp23 + resume |
| Phase 1 s56 | 39/36/24 | 0.906 | 0.945 | 0.926 | 48% | exp18 |

Asym warm 12k is the first run with genuinely heterogeneous routing and competitive BLER.
Phase 2 v1 is the quality ceiling (but collapsed — no compute savings).
Phase 1 s56 is the cheapest (but abandoned large — pure nano+small policy).

**Eval pending:** asym warm 12k test-split eval (job 18986864).

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
| 18986864 | Asym warm 12k test-split eval | running |

## Immediate Next Steps

1. **Get asym warm 12k test eval** — confirm the val numbers hold on test split
2. **Update experiment READMEs** with final results
3. **Generate Pareto plot** (BLER vs FLOPs) with all runs
4. **SNR-binned expert usage analysis** — check if routing is adaptive (hard→large)
   or just uniform. The eval tables have `eval/expert_usage_snr_binned`.
5. **Consider extending asym warm further** (18k–20k) to see if BLER keeps improving
6. **DeepMIMO OOD evaluation** — teammate added the dataset generator, test generalization

## Open Questions

- Is the asym warm routing actually adaptive (SNR-dependent) or just near-uniform?
- Would extending to 20k steps close the BLER gap to Phase 2 v1?
- Can direction B (uniform-prior KL during frozen phase) produce better results
  than asym warm? (Not yet tested)
- Should we try asym warm with different alpha/beta?
- DeepMIMO OOD: how much does BLER degrade on real ray-traced channels?
