# MoE NL Phase 1 v1

## Question

Does the wider nano/small/large expert range (block_dim 8/32/64) produce stronger routing
differentiation than the original small/mid/large range (32/48/64)?

Phase 1 joint-from-scratch training with alpha=1e-3, beta=0.1 (locked from original Phase 1).
Uses state_dim=32 throughout (stem=[32,32]) — lossless vs state_dim=56 per stem bottleneck study.

## Why

The original MoE Phase 1 showed routing (all experts ~33%, entropy ~99.9%) but the quality
gap between experts was too small to make the Pareto curve interesting. The new expert range
spans block_dim 8→64 (8× width difference vs the original 32→64 = 2× difference), which should
produce a much larger BLER gap between nano and large, giving the router a stronger signal.

## Config

- Model: `moe_nl` (nano/small/large experts, state_dim=32)
- `alpha=1e-3`, `beta=0.1` (inherited from Phase 1 sweep winner)
- 10k steps, lr=1e-3, wd=1e-4, constant LR, seed=67
- Validation: 5 SNR bins per profile logged as timeseries

## What To Watch

- `train/router_entropy` — should stay >0.85; collapse means beta needs adjusting
- `train/expert_usage/{nano,small,large}` — should all get meaningful share
- `val/tdlc/snr_bin_*/bler` — should show large preferred at high SNR, nano at low SNR
- `val/uma/snr_bin_*/bler` — UMA routing may be weaker (smaller quality gap per findings)

## Job

`18723728.pbs-m1.metacentrum.cz` — gpu-46gb, walltime 8h

## Results

| Metric | Value | Notes |
|---|---|---|
| Router entropy | — | |
| nano usage | — | |
| small usage | — | |
| large usage | — | |
| val TDLC BER | — | |
| val UMA BER | — | |
