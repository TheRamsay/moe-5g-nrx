# Dense Large s32 Finalization v1

## Question

Does state_dim=32 match state_dim=56 large at 20k steps? Provides the warm-start checkpoint
for the redesigned MoE Phase 2 (nano/small/large_s32 expert family).

## Motivation

The stem bottleneck study confirmed state_dim=32 matches large at 10k steps (best_score=0.2016
vs large 0.199). This run finalizes it to 20k with the same recipe used for small/mid/large,
so all three Phase 2 warm-start checkpoints are trained equally.

## Config

- stem_hidden_dims=[32,32], state_dim=32, block_dim=64, readout=128, 8 blocks
- lr=1e-3, wd=1e-4, constant LR, seed=67, 20k steps
- Validation: 5 SNR bins per profile as timeseries

## Job

`18723729.pbs-m1.metacentrum.cz` — gpu-16gb, walltime 8h

## Results

| Run | val TDLC BER | val UMA BER | best score | artifact |
|---|---|---|---|---|
| large_s32_final20k | — | — | — | — |
