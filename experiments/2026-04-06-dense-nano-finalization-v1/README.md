# Dense Nano Finalization v1

## Question

Finalize nano (block_dim=8, 4 blocks, stem=[64,64], state_dim=56) to 20k steps for MoE
Phase 2 warm-start, using the same recipe as small/mid/large finalization.

## Motivation

Capacity floor study showed nano (90k params) achieves BLER comparable to small at 10k steps
(with a real gap visible at high-SNR bins). Finalizing to 20k gives an equal-footing warm-start
checkpoint for the redesigned MoE expert family (nano/small/large).

## Config

- stem_hidden_dims=[64,64], state_dim=56, block_dim=8, readout=32, 4 blocks
- lr=1e-3, wd=1e-4, constant LR, seed=67, 20k steps
- Validation: 5 SNR bins per profile as timeseries

## Job

`18723723.pbs-m1.metacentrum.cz` — gpu-16gb, walltime 8h

## Results

| Run | val TDLC BER | val UMA BER | best score | artifact |
|---|---|---|---|---|
| nano_final20k | — | — | — | — |
