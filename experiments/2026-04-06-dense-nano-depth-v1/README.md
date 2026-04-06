# Dense Nano Depth v1

## Question

Does depth matter at nano scale (block_hidden_dim=8)? Comparing 2 / 4 / 8 blocks with the
same width and stem to isolate the depth effect.

## Motivation

The current nano expert uses 4 blocks. If quality is flat across depths at block_dim=8,
the expert design can use the cheapest option (2 blocks) for the nano warm-start. If depth
matters significantly, the 4-block design is justified.

## Variants

| Name | num_blocks | block_dim | stem | state_dim | Est. params |
|---|---:|---:|---|---:|---:|
| nano_2blk | 2 | 8 | [64,64] | 56 | ~83k |
| nano_4blk (ref) | 4 | 8 | [64,64] | 56 | ~90k |
| nano_8blk | 8 | 8 | [64,64] | 56 | ~103k |

## Training Recipe

- lr=1e-3, wd=1e-4, constant LR, seed=67, 10k steps
- Validation: 5 SNR bins per profile as timeseries

## Jobs

- `18723730` — nano_2blk, gpu-16gb
- `18723731` — nano_8blk, gpu-16gb

## Results

| Run | val TDLC BER | val UMA BER | best score | val TDLC BLER |
|---|---|---|---|---|
| nano_2blk | — | — | — | — |
| nano_4blk (ref, 10k) | 0.1323 | 0.2814 | 0.2064 | 0.9711 |
| nano_8blk | — | — | — | — |
