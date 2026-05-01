# Middle-Expert Investigation — what does small ACTUALLY do?

## The mystery

Per-expert success rate (2026-04-30): nano and small NEVER decode (0% block-success
on routed samples). Only large decodes (23-29%).

But exp61 v2 (2026-05-01) showed: replacing nano + small with sink + channel_only
matches BLER but **loses FLOPs efficiency** (real_flops 0.78 vs exp26's 0.56).
Router defaults to large for any non-hopeless sample.

So small must be contributing SOMETHING beyond decoding. What?

## Five concrete questions

| # | Question | Method |
|---|---|---|
| Q1 | What's small's actual BIT-level error rate on routed samples? (Random=0.5, partial=0.2, near-decode=0.1) | Per-route BER aggregate |
| Q2 | For samples routed to small, what would large/nano have done? | Counterfactual: force each expert on small's samples |
| Q3 | Is small's channel estimate better than nano's would have been? | MSE-per-expert on the same routed samples |
| Q4 | Is small's 0% success universal, or only at low SNR? | Per-SNR-bin per-expert success rate, all forced |
| Q5 | Are small's bit predictions random (~0 confidence), correct-and-confident, or confident-wrong? | Histogram of \|sigmoid(logits) - 0.5\| per expert per route |

## Method

Loads exp26 best checkpoint. For each of UMa + TDLC test (4k samples each):

1. Forward through stem → captures shared_state via hook
2. Records router's actual choice (selected_expert_index)
3. **Manually calls each expert** on the same shared_state (counterfactual)
4. For each (sample, expert) pair records: BER, BLER, channel MSE, mean-bit-confidence, wrong-bit-confidence

Outputs 5 figures + JSON summary.

## Cluster

`select=1:ncpus=4:ngpus=1:mem=16gb`, walltime 1h. ~5min actual runtime.
