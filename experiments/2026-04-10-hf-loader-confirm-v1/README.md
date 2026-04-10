# HF Loader Confirm V1

Goal: confirm the best HuggingFace loader setting from the first sweep using the
same GPU class for both runs.

## Why This Exists

The initial sweep mixed multiple GPU families, so worker/prefetch settings were
confounded with hardware differences. This follow-up compares only:

- `w2_p1` = `hf_num_workers=2`, `hf_prefetch_factor=1`
- `w2_p2` = `hf_num_workers=2`, `hf_prefetch_factor=2`

Final comparison was run on the same `16 GB` GPU class (`Quadro RTX 5000`).

## Run Design

- Model: `exp05_dense_capacity_large`
- Dataset: `Vack0/moe-5g-nrx`
- Training profile: `mixed`
- Validation: disabled
- Max steps: 300
- Logging cadence: every 25 steps

## Launch

```bash
bash experiments/2026-04-10-hf-loader-confirm-v1/submit.sh qsub
```

## Selection Criterion

Winner is the run with the lower walltime on the same GPU class, assuming both
finish successfully and land in the same training-loss range.

## Results

| Exp name | Job | GPU | Walltime | Peak RAM | Final loss | Final BER |
|---|---:|---|---:|---:|---:|---:|
| `hf_loader_confirm_w2_p1` | `18890316` | Quadro RTX 5000 | **18m10s** | **20.2 GB** | 0.2823 | 0.1404 |
| `hf_loader_confirm_w2_p2` | `18895412` | Quadro RTX 5000 | 29m22s | 35.8 GB | 0.2720 | 0.1363 |

## Decision

Lock the HF loader defaults to:

- `training.hf_num_workers=2`
- `training.hf_prefetch_factor=1`

Interpretation:
- `prefetch=2` increased queued host-side work and memory pressure.
- The small final-loss difference is not worth a 61% walltime increase and a
  much larger RAM footprint.
- For mixed HF dense training, throughput and host stability dominate this tiny
  optimization-noise difference.
