# HF Loader Sweep V1

Goal: identify whether HuggingFace training throughput is limited by worker count,
prefetch, or host resource starvation on MetaCentrum.

## Hypothesis

The current HF path is CPU/RAM bound in mixed-profile training because
`training.hf_num_workers` is applied per profile loader. In `mixed` mode,
`hf_num_workers=4` becomes 8 workers total, which overcommits a 4 CPU / 24 GB
job and leaves the GPU idle.

## Sweep Design

- Model: `exp05_dense_capacity_large`
- Training data: `Vack0/moe-5g-nrx`
- Training profile: `mixed`
- Validation: disabled
- Max steps: 300
- Logging cadence: every 25 steps
- Primary profiling metrics:
  - `train/data_time_s`
  - `train/step_time_s`
  - `train/iter_time_s`
  - `train/samples_per_s`

## Resource Policy

Use the 16 GB dense GPU class, but request more host resources than the default
dense preset:

- `ncpus=8`
- `mem=48gb`
- `gpu_mem=16384mb`

This is meant to be safe for short profiling runs while still landing on the
dense-capable GPU class.

## Variants

| Exp name | per-profile workers | total workers (mixed) | prefetch |
|---|---:|---:|---:|
| `hf_loader_w0_p0` | 0 | 0 | 0 |
| `hf_loader_w1_p1` | 1 | 2 | 1 |
| `hf_loader_w1_p2` | 1 | 2 | 2 |
| `hf_loader_w2_p1` | 2 | 4 | 1 |
| `hf_loader_w2_p2` | 2 | 4 | 2 |

## Launch

```bash
bash experiments/2026-04-10-hf-loader-sweep-v1/submit.sh qsub
```

To print the exact commands without submitting:

```bash
bash experiments/2026-04-10-hf-loader-sweep-v1/submit.sh print
```

## Selection Criterion

Choose the smallest worker/prefetch setting that:

1. Keeps `data_time_s` close to or below `step_time_s`
2. Avoids memory pressure / cgroup kills
3. Improves `samples_per_s` materially over `w0_p0`

If all variants remain data-bound, the next step is code-level optimization in
the HF dataset path rather than further worker tuning.
